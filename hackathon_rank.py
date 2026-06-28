import os
import warnings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Suppress sklearn unpickling warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

try:
    import torch
    cpu_cores = os.cpu_count() or 4
    if cpu_cores > 6:
        optimal_threads = min(12, cpu_cores - 4)
    else:
        optimal_threads = max(1, cpu_cores - 1)
    torch.set_num_threads(optimal_threads)
except ImportError:
    pass

import argparse
import sys
import time
import json
import csv
import numpy as np

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CVHunt")

from src.config import SUBMISSION_PATH, SURROGATE_PATH, ID_MAP_PATH, INDEX_PATH, BOOST_KEYWORDS
from src.data_loader import stream_candidates
from src.jd_parser import parse_jd
from src.features import extract_features
from src.semantic_scorer import (
    create_candidate_text, compute_embeddings, compute_similarity
)
from src.structured_scorer import compute_structured_score
from src.vector_index import load_index, search, scores_from_search
from src.honeypot_detector import run_honeypot_checks
from src.hybrid_aggregator import compute_final_ranking
from src.reasoning_generator import generate_reasoning
from src.validator import validate

def parse_args():
    parser = argparse.ArgumentParser(description="CVHunt - Candidate Discovery and Ranking Engine")
    parser.add_argument(
        "--candidates", 
        required=True, 
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--jd", 
        required=True, 
        help="Path to job_description.md"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to write the submission.csv"
    )
    parser.add_argument(
        "--embeddings", 
        default="", 
        help="Optional path to precomputed candidate_embeddings.npy"
    )
    parser.add_argument(
        "--index", 
        default="", 
        help="Optional path to precomputed faiss_index.bin"
    )
    return parser.parse_args()

def main():
    print("=" * 60)
    print(" CVHUNT - Optimized Candidate Ranking Pipeline ")
    print("=" * 60)
    
    args = parse_args()
    
    total_start = time.time()
    timings = {}
    
    # ----------------------------------------------------
    # Phase 1: Parse JD
    # ----------------------------------------------------
    print("\n[Phase 1] Parsing Job Description...")
    p_start = time.time()
    jd_reqs = parse_jd(args.jd)
    timings["JD Parsing"] = time.time() - p_start
    query_text = jd_reqs.to_embedding_text()
    
    # ----------------------------------------------------
    # Phase 2: First-Stage Vector Search (FAISS)
    # ----------------------------------------------------
    print("\n[Phase 2] Executing first-stage vector search...")
    p_start = time.time()
    
    # Encode query JD text
    print("Encoding query JD vector...")
    jd_embedding = compute_embeddings([query_text])[0]
    
    # Load index and candidate IDs mapping
    print("Loading FAISS index...")
    idx_path = args.index if args.index and os.path.exists(args.index) else INDEX_PATH
    index = load_index(idx_path)
    
    print("Loading candidate IDs map...")
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
        
    # Search top 1000 candidates
    k_search = min(1000, len(candidate_ids))
    retrieved_indices, retrieved_distances = search(jd_embedding, index, k=k_search)
    vector_scores_arr = scores_from_search(retrieved_indices, retrieved_distances, len(candidate_ids))
    
    retrieved_cids = [candidate_ids[idx] for idx in retrieved_indices if idx != -1 and idx < len(candidate_ids)]
    retrieved_cids_set = set(retrieved_cids)
    timings["First-Stage FAISS Search"] = time.time() - p_start
    print(f"Retrieved top {len(retrieved_cids_set)} candidate IDs from FAISS.")

    # ----------------------------------------------------
    # Phase 3: Stream & Filter Candidates
    # ----------------------------------------------------
    print("\n[Phase 3] Streaming & filtering candidates...")
    p_start = time.time()
    candidates = []
    features_list = []
    flag_reasons = {}
    
    for cand in stream_candidates(args.candidates):
        cid = cand["candidate_id"]
        if cid not in retrieved_cids_set:
            continue
            
        candidates.append(cand)
        feats = extract_features(cand, jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills), jd_location=jd_reqs.location_preference)
        features_list.append(feats)
        
        is_flagged, reason = run_honeypot_checks(cand)
        if is_flagged:
            flag_reasons[cid] = reason
            
    # 1. Structured & Surrogate scoring on all 1000 candidates
    print("Computing Structured & Surrogate scores on all 1000 candidates...")
    structured_scores = {}
    structured_breakdowns = {}
    s_path = args.embeddings.replace("candidate_embeddings.npy", "surrogate_model.pkl") if args.embeddings else SURROGATE_PATH
    if not os.path.exists(s_path):
        s_path = SURROGATE_PATH
        
    for cand, feats in zip(candidates, features_list):
        cid = cand["candidate_id"]
        score, breakdown = compute_structured_score(feats, surrogate_path=s_path, ideal_range=jd_reqs.ideal_experience_range)
        structured_scores[cid] = score
        structured_breakdowns[cid] = breakdown
        
    # 2. BM25 keyword matching on all 1000 candidates
    print("Computing BM25 Similarity on all 1000 candidates...")
    from rank_bm25 import BM25Okapi
    bm25_texts = [create_candidate_text(c) for c in candidates]
    tokenized_corpus = [doc.lower().split(" ") for doc in bm25_texts]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query_text.lower().split(" ")
    bm25_raw_scores = bm25_model.get_scores(tokenized_query)
    
    min_bm25 = float(np.min(bm25_raw_scores))
    max_bm25 = float(np.max(bm25_raw_scores))
    
    bm25_scores = {}
    if max_bm25 > min_bm25:
        for idx, cand in enumerate(candidates):
            bm25_scores[cand["candidate_id"]] = float(((bm25_raw_scores[idx] - min_bm25) / (max_bm25 - min_bm25)) * 100.0)
    else:
        for cand in candidates:
            bm25_scores[cand["candidate_id"]] = 50.0
            
    # 3. Build vector scores map
    vector_scores = {}
    for idx, cid in enumerate(candidate_ids):
        if cid in retrieved_cids_set:
            vector_scores[cid] = float(vector_scores_arr[idx])
            
    # 4. Run Intermediate scoring (combines FAISS, BM25, Structured) to select the top 150
    print("Running high-fidelity pre-filtering to select top 150 candidates...")
    from src.hybrid_aggregator import compute_final_ranking
    intermediate_results = compute_final_ranking(
        candidates_data=candidates,
        semantic_scores=None,
        structured_scores=structured_scores,
        structured_breakdowns=structured_breakdowns,
        vector_scores=vector_scores,
        flag_reasons=flag_reasons,
        features_list=features_list,
        bm25_scores=bm25_scores,
        cross_encoder_scores=None,
        jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills),
        apply_mapping=False
    )
    
    # Prune candidates list to top 150 based on intermediate ranking
    top_intermediate = intermediate_results[:150]
    top_cids_set = {x[0] for x in top_intermediate}
    
    # Keep candidates and features_list aligned 1-to-1
    top_candidates = []
    top_features_list = []
    for cid, _, _ in top_intermediate:
        for idx, cand in enumerate(candidates):
            if cand["candidate_id"] == cid:
                top_candidates.append(cand)
                top_features_list.append(features_list[idx])
                break
    candidates = top_candidates
    features_list = top_features_list
    
    timings["Candidate Streaming & Filtering"] = time.time() - p_start
    print(f"Loaded and filtered down to top {len(candidates)} candidates for second-pass ML scoring.")

    # ----------------------------------------------------
    # Phase 4: Second-Stage Scoring
    # ----------------------------------------------------
    print("\n[Phase 4] Scoring candidates...")
    p_start = time.time()
            
    # 4.1 Score 1: Semantic Similarity (mpnet-base) on top 150
    print("Computing Score 1 (Semantic Similarity) on top 150...")
    semantic_scores = {c["candidate_id"]: 0.0 for c in candidates}
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading all-mpnet-base-v2 for semantic scorer...")
        sem_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cpu')
        jd_sem_emb = sem_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        rich_texts = [create_candidate_text(c) for c in candidates]
        cand_sem_embs = sem_model.encode(rich_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        
        similarities = np.dot(cand_sem_embs, jd_sem_emb)
        scaled_scores = (similarities + 1.0) / 2.0 * 100.0
        
        for c, score in zip(candidates, scaled_scores):
            semantic_scores[c["candidate_id"]] = float(score)
    except Exception as e:
        print(f"Warning: Failed to load all-mpnet-base-v2: {e}")
        
    # Apply keyword boost to semantic scores
    for c in candidates:
        cid = c["candidate_id"]
        score = semantic_scores.get(cid, 0.0)
        if score > 0.0:
            profile = c.get("profile", {})
            skills_list = c.get("skills", [])
            skills_text = " ".join([s.get("name", "").lower() for s in skills_list])
            full_text = f"{profile.get('headline', '')} {profile.get('summary', '')} {skills_text}".lower()
            has_jd_term = any(kw in full_text for kw in BOOST_KEYWORDS)
            boosted_score = score + 12.0 if has_jd_term else score
            semantic_scores[cid] = min(100.0, float(boosted_score))
            
    # 4.2 Score 5: CrossEncoder Reranker on top 150
    cross_encoder_scores = {}
    try:
        from sentence_transformers import CrossEncoder
        print("Loading cross-encoder/ms-marco-MiniLM-L-6-v2 for reranker...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu', max_length=512)
        
        pairs = [(query_text, create_candidate_text(c)) for c in candidates]
        raw_logits = reranker.predict(pairs, batch_size=64, show_progress_bar=False)
        
        min_logit = float(np.min(raw_logits))
        max_logit = float(np.max(raw_logits))
        
        if max_logit > min_logit:
            for i, c in enumerate(candidates):
                norm_score = ((raw_logits[i] - min_logit) / (max_logit - min_logit)) * 100.0
                cross_encoder_scores[c["candidate_id"]] = float(norm_score)
        else:
            for c in candidates:
                cross_encoder_scores[c["candidate_id"]] = 50.0
    except Exception as e:
        print(f"Warning: Failed to run CrossEncoder reranking: {e}")
        
    timings["Candidates Scoring"] = time.time() - p_start

    # ----------------------------------------------------
    # Phase 5: Aggregate & Sort
    # ----------------------------------------------------
    print("\n[Phase 5] Aggregating scores & sorting...")
    p_start = time.time()
    ranked_results = compute_final_ranking(
        candidates_data=candidates,
        semantic_scores=semantic_scores,
        structured_scores=structured_scores,
        structured_breakdowns=structured_breakdowns,
        vector_scores=vector_scores,
        flag_reasons=flag_reasons,
        features_list=features_list,
        bm25_scores=bm25_scores,
        cross_encoder_scores=cross_encoder_scores,
        jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills)
    )
    timings["Aggregation & Sorting"] = time.time() - p_start
    print(f"Top candidate score: {ranked_results[0][1]:.3f} (ID: {ranked_results[0][0]})")

    # ----------------------------------------------------
    # Phase 6: Generate Reasoning
    # ----------------------------------------------------
    print("\n[Phase 6] Generating justifications for top 100...")
    p_start = time.time()
    top_100 = ranked_results[:100]
    
    final_rows = []
    for rank_idx, (cid, score, breakdown) in enumerate(top_100):
        rank = rank_idx + 1
        reasoning = generate_reasoning(breakdown, rank)
        final_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning
        })
        
    timings["Reasoning Generation"] = time.time() - p_start

    # ----------------------------------------------------
    # Phase 7: Write Output CSV
    # ----------------------------------------------------
    print(f"\n[Phase 7] Writing results to {args.output}...")
    p_start = time.time()
    
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)
            
    timings["Writing CSV"] = time.time() - p_start
    
    # ----------------------------------------------------
    # Phase 8: Validate Output CSV
    # ----------------------------------------------------
    print("\n[Phase 8] Validating output CSV format...")
    p_start = time.time()
    is_valid = validate(args.output)
    timings["CSV Validation"] = time.time() - p_start
    
    total_elapsed = time.time() - total_start
    
    # Timing Report
    print("\n" + "=" * 40)
    print(" TIMING PROFILE SUMMARY ")
    print("=" * 40)
    for stage, elapsed in timings.items():
        print(f"  {stage:42}: {elapsed:6.3f} seconds")
    print("-" * 40)
    print(f"  {'Total Runtime':42}: {total_elapsed:6.3f} seconds")
    print("=" * 40)
    
    if not is_valid:
        print("CAUTION: Submission is INVALID! Please fix errors listed above.")
        sys.exit(1)
    else:
        print("Success! Submission CSV is 100% compliant.")

if __name__ == '__main__':
    main()
