import argparse
import os
import sys
import time
import json
import csv
import numpy as np

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import SUBMISSION_PATH, SURROGATE_PATH
from src.data_loader import stream_candidates
from src.jd_parser import parse_jd
from src.features import extract_features
from src.semantic_scorer import (
    create_candidate_text, compute_embeddings, compute_similarity, load_embeddings
)
from src.structured_scorer import compute_structured_score
from src.vector_index import load_index, build_index, search, scores_from_search
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
    print(" CVHUNT - Candidate Ranking Pipeline ")
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
    print(f"Location preference: {jd_reqs.location_preference}")
    print(f"Ideal Experience: {jd_reqs.ideal_experience_range} years")
    
    # Construct a rich query representation for embedding retrieval
    query_text = (
        "Senior AI Engineer. Deployed embeddings-based retrieval systems, vector databases, hybrid search. "
        "Experience with sentence-transformers, FAISS, Pinecone, Weaviate, Milvus, Qdrant, OpenSearch, Elasticsearch. "
        "Strong Python, evaluation frameworks (NDCG, MRR, MAP, A/B testing). "
        "Fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank, XGBoost, LightGBM, distributed systems. "
        "Product company experience. Located in Noida or Pune, India. 5-9 years experience."
    )

    # ----------------------------------------------------
    # Phase 2: Stream Candidates & Extract Features
    # ----------------------------------------------------
    print("\n[Phase 2] Streaming candidates & extracting features...")
    p_start = time.time()
    candidates = []
    features_list = []
    flagged_ids = {} # candidate_id -> reason
    
    # We will accumulate the data in memory. Feature extraction takes ~2-3s for 100K.
    for cand in stream_candidates(args.candidates):
        candidates.append(cand)
        cid = cand["candidate_id"]
        
        # Extract features
        feats = extract_features(cand)
        features_list.append(feats)
        
        # Run honeypot detection
        is_flagged, reason = run_honeypot_checks(cand)
        if is_flagged:
            flagged_ids[cid] = reason
            
    total_candidates = len(candidates)
    timings["Candidate Streaming & Feature Extraction"] = time.time() - p_start
    print(f"Processed {total_candidates} candidates.")
    print(f"Flagged {len(flagged_ids)} honeypot candidates.")

    # ----------------------------------------------------
    # Phase 3: Load or Compute Semantic Embeddings
    # ----------------------------------------------------
    print("\n[Phase 3] Loading or computing semantic embeddings...")
    p_start = time.time()
    
    # Encode query JD text
    print("Encoding query JD vector...")
    # This lazily loads SentenceTransformer on CPU
    jd_embedding = compute_embeddings([query_text])[0]
    
    if args.embeddings and os.path.exists(args.embeddings):
        print(f"Loading precomputed candidate embeddings from {args.embeddings}...")
        candidate_embeddings = load_embeddings(args.embeddings)
    else:
        print("Precomputed embeddings not found. Computing embeddings online (this will take time on CPU!)...")
        rich_texts = [create_candidate_text(c) for c in candidates]
        candidate_embeddings = compute_embeddings(rich_texts)
        
    timings["Embeddings Processing"] = time.time() - p_start
    print(f"Embeddings loaded/computed: {candidate_embeddings.shape}")

    # ----------------------------------------------------
    # Phase 4: Load or Build FAISS Index
    # ----------------------------------------------------
    print("\n[Phase 4] Loading or building FAISS index...")
    p_start = time.time()
    if args.index and os.path.exists(args.index):
        print(f"Loading precomputed FAISS index from {args.index}...")
        index = load_index(args.index)
    else:
        print("Precomputed index not found. Building FlatIP FAISS index online...")
        index = build_index(candidate_embeddings)
        
    timings["FAISS Index Processing"] = time.time() - p_start

    # ----------------------------------------------------
    # Phase 5: Compute Scores
    # ----------------------------------------------------
    print("\n[Phase 5] Scoring candidates...")
    p_start = time.time()
    
    # 5.1 Score 1: Semantic Similarity
    print("Computing Score 1 (Semantic Similarity)...")
    semantic_scores = {}
    sem_raw_scores = compute_similarity(jd_embedding, candidate_embeddings)
    for cand, score in zip(candidates, sem_raw_scores):
        semantic_scores[cand["candidate_id"]] = float(score)
        
    # 5.2 Score 2: Structured Scorer
    print("Computing Score 2 (Structured Scoring)...")
    structured_scores = {}
    structured_breakdowns = {}
    
    # Check if surrogate model exists in models/
    # If not provided via path, check models/surrogate_model.pkl
    s_path = args.embeddings.replace("candidate_embeddings.npy", "surrogate_model.pkl") if args.embeddings else SURROGATE_PATH
    if not os.path.exists(s_path):
        s_path = SURROGATE_PATH
        
    for cand, feats in zip(candidates, features_list):
        cid = cand["candidate_id"]
        score, breakdown = compute_structured_score(feats, surrogate_path=s_path)
        structured_scores[cid] = score
        structured_breakdowns[cid] = breakdown
        
    # 5.3 Score 3: FAISS Vector Index Scorer
    print("Computing Score 3 (Vector Retrieval Similarity)...")
    # Search top 2000 (enough to cover our top-100 hybrid aggregates)
    k_search = min(2000, total_candidates)
    retrieved_indices, retrieved_distances = search(jd_embedding, index, k=k_search)
    vector_scores_arr = scores_from_search(retrieved_indices, retrieved_distances, total_candidates)
    
    vector_scores = {}
    for idx, cand in enumerate(candidates):
        vector_scores[cand["candidate_id"]] = float(vector_scores_arr[idx])
        
    timings["Candidates Scoring"] = time.time() - p_start

    # ----------------------------------------------------
    # Phase 6: Aggregate & Sort
    # ----------------------------------------------------
    print("\n[Phase 6] Aggregating scores & sorting...")
    p_start = time.time()
    ranked_results = compute_final_ranking(
        candidates_data=candidates,
        semantic_scores=semantic_scores,
        structured_scores=structured_scores,
        structured_breakdowns=structured_breakdowns,
        vector_scores=vector_scores,
        flagged_ids=flagged_ids,
        features_list=features_list
    )
    timings["Aggregation & Sorting"] = time.time() - p_start
    print(f"Top candidate score: {ranked_results[0][1]:.3f} (ID: {ranked_results[0][0]})")

    # ----------------------------------------------------
    # Phase 7: Generate Reasoning
    # ----------------------------------------------------
    print("\n[Phase 7] Generating justifications for top 100...")
    p_start = time.time()
    top_100 = ranked_results[:100]
    
    final_rows = []
    for rank_idx, (cid, score, breakdown) in enumerate(top_100):
        rank = rank_idx + 1
        reasoning = generate_reasoning(breakdown)
        final_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning
        })
        
    timings["Reasoning Generation"] = time.time() - p_start

    # ----------------------------------------------------
    # Phase 8: Write Output CSV
    # ----------------------------------------------------
    print(f"\n[Phase 8] Writing results to {args.output}...")
    p_start = time.time()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)
            
    timings["Writing CSV"] = time.time() - p_start
    print(f"Successfully wrote {len(final_rows)} rows.")

    # ----------------------------------------------------
    # Phase 9: Validate Output CSV
    # ----------------------------------------------------
    print("\n[Phase 9] Validating output CSV format...")
    p_start = time.time()
    is_valid = validate(args.output)
    timings["CSV Validation"] = time.time() - p_start
    
    total_elapsed = time.time() - total_start
    
    # ----------------------------------------------------
    # Timing Report
    # ----------------------------------------------------
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
