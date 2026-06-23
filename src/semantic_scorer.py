import numpy as np
import os
import json
from src.config import EMBEDDING_MODEL, BATCH_SIZE

# Lazy loading of sentence-transformers model to save import time
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Load model on CPU explicitly
        _model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    return _model

def create_candidate_text(candidate):
    """
    Creates a detailed text representation for the candidate to ensure rich semantic mapping
    of summaries, career history, skills, education, and behavioral signals.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")
    years_exp = profile.get("years_of_experience", 0.0)
    summary = profile.get("summary", "")
    
    career = candidate.get("career_history", [])
    career_parts = []
    for job in career:
        comp = job.get("company", "")
        title = job.get("title", "")
        desc = job.get("description", "")
        career_parts.append(f"{comp}\n{title}\n{desc}")
    career_str = "\n\n".join(career_parts)
    
    skills = [s.get("name", "") for s in candidate.get("skills", [])]
    skills_str = "\n".join(skills)
    
    education = candidate.get("education", [])
    edu_parts = []
    for edu in education:
        inst = edu.get("institution", "")
        deg = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        edu_parts.append(f"{inst} - {deg} in {field}")
    edu_str = "\n".join(edu_parts)
    
    signals = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.0)
    notice = signals.get("notice_period_days", 90)
    behavior_str = f"Response Rate {response_rate:.0%}\nNotice Period {notice} days"
    
    parts = [
        f"{current_title}",
        f"{years_exp:.1f} years experience",
        f"{summary}",
        "Career:",
        f"{career_str}",
        "Skills:",
        f"{skills_str}",
        "Education:",
        f"{edu_str}",
        "Behavior:",
        f"{behavior_str}"
    ]
    return "\n\n".join([p.strip() for p in parts if p.strip()])

def compute_embeddings(texts, batch_size=BATCH_SIZE):
    """
    Computes SentenceTransformer embeddings for a list of texts on CPU.
    Optimized for fast CPU inference.
    """
    import torch
    # Prevent CPU core thrashing by limiting PyTorch threads
    if torch.get_num_threads() > 8:
        torch.set_num_threads(8)
        
    model = get_model()
    show_progress = len(texts) > 1000
    
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    return embeddings

def compute_similarity(jd_embedding, candidate_embeddings):
    """
    Computes cosine similarity between JD embedding and all candidate embeddings.
    Assumes embeddings are normalized. Returns scores scaled to 0-100.
    """
    # Dot product of normalized vectors is Cosine Similarity
    similarities = np.dot(candidate_embeddings, jd_embedding)
    # Scale from [-1, 1] to [0, 100]
    scaled_scores = (similarities + 1.0) / 2.0 * 100.0
    return scaled_scores

def precompute_and_save(candidates, output_path, id_map_path):
    """
    Compute and save candidate embeddings offline.
    """
    print(f"Generating rich texts for {len(candidates)} candidates...")
    texts = [create_candidate_text(c) for c in candidates]
    candidate_ids = [c.get("candidate_id", "") for c in candidates]
    
    print("Computing embeddings...")
    embeddings = compute_embeddings(texts)
    
    print(f"Saving embeddings to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    
    print(f"Saving candidate IDs mapping to {id_map_path}...")
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(candidate_ids, f)
        
    print("Precomputation finished successfully.")

def load_embeddings(path):
    """
    Loads pre-computed embeddings from a .npy file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found at {path}")
    return np.load(path)
