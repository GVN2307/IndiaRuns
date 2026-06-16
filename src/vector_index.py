import faiss
import numpy as np
import os

def build_index(embeddings):
    """
    Builds a FAISS FlatIP (Inner Product) index.
    Assumes embeddings are already normalized (L2 norm = 1.0).
    """
    dim = embeddings.shape[1]
    # Inner product of normalized vectors is Cosine Similarity
    index = faiss.IndexFlatIP(dim)
    # Add embeddings to the index
    index.add(embeddings.astype('float32'))
    return index

def save_index(index, path):
    """
    Saves FAISS index to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_index(path):
    """
    Loads FAISS index from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)

def search(jd_embedding, index, k=1000):
    """
    Searches the FAISS index for the top-k nearest candidates to the JD embedding.
    Returns:
        indices: numpy array of candidate indices.
        distances: numpy array of inner product values (cosine similarities).
    """
    # Ensure jd_embedding is shape (1, dim) and float32
    query = np.expand_dims(jd_embedding.astype('float32'), axis=0)
    distances, indices = index.search(query, k)
    # Return flattened arrays for the single query
    return indices[0], distances[0]

def scores_from_search(indices, distances, total_candidates):
    """
    Converts FAISS search results to a 0-100 score for all candidates.
    Candidates in the top-k get their similarity scaled to 0-100.
    Non-matches (not in top-k) get a baseline low score of 10.0.
    """
    scores = np.full(total_candidates, 10.0, dtype=np.float32)
    
    if len(indices) == 0:
        return scores
        
    # Scale distances from [-1.0, 1.0] to [0.0, 100.0]
    scaled_distances = (distances + 1.0) / 2.0 * 100.0
    
    # Update scores for matching indices
    for idx, score in zip(indices, scaled_distances):
        if 0 <= idx < total_candidates:
            scores[idx] = score
            
    return scores
