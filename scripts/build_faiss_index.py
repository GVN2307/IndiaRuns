import sys
import os
import time
import numpy as np

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.config import EMBEDDINGS_PATH, INDEX_PATH
from src.vector_index import build_index, save_index, search

def main():
    print("=" * 60)
    print(" CVHUNT - FAISS Index Builder ")
    print("=" * 60)
    
    start_time = time.time()
    
    print(f"Loading pre-computed embeddings from {EMBEDDINGS_PATH}...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: Embeddings not found at {EMBEDDINGS_PATH}. Run precompute_embeddings.py first.")
        sys.exit(1)
        
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded embeddings matrix of shape {embeddings.shape}.")
    
    print("Building FAISS FlatIP index...")
    index = build_index(embeddings)
    
    print(f"Saving index to {INDEX_PATH}...")
    save_index(index, INDEX_PATH)
    
    # Run a test query search
    print("Testing index with a dummy search query...")
    dummy_query = embeddings[0] # Search using candidate 0
    indices, distances = search(dummy_query, index, k=5)
    print(f"Closest candidates indices: {indices}")
    print(f"Similarities (dot products): {distances}")
    
    elapsed = time.time() - start_time
    print(f"FAISS index built and saved in {elapsed:.2f} seconds.")
    print("=" * 60)

if __name__ == '__main__':
    main()
