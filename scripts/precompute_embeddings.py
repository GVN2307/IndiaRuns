import sys
import os
import time

# Add base directory to path so we can import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.config import CANDIDATES_PATH, EMBEDDINGS_PATH, ID_MAP_PATH
from src.data_loader import load_all_candidates
from src.semantic_scorer import precompute_and_save

def main():
    print("=" * 60)
    print(" CVHUNT - Embedding Precomputation ")
    print("=" * 60)
    
    start_time = time.time()
    
    print(f"Loading candidates from {CANDIDATES_PATH}...")
    candidates = load_all_candidates(CANDIDATES_PATH)
    print(f"Loaded {len(candidates)} candidates.")
    
    precompute_and_save(candidates, EMBEDDINGS_PATH, ID_MAP_PATH)
    
    elapsed = time.time() - start_time
    print(f"Precomputation completed in {elapsed:.2f} seconds ({elapsed / 60.0:.2f} minutes).")
    print("=" * 60)

if __name__ == '__main__':
    main()
