# CVHunt (talentrank-ai)

CVHunt is a high-performance candidate discovery and ranking system designed for the Redrob Intelligent Candidate Discovery & Ranking Challenge. It retrieves and ranks the top 100 candidates out of a pool of 100,000 candidates for a specific job description under a strict 5-minute execution limit on CPU.

---

## 🚀 Key Improvements in V3
* **Cross-Encoder Reranking**: Integrates a `ms-marco-MiniLM-L-6-v2` Cross-Encoder to perform second-pass semantic reranking on FAISS candidate matches.
* **4-Score Hybrid Scorer**: Combines semantic vectors, classical BM25 keyword matching, structured rules/surrogate models, and FAISS indexing.
* **Graduated Must-Have Gating**: Replaces binary gating with a graduated penalty system based on the count of must-have skills matching the JD.
* **Shipper & Domain Experience Boosts**: Detects systems-shipping experience, retrieval, and evaluation keywords to reward top practitioners.
* **Instant Offline Execution**: Configures PyTorch/Transformers to run in offline mode (`HF_HUB_OFFLINE="1"`), bypassing network checking timeouts and bringing runtime down to **~96 seconds**.

---

## 1. Architecture Overview

```
                        +-------------------------+
                        |   Job Description (JD)  |
                        +------------+------------+
                                     |
                                     v
                              [ jd_parser.py ]
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
       [ JD Requirements ]                        [ JD Embedding ]
                |                                         |
                |                                         v
   +------------+------------+                  +---------+---------+
   |  100K Candidates Pool   |                  |   FAISS FlatIP    |
   |  (candidates.jsonl.gz)  |                  |      Index        |
   +------------+------------+                  +---------+---------+
                |                                         |
                v (data_loader.py)                        v (vector_index.py)
                |                                         |
                +--------------------+--------------------+
                                     |
                                     v [ Candidate Filtering ]
                                     | (Top 1000 Retrieved)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        [Structured Scorer]                        [BM25 Scorer]
        • GBR surrogate model                      • rank-bm25
        • Weight: 45%                              • Weight: 20%
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │ (Vector / FAISS Weight: 15%)
                                     ▼
                          [ Intermediate Ranker ]
                          • Filters honeypots/disqualifications
                          ▼ [ Prune to Top 150 ]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
         [Semantic Scorer]                    [Cross-Encoder Reranker]
         • mpnet-base-v2                      • ms-marco-MiniLM-L-6-v2
         • Weight: 20%                        • Blend: 0.6 * base + 0.4 * CE
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                            [ Score Mapping ]
                            │ (Tiers: 95-100, 70-95, 30-70)
                            ▼
                         [ reasoning_generator.py ]
                         • Dynamic justifications
                            │
                            ▼
                   [ Output submission.csv / UI ]
```

Detailed explanation of each module, its design constraints, mathematical formulations, and engineering decisions can be found in the comprehensive **[SYSTEM_MANUAL.md](file:///c:/Users/veera/Desktop/Codes%20for%20fun/India%20Runs/CVHunt/SYSTEM_MANUAL.md)**.
File concepts and structure summaries are also documented in **[ARCHITECTURE.md](file:///c:/Users/veera/Desktop/Codes%20for%20fun/India%20Runs/CVHunt/ARCHITECTURE.md)**.

---

## 2. The 4-Score Hybrid System

CVHunt aggregates four scoring engines to rank candidates:

1. **Semantic Similarity Scorer (20% weight)**: Generates a dense textual profile for candidates (Headline + Summary + Skills + Career History + Education + Behavior) and calculates the Cosine Similarity against the Job Description embedding using `sentence-transformers/all-mpnet-base-v2`. It includes a `+12.0` point keyword boost for critical JD matches.
2. **BM25 Keyword Scorer (20% weight)**: Computes classical BM25 relevance scores over the candidate rich text documents to capture specific syntax matches.
3. **FAISS Vector Index Scorer (15% weight)**: Uses a local FAISS FlatIP Index built on candidates' `all-MiniLM-L6-v2` embeddings. Candidates in the top search range get their raw search similarity scaled, while others get a baseline minimum score.
4. **Structured Feature Scorer (45% weight)**: Evaluates:
   * Experience years using a Gaussian decay function peaking at 7 years.
   * Product-vs-consulting company ratios.
   * Education tiers (Tier-1, Tier-2, Tier-3 mapping).
    * An optional **Gradient Boosting Regressor** surrogate model trained on heuristic scores to smooth structured ranking (50% blend).

---

## 3. Advanced Filtering & Gating

* **Relative Must-Have Skill Gating**: Candidates are evaluated on the Job Description's must-have skills (e.g. embeddings, RAG, PyTorch).
  * Candidates with less than the minimum required must-have skills are **disqualified** (final score = 0.0).
  * Applies graduated penalty multipliers (from `0.15` to `0.80`) based on the candidate's match ratio of the Job Description's must-have skills.
* **Honeypot Detection**: Runs 8 rules to identify fraudulent profiles (such as claiming credentials before a company was founded, or claiming expert proficiency with 0 experience). Flagged candidates receive a score of `0.0`.
* **Hard Disqualifications**: Auto-rejects candidates with zero product company experience, computer vision engineers without generic AI/ML background, or candidates with $\geq 95\%$ consulting firm ratios.

---

## 4. Setup & Execution

### 📋 Prerequisites
* Python 3.9+
* Pip

### 1. Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\pip install -r requirements.txt

# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Offline Pre-Computation
Compute embeddings, build the FAISS index, and train the surrogate model offline:
```bash
# On Windows PowerShell:
.\.venv\Scripts\python scripts/precompute_embeddings.py
.\.venv\Scripts\python scripts/build_faiss_index.py
.\.venv\Scripts\python scripts/train_surrogate.py
```

### 3. Running the Ranker
Execute the ranker script using the precomputed artifacts:
```bash
.\.venv\Scripts\python hackathon_rank.py --candidates data/candidates.jsonl.gz --jd data/job_description.md --output output/submission.csv --embeddings models/candidate_embeddings.npy --index models/faiss_index.bin
```

---

## 5. Compute Requirements

* **Memory**: $\leq$ 16 GB RAM (runs in under 1.0 GB RAM on Streamlit Community Cloud).
* **CPU Only**: Run fully on CPU. Setting offline variables ensures loading model parameters is instantaneous:
  ```python
  os.environ["HF_HUB_OFFLINE"] = "1"
  os.environ["TRANSFORMERS_OFFLINE"] = "1"
  ```
* **Runtime**: ~96 seconds (well under the 5-minute competition limit).

---

## 6. Output & Validation
The results are output to `output/submission.csv` containing columns: `candidate_id`, `rank`, `score`, `reasoning`. 
* **Target score distribution**:
  * Ranks 1-10: **95.0 – 100.0**
  * Ranks 11-50: **70.0 – 95.0**
  * Ranks 51-100: **30.0 – 70.0**
* **Reasoning Justification**: Dynamically formatted using rank-consistent tone prefixes (`Exceptional fit`, `Strong fit`, `Partial fit`), highlighting years of experience, key matching skills, behavioral signals, and honest recruitment concerns.

---

## 7. Streamlit Cloud Web Dashboard

We provide a memory-optimized recruiter dashboard deployed at **[indiaruns.streamlit.app](https://indiaruns.streamlit.app/)**.

### ⚙️ Memory-Limit Design (1.0 GB RAM Constraints)
Because Streamlit Community Cloud has a hard 1.0 GB container RAM limit, we implemented the following performance constraints:
1. **Lazy Database Streaming:** Instead of loading all 100,000 candidate profiles into memory at startup (which uses 350MB+), candidate profiles are scanned and parsed on-the-fly from the gzip file `data/candidates.jsonl.gz` inside the query thread. We use a fast raw-string index scan to extract only the profiles matching the first-stage FAISS search.
2. **First-Stage Model Unloading:** Immediately after retrieving candidate IDs from the FAISS vector index, the first-stage embedding model (`all-MiniLM-L6-v2`) is deleted and memory is reclaimed via `gc.collect()` before starting the heavier second-stage scoring modules.
3. **Garbage Collection:** We trigger garbage collection after every discovery search thread executes to prevent memory leaks.
4. **Self-Healing Index Downloader:** Since `faiss_index.bin` (~153MB) is too large for GitHub pushes, the application includes a direct download engine that automatically parses Google Drive's virus scan warning forms and downloads the binary file on first load.
5. **Dedent formatting wrapper:** To prevent dynamic HTML tables and skills badges from rendering as plain text code blocks in markdown, all multi-line markdown rendering is automatically dedented.

---

## 8. 🧪 Automated Testing

We provide a comprehensive unit and integration test suite under `tests/` to verify candidate scoring, disqualification thresholds, and ranking computations:

```bash
# Run all tests locally
python -m unittest discover -s tests -p "test_*.py"
```
