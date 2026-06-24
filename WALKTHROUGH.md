# Walkthrough - CVHunt Scoring & Gating Overhaul (V3)

We have successfully completed all Phase 1-6 updates to implement CrossEncoder reranking, BM25 scoring, shipper scores, retrieval/evaluation boosts, and structured recruiter reasonings.

---

## Enhancements Implemented in V3

### 1. Phase 1: Cross-Encoder Reranking
* Integrated `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` to rerank the top candidates retrieved by the FAISS vector index.
* Blended the CrossEncoder score into the final score:
  $$\text{final\_score} = 0.6 \times \text{current\_score} + 0.4 \times \text{rerank\_score}$$

### 2. Phase 2: Better Candidate Representation
* Overhauled `create_candidate_text` to produce a detailed text representation:
  * Current title and years of experience.
  * Profile summary.
  * Complete career history (company name, title, description).
  * List of skills and education.
  * Behavioral signals (Notice Period and Response Rate).

### 3. Phase 3: Shipper Score
* Added detection for system shipping experience using keywords: `"built"`, `"shipped"`, `"launched"`, `"deployed"`, `"production"`, `"users"`, `"latency"`, `"scale"`, `"ab testing"`, `"engagement"`.
* Computed `shipper_score = matches / 10` and applied the bonus: `final_score += shipper_score * 8`.

### 4. Phase 4: Retrieval & Evaluation Experience
* Detected retrieval experience using: `"retrieval"`, `"ranking"`, `"search"`, `"vector"`, `"embedding"`, `"faiss"`, `"pinecone"`, `"milvus"`, `"qdrant"`, `"weaviate"`, `"reranking"`.
* Detected evaluation experience using: `"ndcg"`, `"mrr"`, `"map"`, `"ab testing"`, `"evaluation"`, `"benchmark"`.
* Applied a `+15%` multiplier boost for retrieval experience, and a `+10%` multiplier boost for evaluation experience.

### 5. Phase 5: Add BM25
* Installed and integrated `rank-bm25`.
* Computed global BM25 scores over all 100,000 candidates dynamically.
* Blended the BM25 score into the aggregation:
  * weights: semantic (Embedding)=0.20, BM25=0.20, vector=0.15, structured=0.45.

### 6. Phase 6: Improved Reasoning
* Configured dynamic recruiter reasonings to strictly use the tone tier prefixes:
  * Ranks 1 to 10: `"Exceptional fit"`
  * Ranks 11 to 50: `"Strong fit"`
  * Ranks 51 to 100: `"Partial fit"`
* Every reasoning strictly outputs the format:
  `"{fit_phrase}: {years} years experience. Profile highlights {skills_names}. {behavioral_signal}. {concern_text}"`

---

## Validation & Verification Results

1. **Submission Format**: 100% Valid and compliant with `validate_submission.py`.
2. **Score Distribution**:
   * Ranks 1 to 10: mapped to the **95.0 – 100.0** range.
   * Ranks 11 to 50: mapped to the **70.0 – 95.0** range.
   * Ranks 51 to 100: mapped to the **30.0 – 70.0** range.
3. **No Non-Tech / Disqualified Candidates**: Verified that all non-tech candidates, consulting-only candidates, and honeypots are set to score `0.0`.
4. **Execution Time (Performance Optimization)**:
   * **Issue**: HuggingFace library model checking caused network timeouts in the offline environment, leading to a 7.1-minute runtime.
   * **Fix**: Added `os.environ["HF_HUB_OFFLINE"] = "1"` and `os.environ["TRANSFORMERS_OFFLINE"] = "1"` at the very top of `hackathon_rank.py` to prevent all internet lookup attempts and load model files directly from the local cache.
   * **Result**: Total execution time dropped to **96.48 seconds (1.6 minutes)**, well below the **5-minute (300 seconds)** limit.

### Timing Profile Summary
* **JD Parsing**: 0.001 seconds
* **First-Stage FAISS Search**: 10.676 seconds
* **Candidate Streaming & Filtering**: 7.356 seconds
* **Candidates Scoring (Model Inference)**: 78.258 seconds
* **Aggregation, Sorting & Reasoning**: 0.043 seconds
* **Total Runtime**: **96.480 seconds**
