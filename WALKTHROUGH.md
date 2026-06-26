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
* **Fix**: To prevent score stuffing (e.g. candidate repeating the word "built" 10 times to inflate the score), the logic was updated to count **unique matches only** and **cap the bonus at 5 matches** max:
  $$\text{final\_score} += \min(\text{unique\_shipper\_matches}, 5) \times 0.8$$

### 4. Phase 4: Retrieval & Evaluation Experience
* Centralized and mapped retrieval and evaluation experience keyword lists:
  * `RETRIEVAL_KEYWORDS`: `"retrieval"`, `"ranking"`, `"search"`, `"vector"`, `"embedding"`, `"faiss"`, `"pinecone"`, `"milvus"`, `"qdrant"`, `"weaviate"`, `"reranking"`.
  * `EVALUATION_KEYWORDS`: `"ndcg"`, `"mrr"`, `"map"`, `"ab testing"`, `"evaluation"`, `"benchmark"`.
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
4. **Graduated Gating Relaxation**: Relaxed the gating threshold from a hard reject at $<3$ must-haves to:
   * $< 2$ must-haves: **Disqualified** (final score = 0.0)
   * $< 3$ must-haves: final score $\times 0.15$
   This prevents the exclusion of strong candidates who use semantic variations of key terms.
5. **Execution Time (Performance Optimization)**:
   * **Model Loading Timeout**: Configured `HF_HUB_OFFLINE="1"` and `TRANSFORMERS_OFFLINE="1"` to bypass network timeouts in the offline environment, dropping load time from 3+ minutes to under 1 second.
   * **CPU Core Thrashing**: Added `torch.set_num_threads(4)` at the top of the entry point script to prevent synchronization overhead on high-core (16-core) machines during candidate encoding.
   * **Result**: Total execution time is **278.68 seconds (4.6 minutes)**, successfully meeting the **5-minute (300 seconds)** limit.

### Timing Profile Summary
* **JD Parsing**: 0.002 seconds
* **First-Stage FAISS Search**: 36.764 seconds
* **Candidate Streaming & Filtering**: 26.088 seconds
* **Candidates Scoring (Model Inference)**: 215.630 seconds
* **Aggregation, Sorting & Reasoning**: 0.066 seconds
* **Total Runtime**: **278.682 seconds**

