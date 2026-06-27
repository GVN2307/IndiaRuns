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

## Code Quality & Architecture Cleanups

1. **Dynamic JD Parser**: Overhauled `src/jd_parser.py` to be fully dynamic. Instead of relying solely on a pre-configured skill vocabulary, it dynamically extracts skills (proper nouns, acronyms, mixed-case words, list items, and parenthesized terms) directly from the text within identified job description sections, merging them with vocab matches for complete coverage. It also scans `.docx` files dynamically using `python-docx` (with text fallback), extracts experience ranges and locations via regular expressions, and builds the query embedding text dynamically.
2. **Unified Date Parsing**: Centralized `parse_date` in `src/config.py` and imported it in both `src/features.py` and `src/reasoning_generator.py`, removing duplicate date parsing logic.
3. **Cleaned Imports**: Moved `NON_TECH_TITLE_PATTERN` imports in `src/hybrid_aggregator.py` from inner-function level to module-level imports originating from configuration.
4. **Location Bonus Design**: Documented that location bonuses are applied mutually exclusively: candidates located in Pune/Noida receive the +15% location bonus; otherwise, candidates willing to relocate receive the +10% relocation willingness bonus.

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
   * **CPU Core Thrashing**: Set PyTorch threads dynamically based on CPU core count (`optimal_threads = min(12, cpu_cores - 4)` for high-core machines).
   * **Result**: Total execution time is **92.94 seconds (1.5 minutes)**, successfully meeting the **5-minute (300 seconds)** limit with substantial safety margin.

### Timing Profile Summary
* **JD Parsing**: 0.004 seconds
* **First-Stage FAISS Search**: 7.874 seconds
* **Candidate Streaming & Filtering**: 7.052 seconds
* **Candidates Scoring (Model Inference)**: 77.756 seconds
* **Aggregation, Sorting & Reasoning**: 0.133 seconds
* **Total Runtime**: **92.940 seconds**

---

## Streamlit Cloud Deployment & Memory Optimization Overhaul (V4)

In this phase, we prepared and deployed the candidate discovery and ranking pipeline as a public Streamlit application hosted on Streamlit Community Cloud. Due to the strict **1.0 GB RAM limit** of the hosting container, we performed major architectural and memory optimization cleanups to guarantee stability and prevent Out-Of-Memory (OOM) crashes:

### 1. Memory Optimization & OOM Prevention
* **Lazy Database Streaming (Saved ~350MB RAM):** Removed the startup candidate database caching logic (`load_candidates_db()`). Instead of keeping all 100,000 profile records in memory, candidate profiles are now streamed dynamically from `data/candidates.jsonl.gz` inside `run_interactive_pipeline` using a fast raw-string index scanner (`"candidate_id": "CAND_..."`). This extracts only the 250 FAISS-retrieved profiles on-the-fly, reducing startup database RAM footprint to **0 MB**.
* **Precomputed Results Cache & Lazy Model Loading (Saved ~520MB RAM, Startup Time <1s):** Removed the global model loading at startup. The ML models are loaded lazily inside the pipeline thread only when a live run is initiated. If the user loads the page for the first time, the dashboard loads results instantly from `data/default_results.json`, bypassing both model loading and pipeline execution. This keeps the initial startup memory footprint under 160MB and guarantees the `/healthz` check succeeds immediately.
* **First-Stage Model Memory Freeing (Saved ~120MB):** Immediately after executing the FAISS search, the first-stage embedding model (`all-MiniLM-L6-v2`) is deleted from memory, and `gc.collect()` is run to reclaim ~120MB of RAM before loading the heavier second-stage scoring models.
* **Garbage Collection Integration:** Explicitly triggered `gc.collect()` at the end of every pipeline run to instantly clean up temporary references and keep memory well under the 1.0 GB limit.

### 2. Self-Healing FAISS Index Downloader
* Because the binary FAISS index (`faiss_index.bin` ~153MB) is gitignored due to size, we implemented a robust, warning-bypass downloader to pull it from Google Drive.
* The downloader dynamically parses Google Drive's virus scan warning forms for large files (>100MB), extracts the hidden form action URLs, confirm tokens, and session identifiers, and downloads the binary file successfully using opener cookie sessions.
* Added size checks (>100MB) to prevent loop errors or loading corrupted HTML warning pages into the FAISS engine.

### 3. HTML/CSS Rendering Fixes
* In markdown rendering, any indented block (4+ spaces) is parsed as a code block. When technical skills badges or candidate table rows were concatenated with blank lines or indentation, the markdown engine parsed them as raw HTML code blocks.
* We converted dynamic HTML string generation (candidate list rows and matching skills badges) to use single contiguous inline strings without newlines or indentation. This guarantees the markdown engine parses and renders them as beautiful, styled HTML components.

### 4. Dependency Updates
* Added `torchvision` to `requirements.txt` to suppress optional import warning tracebacks printed in the Streamlit logs by `transformers`.

---

## 🎨 V4 UI/UX Redesign & Pipeline Refactoring

Following recruiter feedback and to improve demonstration clarity for judges, we implemented a significant frontend overhaul and code clean-up:

1. **CSS Decoupling:** Moved static styling rules out of `streamlit_app.py` into a dedicated [styles.css](file:///c:/Users/veera/Desktop/Codes%20for%20fun/India%20Runs/CVHunt/styles.css) file to clean up the code.
2. **Modularized Pipeline:** Refactored the monolithic `run_interactive_pipeline` into isolated helper functions (`retrieve_top_k`, `fetch_candidates`, `score_semantic`, `score_bm25`, `score_cross_encoder`, `score_structured`, `aggregate_scores`) to improve code testability and readability.
3. **O(1) Profiler Lookups:** Optimised candidate lookup in the profiler tab by changing sequential lists searches to fast O(1) dictionary maps (`results_dict` and `candidate_map`).
4. **Weights Expander:** Collapsed advanced weighting sliders under `st.expander` to simplify the UI.
5. **Step-by-Step Status Tracker:** Swapped the generic spinner for an interactive `st.status` steps tracker to show progress during ranking.
6. **KPI Performance Metric:** Added `st.metric` cards to showcase total execution runtime and ranked candidates size.
7. **System Specs Card:** Showcases system configuration parameters showing absolute compliance with air-gapped CPU-only rules.
8. **Download CSV Button:** Added a direct `st.download_button` to download the final CSV.
9. **Architecture Tab:** Added an `Architecture & Pipeline` tab displaying a custom theme-aware inline SVG flowchart and detailed descriptions.


