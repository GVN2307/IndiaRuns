# CVHunt (talentrank-ai)

CVHunt is a high-performance candidate discovery and ranking system designed for the Redrob Intelligent Candidate Discovery & Ranking Challenge. It is designed to rank the top 100 candidates out of a pool of 100,000 candidates for a specific job description.

## Architecture Overview

```
                       +-------------------------+
                       |   Job Description (JD)  |
                       +------------+------------+
                                    |
                                    v
                             [ jd_parser.py ]
                                    |
                                    +-----------------------+
                                    |                       |
                                    v                       v
                             [ JD Requirements ]     [ JD Embedding ]
                                    |                       |
                                    |                       v
+-------------------------+         |             +---------+---------+
|  100K Candidates Pool   |         |             |   FAISS FlatIP    |
|  (candidates.jsonl.gz)  |         |             |      Index        |
+------------+------------+         |             +---------+---------+
             |                      |                       |
             v                      |                       v
      (data_loader.py)              |               (vector_index.py)
             |                      |                       |
             +--------+-------------+                       |
             |        |             |                       |
             v        v             v                       v
      [Semantic]  [Structured]  [Honeypot]               [Vector]
       Scoring     Scoring      Detection                Scoring
       Score 1     Score 2        Cap                    Score 3
        (35%)       (40%)      (max=20)                   (25%)
             |        |             |                       |
             +--------+-------------+-----------------------+
                                    |
                                    v
                          [ hybrid_aggregator.py ]
                        (Aggregates + Multipliers)
                                    |
                                    v
                           [ Tie-breaking ]
                       (Ascending candidate_id)
                                    |
                                    v
                        [ reasoning_generator.py ]
                          (1-2 sentence reasons)
                                    |
                                    v
                            [ validator.py ]
                       (submission.csv output)
```

## The 3-Score Hybrid System

CVHunt aggregates three distinct scorers to compute a robust candidate match:

1. **Semantic Similarity Scorer (35% weight)**: Generates a dense textual profile for each candidate (Headline + Summary + Skills + Career History Descriptions) and calculates the Cosine Similarity against the Job Description embedding using the `sentence-transformers/all-MiniLM-L6-v2` model.
2. **Structured Feature Scorer (40% weight)**: A rule-based feature evaluator matching experience years (with a Gaussian decay centering at 7 years), product/consulting company ratios, title consistency, education tiers, and behavioral signals (recruiter response rate, activity status, notice period). It also blends predictions (50% weight) from a local **Gradient Boosting Regressor** surrogate trained to predict expert recruiter rankings.
3. **Vector Index Scorer (25% weight)**: Uses a fast local **FAISS FlatIP Index** built on candidate embeddings. Returns similarity scores for candidates retrieved in the top $k=2000$ and penalizes non-retrieved candidates with a default low score.

## Why Local Embeddings Over API Calls

* **Latency & Scale**: Running API calls (e.g. OpenAI or Gemini) for 100,000 candidates violates the 5-minute compute budget on CPU and creates high network-bound latency. Local embeddings via `sentence-transformers` on CPU take less than 1 second to score the entire 100K pool when pre-computed.
* **Cost Efficiency**: No API costs are incurred during the ranking step, making the system highly scalable in production.
* **Offline Compliance**: Fully complies with the off-network rule during execution.

## Honeypot Detection

CVHunt implements a 6-tier anomaly detector to identify fake/impossible profiles:
* **Timeline Impossibility**: Detects if candidates claim to have worked at a company before its actual founding year (e.g. working at CRED in 2017 when it was founded in 2018).
* **Skill-Proficiency Mismatch**: Flags candidates claiming "expert/advanced" proficiency with 0 months of experience, or candidates claiming more than 15 expert skills.
* **Title-Description Mismatch**: Identifies candidates holding technical titles (e.g., Senior AI Engineer) but possessing career histories containing only non-technical details (e.g. sales, marketing, customer support).
* **Keyword Stuffing**: Flags profiles listing >20 skills with an average duration of <12 months.
* **Consulting Trap**: Identifies candidates whose career is 100% consulting-only but who have stuffed multiple AI buzzwords to game semantic algorithms.
* **Plain-Language Tier 5**: Flags candidates claiming expert-level AI roles but whose descriptions use generic plain language without mentioning specific libraries (e.g., PyTorch, FAISS) and who graduated from low-tier colleges.

Flagged honeypot candidates have their scores capped at **20.0** to ensure they do not contaminate the top 100.

## Setup & Execution

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
Execute the ranker using the precomputed artifacts:
```bash
.\.venv\Scripts\python hackathon_rank.py --candidates data/candidates.jsonl.gz --jd data/job_description.md --output output/submission.csv --embeddings models/candidate_embeddings.npy --index models/faiss_index.bin
```

## Compute Requirements
* **Memory**: $\leq$ 16 GB RAM
* **CPU**: $\leq$ 5 minutes execution time (usually runs in ~10 seconds with precomputed embeddings)
* **GPU**: None required
* **Network**: None required during ranking

## Approach Summary
CVHunt utilizes a hybrid retrieval and ranking architecture combining dense retrieval with rule-based heuristics. Candidate text summaries are parsed into dense vector representations using a SentenceTransformer model and indexed in a FAISS FlatIP index to enable fast cosine similarity retrieval. To incorporate structured candidate parameters and availability, a feature extraction engine extracts engineering, experience, education, and behavioral features. These features are evaluated using rule-based scoring (penalizing consulting firm locks, prioritizing product experience, and applying a Gaussian experience peak at 7 years) and blended with a lightweight Gradient Boosting regressor surrogate. An anomaly-based honeypot detector runs 6 verification filters to detect impossible profiles, capping flagged candidates at a score of 20. Aggregated scores are adjusted by recruiter response rates and activity multipliers, and sorted with a deterministic tie-breaker on candidate IDs ascending. The system runs end-to-end in ~10 seconds on a single CPU.
