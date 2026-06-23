import os
import re

NON_TECH_TITLE_PATTERN = re.compile(
    r"\b(civil|mechanical|electrical|chemical|sales|marketing|hr|human resources|recruiter|recruiting|"
    r"talent acquisition|accountant|accounting|finance|customer support|data entry|office manager|"
    r"business analyst|financial analyst|operations manager|project manager|graphic designer|"
    r"ui designer|ux designer|product manager|scrum master|it manager|it director)\b|ui/ux designer",
    re.IGNORECASE
)


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.jsonl.gz")
JD_PATH = os.path.join(DATA_DIR, "job_description.md")
SAMPLE_CANDIDATES_PATH = os.path.join(DATA_DIR, "sample_candidates.json")

EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "candidate_embeddings.npy")
INDEX_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")
SURROGATE_PATH = os.path.join(MODELS_DIR, "surrogate_model.pkl")
ID_MAP_PATH = os.path.join(MODELS_DIR, "candidate_ids.json")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ML & Indexing Hyperparameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 128
TOP_K = 100

# Experience Rules
IDEAL_YEARS_MIN = 5
IDEAL_YEARS_MAX = 9
IDEAL_YEARS_PEAK = 7

# Company lists
CONSULTING_COMPANIES = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "Tech Mahindra", "HCL", "Genpact"]
PRODUCT_COMPANIES = ["Swiggy", "Zomato", "Uber", "CRED", "Razorpay", "Ola", "Flipkart", "Amazon", "Google", "Microsoft"]

# Company Founding Years (for timeline checks)
FOUNDING_YEARS = {
    "Swiggy": 2014,
    "Zomato": 2008,
    "Uber": 2009,
    "CRED": 2018,
    "Razorpay": 2014,
    "Ola": 2010,
    "Flipkart": 2007,
    "Amazon": 1994,
    "Google": 1998,
    "Microsoft": 1975,
    "TCS": 1968,
    "Infosys": 1981,
    "Wipro": 1945,
    "Accenture": 1989,
    "Cognizant": 1994,
    "Capgemini": 1967,
    "Tech Mahindra": 1986,
    "HCL": 1976,
    "Redrob AI": 2023,
    "Redrob": 2023,
    "Cure.fit": 2016,
    "Curefit": 2016,
    "PhonePe": 2015,
    "Paytm": 2010,
    "BharatPe": 2018,
    "Groww": 2016,
    "Meesho": 2015,
    "Zepto": 2021,
    "Blinkit": 2013
}

# Skill criteria
MUST_HAVE_SKILLS = [
    "embeddings", "vector search", "retrieval", "ranking", "Python", "LLM", 
    "fine-tuning", "RAG", "sentence-transformers", "FAISS", "Pinecone", 
    "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", 
    "evaluation", "NDCG", "MRR", "MAP", "A/B testing"
]

NICE_TO_HAVE_SKILLS = [
    "LoRA", "QLoRA", "PEFT", "learning-to-rank", "XGBoost", "LightGBM", 
    "distributed systems", "MLOps", "Kubeflow", "MLflow", "BentoML"
]

# Weights and scoring parameters
PROFICIENCY_WEIGHTS = {
    "beginner": 0.2,
    "intermediate": 0.5,
    "advanced": 0.8,
    "expert": 1.0
}

SCORE_WEIGHTS = {
    "semantic": 0.20,
    "bm25": 0.20,
    "vector": 0.15,
    "structured": 0.45
}

BEHAVIORAL_MULTIPLIERS = {
    "inactive_penalty": 0.40,       # last_active > 90 days AND not open_to_work
    "low_response_penalty": 0.40,   # recruiter_response_rate < 0.2
    "long_notice_penalty": 0.60,    # notice_period_days > 90
    "low_completion_penalty": 0.60  # interview_completion_rate < 0.3
}

# Global Reference Date (for calculating active days)
# Matches the execution time of 2026-06-16
REFERENCE_DATE = "2026-06-16"
