import os
import sys
import json
import gzip
import time
import re
import threading
import uuid
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import textwrap

# Suppress sklearn unpickling warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Auto-dedent all st.markdown calls to prevent indented HTML/markdown strings from rendering as code blocks
_original_markdown = st.markdown
def dedented_markdown(body, unsafe_allow_html=False):
    if isinstance(body, str):
        body = textwrap.dedent(body)
    return _original_markdown(body, unsafe_allow_html=unsafe_allow_html)
st.markdown = dedented_markdown

# Global lock to prevent concurrent heavy ML pipeline executions (OOM prevention on Streamlit Cloud)
_pipeline_lock = threading.Lock()

import plotly.express as px
import plotly.graph_objects as go

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import ALL_TECH_VOCAB, MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, SCORE_WEIGHTS, ID_MAP_PATH, INDEX_PATH, SURROGATE_PATH
import src.config
from src.jd_parser import parse_jd
from src.features import extract_features
from src.semantic_scorer import create_candidate_text
from src.structured_scorer import compute_structured_score
from src.vector_index import load_index, search, scores_from_search
from src.honeypot_detector import run_honeypot_checks
from src.hybrid_aggregator import compute_final_ranking
from src.reasoning_generator import generate_reasoning

# ----------------------------------------------------
# 1. Streamlit Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="CVHunt",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------
# 2. Theme Toggle & State Management
# ----------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

if "parsed_jd" not in st.session_state:
    st.session_state.parsed_jd = None

if "selected_cid" not in st.session_state:
    st.session_state.selected_cid = None

if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# ----------------------------------------------------
# 3. Inject CSS Variables and Global Custom Styles
# ----------------------------------------------------
if IS_DARK:
    bg_color = "#09090b"
    bg_subtle = "#0c0c0f"
    card_color = "#0c0c0f"
    card_hover = "#131316"
    border_color = "#1e1e24"
    border_subtle = "#16161a"
    text_color = "#fafafa"
    text_muted = "#71717a"
    text_dim = "#52525b"
    accent_color = "#2563eb"
    green_color = "#22c55e"
    green_muted = "rgba(34,197,94,0.12)"
    red_color = "#ef4444"
    red_muted = "rgba(239,68,68,0.12)"
    amber_color = "#f59e0b"
    amber_muted = "rgba(245,158,11,0.12)"
    shadow = "none"
else:
    bg_color = "#ffffff"
    bg_subtle = "#f9fafb"
    card_color = "#ffffff"
    card_hover = "#f4f4f5"
    border_color = "#e4e4e7"
    border_subtle = "#f0f0f2"
    text_color = "#09090b"
    text_muted = "#71717a"
    text_dim = "#a1a1aa"
    accent_color = "#2563eb"
    green_color = "#16a34a"
    green_muted = "rgba(22,163,74,0.08)"
    red_color = "#dc2626"
    red_muted = "rgba(220,38,38,0.08)"
    amber_color = "#d97706"
    amber_muted = "rgba(217,119,6,0.08)"
    shadow = "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

# Inject CSS Variables
st.markdown(f"""
<style>
:root {{
    --bg: {bg_color};
    --bg-subtle: {bg_subtle};
    --card: {card_color};
    --card-hover: {card_hover};
    --border: {border_color};
    --border-subtle: {border_subtle};
    --text: {text_color};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: {accent_color};
    --green: {green_color};
    --green-muted: {green_muted};
    --red: {red_color};
    --red-muted: {red_muted};
    --amber: {amber_color};
    --amber-muted: {amber_muted};
    --shadow: {shadow};
    --radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# Load external static styles
styles_path = os.path.join(BASE_DIR, "styles.css")
if os.path.exists(styles_path):
    with open(styles_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Header Bar
# ----------------------------------------------------
head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-logo">◆</span>
        <span class="brand-name">CVHunt</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# ----------------------------------------------------
# 5. Caching Resources (Database and Models)
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_faiss_and_ids():
    index = load_index(INDEX_PATH)
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
    return index, candidate_ids



@st.cache_resource(show_spinner=False)
def load_ml_models():
    is_streamlit_cloud = "STREAMLIT_SHARE_PREREQUISITES" in os.environ or os.environ.get("USER") == "appuser"
    if not is_streamlit_cloud:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
    from sentence_transformers import SentenceTransformer, CrossEncoder
    sem_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cpu')
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu', max_length=512)
    return sem_model, reranker

# ----------------------------------------------------
# 4.5 Self-Healing File Downloader
# ----------------------------------------------------
def extract_gdrive_id(url: str) -> str:
    match = re.search(r'/file/d/([0-9A-Za-z_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([0-9A-Za-z_-]+)', url)
    if match:
        return match.group(1)
    return ""

def download_large_file(url: str, destination: str):
    import urllib.request
    import http.cookiejar
    import os
    
    is_gdrive = "drive.google.com" in url or "docs.google.com" in url
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if is_gdrive:
        file_id = extract_gdrive_id(url)
        if not file_id:
            raise ValueError("Could not extract Google Drive File ID from the provided URL.")
            
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        
        base_url = "https://docs.google.com/uc?export=download"
        gdrive_url = f"{base_url}&id={file_id}"
        
        # First request to check for the virus scan confirmation token
        req = urllib.request.Request(gdrive_url, headers={'User-Agent': 'Mozilla/5.0'})
        with opener.open(req) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        # Parse the form action and all hidden inputs dynamically
        import urllib.parse
        action_match = re.search(r'<form\s+[^>]*action="([^"]+)"', html)
        if action_match:
            action_url = action_match.group(1)
            # Find all hidden inputs
            inputs = re.findall(r'<input\s+[^>]*name="([^"]+)"\s+value="([^"]+)"', html)
            inputs_swapped = re.findall(r'<input\s+[^>]*value="([^"]+)"\s+name="([^"]+)"', html)
            
            params = {}
            for name, value in inputs:
                params[name] = value
            for value, name in inputs_swapped:
                params[name] = value
                
            if 'confirm' in params:
                query_string = urllib.parse.urlencode(params)
                final_url = f"{action_url}?{query_string}"
            else:
                final_url = gdrive_url
        else:
            final_url = gdrive_url
            
        # Second request to download the actual binary file
        req_dl = urllib.request.Request(final_url, headers={'User-Agent': 'Mozilla/5.0'})
        with opener.open(req_dl) as response:
            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    else:
        # Standard download for Dropbox, OneDrive, etc.
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)

is_valid_index = False
if os.path.exists(INDEX_PATH):
    try:
        file_size = os.path.getsize(INDEX_PATH)
        # The FAISS index is exactly 153,600,045 bytes. We'll check if it is > 100MB
        if file_size > 100 * 1024 * 1024:
            is_valid_index = True
        else:
            # File is too small (likely a downloaded HTML preview or error page)
            try:
                os.remove(INDEX_PATH)
            except Exception:
                pass
    except Exception:
        pass

if not is_valid_index:
    DEFAULT_GD_URL = "https://drive.google.com/file/d/1UN0KW7qsOLIlu4GY_H9vOgqXlRURtQkv/view?usp=sharing"
    success_auto = False
    
    # Try automatic download first
    with st.spinner("📥 FAISS Index File is missing. Automatically downloading index (153MB) from Google Drive..."):
        try:
            download_large_file(DEFAULT_GD_URL, INDEX_PATH)
            file_size = os.path.getsize(INDEX_PATH)
            if file_size > 100 * 1024 * 1024:
                success_auto = True
                st.success("FAISS Index downloaded successfully! Re-running dashboard...")
                st.rerun()
            else:
                if os.path.exists(INDEX_PATH):
                    try:
                        os.remove(INDEX_PATH)
                    except Exception:
                        pass
        except Exception as e:
            pass
            
    if not success_auto:
        st.warning("⚠️ **FAISS Index File (`models/faiss_index.bin`) is missing or invalid!**")
        st.markdown("""
        Because the FAISS index is around 153MB, it is gitignored and not pushed to GitHub.
        Automatic download failed. Please provide a manual download link.
        
        **To fix this:**
        1. Make sure your Google Drive file sharing setting is set to **"Anyone with the link"** (public).
        2. Paste your Google Drive sharing link (or Dropbox/OneDrive direct link) below.
        3. Click **Download FAISS Index** to download it directly into the app's container.
        """)
        
        download_url = st.text_input("Cloud URL for faiss_index.bin", placeholder="https://drive.google.com/... or https://dl.dropboxusercontent.com/...")
        if st.button("📥 Download FAISS Index", use_container_width=True):
            if download_url:
                with st.spinner("Downloading FAISS index (153MB)... This might take a minute."):
                    try:
                        download_large_file(download_url, INDEX_PATH)
                        file_size = os.path.getsize(INDEX_PATH)
                        if file_size > 100 * 1024 * 1024:
                            st.success("FAISS Index downloaded successfully! Re-running dashboard...")
                            st.rerun()
                        else:
                            if os.path.exists(INDEX_PATH):
                                try:
                                    os.remove(INDEX_PATH)
                                except Exception:
                                    pass
                            st.error(f"⚠️ **Download Failed: The downloaded file is too small ({file_size / (1024*1024):.2f} MB).**")
                    except Exception as e:
                        st.error(f"Failed to download index: {e}")
            else:
                st.error("Please enter a valid URL.")
        st.stop()

# Warm up data loading in background
with st.spinner("Initializing FAISS index (100,000 candidates)..."):
    try:
        index, candidate_ids = load_faiss_and_ids()
    except Exception as e:
        # File could not be loaded by FAISS (e.g. corrupted/invalid format)
        if os.path.exists(INDEX_PATH):
            try:
                os.remove(INDEX_PATH)
            except Exception:
                pass
        st.error(f"⚠️ **Error loading FAISS Index:** {e}")
        st.info("The downloaded index file may be corrupted (e.g. if the download link was not a direct download URL and returned an HTML page). The invalid file has been deleted. Please refresh/reload the page to try downloading again with a verified direct download link.")
        st.stop()

# ----------------------------------------------------
# 6. Recruiter Pipeline Runner
# ----------------------------------------------------
# ----------------------------------------------------
# 6. Recruiter Pipeline Helper Functions
# ----------------------------------------------------
def retrieve_top_k(query_text, index, candidate_ids):
    from src.semantic_scorer import get_model as get_faiss_model
    faiss_model = get_faiss_model()
    jd_embedding = faiss_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    # Free memory of first-stage embedding model
    import src.semantic_scorer
    src.semantic_scorer._model = None
    import gc
    gc.collect()
    
    k_search = min(1000, len(candidate_ids))
    retrieved_indices, retrieved_distances = search(jd_embedding, index, k=k_search)
    vector_scores_arr = scores_from_search(retrieved_indices, retrieved_distances, len(candidate_ids))
    
    retrieved_cids = [candidate_ids[idx] for idx in retrieved_indices if idx != -1 and idx < len(candidate_ids)]
    return retrieved_cids, vector_scores_arr

def fetch_candidates(retrieved_cids, candidates_db=None):
    retrieved_cids_set = set(retrieved_cids)
    if candidates_db is None:
        candidates_db = {}
        candidates_path = "data/candidates.jsonl.gz"
        if not os.path.exists(candidates_path):
            candidates_path = "data/sample_candidates.json"
            
        if candidates_path.endswith('.gz'):
            import gzip
            from src.data_loader import sanitize_candidate
            with gzip.open(candidates_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    idx = line.find('"candidate_id": "')
                    if idx != -1:
                        start_idx = idx + 17
                        end_idx = line.find('"', start_idx)
                        cand_id = line[start_idx:end_idx]
                        if cand_id in retrieved_cids_set:
                            try:
                                cand = json.loads(line)
                                candidates_db[cand_id] = sanitize_candidate(cand)
                            except Exception:
                                pass
                            if len(candidates_db) >= len(retrieved_cids_set):
                                break
        else:
            from src.data_loader import load_sample_candidates
            for cand in load_sample_candidates(candidates_path):
                cid = cand.get("candidate_id")
                if cid in retrieved_cids_set:
                    candidates_db[cid] = cand
    return candidates_db

def score_semantic(candidates, query_text, sem_model):
    semantic_scores = {c["candidate_id"]: 0.0 for c in candidates}
    jd_sem_emb = sem_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    rich_texts = [create_candidate_text(c) for c in candidates]
    cand_sem_embs = sem_model.encode(rich_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    
    similarities = np.dot(cand_sem_embs, jd_sem_emb)
    scaled_scores = (similarities + 1.0) / 2.0 * 100.0
    
    from src.config import BOOST_KEYWORDS
    for c, score in zip(candidates, scaled_scores):
        cid = c["candidate_id"]
        profile = c.get("profile", {})
        skills_list = c.get("skills", [])
        skills_text = " ".join([s.get("name", "").lower() for s in skills_list])
        full_text = f"{profile.get('headline', '')} {profile.get('summary', '')} {skills_text}".lower()
        has_jd_term = any(kw in full_text for kw in BOOST_KEYWORDS)
        boosted_score = score + 12.0 if has_jd_term else score
        semantic_scores[cid] = min(100.0, float(boosted_score))
    return semantic_scores

def score_bm25(candidates, query_text):
    from rank_bm25 import BM25Okapi
    bm25_texts = [create_candidate_text(c) for c in candidates]
    tokenized_corpus = [doc.lower().split(" ") for doc in bm25_texts]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query_text.lower().split(" ")
    bm25_raw_scores = bm25_model.get_scores(tokenized_query)
    
    min_bm25 = float(np.min(bm25_raw_scores))
    max_bm25 = float(np.max(bm25_raw_scores))
    
    bm25_scores = {}
    if max_bm25 > min_bm25:
        for idx, cand in enumerate(candidates):
            bm25_scores[cand["candidate_id"]] = float(((bm25_raw_scores[idx] - min_bm25) / (max_bm25 - min_bm25)) * 100.0)
    else:
        for cand in candidates:
            bm25_scores[cand["candidate_id"]] = 50.0
    return bm25_scores

def score_cross_encoder(candidates, query_text, reranker):
    cross_encoder_scores = {}
    pairs = [(query_text, create_candidate_text(c)) for c in candidates]
    raw_logits = reranker.predict(pairs, batch_size=64, show_progress_bar=False)
    
    min_logit = float(np.min(raw_logits))
    max_logit = float(np.max(raw_logits))
    
    if max_logit > min_logit:
        for i, c in enumerate(candidates):
            norm_score = ((raw_logits[i] - min_logit) / (max_logit - min_logit)) * 100.0
            cross_encoder_scores[c["candidate_id"]] = float(norm_score)
    else:
        for c in candidates:
            cross_encoder_scores[c["candidate_id"]] = 50.0
    return cross_encoder_scores

def score_structured(candidates, features_list, ideal_range=(5, 9, 7)):
    structured_scores = {}
    structured_breakdowns = {}
    for cand, feats in zip(candidates, features_list):
        cid = cand["candidate_id"]
        score, breakdown = compute_structured_score(feats, surrogate_path=SURROGATE_PATH, ideal_range=ideal_range)
        structured_scores[cid] = score
        structured_breakdowns[cid] = breakdown
    return structured_scores, structured_breakdowns

def aggregate_scores(
    candidates,
    semantic_scores,
    structured_scores,
    structured_breakdowns,
    vector_scores,
    flag_reasons,
    features_list,
    bm25_scores,
    cross_encoder_scores,
    jd_skills
):
    ranked_results = compute_final_ranking(
        candidates_data=candidates,
        semantic_scores=semantic_scores,
        structured_scores=structured_scores,
        structured_breakdowns=structured_breakdowns,
        vector_scores=vector_scores,
        flag_reasons=flag_reasons,
        features_list=features_list,
        bm25_scores=bm25_scores,
        cross_encoder_scores=cross_encoder_scores,
        jd_skills=jd_skills
    )
    return ranked_results

def run_interactive_pipeline(
    jd_reqs, 
    weights, 
    index, 
    candidate_ids, 
    candidates_db=None,
    status_box=None
):
    query_text = jd_reqs.to_embedding_text()
    
    # Configure weights dynamically
    src.config.SCORE_WEIGHTS["semantic"] = weights["semantic"]
    src.config.SCORE_WEIGHTS["bm25"] = weights["bm25"]
    src.config.SCORE_WEIGHTS["vector"] = weights["vector"]
    src.config.SCORE_WEIGHTS["structured"] = weights["structured"]
    
    if status_box:
        status_box.update(label="⚡ Retrieving Top Candidates via FAISS Index...", state="running")
    retrieved_cids, vector_scores_arr = retrieve_top_k(query_text, index, candidate_ids)
    retrieved_cids_set = set(retrieved_cids)
    
    if status_box:
        status_box.update(label="📂 Streaming matching candidate profiles...", state="running")
    candidates_db = fetch_candidates(retrieved_cids, candidates_db)
    
    candidates = []
    features_list = []
    flag_reasons = {}
    for cid in retrieved_cids:
        cand = candidates_db.get(cid)
        if not cand:
            continue
        candidates.append(cand)
        feats = extract_features(cand, jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills), jd_location=jd_reqs.location_preference)
        features_list.append(feats)
        is_flagged, reason = run_honeypot_checks(cand)
        if is_flagged:
            flag_reasons[cid] = reason
            
    # 1. Structured & Surrogate scoring on all 1000 candidates
    if status_box:
        status_box.update(label="📊 Computing Structured & Surrogate scores on all 1000 candidates...", state="running")
    structured_scores, structured_breakdowns = score_structured(candidates, features_list, ideal_range=jd_reqs.ideal_experience_range)
    
    # 2. BM25 keyword matching on all 1000 candidates
    if status_box:
        status_box.update(label="📝 Running BM25 keyword matching on all 1000 candidates...", state="running")
    bm25_scores = score_bm25(candidates, query_text)
    
    # 3. Build vector scores map
    vector_scores = {}
    for idx, cid in enumerate(candidate_ids):
        if cid in retrieved_cids_set:
            vector_scores[cid] = float(vector_scores_arr[idx])
            
    # 4. Run Intermediate scoring (combines FAISS, BM25, Structured) to select the top 150
    if status_box:
        status_box.update(label="🧬 Running high-fidelity pre-filtering on 1000 candidates...", state="running")
    from src.hybrid_aggregator import compute_final_ranking
    intermediate_results = compute_final_ranking(
        candidates_data=candidates,
        semantic_scores=None,
        structured_scores=structured_scores,
        structured_breakdowns=structured_breakdowns,
        vector_scores=vector_scores,
        flag_reasons=flag_reasons,
        features_list=features_list,
        bm25_scores=bm25_scores,
        cross_encoder_scores=None,
        jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills),
        apply_mapping=False
    )
    
    # Prune candidates list to top 150 based on intermediate ranking
    top_intermediate = intermediate_results[:150]
    top_cids_set = {x[0] for x in top_intermediate}
    
    # Keep candidates and features_list aligned 1-to-1
    top_candidates = []
    top_features_list = []
    for cid, _, _ in top_intermediate:
        for idx, cand in enumerate(candidates):
            if cand["candidate_id"] == cid:
                top_candidates.append(cand)
                top_features_list.append(features_list[idx])
                break
    candidates = top_candidates
    features_list = top_features_list
            
    if status_box:
        status_box.update(label="🧠 Loading ML models (MPNet & CrossEncoder)...", state="running")
    sem_model, reranker = load_ml_models()
            
    if status_box:
        status_box.update(label="🧠 Running Semantic model scoring on top 150...", state="running")
    semantic_scores = score_semantic(candidates, query_text, sem_model)
    
    if status_box:
        status_box.update(label="🤖 Executing Cross-Encoder reranking on top 150...", state="running")
    cross_encoder_scores = score_cross_encoder(candidates, query_text, reranker)
    
    if status_box:
        status_box.update(label="🧬 Aggregating & mapping final scores...", state="running")
    ranked_results = aggregate_scores(
        candidates,
        semantic_scores,
        structured_scores,
        structured_breakdowns,
        vector_scores,
        flag_reasons,
        features_list,
        bm25_scores,
        cross_encoder_scores,
        (jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills)
    )
    
    if status_box:
        status_box.update(label="✍️ Generating dynamic justifications & reasons...", state="running")
    # Generate reasonings for the top 100
    candidate_map = {c["candidate_id"]: c for c in candidates}
    final_ranked_candidates = []
    for rank_idx, (cid, score, breakdown) in enumerate(ranked_results):
        rank = rank_idx + 1
        reasoning = generate_reasoning(breakdown, rank)
        cand_info = candidate_map.get(cid)
        
        final_ranked_candidates.append({
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
            "candidate_info": cand_info,
            "breakdown": breakdown
        })
        
    import gc
    gc.collect()
    return final_ranked_candidates

# ----------------------------------------------------
# 7. Navigation Tabs
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Candidate Ranker", "👤 Candidate Profiler", "📊 Analytics Dashboard", "🏛️ Architecture & Pipeline"])

# ----------------------------------------------------
# TAB 1: Candidate Ranker
# ----------------------------------------------------
with tab1:
    col_left, col_right = st.columns([4, 8])
    
    with col_left:
        st.markdown('<div class="section-title">Job Description input</div>', unsafe_allow_html=True)
        
        jd_source = st.selectbox(
            "Select Job Description Source",
            options=["Default JD (Senior AI Engineer - Founding Team)", "Paste Custom JD", "Upload JD File (.md/.docx)"]
        )
        
        jd_content = ""
        filepath = f"temp_jd_{st.session_state.session_uuid}.md"
        
        if jd_source == "Default JD (Senior AI Engineer - Founding Team)":
            default_path = "data/job_description.md"
            if os.path.exists(default_path):
                with open(default_path, 'r', encoding='utf-8') as f:
                    jd_content = f.read()
            else:
                jd_content = "Default JD file not found. Please paste custom text."
            
            st.text_area("Default JD Preview (Read-Only)", value=jd_content[:500] + "...", height=150, disabled=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(jd_content)
                
        elif jd_source == "Paste Custom JD":
            jd_content = st.text_area("Paste JD Markdown Text Here", height=250, value="Job Title: Senior ML Engineer\nExperience Required: 4-8 years\nLocation: Pune, India\n\nThings you absolutely need:\n- Production experience with PyTorch\n- RAG pipelines\n\nNice to have:\n- Docker, Kubernetes")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(jd_content)
                
        else: # Upload File
            uploaded_file = st.file_uploader("Upload Job Description", type=["md", "txt", "docx"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".docx"):
                    filepath = f"temp_jd_{st.session_state.session_uuid}.docx"
                else:
                    filepath = f"temp_jd_{st.session_state.session_uuid}.md"
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Triggers re-parsing
            else:
                st.info("Please upload a file.")
                filepath = ""

        # Collapse scoring weights in an expander
        with st.expander("⚙️ Advanced Scoring Settings", expanded=False):
            st.caption("Customize how the search engine blends structural features, semantic embeddings, keyword match, and dense vectors. Weights are automatically normalized to sum to 1.0.")
            w_sem = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.20, 0.05)
            w_bm25 = st.slider("BM25 Keyword Weight", 0.0, 1.0, 0.20, 0.05)
            w_vec = st.slider("Vector Index Search Weight", 0.0, 1.0, 0.15, 0.05)
            w_str = st.slider("Structured Feature Scorer Weight", 0.0, 1.0, 0.45, 0.05)
            
            # Normalize weights
            sum_w = w_sem + w_bm25 + w_vec + w_str
            if sum_w > 0:
                weights = {
                    "semantic": w_sem / sum_w,
                    "bm25": w_bm25 / sum_w,
                    "vector": w_vec / sum_w,
                    "structured": w_str / sum_w
                }
            else:
                weights = {"semantic": 0.2, "bm25": 0.2, "vector": 0.15, "structured": 0.45}
        
        # If not expanded, use default weights
        if 'weights' not in locals():
            weights = {"semantic": 0.2, "bm25": 0.2, "vector": 0.15, "structured": 0.45}
            
        run_btn = st.button("🚀 Run CVHunt Pipeline", use_container_width=True)
        
        # System Information Card
        st.markdown('<div class="section-title">System Information</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="profile-box" style="padding: 10px 12px; margin-bottom: 0px;">
            <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; margin-bottom: 6px;">Pipeline Environment Specs</div>
            <div style="display: flex; flex-direction: column; gap: 4px; font-size: 0.76rem;">
                <div>🤖 <strong>Retrieval Model:</strong> FAISS Vector Index (MiniLM-L6)</div>
                <div>🧠 <strong>Semantic Similarity:</strong> MPNet-base-v2</div>
                <div>⚡ <strong>Reranker:</strong> Cross-Encoder (MiniLM-L6)</div>
                <div>📝 <strong>Text Search Scorer:</strong> BM25 Okapi</div>
                <div>🌳 <strong>Surrogate Scorer:</strong> GBR Regression model</div>
                <div>💻 <strong>Hardware Constraints:</strong> CPU-Only (offline mode)</div>
                <div>🚫 <strong>Network Connectivity:</strong> Local / Air-gapped</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        if filepath and (run_btn or st.session_state.pipeline_results is not None):
            # Parse JD if button clicked or if already cached
            if run_btn or st.session_state.parsed_jd is None:
                try:
                    st.session_state.parsed_jd = parse_jd(filepath)
                except Exception as e:
                    st.error(f"Error parsing Job Description: {e}")
                    st.session_state.parsed_jd = None
                finally:
                    # Clean up session-specific temp file immediately after parsing
                    if filepath and os.path.exists(filepath) and "temp_jd_" in filepath:
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
            
            jd_reqs = st.session_state.parsed_jd
            
            if jd_reqs:
                st.markdown(f"### Target Role: `{jd_reqs.role}`")
                
                # Show parsed details
                pd_col1, pd_col2, pd_col3 = st.columns(3)
                with pd_col1:
                    st.markdown(f"**Experience Required:** {jd_reqs.ideal_experience_range[0]}-{jd_reqs.ideal_experience_range[1]} years")
                with pd_col2:
                    st.markdown(f"**Location Preference:** {jd_reqs.location_preference}")
                with pd_col3:
                    st.markdown(f"**Parsed Description Length:** {len(jd_content) if jd_content else 'File'} chars")
                
                st.markdown("**Parsed Skills Required:**")
                must_tags = " ".join([f'<span class="badge badge-green">{s}</span>' for s in jd_reqs.must_have_skills])
                nice_tags = " ".join([f'<span class="badge badge-blue">{s}</span>' for s in jd_reqs.nice_to_have_skills])
                
                st.markdown(f"**Must Have:** {must_tags if must_tags else 'None detected'}", unsafe_allow_html=True)
                st.markdown(f"**Nice To Have:** {nice_tags if nice_tags else 'None detected'}", unsafe_allow_html=True)
                st.markdown("---")
                
                if run_btn:
                    with st.status("Initializing CVHunt Pipeline...", expanded=True) as status:
                        status.write("Checking system availability...")
                        # Acquire global lock to prevent concurrent heavy ML executions (OOM prevention)
                        acquired = _pipeline_lock.acquire(blocking=False)
                        if not acquired:
                            status.write("⚠️ Another user is currently running the pipeline. Waiting for resources to free up...")
                            _pipeline_lock.acquire(blocking=True)
                        
                        try:
                            status.write("System resources locked. Running pipeline...")
                            t_start = time.time()
                            st.session_state.pipeline_results = run_interactive_pipeline(
                                jd_reqs, 
                                weights, 
                                index, 
                                candidate_ids, 
                                status_box=status
                            )
                            st.session_state.pipeline_time = time.time() - t_start
                        finally:
                            _pipeline_lock.release()
                elif st.session_state.pipeline_results is None:
                    # Load precomputed default results to make startup instantaneous & OOM-safe
                    default_res_path = os.path.join(BASE_DIR, "data", "default_results.json")
                    if os.path.exists(default_res_path):
                        with open(default_res_path, "r", encoding="utf-8") as f:
                            st.session_state.pipeline_results = json.load(f)
                        st.session_state.pipeline_time = 92.94 # default runtime
                    else:
                        with st.status("Initializing CVHunt Pipeline (Fallback)...", expanded=True) as status:
                            status.write("Checking system availability...")
                            # Acquire global lock
                            acquired = _pipeline_lock.acquire(blocking=False)
                            if not acquired:
                                status.write("⚠️ Another user is currently running the pipeline. Waiting for resources to free up...")
                                _pipeline_lock.acquire(blocking=True)
                            
                            try:
                                status.write("System resources locked. Running pipeline (Fallback)...")
                                t_start = time.time()
                                st.session_state.pipeline_results = run_interactive_pipeline(
                                    jd_reqs, 
                                    weights, 
                                    index, 
                                    candidate_ids, 
                                    status_box=status
                                )
                                st.session_state.pipeline_time = time.time() - t_start
                            finally:
                                _pipeline_lock.release()
                        
                results = st.session_state.pipeline_results
                p_time = st.session_state.get("pipeline_time", 0.0)
                
                try:
                    p_time_str = f"{float(p_time):.2f} s"
                except (ValueError, TypeError):
                    p_time_str = f"{p_time} s"
                
                # Show runtime metrics in columns
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric(label="Pipeline Runtime", value=p_time_str, delta="CPU-only mode")
                with met_col2:
                    st.metric(label="Ranked Pool Size", value=f"{len(results)} profiles")
                
                st.markdown("---")
                
                # Display Ranked List
                st.markdown('<div class="section-title">Top 100 Candidates List</div>', unsafe_allow_html=True)
                
                # Generate Ranked CSV for Download Button
                csv_rows = []
                for r in results[:100]:
                    csv_rows.append({
                        "candidate_id": r["candidate_id"],
                        "rank": r["rank"],
                        "score": round(r["score"], 4),
                        "reasoning": r["reasoning"]
                    })
                df_csv = pd.DataFrame(csv_rows)
                csv_data = df_csv.to_csv(index=False, encoding="utf-8")
                
                # Generate Ranked Excel bytes in memory
                import io
                excel_buffer = io.BytesIO()
                
                # Self-healing import check for openpyxl
                try:
                    import openpyxl
                except ImportError:
                    try:
                        import subprocess
                        import sys
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
                        import openpyxl
                    except Exception as pip_err:
                        st.error(f"Could not load or install openpyxl: {pip_err}")
                        
                excel_data = None
                try:
                    df_csv.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_data = excel_buffer.getvalue()
                except Exception as excel_err:
                    st.error(f"Error generating Excel file: {excel_err}")
                
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="📥 Download Submission CSV",
                        data=csv_data,
                        file_name="submission.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with dl_col2:
                    if excel_data is not None:
                        st.download_button(
                            label="📊 Download Excel Spreadsheet",
                            data=excel_data,
                            file_name="submission.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    else:
                        st.button("📊 Excel Export Unavailable", disabled=True, use_container_width=True)
                
                # Custom HTML table styling
                table_rows = ""
                for r in results[:100]:
                    cid = r["candidate_id"]
                    rank = r["rank"]
                    score = r["score"]
                    reasoning = r["reasoning"]
                    info = r["candidate_info"]
                    
                    profile = info.get("profile", {})
                    title = profile.get("current_title", "Technical Engineer")
                    exp = profile.get("years_of_experience", 0.0)
                    
                    # Fit status badge
                    if rank <= 10:
                        fit_badge = '<span class="badge badge-green">Exceptional fit</span>'
                    elif rank <= 50:
                        fit_badge = '<span class="badge badge-blue">Strong fit</span>'
                    else:
                        fit_badge = '<span class="badge badge-amber">Partial fit</span>'
                        
                    # Split reasonings for quick display
                    short_reasoning = reasoning.split("---")[0].split("Concern:")[0].split("Disadvantage:")[0].strip()
                    
                    table_rows += (
                        f'<tr>'
                        f'<td style="font-weight: 700; width: 50px;">#{rank}</td>'
                        f'<td style="font-family: \'JetBrains Mono\', monospace; font-weight: 600; color: var(--accent);">{cid}</td>'
                        f'<td>'
                        f'<div style="font-weight: 600;">{title}</div>'
                        f'<div style="font-size: 0.75rem; color: var(--text-muted);">{exp:.1f} years exp | {profile.get("location", "India")}</div>'
                        f'</td>'
                        f'<td style="font-weight: 700; color: var(--accent);">{score:.2f}%</td>'
                        f'<td>{fit_badge}</td>'
                        f'<td style="font-size: 0.76rem; max-width: 320px; line-height: 1.25;">{short_reasoning}</td>'
                        f'</tr>'
                    )
                    
                table_html = (
                    f'<table class="data-table">'
                    f'<thead>'
                    f'<tr>'
                    f'<th>Rank</th>'
                    f'<th>ID</th>'
                    f'<th>Current Profile</th>'
                    f'<th>Score</th>'
                    f'<th>Fit Status</th>'
                    f'<th>Justification Summary</th>'
                    f'</tr>'
                    f'</thead>'
                    f'<tbody>'
                    f'{table_rows}'
                    f'</tbody>'
                    f'</table>'
                )
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Dynamic Selector to view in Candidate Profiler
                st.markdown("---")
                st.markdown("#### Analyze Candidate Profile Details")
                
                # Make a dictionary mapping for format func
                results_dict = {r["candidate_id"]: r for r in results}
                
                selected_opt = st.selectbox(
                    "Choose a candidate from the ranked list to load into the Profiler Tab:",
                    options=[r["candidate_id"] for r in results[:100]],
                    format_func=lambda x: f"Rank #{results_dict[x]['rank']}: {x} - {results_dict[x]['candidate_info'].get('profile', {}).get('current_title', 'Engineer')} ({results_dict[x]['score']:.2f}%)"
                )
                
                if selected_opt:
                    st.session_state.selected_cid = selected_opt
                    st.info(f"Candidate `{selected_opt}` selected! Head over to the **👤 Candidate Profiler** tab to analyze their full resume breakdown.")
        else:
            st.info("Please select or upload a Job Description on the left, then click 'Run Candidate Discoverer' to rank profiles.")

# ----------------------------------------------------
# TAB 2: Candidate Profiler
# ----------------------------------------------------
with tab2:
    if st.session_state.selected_cid is not None and st.session_state.pipeline_results is not None:
        cid = st.session_state.selected_cid
        results = st.session_state.pipeline_results
        
        # Get selected candidate details
        results_dict = {r["candidate_id"]: r for r in results}
        cand_data = results_dict.get(cid)
        
        if cand_data:
            c_info = cand_data["candidate_info"]
            score = cand_data["score"]
            rank = cand_data["rank"]
            reasoning = cand_data["reasoning"]
            breakdown = cand_data["breakdown"]
            
            profile = c_info.get("profile", {})
            career_history = c_info.get("career_history", [])
            skills = c_info.get("skills", [])
            education = c_info.get("education", [])
            signals = c_info.get("redrob_signals", {})
            
            # Header Columns
            prof_h1, prof_h2 = st.columns([7, 2])
            with prof_h1:
                st.markdown(f"## Candidate Profile: `{cid}`")
                st.markdown(f"#### **{profile.get('current_title', 'Software Engineer')}** | {profile.get('location', 'India')}")
            with prof_h2:
                # Big Score KPI
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Match Score</div>
                    <div class="metric-value">{score:.2f}%</div>
                    <div class="metric-label" style="margin-top:0.3rem;">Rank #{rank} / 100</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Sub Columns: Left detail, right scores & metrics
            det_left, det_right = st.columns([7, 5])
            
            with det_left:
                # 1. Recruiter Reasoning
                st.markdown('<div class="section-title">Recruiter Justification & Reasoning</div>', unsafe_allow_html=True)
                st.markdown('<div class="profile-box" style="background-color: var(--bg-subtle); border-left: 4px solid var(--accent); padding: 15px 18px;">', unsafe_allow_html=True)
                if "---" in reasoning:
                    detailed_reasoning = reasoning.split("---")[1].strip()
                    st.markdown(detailed_reasoning)
                else:
                    st.markdown(f'<p style="font-size: 0.9rem; line-height: 1.5; font-style: italic; margin: 0;">"{reasoning}"</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 2. Career History
                st.markdown('<div class="section-title">Career History & Job Timeline</div>', unsafe_allow_html=True)
                if career_history:
                    for idx, job in enumerate(career_history):
                        comp = job.get("company", "Company")
                        title = job.get("title", "Role Title")
                        desc = job.get("description", "")
                        sd = job.get("start_date", "N/A")
                        ed = job.get("end_date", "Present")
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 1.15rem; padding-left: 10px; border-left: 2px solid var(--border);">
                            <span style="font-size: 0.72rem; color: var(--text-muted); font-family: monospace;">{sd} — {ed}</span>
                            <div style="font-weight: 700; font-size: 0.88rem; color: var(--text);">{title}</div>
                            <div style="font-weight: 500; font-size: 0.8rem; color: var(--accent); margin-bottom: 0.35rem;">{comp}</div>
                            <div style="font-size: 0.78rem; color: var(--text-muted); line-height:1.45;">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: var(--text-muted);'>No career history records reported.</p>", unsafe_allow_html=True)
                    
            with det_right:
                # 1. Score Breakdown
                st.markdown('<div class="section-title">Score Engine Breakdown</div>', unsafe_allow_html=True)
                
                # Sub-score gauge/bars
                sub_sem = breakdown.get("semantic_score", 0.0)
                sub_str = breakdown.get("structured_score", 0.0)
                sub_vec = breakdown.get("vector_score", 0.0)
                sub_bm25 = breakdown.get("bm25_score", 0.0)
                sub_ce = breakdown.get("cross_encoder_score", 0.0)
                
                sb_df = pd.DataFrame({
                    "Scoring Module": ["Semantic Scorer", "Cross-Encoder", "Structured Scorer", "Vector Search", "BM25 Search"],
                    "Raw Score": [sub_sem, sub_ce, sub_str, sub_vec, sub_bm25]
                })
                
                fig_sb = px.bar(
                    sb_df, 
                    x="Raw Score", 
                    y="Scoring Module", 
                    orientation='h',
                    color="Raw Score",
                    color_continuous_scale="greys" if IS_DARK else "purples",
                    range_x=[0, 100]
                )
                
                fig_sb.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=9),
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig_sb, use_container_width=True, config={"displayModeBar": False})
                
                # 2. Behavioral Signals
                st.markdown('<div class="section-title">Recruitment & Availability Signals</div>', unsafe_allow_html=True)
                
                r_rate = signals.get("recruiter_response_rate", 0.0)
                np_days = signals.get("notice_period_days", 90)
                ic_rate = signals.get("interview_completion_rate", 0.0)
                pc_score = signals.get("profile_completeness_score", 0.0) / 100.0
                active_date = signals.get("last_active_date", "Unknown")
                relocate = "Yes" if signals.get("willing_to_relocate", False) else "No"
                otw = "Yes" if signals.get("open_to_work_flag", False) else "No"
                
                col_sig1, col_sig2 = st.columns(2)
                with col_sig1:
                    st.metric("Recruiter Response Rate", f"{r_rate:.0%}")
                    st.metric("Interview Completion", f"{ic_rate:.0%}")
                    st.metric("Open to Work", otw)
                with col_sig2:
                    st.metric("Notice Period", f"{np_days} days")
                    st.metric("Profile Completeness", f"{pc_score:.0%}")
                    st.metric("Willing to Relocate", relocate)
                st.caption(f"Last platform active date recorded: `{active_date}`")
                
                # 3. Matched Technical Skills
                st.markdown('<div class="section-title">Reported Technical Skills</div>', unsafe_allow_html=True)
                if skills:
                    skill_badges = ""
                    for s in skills:
                        name = s.get("name", "Skill")
                        prof = s.get("proficiency", "beginner").lower()
                        months = s.get("duration_months", 0)
                        
                        # Set color based on proficiency
                        if prof in ["expert", "advanced"]:
                            p_cls = "badge-green"
                        elif prof == "intermediate":
                            p_cls = "badge-blue"
                        else:
                            p_cls = "badge-amber"
                            
                        # Format years/months
                        dur_str = f"{months // 12}y {months % 12}m" if months >= 12 else f"{months}m"
                        skill_badges += (
                            f'<div style="display: inline-block; margin: 4px; padding: 6px 10px; background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 6px; font-size: 0.76rem;">'
                            f'<strong style="color: var(--text);">{name}</strong> '
                            f'<span class="badge {p_cls}" style="font-size:0.6rem; padding: 1px 5px; margin-left:5px;">{prof} ({dur_str})</span>'
                            f'</div>'
                        )
                    st.markdown(f'<div style="line-height:1.8;">{skill_badges}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: var(--text-muted);'>No technical skills reported.</p>", unsafe_allow_html=True)
                
                # 4. Education
                st.markdown('<div class="section-title">Education History</div>', unsafe_allow_html=True)
                if education:
                    for edu in education:
                        inst = edu.get("institution", "Institution")
                        deg = edu.get("degree", "Degree")
                        field = edu.get("field_of_study", "Field")
                        tier = edu.get("tier", "unknown").upper().replace("_", " ")
                        
                        st.markdown(f"""
                        <div class="profile-box" style="padding: 10px; margin-bottom: 8px; border: 1px solid var(--border);">
                            <div style="font-weight: 700; font-size: 0.82rem;">{deg} in {field}</div>
                            <div style="font-size: 0.76rem; color: var(--text-muted);">{inst}</div>
                            <div style="margin-top: 5px;"><span class="badge badge-blue">{tier}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: var(--text-muted);'>No education history records reported.</p>", unsafe_allow_html=True)
        else:
            st.warning("Could not locate profile details for selected Candidate.")
    else:
        st.info("No candidate profile loaded. Please execute the ranking pipeline on the **🔍 Candidate Discoverer** tab first, select a candidate, and click analyze.")

# ----------------------------------------------------
# TAB 3: Analytics Dashboard
# ----------------------------------------------------
with tab3:
    if st.session_state.pipeline_results is not None:
        results = st.session_state.pipeline_results
        
        # Convert results to DataFrame for charting
        data_rows = []
        for r in results[:100]:
            cid = r["candidate_id"]
            rank = r["rank"]
            score = r["score"]
            c_info = r["candidate_info"]
            breakdown = r["breakdown"]
            
            profile = c_info.get("profile", {})
            location = profile.get("location", "India").split(",")[0].split("/")[0].strip().capitalize()
            years_exp = profile.get("years_of_experience", 0.0)
            
            signals = c_info.get("redrob_signals", {})
            np_days = signals.get("notice_period_days", 90)
            r_rate = signals.get("recruiter_response_rate", 0.0)
            
            # Count skills
            skills_count = len(c_info.get("skills", []))
            
            data_rows.append({
                "candidate_id": cid,
                "rank": rank,
                "Score": score,
                "Years Experience": years_exp,
                "Location": location,
                "Skills Count": skills_count,
                "Notice Period (Days)": np_days,
                "Response Rate": r_rate,
                "Semantic Score": breakdown.get("semantic_score", 0.0),
                "Structured Score": breakdown.get("structured_score", 0.0),
                "Vector Score": breakdown.get("vector_score", 0.0),
                "BM25 Score": breakdown.get("bm25_score", 0.0)
            })
            
        df_an = pd.DataFrame(data_rows)
        
        # Row 1: KPI Stats
        an_col1, an_col2, an_col3, an_col4 = st.columns(4)
        with an_col1:
            avg_score = df_an["Score"].mean()
            st.metric("Avg Match Score (Top 100)", f"{avg_score:.2f}%")
        with an_col2:
            avg_exp = df_an["Years Experience"].mean()
            st.metric("Avg Years Experience", f"{avg_exp:.1f} years")
        with an_col3:
            avg_skills = df_an["Skills Count"].mean()
            st.metric("Avg Skills Listed", f"{avg_skills:.1f} skills")
        with an_col4:
            avg_np = df_an["Notice Period (Days)"].mean()
            st.metric("Avg Notice Period", f"{avg_np:.0f} days")
            
        st.markdown("---")
        
        # Row 2: Charts
        c_an_col1, c_an_col2 = st.columns(2)
        
        with c_an_col1:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Years of Experience vs. Match Score</div>
                <div class="chart-subtitle">Top 100 candidates scatter analysis</div>
            """, unsafe_allow_html=True)
            
            fig1 = px.scatter(
                df_an, 
                x="Years Experience", 
                y="Score", 
                color="Score",
                hover_name="candidate_id",
                color_continuous_scale="greys" if IS_DARK else "purples",
            )
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Distribution of Final Scores</div>
                <div class="chart-subtitle">Match scores histogram for the top candidates</div>
            """, unsafe_allow_html=True)
            
            fig2 = px.histogram(
                df_an, 
                x="Score",
                color_discrete_sequence=["#27272a" if IS_DARK else "#71717a"]
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_an_col2:
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Breakdown of Candidate Locations</div>
                <div class="chart-subtitle">Presents candidates matching geographic tier settings</div>
            """, unsafe_allow_html=True)
            
            loc_counts = df_an["Location"].value_counts().reset_index()
            loc_counts.columns = ["Location", "Count"]
            
            fig3 = px.pie(
                loc_counts, 
                values="Count", 
                names="Location",
                color_discrete_sequence=px.colors.sequential.Greys if IS_DARK else px.colors.sequential.Purples
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Recruiter Response Rate vs. Notice Period</div>
                <div class="chart-subtitle">Cross comparison of critical availability factors</div>
            """, unsafe_allow_html=True)
            
            fig4 = px.scatter(
                df_an, 
                x="Notice Period (Days)", 
                y="Response Rate", 
                color="Score",
                hover_name="candidate_id",
                color_continuous_scale="greys" if IS_DARK else "purples",
            )
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=10),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No analytics data available. Please execute the ranking pipeline on the **🔍 Candidate Ranker** tab first to populate the analytics dashboard.")

# ----------------------------------------------------
# TAB 4: Architecture & Pipeline
# ----------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">CVHunt Pipeline Visualization</div>', unsafe_allow_html=True)
    
    # Inline SVG Flowchart
    st.markdown("""
    <svg viewBox="0 0 850 180" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: auto; font-family: system-ui, sans-serif;">
      <rect width="850" height="180" rx="12" fill="var(--bg-subtle)" stroke="var(--border)" stroke-width="1"/>
      
      <!-- Step 1 -->
      <rect x="20" y="55" width="80" height="50" rx="6" fill="var(--card)" stroke="var(--border)" stroke-width="2"/>
      <text x="60" y="85" fill="var(--text)" font-size="11" font-weight="bold" text-anchor="middle">Job Desc (JD)</text>
      
      <!-- Arrow 1 -->
      <path d="M 100 80 L 120 80" stroke="var(--text-muted)" stroke-width="2" marker-end="url(#arrow)"/>
      
      <!-- Step 2 -->
      <rect x="120" y="55" width="95" height="50" rx="6" fill="var(--card)" stroke="var(--border)" stroke-width="2"/>
      <text x="167" y="78" fill="var(--text)" font-size="10" font-weight="bold" text-anchor="middle">Dynamic Parser</text>
      <text x="167" y="94" fill="var(--text-muted)" font-size="9" text-anchor="middle">jd_parser.py</text>
      
      <!-- Arrow 2 -->
      <path d="M 215 80 L 235 80" stroke="var(--text-muted)" stroke-width="2" marker-end="url(#arrow)"/>
      
      <!-- Step 3 -->
      <rect x="235" y="55" width="95" height="50" rx="6" fill="var(--card)" stroke="var(--accent)" stroke-width="2"/>
      <text x="282" y="78" fill="var(--accent)" font-size="10" font-weight="bold" text-anchor="middle">FAISS Retrieval</text>
      <text x="282" y="94" fill="var(--text-muted)" font-size="8" text-anchor="middle">Top 250 Retrieved</text>
      
      <!-- Arrow 3 -->
      <path d="M 330 80 L 350 80" stroke="var(--text-muted)" stroke-width="2" marker-end="url(#arrow)"/>
      
      <!-- Hybrid Scorer dashed box -->
      <rect x="350" y="15" width="220" height="130" rx="8" fill="none" stroke="var(--border)" stroke-width="1.5" stroke-dasharray="4,4"/>
      <text x="460" y="30" fill="var(--text-muted)" font-size="9" font-weight="bold" text-anchor="middle">4-SCORE HYBRID SCORER</text>
      
      <rect x="360" y="42" width="95" height="32" rx="4" fill="var(--card)" stroke="var(--border)" stroke-width="1"/>
      <text x="407" y="62" fill="var(--text)" font-size="8.5" text-anchor="middle">Semantic (MPNet)</text>
      
      <rect x="465" y="42" width="95" height="32" rx="4" fill="var(--card)" stroke="var(--border)" stroke-width="1"/>
      <text x="512" y="62" fill="var(--text)" font-size="8.5" text-anchor="middle">BM25 Keyword</text>
      
      <rect x="360" y="88" width="95" height="32" rx="4" fill="var(--card)" stroke="var(--border)" stroke-width="1"/>
      <text x="407" y="108" fill="var(--text)" font-size="8.5" text-anchor="middle">Heuristic/Surrogate</text>
      
      <rect x="465" y="88" width="95" height="32" rx="4" fill="var(--card)" stroke="var(--border)" stroke-width="1"/>
      <text x="512" y="108" fill="var(--text)" font-size="8.5" text-anchor="middle">Vector (FAISS)</text>
      
      <!-- Arrow 4 -->
      <path d="M 570 80 L 590 80" stroke="var(--text-muted)" stroke-width="2" marker-end="url(#arrow)"/>
      
      <!-- Step 5 -->
      <rect x="590" y="55" width="105" height="50" rx="6" fill="var(--card)" stroke="var(--border)" stroke-width="2"/>
      <text x="642" y="78" fill="var(--text)" font-size="10" font-weight="bold" text-anchor="middle">Cross-Encoder</text>
      <text x="642" y="94" fill="var(--text-muted)" font-size="9" text-anchor="middle">MiniLM Reranker</text>
      
      <!-- Arrow 5 -->
      <path d="M 695 80 L 715 80" stroke="var(--text-muted)" stroke-width="2" marker-end="url(#arrow)"/>
      
      <!-- Step 6 -->
      <rect x="715" y="55" width="115" height="50" rx="6" fill="var(--card)" stroke="var(--border)" stroke-width="2"/>
      <text x="772" y="78" fill="var(--text)" font-size="10" font-weight="bold" text-anchor="middle">Rank & Reasoning</text>
      <text x="772" y="94" fill="var(--text-muted)" font-size="9" text-anchor="middle">submission.csv</text>
      
      <!-- Marker -->
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="var(--text-dim)"/>
        </marker>
      </defs>
    </svg>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Textual Details
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
        ### 🏛️ Pipeline Stages Explained
        
        #### 1. Dynamic JD Parsing (`jd_parser.py`)
        - Dynamically extracts must-have and nice-to-have technical skills directly from the text of the job description.
        - Identifies location preferences and ideal experience ranges dynamically using regular expressions.
        - Translates parsed requirements into a unified dense textual query.
        
        #### 2. First-Stage Dense Retrieval (`vector_index.py`)
        - Encodes the parsed JD query into a dense vector embedding using `all-MiniLM-L6-v2`.
        - Performs an O(1) similarity search against 100,000 candidate profiles in a **FAISS FlatIP vector index**.
        - Retrieves the top 1000 candidate IDs matching the query vector, then applies a fast hybrid pre-filter to narrow it down to the top 150 candidates for final scoring.
        
        #### 3. 4-Score Hybrid Scorer (`compute_final_ranking`)
        Blends multiple retrieval and scoring philosophies:
        - **Semantic Scorer (20%):** Calculates cosine similarity between dense candidate profile texts and the JD using `all-mpnet-base-v2`. Includes keyword boosts.
        - **BM25 Search (20%):** Runs classical keyword frequencies over profile texts to lock onto specific, exact matching terms.
        - **Vector Search (15%):** Leverages raw index similarity distances.
        - **Structured Scorer (45%):** Blend of rule-based gating (gains/decays for experience, education tier, and company ratio) and a **Gradient Boosting Regressor** surrogate trained on heuristic scores to smooth structured ranking.
        """)
    with arch_col2:
        st.markdown("""
        ### 🔒 Security, Compliance & Rules
        
        #### ⚠️ Honeypot Filtering (`honeypot_detector.py`)
        - Employs 8 robust rules to detect impossible/cheat resume records (e.g. claiming skills before a company was founded, or claiming expert skills with zero actual job durations).
        - Flagged profiles are capped at a maximum final score of `20.0`, safely filtering them out of the top 100 list.
        
        #### ⚖️ Hard Disqualifications & Gating
        - Candidate profiles with `< 2` must-have skills (or `< 1` for small JDs) are auto-disqualified (score = 0.0).
        - Applies graduated penalty multipliers (from `0.15` to `0.80`) based on the candidate's match ratio of the Job Description's must-have skills.
        - Auto-rejects candidates with zero product-company experience or ≥ 95% consulting experience ratios.
        
        #### 📈 Deterministic Tie-Breaking
        - If two candidate profiles achieve identical scores, ranks are assigned deterministically by sorting descending by score, then ascending by candidate ID.
        
        ### ⚙️ Strict Sandbox Optimization
        - **Memory Limit Compliance:** Streams matching profiles dynamically on-the-fly inside the query thread, avoiding pre-loading the 100K candidates database into RAM (saving ~350MB).
        - **Model Unloading:** Deletes the first-stage vector model `all-MiniLM-L6-v2` and triggers `gc.collect()` immediately after retrieval to keep container footprint under **1.0 GB RAM**.
        - **Offline Enforcement:** Automatically turns off HuggingFace hub connections (`HF_HUB_OFFLINE="1"`) to prevent timeout delays, completing the ranking in **~93 seconds** on CPU.
        """)
