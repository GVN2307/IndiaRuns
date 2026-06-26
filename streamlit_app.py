import os
import sys
import json
import gzip
import time
import re
import numpy as np
import pandas as pd
import streamlit as st
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
    page_title="CVHunt Discoverer",
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

css = f"""
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

/* Hide Streamlit components */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1360px !important;
}}

[data-testid="stHorizontalBlock"] {{ gap: 1.25rem !important; }}
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stHorizontalBlock"]) {{
    margin-bottom: 0.5rem !important;
}}

/* Card Elements */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}}
.metric-label {{
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 0.1rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}}
.metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}}
.metric-delta {{
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 0.3rem;
    padding: 2px 7px;
    border-radius: 5px;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}}
.delta-up {{ color: var(--green); background: var(--green-muted); }}
.delta-down {{ color: var(--red); background: var(--red-muted); }}
.delta-warn {{ color: var(--amber); background: var(--amber-muted); }}

.chart-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.chart-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
}}
.chart-subtitle {{
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}}

/* Navigation Tabs (Pill-style) */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.835rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px;
    margin-bottom: 1rem;
}}

/* Custom Status Badges */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}

/* Custom Tables */
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.65rem 0.8rem;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
    background: var(--bg-subtle);
}}
.data-table td {{
    padding: 0.7rem 0.8rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
    vertical-align: middle;
}}
.data-table tr:hover td {{
    background: var(--card-hover);
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}

/* Brand / Header */
.brand {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}}
.brand-logo {{
    font-size: 1.5rem;
    color: var(--accent);
}}
.brand-name {{
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
}}
.brand-subtitle {{
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-left: 0.5rem;
}}

/* Custom container */
.profile-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}

.section-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ----------------------------------------------------
# 4. Header Bar
# ----------------------------------------------------
head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-logo">◆</span>
        <span class="brand-name">CVHunt Discoverer</span>
        <span class="brand-subtitle">AI-Powered Intelligent Discovery & Reranking</span>
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
def load_candidates_db():
    candidates_path = "data/candidates.jsonl.gz"
    if not os.path.exists(candidates_path):
        # Fallback if gz not present
        candidates_path = "data/sample_candidates.json"
        if not os.path.exists(candidates_path):
            st.error(f"Candidate database file not found at {candidates_path}!")
            return {}
            
    candidates = {}
    if candidates_path.endswith('.gz'):
        with gzip.open(candidates_path, 'rt', encoding='utf-8') as f:
            for line in f:
                cand = json.loads(line)
                candidates[cand["candidate_id"]] = cand
    else:
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                cand = json.loads(line)
                candidates[cand["candidate_id"]] = cand
    return candidates

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

# Warm up data loading in background
with st.spinner("Initializing models & database (100,000 candidates)... This may take 10-15s on first load."):
    index, candidate_ids = load_faiss_and_ids()
    candidates_db = load_candidates_db()
    sem_model, reranker = load_ml_models()

# ----------------------------------------------------
# 6. Recruiter Pipeline Runner
# ----------------------------------------------------
def run_interactive_pipeline(
    jd_reqs, 
    weights, 
    candidates_db, 
    index, 
    candidate_ids, 
    sem_model, 
    reranker
):
    query_text = jd_reqs.to_embedding_text()
    
    # 1. First-Stage FAISS Scorer
    from src.semantic_scorer import get_model as get_faiss_model
    faiss_model = get_faiss_model()
    jd_embedding = faiss_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    k_search = min(250, len(candidate_ids))
    retrieved_indices, retrieved_distances = search(jd_embedding, index, k=k_search)
    vector_scores_arr = scores_from_search(retrieved_indices, retrieved_distances, len(candidate_ids))
    
    retrieved_cids = [candidate_ids[idx] for idx in retrieved_indices if idx != -1 and idx < len(candidate_ids)]
    retrieved_cids_set = set(retrieved_cids)
    
    # 2. Candidate Filtering & Features Extraction
    candidates = []
    features_list = []
    flag_reasons = {}
    
    for cid in retrieved_cids:
        cand = candidates_db.get(cid)
        if not cand:
            continue
            
        candidates.append(cand)
        feats = extract_features(cand, jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills))
        features_list.append(feats)
        
        is_flagged, reason = run_honeypot_checks(cand)
        if is_flagged:
            flag_reasons[cid] = reason
            
    # 3. Score 1: Semantic Scorer (mpnet-base)
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
        
    # 4. Score 4: BM25 Scorer
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
            
    # 5. Score 5: CrossEncoder Reranker
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
            
    # 6. Score 2: Structured Scorer
    structured_scores = {}
    structured_breakdowns = {}
    for cand, feats in zip(candidates, features_list):
        cid = cand["candidate_id"]
        score, breakdown = compute_structured_score(feats, surrogate_path=SURROGATE_PATH)
        structured_scores[cid] = score
        structured_breakdowns[cid] = breakdown
        
    # 7. Vector Scores (FAISS)
    vector_scores = {}
    for idx, cid in enumerate(candidate_ids):
        if cid in retrieved_cids_set:
            vector_scores[cid] = float(vector_scores_arr[idx])
            
    # 8. Configure weights dynamically
    src.config.SCORE_WEIGHTS["semantic"] = weights["semantic"]
    src.config.SCORE_WEIGHTS["bm25"] = weights["bm25"]
    src.config.SCORE_WEIGHTS["vector"] = weights["vector"]
    src.config.SCORE_WEIGHTS["structured"] = weights["structured"]
    
    # 9. Aggregate
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
        jd_skills=(jd_reqs.must_have_skills, jd_reqs.nice_to_have_skills)
    )
    
    # Generate reasonings for the top 100
    final_ranked_candidates = []
    for rank_idx, (cid, score, breakdown) in enumerate(ranked_results):
        rank = rank_idx + 1
        reasoning = generate_reasoning(breakdown, rank)
        
        cand_info = next((c for c in candidates if c["candidate_id"] == cid), None)
        
        final_ranked_candidates.append({
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
            "candidate_info": cand_info,
            "breakdown": breakdown
        })
        
    return final_ranked_candidates

# ----------------------------------------------------
# 7. Navigation Tabs
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Candidate Discoverer", "👤 Candidate Profiler", "📊 Analytics Dashboard"])

# ----------------------------------------------------
# TAB 1: Candidate Discoverer
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
        filepath = "temp_jd.md"
        
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
                    filepath = "temp_jd.docx"
                else:
                    filepath = "temp_jd.md"
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Triggers re-parsing
            else:
                st.info("Please upload a file.")
                filepath = ""

        st.markdown('<div class="section-title">Aggregation Weights</div>', unsafe_allow_html=True)
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
            
        run_btn = st.button("🚀 Run Candidate Discoverer", use_container_width=True)
        
    with col_right:
        if filepath and (run_btn or st.session_state.pipeline_results is not None):
            # Parse JD if button clicked or if already cached
            if run_btn or st.session_state.parsed_jd is None:
                try:
                    st.session_state.parsed_jd = parse_jd(filepath)
                except Exception as e:
                    st.error(f"Error parsing Job Description: {e}")
                    st.session_state.parsed_jd = None
            
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
                
                # Run pipeline
                if run_btn or st.session_state.pipeline_results is None:
                    with st.spinner("Executing FAISS dense search, scoring candidates, and applying behavioral modifiers..."):
                        t_start = time.time()
                        st.session_state.pipeline_results = run_interactive_pipeline(
                            jd_reqs, 
                            weights, 
                            candidates_db, 
                            index, 
                            candidate_ids, 
                            sem_model, 
                            reranker
                        )
                        st.session_state.pipeline_time = time.time() - t_start
                        
                results = st.session_state.pipeline_results
                p_time = st.session_state.get("pipeline_time", 0.0)
                
                st.success(f"Pipeline executed successfully in **{p_time:.2f} seconds**! Discovered and ranked top {len(results)} matching profiles.")
                
                # Display Ranked List
                st.markdown('<div class="section-title">Top 100 Candidates List</div>', unsafe_allow_html=True)
                
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
                    short_reasoning = reasoning.split("Concern:")[0].split("Disadvantage:")[0]
                    
                    table_rows += f"""
                    <tr>
                        <td style="font-weight: 700; width: 50px;">#{rank}</td>
                        <td style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--accent);">{cid}</td>
                        <td>
                            <div style="font-weight: 600;">{title}</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">{exp:.1f} years exp | {profile.get('location', 'India')}</div>
                        </td>
                        <td style="font-weight: 700; color: var(--accent);">{score:.2f}%</td>
                        <td>{fit_badge}</td>
                        <td style="font-size: 0.76rem; max-width: 320px; line-height: 1.25;">{short_reasoning}</td>
                    </tr>
                    """
                    
                st.markdown(f"""
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>ID</th>
                            <th>Current Profile</th>
                            <th>Score</th>
                            <th>Fit Status</th>
                            <th>Justification Summary</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
                
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
        cand_data = next((r for r in results if r["candidate_id"] == cid), None)
        
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
                st.markdown(f"""
                <div class="profile-box" style="background-color: var(--bg-subtle); border-left: 4px solid var(--accent);">
                    <p style="font-size: 0.9rem; line-height: 1.5; font-style: italic; margin: 0;">"{reasoning}"</p>
                </div>
                """, unsafe_allow_html=True)
                
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
                        
                        skill_badges += f"""
                        <div style="display: inline-block; margin: 4px; padding: 6px 10px; background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 6px; font-size: 0.76rem;">
                            <strong style="color: var(--text);">{name}</strong> 
                            <span class="badge {p_cls}" style="font-size:0.6rem; padding: 1px 5px; margin-left:5px;">{prof} ({dur_str})</span>
                        </div>
                        """
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
        st.info("No analytics data available. Please execute the ranking pipeline on the **🔍 Candidate Discoverer** tab first to populate the analytics dashboard.")
