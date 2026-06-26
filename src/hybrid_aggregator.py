import re
import logging
from typing import Dict, Any, List, Tuple, Set
from src.config import (
    SCORE_WEIGHTS, BEHAVIORAL_MULTIPLIERS, 
    SHIPPER_WORDS, RETRIEVAL_KEYWORDS, EVALUATION_KEYWORDS
)

logger = logging.getLogger("CVHunt.Aggregator")

def aggregate_scores(semantic_score: float, structured_score: float, vector_score: float, bm25_score: float) -> float:
    """
    Computes the weighted average of the 4 score engines.
    Weights: semantic=0.20, bm25=0.20, vector=0.15, structured=0.45
    """
    w_sem = SCORE_WEIGHTS.get("semantic", 0.20)
    w_bm25 = SCORE_WEIGHTS.get("bm25", 0.20)
    w_vec = SCORE_WEIGHTS.get("vector", 0.15)
    w_str = SCORE_WEIGHTS.get("structured", 0.45)
    
    return (semantic_score * w_sem) + (bm25_score * w_bm25) + (vector_score * w_vec) + (structured_score * w_str)

def apply_behavioral_modifiers(base_score: float, beh_feat: Dict[str, Any]) -> float:
    """
    Applies behavioral modifiers as multiplicative penalties based on features.
    Uses graduated penalties for notice period, recruiter response rate, and active status.
    Also applies location, relocation, github, and skill assessment bonuses.
    """
    modified_score = base_score
    
    # 1. Inactive graduated penalty (if not open_to_work)
    days_since_active = beh_feat.get("active_status", 365)
    open_to_work = beh_feat.get("open_to_work", False)
    if not open_to_work:
        if days_since_active <= 30:
            active_mult = 1.0
        elif days_since_active <= 90:
            active_mult = 0.9
        elif days_since_active <= 180:
            active_mult = 0.7
        else:
            active_mult = 0.4
        modified_score *= active_mult
        
    # 2. Recruiter response rate graduated penalty
    response_rate = beh_feat.get("response_rate", 0.0)
    if response_rate >= 0.8:
        response_mult = 1.0
    elif response_rate >= 0.5:
        response_mult = 0.9
    elif response_rate >= 0.2:
        response_mult = 0.8
    else:
        response_mult = 0.4
    modified_score *= response_mult
        
    # 3. Notice period graduated penalty
    notice_period = beh_feat.get("notice_period", 90)
    if notice_period <= 30:
        notice_mult = 1.0
    elif notice_period <= 60:
        notice_mult = 0.9
    elif notice_period <= 90:
        notice_mult = 0.8
    else:
        notice_mult = 0.6
    modified_score *= notice_mult
        
    # 4. Low completion penalty (interview_completion_rate < 0.3)
    completion_rate = beh_feat.get("completion_rate", 0.0)
    if completion_rate < 0.3:
        modified_score *= BEHAVIORAL_MULTIPLIERS.get("low_completion_penalty", 0.60)
        
    # 5. Spray-and-pray penalty (applications_submitted_30d > 10)
    if beh_feat.get("spammy_applications", False):
        modified_score *= 0.95
        
    # 6. Profile completeness gate (profile_completeness < 0.5)
    profile_completeness = beh_feat.get("profile_completeness", 0.0)
    if profile_completeness < 0.5:
        modified_score *= 0.50
        
    # 7. Pune/Noida location bonus (+15% / 1.15x)
    if beh_feat.get("is_pune_noida", False):
        modified_score *= 1.15
    # 8. Relocation willingness bonus (+10% / 1.10x)
    elif beh_feat.get("willing_to_relocate", False):
        modified_score *= 1.10
        
    # 9. GitHub activity bonus (+5% / 1.05x)
    github_score = beh_feat.get("github_activity_score", -1.0)
    if github_score > 50:
        modified_score *= 1.05
        
    # 10. Relevant skill assessments bonus (+5% / 1.05x)
    if beh_feat.get("has_high_assessment", False):
        modified_score *= 1.05
        
    return modified_score

def apply_honeypot_penalty(score: float, is_flagged: bool) -> float:
    """
    Eliminates candidate (sets score to 0.0) if flagged as a honeypot.
    """
    if is_flagged:
        return 0.0
    return score

def check_disqualifications(cand: Dict[str, Any], features: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Checks if a candidate is disqualified based on hard requirements.
    Returns (is_disqualified, reason).
    """
    profile = cand.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    skills = cand.get("skills", [])
    skills_text = " ".join([s.get("name", "").lower() for s in skills])
    career = cand.get("career_history", [])
    career_titles = " ".join([job.get("title", "").lower() for job in career])
    career_descs = " ".join([job.get("description", "").lower() for job in career])
    
    full_text = f"{current_title} {headline} {summary} {skills_text} {career_titles} {career_descs}"
    
    # 1. CV Engineer (matches title/headline) with <2 non-CV AI/ML jobs -> REJECT
    cv_pattern = re.compile(r"\b(computer vision|cv|vision engineer|image processing|opencv|object detection|cnn|yolo|segmentation|resnet|image classification|ocr)\b", re.IGNORECASE)
    is_cv_engineer = bool(cv_pattern.search(current_title) or cv_pattern.search(headline))
    if is_cv_engineer:
        ai_ml_title_pattern = re.compile(r"\b(ai|ml|machine learning|nlp|natural language|retrieval|search|ranking|rag|embeddings|llm|deep learning|applied scientist|data scientist)\b", re.IGNORECASE)
        non_cv_jobs_count = 0
        for job in career:
            job_title = job.get("title", "")
            job_desc = job.get("description", "")
            if ai_ml_title_pattern.search(job_title) and not cv_pattern.search(job_title) and not cv_pattern.search(job_desc):
                non_cv_jobs_count += 1
        if non_cv_jobs_count < 2:
            return True, f"CV Engineer with only {non_cv_jobs_count} non-CV AI/ML jobs in career history (requires >=2)"
            
    # 2. Pure research (matches title/headline) with <30% product exp -> REJECT
    research_pattern = re.compile(r"\b(research scientist|researcher|postdoc|applied scientist|fellow|academic|professor|phd scholar|research engineer)\b", re.IGNORECASE)
    is_research = bool(research_pattern.search(current_title) or research_pattern.search(headline))
    exp_features = features.get("experience_features", {})
    product_company_ratio = exp_features.get("product_company_ratio", 0.0)
    if is_research and product_company_ratio < 0.3:
        return True, f"Research candidate with only {product_company_ratio:.0%} product experience (requires >=30%)"
        
    # 3. Non-tech titles with <3 tech jobs -> REJECT
    career_feat = features.get("career_quality", {})
    is_non_tech = career_feat.get("is_non_tech", False)
    title_tech_count = 0
    from src.features import NON_TECH_TITLE_PATTERN
    tech_title_patterns = re.compile(
        r"(software engineer|backend engineer|frontend engineer|data engineer|ml engineer|ai engineer|"
        r"machine learning engineer|data scientist|research scientist|nlp engineer|search engineer|"
        r"computer vision engineer|deep learning engineer|applied scientist|research engineer|"
        r"software developer|backend developer|systems developer|python developer|ml developer|ai developer|"
        r"full stack developer|full-stack developer|data developer|software programmer|python programmer|"
        r"architect|tech lead|engineering lead|founding engineer|founder|"
        r"staff engineer|principal engineer|distinguished engineer|engineering manager|"
        r"ai researcher|platform engineer|infrastructure engineer|site reliability engineer|sre)",
        re.IGNORECASE
    )
    for job in career:
        job_title = job.get("title", "").strip()
        if tech_title_patterns.search(job_title) and not NON_TECH_TITLE_PATTERN.search(job_title):
            title_tech_count += 1
            
    if is_non_tech and title_tech_count < 3:
        return True, f"Non-tech current role '{current_title}' with only {title_tech_count} tech jobs in career history (requires >=3)"
        
    # 4. Consulting-only with >=95% ratio -> REJECT
    consulting_company_ratio = exp_features.get("consulting_company_ratio", 0.0)
    if consulting_company_ratio >= 0.95:
        return True, f"Consulting-only career history ({consulting_company_ratio:.0%} ratio)"
        
    return False, ""

def compute_final_ranking(
    candidates_data: List[Dict[str, Any]], 
    semantic_scores: Dict[str, float], 
    structured_scores: Dict[str, float], 
    structured_breakdowns: Dict[str, Dict[str, float]], 
    vector_scores: Dict[str, float], 
    flag_reasons: Dict[str, str],
    features_list: List[Dict[str, Any]] = None,
    bm25_scores: Dict[str, float] = None,
    cross_encoder_scores: Dict[str, float] = None
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Runs the full aggregation pipeline for all candidates.
    Returns a sorted list of (candidate_id, final_score, candidate_and_breakdown_dict).
    Uses deterministic tie-breaking.
    """
    ranked_candidates = []
    
    for idx, cand in enumerate(candidates_data):
        cid = cand.get("candidate_id", "")
        
        sem_s = semantic_scores.get(cid, 0.0)
        str_s = structured_scores.get(cid, 0.0)
        vec_s = vector_scores.get(cid, 0.0)
        bm25_s = bm25_scores.get(cid, 0.0) if bm25_scores is not None else 0.0
        
        # 1. Aggregate
        agg_score = aggregate_scores(sem_s, str_s, vec_s, bm25_s)
        
        # Reuse pre-extracted features if available
        if features_list is not None and idx < len(features_list):
            features = features_list[idx]
        else:
            from src.features import extract_features
            features = extract_features(cand)
            
        beh_feat = features.get("behavioral_features", {})
        
        # 2. Apply modifiers
        modified_score = apply_behavioral_modifiers(agg_score, beh_feat)
        
        # 3. Apply honeypot penalty (score becomes 0.0 if flagged)
        is_flagged = cid in flag_reasons
        if is_flagged:
            logger.warning(f"Candidate {cid} flagged as honeypot: {flag_reasons.get(cid, '')}")
        final_score = apply_honeypot_penalty(modified_score, is_flagged)
        
        # Hard cap for non-tech titles regardless of other signals
        career_feat = features.get("career_quality", {})
        if career_feat.get("is_non_tech", False):
            final_score = min(5.0, final_score)
            
        # 4. Enforce must-have skill gating on final score (graduated multipliers)
        must_have_count = features.get("skill_features", {}).get("must_have_count", 0)
        if must_have_count < 2:
            final_score = 0.0  # reject
        elif must_have_count < 3:
            final_score *= 0.15
        elif must_have_count < 6:
            final_score *= 0.3
        elif must_have_count < 9:
            final_score *= 0.6
        elif must_have_count < 12:
            final_score *= 0.8
            
        # Count shipper words in candidate's summary and job descriptions (Phase 3)
        profile = cand.get("profile", {})
        summary = profile.get("summary", "").lower()
        career = cand.get("career_history", [])
        career_text = " ".join([job.get("description", "").lower() + " " + job.get("title", "").lower() for job in career])
        full_text_shipper = f"{summary} {career_text}"
        
        # Unique matches only to prevent padding/stuffing
        shipper_matches = sum(1 for word in SHIPPER_WORDS if word in full_text_shipper)
        shipper_capped = min(shipper_matches, 5)
        
        # Apply shipper bonus: final_score += shipper_capped * 0.8 (max 4.0 points)
        final_score += shipper_capped * 0.8
        
        # Check retrieval and evaluation experience (Phase 4)
        skills_text = " ".join([s.get("name", "").lower() for s in cand.get("skills", [])])
        headline = profile.get("headline", "").lower()
        career_titles = " ".join([job.get("title", "").lower() for job in career])
        career_descs = " ".join([job.get("description", "").lower() for job in career])
        full_text_boost = f"{skills_text} {headline} {summary} {career_titles} {career_descs}"
        
        has_retrieval = any(kw in full_text_boost for kw in RETRIEVAL_KEYWORDS)
        has_evaluation = any(kw in full_text_boost for kw in EVALUATION_KEYWORDS)
        
        if has_retrieval:
            final_score *= 1.15
        if has_evaluation:
            final_score *= 1.10
            
        # Apply CrossEncoder score blend (Phase 1)
        if cross_encoder_scores is not None and cid in cross_encoder_scores:
            rerank_s = cross_encoder_scores[cid]
            final_score = 0.6 * final_score + 0.4 * rerank_s
            
        # 5. Check hard disqualifications at the ABSOLUTE LAST step (score becomes 0.0 if disqualified)
        is_disq, disq_reason = check_disqualifications(cand, features)
        if is_disq:
            logger.warning(f"Candidate {cid} disqualified: {disq_reason}")
            final_score = 0.0
            
        # Keep track of individual scores for reasoning generation later
        flagged_reason = flag_reasons.get(cid, "") if isinstance(flag_reasons, dict) else ""
        breakdown = {
            "candidate": cand,
            "semantic_score": sem_s,
            "structured_score": str_s,
            "vector_score": vec_s,
            "bm25_score": bm25_s,
            "cross_encoder_score": cross_encoder_scores.get(cid, 0.0) if cross_encoder_scores is not None else 0.0,
            "rule_based_structured": structured_breakdowns.get(cid, {}).get("rule_based_score", 0.0),
            "surrogate_structured": structured_breakdowns.get(cid, {}).get("surrogate_score", None),
            "is_flagged": is_flagged,
            "flagged_reason": flagged_reason,
            "features": features
        }
        
        ranked_candidates.append((cid, final_score, breakdown))
        
    # Sort with tiebreaker: descending by final_score, ascending by candidate_id
    ranked_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # ----------------------------------------------------
    # Apply Piecewise Score Mapping to Top 100
    # ----------------------------------------------------
    top_100 = ranked_candidates[:100]
    other_candidates = ranked_candidates[100:]
    
    raw_scores = [x[1] for x in top_100]
    
    def map_group(group_idx_list, L, U):
        if not group_idx_list:
            return []
        
        group_raws = [raw_scores[i] for i in group_idx_list]
        max_raw = max(group_raws)
        min_raw = min(group_raws)
        
        mapped_group_results = []
        n_items = len(group_idx_list)
        
        for idx_in_group, global_idx in enumerate(group_idx_list):
            cid, raw_val, breakdown = top_100[global_idx]
            
            if max_raw > min_raw:
                val_mapped = L + 0.1 + (U - L - 0.2) * (raw_val - min_raw) / (max_raw - min_raw)
            else:
                val_mapped = (L + U) / 2.0
                
            rank_factor = 0.05 * (n_items - 1 - 2 * idx_in_group) / max(1, n_items - 1)
            val_mapped += rank_factor
            val_mapped = max(L, min(U, val_mapped))
            mapped_group_results.append((cid, round(val_mapped, 6), breakdown))
            
        # Ensure strict descending
        for i in range(1, len(mapped_group_results)):
            if mapped_group_results[i][1] >= mapped_group_results[i-1][1]:
                mapped_group_results[i] = (
                    mapped_group_results[i][0],
                    round(mapped_group_results[i-1][1] - 0.0001, 6),
                    mapped_group_results[i][2]
                )
        return mapped_group_results

    # Map Group 1: ranks 1-10 (indices 0-9)
    mapped_g1 = map_group(list(range(0, min(10, len(top_100)))), 95.0, 100.0)
    # Map Group 2: ranks 11-50 (indices 10-49)
    mapped_g2 = map_group(list(range(10, min(50, len(top_100)))), 70.0, 95.0)
    # Map Group 3: ranks 51-100 (indices 50-99)
    mapped_g3 = map_group(list(range(50, min(100, len(top_100)))), 30.0, 70.0)
    
    final_top_100 = mapped_g1 + mapped_g2 + mapped_g3
    
    for i in range(1, len(final_top_100)):
        if final_top_100[i][1] >= final_top_100[i-1][1]:
            final_top_100[i] = (
                final_top_100[i][0],
                round(final_top_100[i-1][1] - 0.0001, 6),
                final_top_100[i][2]
            )
            
    return final_top_100 + other_candidates
