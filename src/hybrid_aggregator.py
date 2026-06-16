from typing import Dict, Any, List, Tuple, Set
from src.config import SCORE_WEIGHTS, BEHAVIORAL_MULTIPLIERS

def aggregate_scores(semantic_score: float, structured_score: float, vector_score: float) -> float:
    """
    Computes the weighted average of the 3 score engines.
    Weights: semantic=0.35, structured=0.40, vector=0.25
    """
    w_sem = SCORE_WEIGHTS.get("semantic", 0.35)
    w_str = SCORE_WEIGHTS.get("structured", 0.40)
    w_vec = SCORE_WEIGHTS.get("vector", 0.25)
    
    return (semantic_score * w_sem) + (structured_score * w_str) + (vector_score * w_vec)

def apply_behavioral_modifiers(base_score: float, beh_feat: Dict[str, Any]) -> float:
    """
    Applies behavioral modifiers as multiplicative penalties based on features.
    """
    modified_score = base_score
    
    # 1. Inactive penalty (last_active_date > 90 days AND not open_to_work)
    # Note: beh_feat["active_status"] is days_since_active
    days_since_active = beh_feat.get("active_status", 365)
    open_to_work = beh_feat.get("open_to_work", False)
    if days_since_active > 90 and not open_to_work:
        modified_score *= BEHAVIORAL_MULTIPLIERS.get("inactive_penalty", 0.75)
        
    # 2. Low response penalty (recruiter_response_rate < 0.2)
    response_rate = beh_feat.get("response_rate", 0.0)
    if response_rate < 0.2:
        modified_score *= BEHAVIORAL_MULTIPLIERS.get("low_response_penalty", 0.85)
        
    # 3. Long notice penalty (notice_period_days > 90)
    notice_period = beh_feat.get("notice_period", 90)
    if notice_period > 90:
        modified_score *= BEHAVIORAL_MULTIPLIERS.get("long_notice_penalty", 0.90)
        
    # 4. Low completion penalty (interview_completion_rate < 0.3)
    completion_rate = beh_feat.get("completion_rate", 0.0)
    if completion_rate < 0.3:
        modified_score *= BEHAVIORAL_MULTIPLIERS.get("low_completion_penalty", 0.90)
        
    return modified_score

def apply_honeypot_penalty(score: float, is_flagged: bool) -> float:
    """
    Caps the final score at 20.0 if flagged as a honeypot.
    """
    if is_flagged:
        return min(20.0, score)
    return score

def compute_final_ranking(
    candidates_data: List[Dict[str, Any]], 
    semantic_scores: Dict[str, float], 
    structured_scores: Dict[str, float], 
    structured_breakdowns: Dict[str, Dict[str, float]], 
    vector_scores: Dict[str, float], 
    flagged_ids: Set[str],
    features_list: List[Dict[str, Any]] = None
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
        
        # 1. Aggregate
        agg_score = aggregate_scores(sem_s, str_s, vec_s)
        
        # Reuse pre-extracted features if available
        if features_list is not None and idx < len(features_list):
            features = features_list[idx]
        else:
            from src.features import extract_features
            features = extract_features(cand)
            
        beh_feat = features.get("behavioral_features", {})
        
        # 2. Apply modifiers
        modified_score = apply_behavioral_modifiers(agg_score, beh_feat)
        
        # 3. Apply honeypot penalty
        is_flagged = cid in flagged_ids
        final_score = apply_honeypot_penalty(modified_score, is_flagged)
        
        # Keep track of individual scores for reasoning generation later
        breakdown = {
            "candidate": cand,
            "semantic_score": sem_s,
            "structured_score": str_s,
            "vector_score": vec_s,
            "rule_based_structured": structured_breakdowns.get(cid, {}).get("rule_based_score", 0.0),
            "surrogate_structured": structured_breakdowns.get(cid, {}).get("surrogate_score", None),
            "is_flagged": is_flagged,
            "flagged_reason": flagged_ids.get(cid) if isinstance(flagged_ids, dict) else "",
            "features": features
        }
        
        ranked_candidates.append((cid, final_score, breakdown))
        
    # 4. Sort with tiebreaker: descending by final_score, ascending by candidate_id
    # Since Python's sort is stable, we can sort by candidate_id first, then by final_score descending
    # To do this in a single sort call:
    ranked_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    return ranked_candidates
