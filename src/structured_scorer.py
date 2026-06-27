import math
import os
import joblib
from typing import Dict, Any, Tuple

# Global variable to cache the surrogate model
_surrogate_model = None

def get_surrogate_model(model_path: str):
    global _surrogate_model
    if _surrogate_model is None and os.path.exists(model_path):
        try:
            _surrogate_model = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: Failed to load surrogate model: {e}")
    return _surrogate_model

def compute_structured_score(
    features: Dict[str, Any], 
    surrogate_path: str = "",
    ideal_range: Tuple[int, int, int] = (5, 9, 7)
) -> Tuple[float, Dict[str, float]]:
    """
    Computes a 0-100 structured score using rule-based criteria.
    Blends in surrogate model predictions if available.
    """
    skill_feat = features.get("skill_features", {})
    exp_feat = features.get("experience_features", {})
    edu_feat = features.get("education_features", {})
    beh_feat = features.get("behavioral_features", {})
    career_feat = features.get("career_quality", {})

    # ----------------------------------------------------
    # 1. Skill Score (0-100)
    # ----------------------------------------------------
    # Exponential skill scoring: 100 * (1 - exp(-must_have_count/3.0))
    must_have_count = skill_feat.get("must_have_count", 0)
    skill_score = 100.0 * (1.0 - math.exp(-must_have_count / 3.0))
    
    # Penalize keyword stuffing
    if skill_feat.get("keyword_stuffing_flag", False):
        skill_score *= 0.5

    # ----------------------------------------------------
    # 2. Experience Score (0-100)
    # ----------------------------------------------------
    years = exp_feat.get("years", 0.0)
    exp_min, exp_max, exp_peak = ideal_range
    # Step-function experience score adjusted dynamically based on JD requirements
    if exp_min <= years <= exp_max:
        step_score = 100.0
    elif (max(0.0, exp_min - 2) <= years < exp_min) or (exp_max < years <= exp_max + 3):
        step_score = 70.0
    elif (max(0.0, exp_min - 5) <= years < max(0.0, exp_min - 2)) or (exp_max + 3 < years <= exp_max + 6):
        step_score = 40.0
    else:
        step_score = 20.0
    
    # Multiplicative bonuses
    prod_mult = 1.0 + 0.2 * exp_feat.get("product_company_ratio", 0.0)
    cons_mult = 1.0 - 0.2 * exp_feat.get("consulting_company_ratio", 0.0)
    title_mult = 0.8 + 0.2 * exp_feat.get("title_consistency", 0.0)
    
    experience_score = step_score * prod_mult * cons_mult * title_mult
    # NEW: If zero tech titles in entire career history, heavily penalize experience score
    if exp_feat.get("title_consistency", 0.0) == 0.0:
        experience_score *= 0.1  # 90% penalty
    if exp_feat.get("production_shipping_flag", False):
        experience_score *= 1.05
    experience_score = max(0.0, min(100.0, experience_score))

    # ----------------------------------------------------
    # 3. Education Score (0-100)
    # ----------------------------------------------------
    edu_score = (
        edu_feat.get("tier_score", 0.2) * 50.0 +
        edu_feat.get("field_relevance", 0.2) * 30.0 +
        edu_feat.get("degree_level", 0.5) * 20.0
    )
    education_score = max(0.0, min(100.0, edu_score))

    # ----------------------------------------------------
    # 4. Behavioral Score (0-100)
    # ----------------------------------------------------
    active_days = beh_feat.get("active_status", 365)
    if active_days <= 30:
        active_score = 1.0
    elif active_days <= 90:
        active_score = 0.8
    elif active_days <= 180:
        active_score = 0.5
    else:
        active_score = 0.2
        
    beh_score = (
        beh_feat.get("response_rate", 0.0) * 30.0 +
        active_score * 30.0 +
        beh_feat.get("completion_rate", 0.0) * 20.0 +
        beh_feat.get("profile_completeness", 0.0) * 20.0
    )
    behavioral_score = max(0.0, min(100.0, beh_score))

    # ----------------------------------------------------
    # 5. Career Narrative Score (0-100) (additional quality metric)
    # ----------------------------------------------------
    career_narrative_score = (
        career_feat.get("progression", 0.5) * 40.0 +
        career_feat.get("current_role_relevance", 0.1) * 60.0
    )

    # Base rule-based score calculation (career narrative gate multiplier is removed)
    rule_score = (
        skill_score * 0.45 +
        experience_score * 0.35 +
        education_score * 0.10 +
        behavioral_score * 0.10
    )
    
    final_score = rule_score
    surrogate_score = None
    
    if surrogate_path:
        model = get_surrogate_model(surrogate_path)
        if model is not None:
            from src.features import flatten_features
            try:
                feature_vector = flatten_features(features)
                surrogate_score = float(model.predict([feature_vector])[0])
                surrogate_score = max(0.0, min(100.0, surrogate_score))
                # 50% Rule-based, 50% ML surrogate prediction
                final_score = 0.5 * rule_score + 0.5 * surrogate_score
            except Exception as e:
                print(f"Warning: Surrogate prediction failed: {e}. Using rule-based score.")

    # Enforce soft role relevance penalty
    role_rel = career_feat.get("current_role_relevance", 0.05)
    relevance_mult = math.sqrt(role_rel)
    rule_score *= relevance_mult
    final_score *= relevance_mult

    breakdown = {
        "skill_score": skill_score,
        "experience_score": experience_score,
        "education_score": education_score,
        "behavioral_score": behavioral_score,
        "career_narrative_score": career_narrative_score,
        "rule_based_score": rule_score,
    }
    if surrogate_score is not None:
        breakdown["surrogate_score"] = surrogate_score

    return final_score, breakdown
