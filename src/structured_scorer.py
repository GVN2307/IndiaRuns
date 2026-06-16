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

def compute_structured_score(features: Dict[str, Any], surrogate_path: str = "") -> Tuple[float, Dict[str, float]]:
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
    # Base match: Must-have skills get 75% of weight, Nice-to-have get 25%
    # Using 10 must-haves and 5 nice-to-haves as scaling factors
    must_have_ratio = min(1.0, skill_feat.get("must_have_count", 0) / 10.0)
    nice_to_have_ratio = min(1.0, skill_feat.get("nice_to_have_count", 0) / 5.0)
    
    base_skill = (must_have_ratio * 75.0) + (nice_to_have_ratio * 25.0)
    
    # Scale base_skill by avg proficiency weight (which is between 0.2 and 1.0)
    # Map proficiency: 1.0 (expert) doesn't change it, 0.5 reduces it
    prof_multiplier = 0.5 + 0.5 * skill_feat.get("avg_proficiency", 0.5)
    skill_score = base_skill * prof_multiplier
    
    # Penalize keyword stuffing
    if skill_feat.get("keyword_stuffing_flag", False):
        skill_score *= 0.5

    # ----------------------------------------------------
    # 2. Experience Score (0-100)
    # ----------------------------------------------------
    years = exp_feat.get("years", 0.0)
    # Gaussian decay centered at 7 years. sigma = 2.0 gives ~60% score at 5 and 9 years.
    gaussian_score = 100.0 * math.exp(-((years - 7.0) ** 2) / (2.0 * (2.0 ** 2)))
    
    # Product company bonus (+20 lpa max)
    prod_bonus = 20.0 * exp_feat.get("product_company_ratio", 0.0)
    
    # Consulting company penalty (-20 lpa max)
    cons_penalty = 20.0 * exp_feat.get("consulting_company_ratio", 0.0)
    
    # Title consistency scaling
    title_mult = 0.8 + 0.2 * exp_feat.get("title_consistency", 0.0)
    
    experience_score = (gaussian_score + prod_bonus - cons_penalty) * title_mult
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

    # Base rule-based score calculation
    rule_score = (
        skill_score * 0.40 +
        experience_score * 0.30 +
        education_score * 0.15 +
        behavioral_score * 0.15
    )
    
    # Weave career narrative score subtly as a 10% modifier on rule_score
    rule_score = rule_score * 0.9 + career_narrative_score * 0.1
    
    # ----------------------------------------------------
    # Blend with ML surrogate prediction if available
    # ----------------------------------------------------
    final_score = rule_score
    surrogate_score = None
    
    if surrogate_path:
        model = get_surrogate_model(surrogate_path)
        if model is not None:
            # Flatten features dictionary for prediction
            # MUST match features used in train_surrogate.py
            feature_vector = [
                skill_feat.get("must_have_count", 0),
                skill_feat.get("nice_to_have_count", 0),
                skill_feat.get("avg_proficiency", 0.0),
                skill_feat.get("weighted_score", 0.0),
                float(skill_feat.get("keyword_stuffing_flag", False)),
                exp_feat.get("years", 0.0),
                exp_feat.get("product_company_ratio", 0.0),
                exp_feat.get("consulting_company_ratio", 0.0),
                exp_feat.get("title_consistency", 0.0),
                exp_feat.get("description_quality", 0.0),
                edu_feat.get("tier_score", 0.0),
                edu_feat.get("field_relevance", 0.0),
                edu_feat.get("degree_level", 0.0),
                float(beh_feat.get("open_to_work", False)),
                beh_feat.get("response_rate", 0.0),
                float(beh_feat.get("active_status", 365)),
                beh_feat.get("completion_rate", 0.0),
                beh_feat.get("profile_completeness", 0.0),
                career_feat.get("progression", 0.0),
                career_feat.get("current_role_relevance", 0.0)
            ]
            try:
                surrogate_score = float(model.predict([feature_vector])[0])
                surrogate_score = max(0.0, min(100.0, surrogate_score))
                # 50% Rule-based, 50% ML surrogate prediction
                final_score = 0.5 * rule_score + 0.5 * surrogate_score
            except Exception as e:
                print(f"Warning: Surrogate prediction failed: {e}. Using rule-based score.")

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
