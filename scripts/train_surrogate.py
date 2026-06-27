import sys
import os
import time
import random
import json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
import pickle

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.config import CANDIDATES_PATH, SURROGATE_PATH, JD_PATH
from src.data_loader import load_all_candidates
from src.features import extract_features, flatten_features
from src.structured_scorer import compute_structured_score

def get_heuristic_score(cand, features):
    """
    Fallback high-quality labeling function that mimics an expert recruiter.
    Computes a score based on technical features, experience, and behavioral signals.
    """
    # Use our structured score breakdown as the base label
    score, breakdown = compute_structured_score(features)
    
    # Add minor noise to simulate human rating variability
    # (keeps it within 0-100 range)
    noise = random.normalvariate(0, 3)
    final_score = max(0.0, min(100.0, score + noise))
    return final_score

def get_llm_score(cand, jd_text, client):
    """
    Calls OpenAI's gpt-4o-mini to get an expert rating for a candidate.
    """
    # Create simple candidate summary to avoid token overflow
    profile = cand.get("profile", {})
    skills = [s.get("name", "") for s in cand.get("skills", [])]
    career = [f"{job.get('title')} at {job.get('company')}: {job.get('description')}" for job in cand.get("career_history", [])[:3]]
    
    cand_summary = f"""
    Title: {profile.get('current_title')}
    Experience: {profile.get('years_of_experience')} years
    Headline: {profile.get('headline')}
    Summary: {profile.get('summary')}
    Skills: {', '.join(skills[:15])}
    Career History:
    {chr(10).join(career)}
    """
    
    prompt = f"""You are an expert recruiter. Score this candidate 0-100 for this Job Description (JD).
    
    Job Description:
    {jd_text[:1500]}
    
    Candidate Profile:
    {cand_summary}
    
    Return JSON only in this format:
    {{
      "score": <float 0-100>,
      "reasoning": "<1-2 sentences justification>",
      "strengths": ["strength1", "strength2"],
      "gaps": ["gap1", "gap2"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=250
        )
        data = json.loads(response.choices[0].message.content)
        return float(data.get("score", 50.0))
    except Exception as e:
        print(f"  Warning: LLM API call failed: {e}. Falling back to heuristic.")
        return None

def main():
    print("=" * 60)
    print(" CVHUNT - Surrogate Model Trainer ")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Load candidates
    print(f"Loading candidates from {CANDIDATES_PATH}...")
    candidates = load_all_candidates(CANDIDATES_PATH)
    print(f"Loaded {len(candidates)} candidates.")
    
    # Deterministically sample 500 candidates
    random.seed(42)
    sample_size = min(500, len(candidates))
    sampled_candidates = random.sample(candidates, sample_size)
    print(f"Sampled {sample_size} candidates for training.")
    
    # 2. Check OpenAI API setup
    api_key = os.environ.get("OPENAI_API_KEY")
    client = None
    jd_text = ""
    
    if api_key:
        print("OpenAI API key detected. Attempting to use gpt-4o-mini for labels.")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            if os.path.exists(JD_PATH):
                with open(JD_PATH, 'r', encoding='utf-8') as f:
                    jd_text = f.read()
        except ImportError:
            print("Warning: openai package not installed. Falling back to heuristic labeling.")
            client = None
    else:
        print("No OpenAI API key found in environment. Using high-quality heuristic labels.")
        
    # 3. Extract features and labels
    X = []
    y = []
    
    print("Extracting features and generating labels...")
    for idx, cand in enumerate(sampled_candidates):
        features = extract_features(cand)
        
        feature_vector = flatten_features(features)
        
        # Determine label (score)
        score = None
        if client:
            print(f"({idx+1}/{sample_size}) Querying LLM for {cand.get('candidate_id')}...")
            score = get_llm_score(cand, jd_text, client)
            # rate limit throttle
            time.sleep(0.1)
            
        if score is None:
            score = get_heuristic_score(cand, features)
            
        X.append(feature_vector)
        y.append(score)
        
    X = np.array(X)
    y = np.array(y)
    
    # 4. Train/Test split & train regressor
    print("Training GradientBoostingRegressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=43
    )
    reg.fit(X_train, y_train)
    
    # Evaluate
    train_score = reg.score(X_train, y_train)
    test_score = reg.score(X_test, y_test)
    print(f"Model R^2 - Train: {train_score:.4f}, Test: {test_score:.4f}")
    
    # Feature importances
    importances = reg.feature_importances_
    feature_names = [
        "must_have_skills", "nice_to_have_skills", "avg_proficiency", "weighted_skill_score", "keyword_stuffing",
        "experience_years", "product_ratio", "consulting_ratio", "title_consistency", "description_quality",
        "edu_tier", "edu_field", "edu_degree", "open_to_work", "response_rate", "active_days",
        "completion_rate", "profile_completeness", "progression", "current_role_relevance"
    ]
    
    print("\nFeature Importances:")
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")
        
    # Save model
    print(f"\nSaving model to {SURROGATE_PATH}...")
    os.makedirs(os.path.dirname(SURROGATE_PATH), exist_ok=True)
    with open(SURROGATE_PATH, "wb") as f:
        pickle.dump(reg, f)
    
    # Save feature importances to text file
    importance_path = os.path.join(os.path.dirname(SURROGATE_PATH), "feature_importances.txt")
    with open(importance_path, 'w', encoding='utf-8') as f:
        for i in sorted_idx:
            f.write(f"{feature_names[i]}: {importances[i]:.4f}\n")
            
    elapsed = time.time() - start_time
    print(f"Surrogate model trained and saved in {elapsed:.2f} seconds.")
    print("=" * 60)

if __name__ == '__main__':
    main()
