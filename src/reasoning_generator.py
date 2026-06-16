from typing import Dict, Any

def generate_reasoning(breakdown: Dict[str, Any]) -> str:
    """
    Generates a 1-2 sentence recruiter justification based on candidate data.
    Template: "{current_title} with {years} yrs; {key_strengths}; {key_concerns if any}; response rate {rate}."
    """
    candidate = breakdown.get("candidate", {})
    profile = candidate.get("profile", {})
    features = breakdown.get("features", {})
    
    # Extract basics
    title = profile.get("current_title", "Engineer").strip()
    if not title:
        title = "AI Professional"
        
    years = profile.get("years_of_experience", 0.0)
    
    # Extract strengths
    strengths = []
    
    # 1. Skills
    skills_list = candidate.get("skills", [])
    skills_names = [s.get("name", "") for s in skills_list]
    
    # Core technical matches
    matched_must = []
    core_keywords = ["embedding", "vector search", "retrieval", "ranking", "rag", "faiss", "pinecone", "milvus", "llm", "fine-tuning"]
    for s in skills_names:
        for kw in core_keywords:
            if kw in s.lower() and len(matched_must) < 3:
                matched_must.append(s)
                break
                
    if matched_must:
        strengths.append(f"strong {', '.join(matched_must)} background")
    else:
        strengths.append("relevant technical skills")
        
    # 2. Product company experience
    prod_ratio = features.get("experience_features", {}).get("product_company_ratio", 0.0)
    if prod_ratio > 0.5:
        strengths.append("strong product company pedigree")
    elif prod_ratio > 0:
        strengths.append("some product company exposure")
        
    # 3. Education tier
    edu_tier = features.get("education_features", {}).get("tier_score", 0.2)
    if edu_tier == 1.0:
        strengths.append("tier-1 academic background")
    elif edu_tier == 0.8:
        strengths.append("tier-2 academic background")
        
    # 4. GitHub activity
    github_score = candidate.get("redrob_signals", {}).get("github_activity_score", -1.0)
    if github_score > 70:
        strengths.append("highly active open-source contributor")
        
    # Compile strengths
    strengths_str = "; ".join(strengths[:3])
    if not strengths_str:
        strengths_str = "matching technical profile"
        
    # Extract concerns / limitations
    concerns = []
    
    # 1. Notice period
    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")
    elif notice > 60:
        concerns.append(f"notice period of {notice} days")
        
    # 2. Consulting firm heavy
    cons_ratio = features.get("experience_features", {}).get("consulting_company_ratio", 0.0)
    if cons_ratio > 0.8:
        concerns.append("consulting-only career history")
    elif cons_ratio > 0.5:
        concerns.append("consulting-heavy career history")
        
    # 3. Not open to work
    open_to_work = candidate.get("redrob_signals", {}).get("open_to_work_flag", False)
    if not open_to_work:
        concerns.append("not marked open to work")
        
    # Compile concerns
    concerns_str = "; ".join(concerns[:2])
    
    # Recruiter response rate
    response_rate = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0.0)
    
    # Construct final text
    if concerns_str:
        reasoning = f"{title} with {years:.1f} yrs; {strengths_str}; concern: {concerns_str}; response rate {response_rate:.2f}."
    else:
        reasoning = f"{title} with {years:.1f} yrs; {strengths_str}; response rate {response_rate:.2f}."
        
    return reasoning
