import re
from typing import Dict, Any, Set, Tuple
from src.config import FOUNDING_YEARS, CONSULTING_COMPANIES, NON_TECH_TITLE_PATTERN

def check_timeline_impossibility(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    career = candidate.get("career_history", [])
    for job in career:
        company = job.get("company", "").strip()
        start_date = job.get("start_date", "")
        
        for name, year in FOUNDING_YEARS.items():
            if name.lower() in company.lower() and start_date:
                try:
                    start_year = int(start_date.split("-")[0])
                    if start_year < year:
                        return True, f"Worked at {company} in {start_year} but founded in {year}"
                except (ValueError, IndexError):
                    continue
    return False, ""

def check_skill_proficiency_mismatch(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    skills = candidate.get("skills", [])
    expert_zero = [s for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0]
    if expert_zero:
        return True, f"Expert/Advanced skills with 0 months duration: {[s['name'] for s in expert_zero]}"
    
    adv_count = sum(1 for s in skills if s.get("proficiency") in ["expert", "advanced"])
    if adv_count > 15:
        return True, f"Too many expert/advanced skills: {adv_count}"
        
    return False, ""

def check_title_description_mismatch(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "").lower()
    title = profile.get("current_title", "").lower()
    
    tech_keywords = ["ai", "ml", "machine learning", "nlp", "engineer", "scientist", "retrieval", "search", "ranking", "developer", "programmer", "architect"]
    is_tech_title = any(kw in headline or kw in title for kw in tech_keywords)
    
    if is_tech_title:
        career = candidate.get("career_history", [])
        desc_text = " ".join([job.get("description", "").lower() for job in career])
        summary = profile.get("summary", "").lower()
        full_text = desc_text + " " + summary
        
        # Core technical content words
        tech_content = ["code", "python", "model", "system", "pipeline", "data", "develop", "build", "program", "train", "algorithm", "software", "git", "sql"]
        has_tech_content = any(term in full_text for term in tech_content)
        
        # Non-technical content words
        non_tech_content = ["sales", "marketing", "hr", "recruiting", "talent acquisition", "customer support", "accounting", "finance"]
        has_non_tech_content = any(term in full_text for term in non_tech_content)
        
        if not has_tech_content and has_non_tech_content:
            return True, "Technical title but job descriptions contain only non-technical content (sales/marketing/etc.)"
            
    return False, ""

def check_keyword_stuffing(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    skills = candidate.get("skills", [])
    if len(skills) > 20:
        durations = [s.get("duration_months", 0) for s in skills]
        avg_dur = sum(durations) / len(durations) if durations else 0
        if avg_dur < 12:
            return True, f"Keyword stuffing: {len(skills)} skills with avg duration {avg_dur:.2f} months"
    return False, ""

def check_consulting_trap(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    career = candidate.get("career_history", [])
    if not career:
        return False, ""
        
    consulting_set = {cc.lower() for cc in CONSULTING_COMPANIES}
    all_consulting = all(any(c in job.get("company", "").lower() for c in consulting_set) for job in career)
    
    if all_consulting:
        profile = candidate.get("profile", {})
        headline = profile.get("headline", "").lower()
        summary = profile.get("summary", "").lower()
        skills = " ".join([s.get("name", "").lower() for s in candidate.get("skills", [])])
        full_text = headline + " " + summary + " " + skills
        
        # Count AI/ML buzzwords
        ai_buzzwords = ["embedding", "vector search", "retrieval", "ranking", "llm", "fine-tuning", "rag", "sentence-transformers", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "ndcg", "mrr", "map", "lora", "qlora", "peft", "learning-to-rank", "xgboost", "lightgbm"]
        stuffed_count = sum(1 for kw in ai_buzzwords if kw in full_text)
        if stuffed_count > 8:
            return True, f"Consulting-heavy candidate with {stuffed_count} AI keywords stuffed"
            
    return False, ""

def check_plain_language_tier5(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "").lower()
    
    is_high_skill_claim = any(kw in headline for kw in ["ai engineer", "ml engineer", "nlp engineer", "search engineer", "vector search"])
    if is_high_skill_claim:
        career = candidate.get("career_history", [])
        desc_text = " ".join([job.get("description", "").lower() for job in career])
        
        # Check for lack of specific tool names
        specific_tech = ["tensorflow", "pytorch", "transformers", "numpy", "pandas", "scikit", "faiss", "pinecone", "milvus", "weaviate", "spacy", "nltk", "bert", "gpt", "rag", "dense", "similarity", "cosine"]
        has_specific_tech = any(tech in desc_text for tech in specific_tech)
        
        # Check for basic/plain phrases
        plain_phrases = ["helped with", "assisted in", "did some", "learned about", "worked on", "interested in"]
        uses_plain = sum(1 for p in plain_phrases if p in desc_text)
        
        education = candidate.get("education", [])
        all_tier4_or_unknown = all(edu.get("tier") in ["tier_4", "unknown"] for edu in education) if education else True
        
        if not has_specific_tech and uses_plain >= 2 and all_tier4_or_unknown:
            return True, "High-skill claims but plain language descriptions and low-tier education"
            
    return False, ""

def check_non_tech_title_with_tech_skills(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Flags candidates with non-technical job titles who claim
    an unrealistic number of AI/ML skills (title-chasers / honeypots).
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    
    is_non_tech = bool(NON_TECH_TITLE_PATTERN.search(title))
    if not is_non_tech:
        return False, ""
        
    # Count AI/ML-related skills
    ai_keywords = [
        "ai", "ml", "embedding", "vector", "llm", "rag", "fine-tuning",
        "retrieval", "ranking", "faiss", "pinecone", "milvus", "weaviate",
        "sentence-transformer", "transformer", "pytorch", "tensorflow",
        "information retrieval", "vector search", "ndcg", "mrr", "map"
    ]
    skills = candidate.get("skills", [])
    ai_skill_count = sum(
        1 for s in skills
        if any(kw in s.get("name", "").lower() for kw in ai_keywords)
    )
    
    if ai_skill_count >= 3:
        return True, f"Non-technical title '{title}' with {ai_skill_count} AI/ML skills (likely keyword-stuffing)"
        
    return False, ""

def check_non_tech_high_exp_low_skill(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Flags non-technical titles that have high years of experience but 
    minimal actual AI/ML skills - likely false positives from experience Gaussian.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "").lower()
    
    is_non_tech = bool(NON_TECH_TITLE_PATTERN.search(title))
    if not is_non_tech:
        return False, ""
    
    years = profile.get("years_of_experience", 0.0)
    if years < 5:  # Only flag those in the sweet spot
        return False, ""
    
    # Count actual must-have skills
    from src.config import MUST_HAVE_SKILLS
    must_have_set = {s.lower() for s in MUST_HAVE_SKILLS}
    skills = candidate.get("skills", [])
    must_have_count = sum(
        1 for s in skills
        if any(kw in s.get("name", "").lower() for kw in must_have_set)
    )
    
    # If in experience sweet spot but has <2 must-have skills, likely false positive
    if must_have_count < 2:
        return True, f"Non-technical title '{title}' with {years} years exp but only {must_have_count} must-have skills"
        
    return False, ""

def run_honeypot_checks(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Runs all 8 honeypot checks on a candidate record.
    Returns:
        is_flagged: True if flagged by any check.
        reason: Description of the violation.
    """
    checks = [
        check_timeline_impossibility,
        check_skill_proficiency_mismatch,
        check_title_description_mismatch,
        check_keyword_stuffing,
        check_consulting_trap,
        check_plain_language_tier5,
        check_non_tech_title_with_tech_skills,
        check_non_tech_high_exp_low_skill
    ]
    
    for check in checks:
        flagged, reason = check(candidate)
        if flagged:
            return True, reason
            
    return False, ""

def detect_honeypots_in_pool(candidates_list) -> Tuple[Set[str], Dict[str, str]]:
    """
    Scans a pool of candidates and returns:
        flagged_ids: Set of candidate_ids that are honeypots.
        flag_reasons: Dict of candidate_id -> explanation of why flagged.
    """
    flagged_ids = set()
    flag_reasons = {}
    
    for cand in candidates_list:
        cid = cand.get("candidate_id", "")
        flagged, reason = run_honeypot_checks(cand)
        if flagged:
            flagged_ids.add(cid)
            flag_reasons[cid] = reason
            
    return flagged_ids, flag_reasons
