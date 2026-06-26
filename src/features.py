import re
from datetime import datetime
from typing import Dict, Any, List
from src.config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, PRODUCT_COMPANIES, CONSULTING_COMPANIES, 
    PROFICIENCY_WEIGHTS, REFERENCE_DATE, NON_TECH_TITLE_PATTERN, parse_date
)

def is_skill_match(skill_name: str, target_skills: set) -> bool:
    s_lower = skill_name.lower()
    if s_lower in target_skills:
        return True
    for target in target_skills:
        if target in s_lower:
            idx = s_lower.find(target)
            while idx != -1:
                before = idx > 0 and s_lower[idx-1].isalnum()
                after = idx + len(target) < len(s_lower) and s_lower[idx + len(target)].isalnum()
                if not before and not after:
                    return True
                idx = s_lower.find(target, idx + 1)
    return False

def extract_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts all structured features from a candidate record.
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})
    
    # ----------------------------------------------------
    # 1. Skill Features
    # ----------------------------------------------------
    must_have_count = 0
    nice_to_have_count = 0
    total_prof_weight = 0.0
    weighted_skill_score = 0.0
    matched_skills_count = 0
    
    # Convert lists to sets for fast lookup
    must_have_set = {s.lower() for s in MUST_HAVE_SKILLS}
    nice_to_have_set = {s.lower() for s in NICE_TO_HAVE_SKILLS}
    
    for s in skills:
        s_name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        weight = PROFICIENCY_WEIGHTS.get(prof, 0.2)
        
        is_must = is_skill_match(s_name, must_have_set)
        is_nice = is_skill_match(s_name, nice_to_have_set)
        
        if is_must:
            must_have_count += 1
            matched_skills_count += 1
            total_prof_weight += weight
            weighted_skill_score += weight * (dur / 12.0)
        elif is_nice:
            nice_to_have_count += 1
            matched_skills_count += 1
            total_prof_weight += weight
            weighted_skill_score += weight * (dur / 12.0) * 0.5 # Nice-to-haves get half weight in duration scoring

    avg_proficiency = total_prof_weight / matched_skills_count if matched_skills_count > 0 else 0.0
    
    # Keyword stuffing flag: >20 skills with average duration < 12 months
    keyword_stuffing_flag = False
    if len(skills) > 20:
        durations = [s.get("duration_months", 0) for s in skills]
        avg_dur = sum(durations) / len(durations) if durations else 0
        if avg_dur < 12:
            keyword_stuffing_flag = True

    skill_features = {
        "must_have_count": must_have_count,
        "nice_to_have_count": nice_to_have_count,
        "avg_proficiency": avg_proficiency,
        "weighted_score": weighted_skill_score,
        "keyword_stuffing_flag": keyword_stuffing_flag
    }

    # ----------------------------------------------------
    # 2. Experience Features
    # ----------------------------------------------------
    years = profile.get("years_of_experience", 0.0)
    
    product_company_count = 0
    consulting_company_count = 0
    total_jobs = len(career_history)
    title_tech_count = 0
    desc_word_count = 0
    tech_word_mentions = 0
    
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
    tech_keyword_patterns = re.compile(r"(embeddings|vector|retrieval|ranking|llm|fine-tuning|rag|similarity|search|python|code|pytorch|tensorflow|mlflow|pipeline|model|algorithm|deploy)", re.IGNORECASE)
    production_pattern = re.compile(r"\b(shipped|deployed|production|a/b test|a/b testing)\b", re.IGNORECASE)
    has_production_exp = False
    
    for job in career_history:
        company = job.get("company", "").strip()
        title = job.get("title", "").strip()
        desc = job.get("description", "").strip()
        
        # Check product vs consulting company (case-insensitive substring match)
        is_product = any(pc.lower() in company.lower() for pc in PRODUCT_COMPANIES)
        is_consulting = any(cc.lower() in company.lower() for cc in CONSULTING_COMPANIES)
        
        if is_product:
            product_company_count += 1
        if is_consulting:
            consulting_company_count += 1
            
        # Title consistency (check if title matches tech roles and is not a non-tech title)
        if tech_title_patterns.search(title) and not NON_TECH_TITLE_PATTERN.search(title):
            title_tech_count += 1
            
        # Description quality
        desc_words = desc.split()
        desc_word_count += len(desc_words)
        
        # Tech keyword count in descriptions
        tech_word_mentions += len(tech_keyword_patterns.findall(desc))
        
        # Check for production/shipping keywords
        if production_pattern.search(desc) or production_pattern.search(title):
            has_production_exp = True

    product_company_ratio = product_company_count / total_jobs if total_jobs > 0 else 0.0
    consulting_company_ratio = consulting_company_count / total_jobs if total_jobs > 0 else 0.0
    title_consistency = title_tech_count / total_jobs if total_jobs > 0 else 0.0
    avg_desc_len = desc_word_count / total_jobs if total_jobs > 0 else 0.0
    
    # Map description quality to a 0.0 - 1.0 scale
    description_quality = min(1.0, (avg_desc_len / 100.0) * 0.7 + (tech_word_mentions / 5.0) * 0.3) if total_jobs > 0 else 0.0

    experience_features = {
        "years": years,
        "product_company_ratio": product_company_ratio,
        "consulting_company_ratio": consulting_company_ratio,
        "title_consistency": title_consistency,
        "description_quality": description_quality,
        "production_shipping_flag": has_production_exp
    }

    # ----------------------------------------------------
    # 3. Education Features
    # ----------------------------------------------------
    tier_score = 0.2
    field_relevance = 0.2
    degree_level = 0.5
    
    field_patterns = re.compile(r"(computer science|data science|information technology|mathematics|statistics|software|engineering|physics)", re.IGNORECASE)
    master_patterns = re.compile(r"(m\.tech|ms|master|m\.s\.)", re.IGNORECASE)
    phd_patterns = re.compile(r"(phd|ph\.d|doctor)", re.IGNORECASE)
    bachelor_patterns = re.compile(r"(b\.tech|be|b\.s\.|bachelor|b\.sc)", re.IGNORECASE)
    
    for edu in education:
        tier = edu.get("tier", "unknown")
        # Tier score
        if tier == "tier_1":
            tier_score = max(tier_score, 1.0)
        elif tier == "tier_2":
            tier_score = max(tier_score, 0.8)
        elif tier == "tier_3":
            tier_score = max(tier_score, 0.6)
        elif tier == "tier_4":
            tier_score = max(tier_score, 0.4)
            
        # Field relevance
        field = edu.get("field_of_study", "").lower()
        if "computer science" in field or "data science" in field:
            field_relevance = max(field_relevance, 1.0)
        elif field_patterns.search(field):
            field_relevance = max(field_relevance, 0.8)
        elif "engineering" in field:
            field_relevance = max(field_relevance, 0.6)
            
        # Degree level
        degree = edu.get("degree", "").lower()
        if phd_patterns.search(degree):
            degree_level = max(degree_level, 1.0)
        elif master_patterns.search(degree):
            degree_level = max(degree_level, 0.9)
        elif bachelor_patterns.search(degree):
            degree_level = max(degree_level, 0.8)

    education_features = {
        "tier_score": tier_score,
        "field_relevance": field_relevance,
        "degree_level": degree_level
    }

    # ----------------------------------------------------
    # 4. Behavioral Features
    # ----------------------------------------------------
    open_to_work = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    notice_period = signals.get("notice_period_days", 90)
    completion_rate = signals.get("interview_completion_rate", 0.0)
    profile_completeness = signals.get("profile_completeness_score", 0.0) / 100.0
    
    applications_30d = signals.get("applications_submitted_30d", 0)
    spammy_flag = applications_30d > 10

    # Calculate days since active using REFERENCE_DATE
    ref_dt = parse_date(REFERENCE_DATE)
    act_dt = parse_date(signals.get("last_active_date", ""))
    
    if ref_dt and act_dt:
        days_since_active = max(0, (ref_dt - act_dt).days)
    else:
        days_since_active = 365 # Default to inactive if date is missing

    # Location preferences
    location = profile.get("location", "").lower()
    is_pune_noida = "pune" in location or "noida" in location
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # GitHub activity score
    github_score = signals.get("github_activity_score", -1.0)

    # Skill assessments check
    has_high_assessment = False
    assessment_scores = signals.get("skill_assessment_scores", {})
    if isinstance(assessment_scores, dict):
        for s_name, score in assessment_scores.items():
            if isinstance(score, (int, float)) and score > 80:
                has_high_assessment = True
                break

    behavioral_features = {
        "open_to_work": open_to_work,
        "response_rate": response_rate,
        "active_status": days_since_active, # Lower is better (more active)
        "notice_period": notice_period,
        "completion_rate": completion_rate,
        "profile_completeness": profile_completeness,
        "applications_submitted_30d": applications_30d,
        "spammy_applications": spammy_flag,
        "is_pune_noida": is_pune_noida,
        "willing_to_relocate": willing_to_relocate,
        "github_activity_score": github_score,
        "has_high_assessment": has_high_assessment
    }

    # ----------------------------------------------------
    # 5. Career Quality
    # ----------------------------------------------------
    # Progression: check if older roles were junior and recent roles are senior
    progression = 0.5
    if len(career_history) > 1:
        # Sort jobs by start_date ascending (oldest first)
        jobs_sorted = []
        for job in career_history:
            sd = parse_date(job.get("start_date", ""))
            if sd:
                jobs_sorted.append((sd, job.get("title", "").lower()))
        
        if len(jobs_sorted) > 1:
            jobs_sorted.sort(key=lambda x: x[0])
            first_title = jobs_sorted[0][1]
            last_title = jobs_sorted[-1][1]
            
            is_first_junior = any(kw in first_title for kw in ["intern", "junior", "associate", "analyst"])
            is_last_senior = any(kw in last_title for kw in ["senior", "lead", "staff", "principal", "manager", "director"])
            
            if is_first_junior and is_last_senior:
                progression = 1.0
            elif is_last_senior or ("senior" in last_title or "lead" in last_title):
                progression = 0.8
            else:
                progression = 0.6
    else:
        # Only one job: check if senior
        if career_history:
            title = career_history[0].get("title", "").lower()
            if any(kw in title for kw in ["senior", "lead", "staff", "principal", "manager"]):
                progression = 0.8
            else:
                progression = 0.6
                
    # Current role relevance to Senior AI Engineer
    current_title = profile.get("current_title", "").lower()
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    
    current_role_relevance = 0.05
    is_non_tech_title = bool(NON_TECH_TITLE_PATTERN.search(current_title))

    if is_non_tech_title:
        # Non-tech titles can only get minimal relevance even if they mention AI in summary
        # They need STRONG evidence: must-have skills AND tech job history
        has_strong_tech_evidence = (
            skill_features.get("must_have_count", 0) >= 3 and  # Actually has the skills
            experience_features.get("title_consistency", 0) > 0.5      # Tech titles in career history
        )
        if has_strong_tech_evidence:
            current_role_relevance = 0.15  # Capped low even with evidence
        else:
            current_role_relevance = 0.05  # Default for non-tech
    else:
        # Tech titles - existing logic applies
        if any(kw in current_title or kw in headline for kw in ["ai", "ml", "machine learning", "nlp", "vector search", "retrieval", "ranking"]):
            current_role_relevance = 1.0
        elif "data scientist" in current_title or "data scientist" in headline:
            eng_pattern = re.compile(r"\b(software engineer|backend engineer|frontend engineer|data engineer|ml engineer|ai engineer|machine learning engineer|systems engineer|platform engineer|infrastructure engineer|developer|programmer)\b", re.IGNORECASE)
            has_eng_exp = any(eng_pattern.search(job.get("title", "")) for job in career_history)
            current_role_relevance = 0.6 if has_eng_exp else 0.2
        elif "deep learning" in current_title or "deep learning" in headline:
            current_role_relevance = 0.6
        elif "computer vision" in current_title or "computer vision" in headline:
            # JD says CV without NLP/IR is a disqualifier
            has_nlp = any(kw in summary for kw in ["nlp", "natural language", "text", "retrieval", "ranking"])
            current_role_relevance = 0.5 if has_nlp else 0.10
        elif any(kw in current_title or kw in headline for kw in ["software engineer", "backend", "data engineer"]):
            current_role_relevance = 0.6
        elif any(kw in summary for kw in ["ai", "ml", "machine learning", "nlp", "retrieval"]):
            current_role_relevance = 0.4
        else:
            current_role_relevance = 0.05

    # Penalize "Junior" titles with high experience
    if any(jw in current_title for jw in ["junior", "associate", "intern"]) and years > 4:
        current_role_relevance *= 0.3  # Heavy penalty

    career_quality = {
        "progression": progression,
        "current_role_relevance": current_role_relevance,
        "is_non_tech": bool(NON_TECH_TITLE_PATTERN.search(current_title))
    }

    # Penalize skill claims from non-tech titles severely
    if career_quality.get("is_non_tech", False):
        skill_features["must_have_count"] = int(skill_features.get("must_have_count", 0) * 0.2)
        skill_features["nice_to_have_count"] = int(skill_features.get("nice_to_have_count", 0) * 0.2)
        skill_features["avg_proficiency"] *= 0.3
        skill_features["weighted_score"] *= 0.1

    return {
        "skill_features": skill_features,
        "experience_features": experience_features,
        "education_features": education_features,
        "behavioral_features": behavioral_features,
        "career_quality": career_quality
    }

def flatten_features(features: Dict[str, Any]) -> List[float]:
    """
    Flattens the candidate features dictionary into a 20-element 1D list in a fixed order.
    MUST be consistent between structured_scorer.py and train_surrogate.py.
    """
    skill_feat = features.get("skill_features", {})
    exp_feat = features.get("experience_features", {})
    edu_feat = features.get("education_features", {})
    beh_feat = features.get("behavioral_features", {})
    career_feat = features.get("career_quality", {})
    
    return [
        float(skill_feat.get("must_have_count", 0)),
        float(skill_feat.get("nice_to_have_count", 0)),
        float(skill_feat.get("avg_proficiency", 0.0)),
        float(skill_feat.get("weighted_score", 0.0)),
        float(skill_feat.get("keyword_stuffing_flag", False)),
        float(exp_feat.get("years", 0.0)),
        float(exp_feat.get("product_company_ratio", 0.0)),
        float(exp_feat.get("consulting_company_ratio", 0.0)),
        float(exp_feat.get("title_consistency", 0.0)),
        float(exp_feat.get("description_quality", 0.0)),
        float(edu_feat.get("tier_score", 0.0)),
        float(edu_feat.get("field_relevance", 0.0)),
        float(edu_feat.get("degree_level", 0.0)),
        float(beh_feat.get("open_to_work", False)),
        float(beh_feat.get("response_rate", 0.0)),
        float(beh_feat.get("active_status", 365)),
        float(beh_feat.get("completion_rate", 0.0)),
        float(beh_feat.get("profile_completeness", 0.0)),
        float(career_feat.get("progression", 0.0)),
        float(career_feat.get("current_role_relevance", 0.0))
    ]
