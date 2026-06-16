import gzip
import json
import os

def stream_candidates(filepath):
    """
    Generator yielding one candidate dict at a time.
    Handles both .jsonl and .jsonl.gz files.
    """
    is_gz = filepath.endswith('.gz')
    open_func = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'
    
    try:
        with open_func(filepath, mode, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                    # Gracefully clean up fields to prevent KeyErrors later
                    yield sanitize_candidate(candidate)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

def load_all_candidates(filepath):
    """
    Load all candidates into a list (for smaller samples or training scripts).
    """
    return list(stream_candidates(filepath))

def load_sample_candidates(filepath):
    """
    Load the 50 sample candidates (usually stored as a JSON array of dicts).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
            return [sanitize_candidate(c) for c in candidates]
    except Exception as e:
        print(f"Error loading sample candidates: {e}")
        return []

def sanitize_candidate(candidate):
    """
    Gracefully handles missing or null fields by setting default structures.
    """
    if not isinstance(candidate, dict):
        return {}
        
    candidate.setdefault("candidate_id", "UNKNOWN")
    
    # Profile block
    profile = candidate.setdefault("profile", {})
    if not isinstance(profile, dict):
        profile = {}
        candidate["profile"] = profile
    profile.setdefault("anonymized_name", "Anonymized Candidate")
    profile.setdefault("headline", "")
    profile.setdefault("summary", "")
    profile.setdefault("location", "")
    profile.setdefault("country", "")
    profile.setdefault("years_of_experience", 0.0)
    profile.setdefault("current_title", "")
    profile.setdefault("current_company", "")
    profile.setdefault("current_company_size", "unknown")
    profile.setdefault("current_industry", "")
    
    # Ensure years of experience is a float
    try:
        profile["years_of_experience"] = float(profile.get("years_of_experience") or 0.0)
    except (ValueError, TypeError):
        profile["years_of_experience"] = 0.0

    # Career History block
    career = candidate.get("career_history")
    if not isinstance(career, list):
        candidate["career_history"] = []
    else:
        for job in candidate["career_history"]:
            if not isinstance(job, dict):
                continue
            job.setdefault("company", "")
            job.setdefault("title", "")
            job.setdefault("start_date", "")
            job.setdefault("end_date", None)
            job.setdefault("duration_months", 0)
            job.setdefault("is_current", False)
            job.setdefault("industry", "")
            job.setdefault("company_size", "unknown")
            job.setdefault("description", "")

    # Education block
    education = candidate.get("education")
    if not isinstance(education, list):
        candidate["education"] = []
    else:
        for edu in candidate["education"]:
            if not isinstance(edu, dict):
                continue
            edu.setdefault("institution", "")
            edu.setdefault("degree", "")
            edu.setdefault("field_of_study", "")
            edu.setdefault("start_year", 0)
            edu.setdefault("end_year", 0)
            edu.setdefault("grade", None)
            edu.setdefault("tier", "unknown")

    # Skills block
    skills = candidate.get("skills")
    if not isinstance(skills, list):
        candidate["skills"] = []
    else:
        for s in candidate["skills"]:
            if not isinstance(s, dict):
                continue
            s.setdefault("name", "")
            s.setdefault("proficiency", "beginner")
            s.setdefault("endorsements", 0)
            s.setdefault("duration_months", 0)

    # Redrob Signals block
    signals = candidate.setdefault("redrob_signals", {})
    if not isinstance(signals, dict):
        signals = {}
        candidate["redrob_signals"] = signals
    signals.setdefault("profile_completeness_score", 0.0)
    signals.setdefault("signup_date", "")
    signals.setdefault("last_active_date", "")
    signals.setdefault("open_to_work_flag", False)
    signals.setdefault("profile_views_received_30d", 0)
    signals.setdefault("applications_submitted_30d", 0)
    signals.setdefault("recruiter_response_rate", 0.0)
    signals.setdefault("avg_response_time_hours", 24.0)
    signals.setdefault("skill_assessment_scores", {})
    signals.setdefault("connection_count", 0)
    signals.setdefault("endorsements_received", 0)
    signals.setdefault("notice_period_days", 90)
    
    expected_salary = signals.setdefault("expected_salary_range_inr_lpa", {})
    if not isinstance(expected_salary, dict):
        expected_salary = {"min": 0, "max": 0}
        signals["expected_salary_range_inr_lpa"] = expected_salary
    expected_salary.setdefault("min", 0.0)
    expected_salary.setdefault("max", 0.0)
    
    signals.setdefault("preferred_work_mode", "hybrid")
    signals.setdefault("willing_to_relocate", False)
    signals.setdefault("github_activity_score", -1.0)
    signals.setdefault("search_appearance_30d", 0)
    signals.setdefault("saved_by_recruiters_30d", 0)
    signals.setdefault("interview_completion_rate", 0.0)
    signals.setdefault("offer_acceptance_rate", -1.0)
    signals.setdefault("verified_email", False)
    signals.setdefault("verified_phone", False)
    signals.setdefault("linkedin_connected", False)

    return candidate
