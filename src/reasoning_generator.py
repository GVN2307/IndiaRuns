from typing import Dict, Any
from datetime import datetime
from src.config import JD_REASONING_KEYWORDS, parse_date

def generate_reasoning(breakdown: Dict[str, Any], rank: int = 50) -> str:
    """
    Generates a high-quality, structured recruiter justification with Pros and Cons.
    The output is formatted as Markdown for the detailed candidate profiler,
    with a short single-line summary at the top separated by a delimiter (---)
    for quick table rendering and backwards compatibility.
    """
    candidate = breakdown.get("candidate", {})
    profile = candidate.get("profile", {})
    features = breakdown.get("features", {})
    signals = candidate.get("redrob_signals", {})
    
    # 1. Fit Tier & Experience
    if rank <= 10:
        fit_phrase = "🏆 Exceptional Match (Top 10)"
        short_fit = "Exceptional fit"
    elif rank <= 50:
        fit_phrase = "✅ Strong Match / Qualified"
        short_fit = "Strong fit"
    else:
        fit_phrase = "⚠️ Partial Match / Under Review"
        short_fit = "Partial fit"
        
    years = profile.get("years_of_experience", 0.0)
    years_str = f"{years:.1f} years experience"
    
    # 2. Extract matched skills
    skills_list = candidate.get("skills", [])
    skills_names = [s.get("name", "") for s in skills_list]
    matched_jd = []
    for kw in JD_REASONING_KEYWORDS:
        if any(kw.lower() in s.lower() for s in skills_names):
            matched_jd.append(kw)
    if not matched_jd:
        matched_jd = ["Python", "Software Engineering"]
        
    skills_matched_str = ", ".join(matched_jd[:3])
    skills_phrase = f"Profile highlights {skills_matched_str}"
    
    # 3. Behavioral signal
    open_to_work = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    
    if open_to_work and response_rate >= 0.8:
        beh_signal = "Candidate is actively open to work with a high response rate"
        short_beh = "Actively looking & highly responsive"
    elif response_rate >= 0.8:
        beh_signal = "Highly responsive candidate"
        short_beh = "Highly responsive"
    elif open_to_work:
        beh_signal = "Active candidate open to new opportunities"
        short_beh = "Open to work"
    else:
        beh_signal = "Demonstrates good platform engagement"
        short_beh = "Standard activity"
        
    # Get sub-features dictionaries
    beh_feat = features.get("behavioral_features", {})
    exp_feat = features.get("experience_features", {})
    edu_feat = features.get("education_features", {})
    career_feat = features.get("career_quality", {})
    skill_feat = features.get("skill_features", {})
    
    must_have_count = skill_feat.get("must_have_count", 0)
    notice = signals.get("notice_period_days", 90)
    active_days = beh_feat.get("active_status", 365)
    
    # 4. Concern collection
    concerns = []
    if notice > 90:
        concerns.append(f"{notice}-day notice period may delay onboarding")
    elif notice > 60:
        concerns.append(f"{notice}-day notice period")
    if response_rate < 0.2:
        concerns.append("very low recruiter responsiveness")
    if active_days > 180:
        concerns.append("inactive for over 6 months")
    if rank > 50 and must_have_count < 6:
        concerns.append("limited must-have skill matches")
        
    if concerns:
        short_concern = "Concern: " + "; ".join(concerns[:2]) + "."
    else:
        short_concern = "No major onboarding concerns."
        
    # Generate single-line summary (same format as legacy)
    short_summary = f"{short_fit}: {years_str}. {skills_phrase}. {short_beh}. {short_concern}"
    
    # 5. Compile Markdown Pros list
    pros = []
    pros.append(f"**Core Expertise:** Matches {must_have_count} requested skills (Highlights: {', '.join(matched_jd[:4])})")
    pros.append(f"**Platform Engagement:** {beh_signal} ({response_rate:.0%} response rate)")
    
    if notice <= 30:
        pros.append(f"**Availability:** Rapid onboarding ({notice}-day notice period)")
        
    if beh_feat.get("is_pune_noida", False):
        pros.append("**Geographic Fit:** Located in preferred hiring hubs (Pune/Noida)")
    elif beh_feat.get("willing_to_relocate", False):
        pros.append("**Flexibility:** Stated willingness to relocate to Pune/Noida")
        
    if beh_feat.get("has_high_assessment", False):
        pros.append("**Verified Assessment:** Scored outstanding (>80%) on verified technical assessments")
        
    progression = career_feat.get("progression", 0.5)
    if progression >= 0.8:
        pros.append("**Career Growth:** Demonstrated upward career progression and role expansion")
        
    # 6. Compile Markdown Cons list
    cons = []
    if notice > 60:
        cons.append(f"**Notice Period:** Long notice period ({notice} days) may delay onboarding timeline")
    if response_rate < 0.2:
        cons.append(f"**Response Rate:** Low responsiveness ({response_rate:.0%}); outreach might require follow-ups")
    if active_days > 90:
        active_months = round(active_days / 30.0, 1)
        cons.append(f"**Activity Gap:** Platform inactivity for {active_months} months (last active {active_days} days ago)")
    if must_have_count < 6:
        cons.append(f"**Skill Gaps:** Lacks {10 - must_have_count} must-have skills requested in the job description")
        
    if not cons:
        cons.append("**None:** No major professional or behavioral red flags detected")
        
    # Build full structured markdown
    pros_md = "\n".join([f"* {p}" for p in pros])
    cons_md = "\n".join([f"* {c}" for c in cons])
    
    detailed_markdown = (
        f"### 📋 Candidate Evaluation\n"
        f"* **Fit Level:** {fit_phrase}\n"
        f"* **Experience:** {years:.1f} years total experience\n\n"
        f"#### 🟢 Pros & Strengths\n"
        f"{pros_md}\n\n"
        f"#### 🔴 Cons & Concerns\n"
        f"{cons_md}"
    )
    
    # Return both separated by a delimiter
    return f"{short_summary}\n\n---\n\n{detailed_markdown}"
