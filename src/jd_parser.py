import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from src.config import MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, IDEAL_YEARS_MIN, IDEAL_YEARS_MAX, IDEAL_YEARS_PEAK

@dataclass
class JDRequirements:
    must_have_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    ideal_experience_range: Tuple[int, int, int] = (5, 9, 7) # (min, max, peak)
    required_signals: Dict[str, str] = field(default_factory=dict)
    location_preference: str = "Pune/Noida/India"
    disqualifiers: List[str] = field(default_factory=list)
    behavioral_importance: Dict[str, float] = field(default_factory=dict)

def parse_jd(filepath: str) -> JDRequirements:
    """
    Parses job_description.md and extracts key requirements.
    Falls back to hardcoded defaults from config if sections aren't parsed successfully.
    """
    reqs = JDRequirements(
        must_have_skills=MUST_HAVE_SKILLS.copy(),
        nice_to_have_skills=NICE_TO_HAVE_SKILLS.copy(),
        ideal_experience_range=(IDEAL_YEARS_MIN, IDEAL_YEARS_MAX, IDEAL_YEARS_PEAK),
        required_signals={
            "product_experience": "wants product-company experience",
            "behavioral": "response rate, active status, notice period matter"
        },
        location_preference="Pune/Noida/India",
        disqualifiers=[
            "pure research", "consulting-only", "title-chasers", 
            "framework enthusiasts", "LangChain-only under 12 months", 
            "computer vision without NLP/IR"
        ],
        behavioral_importance={
            "response_rate": 0.3,
            "last_active": 0.3,
            "notice_period": 0.2,
            "interview_completion": 0.2
        }
    )
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Try to parse experience range (e.g. 5–9 years or 5-9 years)
        exp_match = re.search(r'Experience Required:\s*(\d+)[\u2013-](\d+)\s*years', content, re.IGNORECASE)
        if exp_match:
            exp_min = int(exp_match.group(1))
            exp_max = int(exp_match.group(2))
            exp_peak = int((exp_min + exp_max) / 2)
            reqs.ideal_experience_range = (exp_min, exp_max, exp_peak)
            
        # Try to parse location preference
        loc_match = re.search(r'Location:\s*([^\n]+)', content, re.IGNORECASE)
        if loc_match:
            reqs.location_preference = loc_match.group(1).strip()
            
    except Exception as e:
        print(f"Warning: Could not parse job description file: {e}. Using defaults.")
        
    return reqs
