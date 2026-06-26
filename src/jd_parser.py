import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from src.config import MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, IDEAL_YEARS_MIN, IDEAL_YEARS_MAX, IDEAL_YEARS_PEAK

@dataclass
class JDRequirements:
    role: str = "Senior AI Engineer"
    must_have_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    ideal_experience_range: Tuple[int, int, int] = (5, 9, 7) # (min, max, peak)
    required_signals: Dict[str, str] = field(default_factory=dict)
    location_preference: str = "Pune/Noida/India"
    disqualifiers: List[str] = field(default_factory=list)
    behavioral_importance: Dict[str, float] = field(default_factory=dict)
    description_summary: str = ""

    def to_embedding_text(self) -> str:
        parts = [
            f"Role: {self.role}",
            f"Required skills: {', '.join(self.must_have_skills[:10])}",
            f"Preferred skills: {', '.join(self.nice_to_have_skills[:8])}",
            f"Experience: {self.ideal_experience_range[0]}-{self.ideal_experience_range[1]} years",
            f"Location: {self.location_preference}"
        ]
        if self.description_summary:
            parts.append(self.description_summary)
        return ". ".join(parts)

def parse_jd(filepath: str) -> JDRequirements:
    """
    Parses a job description file (.md, .txt, or .docx) dynamically.
    Scans for job title, experience bounds, location, and matching skills.
    """
    content = ""
    
    # 1. Read file based on extension
    if filepath.endswith('.docx'):
        try:
            import docx
            doc = docx.Document(filepath)
            content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            print(f"Warning: Failed to parse docx {filepath}: {e}. Trying text fallback.")
            
    if not content:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}. Using defaults.")
            content = ""

    # Defaults
    role = "Senior AI Engineer"
    must_have_skills = []
    nice_to_have_skills = []
    ideal_experience_range = (IDEAL_YEARS_MIN, IDEAL_YEARS_MAX, IDEAL_YEARS_PEAK)
    location_preference = "Pune/Noida/India"
    description_summary = ""

    if content:
        # 2. Extract Job Title / Role
        role_match = re.search(r'(?:Job Description|Job Title|Role|Title):\s*([^\n|]+)', content, re.IGNORECASE)
        if role_match:
            role = role_match.group(1).strip().split("—")[0].strip()
        else:
            # Try to grab the first non-empty line as the role heading
            first_lines = [line.strip() for line in content.split("\n") if line.strip()]
            if first_lines:
                role = first_lines[0].replace("#", "").strip().split("—")[0].strip()

        # 3. Extract Experience Range
        exp_match = re.search(r'(?:Experience Required|Experience):\s*(\d+)[\u2013-](\d+)\s*years', content, re.IGNORECASE)
        if exp_match:
            exp_min = int(exp_match.group(1))
            exp_max = int(exp_match.group(2))
            exp_peak = int((exp_min + exp_max) / 2)
            ideal_experience_range = (exp_min, exp_max, exp_peak)

        # 4. Extract Location
        loc_match = re.search(r'(?:Location):\s*([^\n|]+)', content, re.IGNORECASE)
        if loc_match:
            location_preference = loc_match.group(1).strip()

        # 5. Extract Skills dynamically by scanning document sections
        content_lower = content.lower()
        must_have_idx = -1
        nice_to_have_idx = -1
        
        must_have_headers = ["absolutely need", "must have", "requirements", "essential", "required skills", "skills inventory"]
        nice_to_have_headers = ["nice to have", "preferred", "would like", "desirable", "plus", "wishlist"]
        
        for h in must_have_headers:
            idx = content_lower.find(h)
            if idx != -1:
                must_have_idx = idx
                break
        for h in nice_to_have_headers:
            idx = content_lower.find(h)
            if idx != -1:
                nice_to_have_idx = idx
                break

        all_config_skills = MUST_HAVE_SKILLS + NICE_TO_HAVE_SKILLS
        
        if must_have_idx != -1 and nice_to_have_idx != -1:
            if must_have_idx < nice_to_have_idx:
                must_have_sec = content_lower[must_have_idx:nice_to_have_idx]
                nice_to_have_sec = content_lower[nice_to_have_idx:]
            else:
                nice_to_have_sec = content_lower[nice_to_have_idx:must_have_idx]
                must_have_sec = content_lower[must_have_idx:]
                
            for skill in all_config_skills:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(must_have_sec):
                    must_have_skills.append(skill)
                elif pattern.search(nice_to_have_sec):
                    nice_to_have_skills.append(skill)
        else:
            # Differentiate skills based on complete document scan and baseline group mapping
            for skill in MUST_HAVE_SKILLS:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(content_lower):
                    must_have_skills.append(skill)
            for skill in NICE_TO_HAVE_SKILLS:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(content_lower):
                    nice_to_have_skills.append(skill)

        # 6. Extract dynamic description summary (first few long sentences)
        clean_text = re.sub(r'#.*', '', content)
        clean_text = re.sub(r'(?:location|experience|employment type):.*', '', clean_text, flags=re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r'[.\n]', clean_text) if s.strip()]
        
        desc_sentences = []
        for s in sentences:
            if len(s) > 30 and not any(h in s.lower() for h in must_have_headers + nice_to_have_headers):
                desc_sentences.append(s)
                if len(desc_sentences) >= 3:
                    break
        description_summary = ". ".join(desc_sentences)

    # Fallbacks to baseline config lists if none were parsed
    if not must_have_skills:
        must_have_skills = MUST_HAVE_SKILLS.copy()
    if not nice_to_have_skills:
        nice_to_have_skills = NICE_TO_HAVE_SKILLS.copy()

    return JDRequirements(
        role=role,
        must_have_skills=must_have_skills,
        nice_to_have_skills=nice_to_have_skills,
        ideal_experience_range=ideal_experience_range,
        required_signals={
            "product_experience": "wants product-company experience",
            "behavioral": "response rate, active status, notice period matter"
        },
        location_preference=location_preference,
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
        },
        description_summary=description_summary
    )
