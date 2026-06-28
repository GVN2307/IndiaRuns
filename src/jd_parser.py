import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from src.config import MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, IDEAL_YEARS_MIN, IDEAL_YEARS_MAX, IDEAL_YEARS_PEAK, ALL_TECH_VOCAB

COMMON_NON_SKILLS = {
    'we', 'our', 'us', 'you', 'your', 'they', 'their', 'the', 'this', 'that', 'these', 'those',
    'strong', 'hands-on', 'good', 'experience', 'production', 'required', 'preferred', 'excellent',
    'working', 'prior', 'background', 'degree', 'university', 'college', 'team', 'client', 'customer',
    'project', 'product', 'role', 'design', 'job', 'title', 'description', 'company', 'location',
    'employment', 'type', 'years', 'year', 'month', 'months', 'week', 'weeks', 'day', 'days',
    'hiring', 'candidate', 'candidates', 'engineer', 'engineers', 'developer', 'developers',
    'specialist', 'manager', 'lead', 'architect', 'infrastructure', 'systems', 'framework',
    'frameworks', 'libraries', 'library', 'tool', 'tools', 'platform', 'platforms', 'database',
    'databases', 'search', 'retrieval', 'ranking', 'matching', 'learning', 'intelligence',
    'science', 'technology', 'technologies', 'skills', 'skill', 'inventory', 'things', 'people',
    'career', 'time', 'full-time', 'part-time', 'hybrid', 'office', 'remote', 'relocation',
    'ability', 'knowledge', 'understanding', 'familiarity', 'expert', 'expertise', 'intermediate',
    'beginner', 'advanced', 'proficient', 'proficiency', 'track', 'record', 'proven', 'demonstrated',
    'industry', 'standards', 'best', 'practices', 'methods', 'methodologies', 'process', 'processes',
    'development', 'engineering', 'deployment', 'operations', 'management', 'architecture',
    'building', 'shipping', 'launching', 'deploying', 'running', 'scaling', 'optimizing',
    'evaluating', 'designing', 'implementing', 'maintaining', 'monitoring', 'testing',
    'academic', 'labs', 'research', 'commercial', 'applications', 'solutions', 'problems',
    'challenges', 'code', 'quality', 'clean', 'robust', 'scalable', 'efficient', 'reliable',
    'high-quality', 'performance', 'latency', 'throughput', 'concurrency', 'distributed',
    'scale', 'volume', 'size', 'metrics', 'feedback', 'loops', 'users', 'customers',
    'recruiters', 'recruiting', 'talent', 'acquisition', 'hr-tech', 'marketplace',
    'first', 'second', 'third', 'last', 'next', 'previous', 'current', 'future',
    'many', 'much', 'some', 'any', 'all', 'every', 'each', 'none', 'no', 'not', 'only',
    'also', 'too', 'very', 'quite', 'rather', 'extremely', 'really', 'especially',
    'please', 'read', 'carefully', 'honest', 'different', 'differently', 'most', 'least',
    'raised', 'round', 'series', 'funding', 'early-stage', 'startup', 'startups',
    'google', 'meta', 'amazon', 'microsoft', 'apple', 'netflix', 'swiggy', 'zomato', 'cred',
    'razorpay', 'ola', 'flipkart', 'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'tech', 'mahindra', 'hcl', 'redrob', 'curefit', 'phonepe', 'paytm',
    'bharatpe', 'groww', 'meesho', 'zepto', 'blinkit', 'cure.fit', 'some', 'similar', 
    'something', 'absolutely', 'need', 'like', 'reject', 'for', 'want', 'explicitly',
    'do', 'not', 'vibe', 'check', 'culture-fit', 'matters', 'more', 'at', 'stage',
    'than', 'skills-fit', 'teachable', 'rest', 'mostly', 'isn\'t', 'async-first',
    'write', 'lot', 'painful', 'disagree', 'openly', 'decide', 'quickly', 'abrasive',
    'move', 'fast', 'break', 'caveat', 'internal', 'assumptions', 'user-facing',
    'stable', 'mature', 'codebase', 'productive', 'unstable', 'read', 'between',
    'lines', 'ideal', 'imagining', 'roughly', 'total', 'applied', 'roles', 'companies',
    'services', 'shipped', 'at least', 'one', 'end-to-end', 'recommendation', 'meaningful',
    'opinions', 'dense', 'integration', 'when', 'prompt', 'defend', 'reference',
    'actually', 'built', 'located', 'willing', 'relocate', 'noida', 'pune', 'active',
    'talk', 'aware', 'narrow', 'profile', 'expecting', 'find', 'matches', 'pool',
    'ok', 'great', 'maybes', 'final', 'note', 'participants', 'challenge', 'intelligent',
    'discovery', 'right', 'answer', 'contains', 'ai', 'keywords', 'trap', 'dataset',
    'involves', 'reasoning', 'gap', 'means', 'tier', 'user', 'words', 'profile',
    'history', 'shows', 'availability', 'availability', 'recruiter', 'response',
    'rate', 'logged', 'purposes', 'available', 'down-weight', 'appropriately', 'good', 'luck'
}

def clean_skill_term(term: str) -> str:
    term = term.strip().strip('"\'`.,:-()[]{}')
    if not term:
        return ""
    
    term_lower = term.lower()
    if term_lower in COMMON_NON_SKILLS:
        return ""
        
    start_prefixes = [
        "experience with ", "production experience with ", "hands-on experience with ", 
        "strong ", "familiarity with ", "knowledge of ", "understanding of ", 
        "working with ", "expert in ", "expert ", "good ", "excellent "
    ]
    for prefix in start_prefixes:
        if term_lower.startswith(prefix):
            term = term[len(prefix):].strip()
            term_lower = term.lower()
            
    end_suffixes = [
        " experience", " frameworks", " systems", " technologies", " tools", 
        " libraries", " databases", " platform", " platforms", " models", 
        " methods", " practices", " concepts"
    ]
    for suffix in end_suffixes:
        if term_lower.endswith(suffix):
            term = term[:-len(suffix)].strip()
            term_lower = term.lower()
            
    term = term.strip().strip('"\'`.,:-()[]{}')
    term_lower = term.lower()
    
    if len(term) < 2:
        if term_lower not in ['c', 'r']:
            return ""
            
    if len(term) > 30:
        return ""
        
    if term_lower in COMMON_NON_SKILLS:
        return ""
        
    if term.isdigit():
        return ""
        
    words = term_lower.split()
    if all(w in COMMON_NON_SKILLS for w in words):
        return ""
        
    return term

def extract_dynamic_skills(text: str) -> List[str]:
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    skills = []
    for sentence in sentences:
        # 1. Extract from parentheses
        parentheses_matches = re.findall(r'\(([^)]+)\)', sentence)
        for pm in parentheses_matches:
            parts = re.split(r',|;| or | and ', pm, flags=re.IGNORECASE)
            for part in parts:
                cleaned = clean_skill_term(part)
                if cleaned:
                    skills.append(cleaned)
                    
        # 2. Extract after dash or colon
        dash_match = re.search(r'(?:—|--|-:|:)\s*(.+)$', sentence)
        if dash_match:
            after_dash = dash_match.group(1)
            parts = re.split(r',|;| or | and ', after_dash, flags=re.IGNORECASE)
            for part in parts:
                cleaned = clean_skill_term(part)
                if cleaned:
                    skills.append(cleaned)
                    
        # 3. Short bullet points or lines
        if sentence.startswith(('-', '*', '•')) or re.match(r'^\d+\.', sentence):
            clean_line = re.sub(r'^[-\*•]|\d+\.\s*', '', sentence).strip()
            if len(clean_line) < 35:
                cleaned = clean_skill_term(clean_line)
                if cleaned:
                    skills.append(cleaned)
                    
        # 4. Acronyms (2-6 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', sentence)
        for ac in acronyms:
            cleaned = clean_skill_term(ac)
            if cleaned:
                skills.append(cleaned)
                
        # 5. Mixed case words
        mixed_case = re.findall(r'\b[a-zA-Z0-9_.-]*[a-z][A-Z][a-zA-Z0-9_.-]*\b', sentence)
        for mc in mixed_case:
            cleaned = clean_skill_term(mc)
            if cleaned:
                skills.append(cleaned)
                
        # 6. Capitalized words (proper nouns)
        words = re.findall(r'\b[a-zA-Z0-9_+#.-]+\b', sentence)
        for i, word in enumerate(words):
            if word and word[0].isupper():
                if i == 0:
                    if not word.isupper() and not any(c.isupper() for c in word[1:]):
                        continue
                cleaned = clean_skill_term(word)
                if cleaned:
                    skills.append(cleaned)
                    
    seen = set()
    unique_skills = []
    for s in skills:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique_skills.append(s)
            
    return unique_skills

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

    if not content or len(content.strip()) < 30:
        raise ValueError("Uploaded file is empty or too short (minimum 30 characters required).")

    # Validate if it looks like a Job Description
    content_lower = content.lower()
    recruitment_keywords = [
        "experience", "skill", "job", "role", "title", "position", "require", 
        "look for", "responsibilit", "qualification", "candidate", "engineer", 
        "developer", "analyst", "manager", "architect"
    ]
    has_recruitment_term = any(k in content_lower for k in recruitment_keywords)
    has_tech_skill = any(re.search(rf'\b{re.escape(s.lower())}\b', content_lower) for s in MUST_HAVE_SKILLS + NICE_TO_HAVE_SKILLS)
    dyn_skills = extract_dynamic_skills(content)
    
    if not has_recruitment_term and not has_tech_skill and not dyn_skills:
        raise ValueError("Invalid Job Description: The file content does not contain any recognizable job role, requirements, or professional skills.")

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
        lines = content.split('\n')
        must_have_idx = -1
        nice_to_have_idx = -1
        boundary_idx = -1
        
        must_have_patterns = ["absolutely need", "must have", "requirements", "essential", "required skills", "skills inventory"]
        nice_to_have_patterns = ["nice to have", "preferred", "would like", "like you to have", "like to have", "desirable", "plus", "wishlist"]
        boundary_patterns = ["do not want", "don't want", "explicitly do not", "location", "comp,", "logistics", "vibe check", "benefits", "compensation", "about us", "about company", "how to apply"]

        char_pos = 0
        line_positions = []
        for line in lines:
            line_positions.append((char_pos, line))
            char_pos += len(line) + 1
            
        for pos, line in line_positions:
            line_lower = line.lower()
            if len(line) > 100:
                continue
                
            if must_have_idx == -1:
                if any(p in line_lower for p in must_have_patterns):
                    must_have_idx = pos
                    continue
                    
            if nice_to_have_idx == -1:
                if any(p in line_lower for p in nice_to_have_patterns):
                    if must_have_idx == -1 or pos > must_have_idx:
                        nice_to_have_idx = pos
                        continue
                        
            if boundary_idx == -1:
                if any(p in line_lower for p in boundary_patterns):
                    if pos > max(must_have_idx, nice_to_have_idx):
                        boundary_idx = pos
                        continue

        content_lower = content.lower()
        must_have_sec = ""
        nice_to_have_sec = ""
        
        if must_have_idx != -1 and nice_to_have_idx != -1:
            if must_have_idx < nice_to_have_idx:
                must_have_sec = content[must_have_idx:nice_to_have_idx]
                if boundary_idx != -1 and boundary_idx > nice_to_have_idx:
                    nice_to_have_sec = content[nice_to_have_idx:boundary_idx]
                else:
                    nice_to_have_sec = content[nice_to_have_idx:]
            else:
                nice_to_have_sec = content[nice_to_have_idx:must_have_idx]
                if boundary_idx != -1 and boundary_idx > must_have_idx:
                    must_have_sec = content[must_have_idx:boundary_idx]
                else:
                    must_have_sec = content[must_have_idx:]
                    
            # Extract vocab-based skills
            for skill in ALL_TECH_VOCAB:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(must_have_sec.lower()):
                    must_have_skills.append(skill)
                elif pattern.search(nice_to_have_sec.lower()):
                    nice_to_have_skills.append(skill)
                    
            # Extract dynamic skills
            dyn_must = extract_dynamic_skills(must_have_sec)
            dyn_nice = extract_dynamic_skills(nice_to_have_sec)
            
            # Combine them
            for skill in dyn_must:
                if skill.lower() not in [s.lower() for s in must_have_skills]:
                    must_have_skills.append(skill)
            for skill in dyn_nice:
                if skill.lower() not in [s.lower() for s in nice_to_have_skills]:
                    nice_to_have_skills.append(skill)
                    
        elif must_have_idx != -1:
            must_have_sec = content[must_have_idx:boundary_idx] if (boundary_idx != -1 and boundary_idx > must_have_idx) else content[must_have_idx:]
            for skill in ALL_TECH_VOCAB:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(must_have_sec.lower()):
                    must_have_skills.append(skill)
            dyn_must = extract_dynamic_skills(must_have_sec)
            for skill in dyn_must:
                if skill.lower() not in [s.lower() for s in must_have_skills]:
                    must_have_skills.append(skill)
                    
        elif nice_to_have_idx != -1:
            nice_to_have_sec = content[nice_to_have_idx:boundary_idx] if (boundary_idx != -1 and boundary_idx > nice_to_have_idx) else content[nice_to_have_idx:]
            for skill in ALL_TECH_VOCAB:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(nice_to_have_sec.lower()):
                    nice_to_have_skills.append(skill)
            dyn_nice = extract_dynamic_skills(nice_to_have_sec)
            for skill in dyn_nice:
                if skill.lower() not in [s.lower() for s in nice_to_have_skills]:
                    nice_to_have_skills.append(skill)
                    
        else:
            # Fallback when no section is found: parse whole document
            for skill in ALL_TECH_VOCAB:
                pattern = re.compile(rf'\b{re.escape(skill.lower())}\b')
                if pattern.search(content_lower):
                    if skill in MUST_HAVE_SKILLS:
                        must_have_skills.append(skill)
                    else:
                        nice_to_have_skills.append(skill)
            dyn_all = extract_dynamic_skills(content)
            for skill in dyn_all:
                skill_lower = skill.lower()
                if skill_lower not in [s.lower() for s in must_have_skills] and skill_lower not in [s.lower() for s in nice_to_have_skills]:
                    if any(ns.lower() == skill_lower for ns in NICE_TO_HAVE_SKILLS):
                        nice_to_have_skills.append(skill)
                    else:
                        must_have_skills.append(skill)

        # 6. Extract dynamic description summary (first few long sentences)
        clean_text = re.sub(r'#.*', '', content)
        clean_text = re.sub(r'(?:location|experience|employment type):.*', '', clean_text, flags=re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r'[.\n]', clean_text) if s.strip()]
        
        desc_sentences = []
        for s in sentences:
            if len(s) > 30 and not any(h in s.lower() for h in must_have_patterns + nice_to_have_patterns):
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
