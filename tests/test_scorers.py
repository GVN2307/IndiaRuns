import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.structured_scorer import compute_structured_score
# Let's import features and disqualifications
from src.hybrid_aggregator import check_disqualifications

class TestScorers(unittest.TestCase):
    def test_structured_score_bounds(self):
        # Test default scoring with ideal experience
        features = {
            "skill_features": {
                "must_have_count": 5,
                "nice_to_have_count": 3,
                "weighted_skill_score": 80.0
            },
            "experience_features": {
                "experience_years": 7.0,
                "average_proficiency": 3.5,
                "product_company_ratio": 0.8,
                "consulting_company_ratio": 0.0
            },
            "career_quality": {
                "title_consistency": 100.0,
                "is_non_tech": False,
                "progression": 10.0
            },
            "education": {
                "edu_tier": 1,
                "edu_field": 1,
                "edu_degree": 2
            },
            "behavioral_features": {
                "open_to_work": True,
                "active_status": 10,
                "response_rate": 0.9,
                "notice_period": 15,
                "completion_rate": 0.8,
                "profile_completeness": 0.9,
                "spammy_applications": False,
                "is_pune_noida": True
            }
        }
        
        score, breakdown = compute_structured_score(features, ideal_range=(5, 9, 7))
        self.assertGreaterEqual(score, 0.0)
        self.assertIn("skill_score", breakdown)
        self.assertIn("experience_score", breakdown)
        self.assertIn("education_score", breakdown)
        self.assertIn("behavioral_score", breakdown)
        self.assertIn("career_narrative_score", breakdown)

    def test_disqualifications(self):
        # Candidate 1: Research scientist with 0% product experience (Requires >= 30%)
        cand_research = {
            "candidate_id": "CAND_TEST_RES",
            "profile": {
                "current_title": "Research Scientist",
                "headline": "AI Researcher at University",
                "summary": "Academic research in Deep Learning."
            },
            "skills": [{"name": "PyTorch"}, {"name": "LLMs"}],
            "career_history": [
                {"title": "Research Assistant", "description": "Academic lab research."}
            ]
        }
        features_res = {
            "experience_features": {
                "product_company_ratio": 0.0,
                "consulting_company_ratio": 0.0
            },
            "career_quality": {
                "is_non_tech": False
            }
        }
        
        is_disq, reason = check_disqualifications(cand_research, features_res)
        self.assertTrue(is_disq)
        self.assertIn("Research candidate", reason)

        # Candidate 2: Consulting-only history (Requires < 95% consulting ratio)
        cand_consulting = {
            "candidate_id": "CAND_TEST_CONS",
            "profile": {
                "current_title": "Software Engineer",
                "headline": "Consultant"
            },
            "career_history": [
                {"title": "Software Consultant", "description": "Client work."}
            ]
        }
        features_cons = {
            "experience_features": {
                "product_company_ratio": 0.0,
                "consulting_company_ratio": 1.0
            },
            "career_quality": {
                "is_non_tech": False
            }
        }
        is_disq, reason = check_disqualifications(cand_consulting, features_cons)
        self.assertTrue(is_disq)
        self.assertIn("Consulting-only", reason)

if __name__ == "__main__":
    unittest.main()
