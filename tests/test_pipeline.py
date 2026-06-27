import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_aggregator import compute_final_ranking

class TestPipeline(unittest.TestCase):
    def test_score_aggregation_and_gating(self):
        # 1. Create a dummy candidate
        candidates = [{
            "candidate_id": "CAND_TEST_001",
            "profile": {
                "current_title": "Senior AI Engineer",
                "headline": "LLM & RAG Expert",
                "summary": "Building production AI agents."
            },
            "skills": [{"name": "Python"}, {"name": "PyTorch"}, {"name": "LLM"}],
            "career_history": [
                {"title": "AI Engineer", "description": "Developed search and retrieval systems."}
            ]
        }]
        
        # Mock scores
        semantic_scores = {"CAND_TEST_001": 85.0}
        structured_scores = {"CAND_TEST_001": 90.0}
        vector_scores = {"CAND_TEST_001": 80.0}
        bm25_scores = {"CAND_TEST_001": 75.0}
        cross_encoder_scores = {"CAND_TEST_001": 88.0}
        flag_reasons = {}
        
        # Ideal skills mock
        jd_skills = (["python", "pytorch"], ["llm"])
        
        # Extract features mock structure matching features.py output
        features_list = [{
            "skill_features": {
                "must_have_count": 2,
                "nice_to_have_count": 1,
                "weighted_skill_score": 90.0
            },
            "experience_features": {
                "experience_years": 5.0,
                "average_proficiency": 4.0,
                "product_company_ratio": 0.8,
                "consulting_company_ratio": 0.0
            },
            "career_quality": {
                "title_consistency": 100.0,
                "is_non_tech": False,
                "progression": 5.0
            },
            "education": {
                "edu_tier": 2,
                "edu_field": 1,
                "edu_degree": 2
            },
            "behavioral_features": {
                "open_to_work": True,
                "active_status": 5,
                "response_rate": 0.95,
                "notice_period": 30,
                "completion_rate": 0.9,
                "profile_completeness": 0.9,
                "spammy_applications": False,
                "is_pune_noida": True
            }
        }]
        
        # Execute compute_final_ranking
        ranked = compute_final_ranking(
            candidates_data=candidates,
            semantic_scores=semantic_scores,
            structured_scores=structured_scores,
            structured_breakdowns={},
            vector_scores=vector_scores,
            flag_reasons=flag_reasons,
            features_list=features_list,
            bm25_scores=bm25_scores,
            cross_encoder_scores=cross_encoder_scores,
            jd_skills=jd_skills,
            apply_mapping=False
        )
        
        self.assertEqual(len(ranked), 1)
        cid, score, data = ranked[0]
        self.assertEqual(cid, "CAND_TEST_001")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 120.0)  # Bonuses can exceed 100 before mapping

if __name__ == "__main__":
    unittest.main()
