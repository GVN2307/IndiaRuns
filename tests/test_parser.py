import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.jd_parser import parse_jd

class TestParser(unittest.TestCase):
    def setUp(self):
        self.temp_valid = "temp_valid_test.md"
        self.temp_invalid = "temp_invalid_test.md"
        
    def tearDown(self):
        for f in [self.temp_valid, self.temp_invalid]:
            if os.path.exists(f):
                os.remove(f)

    def test_valid_jd_parsing(self):
        valid_text = """
        Job Title: Senior ML Engineer
        Experience Required: 3-5 years
        Location: Pune, India
        
        Requirements:
        - Deep understanding of PyTorch
        - Experience deploying RAG pipelines
        """
        with open(self.temp_valid, "w", encoding="utf-8") as f:
            f.write(valid_text)
            
        # Should parse successfully without error
        jd_reqs = parse_jd(self.temp_valid)
        self.assertEqual(jd_reqs.role, "Senior ML Engineer")
        self.assertIn("pytorch", [s.lower() for s in jd_reqs.must_have_skills])

    def test_empty_jd_raises_error(self):
        with open(self.temp_invalid, "w", encoding="utf-8") as f:
            f.write("   ")
            
        with self.assertRaises(ValueError) as ctx:
            parse_jd(self.temp_invalid)
        self.assertIn("empty or too short", str(ctx.exception))

    def test_unrelated_text_raises_error(self):
        random_text = "This is a random text about a recipe for chocolate chip cookies. It has no special instructions or ingredients."
        with open(self.temp_invalid, "w", encoding="utf-8") as f:
            f.write(random_text)
            
        with self.assertRaises(ValueError) as ctx:
            parse_jd(self.temp_invalid)
        self.assertIn("does not contain any recognizable job role", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
