import subprocess
import os
import sys

def validate(csv_path: str) -> bool:
    """
    Executes the official validate_submission.py script on the output CSV.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validator_path = os.path.join(base_dir, "data", "validate_submission.py")
    
    if not os.path.exists(validator_path):
        print(f"Warning: validate_submission.py not found at {validator_path}")
        return False
        
    try:
        result = subprocess.run(
            [sys.executable, validator_path, csv_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Submission CSV format is fully VALID.")
            print(result.stdout.strip())
            return True
        else:
            print("Submission CSV format is INVALID!")
            print("Validator Output:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error running validation: {e}")
        return False
