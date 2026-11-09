"""
run_all.py  --  RUN THE WHOLE AI PIPELINE WITH ONE COMMAND:   python run_all.py
Each step reads the files the previous step wrote, so the ORDER matters:
  02 clean -> 03 tag courses -> 04 simulate learners -> 05 gap model -> 06 recommender -> 07 evaluate
(File 01 is framework.py - it is imported by the others, not run.)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["02_clean_catalogue.py", "03_tag_courses.py", "04_simulate_learners.py",
         "05_train_gap_model.py", "06_train_recommender.py", "07_evaluate.py"]

for step in STEPS:
    print(f"\n{'=' * 70}\n  RUNNING {step}\n{'=' * 70}")
    code = subprocess.call([sys.executable, str(HERE / step)], cwd=HERE)
    if code != 0:
        sys.exit(f"Step {step} failed")
print("\nAll done. Models are in ml/models, numbers in ml/reports.")
