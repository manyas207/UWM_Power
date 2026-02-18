"""
Run full battery design analysis (Steps 1–4).
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

steps = [
    "step1_peak_shaving.py",
    "step2_resiliency.py",
    "step3_solar_excess.py",
    "step4_combined_sizing.py",
]

for step in steps:
    print("\n")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"ERROR: {step} failed with code {result.returncode}")
        sys.exit(result.returncode)

print("\n" + "=" * 60)
print("All steps completed successfully.")
print("=" * 60)
