"""
Run full battery design analysis.
Uses temporal simulation (load+PV aligned) for sizing; step2 for resiliency.
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

steps = [
    "step2_resiliency.py",      # Resiliency load (critical buildings)
    "temporal_simulation.py",   # Load+PV temporal analysis → sizing (writes step1,3,4 outputs)
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
