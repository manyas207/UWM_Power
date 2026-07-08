import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data = np.loadtxt(PROJECT_ROOT / "results" / "load_profile_tem.csv", delimiter=",", skiprows=1)
load_kw = data[:, 1]  # Column 1 = Load (kW)
avg_daily_load_kwh = load_kw.mean()  # mean kW × 24 h = kWh per average day
print(f"Average daily load: {avg_daily_load_kwh:.1f} kWh")