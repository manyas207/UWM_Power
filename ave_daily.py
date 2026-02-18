import numpy as np

data = np.loadtxt("results/load_profile_tem.csv", delimiter=",", skiprows=1)
load_kw = data[:, 1]  # Column 1 = Load (kW)
avg_daily_load_kwh = load_kw.mean() * 24  # mean kW × 24 h = kWh per average day
print(f"Average daily load: {avg_daily_load_kwh:.1f} kWh")