import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Save PNGs in a dedicated time_series plots folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", "time_series")
os.makedirs(PLOTS_DIR, exist_ok=True)
print("Saving time-series plots to:", PLOTS_DIR)

df = pd.read_csv("load_profile_tem.csv")

plt.figure(figsize=(12, 4))
plt.plot(df["Hour"], df["Load (kW)"])
plt.xlabel("Hour of Year")
plt.ylabel("Load (kW)")
plt.title("Hourly Energy Demand Over the Year")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "hourly_demand_timeseries.png"), dpi=300)
print("Plot saved: hourly_demand_timeseries.png")
plt.show()
plt.close()




df = pd.read_csv("load_profile_tem.csv")

# 1-week moving average
df["Smoothed_Load"] = df["Load (kW)"].rolling(window=24*7, center=True).mean()

plt.figure(figsize=(12, 4))
plt.plot(df["Hour"], df["Load (kW)"], alpha=0.3, label="Hourly Load")
plt.plot(df["Hour"], df["Smoothed_Load"], linewidth=2, label="Weekly Average")
plt.xlabel("Hour of Year")
plt.ylabel("Load (kW)")
plt.title("Smoothed Energy Demand Trend")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "hourly_demand_smoothed.png"), dpi=300)
print("Plot saved: hourly_demand_smoothed.png")
plt.show()
plt.close()



df = pd.read_csv("load_profile_tem.csv")

df["Hour_of_Day"] = (df["Hour"] - 1) % 24

daily_profile = df.groupby("Hour_of_Day")["Load (kW)"].mean()

plt.figure(figsize=(6, 4))
plt.plot(daily_profile.index, daily_profile.values, marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Average Load (kW)")
plt.title("Typical Daily Load Profile")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "typical_daily_profile.png"), dpi=300)
print("Plot saved: typical_daily_profile.png")
plt.show()
plt.close()



df = pd.read_csv("load_profile_tem.csv")

# Approximate month index
df["Month"] = ((df["Hour"] - 1) // (24 * 30)) + 1

monthly = df.groupby("Month")["Load (kW)"].mean()

plt.figure(figsize=(6, 4))
plt.plot(monthly.index, monthly.values, marker="o")
plt.xlabel("Month")
plt.ylabel("Average Load (kW)")
plt.title("Monthly Average Energy Demand")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "monthly_average_load.png"), dpi=300)
print("Plot saved: monthly_average_load.png")
plt.show()
plt.close()






df = pd.read_csv("load_profile_tem.csv")

t = df["Hour"].values
y = df["Load (kW)"].values

# Design matrix (daily + yearly cycles)
X = np.column_stack([
    np.ones(len(t)),
    np.sin(2 * np.pi * t / 24),
    np.cos(2 * np.pi * t / 24),
    np.sin(2 * np.pi * t / 8760),
    np.cos(2 * np.pi * t / 8760)
])

coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ coeffs

plt.figure(figsize=(12, 4))
plt.plot(t, y, alpha=0.3, label="Actual Load")
plt.plot(t, y_hat, linewidth=2, label="Sinusoidal Model")
plt.xlabel("Hour of Year")
plt.ylabel("Load (kW)")
plt.title("Analytical Energy Demand Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sinusoidal_load_model.png"), dpi=300)
print("Plot saved: sinusoidal_load_model.png")
plt.show()
plt.close()
