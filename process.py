import pandas as pd

INPUT_CSV = "UWM_Power.csv"
OUTPUT_CSV = "load_profile_tem.csv"

# Read CSV
df = pd.read_csv(INPUT_CSV)

# Parse time
df["Time"] = pd.to_datetime(df["Time"])

# Clean building columns
building_cols = df.columns.drop("Time")
df[building_cols] = (
    df[building_cols]
    .replace(" kW", "", regex=True)
    .astype(float)
)

# Sum all buildings per 15-minute interval (still kW)
df["Total_Load_kW"] = df[building_cols].sum(axis=1)

# Sort by time
df = df.sort_values("Time").reset_index(drop=True)

# Continuous hour index (4 rows = 1 hour)
df["Hour"] = (df.index // 4) + 1

# AVERAGE the 4 quarter-hour values per hour
hourly = (
    df.groupby("Hour")["Total_Load_kW"]
    .mean()
    .reset_index()
)

# Rename columns to exact required format
hourly.columns = ["Hour", "Load (kW)"]

# Save
hourly.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
