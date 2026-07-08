"""
Step 2: Resiliency Sizing
-------------------------
Sizes battery to supply 25% of critical buildings' average daily load for 3 days.
Critical buildings: dorms + Carson Gulley + DeJope (dining).
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

LOAD_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "UWM_Power.csv")

# Critical buildings from challenge (substrings to match column names in UWM_Power.csv)
RESILIENCY_BUILDINGS = [
    "DeJope", "Goodnight", "Phillips", "Bradley", "Jones", "Sullivan",
    "Chamberlin", "Cole", "Kronshage", "Leopold", "Mack House", "Jorns",
    "Turner", "Gilman", "Humphrey", "Adams", "Tripp", "Slichter",
    "Carson Gulley",
]

def main():
    print("=" * 60)
    print("STEP 2: Resiliency Sizing")
    print("=" * 60)

    df = pd.read_csv(LOAD_CSV)
    # Clean columns
    building_cols = [c for c in df.columns if c != "Time"]
    df[building_cols] = df[building_cols].replace(" kW", "", regex=True).astype(float)

    # Match resiliency buildings to actual columns
    selected = []
    for col in df.columns:
        if col == "Time":
            continue
        for b in RESILIENCY_BUILDINGS:
            if b.lower() in col.lower():
                selected.append(col)
                break
    selected = list(dict.fromkeys(selected))  # preserve order, remove dupes

    critical_df = df[["Time"] + selected].copy()
    critical_cols = [c for c in critical_df.columns if c != "Time"]
    if not critical_cols:
        raise ValueError("No matching critical building columns found. Check RESILIENCY_BUILDINGS.")

    critical_load = critical_df[critical_cols].sum(axis=1)
    # 15-min data -> hourly
    critical_hourly = critical_load.values.reshape(-1, 4).mean(axis=1)

    avg_hourly_kw = float(np.mean(critical_hourly))
    avg_daily_kwh = avg_hourly_kw * 24
    resiliency_load_pct = 0.25
    resiliency_daily_kwh = resiliency_load_pct * avg_daily_kwh
    outage_days = 3
    e_resiliency_kwh = outage_days * resiliency_daily_kwh

    # Power: peak of 25% of hourly critical load
    resiliency_hourly_kw = resiliency_load_pct * critical_hourly
    p_resiliency_kw = float(np.max(resiliency_hourly_kw))

    print(f"\nCritical buildings: {len(critical_cols)}")
    print(f"Aggregated avg daily load:  {avg_daily_kwh:.1f} kWh/day")
    print(f"25% resiliency load:        {resiliency_daily_kwh:.1f} kWh/day")
    print(f"3-day energy required:      {e_resiliency_kwh:.0f} kWh")
    print(f"Peak 25% load (power):      {p_resiliency_kw:.1f} kW")
    print(f"\nBattery sizing for resiliency:")
    print(f"  Power rating:  ≥ {p_resiliency_kw:.1f} kW")
    print(f"  Energy:        ≥ {e_resiliency_kwh:.0f} kWh")

    results = {
        "avg_daily_kwh": avg_daily_kwh,
        "resiliency_daily_kwh": resiliency_daily_kwh,
        "e_resiliency_kwh": e_resiliency_kwh,
        "p_resiliency_kw": p_resiliency_kw,
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step2_resiliency_results.csv"),
        index=False,
    )
    print(f"\nResults saved to {OUTPUTS_DIR}/step2_resiliency_results.csv")
    return results

if __name__ == "__main__":
    main()
