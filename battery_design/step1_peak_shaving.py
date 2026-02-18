"""
Step 1: Peak Load Reduction Sizing
----------------------------------
Sizes battery power and energy to achieve 25% reduction in campus peak demand.
Peak occurs in July. Target: new peak = 0.75 × original peak.
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Paths
LOAD_CSV = os.path.join(PROJECT_ROOT, "UWM_Power.csv")
HOURLY_LOAD = os.path.join(PROJECT_ROOT, "results", "load_profile_tem.csv")

def load_campus_hourly():
    """Load or compute hourly campus load."""
    if os.path.exists(HOURLY_LOAD):
        df = pd.read_csv(HOURLY_LOAD)
        print(f"Using hourly load from {HOURLY_LOAD}")
        return df["Load (kW)"].values
    # Fallback: compute from 15-min data
    print(f"Using 15-min data from {LOAD_CSV} (hourly file not found)")
    df = pd.read_csv(LOAD_CSV)
    cols = [c for c in df.columns if c != "Time"]
    df[cols] = df[cols].replace(" kW", "", regex=True).astype(float)
    total = df[cols].sum(axis=1)
    # Average to hourly (4 rows per hour)
    total = total.values.reshape(-1, 4).mean(axis=1)
    return total

def main():
    print("=" * 60)
    print("STEP 1: Peak Load Reduction Sizing")
    print("=" * 60)

    load_kw = load_campus_hourly()
    peak_actual_kw = float(np.max(load_kw))

    # 25% reduction target
    reduction_pct = 0.25
    peak_target_kw = peak_actual_kw * (1 - reduction_pct)
    p_shave_kw = peak_actual_kw - peak_target_kw

    # Peak typically lasts 2-4 hours; use 3 hours as default
    peak_duration_hours = 3
    e_peak_kwh = p_shave_kw * peak_duration_hours

    # Identify peak month (July = hours ~4344-4415 in 1-indexed, or similar)
    # Hour 1 = Jan 1 00:00; July 1 ≈ hour 3624 (181 days * 24)
    hours = np.arange(1, len(load_kw) + 1)
    july_hours = (hours >= 3624) & (hours < 3864)  # July
    july_peak = float(np.max(load_kw[july_hours])) if np.any(july_hours) else peak_actual_kw

    print(f"\nCampus peak demand:        {peak_actual_kw:.1f} kW")
    print(f"Target (75% of peak):      {peak_target_kw:.1f} kW")
    print(f"Peak shaving required:     {p_shave_kw:.1f} kW")
    print(f"July peak (if available):  {july_peak:.1f} kW")
    print(f"\nBattery sizing for peak shaving:")
    print(f"  Power rating:  ≥ {p_shave_kw:.1f} kW")
    print(f"  Energy (3h):   ≥ {e_peak_kwh:.0f} kWh")

    results = {
        "peak_actual_kw": peak_actual_kw,
        "peak_target_kw": peak_target_kw,
        "p_shave_kw": p_shave_kw,
        "e_peak_kwh": e_peak_kwh,
        "peak_duration_hours": peak_duration_hours,
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step1_peak_shaving_results.csv"),
        index=False,
    )
    print(f"\nResults saved to {OUTPUTS_DIR}/step1_peak_shaving_results.csv")
    return results

if __name__ == "__main__":
    main()
