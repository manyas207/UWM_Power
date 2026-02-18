"""
Step 4: Combined Battery Sizing
-------------------------------
Combines requirements from Steps 1–3 and recommends a battery configuration.
"""

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")

def main():
    print("=" * 60)
    print("STEP 4: Combined Battery Sizing")
    print("=" * 60)

    # Load results from previous steps
    s1 = pd.read_csv(os.path.join(OUTPUTS_DIR, "step1_peak_shaving_results.csv")).iloc[0]
    s2 = pd.read_csv(os.path.join(OUTPUTS_DIR, "step2_resiliency_results.csv")).iloc[0]
    s3 = pd.read_csv(os.path.join(OUTPUTS_DIR, "step3_solar_excess_results.csv")).iloc[0]

    p_peak = s1["p_shave_kw"]
    e_peak = s1["e_peak_kwh"]
    p_resil = s2["p_resiliency_kw"]
    e_resil = s2["e_resiliency_kwh"]
    p_solar = s3["p_solar_kw"]
    e_solar = s3["e_solar_kwh"]

    # Combined sizing (shared battery)
    p_battery_kw = max(p_peak, p_resil, p_solar)
    e_battery_kwh = max(e_peak, e_resil, e_solar)

    # Round to practical sizes
    p_round = round(p_battery_kw / 50) * 50
    e_round = round(e_battery_kwh / 100) * 100
    p_round = max(p_round, 50)
    e_round = max(e_round, 100)

    print("\nRequirements by use case:")
    print(f"  Peak shaving:    P ≥ {p_peak:.0f} kW,  E ≥ {e_peak:.0f} kWh")
    print(f"  Resiliency:      P ≥ {p_resil:.0f} kW,  E ≥ {e_resil:.0f} kWh")
    print(f"  Solar excess:    P ≥ {p_solar:.0f} kW,  E ≥ {e_solar:.0f} kWh")

    print("\nCombined (shared battery):")
    print(f"  Power:  {p_battery_kw:.0f} kW  →  recommended: {p_round:.0f} kW")
    print(f"  Energy: {e_battery_kwh:.0f} kWh →  recommended: {e_round:.0f} kWh")

    results = {
        "p_battery_kw": p_battery_kw,
        "e_battery_kwh": e_battery_kwh,
        "p_recommended_kw": p_round,
        "e_recommended_kwh": e_round,
        "peak_shaving_p_kw": p_peak,
        "peak_shaving_e_kwh": e_peak,
        "resiliency_p_kw": p_resil,
        "resiliency_e_kwh": e_resil,
        "solar_excess_p_kw": p_solar,
        "solar_excess_e_kwh": e_solar,
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step4_combined_results.csv"),
        index=False,
    )
    print(f"\nResults saved to {OUTPUTS_DIR}/step4_combined_results.csv")
    return results

if __name__ == "__main__":
    main()
