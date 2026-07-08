"""
Step 3: Solar Excess Absorption Sizing
--------------------------------------
Sizes battery to capture PV generation exceeding grid export limits.
- Bakke: 587.05 kW install, 400 kVA grid limit → excess up to 187.05 kW
- Vet: 525 kW install, 200 kVA grid limit → excess up to 325 kW
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Solar excess limits (kW) - from challenge
BAKKE = {"install_kw": 587.05, "grid_limit_kva": 400, "excess_max_kw": 187.05}
VET = {"install_kw": 525, "grid_limit_kva": 200, "excess_max_kw": 325}

# Helioscope simulation file(s) - use project simulation as proxy for PV profile
# If you have separate Bakke and Vet simulations, point to them here
PV_SIMULATION = os.path.join(PROJECT_ROOT, "data", "raw", "simulation_17530461_hourly_data 2.csv")

def load_pv_profile(path):
    """Load PV AC power from Helioscope simulation."""
    df = pd.read_csv(path)
    for col in ["ac_power", "grid_power", "module_power"]:
        if col in df.columns and df[col].max() > 0:
            return df[col].fillna(0).values
    # Fallback: use total_irradiance scaled if power columns are zero
    if "total_irradiance" in df.columns:
        irr = df["total_irradiance"].fillna(0).values
        scale = 0.2  # rough kW per W/m²
        return irr * scale
    raise ValueError("No usable PV power column in simulation file")

def main():
    print("=" * 60)
    print("STEP 3: Solar Excess Absorption Sizing")
    print("=" * 60)

    p_solar_kw = BAKKE["excess_max_kw"] + VET["excess_max_kw"]
    e_solar_kwh = p_solar_kw * 4 * 90  # default rough estimate

    if not os.path.exists(PV_SIMULATION):
        print(f"\nPV simulation not found: {PV_SIMULATION}")
        print("Using theoretical max excess for sizing.")
        e_solar_kwh = p_solar_kw * 4 * 90  # ~90 summer days, 4h each
        print(f"\nTotal max excess: {p_solar_kw:.1f} kW (Bakke {BAKKE['excess_max_kw']} + Vet {VET['excess_max_kw']})")
        print(f"Estimated seasonal excess energy: ~{e_solar_kwh:.0f} kWh (rough)")
    else:
        pv = load_pv_profile(PV_SIMULATION)
        # Scale simulation to represent combined Bakke + Vet
        # Simulation may be for one site; scale to combined install: 587 + 525 = 1112 kW
        sim_max = np.max(pv)
        if sim_max > 0:
            scale = (BAKKE["install_kw"] + VET["install_kw"]) / sim_max
            pv_scaled = pv * scale
        else:
            pv_scaled = pv

        # Compute excess per site (simplified: assume simulation is combined or use same profile)
        # Split proportionally: Bakke fraction = 587/(587+525)
        bakke_frac = BAKKE["install_kw"] / (BAKKE["install_kw"] + VET["install_kw"])
        vet_frac = 1 - bakke_frac

        bakke_gen = pv_scaled * bakke_frac
        vet_gen = pv_scaled * vet_frac

        bakke_excess = np.maximum(0, bakke_gen - BAKKE["grid_limit_kva"])
        vet_excess = np.maximum(0, vet_gen - VET["grid_limit_kva"])
        total_excess = bakke_excess + vet_excess

        p_solar_kw = float(np.max(total_excess))
        annual_excess_kwh = float(np.sum(total_excess))
        annual_excess_hours = np.sum(total_excess > 0)
        # Battery energy: need to absorb excess during sun hours, then discharge to load
        # Use max single-day excess or 6h of peak as proxy for storage need
        daily_excess = np.array([np.sum(total_excess[i:i+24]) for i in range(0, len(total_excess)-24, 24)])
        e_solar_kwh = max(
            float(np.max(daily_excess)) if len(daily_excess) > 0 else 0,
            p_solar_kw * 6,  # 6h at peak
        )
        e_solar_kwh = max(e_solar_kwh, 500)  # minimum 500 kWh

        print(f"\nBakke: install {BAKKE['install_kw']} kW, grid limit {BAKKE['grid_limit_kva']} kVA")
        print(f"Vet:   install {VET['install_kw']} kW, grid limit {VET['grid_limit_kva']} kVA")
        print(f"\nFrom PV simulation:")
        print(f"  Peak excess (combined):  {p_solar_kw:.1f} kW")
        print(f"  Annual excess energy:    {annual_excess_kwh:.0f} kWh")
        print(f"  Max single-day excess:   {np.max(daily_excess) if len(daily_excess) > 0 else 0:.0f} kWh")
        print(f"  Hours with excess:       {annual_excess_hours}")

    # Power: must charge at peak excess (use max of theoretical and simulated)
    p_solar_kw = max(p_solar_kw, BAKKE["excess_max_kw"] + VET["excess_max_kw"])

    print(f"\nBattery sizing for solar excess:")
    print(f"  Charge power:  ≥ {p_solar_kw:.1f} kW")
    print(f"  Energy:        ≥ {e_solar_kwh:.0f} kWh (annual excess to absorb)")

    results = {
        "p_solar_kw": p_solar_kw,
        "e_solar_kwh": e_solar_kwh,
        "bakke_excess_max_kw": BAKKE["excess_max_kw"],
        "vet_excess_max_kw": VET["excess_max_kw"],
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step3_solar_excess_results.csv"),
        index=False,
    )
    print(f"\nResults saved to {OUTPUTS_DIR}/step3_solar_excess_results.csv")
    return results

if __name__ == "__main__":
    main()
