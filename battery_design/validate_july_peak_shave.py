"""
UW Madison battery requirement — peak shave validation (15-min UWM_Power + PV)

Requirement (paraphrased):
- Campus **annual peak load is in July** (use July as the baseline month for peak).
- **25% reduction** in peak grid demand: new peak = 75% × (July peak grid import, PV, no battery).
  Example: 10 MW peak → 7.5 MW threshold; the battery covers demand above that.
- **Staying below the same threshold during other high-load periods** is optimal but **not required**.

This script:
1. Computes July baseline peak: max grid import without battery, restricted to calendar July.
2. Sets threshold = 0.75 × that July peak.
3. Simulates the battery with `grid_import_cap_kw = threshold` (shave to threshold).
4. **Required:** max grid import in **July** (with battery) ≤ threshold.
5. **Optional / optimal:** report whether **any month** exceeds the threshold (not required to pass).

If the uploaded year’s **calendar maximum** occurs outside July, a warning is printed; the
threshold is still based on **July only**, per the written requirement.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np

from temporal_simulation import (
    LOAD_CSV,
    PV_SIMULATION,
    BAKKE,
    VET,
    load_pv_bakke_vet,
    _align_series_to_length,
    simulate_battery,
    OUTPUTS_DIR,
)

STEP2 = os.path.join(OUTPUTS_DIR, "step2_resiliency_results.csv")


def main():
    df = pd.read_csv(LOAD_CSV)
    cols = [c for c in df.columns if c != "Time"]
    df[cols] = df[cols].replace(" kW", "", regex=True).astype(float)
    load_kw = df[cols].sum(axis=1).values.astype(float)
    ts = pd.to_datetime(df["Time"], format="%m/%d/%y %H:%M")
    month = ts.dt.month.to_numpy()
    july = month == 7

    n = len(load_kw)
    bakke_h, vet_h = load_pv_bakke_vet(PV_SIMULATION)
    bakke_kw = _align_series_to_length(bakke_h, n)
    vet_kw = _align_series_to_length(vet_h, n)

    pv_within_kw = np.minimum(bakke_kw, BAKKE["grid_limit_kva"]) + np.minimum(vet_kw, VET["grid_limit_kva"])
    pv_charge_only_kw = np.maximum(0.0, bakke_kw - BAKKE["grid_limit_kva"]) + np.maximum(
        0.0, vet_kw - VET["grid_limit_kva"]
    )

    grid_no_batt = np.maximum(0.0, load_kw - pv_within_kw)
    annual_peak = float(np.max(grid_no_batt))
    july_peak_no_batt = float(np.max(grid_no_batt[july])) if np.any(july) else 0.0
    peak_idx = int(np.argmax(grid_no_batt))
    peak_month = int(month[peak_idx])

    # 25% reduction → new peak = 75% of July baseline (requirement example: 10 MW → 7.5 MW)
    threshold_kw = 0.75 * july_peak_no_batt

    e_res = float(pd.read_csv(STEP2).iloc[0]["e_resiliency_kwh"]) if os.path.exists(STEP2) else 368.0
    dt_hours = 15.0 / 60.0

    configs = [
        ("REopt_battery1", 329.0, 2025.0),
        ("REopt_unlimited", 531.0, 3968.0),
    ]

    print("=" * 64)
    print("UW MADISON PEAK SHAVE VALIDATION (July baseline, 15-min)")
    print("=" * 64)
    print(f"Timesteps: {n} | dt = {dt_hours} h")
    print()
    print("Baseline (grid import, PV, no battery):")
    print(f"  July peak (used for 25% rule):     {july_peak_no_batt:.2f} kW")
    print(f"  Calendar max this file:           {annual_peak:.2f} kW in month {peak_month} (1=Jan … 7=Jul)")
    if peak_month != 7:
        print(
            "  NOTE: This file’s single worst interval is not in July; July peak is still used\n"
            "        as the requirement states campus annual peak is in July."
        )
    print()
    print(f"Threshold (75% of July peak):       {threshold_kw:.2f} kW  (demand above this → battery)")
    print(f"Resiliency SOC floor:               {e_res:.2f} kWh")
    print()
    print("Controller: discharge to cap grid import at threshold (simulate_battery).")
    print()
    print("— Required: July max grid import (with battery) ≤ threshold —")
    print("— Optional: no other month > threshold (optimal, not required) —")
    print()

    rows = []
    for cfg_name, p_kw, e_kwh in configs:
        summ, series = simulate_battery(
            load_kw,
            pv_within_kw,
            e_kwh,
            p_kw,
            dt_hours,
            soc_min_kwh=float(e_res),
            pv_charge_only_kw=pv_charge_only_kw,
            export_limit_kw=None,
            grid_import_cap_kw=threshold_kw,
        )
        g = series["grid_import_kw"]
        max_july = float(np.max(g[july])) if np.any(july) else 0.0
        max_year = float(np.max(g)) if len(g) else 0.0
        tol = 1e-2
        pass_required = max_july <= threshold_kw + tol
        exceeds_non_july = bool(np.any((g > threshold_kw + tol) & ~july))
        pass_optional_year = max_year <= threshold_kw + tol

        # Months where any interval exceeds threshold (informational)
        bad_months = sorted(set(month[g > threshold_kw + tol].tolist())) if len(g) else []

        rows.append(
            {
                "config": cfg_name,
                "P_kW": p_kw,
                "E_kWh": e_kwh,
                "july_peak_no_batt_kW": july_peak_no_batt,
                "threshold_kW": threshold_kw,
                "peak_july_with_batt_kW": max_july,
                "peak_year_with_batt_kW": max_year,
                "pass_required_july": pass_required,
                "pass_optional_no_month_exceeds": pass_optional_year,
                "months_any_interval_exceeds_threshold": str(bad_months) if bad_months else "",
            }
        )

        req = "PASS" if pass_required else "FAIL"
        opt = "PASS (optimal)" if pass_optional_year else "FAIL (acceptable per spec if July passes)"
        print(f"{cfg_name} ({p_kw:.0f} kW / {e_kwh:.0f} kWh)")
        print(f"  Required — July peak with battery: {max_july:.2f} kW  (≤ {threshold_kw:.2f} kW)  →  {req}")
        print(f"  Optional — Year peak with battery: {max_year:.2f} kW  →  {opt}")
        if bad_months and not pass_optional_year:
            print(f"  Months with any interval > threshold: {bad_months}")
        print()

    out = os.path.join(OUTPUTS_DIR, "july_peak_validation.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
