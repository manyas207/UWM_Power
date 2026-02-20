"""
Temporal Simulation: Load–PV–Battery with SOC dynamics (15-min capable)
----------------------------------------------------------------------
Simulates battery charging/discharging against *net load*:

  net_load_kW = load_kW - pv_kW

at a fixed time-step (default: 15 minutes). The battery SOC is updated each
step, respecting:
- energy capacity (kWh)
- power limit (kW)
- charge/discharge efficiencies
- optional minimum SOC reserve (resiliency)

PV is assumed to serve load first; any remaining PV can charge the battery,
then export to grid up to an optional export limit; remaining surplus is curtailed.
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
PV_SIMULATION = os.path.join(PROJECT_ROOT, "simulation_17530461_hourly_data 2.csv")

# Bakke and Vet
BAKKE = {"install_kw": 587.05, "grid_limit_kva": 400}
VET = {"install_kw": 525, "grid_limit_kva": 200}

E_RESILIENCE_KWH = 368


def load_campus_load_kw(time_step_min: int = 15):
    """
    Load campus total load (kW) at desired resolution.

    - If `time_step_min == 60` and `results/load_profile_tem.csv` exists, uses it.
    - Otherwise reads 15-min `UWM_Power.csv` and aggregates as needed.
    """
    if time_step_min == 60 and os.path.exists(HOURLY_LOAD):
        df = pd.read_csv(HOURLY_LOAD)
        return df["Load (kW)"].astype(float).values

    df = pd.read_csv(LOAD_CSV)
    cols = [c for c in df.columns if c != "Time"]
    df[cols] = df[cols].replace(" kW", "", regex=True).astype(float)
    load_15 = df[cols].sum(axis=1).values  # 15-min kW

    if time_step_min == 15:
        return load_15
    if time_step_min % 15 != 0:
        raise ValueError(f"time_step_min must be a multiple of 15; got {time_step_min}")

    k = time_step_min // 15
    n = (len(load_15) // k) * k
    return load_15[:n].reshape(-1, k).mean(axis=1)


def load_pv_bakke_vet(path):
    """Load PV and scale to Bakke and Vet. Returns (bakke_kw, vet_kw) hourly."""
    df = pd.read_csv(path)
    for col in ["ac_power", "grid_power", "module_power"]:
        if col in df.columns and df[col].max() > 0:
            pv = df[col].fillna(0).values
            break
    else:
        if "total_irradiance" in df.columns:
            pv = df["total_irradiance"].fillna(0).values * 0.2
        else:
            raise ValueError("No usable PV column")
    mx = np.max(pv)
    if mx > 0:
        bakke = pv * (BAKKE["install_kw"] / mx)
        vet = pv * (VET["install_kw"] / mx)
    else:
        bakke = np.zeros_like(pv)
        vet = np.zeros_like(pv)
    return bakke, vet


def _align_series_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Trim/pad/repeat a series to match `target_len`."""
    x = np.asarray(x, dtype=float)
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    if len(x) > 0 and target_len % len(x) == 0:
        reps = target_len // len(x)
        return np.repeat(x, reps)
    if len(x) == 0:
        return np.zeros(target_len, dtype=float)
    return np.pad(x, (0, target_len - len(x)), mode="edge")


def simulate_battery(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    E_cap_kwh: float,
    P_cap_kw: float,
    dt_hours: float,
    *,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
    soc0_frac: float = 0.5,
    soc_min_kwh: float = 0.0,
    pv_charge_only_kw: np.ndarray | None = None,
    export_limit_kw: float | None = None,
    grid_import_cap_kw: float | None = None,
):
    """
    Time-step battery simulation with SOC dynamics.

    Conventions:
    - `batt_kw` > 0 means discharging (supplying load)
    - `batt_kw` < 0 means charging
    """
    load_kw = np.asarray(load_kw, dtype=float)
    pv_kw = np.asarray(pv_kw, dtype=float)
    pv_charge_only_kw = (
        np.zeros_like(pv_kw) if pv_charge_only_kw is None else np.asarray(pv_charge_only_kw, dtype=float)
    )
    if len(load_kw) != len(pv_kw):
        raise ValueError("load_kw and pv_kw must have same length")
    if len(pv_charge_only_kw) != len(pv_kw):
        raise ValueError("pv_charge_only_kw must have same length as pv_kw")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")

    n = len(load_kw)
    soc = max(soc_min_kwh, soc0_frac * E_cap_kwh) if E_cap_kwh > 0 else 0.0

    grid_import = np.zeros(n, dtype=float)
    grid_export = np.zeros(n, dtype=float)
    curtailed = np.zeros(n, dtype=float)
    batt_kw = np.zeros(n, dtype=float)
    soc_kwh = np.zeros(n, dtype=float)

    for i in range(n):
        load = max(0.0, load_kw[i])
        pv = max(0.0, pv_kw[i])
        pv_charge_only = max(0.0, pv_charge_only_kw[i])

        # PV serves load first
        pv_to_load = min(load, pv)
        remaining_load = load - pv_to_load
        pv_surplus_within = pv - pv_to_load
        pv_surplus_charge_only = pv_charge_only

        # Discharge to reduce grid import.
        # If `grid_import_cap_kw` is set, only discharge enough to cap import to that level.
        discharge_need = remaining_load
        if grid_import_cap_kw is not None:
            discharge_need = max(0.0, remaining_load - grid_import_cap_kw)

        if E_cap_kwh > 0 and P_cap_kw > 0 and discharge_need > 0:
            dis_soc_limited_kw = (max(0.0, soc - soc_min_kwh) * eta_dis) / dt_hours
            p_dis = min(discharge_need, P_cap_kw, dis_soc_limited_kw)
            remaining_load -= p_dis
            soc -= (p_dis / eta_dis) * dt_hours
            batt_kw[i] += p_dis

        # Charge using any PV surplus. Priority: charge-only PV (otherwise curtailed), then within-limit PV surplus.
        pv_available_for_charge = pv_surplus_charge_only + pv_surplus_within
        if E_cap_kwh > 0 and P_cap_kw > 0 and pv_available_for_charge > 0:
            ch_room_limited_kw = (max(0.0, E_cap_kwh - soc) / eta_ch) / dt_hours
            p_ch = min(pv_available_for_charge, P_cap_kw, ch_room_limited_kw)

            # draw down charge-only PV first (avoids curtailment)
            use_charge_only = min(p_ch, pv_surplus_charge_only)
            pv_surplus_charge_only -= use_charge_only
            remaining_charge = p_ch - use_charge_only
            pv_surplus_within -= remaining_charge

            soc += (p_ch * eta_ch) * dt_hours
            batt_kw[i] -= p_ch

        # Export / curtail remaining PV surplus:
        # - within-limit PV can be exported (optionally capped by export_limit_kw)
        # - charge-only PV cannot be exported; any remainder is curtailed
        if pv_surplus_within > 0:
            if export_limit_kw is None:
                grid_export[i] = pv_surplus_within
            else:
                grid_export[i] = min(pv_surplus_within, export_limit_kw)
                curtailed[i] += max(0.0, pv_surplus_within - grid_export[i])
        if pv_surplus_charge_only > 0:
            curtailed[i] += pv_surplus_charge_only

        # Grid import is whatever load remains
        grid_import[i] = remaining_load
        soc = min(max(soc_min_kwh, soc), E_cap_kwh) if E_cap_kwh > 0 else 0.0
        soc_kwh[i] = soc

    summary = {
        "peak_grid_import_kw": float(np.max(grid_import)) if n else 0.0,
        "total_grid_import_kwh": float(np.sum(grid_import) * dt_hours),
        "total_grid_export_kwh": float(np.sum(grid_export) * dt_hours),
        "total_curtailed_kwh": float(np.sum(curtailed) * dt_hours),
        "soc_min_observed_kwh": float(np.min(soc_kwh)) if n else 0.0,
        "soc_max_observed_kwh": float(np.max(soc_kwh)) if n else 0.0,
    }
    series = {
        "grid_import_kw": grid_import,
        "grid_export_kw": grid_export,
        "curtailed_kw": curtailed,
        "batt_kw": batt_kw,
        "soc_kwh": soc_kwh,
    }
    return summary, series


def main():
    print("=" * 60)
    print("TEMPORAL SIMULATION (net load + SOC dynamics)")
    print("=" * 60)

    # Time step (minutes). Default to 15-min to match UWM_Power.csv cadence.
    time_step_min = int(os.environ.get("TIME_STEP_MIN", "15"))
    dt_hours = time_step_min / 60.0

    # Load data
    load_kw = load_campus_load_kw(time_step_min=time_step_min)
    n = len(load_kw)

    if os.path.exists(PV_SIMULATION):
        bakke_h, vet_h = load_pv_bakke_vet(PV_SIMULATION)  # expected hourly
    else:
        bakke_h = np.zeros(8760, dtype=float)
        vet_h = np.zeros(8760, dtype=float)
        print("Warning: No PV simulation; using zero PV")

    step2_path = os.path.join(OUTPUTS_DIR, "step2_resiliency_results.csv")
    e_res = pd.read_csv(step2_path).iloc[0]["e_resiliency_kwh"] if os.path.exists(step2_path) else E_RESILIENCE_KWH

    # Align PV to load resolution by repeating/trim/pad
    bakke_kw = _align_series_to_length(bakke_h, n)
    vet_kw = _align_series_to_length(vet_h, n)

    # PV interconnect limit mode:
    # - per_site: enforce Bakke 400 kVA and Vet 200 kVA separately (excess is "charge-only", otherwise curtailed)
    # - none: treat PV as simple net-load offset (no per-site curtailment)
    pv_limit_mode = os.environ.get("PV_LIMIT_MODE", "per_site").strip().lower()
    if pv_limit_mode not in {"per_site", "none"}:
        raise ValueError("PV_LIMIT_MODE must be 'per_site' or 'none'")

    if pv_limit_mode == "per_site":
        pv_within_kw = np.minimum(bakke_kw, BAKKE["grid_limit_kva"]) + np.minimum(vet_kw, VET["grid_limit_kva"])
        pv_charge_only_kw = np.maximum(0.0, bakke_kw - BAKKE["grid_limit_kva"]) + np.maximum(
            0.0, vet_kw - VET["grid_limit_kva"]
        )
    else:
        pv_within_kw = bakke_kw + vet_kw
        pv_charge_only_kw = np.zeros_like(pv_within_kw)

    # Optional export limit for *within-limit* PV surplus.
    # In per_site mode, PV is already capped at the interconnect limits, so export_limit_kw defaults to unlimited.
    export_limit_env = os.environ.get("EXPORT_LIMIT_KW", "").strip().lower()
    if export_limit_env in {"none", "unlimited", "inf"}:
        export_limit_kw = None
    elif export_limit_env == "":
        export_limit_kw = None if pv_limit_mode == "per_site" else float(BAKKE["grid_limit_kva"] + VET["grid_limit_kva"])
    else:
        export_limit_kw = float(export_limit_env)

    # Peak grid import with PV but without battery (net load definition)
    grid_import_no_batt = np.maximum(0.0, load_kw - pv_within_kw)
    peak_grid_no_batt = float(np.max(grid_import_no_batt)) if n else 0.0
    peak_target = peak_grid_no_batt * 0.75

    print(f"\nData: {n} steps @ {time_step_min}-min (dt={dt_hours:.2f} h)")
    print(f"Peak grid (PV, no batt):   {peak_grid_no_batt:.1f} kW")
    print(f"Target (75% of peak):      {peak_target:.1f} kW")
    print(f"Resiliency reserve:        {e_res:.0f} kWh")
    print(f"PV limit mode:             {pv_limit_mode}")
    print(f"PV export limit:           {'unlimited' if export_limit_kw is None else f'{export_limit_kw:.0f} kW'}")

    # Lightweight sweep: ~12 configs, early exit on first valid
    p_candidates = [200, 300, 400, 500]
    e_candidates = [1000, 1500, 2000, 2500, 3000]

    best = None
    for E_cap in e_candidates:
        if E_cap < e_res + 100:
            continue
        for P_cap in p_candidates:
            summary, _series = simulate_battery(
                load_kw,
                pv_within_kw,
                E_cap,
                P_cap,
                dt_hours,
                soc_min_kwh=float(e_res),
                pv_charge_only_kw=pv_charge_only_kw,
                export_limit_kw=export_limit_kw,
                grid_import_cap_kw=peak_target,
            )
            if summary["peak_grid_import_kw"] <= peak_target:
                best = {
                    "E_cap_kWh": E_cap,
                    "P_cap_kW": P_cap,
                    "Peak_Grid_kW": summary["peak_grid_import_kw"],
                    "Curtailed_kWh": summary["total_curtailed_kwh"],
                }
                break
        if best is not None:
            break

    if best is None:
        # One fallback run with larger size
        summary, _series = simulate_battery(
            load_kw,
            pv_within_kw,
            3500,
            600,
            dt_hours,
            soc_min_kwh=float(e_res),
            pv_charge_only_kw=pv_charge_only_kw,
            export_limit_kw=export_limit_kw,
            grid_import_cap_kw=peak_target,
        )
        best = {
            "E_cap_kWh": 3500,
            "P_cap_kW": 600,
            "Peak_Grid_kW": summary["peak_grid_import_kw"],
            "Curtailed_kWh": summary["total_curtailed_kwh"],
        }
        print(
            f"\n25% target not met. Best: {best['E_cap_kWh']} kWh / {best['P_cap_kW']} kW"
            f" → peak {best['Peak_Grid_kW']:.1f} kW"
        )
    else:
        print(f"\nMinimum viable: {best['E_cap_kWh']} kWh / {best['P_cap_kW']} kW")

    # Solar curtailment potential (no battery), given export limit
    no_batt_summary, _no_batt_series = simulate_battery(
        load_kw,
        pv_within_kw,
        E_cap_kwh=0.0,
        P_cap_kw=0.0,
        dt_hours=dt_hours,
        pv_charge_only_kw=pv_charge_only_kw,
        export_limit_kw=export_limit_kw,
    )

    pd.DataFrame([best]).to_csv(os.path.join(OUTPUTS_DIR, "temporal_sizing_results.csv"), index=False)

    step1_results = {
        "peak_actual_kw": peak_grid_no_batt,
        "peak_target_kw": peak_target,
        "p_shave_kw": peak_grid_no_batt - peak_target,
        "e_peak_kwh": (peak_grid_no_batt - peak_target) * 3,
        "peak_duration_hours": 3,
        "source": f"temporal_{time_step_min}min",
    }
    pd.DataFrame([step1_results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step1_peak_shaving_results.csv"), index=False
    )

    step3_results = {
        "p_solar_kw": float(np.max(_no_batt_series["curtailed_kw"])) if n else 0.0,
        "e_solar_kwh": float(no_batt_summary["total_curtailed_kwh"]),
        "bakke_excess_max_kw": BAKKE["install_kw"] - BAKKE["grid_limit_kva"],
        "vet_excess_max_kw": VET["install_kw"] - VET["grid_limit_kva"],
    }
    pd.DataFrame([step3_results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step3_solar_excess_results.csv"), index=False
    )

    step4_results = {
        "p_battery_kw": best["P_cap_kW"],
        "e_battery_kwh": best["E_cap_kWh"],
        "p_recommended_kw": best["P_cap_kW"],
        "e_recommended_kwh": best["E_cap_kWh"],
        "peak_grid_achieved_kw": best["Peak_Grid_kW"],
        "source": f"temporal_{time_step_min}min_simulation",
    }
    pd.DataFrame([step4_results]).to_csv(
        os.path.join(OUTPUTS_DIR, "step4_combined_results.csv"), index=False
    )

    print(f"\nResults saved to {OUTPUTS_DIR}/")
    return best


if __name__ == "__main__":
    main()
