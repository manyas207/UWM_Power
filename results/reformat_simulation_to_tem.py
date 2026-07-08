"""
Reformat Helioscope hourly simulation to "tem" CSV format.

Creates a 2-column file shaped like `results/load_profile_tem.csv`:

  Hour,<kW-* column>

By default, writes *two* files:
- `results/pv_profile_tem_ac.csv` with column name "Load (kW-AC)" from `ac_power`
- `results/pv_profile_tem_dc.csv` with column name "Load (kW-DC)" from `module_power`

This preserves the same 2-column "tem" shape while making units explicit.

Additionally writes a normalized profile suitable for many utility / challenge
data requirements:

  (kW-AC output) / (kW-DC nameplate)

per timestep. This is a dimensionless ratio whose units are often written as
"kW-AC/kW-DC nameplate".
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


AC_POWER_COLS = ("ac_power", "grid_power")
DC_POWER_COLS = ("module_power", "module_mpp_power", "actual_dc_power", "optimal_dc_power")
NAMEPLATE_COLS = ("nameplate_power",)


def _pick_power_column(df: pd.DataFrame, preferred: tuple[str, ...]) -> str:
    for col in preferred:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if float(s.max()) > 0:
                return col
    raise ValueError(
        f"No usable power column found. Tried {preferred}. Available columns: {list(df.columns)[:30]}..."
    )


def _to_kw(values: np.ndarray) -> np.ndarray:
    """
    Heuristic unit conversion:
    - Helioscope exports power in Watts for many fields; if max is large, convert W -> kW.
    """
    values = np.asarray(values, dtype=float)
    mx = float(np.nanmax(values)) if values.size else 0.0
    # If peak is > 5,000, it's almost certainly W (e.g., 500,000 W = 500 kW).
    if mx > 5000:
        return values / 1000.0
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(
            Path(__file__).resolve().parents[1], "data", "raw", "simulation_17530461_hourly_data 2.csv"
        ),
        help="Path to Helioscope simulation CSV.",
    )
    parser.add_argument(
        "--output-ac",
        default=os.path.join(Path(__file__).resolve().parent, "pv_profile_tem_ac.csv"),
        help="Output CSV path for AC power (2 columns).",
    )
    parser.add_argument(
        "--output-dc",
        default=os.path.join(Path(__file__).resolve().parent, "pv_profile_tem_dc.csv"),
        help="Output CSV path for DC power (2 columns).",
    )
    parser.add_argument(
        "--output-norm",
        default=os.path.join(Path(__file__).resolve().parent, "pv_profile_tem_norm.csv"),
        help="Output CSV path for normalized kW-AC/kW-DC-nameplate (2 columns).",
    )
    parser.add_argument(
        "--ac-column-name",
        default="Load (kW-AC)",
        help="Name for the AC kW column in the output.",
    )
    parser.add_argument(
        "--dc-column-name",
        default="Load (kW-DC)",
        help="Name for the DC kW column in the output.",
    )
    parser.add_argument(
        "--norm-column-name",
        default="PV (kW-AC/kW-DC nameplate)",
        help="Name for the normalized column in the output.",
    )
    parser.add_argument(
        "--ac-power-column",
        default="",
        help="Force a specific AC input power column (otherwise auto-picks).",
    )
    parser.add_argument(
        "--dc-power-column",
        default="",
        help="Force a specific DC input power column (otherwise auto-picks).",
    )
    parser.add_argument(
        "--nameplate-column",
        default="",
        help="Force a specific DC nameplate column (otherwise auto-picks, e.g. nameplate_power).",
    )
    parser.add_argument(
        "--dc-nameplate-kw",
        type=float,
        default=0.0,
        help="Override DC nameplate in kW-DC (used if nameplate column missing/unusable).",
    )

    args = parser.parse_args()

    inp = Path(args.input)
    out_ac = Path(args.output_ac)
    out_dc = Path(args.output_dc)
    out_norm = Path(args.output_norm)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    ac_col = args.ac_power_column.strip() or _pick_power_column(df, AC_POWER_COLS)
    dc_col = args.dc_power_column.strip() or _pick_power_column(df, DC_POWER_COLS)
    nameplate_col = args.nameplate_column.strip()
    if not nameplate_col:
        # optional; may not exist in every export
        nameplate_col = df.columns.intersection(NAMEPLATE_COLS).tolist()[0] if any(c in df.columns for c in NAMEPLATE_COLS) else ""

    ac_raw = pd.to_numeric(df[ac_col], errors="coerce").fillna(0.0).values
    dc_raw = pd.to_numeric(df[dc_col], errors="coerce").fillna(0.0).values
    ac_kw = _to_kw(ac_raw)
    dc_kw = _to_kw(dc_raw)

    # Determine DC nameplate (kW-DC)
    dc_nameplate_kw: float = float(args.dc_nameplate_kw)
    if dc_nameplate_kw <= 0 and nameplate_col:
        nameplate_raw = pd.to_numeric(df[nameplate_col], errors="coerce").fillna(0.0).values
        nameplate_kw = _to_kw(nameplate_raw)
        dc_nameplate_kw = float(np.nanmax(nameplate_kw)) if nameplate_kw.size else 0.0
    if dc_nameplate_kw <= 0:
        raise ValueError(
            "Could not determine DC nameplate kW. Provide --dc-nameplate-kw or ensure a usable "
            "nameplate column exists (e.g. nameplate_power)."
        )

    norm_kwac_per_kwdc = ac_kw / dc_nameplate_kw if dc_nameplate_kw > 0 else np.zeros_like(ac_kw)

    # Prefer `hour_index` if present; else sequential 1..N
    if "hour_index" in df.columns:
        hour = pd.to_numeric(df["hour_index"], errors="coerce").fillna(0).astype(int).values
        # Some exports can have 0/NaN padding; normalize to 1..N if needed
        if int(np.nanmin(hour)) <= 0 or len(np.unique(hour)) != len(hour):
            hour = np.arange(1, len(ac_kw) + 1)
    else:
        hour = np.arange(1, len(ac_kw) + 1)

    # Write AC file
    out_ac.parent.mkdir(parents=True, exist_ok=True)
    out_df_ac = pd.DataFrame({"Hour": hour, args.ac_column_name: ac_kw})
    out_df_ac.to_csv(out_ac, index=False)

    # Write DC file
    out_dc.parent.mkdir(parents=True, exist_ok=True)
    out_df_dc = pd.DataFrame({"Hour": hour, args.dc_column_name: dc_kw})
    out_df_dc.to_csv(out_dc, index=False)

    # Write normalized file (kW-AC per 1 kW-DC nameplate)
    out_norm.parent.mkdir(parents=True, exist_ok=True)
    out_df_norm = pd.DataFrame({"Hour": hour, args.norm_column_name: norm_kwac_per_kwdc})
    out_df_norm.to_csv(out_norm, index=False)

    # Also write a compatibility file name if it exists in workflows (AC by default)
    compat_path = out_ac.parent / "pv_profile_tem.csv"
    out_df_norm.to_csv(compat_path, index=False)

    print(f"Wrote {len(out_df_ac)} rows to {out_ac}")
    print(f"Wrote {len(out_df_dc)} rows to {out_dc}")
    print(f"Wrote {len(out_df_norm)} rows to {out_norm}")
    print(f"Wrote {len(out_df_norm)} rows to {compat_path} (normalized)")
    print(f"AC input power column: {ac_col} | peak (kW-AC): {float(np.max(ac_kw)):.2f}")
    print(f"DC input power column: {dc_col} | peak (kW-DC): {float(np.max(dc_kw)):.2f}")
    if nameplate_col:
        print(f"Nameplate column: {nameplate_col} | DC nameplate (kW-DC): {dc_nameplate_kw:.2f}")
    else:
        print(f"DC nameplate override (kW-DC): {dc_nameplate_kw:.2f}")
    print(f"Peak normalized output (kW-AC/kW-DC): {float(np.max(norm_kwac_per_kwdc)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

