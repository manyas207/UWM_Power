"""
Reformat Helioscope hourly simulation to "tem" CSV format.

Creates a 2-column file shaped like `results/load_profile_tem.csv`:

  Hour,<kW column>

Default output is `results/pv_profile_tem.csv` with column name "Load (kW)"
so it can be ingested by the same simple loader pattern (Hour + kW column).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PREFERRED_POWER_COLS = ("ac_power", "grid_power", "module_power")


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
        default=os.path.join(Path(__file__).resolve().parents[1], "simulation_17530461_hourly_data 2.csv"),
        help="Path to Helioscope simulation CSV.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(Path(__file__).resolve().parent, "pv_profile_tem.csv"),
        help="Output CSV path (2 columns).",
    )
    parser.add_argument(
        "--kW-column-name",
        default="Load (kW)",
        help='Name for the kW column in the output (default matches load_profile_tem.csv).',
    )
    parser.add_argument(
        "--power-column",
        default="",
        help="Force a specific input power column (otherwise auto-picks).",
    )

    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    power_col = args.power_column.strip() or _pick_power_column(df, PREFERRED_POWER_COLS)

    power_raw = pd.to_numeric(df[power_col], errors="coerce").fillna(0.0).values
    power_kw = _to_kw(power_raw)

    # Prefer `hour_index` if present; else sequential 1..N
    if "hour_index" in df.columns:
        hour = pd.to_numeric(df["hour_index"], errors="coerce").fillna(0).astype(int).values
        # Some exports can have 0/NaN padding; normalize to 1..N if needed
        if int(np.nanmin(hour)) <= 0 or len(np.unique(hour)) != len(hour):
            hour = np.arange(1, len(power_kw) + 1)
    else:
        hour = np.arange(1, len(power_kw) + 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"Hour": hour, args.kW_column_name: power_kw})
    out_df.to_csv(out, index=False)

    print(f"Wrote {len(out_df)} rows to {out}")
    print(f"Input power column: {power_col}")
    print(f"Peak power (kW): {float(np.max(power_kw)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

