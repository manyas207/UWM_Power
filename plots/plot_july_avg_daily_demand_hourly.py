from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_kw_strings_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].replace(" kW", "", regex=True).astype(float)
    return out


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_csv = project_root / "data" / "raw" / "UWM_Power.csv"
    out_png = project_root / "plots" / "july_avg_daily_demand_hourly.png"

    df = pd.read_csv(load_csv)
    # UWM_Power.csv timestamps look like "1/1/23 0:00"
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%y %H:%M")
    load_cols = [c for c in df.columns if c != "Time"]
    df = _parse_kw_strings_to_float(df, load_cols)

    # Campus total demand (kW) at 15-minute resolution
    df["campus_kw"] = df[load_cols].sum(axis=1)

    # Filter July
    july = df[df["Time"].dt.month == 7].copy()
    if july.empty:
        raise ValueError("No July rows found in UWM_Power.csv after parsing timestamps.")

    # Convert to hourly mean per day, then average across all July days
    july["date"] = july["Time"].dt.date
    july["hour"] = july["Time"].dt.hour
    hourly_by_day = july.groupby(["date", "hour"], as_index=False)["campus_kw"].mean()
    avg_hourly = hourly_by_day.groupby("hour", as_index=False)["campus_kw"].mean()

    # Plot
    import matplotlib.pyplot as plt  # local import to avoid hard dependency until run

    plt.figure(figsize=(10, 5))
    plt.plot(avg_hourly["hour"], avg_hourly["campus_kw"], marker="o", linewidth=2)
    plt.xticks(np.arange(0, 24, 1))
    plt.grid(True, alpha=0.3)
    plt.title("Average Daily Campus Demand in July (Hourly)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand (kW)")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

