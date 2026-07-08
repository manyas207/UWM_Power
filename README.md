# UWM Campus Power + Battery Sizing

Portfolio project focused on campus-scale energy analytics and battery storage sizing for UW-Madison load + PV profiles.  
The repository combines data processing, exploratory time-series analysis, and a multi-constraint battery design workflow (peak shaving, resiliency, and PV excess absorption).

## Project Highlights

- Builds hourly campus load profiles from 15-minute building-level demand data.
- Integrates solar generation simulation outputs to model net load behavior.
- Evaluates battery sizing tradeoffs across power (kW) and energy (kWh) constraints.
- Produces reproducible CSV outputs and figures suitable for technical reports.
- Includes a challenge-oriented methodology in `battery_design/`.

## Repository Structure

```text
UWM_Power/
├── data/
│   └── raw/                    # source datasets (.csv, .xlsx)
├── docs/
│   └── references/             # background PDFs and supporting documents
├── scripts/                    # analysis and preprocessing scripts
├── battery_design/             # step-by-step sizing methodology
├── results/                    # generated tabular outputs
└── plots/                      # generated visual outputs
```

## Tech Stack

- **Language**: Python
- **Core libraries**: `pandas`, `numpy`, `matplotlib`
- **Supporting tools**: `pathlib`, `argparse` for CLIs, `datetime`/`pandas` time handling

## Results Snapshot

- **Peak demand reduction**: Battery scenarios achieve up to ~25% reduction in modeled campus peak grid import relative to the baseline July peak (see `battery_design/PEAK_SHAVE_VALIDATION_REPORT.md`).
- **Energy usage impact**: Battery configurations in `scripts/battery_sizing_analysis.py` reduce annual grid energy consumption while increasing PV self-consumption and reducing curtailment.
- **Resiliency sizing**: `battery_design/step2_resiliency.py` computes a multi-day outage requirement for critical buildings and folds it into the combined recommended configuration in `battery_design/outputs/step4_combined_results.csv`.

## Quick Start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the primary pipeline:

```bash
# 1) Build hourly load profile from raw building data
python scripts/process.py

# 2) Generate time-series demand plots
python scripts/time_series_model.py

# 3) Run sizing analysis against load + PV data
python scripts/battery_sizing_analysis.py
```

Run the challenge-style battery design workflow:

```bash
python battery_design/run_all.py
```

## Key Outputs

- `results/load_profile_tem.csv` - hourly campus demand profile.
- `results/battery_sizing_results.csv` - battery sizing tradeoff summary.
- `battery_design/outputs/step4_combined_results.csv` - recommended sizing from constraints.
- `plots/` - generated figures for demand patterns and battery impact.

## Why This Project Matters

This project demonstrates practical skills relevant to energy systems and data roles:

- data cleaning and transformation of high-resolution time-series data,
- modeling tradeoffs under real engineering constraints,
- translating analysis into decision-ready outputs.