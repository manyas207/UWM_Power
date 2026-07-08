# Peak-Shave Validation Summary Report

## Overview

This work checks whether **candidate battery systems** meet the UW–Madison goal of **reducing peak grid demand by 25%** relative to the **July** peak, using the project’s **15-minute** campus load series and a **PV + battery** simulation. Solar offsets load first; the battery discharges so that grid import stays at or below a fixed **threshold** derived from July (75% of July’s peak grid import without the battery).

**What was validated**

- **Required:** After the battery is modeled, the **maximum grid import in calendar July** must not exceed **75% of July’s pre-battery peak** (the “7.5 MW” line if July peaked at “10 MW”).
- **Optional:** Whether any **other month** ever exceeds that same threshold (desirable for demand costs, but not required by the sponsor wording).

**How it was done**

A script [`validate_july_peak_shave.py`](validate_july_peak_shave.py) loads [`data/raw/UWM_Power.csv`](../data/raw/UWM_Power.csv), aligns Bakke/Vet PV from [`data/raw/simulation_17530461_hourly_data 2.csv`](../data/raw/simulation_17530461_hourly_data%202.csv), computes the July baseline and threshold, then runs [`simulate_battery`](temporal_simulation.py) from [`temporal_simulation.py`](temporal_simulation.py) for a **full year** so state-of-charge before and after July is realistic. Outputs are written to [`outputs/july_peak_validation.csv`](outputs/july_peak_validation.csv).

**Headline outcome (latest run, see Results below)**

- Both tested configurations (**329 kW / 2,025 kWh** and **531 kW / 3,968 kWh**) **pass** the required July check and the optional year-wide check on the current dataset.
- On this file, the **single worst 15-minute grid-import interval** occurs in **June**, not July; the rule still uses **July** for the baseline peak, per the written requirement. That distinction is noted in the script output and in the Results section.

---

## Results

Values below come from the last run of `validate_july_peak_shave.py`, saved in [`outputs/july_peak_validation.csv`](outputs/july_peak_validation.csv).

### Baseline and threshold (same for all battery rows)

| Quantity | Value | Notes |
|----------|------:|--------|
| July peak grid import (PV, **no** battery) | **726.64 kW** | Max over all **July** 15-minute intervals |
| **Threshold (75% of July peak)** | **544.98 kW** | Target “new peak” from the grid; demand above this is met by the battery in the model |
| Resiliency SOC floor (simulation) | **368.30 kWh** | From `step2_resiliency_results.csv`; limits usable discharge |

**Data note:** In this file, the **calendar maximum** grid import without battery (**853.27 kW**) occurs in **June**, not July. The validation **still** uses the **July** peak (**726.64 kW**) to set the threshold, matching the sponsor statement that the campus annual peak is in July.

### Outcomes by battery configuration

| Configuration | Power (kW) | Energy (kWh) | Peak July **with** battery (kW) | Peak **year** with battery (kW) | Required (July ≤ threshold) | Optional (no month above threshold) |
|---------------|-------------:|---------------:|----------------------------------:|----------------------------------:|:----------------------------:|:-------------------------------------:|
| REopt_battery1 | 329 | 2,025 | 544.98 | 544.98 | **Pass** | **Pass** |
| REopt_unlimited | 531 | 3,968 | 544.98 | 544.98 | **Pass** | **Pass** |

**Interpretation**

- Both configurations achieve a **July** maximum grid import equal to the **threshold** (545 kW when rounded), i.e. the modeled **25% reduction from the July baseline peak** (726.64 → 544.98 kW).
- **Year peak** matches the July threshold in this run because the controller applies the **same** cap every timestep; the battery shaves **all** intervals that would otherwise exceed the cap, not only July.
- The two battery sizes **tie** on these metrics here: neither is the limiting factor; the **import cap** binds first. For other tariffs or controls, larger or smaller packs could diverge.

---

## 1. Requirement (as modeled)

The sponsor language was interpreted as follows for the **validation script**:

| Concept | Model implementation |
|--------|----------------------|
| Campus annual peak occurs in **July** | Baseline peak for the **25% rule** is taken from **calendar July only**: maximum **grid import** (after PV reduces load, **no battery**). |
| **25% reduction** in peak demand from the grid | **Threshold** = **75% × July baseline peak** (e.g. 10 MW → 7.5 MW). |
| Demand above the threshold | Covered by the **battery** (and PV already applied first). |
| Staying under the threshold in **other** months | **Optimal**, **not required** — reported separately. |

Grid import without battery uses the same definition as the rest of `battery_design`: load minus PV that counts toward serving load within interconnect limits (Bakke / Vet).

---

## 2. What was built

### New / primary artifact

- **[`validate_july_peak_shave.py`](validate_july_peak_shave.py)** — End-to-end check that:
  1. Loads [`data/raw/UWM_Power.csv`](../data/raw/UWM_Power.csv) at **15-minute** resolution (all building columns summed to campus kW).
  2. Loads and aligns PV from [`data/raw/simulation_17530461_hourly_data 2.csv`](../data/raw/simulation_17530461_hourly_data%202.csv), scaled to Bakke and Vet sizes, with **per-site grid limits** (400 kVA Bakke, 200 kVA Vet).
  3. Computes **July** baseline peak: `max(load - pv_within_limit)` over July only.
  4. Sets `threshold_kw = 0.75 * july_peak_no_batt`.
  5. Runs **[`simulate_battery`](temporal_simulation.py)** from [`temporal_simulation.py`](temporal_simulation.py) with `grid_import_cap_kw = threshold_kw` for a **full year** (preserves SOC dynamics before/after July).
  6. **Required pass:** maximum grid import **in July** with battery ≤ `threshold_kw`.
  7. **Optional:** whether **any** timestep in **non-July** months exceeds `threshold_kw` (listed by month if so).

### Output

- **[`outputs/july_peak_validation.csv`](outputs/july_peak_validation.csv)** — One row per tested battery configuration with baseline peaks, threshold, achieved peaks, and pass/fail flags.

### Dependency (unchanged logic, reused)

- **[`temporal_simulation.py`](temporal_simulation.py)** — Provides `simulate_battery`, PV/load alignment, and Bakke/Vet constants. No change was *required* for this report; the validator **imports** it.

---

## 3. Explanation of the code

### 3.1 Data preparation (`validate_july_peak_shave.py`)

1. **Load** — Reads `data/raw/UWM_Power.csv`, strips `" kW"` from numeric columns, sums all building columns into a single **campus load** array (kW at each 15-minute step).

2. **Time** — Parses the `Time` column (`%m/%d/%y %H:%M`) to build a boolean mask **`july`** for calendar July.

3. **PV** — Calls `load_pv_bakke_vet` (hourly profile), then `_align_series_to_length` to match the load length (typically 35,040 steps for a full year at 15 minutes).

4. **Interconnect split** — For each timestep:
   - `pv_within_kw`: PV that can offset grid import subject to Bakke/Vet **export limits** (modeled as caps on “within-limit” production).
   - `pv_charge_only_kw`: PV in excess of those limits (can charge the battery or be curtailed; see `simulate_battery`).

5. **Baseline grid import (no battery)** —  
   `grid_no_batt = max(0, load_kw - pv_within_kw)`  
   July baseline peak = `max(grid_no_batt[july])`.

6. **Threshold** — `threshold_kw = 0.75 * july_peak_no_batt`.

7. **Resiliency floor** — Minimum SOC reserve (kWh) is read from [`outputs/step2_resiliency_results.csv`](outputs/step2_resiliency_results.csv) so the battery does not discharge below the modeled reserve.

### 3.2 Battery simulation (`simulate_battery` in `temporal_simulation.py`)

The simulator steps **forward in time** with a fixed timestep (here **0.25 h** for 15 minutes). Each step:

1. **PV serves load first** — Reduces remaining load before the grid or battery.

2. **Discharge for peak shaving** — If `grid_import_cap_kw` is set, the model computes how much discharge is needed so that **grid import** does not exceed the cap (subject to **power limit**, **SOC** above `soc_min_kwh`, and discharge efficiency).

3. **Charge** — Uses PV surplus (including charge-only PV) up to **power** and **energy headroom**, with charge efficiency.

4. **Export / curtail** — Remaining within-limit PV may export; excess that cannot be stored may be curtailed.

**Important:** Passing `grid_import_cap_kw=threshold_kw` applies that **same** cap at **every** timestep for the whole year. That enforces “never import from the grid above the July-derived threshold,” which also tends to shave other months. The **sponsor text** only *requires* July; the script still *reports* whether other months ever exceed the threshold for transparency.

### 3.3 Battery configurations tested

Two REopt-style sizes are compared (names are labels only):

| Label | Power (kW) | Energy (kWh) |
|-------|------------|--------------|
| REopt_battery1 | 329 | 2,025 |
| REopt_unlimited | 531 | 3,968 |

### 3.4 Calendar peak vs. July (data caveat)

If the **single worst 15-minute interval** of `grid_no_batt` falls **outside** July (e.g. in June in the current file), the script **still** uses **July** for the baseline peak, per the written requirement, and prints a **NOTE**. This avoids silently switching the rule to “calendar annual max” when the sponsor text says the **annual** peak is in July.

---

## 4. How to run

From the project root:

```bash
cd battery_design && python validate_july_peak_shave.py
```

Ensure `data/raw/UWM_Power.csv`, `data/raw/simulation_17530461_hourly_data 2.csv`, and `outputs/step2_resiliency_results.csv` exist as expected by [`temporal_simulation.py`](temporal_simulation.py). Re-running updates [`outputs/july_peak_validation.csv`](outputs/july_peak_validation.csv); refresh the **Results** section if numbers change.

---

## 5. Closing summary

- **Goal:** Verify that candidate batteries can meet **25% July peak reduction** on **15-minute** data, with optional reporting for the rest of the year.
- **Mechanism:** July baseline peak → 75% threshold → `simulate_battery` with `grid_import_cap_kw` set to that threshold; **required** check on July max grid import with battery.
- **Code:** [`validate_july_peak_shave.py`](validate_july_peak_shave.py) orchestrates data; [`temporal_simulation.py`](temporal_simulation.py) implements physics and dispatch rules.
- **Results:** Stored in [`outputs/july_peak_validation.csv`](outputs/july_peak_validation.csv); summarized in the **Results** section above.
