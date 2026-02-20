# Battery Energy Storage Design Methodology for UW Madison

This document explains the step-by-step approach to sizing a battery system that meets all challenge requirements.

---

## Requirements Summary

| Requirement | Target | Duration |
|-------------|--------|----------|
| **Peak load reduction** | 25% reduction in campus peak demand | Annual peak (July) |
| **Resiliency** | Critical loads maintained during grid outage | 3 days |
| **Solar excess absorption** | Capture Bakke (187 kW) + Vet (325 kW) excess | When PV > grid limit |
| **Critical buildings** | Dorms + Carson Gulley + DeJope dining | — |

---

## Step 1: Peak Load Reduction Sizing

**Goal:** Size battery power (kW) and energy (kWh) to shave 25% off the campus peak.

**Process:**
1. Load campus total load profile at **15-minute resolution** (`UWM_Power.csv`).
2. Load PV generation profile (Helioscope) and apply **interconnect export limits per site**:
   - Bakke export cap: 400 kVA
   - Vet export cap: 200 kVA
3. Compute **grid import without battery** at each timestep:
   - `grid_import_no_batt = max(0, load - pv_within_limit)`
4. Identify annual peak grid import (with PV, no battery) in kW:
   - `Peak_actual = max(grid_import_no_batt)`
5. Compute target peak:
   - `Peak_target = 0.75 × Peak_actual`
6. Simulate a battery with SOC dynamics (15-min timesteps) and a peak-shaving controller:
   - **Discharge only enough** to cap grid import at `Peak_target`
   - **Charge** using PV surplus (including PV above interconnect limits, which otherwise would be curtailed)
   - Enforce power/energy limits, efficiency, and minimum SOC reserve for resiliency
7. Sweep candidate (P,E) sizes and select the **minimum** that achieves:
   - `peak_grid_import_with_batt ≤ Peak_target`

**Output:** Minimum battery P_peak (kW) and E_peak (kWh) for peak shaving.

**Current results (from `battery_design/outputs/step1_peak_shaving_results.csv`):**
- **Peak_actual (grid import, PV, no batt):** 853.27 kW
- **Peak_target (75%):** 639.95 kW
- **Peak shaving needed:** 213.32 kW
- **Peak-shaving energy proxy (3h):** 639.95 kWh (proxy, not the SOC simulation capacity)

---

## Step 2: Resiliency Sizing

**Goal:** Size battery to supply 25% of critical buildings' average daily load for 3 days.

**Critical buildings (from challenge):**
- **Dorms:** DeJope, Goodnight, Phillips, Bradley, Jones, Sullivan, Cole, Chamberlin, Kronshage, Leopold, Mack House, Jorns Halls, Turner, Gilman, Humphrey, Adams, Tripp, Slichter
- **Dining:** Carson Gulley Center, DeJope (dining)

**Process:**
1. Extract load for each critical building from UWM_Power.csv.
2. Sum to get aggregated critical load time series.
3. Compute **average daily load** = mean hourly load × 24 (kWh/day).
4. Resiliency load = 25% × average daily load (kWh/day).
5. Energy for 3 days: `E_resiliency = 3 × (0.25 × avg_daily_load_kWh)`
6. **Power:** Assume critical load is roughly constant; use peak of (25% × hourly critical load) as P_resiliency.

**Output:** E_resiliency (kWh), P_resiliency (kW).

**Current results (from `battery_design/outputs/step2_resiliency_results.csv`):**
- **Energy reserve for 3 days @ 25% avg critical load:** 368.30 kWh
- **Power requirement (peak 25% critical load):** 17.08 kW

---

## Step 3: Solar Excess Absorption Sizing

**Goal:** Size battery to capture PV generation that exceeds grid export limits.

**Given:**
- **Bakke:** Installation 587.05 kW, grid limit 400 kVA → excess up to **187.05 kW**
- **Vet:** Installation 525 kW, grid limit 200 kVA → excess up to **325 kW**
- **Total potential excess:** ~512 kW (when both sites exceed limits simultaneously)

**Process:**
1. Use Helioscope PV simulation (hourly) and scale to Bakke and Vet.
2. Apply **per-site interconnect export limits** each timestep:
   - `excess_bakke = max(0, bakke_gen - 400)`
   - `excess_vet   = max(0, vet_gen   - 200)`
   - `excess_total = excess_bakke + excess_vet`
3. In the temporal model, this `excess_total` is treated as **charge-only PV**:
   - It **cannot** be exported to the grid
   - If the battery cannot accept it (power/energy constrained), it is **curtailed**
4. **Power requirement:** To eliminate curtailment at the worst moment, the battery must be able to **charge** at least:
   - `P_charge ≥ max(excess_total)` (up to ~512 kW when both sites peak)
5. **Energy requirement:** Depends on how long excess persists; the annual excess energy sets the scale of the curtailment problem, but eliminating curtailment typically requires enough kWh to avoid filling during sustained events (plus a discharge strategy to make room later).

**Output:** P_solar (kW), E_solar (kWh) to absorb excess PV.

**Current results (from `battery_design/outputs/step3_solar_excess_results.csv`):**
- **Peak excess power (Bakke+Vet above interconnect caps):** 512.05 kW
- **Annual excess energy above caps (curtailed if no battery):** 330,234.91 kWh
- **Theoretical site maxima:** Bakke 187.05 kW, Vet 325.00 kW

---

## Step 4: Combined Battery Sizing

**Goal:** Single or multiple batteries that meet all three use cases.

**Options:**
- **Shared system:** One battery serves peak shaving + resiliency + solar absorption.
- **Distributed:** Separate batteries for different functions.

**Sizing logic (shared):**
- **Power (kW):** `P_battery = max(P_peak, P_resiliency, P_solar)`
- **Energy (kWh):** Depends on dispatch strategy. Conservative: `E_battery = max(E_peak, E_resiliency, E_solar)`. If use cases don't overlap in time, may use smaller capacity with shared storage.

**Dispatch priorities (during normal operation):**
1. Solar excess → charge when PV > grid limit.
2. Peak shaving → discharge during peak hours (July).
3. Resiliency → reserve energy for outages (may set a minimum SOC).

**Current combined result (from `battery_design/outputs/step4_combined_results.csv`):**
- **Minimum viable to meet 25% peak reduction (with SOC reserve):** **300 kW / 1000 kWh**
- **Achieved peak grid import:** 639.95 kW

**Note on solar curtailment vs peak shaving:**
- The **300 kW / 1000 kWh** battery is sized to meet the **peak reduction target** with a simple peak-shaving controller.
- Fully eliminating PV curtailment driven by Bakke+Vet interconnect caps would generally require **charge power closer to 512 kW** (and enough kWh + dispatch to avoid saturating during long excess periods).

---

## Step 5: Validation

1. Run **15-minute** simulation with combined load, PV (with per-site interconnect limits), and battery SOC dynamics.
2. Check: Peak demand ≤ 75% of original peak.
3. Check: 3-day outage scenario for critical loads.
4. Check: Solar excess curtailed = 0 (or minimal).

---

## File Structure

```
battery_design/
├── METHODOLOGY.md          # This document
├── temporal_simulation.py  # Load+PV temporal sizing (primary)
├── step2_resiliency.py     # Critical buildings, 3-day energy
├── step1_peak_shaving.py   # Standalone peak sizing (legacy)
├── step3_solar_excess.py   # Standalone solar excess (legacy)
├── step4_combined_sizing.py # Combine & recommend (legacy)
├── run_all.py              # Runs step2 + temporal_simulation
└── outputs/                # Generated results
```
