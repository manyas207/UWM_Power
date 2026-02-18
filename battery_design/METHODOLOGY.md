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
1. Load campus total load profile (15-min or hourly).
2. Identify annual peak demand (typically July) in kW.
3. Compute target: `Peak_target = 0.75 × Peak_actual`
4. Peak shaving need: `P_shave = Peak_actual - Peak_target`
5. **Power rating:** Battery must discharge ≥ P_shave kW during peak hours.
6. **Energy:** Duration depends on peak shape. Typically 2–4 hours of discharge at P_shave.

**Output:** Minimum battery P_peak (kW) and E_peak (kWh) for peak shaving.

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

---

## Step 3: Solar Excess Absorption Sizing

**Goal:** Size battery to capture PV generation that exceeds grid export limits.

**Given:**
- **Bakke:** Installation 587.05 kW, grid limit 400 kVA → excess up to **187.05 kW**
- **Vet:** Installation 525 kW, grid limit 200 kVA → excess up to **325 kW**
- **Total potential excess:** ~512 kW (when both sites exceed limits simultaneously)

**Process:**
1. Use Helioscope PV simulation for each site (or combined) to get hourly generation.
2. At each hour: `Excess = max(0, PV_gen - Grid_limit)` per site.
3. Sum excess over the year to get total excess energy (kWh).
4. **Power rating:** Must charge at least max(excess) ≈ 512 kW when both sites peak.
5. **Energy:** Depends on duration of excess events. Run hourly simulation to find typical duration and total excess per day.

**Output:** P_solar (kW), E_solar (kWh) to absorb excess PV.

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

---

## Step 5: Validation

1. Run hourly simulation with combined load, PV, and battery.
2. Check: Peak demand ≤ 75% of original peak.
3. Check: 3-day outage scenario for critical loads.
4. Check: Solar excess curtailed = 0 (or minimal).

---

## File Structure

```
battery_design/
├── METHODOLOGY.md          # This document
├── step1_peak_shaving.py   # Campus peak & 25% target
├── step2_resiliency.py     # Critical buildings, 3-day energy
├── step3_solar_excess.py   # Bakke + Vet excess absorption
├── step4_combined_sizing.py # Combine & recommend
├── run_all.py              # Run full analysis
└── outputs/                # Generated results
```
