import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set working directory to script location and define output folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BATTERY_PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots", "battery")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BATTERY_PLOTS_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)
print("Working directory:", SCRIPT_DIR)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n" + "="*60)
print("Loading data...")
print("="*60)

# Load hourly load profile
load_df = pd.read_csv(os.path.join(RESULTS_DIR, "load_profile_tem.csv"))
print(f"Load data: {len(load_df)} hours")
print(f"Load range: {load_df['Load (kW)'].min():.1f} - {load_df['Load (kW)'].max():.1f} kW")

# Load solar PV simulation data
pv_df = pd.read_csv("simulation_17530461_hourly_data 2.csv")
print(f"PV data: {len(pv_df)} hours")

# Check which PV power column to use
pv_power_col = None
for col in ['ac_power', 'grid_power', 'optimal_dc_power', 'actual_dc_power', 'module_power']:
    if col in pv_df.columns:
        max_val = pv_df[col].max()
        if max_val > 0:
            pv_power_col = col
            print(f"Using PV column: {col} (max: {max_val:.1f} kW)")
            break

if pv_power_col is None:
    # If all power columns are zero, check nameplate and use irradiance-based estimate
    if 'nameplate_power' in pv_df.columns:
        nameplate = pv_df['nameplate_power'].max()
        print(f"Warning: PV power columns are zero. Nameplate: {nameplate:.1f} kW")
        # Use total_irradiance if available to estimate
        if 'total_irradiance' in pv_df.columns:
            pv_power_col = 'total_irradiance'
            print("Using total_irradiance as proxy (will need scaling)")
        else:
            print("ERROR: No usable PV power data found!")
            exit(1)
    else:
        print("ERROR: No PV power data found!")
        exit(1)

# Merge load and PV data by hour index
df = load_df.merge(
    pv_df[["hour_index", pv_power_col]], 
    left_on="Hour", 
    right_on="hour_index",
    how="inner"
)

df["PV_kW"] = df[pv_power_col].fillna(0)

# If PV is in irradiance units, scale to approximate power (rough estimate)
if pv_power_col == 'total_irradiance':
    # Rough conversion: assume 1 kW/m² irradiance ≈ 0.2 kW AC per kW nameplate
    # This is a placeholder - adjust based on your actual system
    if 'nameplate_power' in pv_df.columns:
        nameplate = pv_df['nameplate_power'].max()
        df["PV_kW"] = df["PV_kW"] * (nameplate / 1000) * 0.2  # rough scaling
    else:
        df["PV_kW"] = df["PV_kW"] * 0.2  # assume 200W per kW/m²

print(f"Merged data: {len(df)} hours")
print(f"PV range: {df['PV_kW'].min():.1f} - {df['PV_kW'].max():.1f} kW")
print(f"PV annual generation: {df['PV_kW'].sum():.0f} kWh")

# ============================================================================
# 2. COMPUTE NET LOAD (PV only, no battery)
# ============================================================================

df["NetLoad_no_batt_kW"] = df["Load (kW)"] - df["PV_kW"]
df["GridImport_no_batt_kW"] = df["NetLoad_no_batt_kW"].clip(lower=0)
df["PVExport_kW"] = (-df["NetLoad_no_batt_kW"]).clip(lower=0)

print("\n" + "="*60)
print("PV-only analysis:")
print("="*60)
print(f"Peak grid demand (with PV): {df['GridImport_no_batt_kW'].max():.1f} kW")
print(f"Annual grid energy: {df['GridImport_no_batt_kW'].sum():.0f} kWh")
print(f"PV self-consumption: {df['PV_kW'].sum() - df['PVExport_kW'].sum():.0f} kWh")
print(f"PV export/curtailment: {df['PVExport_kW'].sum():.0f} kWh")
print(f"PV offset percentage: {(df['PV_kW'].sum() - df['PVExport_kW'].sum()) / df['Load (kW)'].sum() * 100:.1f}%")

# ============================================================================
# 3. BATTERY SIMULATION FUNCTION
# ============================================================================

def simulate_battery(df, E_cap_kwh, P_cap_kw, eta_ch=0.95, eta_dis=0.95, soc_initial=0.5):
    """
    Simulate battery operation with self-consumption + peak shaving strategy.
    
    Parameters:
    - E_cap_kwh: Battery energy capacity (kWh)
    - P_cap_kw: Maximum charge/discharge power (kW)
    - eta_ch: Charging efficiency (0-1)
    - eta_dis: Discharging efficiency (0-1)
    - soc_initial: Initial state of charge (0-1)
    
    Returns:
    - DataFrame with battery power, SOC, and grid import columns added
    """
    soc = soc_initial * E_cap_kwh  # Start at initial SOC
    soc_list = []
    batt_power = []  # Positive = discharge, Negative = charge
    grid_import = []
    pv_curtailed = []
    
    for _, row in df.iterrows():
        net_load = row["Load (kW)"] - row["PV_kW"]  # Positive = need power
        
        p_batt = 0.0
        pv_curt = 0.0
        
        if net_load > 0:
            # Need power: discharge battery to reduce grid import
            # Available discharge = min(power limit, energy available)
            p_dis_max = min(P_cap_kw, soc * eta_dis)
            p_batt = min(net_load, p_dis_max)
            soc -= p_batt / eta_dis  # Discharge reduces SOC
        else:
            # Surplus PV: charge battery
            surplus = -net_load
            # Available charge = min(power limit, remaining capacity)
            p_ch_max = min(P_cap_kw, (E_cap_kwh - soc) / eta_ch)
            p_batt = -min(surplus, p_ch_max)  # Negative for charging
            soc -= p_batt * eta_ch  # Charge increases SOC (subtract negative)
            
            # Any remaining surplus after battery is full = curtailed
            pv_curt = max(0, surplus - abs(p_batt))
        
        # Ensure SOC stays within bounds
        soc = max(0, min(E_cap_kwh, soc))
        
        # Grid import after battery
        grid = net_load - p_batt
        grid_import.append(max(grid, 0.0))
        batt_power.append(p_batt)
        soc_list.append(soc)
        pv_curtailed.append(pv_curt)
    
    df_out = df.copy()
    df_out["BattPower_kW"] = batt_power
    df_out["SOC_kWh"] = soc_list
    df_out["GridImport_with_batt_kW"] = grid_import
    df_out["PV_Curtailed_kW"] = pv_curtailed
    
    return df_out

# ============================================================================
# 4. TEST MULTIPLE BATTERY SIZES
# ============================================================================

print("\n" + "="*60)
print("Running battery sizing simulations...")
print("="*60)

# Define battery size candidates
battery_configs = [
    {"E_cap_kwh": 0, "P_cap_kw": 0, "name": "No Battery"},
    {"E_cap_kwh": 250, "P_cap_kw": 125, "name": "250 kWh / 125 kW"},
    {"E_cap_kwh": 500, "P_cap_kw": 250, "name": "500 kWh / 250 kW"},
    {"E_cap_kwh": 1000, "P_cap_kw": 500, "name": "1000 kWh / 500 kW"},
    {"E_cap_kwh": 2000, "P_cap_kw": 1000, "name": "2000 kWh / 1000 kW"},
]

results = []

for config in battery_configs:
    print(f"\nTesting: {config['name']}")
    
    if config["E_cap_kwh"] == 0:
        # No battery case
        df_sim = df.copy()
        df_sim["BattPower_kW"] = 0
        df_sim["SOC_kWh"] = 0
        df_sim["GridImport_with_batt_kW"] = df_sim["GridImport_no_batt_kW"]
        df_sim["PV_Curtailed_kW"] = df_sim["PVExport_kW"]
    else:
        df_sim = simulate_battery(
            df, 
            E_cap_kwh=config["E_cap_kwh"],
            P_cap_kw=config["P_cap_kw"],
            eta_ch=0.95,
            eta_dis=0.95,
            soc_initial=0.5
        )
    
    # Calculate metrics
    peak_grid = df_sim["GridImport_with_batt_kW"].max()
    annual_grid_energy = df_sim["GridImport_with_batt_kW"].sum()
    battery_throughput = abs(df_sim["BattPower_kW"]).sum()
    pv_curtailed = df_sim["PV_Curtailed_kW"].sum()
    
    # Compare to baseline (no PV, no battery)
    baseline_peak = df["Load (kW)"].max()
    baseline_energy = df["Load (kW)"].sum()
    
    peak_reduction = baseline_peak - peak_grid
    energy_reduction = baseline_energy - annual_grid_energy
    
    results.append({
        "Config": config["name"],
        "E_cap_kWh": config["E_cap_kwh"],
        "P_cap_kW": config["P_cap_kw"],
        "Peak_Grid_kW": peak_grid,
        "Peak_Reduction_kW": peak_reduction,
        "Peak_Reduction_%": (peak_reduction / baseline_peak * 100),
        "Annual_Grid_kWh": annual_grid_energy,
        "Energy_Reduction_kWh": energy_reduction,
        "Energy_Reduction_%": (energy_reduction / baseline_energy * 100),
        "Battery_Throughput_kWh": battery_throughput,
        "PV_Curtailed_kWh": pv_curtailed,
        "df": df_sim  # Store full dataframe for plotting
    })
    
    print(f"  Peak grid demand: {peak_grid:.1f} kW (reduction: {peak_reduction:.1f} kW)")
    print(f"  Annual grid energy: {annual_grid_energy:.0f} kWh (reduction: {energy_reduction:.0f} kWh)")
    print(f"  Battery throughput: {battery_throughput:.0f} kWh")
    print(f"  PV curtailed: {pv_curtailed:.0f} kWh")

# ============================================================================
# 5. CREATE SUMMARY TABLE
# ============================================================================

results_df = pd.DataFrame(results)
results_df = results_df.drop(columns=['df'])  # Remove dataframe column for display

print("\n" + "="*60)
print("BATTERY SIZING SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

# Save summary to CSV
summary_path = os.path.join(RESULTS_DIR, "battery_sizing_results.csv")
results_df.to_csv(summary_path, index=False)
print(f"\nSummary saved to: {summary_path}")

# ============================================================================
# 6. CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("Generating plots...")
print("="*60)

# Plot 1: Net load comparison (PV only vs PV + Battery)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Select a representative week (e.g., week 20 = hours 336-503)
week_start = 336
week_end = 504
week_hours = df["Hour"].between(week_start, week_end)

# Top plot: PV only
ax1 = axes[0]
ax1.plot(df.loc[week_hours, "Hour"], 
         df.loc[week_hours, "GridImport_no_batt_kW"], 
         label="Grid Import (PV only)", linewidth=1.5, alpha=0.7)
ax1.plot(df.loc[week_hours, "Hour"], 
         df.loc[week_hours, "Load (kW)"], 
         label="Load", linewidth=1, alpha=0.5, linestyle="--")
ax1.set_xlabel("Hour of Year")
ax1.set_ylabel("Power (kW)")
ax1.set_title("Net Load Profile: PV Only (Sample Week)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: PV + Battery (best case)
best_config_idx = results_df[results_df["E_cap_kWh"] > 0]["Peak_Reduction_%"].idxmax()
best_config = results[best_config_idx]
best_df = best_config["df"]

ax2 = axes[1]
ax2.plot(best_df.loc[week_hours, "Hour"], 
         best_df.loc[week_hours, "GridImport_with_batt_kW"], 
         label=f"Grid Import ({best_config['Config']})", linewidth=1.5, color='green')
ax2.plot(df.loc[week_hours, "Hour"], 
         df.loc[week_hours, "Load (kW)"], 
         label="Load", linewidth=1, alpha=0.5, linestyle="--")
ax2.fill_between(best_df.loc[week_hours, "Hour"],
                 best_df.loc[week_hours, "GridImport_with_batt_kW"],
                 df.loc[week_hours, "GridImport_no_batt_kW"],
                 alpha=0.3, color='green', label="Battery Reduction")
ax2.set_xlabel("Hour of Year")
ax2.set_ylabel("Power (kW)")
ax2.set_title(f"Net Load Profile: PV + Battery ({best_config['Config']})")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BATTERY_PLOTS_DIR, "battery_net_load_comparison.png"), dpi=300)
print("Saved: battery_net_load_comparison.png")
plt.close()

# Plot 2: Battery sizing trade-offs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Peak reduction vs battery size
ax = axes[0, 0]
batt_configs = results_df[results_df["E_cap_kWh"] > 0]
ax.plot(batt_configs["E_cap_kWh"], batt_configs["Peak_Reduction_kW"], 
        marker='o', linewidth=2, markersize=8)
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("Peak Reduction (kW)")
ax.set_title("Peak Demand Reduction vs Battery Size")
ax.grid(True, alpha=0.3)

# Energy reduction vs battery size
ax = axes[0, 1]
ax.plot(batt_configs["E_cap_kWh"], batt_configs["Energy_Reduction_kWh"], 
        marker='s', linewidth=2, markersize=8, color='orange')
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("Annual Energy Reduction (kWh)")
ax.set_title("Annual Energy Reduction vs Battery Size")
ax.grid(True, alpha=0.3)

# Battery throughput vs size
ax = axes[1, 0]
ax.plot(batt_configs["E_cap_kWh"], batt_configs["Battery_Throughput_kWh"], 
        marker='^', linewidth=2, markersize=8, color='green')
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("Battery Throughput (kWh/year)")
ax.set_title("Battery Utilization vs Size")
ax.grid(True, alpha=0.3)

# PV curtailment vs battery size
ax = axes[1, 1]
ax.plot(batt_configs["E_cap_kWh"], batt_configs["PV_Curtailed_kWh"], 
        marker='d', linewidth=2, markersize=8, color='red')
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("PV Curtailed (kWh/year)")
ax.set_title("PV Curtailment vs Battery Size")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BATTERY_PLOTS_DIR, "battery_sizing_tradeoffs.png"), dpi=300)
print("Saved: battery_sizing_tradeoffs.png")
plt.close()

# Plot 3: Battery SOC and power for best configuration
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

week_hours_idx = df["Hour"].between(week_start, week_end)

# Load and PV
ax = axes[0]
ax.plot(df.loc[week_hours_idx, "Hour"], df.loc[week_hours_idx, "Load (kW)"], 
        label="Load", linewidth=1.5)
ax.plot(df.loc[week_hours_idx, "Hour"], df.loc[week_hours_idx, "PV_kW"], 
        label="PV Generation", linewidth=1.5, color='orange')
ax.set_ylabel("Power (kW)")
ax.set_title(f"Load and PV Generation ({best_config['Config']})")
ax.legend()
ax.grid(True, alpha=0.3)

# Battery power
ax = axes[1]
ax.plot(best_df.loc[week_hours_idx, "Hour"], 
        best_df.loc[week_hours_idx, "BattPower_kW"], 
        linewidth=1.5, color='purple')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax.set_ylabel("Battery Power (kW)")
ax.set_title("Battery Charge/Discharge (positive = discharge)")
ax.grid(True, alpha=0.3)

# Battery SOC
ax = axes[2]
ax.plot(best_df.loc[week_hours_idx, "Hour"], 
        best_df.loc[week_hours_idx, "SOC_kWh"], 
        linewidth=1.5, color='green')
ax.set_xlabel("Hour of Year")
ax.set_ylabel("State of Charge (kWh)")
ax.set_title("Battery State of Charge")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BATTERY_PLOTS_DIR, "battery_operation_detail.png"), dpi=300)
print("Saved: battery_operation_detail.png")
plt.close()

# Plot 4: Load duration curves
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by descending order for load duration curve
baseline_sorted = np.sort(df["Load (kW)"].values)[::-1]
pv_only_sorted = np.sort(df["GridImport_no_batt_kW"].values)[::-1]
batt_sorted = np.sort(best_df["GridImport_with_batt_kW"].values)[::-1]

hours = np.arange(1, len(baseline_sorted) + 1)

ax.plot(hours, baseline_sorted, label="Baseline (No PV, No Battery)", 
        linewidth=2, alpha=0.7)
ax.plot(hours, pv_only_sorted, label="PV Only", linewidth=2, alpha=0.7)
ax.plot(hours, batt_sorted, label=f"PV + Battery ({best_config['Config']})", 
        linewidth=2, alpha=0.7)

ax.set_xlabel("Hours per Year")
ax.set_ylabel("Power (kW)")
ax.set_title("Load Duration Curves")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1000)  # Focus on top 1000 hours

plt.tight_layout()
plt.savefig(os.path.join(BATTERY_PLOTS_DIR, "load_duration_curves.png"), dpi=300)
print("Saved: load_duration_curves.png")
plt.close()

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
print("\nGenerated files:")
print(f"  - {summary_path}")
print(f"  - {os.path.join(BATTERY_PLOTS_DIR, 'battery_net_load_comparison.png')}")
print(f"  - {os.path.join(BATTERY_PLOTS_DIR, 'battery_sizing_tradeoffs.png')}")
print(f"  - {os.path.join(BATTERY_PLOTS_DIR, 'battery_operation_detail.png')}")
print(f"  - {os.path.join(BATTERY_PLOTS_DIR, 'load_duration_curves.png')}")
