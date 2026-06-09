"""
dataset.py
Generate a synthetic dataset for predictive maintenance of elevator lifts.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set random seed for consistent generation
np.random.seed(42)

# Configuration
NUM_ENTRIES_PER_LIFT = 2000
HOURS_PER_STEP = 12
FNAME = "predictive_maintenance_lifts.csv"
MAX_STEPS_BEF_MAINTENANCE = 180  # 180 steps * 12 hours = 90 days (3 months)

# Define 3 lift models with different starting ages
lifts = [
    {"id": "LIFT_001", "model": "Otis Gen2", "age_start": 15000},
    {"id": "LIFT_002", "model": "Schindler 3300", "age_start": 32000},
    {"id": "LIFT_003", "model": "KONE MonoSpace", "age_start": 8500},
]

# "Healthy" starting metrics
baselines = {
    "ARM_DIST": 0.5,  # mm
    "DOOR_DIST": 2.0,  # mm
    "FLOOR_DIST": 0.0,  # mm (perfectly flush)
    "ROPE_MFL": 10.0,  # mV
    "BEARING_TEMP": 35.0,  # Celsius
}

# "Breakdown" metrics
criticals = {
    "ARM_DIST": 1.5,
    "DOOR_DIST": 8.0,
    "FLOOR_DIST": 15.0,
    "ROPE_MFL": 50.0,
    "BEARING_TEMP": 80.0,
}

# Base degradation rates per 12-hour step (scaled to hit critical around 100-150 steps)
base_rates = {
    "ARM_DIST": (criticals["ARM_DIST"] - baselines["ARM_DIST"]) / 140,
    "DOOR_DIST": (criticals["DOOR_DIST"] - baselines["DOOR_DIST"]) / 100,
    "FLOOR_DIST": (criticals["FLOOR_DIST"] - baselines["FLOOR_DIST"]) / 150,
    "ROPE_MFL": (criticals["ROPE_MFL"] - baselines["ROPE_MFL"])
    / 500,  # Rope degrades slower naturally
    "BEARING_TEMP": (criticals["BEARING_TEMP"] - baselines["BEARING_TEMP"]) / 120,
}

rows = []
start_time = datetime(2023, 1, 1, 0, 0, 0)

for lift in lifts:
    age = lift["age_start"]
    curr_time = start_time
    time_since_maintenance = 0

    # Pure base values (without sensor noise)
    current_base = baselines.copy()

    # Randomise degradation rates for the current maintenance cycle (simulates independent root-causes)
    cycle_rates = {k: v * np.random.uniform(0.5, 2.5) for k, v in base_rates.items()}

    for _ in range(NUM_ENTRIES_PER_LIFT):
        # Calculate realistic sensor readings (base + degradation rate + noise)
        row_sensors = {
            "ARM_DIST": current_base["ARM_DIST"] + np.random.normal(0, 0.02),
            "DOOR_DIST": current_base["DOOR_DIST"] + np.random.normal(0, 0.1),
            "FLOOR_DIST": current_base["FLOOR_DIST"] + np.random.normal(0, 0.2),
            "ROPE_MFL": current_base["ROPE_MFL"] + np.random.normal(0, 0.5),
            "BEARING_TEMP": current_base["BEARING_TEMP"] + np.random.normal(0, 1.0),
        }

        # Ensure minimum absolute physics rules apply (e.g. distance from flush is absolute)
        row_sensors["ARM_DIST"] = max(0.1, row_sensors["ARM_DIST"])
        row_sensors["DOOR_DIST"] = max(0.1, row_sensors["DOOR_DIST"])
        row_sensors["FLOOR_DIST"] = abs(row_sensors["FLOOR_DIST"])
        row_sensors["ROPE_MFL"] = max(1.0, row_sensors["ROPE_MFL"])
        row_sensors["BEARING_TEMP"] = max(20.0, row_sensors["BEARING_TEMP"])

        # Check whether the lift is due for maintenance
        breakdown_triggered = any(row_sensors[k] >= v for k, v in criticals.items())
        scheduled_maintenance = time_since_maintenance >= MAX_STEPS_BEF_MAINTENANCE
        maintenance_done = int(breakdown_triggered or scheduled_maintenance)

        # Write row to dataset
        rows.append(
            {
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "lift_id": lift["id"],
                "lift_model": lift["model"],
                "lift_age_hours": age,
                "ARM_DIST_mm": round(row_sensors["ARM_DIST"], 3),
                "DOOR_DIST_mm": round(row_sensors["DOOR_DIST"], 3),
                "FLOOR_DIST_mm": round(row_sensors["FLOOR_DIST"], 3),
                "ROPE_MFL_mV": round(row_sensors["ROPE_MFL"], 3),
                "BEARING_TEMP_C": round(row_sensors["BEARING_TEMP"], 3),
                "maintenance_done": maintenance_done,
            }
        )

        # Update states for the next step (12 hours later)
        if maintenance_done:
            # Reset values to healthy baselines
            current_base = baselines.copy()
            # Reroll degradation rates so the next cycle fails differently
            cycle_rates = {
                k: v * np.random.uniform(0.5, 2.5) for k, v in base_rates.items()
            }
            time_since_maintenance = 0
        else:
            # Degrade components further
            for k in current_base:
                current_base[k] += cycle_rates[k]
            time_since_maintenance += 1

        # Increment time
        age += HOURS_PER_STEP
        curr_time += timedelta(hours=HOURS_PER_STEP)

# Convert to pd.DataFrame
df = pd.DataFrame(rows)
df.to_csv(FNAME, index=False)
