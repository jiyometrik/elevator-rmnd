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
START_TIME = datetime(2023, 1, 1, 0, 0, 0)
NUM_ENTRIES_PER_LIFT = 20_000
HOURS_PER_STEP = 12
OUTPUT_FILENAME = "predictive_maintenance_lifts"
MAX_STEPS_BEF_MAINTENANCE = 180  # 180 steps * 12 hours = 90 days (3 months)
GENERATE_CSV = False
GENERATE_PKL = True

# Define 3 lift models with different starting ages
LIFTS = [
    # NOTE the `id` field must be determined during actual preprocessing
    {"id": 1, "model": "Otis Gen2", "age_start": 15000},
    {"id": 2, "model": "Schindler 3300", "age_start": 32000},
    {"id": 3, "model": "KONE MonoSpace", "age_start": 8500},
]

# "Healthy" starting metrics
BASELINES = {
    "ARM_DIST": 0.5,  # mm
    "DOOR_DIST": 2.0,  # mm
    "FLOOR_DIST": 0.0,  # mm (perfectly flush)
    "ROPE_MFL": 10.0,  # mV
    "BEARING_TEMP": 35.0,  # Celsius
}

# "Breakdown" metrics
CRITICALS = {
    "ARM_DIST": 1.5,
    "DOOR_DIST": 8.0,
    "FLOOR_DIST": 15.0,
    "ROPE_MFL": 50.0,
    "BEARING_TEMP": 80.0,
}

# Base degradation rates per 12-hour step (scaled to hit critical around 200--300 steps)
BASE_RATES = {
    "ARM_DIST": (CRITICALS["ARM_DIST"] - BASELINES["ARM_DIST"]) / 280,
    "DOOR_DIST": (CRITICALS["DOOR_DIST"] - BASELINES["DOOR_DIST"]) / 200,
    "FLOOR_DIST": (CRITICALS["FLOOR_DIST"] - BASELINES["FLOOR_DIST"]) / 300,
    # Ropes naturally degrade slower
    "ROPE_MFL": (CRITICALS["ROPE_MFL"] - BASELINES["ROPE_MFL"]) / 1000,
    "BEARING_TEMP": (CRITICALS["BEARING_TEMP"] - BASELINES["BEARING_TEMP"]) / 240,
}


def compute_rul(record: pd.Series, avail_maintenances: pd.DataFrame) -> float:
    """
    Compute the remaining useful life (RUL) in h for a given record.
    NOTE If real-world data is used, this method must be done during preprocessing.
    """
    time_now = record["timestamp"]
    # Get the next available maintenance (should be greater than the current time)
    next_available_maintenance = avail_maintenances[
        avail_maintenances["timestamp"] >= time_now
    ]["timestamp"].min()
    # If no further maintenance exists, then RUL is np.inf
    if pd.isna(next_available_maintenance):
        return np.inf  # np.infs will be filtered out during training
    # If there exists a further maintenance, compute the RUL in hours
    time_now = pd.to_datetime(time_now).to_pydatetime()
    next_available_maintenance = pd.to_datetime(
        next_available_maintenance
    ).to_pydatetime()
    return (next_available_maintenance - time_now).total_seconds() / 3600


class LiftDataGenerator:
    """
    Manages state and generates data entries for a single lift.
    """

    def __init__(self, lift: dict, start_time: datetime):
        self.lift = lift
        self.age = lift["age_start"]
        self.cur_time = start_time
        self.time_since_maintenance = 0
        self.baselines = BASELINES.copy()
        # Randomise degradation rates for the current maintenance cycle
        self.cycle_rates = {
            k: v * np.random.uniform(0.5, 2.5) for k, v in BASE_RATES.items()
        }

    def generate_entry(self) -> dict:
        """Generate a single entry and update internal state."""
        # Calculate realistic sensor readings (base + degradation rate + noise)
        sensor_rows = {
            "ARM_DIST": self.baselines["ARM_DIST"] + np.random.normal(0, 0.02),
            "DOOR_DIST": self.baselines["DOOR_DIST"] + np.random.normal(0, 0.1),
            "FLOOR_DIST": self.baselines["FLOOR_DIST"] + np.random.normal(0, 0.2),
            "ROPE_MFL": self.baselines["ROPE_MFL"] + np.random.normal(0, 0.5),
            "BEARING_TEMP": self.baselines["BEARING_TEMP"] + np.random.normal(0, 1.0),
        }

        # Ensure minimum absolute physics rules apply
        sensor_rows["ARM_DIST"] = max(0.1, sensor_rows["ARM_DIST"])
        sensor_rows["DOOR_DIST"] = max(0.1, sensor_rows["DOOR_DIST"])
        sensor_rows["FLOOR_DIST"] = abs(sensor_rows["FLOOR_DIST"])
        sensor_rows["ROPE_MFL"] = max(1.0, sensor_rows["ROPE_MFL"])
        sensor_rows["BEARING_TEMP"] = max(20.0, sensor_rows["BEARING_TEMP"])

        # Check whether the lift is due for maintenance
        breakdown_triggered = any(sensor_rows[k] >= v for k, v in CRITICALS.items())
        scheduled_maintenance = self.time_since_maintenance >= MAX_STEPS_BEF_MAINTENANCE
        maintenance_done = int(breakdown_triggered or scheduled_maintenance)

        # Create row entry
        row = {
            "timestamp": self.cur_time.strftime("%Y-%m-%d %H:%M:%S"),
            "lift_id": self.lift["id"],
            "lift_model": self.lift["model"],
            "lift_age_hours": self.age,
            "ARM_DIST_mm": round(sensor_rows["ARM_DIST"], 3),
            "DOOR_DIST_mm": round(sensor_rows["DOOR_DIST"], 3),
            "FLOOR_DIST_mm": round(sensor_rows["FLOOR_DIST"], 3),
            "ROPE_MFL_mV": round(sensor_rows["ROPE_MFL"], 3),
            "BEARING_TEMP_C": round(sensor_rows["BEARING_TEMP"], 3),
            "maintenance_done": maintenance_done,
        }

        # Update states for the next step
        if maintenance_done:
            self.baselines = BASELINES.copy()
            self.cycle_rates = {
                k: v * np.random.uniform(0.5, 2.5) for k, v in BASE_RATES.items()
            }
            self.time_since_maintenance = 0
        else:
            for k in self.baselines:
                self.baselines[k] += self.cycle_rates[k]
            self.time_since_maintenance += 1

        # Increment time and age
        self.age += HOURS_PER_STEP
        self.cur_time += timedelta(hours=HOURS_PER_STEP)

        return row


def generate_all_entries(
    lift_models: list[dict] = LIFTS,
    start_time: datetime = START_TIME,
    n_entries_per_lift: int = NUM_ENTRIES_PER_LIFT,
) -> pd.DataFrame:
    """Generate all dataset entries for all lifts and return as a DataFrame."""
    rows = []
    for lift in lift_models:
        generator = LiftDataGenerator(lift, start_time)
        for _ in range(n_entries_per_lift):
            rows.append(generator.generate_entry())
    return pd.DataFrame(rows)


def append_ruls(lift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by computing RUL for each entry.
    NOTE If real-world data is used, this method must be done during preprocessing.
    """
    lift_models = lift_df["lift_model"].unique()
    lift_df = lift_df.copy()
    maintenance_events = lift_df[lift_df["maintenance_done"] == 1]
    for lift_model in lift_models:
        # Get a list of all maintenance events for the current lift model
        available_maintenance_events = maintenance_events[
            maintenance_events["lift_model"] == lift_model
        ]
        # Get all log events for the current lift model
        log_events = lift_df[lift_df["lift_model"] == lift_model]
        # Then compute the RUL for each log event for each lift model
        for idx, entry in log_events.iterrows():
            rul = compute_rul(entry, available_maintenance_events)
            lift_df.at[idx, "RUL_hrs"] = rul

    return lift_df


if __name__ == "__main__":
    # Generate all dataset entries
    df = generate_all_entries()
    # Compute RULs
    df = append_ruls(df)
    # Export as CSV and as pickle
    if GENERATE_CSV:
        df.to_csv(OUTPUT_FILENAME + ".csv", index=False)
    if GENERATE_PKL:
        df.to_pickle(OUTPUT_FILENAME + ".pkl")
