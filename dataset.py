"""
dataset.py
Generate a fake lift dataset for testing future lift models
"""

import datetime
import os
from typing import Literal

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DATE = datetime.datetime(2026, month=6, day=1)  # can be changed
END_DATE = datetime.datetime(2026, month=7, day=1)

np.random.seed(43)

WEIGHTS = {
    "arm_dist": 0.3,
    "door_dist": 0.1,
    "floor_dist": 0.2,
    "rope_mfl": 0.3,
    "bearing_temp": 0.1,
}
AMBIENT_TEMP = 28
THRESHOLDS = {
    "arm_dist": 0.65,
    "door_dist": 6,
    "floor_dist": 8,
    "rope_mfl": 3,
    "bearing_temp": 30 + AMBIENT_TEMP,
    "dsi": 1,
}


def export_dataset(df: pd.DataFrame, fname: str = "lift_data.csv") -> None:
    """
    Exports the lift dataset created to a CSV file, given a specified filepath
    """
    df.to_csv(os.path.join(os.path.dirname(__file__), fname))


def get_dsi(
    entry: dict, weights: dict = WEIGHTS, thresholds: dict = THRESHOLDS
) -> float:
    """
    Aggregates the DSI (Degradation Severity Index) for a given log entry,
    using the specified weights and thresholds for each component.
    """
    return sum(
        entry[component] / thresholds[component] * weight
        for component, weight in weights.items()
    )


def generate_entry(
    timestamp: datetime.datetime,
    start_date: datetime.datetime = START_DATE,
    end_date: datetime.datetime = END_DATE,
) -> dict:
    """
    Generate a single log entry for the lift dataset, given a timestamp.
    The log entry contains the following components:
    - `timestamp': the date and time of the log entry
    - `arm_dist': the distance between the arm and the shaft in mm
    - `door_dist': the door gap in mm
    - `floor_dist': the distance between the cab floor and landing in mm
    - `rope_mfl': the magnetic flux leakage of the rope in mT
    - `bearing_temp': the temperature of the bearing in deg C
    - `dsi`: the degradation severity index, given by the `get_dsi` function
    """
    time_elapsed = (timestamp - start_date).total_seconds() / (
        end_date - start_date
    ).total_seconds()
    arm_dist = min(max(0, np.random.normal(loc=0.5 * time_elapsed, scale=0.1)), 1.0)
    door_dist = min(max(0, np.random.normal(loc=3.9 * time_elapsed, scale=0.5)), 7.8)
    floor_dist = min(max(0, np.random.normal(loc=4.45 * time_elapsed, scale=0.5)), 8.9)
    rope_mfl = min(max(0, np.random.normal(loc=25 * time_elapsed, scale=5)), 50)
    bearing_temp = min(
        max(32, np.random.normal(loc=62 * time_elapsed + 32, scale=5)), 75
    )

    entry = {
        "timestamp": timestamp,
        "arm_dist": arm_dist,
        "door_dist": door_dist,
        "floor_dist": floor_dist,
        "rope_mfl": rope_mfl,
        "bearing_temp": bearing_temp,
    }
    entry["dsi"] = get_dsi(entry)
    assert all(
        component in entry for component in WEIGHTS.keys()
    ), "Missing components in log entry"
    return entry


def generate_dataset(
    log_frequency: Literal["hourly", "daily"] = "hourly",
    start_date: datetime.datetime = START_DATE,
    end_date: datetime.datetime = END_DATE,
) -> pd.DataFrame:
    """
    Generates a dataset according to
    `Proposal for Passenger Lift Remote Maintenance and Diagnostics (RM&D) Systems'
    with the following columns:
    - `timestamp': the date and time of the log entry
    - `arm_dist': the distance between the arm and the shaft in mm
    - `door_dist': the door gap in mm
    - `floor_dist': the distance between the cab floor and landing in mm
    - `rope_mfl': the magnetic flux leakage of the rope in mT
    - `bearing_temp': the temperature of the bearing in deg C
    """
    # Generate timestamps based on the specified log frequency
    timedelta = None
    match log_frequency:
        case "hourly":
            timedelta = datetime.timedelta(hours=1)
        case "daily":
            timedelta = datetime.timedelta(days=1)
        case _:
            raise ValueError("invalid `log_frequency`. Must be 'hourly' or 'daily'.")
    entries = []
    for timestamp in pd.date_range(start_date, end_date, freq=timedelta):
        entry = generate_entry(
            timestamp=timestamp, start_date=start_date, end_date=end_date
        )
        entries.append(entry)
    df = pd.DataFrame(entries)
    return df


if __name__ == "__main__":
    logs = generate_dataset()
    print(logs.head(20))
    export_dataset(logs, fname="liftdata_v2.csv")
    # export_dataset(lifts)
