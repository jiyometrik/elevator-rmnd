"""
dataset.py
Generate a fake lift dataset for testing future lift models
"""

import os
import random

import numpy as np
import pandas as pd
from netron import start

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = 2 * 100
N_STORIES = 4

np.random.seed(67)


def generate_dataset(
    n_samples: int = N_SAMPLES, n_stories: int = N_STORIES
) -> pd.DataFrame:
    """
    Generates the fake lift dataset and returns a pandas.DataFrame, with the following columns:
    - `start_storey`: an integer representing the storey at which the lift starts its ascent/descent
    - `end_storey`: an integer representing the storey at which the lift starts its ascent/descent
    - `microswitch_state`: a Boolean representing the state of brake armature microswitches
    - `armature_dist`: a float representing the distance between the armature and brake pad
    - `brakecoil_current`: a float representing the current in the brake coil
    - `door_dist`: a float representing the distance between the cab doors
    - `door_dist_dot`: a float representing the rate of change of the distance between the cab doors
    - `floor_dist`: a float representing the distance between the cab floor and the landing floor
    - `rope_tension`: a float representing the tension in the steel rope
    - `rope_flux`: a float representing the magnetic flux leak from the steel rope
    - `bearing_temp`: a float representing the ambient temperature surrounding the ball bearings
    """
    # Generate `start_storey` and `end_storey`, but make sure `start_storey[i]` != `end_storey[i]` ∀i
    start_stories = np.ones(n_samples) + np.random.randint(n_stories, size=n_samples)
    end_stories = np.mod(
        np.ones(n_samples) + np.abs(start_stories + np.random.choice([-2, -1, 1, 2])),
        n_stories,
    )

    assert all(i != 0 for i in start_stories - end_stories)

    # Generate `microswitch_state`: 1 = "open", 0 = "closed"
    microswitch_states = np.random.randint(2, size=n_samples)
    # Generate `armature_dist`, in mm, scaled to [0, 10]
    armature_dists = 10 * np.random.chisquare(1, size=n_samples)
    armature_dists /= armature_dists.max()
    # Generate `brakecoil_current`, in A, scaled to [0.5, 5]
    brakecoil_currents = np.random.rayleigh(2, size=n_samples)
    # Generate `door_dist` in mm, scaled to [0, 10]
    door_dists = np.random.chisquare(3, size=n_samples) ** 1.1
    door_dists = np.sort(door_dists)
    door_dist_dots = np.gradient(door_dists)

    # Generate `floor_dist` in mm, scaled to [0, 10]
    floor_dist = np.random.chisquare(3, size=n_samples) ** 1.03

    # Generate `rope_tension` in kN, scaled to [5, 21]
    rope_tensions = 5 * np.random.wald(3, 2.5, size=n_samples)

    # Generate `rope_flux` in mT, scaled to [0, 20]
    rope_fluxes = np.random.pareto(3, size=n_samples)

    # Generate `bearing_temp` in deg C, scaled to [30, 70]
    bearing_temps = (
        10 * np.random.noncentral_chisquare(1, 2.5, size=n_samples) ** 1 + 25
    )

    # Generate empty list for urgency levels
    urgency_levels = np.zeros_like(n_samples)

    df = pd.DataFrame(
        {
            "start_storey": start_stories.astype(int),
            "end_storey": end_stories.astype(int),
            "microswitch_state": microswitch_states.astype(bool),
            "armature_dist": armature_dists,
            "brakecoil_current": brakecoil_currents,
            "door_dist": door_dists,
            "door_dist_dot": door_dist_dots,
            "floor_dist": floor_dist,
            "rope_tension": rope_tensions,
            "rope_flux": rope_fluxes,
            "bearing_temp": bearing_temps,
            "urgency": urgency_levels,
        }
    )
    return df


def export_dataset(df: pd.DataFrame, fname: str = "lift_data.csv") -> None:
    """
    Exports the lift dataset created to a CSV file, given a specified filepath
    """
    df.to_csv(os.path.join(os.path.dirname(__file__), fname))


if __name__ == "__main__":
    lifts = generate_dataset()
    export_dataset(lifts)
