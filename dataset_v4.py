"""
Predictive Maintenance Lift / Elevator Simulator
=================================================
Generates a multi-year, hourly time-series dataset for a fleet of lifts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM & SENSOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_FILENAME: str = "liftdata_v4"

SUBSYSTEM_DEFS: Dict[str, dict] = {
    "arm": {
        "sensors": ["arm_dist_mm"],
        "wear_rate_per_trip": 3.0e-6,
        "maint_restore": 0.015,
        "cascade": [],
    },
    "door": {
        "sensors": ["door_dist_mm"],
        "wear_rate_per_trip": 1.2e-5,  # doors see the most cycling
        "maint_restore": 0.020,
        "cascade": [],
    },
    "drive": {
        "sensors": ["vibration_ms2", "motor_temp_c"],
        "wear_rate_per_trip": 8.0e-6,
        "maint_restore": 0.018,
        "cascade": ["safety"],  # drive stress → ropes/brakes
    },
    "bearing": {
        "sensors": ["bearing_temp_c"],
        "wear_rate_per_trip": 6.0e-6,
        "maint_restore": 0.015,
        "cascade": ["drive"],
    },
    "safety": {
        "sensors": ["brake_wear_pct", "rope_degradation_mv"],
        "wear_rate_per_trip": 5.0e-6,
        "maint_restore": 0.014,
        "cascade": [],
    },
    "leveling": {
        "sensors": ["leveling_accuracy_mm"],
        "wear_rate_per_trip": 7.0e-6,
        "maint_restore": 0.016,
        "cascade": [],
    },
    "hydraulic": {
        "sensors": ["oil_pressure_bar"],
        "wear_rate_per_trip": 5.5e-6,
        "maint_restore": 0.013,
        "cascade": ["drive"],
    },
}

SENSOR_NAMES: List[str] = [
    "arm_dist_mm",
    "door_dist_mm",
    "leveling_accuracy_mm",
    "vibration_ms2",
    "motor_temp_c",
    "bearing_temp_c",
    "brake_wear_pct",
    "rope_degradation_mv",
    "oil_pressure_bar",
]

# Build lookup structures once at import time
_SUB_NAMES: List[str] = list(SUBSYSTEM_DEFS.keys())
_N_SUB: int = len(_SUB_NAMES)
_SUB_IDX: Dict[str, int] = {name: i for i, name in enumerate(_SUB_NAMES)}
_SENSOR_TO_SUB: Dict[str, str] = {}
for _sn, _si in SUBSYSTEM_DEFS.items():
    for _s in _si["sensors"]:
        _SENSOR_TO_SUB[_s] = _sn
_SENSOR_SUB_IDX: List[int] = [_SUB_IDX[_SENSOR_TO_SUB[s]] for s in SENSOR_NAMES]

# Pre-compute cascade indices per subsystem
_CASCADE_IDX: List[List[int]] = [
    [_SUB_IDX[c] for c in SUBSYSTEM_DEFS[sn]["cascade"]] for sn in _SUB_NAMES
]


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SimConfig:
    """Immutable simulation parameters.  Override via CLI or constructor."""

    simulation_years: int = 5
    start_time: datetime = datetime(2024, 1, 1)
    seed: int = 42

    # Random (non-wear) breakdown rate  (Poisson, independent of wear)
    baseline_breakdown_rate_per_year: float = 0.3

    # Maintenance
    scheduled_maint_days: int = 90
    breakdown_restore_frac: float = 0.85  # fraction removed after breakdown

    # Pre-failure degradation ramp
    ramp_threshold: float = 0.65  # wear level that triggers ramp
    ramp_hours: int = 400
    ramp_max_multiplier: float = 6.0

    # Cascade damage on breakdown
    cascade_damage: float = 0.30

    # Sensor noise (σ)
    sensor_noise: Dict[str, float] = field(
        default_factory=lambda: {
            "arm_dist_mm": 0.02,
            "door_dist_mm": 0.08,
            "leveling_accuracy_mm": 0.20,
            "vibration_ms2": 0.05,
            "motor_temp_c": 1.2,
            "bearing_temp_c": 1.0,
            "brake_wear_pct": 0.8,
            "rope_degradation_mv": 0.5,
            "oil_pressure_bar": 0.25,
        }
    )

    # Sensor baselines (healthy) and critical (failure) values
    baselines: Dict[str, float] = field(
        default_factory=lambda: {
            "arm_dist_mm": 0.5,
            "door_dist_mm": 2.0,
            "leveling_accuracy_mm": 0.0,
            "vibration_ms2": 0.30,
            "motor_temp_c": 35.0,
            "bearing_temp_c": 35.0,
            "brake_wear_pct": 0.0,
            "rope_degradation_mv": 10.0,
            "oil_pressure_bar": 12.0,
        }
    )
    criticals: Dict[str, float] = field(
        default_factory=lambda: {
            "arm_dist_mm": 1.5,
            "door_dist_mm": 9.0,
            "leveling_accuracy_mm": 18.0,
            "vibration_ms2": 3.2,
            "motor_temp_c": 100.0,
            "bearing_temp_c": 92.0,
            "brake_wear_pct": 98.0,
            "rope_degradation_mv": 58.0,
            "oil_pressure_bar": 3.5,
        }
    )

    # Seasonal ambient-temperature amplitude (°C)
    seasonal_amplitude_c: float = 10.0

    @property
    def total_hours(self) -> int:
        return self.simulation_years * 365 * 24


# ═══════════════════════════════════════════════════════════════════════════
# TRAFFIC PROFILE  (7 weekdays × 24 hours)
# ═══════════════════════════════════════════════════════════════════════════


def _build_traffic_profile() -> np.ndarray:
    weekday = np.array(
        [
            2,
            1,
            1,
            1,
            1,
            2,
            3,
            12,
            12,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            12,
            12,
            5,
            4,
            3,
            3,
            2,
        ],
        dtype=np.float64,
    )
    weekend = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            3,
            4,
            6,
            7,
            7,
            7,
            7,
            6,
            5,
            5,
            5,
            4,
            3,
            3,
            2,
            2,
            1,
        ],
        dtype=np.float64,
    )
    profile = np.zeros((7, 24), dtype=np.float64)
    for d in range(5):
        profile[d] = weekday
    for d in range(5, 7):
        profile[d] = weekend
    return profile


_TRAFFIC_PROFILE = _build_traffic_profile()


# ═══════════════════════════════════════════════════════════════════════════
# LIFT SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════


class LiftSimulator:
    """Simulate one lift over the full timeline.

    The inner loop is intentionally kept in pure Python (no per-step NumPy
    calls on small arrays) for minimal overhead.  Sensor-value mapping and
    noise addition are vectorised in a post-processing pass.
    """

    def __init__(
        self,
        lift_id: str,
        model: str,
        initial_age_years: float,
        cfg: SimConfig,
        rng: np.random.Generator,
    ) -> None:
        self.lift_id = lift_id
        self.model = model
        self.initial_age_hours = initial_age_years * 8760.0
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.total_hours

        # ── Pre-compute time arrays ──────────────────────────────────────
        hours = np.arange(self.n, dtype=np.int64)
        self.timestamps = np.datetime64(cfg.start_time, "s") + (hours * 3600).astype(
            "timedelta64[s]"
        )
        dti = pd.DatetimeIndex(self.timestamps)
        self.hour_of_day = dti.hour.values.astype(np.int8)
        self.day_of_week = dti.dayofweek.values.astype(np.int8)
        self.day_of_year = dti.dayofyear.values.astype(np.int16)

        # ── Traffic ──────────────────────────────────────────────────────
        base = _TRAFFIC_PROFILE[self.day_of_week, self.hour_of_day]
        noise = rng.normal(0, 0.20, size=self.n) * base
        self.trips = np.clip(np.round(base + noise), 0, None).astype(np.int32)

        # ── Seasonal temperature offset ──────────────────────────────────
        phase = 2.0 * np.pi * (self.day_of_year.astype(np.float64) - 15) / 365.0
        self.ambient_offset = (cfg.seasonal_amplitude_c * np.sin(phase)).astype(
            np.float32
        )

        # ── Per-lift wear-rate scatter (±30 %) ───────────────────────────
        base_rates = np.array(
            [SUBSYSTEM_DEFS[s]["wear_rate_per_trip"] for s in _SUB_NAMES],
            dtype=np.float64,
        )
        self.wear_rates = base_rates * rng.uniform(0.7, 1.3, size=_N_SUB)

        # ── Per-subsystem additive maintenance restore ───────────────────
        self.maint_restore = np.array(
            [SUBSYSTEM_DEFS[s]["maint_restore"] for s in _SUB_NAMES],
            dtype=np.float64,
        )

    # ──────────────────────────────────────────────────────────────────────
    def simulate(self) -> pd.DataFrame:
        """Run the simulation and return a DataFrame."""
        cfg = self.cfg
        rng = self.rng
        n = self.n
        n_sub = _N_SUB

        # ── Pre-draw all random numbers we need ──────────────────────────
        # This avoids per-step RNG calls which have Python overhead.
        jitter = rng.uniform(0.6, 1.4, size=(n, n_sub)).astype(np.float64)
        obstruction_draws = rng.exponential(1.0, size=n)
        random_bd_draws = rng.exponential(1.0, size=64)  # enough for all breakdowns
        random_bd_sub_draws = rng.integers(0, n_sub, size=64)

        # ── Output arrays ────────────────────────────────────────────────
        out_wear = np.empty((n, n_sub), dtype=np.float32)
        out_severity = np.empty(n, dtype=np.int8)
        out_breakdown = np.zeros(n, dtype=np.bool_)
        out_maint = np.zeros(n, dtype=np.bool_)
        out_maint_type = np.zeros(n, dtype=np.int8)  # 0=none, 1=scheduled, 2=breakdown
        out_breakdown_count = np.empty(n, dtype=np.int32)
        out_hours_since_maint = np.empty(n, dtype=np.float32)
        out_trip_cumul = np.empty(n, dtype=np.int64)
        out_door_cycles = np.empty(n, dtype=np.int64)
        out_runtime = np.empty(n, dtype=np.float32)
        out_door_obstructions = np.empty(n, dtype=np.int64)
        out_primary_sub = np.zeros(n, dtype=np.int8)  # 0 = none, 1..6 = subsystem

        # ── Mutable state ────────────────────────────────────────────────
        wear = [0.0] * n_sub  # plain list → faster indexing
        age_h = self.initial_age_hours
        cumul_trips = 0
        cumul_obstr = 0
        cumul_runtime = 0.0
        bd_count = 0
        last_maint_h = age_h

        # Schedule first random breakdown
        rate_inv = (
            8760.0 / cfg.baseline_breakdown_rate_per_year
            if cfg.baseline_breakdown_rate_per_year > 0
            else 1e18
        )
        bd_draw_idx = 0
        random_bd_h = age_h + random_bd_draws[bd_draw_idx] * rate_inv
        bd_draw_idx += 1

        # Ramp state
        ramp_active = [False] * n_sub
        ramp_start = [0.0] * n_sub

        # Local copies for inner loop speed
        wear_rates = self.wear_rates  # ndarray, indexed by int
        maint_rest = self.maint_restore
        trips_arr = self.trips
        ramp_thr = cfg.ramp_threshold
        ramp_hrs = float(cfg.ramp_hours)
        ramp_mul = cfg.ramp_max_multiplier
        maint_int_h = float(cfg.scheduled_maint_days * 24)
        bd_restore = cfg.breakdown_restore_frac
        casc_dmg = cfg.cascade_damage
        cascade_idx = _CASCADE_IDX

        for t in range(n):
            trips_t = int(trips_arr[t])

            # ── 1. Accumulate wear ───────────────────────────────────────
            for si in range(n_sub):
                rate = wear_rates[si] * trips_t * jitter[t, si]
                if ramp_active[si]:
                    frac = (age_h - ramp_start[si]) / ramp_hrs
                    if frac > 1.0:
                        frac = 1.0
                    rate *= 1.0 + (ramp_mul - 1.0) * frac * frac
                w = wear[si] + rate
                if w > 1.0:
                    w = 1.0
                wear[si] = w

            # ── 2. Activate ramps ────────────────────────────────────────
            for si in range(n_sub):
                if not ramp_active[si] and wear[si] >= ramp_thr:
                    ramp_active[si] = True
                    ramp_start[si] = age_h

            # ── 3. Breakdown check ───────────────────────────────────────
            bd_this = False
            primary = 0  # 0 = none

            # a) Wear-driven
            for si in range(n_sub):
                if wear[si] >= 1.0:
                    bd_this = True
                    primary = si + 1
                    break

            # b) Random (Poisson)
            if not bd_this and age_h >= random_bd_h:
                bd_this = True
                primary = int(random_bd_sub_draws[min(bd_draw_idx, 63)]) + 1

            if bd_this:
                pi = primary - 1
                wear[pi] = 1.0

                # Cascade
                for ci in cascade_idx[pi]:
                    w = wear[ci] + casc_dmg
                    wear[ci] = w if w < 1.0 else 1.0

                bd_count += 1

                # Breakdown repair (fractional)
                keep = 1.0 - bd_restore
                for si in range(n_sub):
                    wear[si] *= keep
                    ramp_active[si] = False

                last_maint_h = age_h

                # Schedule next random breakdown
                if bd_draw_idx < 64:
                    random_bd_h = age_h + random_bd_draws[bd_draw_idx] * rate_inv
                    bd_draw_idx += 1
                else:
                    # Extremely unlikely to need >64 breakdowns; extend
                    random_bd_h = age_h + rng.exponential(rate_inv)

            # ── 4. Scheduled maintenance ─────────────────────────────────
            hours_since = age_h - last_maint_h
            scheduled = False
            if hours_since >= maint_int_h and not bd_this:
                scheduled = True
                deact_thr = ramp_thr - 0.10
                for si in range(n_sub):
                    w = wear[si] - maint_rest[si]
                    if w < 0.0:
                        w = 0.0
                    wear[si] = w
                    if ramp_active[si] and w < deact_thr:
                        ramp_active[si] = False
                last_maint_h = age_h

            # ── 5. Severity ──────────────────────────────────────────────
            max_w = 0.0
            for si in range(n_sub):
                if wear[si] > max_w:
                    max_w = wear[si]

            if bd_this:
                sev = 3
            elif max_w >= 0.95:
                sev = 2
            elif max_w >= 0.80:
                sev = 1
            else:
                sev = 0

            # ── 6. Counters ─────────────────────────────────────────────
            cumul_trips += trips_t
            cumul_runtime += trips_t * 0.025
            cumul_obstr += int(trips_t * 0.002 * obstruction_draws[t])

            # ── 7. Store ─────────────────────────────────────────────────
            for si in range(n_sub):
                out_wear[t, si] = wear[si]
            out_severity[t] = sev
            out_breakdown[t] = bd_this
            out_maint[t] = bd_this or scheduled
            out_maint_type[t] = 2 if bd_this else (1 if scheduled else 0)
            out_breakdown_count[t] = bd_count
            out_hours_since_maint[t] = hours_since
            out_trip_cumul[t] = cumul_trips
            out_door_cycles[t] = cumul_trips  # 1 door cycle per trip
            out_runtime[t] = cumul_runtime
            out_door_obstructions[t] = cumul_obstr
            out_primary_sub[t] = primary

            age_h += 1.0

        # ═════════════════════════════════════════════════════════════════
        # POST-PROCESSING: wear → sensor readings (vectorised)
        # ═════════════════════════════════════════════════════════════════
        baselines_arr = np.array(
            [cfg.baselines[s] for s in SENSOR_NAMES], dtype=np.float64
        )
        ranges_arr = np.array(
            [cfg.criticals[s] - cfg.baselines[s] for s in SENSOR_NAMES],
            dtype=np.float64,
        )
        sensor_sub = np.array(_SENSOR_SUB_IDX, dtype=np.int32)

        sensor_wear = out_wear[:, sensor_sub]  # (n, 8)
        sensor_readings = baselines_arr + sensor_wear * ranges_arr  # (n, 8)

        # Seasonal offset on temperature sensors
        for i, s in enumerate(SENSOR_NAMES):
            if s in ("motor_temp_c", "bearing_temp_c"):
                sensor_readings[:, i] += self.ambient_offset

        # Physical clamps
        _CLAMP = {
            "arm_dist_mm": (0.1, 3.0),
            "oil_pressure_bar": (3.0, 12.5),
            "brake_wear_pct": (0.0, 100.0),
            "door_dist_mm": (0.5, 15.0),
            "vibration_ms2": (0.05, 10.0),
            "leveling_accuracy_mm": (0.0, 25.0),
            "motor_temp_c": (20.0, 120.0),
            "bearing_temp_c": (20.0, 110.0),
            "rope_degradation_mv": (5.0, 65.0),
        }
        for i, s in enumerate(SENSOR_NAMES):
            if s in _CLAMP:
                lo, hi = _CLAMP[s]
                np.clip(sensor_readings[:, i], lo, hi, out=sensor_readings[:, i])

        # Add sensor noise
        noise_std = np.array(
            [cfg.sensor_noise.get(s, 0.0) for s in SENSOR_NAMES], dtype=np.float64
        )
        noisy = (
            sensor_readings + rng.normal(0, 1, size=(n, len(SENSOR_NAMES))) * noise_std
        )
        noisy = np.round(noisy, 3)

        for i, s in enumerate(SENSOR_NAMES):
            if s in _CLAMP:
                lo, hi = _CLAMP[s]
                np.clip(noisy[:, i], lo, hi, out=noisy[:, i])

        # ── Build DataFrame ──────────────────────────────────────────────
        # Map maintenance type ints → strings
        maint_type_map = np.array(["", "scheduled", "breakdown"])
        maint_type_str = maint_type_map[out_maint_type]

        # Map primary subsystem ints → names
        sub_name_map = np.array([""] + _SUB_NAMES)  # index 0 = no failure
        primary_str = sub_name_map[out_primary_sub]

        data: dict = {
            "timestamp": pd.DatetimeIndex(self.timestamps),
            "lift_id": self.lift_id,
            "lift_model": self.model,
            "lift_age_days": np.round(
                (self.initial_age_hours + np.arange(n, dtype=np.float64)) / 24.0, 1
            ),
            "trip_count": out_trip_cumul,
            "door_cycles": out_door_cycles,
            "runtime_hours": np.round(out_runtime, 1),
            "door_obstructions": out_door_obstructions,
            "trips_last_hour": self.trips,
        }
        for i, s in enumerate(SENSOR_NAMES):
            data[s] = noisy[:, i]

        for si, sn in enumerate(_SUB_NAMES):
            data[f"wear_{sn}"] = out_wear[:, si]

        data.update(
            {
                "failure_severity": out_severity,
                "breakdown_occurred": out_breakdown,
                "primary_failure_subsystem": primary_str,
                "maintenance_done": out_maint,
                "maintenance_type": maint_type_str,
                "hours_since_maintenance": out_hours_since_maint,
                "breakdown_count": out_breakdown_count,
            }
        )

        self._breakdown_count = bd_count
        return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════
# FLEET RUNNER
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_FLEET: List[Tuple[str, str, float]] = [
    ("LIFT_001", "Otis Gen2", 1.2),
    ("LIFT_002", "Schindler 3300", 2.5),
    ("LIFT_003", "KONE MonoSpace", 0.8),
    ("LIFT_004", "Otis Gen2", 4.0),
    ("LIFT_005", "Schindler 3300", 0.0),
]


def run_simulation(
    cfg: SimConfig | None = None,
    fleet: List[Tuple[str, str, float]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full fleet simulation and return the combined DataFrame."""
    if cfg is None:
        cfg = SimConfig()
    if fleet is None:
        fleet = DEFAULT_FLEET

    master_rng = np.random.default_rng(cfg.seed)

    if verbose:
        exp_random = (
            cfg.baseline_breakdown_rate_per_year * cfg.simulation_years * len(fleet)
        )
        print("=" * 70)
        print("  Lift Predictive-Maintenance Simulator")
        print(
            f"  {len(fleet)} lifts × {cfg.simulation_years} years = "
            f"{cfg.total_hours * len(fleet):,} hourly records"
        )
        print(
            f"  Random breakdown rate : {cfg.baseline_breakdown_rate_per_year}/lift/yr "
            f"(≈{exp_random:.0f} random events)"
        )
        print(f"  + wear-driven breakdowns (subsystem wear → 100 %)")
        print(f"  Maintenance every {cfg.scheduled_maint_days} days (additive restore)")
        print("=" * 70)

    frames: List[pd.DataFrame] = []
    t0 = time.perf_counter()

    for lift_id, model, init_age in fleet:
        lift_rng = np.random.default_rng(master_rng.integers(0, 2**32))
        sim = LiftSimulator(lift_id, model, init_age, cfg, lift_rng)
        df = sim.simulate()
        frames.append(df)

        if verbose:
            bd = int(df["breakdown_occurred"].sum())
            sev = df["failure_severity"].value_counts().sort_index()
            parts = ", ".join(f"s{k}={v}" for k, v in sev.items())
            print(
                f"  {lift_id} ({model:>18s}, age {init_age:4.1f}y): "
                f"{bd:2d} breakdowns  [{parts}]"
            )

    combined = pd.concat(frames, ignore_index=True)
    elapsed = time.perf_counter() - t0

    if verbose:
        _print_summary(combined, fleet, cfg, elapsed)

    return combined


def _print_summary(df: pd.DataFrame, fleet, cfg: SimConfig, elapsed: float) -> None:
    n_lifts = len(fleet)
    bd_total = int(df["breakdown_occurred"].sum())
    sev_counts = df["failure_severity"].value_counts().sort_index()

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print(f"  Records              : {len(df):>10,}")
    print(f"  Breakdowns           : {bd_total:>10}")
    print(f"  Avg breakdowns/lift  : {bd_total / n_lifts:>10.2f}")
    print(f"  Wall-clock time      : {elapsed:>10.2f} s")
    print()
    labels = {0: "healthy", 1: "warning", 2: "critical", 3: "breakdown"}
    print("  Severity distribution:")
    for sev in sorted(sev_counts.index):
        pct = sev_counts[sev] / len(df) * 100
        print(
            f"    {sev} ({labels.get(sev, '?'):>9s}): {sev_counts[sev]:>10,}  ({pct:5.2f} %)"
        )

    bd_df = df[df["breakdown_occurred"]]
    if len(bd_df) > 0:
        print()
        print("  Breakdowns by lift × subsystem:")
        ct = (
            bd_df.groupby(["lift_id", "primary_failure_subsystem"])
            .size()
            .reset_index(name="n")
        )
        for _, r in ct.iterrows():
            print(f"    {r['lift_id']}: {r['primary_failure_subsystem']} × {r['n']}")
        print()
        cols = [
            "timestamp",
            "lift_id",
            "primary_failure_subsystem",
            "failure_severity",
            "trip_count",
        ]
        print("  Sample breakdown events:")
        print("  " + bd_df[cols].head(12).to_string(index=False).replace("\n", "\n  "))

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════


def compute_rul(record: pd.Series, avail_maintenances: pd.DataFrame) -> float:
    """
    Compute the remaining useful life (RUL) in hours for a given record.
    """
    time_now = record["timestamp"]
    next_available_maintenance = avail_maintenances[
        avail_maintenances["timestamp"] >= time_now
    ]["timestamp"].min()
    if pd.isna(next_available_maintenance):
        return np.inf
    time_now = pd.to_datetime(time_now).to_pydatetime()
    next_available_maintenance = pd.to_datetime(
        next_available_maintenance
    ).to_pydatetime()
    return (next_available_maintenance - time_now).total_seconds() / 3600


def preprocess(lift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed features to the dataset:
    * RUL for each entry
    * Age since last maintenance (scaled to maintenance cycle)
    * Meta features (ratios, products)
    * Cumulative features
    * Time derivatives for sensor metrics
    """
    lift_models = lift_df["lift_model"].unique()
    lift_df = lift_df.copy()

    # Compute lift_age_hours from lift_age_days if not already present
    if "lift_age_hours" not in lift_df.columns:
        lift_df["lift_age_hours"] = lift_df["lift_age_days"] * 24.0

    # Add RULs
    maintenance_events = lift_df[lift_df["maintenance_done"] == 1]
    for lift_model in lift_models:
        available_maintenance_events = maintenance_events[
            maintenance_events["lift_model"] == lift_model
        ]
        log_events = lift_df[lift_df["lift_model"] == lift_model]
        for idx, entry in log_events.iterrows():
            rul = compute_rul(entry, available_maintenance_events)
            lift_df.at[idx, "RUL_hrs"] = rul

    # Add age since last maintenance
    lift_df["age_since_last_maint"] = lift_df.groupby("lift_id")[
        "lift_age_hours"
    ].transform(lambda x: x - x.where(lift_df["maintenance_done"] == 1).ffill())
    expected_interval = lift_df.groupby("lift_model")["age_since_last_maint"].transform(
        "median"
    )
    lift_df["age_since_last_maint"] = (
        lift_df["age_since_last_maint"] / expected_interval
    )

    # Meta features
    lift_df["arm_door_ratio"] = lift_df["arm_dist_mm"] / (
        lift_df["door_dist_mm"] + 1e-6
    )
    lift_df["floor_door_ratio"] = lift_df["leveling_accuracy_mm"] / (
        lift_df["door_dist_mm"] + 1e-6
    )
    lift_df["temp_x_rope"] = lift_df["bearing_temp_c"] * lift_df["rope_degradation_mv"]

    # Cumulative features
    lift_df["cumulative_rope_degradation"] = lift_df.groupby("lift_id")[
        "rope_degradation_mv"
    ].cumsum()
    lift_df["cumulative_bearing_heat"] = lift_df.groupby("lift_id")[
        "bearing_temp_c"
    ].cumsum()

    # Time derivatives for sensor metrics
    sensor_cols = [
        "arm_dist_mm",
        "door_dist_mm",
        "leveling_accuracy_mm",
        "rope_degradation_mv",
        "bearing_temp_c",
    ]
    for col in sensor_cols:
        lift_df[col + "_per_hr"] = np.round(lift_df[col].diff().fillna(0), 4)

    return lift_df


# MAIN


def main() -> None:
    """
    The main loop of the data generation process
    """
    cfg = SimConfig()
    fleet = DEFAULT_FLEET

    df = run_simulation(cfg=cfg, fleet=fleet, verbose=True)

    # Apply preprocessing
    df = preprocess(df)

    # Select and rename only the required columns
    df_filtered = pd.DataFrame(
        {
            "LIFT_ID": df["lift_id"],
            "LIFT_MODEL": df["lift_model"],
            "MODEL_ID": df["lift_model"].astype("category").cat.codes,
            "LIFT_AGE_HR": df["lift_age_days"] * 24.0,
            "AGE_SINCE_LAST_MNT": df["age_since_last_maint"],
            "ARM_DIST_mm": df["arm_dist_mm"],
            "ARM_DIST_DELTA": df["arm_dist_mm_per_hr"],
            "DOOR_DIST_mm": df["door_dist_mm"],
            "DOOR_DIST_DELTA": df["door_dist_mm_per_hr"],
            "FLOOR_DIST_mm": df["leveling_accuracy_mm"],
            "FLOOR_DIST_DELTA": df["leveling_accuracy_mm_per_hr"],
            "FLOOR_DOOR_RATIO": df["floor_door_ratio"],
            "ROPE_MFL_mV": df["rope_degradation_mv"],
            "ROPE_MFL_DELTA": df["rope_degradation_mv_per_hr"],
            "ROPE_MFL_CUM": df["cumulative_rope_degradation"],
            "BEARING_TEMP_C": df["bearing_temp_c"],
            "BEARING_TEMP_DELTA": df["bearing_temp_c_per_hr"],
            "BEARING_TEMP_CUM": df["cumulative_bearing_heat"],
            "TEMP_X_ROPE": df["temp_x_rope"],
            "RUL_HR": df["RUL_hrs"],
        }
    )

    output_path = f"{OUTPUT_FILENAME}.pkl"
    df_filtered.to_pickle(output_path)
    print(f"\n  Saved → {output_path}  ({len(df_filtered):,} rows)")


if __name__ == "__main__":
    main()
