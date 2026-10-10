"""Synthetic telecom network telemetry generator.

Approach
--------
Each cell site is assigned one scenario. A scenario is described by two
things: a *stress profile* (a value between 0 and 1 at every 15-minute
timestamp) and an *end state* (the KPI values reached when stress is 1).
Every KPI is then interpolated between the site's own healthy baseline and
the scenario end state:

    kpi(t) = baseline + stress(t) * (end_state - baseline) + gaussian noise

Because all KPIs share one stress profile, they degrade together, which gives
the correlations seen in real networks:

* CPU up        -> latency up, throughput down
* users up      -> CPU up, packet loss up, latency up, handover failures up
* SINR down     -> throughput down, packet loss up, handover failures up

Alarm counts and site status are not scripted. They are derived at every
timestamp from the generated KPI values and the thresholds in app.config, so
the labels always agree with the data a downstream detector will see.

Scenarios
---------
HEALTHY          KPIs stay at baseline with small noise; status stays Healthy.
DEGRADING        Slow, accelerating drift over 24 hours (a leak or ageing
                 hardware). Expect Degraded around hour 16-18, and possibly
                 Critical in the final hour.
CONGESTION       Normal for 18 hours, then user count spikes and drags CPU,
                 latency, loss and throughput with it (an event or overload).
FAILING          KPIs cross critical thresholds in the last 2-4 hours and the
                 site ends Critical or Failed (a developing hardware fault).
INTERMITTENT     Three or four spikes of 1-2 hours that partly recover
                 between peaks (flapping links or unstable power).
SUDDEN_FAILURE   Normal for about 20 hours, then every KPI crashes within
                 roughly an hour (a hard outage with no early warning).

Reproducibility
---------------
All randomness comes from numpy generators seeded with SEED = 42, so the KPI
values are identical on every run. Only the timestamps move, because they
are anchored to the current UTC time.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import (
    ALARM_COUNT_CRITICAL,
    ALARM_COUNT_WARNING,
    CPU_CRITICAL,
    CPU_WARNING,
    HANDOVER_FAILURE_RATE_CRITICAL,
    HANDOVER_FAILURE_RATE_WARNING,
    HEALTH_SCORE_DEGRADED,
    HEALTH_SCORE_WARNING,
    HOURS_OF_DATA,
    LATENCY_CRITICAL_5G,
    LATENCY_CRITICAL_LTE,
    LATENCY_WARNING_5G,
    LATENCY_WARNING_LTE,
    MEMORY_CRITICAL,
    MEMORY_WARNING,
    PACKET_LOSS_CRITICAL,
    PACKET_LOSS_WARNING,
    SINR_CRITICAL,
    SINR_WARNING,
    SITE_COUNT,
    THROUGHPUT_DROP_CRITICAL,
    THROUGHPUT_DROP_WARNING,
)
from app.models import SiteMetric

# --- General settings -------------------------------------------------------

# Fixed seed so every run produces identical KPI values
SEED = 42
# Telemetry cadence of 15 minutes matches common RAN performance counters
INTERVAL_MINUTES = 15
POINTS_PER_HOUR = 60 // INTERVAL_MINUTES
# Column order follows the SiteMetric schema so the CSV and model never drift
COLUMNS = list(SiteMetric.model_fields)
SCENARIOS_FILENAME = "site_scenarios.json"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# --- Site layout ------------------------------------------------------------

CITIES = [
    "Dallas", "Houston", "Chicago", "Phoenix", "Seattle", "Denver", "Atlanta",
    "Miami", "Boston", "Portland", "Austin", "San_Diego", "Detroit",
    "Minneapolis", "Tampa",
]
# 5G is the larger share because most new capacity is deployed on 5G
NETWORK_TYPE_COUNTS = {"5G": 30, "LTE": 20}
# Healthy sites dominate, as in a real network; the rest cover each failure mode
SCENARIO_COUNTS = {
    "HEALTHY": 25,
    "DEGRADING": 8,
    "CONGESTION": 7,
    "FAILING": 5,
    "INTERMITTENT": 3,
    "SUDDEN_FAILURE": 2,
}
SITE_NUMBER_START = 1000
# A step larger than one keeps IDs from looking sequential and stays 4 digits
SITE_NUMBER_STEP = 17

# --- Baselines and noise ----------------------------------------------------

# Healthy (low, high) ranges per network type; alarms are always 0 when healthy
BASELINE_RANGES = {
    "5G": {
        "cpu_utilization": (30.0, 55.0),
        "memory_utilization": (35.0, 55.0),
        "latency_ms": (5.0, 15.0),
        "packet_loss_pct": (0.01, 0.5),
        "throughput_mbps": (800.0, 1200.0),
        "connected_users": (50.0, 200.0),
        "signal_quality_sinr": (10.0, 25.0),
    },
    "LTE": {
        "cpu_utilization": (35.0, 60.0),
        "memory_utilization": (40.0, 60.0),
        "latency_ms": (20.0, 40.0),
        "packet_loss_pct": (0.1, 1.0),
        "throughput_mbps": (100.0, 300.0),
        "connected_users": (100.0, 400.0),
        "signal_quality_sinr": (5.0, 20.0),
    },
}
# Baselines are drawn from the inner part of each range so that noise on a
# healthy site cannot push a KPI over a warning threshold
BASELINE_MARGIN_FRACTION = 0.1
# Noise sigma per site is 2-5% of each KPI's baseline value
NOISE_SIGMA_FRACTION_RANGE = (0.02, 0.05)

# Handover failure rate (%) baselines. 5G cells are smaller, so devices hand
# over more often and fail slightly more often than on LTE.
HANDOVER_BASELINE_RANGES = {"5G": (1.0, 4.0), "LTE": (0.5, 3.0)}
# Handover has its own random stream, so adding this KPI leaves every other
# KPI's generated values unchanged
# Percentage points of handover failure per 100% change in user count; more
# users mean more handover attempts competing for target-cell resources
HANDOVER_USER_COUPLING = 3.0
# Percentage points of handover failure per dB of lost SINR; weak signal at the
# cell edge is the classic cause of failed handovers
HANDOVER_SINR_COUPLING = 0.25

# --- Data quality limits ----------------------------------------------------

PERCENT_MIN = 0.0
PERCENT_MAX = 100.0
# Below about -20 dB a link cannot be decoded at all, so there is no need to go lower
SINR_MIN_DB = -20.0
VALUE_DECIMALS = 2

# --- Scenario timing --------------------------------------------------------

# Squaring the ramp keeps early hours calm so the first warning lands near hour 16
DEGRADING_RAMP_EXPONENT = 2.0
# Congestion begins 6 hours before the end, so the first 18 hours are normal
CONGESTION_WINDOW_HOURS = 6
# Users surge over two hours, as at the end of an event or a commute
CONGESTION_RAMP_HOURS = 2
FAILING_WINDOW_HOURS = (2.0, 4.0)
# Sudden failure starts 2-3 hours before the end, so about 20+ hours are normal
SUDDEN_WINDOW_HOURS = (2.0, 3.0)
# Four steps (one hour) is fast enough to read as a crash, not a drift
SUDDEN_RAMP_STEPS = 4
INTERMITTENT_SPIKE_COUNT = (3, 4)
# A spike lasts 4-8 steps, which is 1-2 hours at 15-minute intervals
INTERMITTENT_SPIKE_STEPS = (4, 8)
# Between spikes KPIs only partly recover, leaving a small residual stress
INTERMITTENT_FLOOR_STRESS = 0.05
# Peak stress below 1 keeps spikes in Degraded territory instead of Failed
INTERMITTENT_PEAK_STRESS = 0.8

# --- Scenario end states (KPI values reached at stress = 1) -----------------

# Ranges are (low, high); a value is drawn per site so no two sites match.
# latency_5g / latency_lte are absolute values; latency_mult scales baseline.
SCENARIO_END_STATES = {
    "DEGRADING": {
        "cpu": (83.0, 87.0),
        "memory": (65.0, 75.0),
        "latency_5g": (50.0, 60.0),
        "latency_lte": (85.0, 95.0),
        "loss": (3.5, 4.5),
        "throughput_ratio": (0.55, 0.65),
        "sinr_drop": (7.0, 9.0),
        "users_mult": (1.1, 1.3),
        "handover": (11.0, 13.0),
    },
    "CONGESTION": {
        "cpu": (80.0, 95.0),
        "memory": (60.0, 75.0),
        "latency_mult": (2.0, 2.5),
        "loss": (3.0, 6.0),
        "throughput_ratio": (0.5, 0.7),
        "sinr_drop": (1.0, 3.0),
        "users_mult": (1.6, 2.0),
        "handover": (10.0, 18.0),
    },
    "FAILING": {
        "cpu": (96.0, 99.0),
        "memory": (88.0, 97.0),
        "latency_5g": (110.0, 160.0),
        "latency_lte": (210.0, 280.0),
        "loss": (8.0, 12.0),
        "throughput_ratio": (0.05, 0.18),
        "sinr_drop": (10.0, 14.0),
        "users_mult": (0.9, 1.2),
        "handover": (17.0, 24.0),
    },
    "INTERMITTENT": {
        "cpu": (80.0, 88.0),
        "memory": (60.0, 70.0),
        "latency_5g": (40.0, 48.0),
        "latency_lte": (75.0, 90.0),
        "loss": (2.5, 3.5),
        "throughput_ratio": (0.55, 0.65),
        "sinr_drop": (6.0, 8.0),
        "users_mult": (1.1, 1.3),
        # Value reached at the spike peak; scaled up to a full-stress end state
        "handover": (12.0, 16.0),
    },
    "SUDDEN_FAILURE": {
        "cpu": (96.0, 99.0),
        "memory": (90.0, 98.0),
        "latency_5g": (120.0, 180.0),
        "latency_lte": (220.0, 300.0),
        "loss": (9.0, 15.0),
        "throughput_ratio": (0.02, 0.12),
        "sinr_drop": (12.0, 16.0),
        "users_mult": (0.6, 1.0),
        "handover": (18.0, 22.0),
    },
}

# --- Status and alarm logic -------------------------------------------------

LATENCY_THRESHOLDS = {
    "5G": (LATENCY_WARNING_5G, LATENCY_CRITICAL_5G),
    "LTE": (LATENCY_WARNING_LTE, LATENCY_CRITICAL_LTE),
}
# Each unit of threshold severity costs this many sub-score points, so a KPI
# exactly at its critical threshold scores 40 and lands in the Critical band
HEALTH_PENALTY_PER_SEVERITY = 60.0
HEALTH_SUBSCORE_MAX = 100.0
# Alarm count drawn per site state: (low, high), inclusive
ALARM_RANGE_NONE = (0, 0)
ALARM_RANGE_ONE_WARNING = (1, 3)
ALARM_RANGE_MULTI_WARNING = (3, 4)
ALARM_RANGE_ONE_CRITICAL = (5, 7)
ALARM_RANGE_TWO_CRITICAL = (6, 8)
ALARM_RANGE_SEVERE = (8, 10)
MULTI_WARNING_MIN_KPIS = 2
SEVERE_MIN_CRITICAL_KPIS = 3
TWO_CRITICAL_KPIS = 2
# Failed requires multiple KPIs beyond critical and more than 7 alarms
FAILED_MIN_CRITICAL_KPIS = 2
FAILED_ALARM_FLOOR = 7

STATUS_HEALTHY = "Healthy"
STATUS_DEGRADED = "Degraded"
STATUS_CRITICAL = "Critical"
STATUS_FAILED = "Failed"


# --- Internal helpers -------------------------------------------------------

def _upper_severity(value: float, warning: float, critical: float) -> float:
    """Return 0 below warning, 1 at critical, scaled linearly in between."""
    return max(0.0, (value - warning) / (critical - warning))


def _lower_severity(value: float, warning: float, critical: float) -> float:
    """Return 0 above warning, 1 at critical, for KPIs where low is bad."""
    return max(0.0, (warning - value) / (warning - critical))


def _draw_baselines(network_type: str, rng: np.random.Generator) -> dict[str, float]:
    """Draw one healthy baseline per KPI from the inner part of each range."""
    baselines = {}
    for kpi, (low, high) in BASELINE_RANGES[network_type].items():
        margin = (high - low) * BASELINE_MARGIN_FRACTION
        baselines[kpi] = float(rng.uniform(low + margin, high - margin))
    return baselines


def _draw_end_state(
    scenario: str,
    network_type: str,
    baselines: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Pick the KPI values this site reaches at full stress."""
    if scenario == "HEALTHY":
        return dict(baselines)

    spec = SCENARIO_END_STATES[scenario]

    def pick(key: str) -> float:
        low, high = spec[key]
        return float(rng.uniform(low, high))

    if "latency_mult" in spec:
        latency = baselines["latency_ms"] * pick("latency_mult")
    else:
        latency = pick(f"latency_{network_type.lower()}")

    return {
        "cpu_utilization": pick("cpu"),
        "memory_utilization": pick("memory"),
        "latency_ms": latency,
        "packet_loss_pct": pick("loss"),
        "throughput_mbps": baselines["throughput_mbps"] * pick("throughput_ratio"),
        "connected_users": baselines["connected_users"] * pick("users_mult"),
        "signal_quality_sinr": baselines["signal_quality_sinr"] - pick("sinr_drop"),
    }


def _draw_handover_levels(
    scenario: str, network_type: str, rng: np.random.Generator
) -> tuple[float, float]:
    """Draw the healthy baseline and full-stress end value for handover failures."""
    low, high = HANDOVER_BASELINE_RANGES[network_type]
    margin = (high - low) * BASELINE_MARGIN_FRACTION
    baseline = float(rng.uniform(low + margin, high - margin))
    if scenario == "HEALTHY":
        return baseline, baseline

    target = float(rng.uniform(*SCENARIO_END_STATES[scenario]["handover"]))
    if scenario == "INTERMITTENT":
        # The target is the value at the spike peak, which reaches only part of full stress
        target = baseline + (target - baseline) / INTERMITTENT_PEAK_STRESS
    return baseline, target


def _build_stress_profile(
    scenario: str, n_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Return the stress value (0 to 1) for every timestamp of a scenario."""
    idx = np.arange(n_points, dtype=float)
    last = n_points - 1

    if scenario == "HEALTHY":
        return np.zeros(n_points)

    if scenario == "DEGRADING":
        return (idx / last) ** DEGRADING_RAMP_EXPONENT

    if scenario == "CONGESTION":
        onset = n_points - CONGESTION_WINDOW_HOURS * POINTS_PER_HOUR
        ramp = CONGESTION_RAMP_HOURS * POINTS_PER_HOUR
        return np.clip((idx - onset) / ramp, 0.0, 1.0)

    if scenario == "FAILING":
        window = rng.uniform(*FAILING_WINDOW_HOURS) * POINTS_PER_HOUR
        return np.clip((idx - (last - window)) / window, 0.0, 1.0)

    if scenario == "SUDDEN_FAILURE":
        window = rng.uniform(*SUDDEN_WINDOW_HOURS) * POINTS_PER_HOUR
        return np.clip((idx - (last - window)) / SUDDEN_RAMP_STEPS, 0.0, 1.0)

    if scenario == "INTERMITTENT":
        stress = np.full(n_points, INTERMITTENT_FLOOR_STRESS)
        n_spikes = int(rng.integers(INTERMITTENT_SPIKE_COUNT[0], INTERMITTENT_SPIKE_COUNT[1] + 1))
        segment = n_points // n_spikes
        height = INTERMITTENT_PEAK_STRESS - INTERMITTENT_FLOOR_STRESS
        for k in range(n_spikes):
            width = int(rng.integers(INTERMITTENT_SPIKE_STEPS[0], INTERMITTENT_SPIKE_STEPS[1] + 1))
            start = k * segment + int(rng.integers(0, segment - width + 1))
            shape = np.sin(np.pi * (np.arange(width) + 1) / (width + 1))
            stress[start:start + width] += height * shape
        return stress

    raise ValueError(f"Unknown scenario: {scenario}")


def _classify_row(
    values: dict[str, float],
    network_type: str,
    baseline_throughput: float,
    rng: np.random.Generator,
) -> tuple[int, str]:
    """Derive alarm count and site status from one timestamp's KPI values."""
    lat_warn, lat_crit = LATENCY_THRESHOLDS[network_type]
    throughput_ratio = values["throughput_mbps"] / baseline_throughput

    severities = [
        _upper_severity(values["cpu_utilization"], CPU_WARNING, CPU_CRITICAL),
        _upper_severity(values["memory_utilization"], MEMORY_WARNING, MEMORY_CRITICAL),
        _upper_severity(values["latency_ms"], lat_warn, lat_crit),
        _upper_severity(values["packet_loss_pct"], PACKET_LOSS_WARNING, PACKET_LOSS_CRITICAL),
        _lower_severity(throughput_ratio, THROUGHPUT_DROP_WARNING, THROUGHPUT_DROP_CRITICAL),
        _lower_severity(values["signal_quality_sinr"], SINR_WARNING, SINR_CRITICAL),
        _upper_severity(
            values["handover_failure_rate"],
            HANDOVER_FAILURE_RATE_WARNING,
            HANDOVER_FAILURE_RATE_CRITICAL,
        ),
    ]
    n_critical = sum(1 for s in severities if s >= 1.0)
    n_warning = sum(1 for s in severities if 0.0 < s < 1.0)

    if n_critical >= SEVERE_MIN_CRITICAL_KPIS:
        alarm_range = ALARM_RANGE_SEVERE
    elif n_critical == TWO_CRITICAL_KPIS:
        alarm_range = ALARM_RANGE_TWO_CRITICAL
    elif n_critical == 1:
        alarm_range = ALARM_RANGE_ONE_CRITICAL
    elif n_warning >= MULTI_WARNING_MIN_KPIS:
        alarm_range = ALARM_RANGE_MULTI_WARNING
    elif n_warning == 1:
        alarm_range = ALARM_RANGE_ONE_WARNING
    else:
        alarm_range = ALARM_RANGE_NONE
    alarms = int(rng.integers(alarm_range[0], alarm_range[1] + 1))

    # Health score is the mean of per-KPI sub-scores, alarms included
    alarm_severity = _upper_severity(alarms, ALARM_COUNT_WARNING, ALARM_COUNT_CRITICAL)
    sub_scores = [
        max(0.0, HEALTH_SUBSCORE_MAX - HEALTH_PENALTY_PER_SEVERITY * s)
        for s in severities + [alarm_severity]
    ]
    health_score = sum(sub_scores) / len(sub_scores)

    if n_critical >= FAILED_MIN_CRITICAL_KPIS and alarms > FAILED_ALARM_FLOOR:
        status = STATUS_FAILED
    elif n_critical >= 1 or alarms >= ALARM_COUNT_CRITICAL or health_score < HEALTH_SCORE_DEGRADED:
        status = STATUS_CRITICAL
    elif n_warning >= 1 or alarms >= ALARM_COUNT_WARNING or health_score < HEALTH_SCORE_WARNING:
        status = STATUS_DEGRADED
    else:
        status = STATUS_HEALTHY
    return alarms, status


def _plan_sites(rng: np.random.Generator) -> list[tuple[str, str, str]]:
    """Return (site_id, network_type, scenario) for every site."""
    network_types = [nt for nt, count in NETWORK_TYPE_COUNTS.items() for _ in range(count)]
    scenarios = [sc for sc, count in SCENARIO_COUNTS.items() for _ in range(count)]
    if len(network_types) != SITE_COUNT or len(scenarios) != SITE_COUNT:
        raise ValueError(
            f"Network type counts ({len(network_types)}) and scenario counts "
            f"({len(scenarios)}) must both equal SITE_COUNT ({SITE_COUNT})"
        )
    scenarios = [scenarios[i] for i in rng.permutation(SITE_COUNT)]

    plan = []
    for i, (network_type, scenario) in enumerate(zip(network_types, scenarios)):
        city = CITIES[i % len(CITIES)].upper()
        number = SITE_NUMBER_START + i * SITE_NUMBER_STEP
        plan.append((f"{city}_{network_type}_RAN_{number}", network_type, scenario))
    return plan


# --- Public API -------------------------------------------------------------

def generate_site_telemetry(
    site_id: str,
    network_type: str,
    scenario: str,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate 24 hours of telemetry for one site following its assigned scenario."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_points = len(timestamps)

    baselines = _draw_baselines(network_type, rng)
    end_state = _draw_end_state(scenario, network_type, baselines, rng)
    stress = _build_stress_profile(scenario, n_points, rng)
    noise_fraction = rng.uniform(*NOISE_SIGMA_FRACTION_RANGE)

    series = {}
    noises = {}
    for kpi, baseline in baselines.items():
        clean = baseline + stress * (end_state[kpi] - baseline)
        noise = rng.normal(0.0, noise_fraction * abs(baseline), n_points)
        series[kpi] = clean + noise
        noises[kpi] = noise

    # Handover failures follow the scenario stress, and pick up the same random
    # movement as user count and SINR so they stay correlated reading to reading
    handover_rng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
    handover_base, handover_end = _draw_handover_levels(scenario, network_type, handover_rng)
    handover = handover_base + stress * (handover_end - handover_base)
    handover += handover_rng.normal(0.0, noise_fraction * handover_base, n_points)
    handover += HANDOVER_USER_COUPLING * noises["connected_users"] / baselines["connected_users"]
    handover -= HANDOVER_SINR_COUPLING * noises["signal_quality_sinr"]
    series["handover_failure_rate"] = np.clip(handover, PERCENT_MIN, PERCENT_MAX)

    series["cpu_utilization"] = np.clip(series["cpu_utilization"], PERCENT_MIN, PERCENT_MAX)
    series["memory_utilization"] = np.clip(series["memory_utilization"], PERCENT_MIN, PERCENT_MAX)
    series["packet_loss_pct"] = np.clip(series["packet_loss_pct"], PERCENT_MIN, PERCENT_MAX)
    series["latency_ms"] = np.maximum(series["latency_ms"], 0.0)
    series["throughput_mbps"] = np.maximum(series["throughput_mbps"], 0.0)
    series["connected_users"] = np.maximum(np.round(series["connected_users"]), 0.0)
    series["signal_quality_sinr"] = np.maximum(series["signal_quality_sinr"], SINR_MIN_DB)

    records = []
    for i, timestamp in enumerate(timestamps):
        values = {kpi: round(float(series[kpi][i]), VALUE_DECIMALS) for kpi in series}
        alarms, status = _classify_row(
            values, network_type, baselines["throughput_mbps"], rng
        )
        record = {
            "site_id": site_id,
            "timestamp": timestamp,
            "network_type": network_type,
            **values,
            "connected_users": int(values["connected_users"]),
            "alarm_count": alarms,
            "site_status": status,
        }
        SiteMetric.model_validate(record)
        records.append(record)
    return records


def generate_dataset() -> pd.DataFrame:
    """Generate complete dataset for all 50 sites. Returns DataFrame with all required columns."""
    rng = np.random.default_rng(SEED)

    # Anchor to the current UTC time, floored to the 15-minute grid
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    n_points = HOURS_OF_DATA * POINTS_PER_HOUR
    start = now.floor(f"{INTERVAL_MINUTES}min") - pd.Timedelta(hours=HOURS_OF_DATA)
    timestamps = pd.date_range(start=start, periods=n_points, freq=f"{INTERVAL_MINUTES}min")

    records = []
    scenarios = {}
    for site_id, network_type, scenario in _plan_sites(rng):
        site_rng = np.random.default_rng(int(rng.integers(2**32)))
        records.extend(
            generate_site_telemetry(site_id, network_type, scenario, timestamps, site_rng)
        )
        scenarios[site_id] = scenario

    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    df.attrs["site_scenarios"] = scenarios
    return df


def save_dataset(df: pd.DataFrame, filepath: Path) -> None:
    """Save dataset to CSV. Also save site_scenarios.json mapping."""
    scenarios = df.attrs.get("site_scenarios")
    if scenarios is None:
        raise ValueError("DataFrame has no site_scenarios metadata; use generate_dataset()")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, date_format=TIMESTAMP_FORMAT)
    scenarios_path = filepath.parent / SCENARIOS_FILENAME
    scenarios_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset from CSV with proper dtypes. Parse timestamps."""
    dtypes = {
        "site_id": "string",
        "network_type": "string",
        "site_status": "string",
        "connected_users": "int64",
        "alarm_count": "int64",
    }
    df = pd.read_csv(filepath, dtype=dtypes, parse_dates=["timestamp"])
    scenarios_path = Path(filepath).parent / SCENARIOS_FILENAME
    if scenarios_path.exists():
        df.attrs["site_scenarios"] = json.loads(scenarios_path.read_text(encoding="utf-8"))
    return df
