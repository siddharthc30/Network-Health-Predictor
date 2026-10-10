"""Anomaly detection for network telemetry.

Two complementary methods are used, because each one is blind to what the
other sees.

Statistical (rolling z-score)
    Each KPI of each site is compared with the mean and standard deviation of
    that site's own preceding window. A value more than 2.5 standard
    deviations away is flagged. This method is interpretable: every anomaly
    names one KPI, the value, the range that was expected, and how far off it
    was. That makes it the source for explanations shown to engineers. Its
    weakness is that it looks at one KPI at a time, so a pattern where every
    KPI shifts a little (CPU up 8%, latency up 15%, SINR down 2 dB) can stay
    under every individual threshold.

Isolation Forest (multivariate)
    An IsolationForest is trained on all nine KPIs at once and isolates
    rows that are unusual as a combination. It catches those joint shifts and
    gives a single anomaly score per row, which is useful to boost failure
    prediction. Its weakness is that it cannot say which KPI is responsible.

Parameter choices follow the scikit-learn documentation for
sklearn.ensemble.IsolationForest
(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
and the "Novelty and Outlier Detection" user guide:

* n_estimators=100 is the library default, and the documentation notes the
  path length converges well before that many trees.
* max_samples="auto" uses min(256, n_samples) per tree. Isolation works with
  small subsamples, and larger ones add cost without better isolation.
* contamination=0.1 sets the score threshold so that 10% of rows are flagged,
  matching the share of sites with planted faults.
* random_state=42 makes the forest reproducible.

The z-score baseline for each point uses only the rows before it, so a value
is never compared against a window that already contains itself.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config import MAX_RESULTS
from app.models import Anomaly

# --- KPI columns -----------------------------------------------------------

KPI_COLUMNS = [
    "cpu_utilization",
    "memory_utilization",
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "connected_users",
    "signal_quality_sinr",
    "handover_failure_rate",
    "alarm_count",
]

# --- Statistical method settings -------------------------------------------

# Twelve 15-minute samples is a 3-hour baseline: long enough to average out
# noise, short enough to adapt to daily traffic shape
DEFAULT_WINDOW_SIZE = 12
# 2.5 sigma flags about 1% of normally distributed noise, a workable alert rate
Z_WARNING = 2.5
# 3.5 sigma is rare under noise alone, so a larger deviation is a real event
Z_CRITICAL = 3.5
# A window that is almost flat would turn tiny moves into huge z-scores, so the
# std is never smaller than 5% of the mean (top of the generator noise range)
STD_FLOOR_RATIO = 0.05
# Absolute std floors cover KPIs whose mean is near zero. Each is the smallest
# change worth caring about in that KPI's own unit.
STD_FLOOR_ABSOLUTE = {
    "cpu_utilization": 1.0,
    "memory_utilization": 1.0,
    "latency_ms": 1.0,
    "packet_loss_pct": 0.1,
    "throughput_mbps": 5.0,
    "connected_users": 2.0,
    "signal_quality_sinr": 0.5,
    "handover_failure_rate": 0.5,
    "alarm_count": 1.0,
}
DESCRIPTION_DECIMALS = 2

# --- Isolation Forest settings ---------------------------------------------

# The task expects roughly 10% of rows to be anomalous
IFOREST_CONTAMINATION = 0.1
# Library default; path lengths converge well before 100 trees
IFOREST_N_ESTIMATORS = 100
RANDOM_STATE = 42

# Summary of anomalies
SEVERITY_LEVELS = ("warning", "critical")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when the DataFrame lacks needed columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def detect_statistical_anomalies(
    df: pd.DataFrame, site_id: str, window_size: int = DEFAULT_WINDOW_SIZE
) -> list[Anomaly]:
    """Flag KPI values far from the site's rolling mean using z-scores."""
    _require_columns(df, ["site_id", "timestamp", *KPI_COLUMNS])
    site = df[df["site_id"] == site_id].sort_values("timestamp")
    if len(site) <= window_size:
        return []

    anomalies = []
    for kpi in KPI_COLUMNS:
        values = site[kpi].astype(float)
        # shift(1) keeps the current value out of its own baseline
        history = values.shift(1).rolling(window_size, min_periods=window_size)
        mean = history.mean()
        std = history.std()
        floor = np.maximum(STD_FLOOR_RATIO * mean.abs(), STD_FLOOR_ABSOLUTE[kpi])
        std = np.maximum(std, floor)

        z_score = (values - mean) / std
        flagged = z_score.abs() > Z_WARNING

        for position in np.flatnonzero(flagged.to_numpy()):
            z = float(z_score.iloc[position])
            low = float(mean.iloc[position] - Z_WARNING * std.iloc[position])
            high = float(mean.iloc[position] + Z_WARNING * std.iloc[position])
            value = float(values.iloc[position])
            direction = "above" if z > 0 else "below"
            anomalies.append(
                Anomaly(
                    site_id=site_id,
                    timestamp=site["timestamp"].iloc[position],
                    metric_name=kpi,
                    current_value=value,
                    expected_range_low=low,
                    expected_range_high=high,
                    severity="critical" if abs(z) > Z_CRITICAL else "warning",
                    description=(
                        f"{kpi} is {round(value, DESCRIPTION_DECIMALS)}, "
                        f"{round(abs(z), 1)} standard deviations {direction} its "
                        f"rolling mean (expected "
                        f"{round(low, DESCRIPTION_DECIMALS)} to "
                        f"{round(high, DESCRIPTION_DECIMALS)})"
                    ),
                )
            )
    return anomalies


def detect_ml_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_anomaly and anomaly_score columns using an Isolation Forest.

    anomaly_score is the negated sklearn score_samples value, so it lies in
    (0, 1) and a higher value means a more anomalous row.
    """
    _require_columns(df, KPI_COLUMNS)
    features = df[KPI_COLUMNS].astype(float)

    forest = IsolationForest(
        n_estimators=IFOREST_N_ESTIMATORS,
        contamination=IFOREST_CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    forest.fit(features)

    result = df.copy()
    result["is_anomaly"] = forest.predict(features) == -1
    result["anomaly_score"] = -forest.score_samples(features)
    return result


def detect_all_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Anomaly]]:
    """Run both methods; return the augmented DataFrame and per-KPI anomalies."""
    augmented = detect_ml_anomalies(df)

    anomalies = []
    for site_id in df["site_id"].unique():
        anomalies.extend(detect_statistical_anomalies(df, site_id))
    return augmented, anomalies


def get_anomaly_summary(anomalies: list[Anomaly]) -> dict:
    """Summarize anomalies by severity, by KPI, and by most affected site."""
    severity_counts = Counter(a.severity for a in anomalies)
    metric_counts = Counter(a.metric_name for a in anomalies)
    site_counts = Counter(a.site_id for a in anomalies)
    critical_by_site = Counter(a.site_id for a in anomalies if a.severity == "critical")

    top_sites = [
        {
            "site_id": site_id,
            "anomaly_count": count,
            "critical_count": critical_by_site.get(site_id, 0),
        }
        for site_id, count in site_counts.most_common(MAX_RESULTS)
    ]
    return {
        "total_anomalies": len(anomalies),
        "by_severity": {level: severity_counts.get(level, 0) for level in SEVERITY_LEVELS},
        "by_metric": dict(metric_counts.most_common()),
        "top_affected_sites": top_sites,
    }
