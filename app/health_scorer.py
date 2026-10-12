"""Network health scoring engine.

Methodology
-----------
Every telemetry row is reduced to one health score from 0 to 100 in two steps.

1. Sub-scores. Each of eight KPIs is mapped to a 0-100 sub-score with a
   piecewise linear curve through three anchor points taken from config:

       at or better than "good"   -> 100
       at the warning threshold   ->  60
       at the critical threshold  ->  20
       beyond critical            ->  falls on the same slope to 0, then clamped

   The curve is continuous, so a small KPI change never causes a large score
   jump.

2. Weighted average. Sub-scores are combined with fixed weights that sum
   to 1.0 and rounded to an integer. The integer score maps to a status:

       80-100  Healthy
       60-79   Warning
       40-59   Degraded
        0-39   Critical

   The band edges are the HEALTH_SCORE_* constants in app.config.

Weights
-------
Latency and packet loss carry the most weight (0.18 each) because they map
directly to the packet delay budget and packet error rate in 3GPP 5QI QoS
profiles, so they are the closest proxy for user experience. CPU and
throughput follow at 0.135: CPU is the best leading indicator of overload and
throughput measures delivered capacity. Handover failure rate carries 0.10:
it is the direct measure of mobility quality and causes dropped calls.
Memory, SINR, and alarms carry 0.09 each: memory changes slowly, SINR is
already reflected in throughput and loss, and alarms corroborate the KPIs
rather than lead them. When handover failure rate was added, the original
seven weights were each scaled by 0.9 so the total stays 1.0.

Thresholds
----------
Warning and critical thresholds, the "good" reference points, and the per
technology latency and throughput levels all live in app.config together
with the reasoning for each value. This module contains no threshold values.
Component sub-scores are kept so the cause of a low score can be explained.
"""

import numpy as np
import pandas as pd

from app.config import (
    ALARM_COUNT_CRITICAL,
    ALARM_COUNT_GOOD,
    ALARM_COUNT_WARNING,
    CPU_CRITICAL,
    CPU_GOOD,
    CPU_WARNING,
    HANDOVER_FAILURE_RATE_CRITICAL,
    HANDOVER_FAILURE_RATE_GOOD,
    HANDOVER_FAILURE_RATE_WARNING,
    HEALTH_SCORE_CRITICAL,
    HEALTH_SCORE_DEGRADED,
    HEALTH_SCORE_HEALTHY,
    HEALTH_SCORE_WARNING,
    LATENCY_CRITICAL_5G,
    LATENCY_CRITICAL_LTE,
    LATENCY_GOOD_5G,
    LATENCY_GOOD_LTE,
    LATENCY_WARNING_5G,
    LATENCY_WARNING_LTE,
    MAX_RESULTS,
    MEMORY_CRITICAL,
    MEMORY_GOOD,
    MEMORY_WARNING,
    PACKET_LOSS_CRITICAL,
    PACKET_LOSS_GOOD,
    PACKET_LOSS_WARNING,
    SINR_CRITICAL,
    SINR_GOOD,
    SINR_WARNING,
    SUBSCORE_CRITICAL,
    SUBSCORE_FLOOR,
    SUBSCORE_GOOD,
    SUBSCORE_WARNING,
    THROUGHPUT_CRITICAL_5G_MBPS,
    THROUGHPUT_CRITICAL_LTE_MBPS,
    THROUGHPUT_GOOD_5G_MBPS,
    THROUGHPUT_GOOD_LTE_MBPS,
    THROUGHPUT_WARNING_5G_MBPS,
    THROUGHPUT_WARNING_LTE_MBPS,
)
from app.models import HealthScore

# --- KPI weights (must sum to 1.0) -----------------------------------------

KPI_WEIGHTS = {
    # Latency weighted 0.18: maps to the 3GPP 5QI packet delay budget, the
    # strongest single driver of perceived user experience
    "latency_score": 0.18,
    # Packet loss weighted 0.18: maps to the 5QI packet error rate and breaks
    # voice, video, and TCP throughput quickly
    "packet_loss_score": 0.18,
    # CPU weighted 0.135: best leading indicator of overload before users notice
    "cpu_score": 0.135,
    # Throughput weighted 0.135: measures the capacity actually delivered
    "throughput_score": 0.135,
    # Handover failures weighted 0.10: direct measure of mobility quality and a
    # cause of dropped calls, but only affects users who are moving
    "handover_score": 0.10,
    # Memory weighted 0.09: changes slowly, and failures show up in CPU and latency
    "memory_score": 0.09,
    # SINR weighted 0.09: radio quality is already reflected in throughput and loss
    "sinr_score": 0.09,
    # Alarms weighted 0.09: corroborate KPI problems rather than lead them
    "alarm_score": 0.09,
}
WEIGHT_SUM_TOLERANCE = 1e-9

if abs(sum(KPI_WEIGHTS.values()) - 1.0) > WEIGHT_SUM_TOLERANCE:
    raise ValueError("KPI_WEIGHTS must sum to 1.0")

# --- Per-technology anchors: (good, warning, critical) ----------------------

LATENCY_ANCHORS = {
    "5G": (LATENCY_GOOD_5G, LATENCY_WARNING_5G, LATENCY_CRITICAL_5G),
    "LTE": (LATENCY_GOOD_LTE, LATENCY_WARNING_LTE, LATENCY_CRITICAL_LTE),
}
THROUGHPUT_ANCHORS = {
    "5G": (THROUGHPUT_GOOD_5G_MBPS, THROUGHPUT_WARNING_5G_MBPS, THROUGHPUT_CRITICAL_5G_MBPS),
    "LTE": (THROUGHPUT_GOOD_LTE_MBPS, THROUGHPUT_WARNING_LTE_MBPS, THROUGHPUT_CRITICAL_LTE_MBPS),
}
CPU_ANCHORS = (CPU_GOOD, CPU_WARNING, CPU_CRITICAL)
MEMORY_ANCHORS = (MEMORY_GOOD, MEMORY_WARNING, MEMORY_CRITICAL)
PACKET_LOSS_ANCHORS = (PACKET_LOSS_GOOD, PACKET_LOSS_WARNING, PACKET_LOSS_CRITICAL)
SINR_ANCHORS = (SINR_GOOD, SINR_WARNING, SINR_CRITICAL)
ALARM_ANCHORS = (ALARM_COUNT_GOOD, ALARM_COUNT_WARNING, ALARM_COUNT_CRITICAL)
HANDOVER_ANCHORS = (
    HANDOVER_FAILURE_RATE_GOOD,
    HANDOVER_FAILURE_RATE_WARNING,
    HANDOVER_FAILURE_RATE_CRITICAL,
)

# score name -> (raw KPI column, anchors); anchors are per network type for
# latency and throughput, a plain (good, warning, critical) tuple otherwise
SUBSCORE_SPEC = {
    "cpu_score": ("cpu_utilization", CPU_ANCHORS),
    "memory_score": ("memory_utilization", MEMORY_ANCHORS),
    "latency_score": ("latency_ms", LATENCY_ANCHORS),
    "packet_loss_score": ("packet_loss_pct", PACKET_LOSS_ANCHORS),
    "throughput_score": ("throughput_mbps", THROUGHPUT_ANCHORS),
    "sinr_score": ("signal_quality_sinr", SINR_ANCHORS),
    "handover_score": ("handover_failure_rate", HANDOVER_ANCHORS),
    "alarm_score": ("alarm_count", ALARM_ANCHORS),
}

SCORE_COLUMN = "health_score"
STATUS_COLUMN = "health_status"
COMPONENT_COLUMNS = list(KPI_WEIGHTS)
SUMMARY_WORST_COLUMNS = ["site_id", "network_type", SCORE_COLUMN, STATUS_COLUMN]


def _subscore(value, anchors: tuple[float, float, float]):
    """Map a KPI value (scalar or array) to a 0-100 sub-score.

    Works for KPIs where higher is worse (good < critical) and where lower is
    worse (good > critical). Beyond critical the curve keeps its slope until
    it reaches the floor, then clamps.
    """
    good, warning, critical = anchors
    slope_span = (critical - warning) * (SUBSCORE_CRITICAL - SUBSCORE_FLOOR) / (
        SUBSCORE_WARNING - SUBSCORE_CRITICAL
    )
    floor_point = critical + slope_span
    points = [good, warning, critical, floor_point]
    scores = [SUBSCORE_GOOD, SUBSCORE_WARNING, SUBSCORE_CRITICAL, SUBSCORE_FLOOR]
    if good > critical:
        # np.interp needs ascending x values, so flip for lower-is-worse KPIs
        points, scores = points[::-1], scores[::-1]
    return np.interp(value, points, scores)


def _component_scores(data, network_type: str) -> dict:
    """Compute all eight sub-scores for a row (Series) or group (DataFrame)."""
    if network_type not in LATENCY_ANCHORS:
        raise ValueError(f"Unknown network_type: {network_type}")
    return {
        score: _subscore(
            data[column],
            anchors[network_type] if isinstance(anchors, dict) else anchors,
        )
        for score, (column, anchors) in SUBSCORE_SPEC.items()
    }


def _weighted_score(components: dict):
    """Combine sub-scores into the rounded weighted average."""
    total = sum(KPI_WEIGHTS[name] * components[name] for name in KPI_WEIGHTS)
    return np.rint(total).astype(int) if isinstance(total, np.ndarray) else int(np.rint(total))


def _status_for_score(score: float) -> str:
    """Map a 0-100 score to a health status label."""
    if score >= HEALTH_SCORE_HEALTHY:
        return "Healthy"
    if score >= HEALTH_SCORE_WARNING:
        return "Warning"
    if score >= HEALTH_SCORE_DEGRADED:
        return "Degraded"
    return "Critical"


def calculate_health_score(row: pd.Series, network_type: str) -> HealthScore:
    """Score one telemetry row and return the score, status, and sub-scores."""
    components = _component_scores(row, network_type)
    score = _weighted_score(components)
    return HealthScore(
        site_id=row["site_id"],
        timestamp=row["timestamp"],
        score=score,
        status=_status_for_score(score),
        component_scores={name: round(float(value), 1) for name, value in components.items()},
    )


def score_all_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Score every row; return a copy with score, status, and sub-score columns."""
    scored = df.copy()
    for column in COMPONENT_COLUMNS:
        scored[column] = np.nan

    for network_type, group in df.groupby("network_type"):
        components = _component_scores(group, str(network_type))
        for column, values in components.items():
            scored.loc[group.index, column] = np.round(values, 1)
        scored.loc[group.index, SCORE_COLUMN] = _weighted_score(components)

    scored[SCORE_COLUMN] = scored[SCORE_COLUMN].astype(int)
    scored[STATUS_COLUMN] = scored[SCORE_COLUMN].map(_status_for_score)
    return scored


def get_current_health_summary(scored_df: pd.DataFrame) -> dict:
    """Summarize each site's latest score: status counts, average, worst sites."""
    latest = (
        scored_df.sort_values("timestamp")
        .groupby("site_id", as_index=False)
        .tail(1)
        .sort_values([SCORE_COLUMN, "site_id"])
    )
    counts = latest[STATUS_COLUMN].value_counts()

    worst = latest.head(MAX_RESULTS)
    worst_sites = []
    for _, row in worst.iterrows():
        entry = {
            "site_id": row["site_id"],
            "network_type": row["network_type"],
            "health_score": int(row[SCORE_COLUMN]),
            "health_status": row[STATUS_COLUMN],
            "weakest_kpi": row[COMPONENT_COLUMNS].astype(float).idxmin(),
        }
        worst_sites.append(entry)

    return {
        "total_sites": int(len(latest)),
        "healthy_count": int(counts.get("Healthy", 0)),
        "warning_count": int(counts.get("Warning", 0)),
        "degraded_count": int(counts.get("Degraded", 0)),
        "critical_count": int(counts.get("Critical", 0)),
        "average_score": round(float(latest[SCORE_COLUMN].mean()), 1),
        "worst_sites": worst_sites,
    }
