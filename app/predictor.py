"""Failure prediction engine.

Overview
--------
For each site the engine produces a risk score from 0.0 to 1.0, a predicted
failure type, an estimated time to failure, a confidence value, and the KPIs
that support the call. Everything is computed from the recent health score
history, the current KPI values, and the detected anomalies.

Risk scoring methodology
------------------------
The risk score is a weighted sum of five signals. Each signal is first
normalized to 0-1, then multiplied by its weight (weights sum to 1.0):

1. Health score trend (0.30). The slope of the health score over the last
   4 hours. A falling score is the most direct evidence that a site is
   moving toward failure, so it carries the largest weight. A decline of
   5 points per hour or more is high risk, 2 or more is medium.
2. Current health score (0.20). A site already in a poor band is more likely
   to fail than a healthy one regardless of direction. Below 40 is high risk,
   40-60 medium, above 60 low.
3. KPI threshold proximity (0.20). How many KPIs are within 15% of their
   critical threshold. One KPI near its limit can be noise; several together
   mean the site is running out of margin on many fronts.
4. Anomaly density (0.15). The count of anomalies in the last 2 hours. This
   catches sudden jumps that a 4-hour trend smooths over.
5. Trend acceleration (0.15). The 2-hour slope compared with the 4-hour
   slope. Degradation that is speeding up leaves less time to react than a
   steady decline at the same current score.

Failure type
------------
Each KPI is turned into a stress value (1 minus its health sub-score / 100).
Each failure type has a set of KPIs that point to it, each with a specificity
weight: how uniquely that KPI indicates that failure mode. A failure type's
score is the largest specificity-weighted stress among its KPIs, and the
highest-scoring type names the failure:

* congestion            user surge (1.0), latency and packet loss (0.5)
* capacity_exhaustion   CPU and memory (0.8)
* signal_degradation    SINR (1.0), throughput (0.5)
* hardware_degradation  alarms and packet loss (0.5)

Specificity weights keep symptoms from drowning out causes. Alarms and
latency rise in almost every fault, so on their own they say little about the
root cause, while a user surge or a SINR collapse is a much more specific
signal. The label is a heuristic: when many KPIs collapse together the root
cause is genuinely ambiguous.

Failure window
--------------
The window is the time for the health score to fall to the Critical band
(below 40) if it keeps falling at its recent rate, using the faster of the
2-hour and 4-hour slopes. A site already below that line has a window of 0.
A site that is not falling gets the maximum window, meaning no crossing is
expected. Limitation: this is linear extrapolation. It assumes the rate of
decline stays constant, but real failures often accelerate (the acceleration
signal partly compensates) or stop abruptly, so the window is an estimate
with wide error, not a forecast.

Confidence
----------
Confidence combines three things: how well a straight line fits the 4-hour
health trend (R squared), how many of the five risk signals agree that risk
is meaningful, and how much of the expected history is actually present.
High confidence needs a clean trend, several signals pointing the same way,
and a full window of data.

Why a transparent model
-----------------------
A black-box classifier would need labeled failure history that is not
available here, would be hard for network operations teams to audit, and
could not say why a given site was flagged. This model uses weights and
thresholds that an engineer can read, argue about, and tune. Every
prediction lists the signals behind it, which is what an on-call engineer
needs to decide whether to dispatch a team.

Limitations
-----------
Weights are expert-set rather than learned, so they should be recalibrated
against real outage data when it exists. Linear trends can miss abrupt
failures that give no warning, which is the sudden-failure case.
"""

from collections import Counter

import numpy as np
import pandas as pd

from app.config import (
    ALARM_COUNT_CRITICAL,
    ALARM_COUNT_WARNING,
    CPU_CRITICAL,
    CPU_WARNING,
    HEALTH_SCORE_DEGRADED,
    HEALTH_SCORE_HEALTHY,
    HEALTH_SCORE_WARNING,
    LATENCY_CRITICAL_5G,
    LATENCY_CRITICAL_LTE,
    LATENCY_WARNING_5G,
    LATENCY_WARNING_LTE,
    MAX_RESULTS,
    MEMORY_CRITICAL,
    MEMORY_WARNING,
    PACKET_LOSS_CRITICAL,
    PACKET_LOSS_WARNING,
    PREDICTION_WINDOW_HOURS,
    SINR_CRITICAL,
    SINR_WARNING,
    SUBSCORE_GOOD,
    THROUGHPUT_CRITICAL_5G_MBPS,
    THROUGHPUT_CRITICAL_LTE_MBPS,
    THROUGHPUT_WARNING_5G_MBPS,
    THROUGHPUT_WARNING_LTE_MBPS,
)
from app.health_scorer import COMPONENT_COLUMNS, score_all_sites
from app.models import Anomaly, Prediction

# --- Risk signal weights (sum to 1.0) --------------------------------------

# Trend is the most direct evidence of movement toward failure
WEIGHT_HEALTH_TREND = 0.30
# Current level matters, but a low stable score is less urgent than a falling one
WEIGHT_CURRENT_HEALTH = 0.20
# Several KPIs near their limits shows margin running out on many fronts
WEIGHT_KPI_PROXIMITY = 0.20
# Recent anomalies catch jumps that smooth trends miss, but are noisier
WEIGHT_ANOMALY_DENSITY = 0.15
# Acceleration shortens the time left to react, so it matters at a modest weight
WEIGHT_ACCELERATION = 0.15
SIGNAL_WEIGHTS = {
    "health_trend": WEIGHT_HEALTH_TREND,
    "current_health": WEIGHT_CURRENT_HEALTH,
    "kpi_proximity": WEIGHT_KPI_PROXIMITY,
    "anomaly_density": WEIGHT_ANOMALY_DENSITY,
    "acceleration": WEIGHT_ACCELERATION,
}
WEIGHT_SUM_TOLERANCE = 1e-9
if abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) > WEIGHT_SUM_TOLERANCE:
    raise ValueError("Risk signal weights must sum to 1.0")

# --- Windows and data requirements -----------------------------------------

# Four hours is long enough to see a trend, short enough to reflect now
TREND_WINDOW_HOURS = 4
# Two hours is the recent half of the trend window, used for acceleration
SHORT_WINDOW_HOURS = 2
# Anomalies older than the short window are treated as already absorbed
ANOMALY_WINDOW_HOURS = 2
# Samples per hour at the 15-minute telemetry cadence
SAMPLES_PER_HOUR = 4
# A line through fewer than 4 points says little about direction
MIN_TREND_POINTS = 4
FULL_WINDOW_POINTS = TREND_WINDOW_HOURS * SAMPLES_PER_HOUR

# --- Trend classification --------------------------------------------------

# A change under 5% of the average value over the window counts as stable
STABLE_CHANGE_FRACTION = 0.05
# Guards the division when the average value is near zero
MIN_REFERENCE_VALUE = 1.0

# --- Signal normalization --------------------------------------------------

# Health score decline per hour that counts as medium and high risk
DECLINE_MEDIUM_PER_HOUR = 2.0
DECLINE_HIGH_PER_HOUR = 5.0
RISK_NONE = 0.0
RISK_MEDIUM = 0.5
RISK_HIGH = 1.0
# A KPI within 15% of its critical threshold is treated as near critical
PROXIMITY_MARGIN = 0.15
# Three KPIs near critical at once is a multi-front problem, so risk saturates
PROXIMITY_SATURATION_COUNT = 3
# Ten anomalies in two hours is dense across several KPIs for one site
ANOMALY_SATURATION_COUNT = 10
# Acceleration of 5 points/hour (the high-decline rate) saturates the signal
ACCELERATION_SATURATION = 5.0
# A user count that doubles is a full congestion signal
USER_SURGE_SATURATION = 1.0

# --- Failure window --------------------------------------------------------

# Health score below this value is the Critical band
FAILURE_SCORE_LEVEL = HEALTH_SCORE_DEGRADED
# Declines slower than 0.5 points/hour would take longer than the cap anyway
MIN_DECLINE_FOR_EXTRAPOLATION = 0.5
# Cap at one full day: beyond that a linear estimate is not meaningful
MAX_WINDOW_HOURS = 24.0

# --- Confidence ------------------------------------------------------------

# Agreement between signals is the strongest sign the call is robust
CONFIDENCE_WEIGHT_AGREEMENT = 0.5
# Trend fit quality adds certainty when the decline is steady
CONFIDENCE_WEIGHT_FIT = 0.3
# Missing history makes every signal less reliable
CONFIDENCE_WEIGHT_DATA = 0.2
# A signal counts as agreeing when it reaches at least the medium risk level
AGREEMENT_MIN_CONTRIBUTION = RISK_MEDIUM

# --- Reporting -------------------------------------------------------------

# Only sites above this risk are reported, which filters out noise-level risk
RISK_REPORT_THRESHOLD = 0.3
# Risk level boundaries for the summary
RISK_LEVEL_HIGH = 0.7
RISK_LEVEL_MEDIUM = 0.5
# A KPI is listed as supporting evidence when its stress passes this value
SUPPORTING_STRESS_MIN = 0.2
MAX_SUPPORTING_KPIS = 5
ROUND_DECIMALS = 3
# An estimated window at or under this many hours is urgent
URGENT_WINDOW_HOURS = 1.0

# --- KPI and failure type mapping ------------------------------------------

KPI_COMPONENT = dict(
    zip(
        [
            "cpu_utilization",
            "memory_utilization",
            "latency_ms",
            "packet_loss_pct",
            "throughput_mbps",
            "signal_quality_sinr",
            "alarm_count",
        ],
        ["cpu_score", "memory_score", "latency_score", "packet_loss_score",
         "throughput_score", "sinr_score", "alarm_score"],
    )
)
USER_KPI = "connected_users"
LOWER_IS_WORSE = {"throughput_mbps", "signal_quality_sinr"}

# Specificity weight of each KPI for each failure type: how uniquely the KPI
# points to that failure mode. Symptoms that rise in almost every fault
# (alarms, latency, packet loss) get 0.5 so root-cause KPIs win ties.
FAILURE_TYPE_SPECIFICITY = {
    "congestion": {USER_KPI: 1.0, "latency_ms": 0.5, "packet_loss_pct": 0.5},
    "capacity_exhaustion": {"cpu_utilization": 0.8, "memory_utilization": 0.8},
    "signal_degradation": {"signal_quality_sinr": 1.0, "throughput_mbps": 0.5},
    "hardware_degradation": {"alarm_count": 0.5, "packet_loss_pct": 0.5},
}
RECOMMENDED_ACTIONS = {
    "congestion": "Rebalance traffic to neighboring cells and review capacity for this site.",
    "capacity_exhaustion": "Check CPU and memory consumers, restart runaway processes, "
                           "and plan a capacity upgrade.",
    "signal_degradation": "Inspect antennas, feeders, and interference sources; "
                          "schedule a radio optimization.",
    "hardware_degradation": "Dispatch a field engineer to inspect hardware and review "
                            "active alarms.",
}
URGENT_ACTION_PREFIX = "URGENT: "

RISK_LEVELS = ("high", "medium", "low")


# --- Trend -----------------------------------------------------------------

def calculate_trend(values: list[float], timestamps: list, n_points: int | None = None) -> dict:
    """Fit a line to the latest points and describe the trend.

    Returns slope (units per sample), r_squared, direction, and
    rate_of_change_per_hour. Pass n_points to fit only the most recent points.
    """
    values = list(values)
    timestamps = list(timestamps)
    if n_points is not None:
        values, timestamps = values[-n_points:], timestamps[-n_points:]

    flat = {"slope": 0.0, "r_squared": 0.0, "direction": "stable",
            "rate_of_change_per_hour": 0.0}
    if len(values) < MIN_TREND_POINTS:
        return flat

    times = pd.to_datetime(pd.Series(timestamps))
    hours = ((times - times.iloc[0]).dt.total_seconds() / 3600.0).to_numpy()
    y = np.asarray(values, dtype=float)
    span = float(hours[-1] - hours[0])
    if span <= 0:
        return flat

    rate_per_hour, intercept = np.polyfit(hours, y, 1)
    residual = float(np.sum((y - (rate_per_hour * hours + intercept)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0

    reference = max(abs(float(y.mean())), MIN_REFERENCE_VALUE)
    relative_change = abs(rate_per_hour * span) / reference
    if relative_change < STABLE_CHANGE_FRACTION:
        direction = "stable"
    else:
        direction = "increasing" if rate_per_hour > 0 else "decreasing"

    return {
        "slope": float(rate_per_hour * span / (len(y) - 1)),
        "r_squared": float(max(0.0, r_squared)),
        "direction": direction,
        "rate_of_change_per_hour": float(rate_per_hour),
    }


# --- Helpers ---------------------------------------------------------------

def _critical_thresholds(network_type: str) -> dict[str, tuple[float, float]]:
    """Return (warning, critical) per KPI for one network technology."""
    if network_type == "5G":
        latency = (LATENCY_WARNING_5G, LATENCY_CRITICAL_5G)
        throughput = (THROUGHPUT_WARNING_5G_MBPS, THROUGHPUT_CRITICAL_5G_MBPS)
    elif network_type == "LTE":
        latency = (LATENCY_WARNING_LTE, LATENCY_CRITICAL_LTE)
        throughput = (THROUGHPUT_WARNING_LTE_MBPS, THROUGHPUT_CRITICAL_LTE_MBPS)
    else:
        raise ValueError(f"Unknown network_type: {network_type}")
    return {
        "cpu_utilization": (CPU_WARNING, CPU_CRITICAL),
        "memory_utilization": (MEMORY_WARNING, MEMORY_CRITICAL),
        "latency_ms": latency,
        "packet_loss_pct": (PACKET_LOSS_WARNING, PACKET_LOSS_CRITICAL),
        "throughput_mbps": throughput,
        "signal_quality_sinr": (SINR_WARNING, SINR_CRITICAL),
        "alarm_count": (ALARM_COUNT_WARNING, ALARM_COUNT_CRITICAL),
    }


def _is_near_critical(kpi: str, value: float, warning: float, critical: float) -> bool:
    """True when a KPI is within the margin of, or beyond, its critical value."""
    # A zero critical value (SINR) has no proportional margin, so use warning
    scale = abs(critical) if critical != 0 else abs(warning)
    margin = PROXIMITY_MARGIN * scale
    if kpi in LOWER_IS_WORSE:
        return value <= critical + margin
    return value >= critical - margin


def _align_scores(site: pd.DataFrame, original: pd.DataFrame, scores: pd.Series) -> pd.Series:
    """Align health scores to the time-sorted site rows."""
    aligned = scores.reindex(site.index)
    if aligned.isna().any():
        if len(scores) != len(original):
            raise ValueError("health_scores must align with site_df by index or position")
        aligned = pd.Series(scores.to_numpy(), index=original.index).reindex(site.index)
    return aligned.astype(float)


def _decline_rate(trend: dict) -> float:
    """Return decline in points per hour (positive when the score is falling)."""
    return -trend["rate_of_change_per_hour"]


def _failure_type_scores(stress: dict[str, float]) -> dict[str, float]:
    """Largest specificity-weighted stress for each failure type."""
    return {
        failure_type: max(weight * stress[kpi] for kpi, weight in kpis.items())
        for failure_type, kpis in FAILURE_TYPE_SPECIFICITY.items()
    }


def _build_explanation(
    failure_type: str,
    score: float,
    decline: float,
    near_critical: list[str],
    recent_anomalies: int,
    accelerating: bool,
    window: float,
) -> str:
    """Compose a short human-readable justification."""
    parts = [f"Health score is {score:.0f}"]
    if decline > 0:
        parts[0] += f" and falling {decline:.1f} points/hour"
    if near_critical:
        parts.append(f"{len(near_critical)} KPI(s) near critical ({', '.join(near_critical)})")
    if recent_anomalies:
        parts.append(f"{recent_anomalies} anomalies in the last {ANOMALY_WINDOW_HOURS} hours")
    if accelerating:
        parts.append("the decline is accelerating")
    timing = (
        "already in the critical band"
        if window <= 0
        else f"estimated to reach the critical band in {window:.1f} hours"
    )
    return f"Likely {failure_type.replace('_', ' ')}: " + "; ".join(parts) + f"; {timing}."


# --- Prediction ------------------------------------------------------------

def predict_failure(
    site_id: str,
    site_df: pd.DataFrame,
    anomalies: list[Anomaly],
    health_scores: pd.Series,
) -> Prediction | None:
    """Predict failure risk for one site; None when there is too little history."""
    if len(site_df) < MIN_TREND_POINTS:
        return None

    site = site_df.sort_values("timestamp")
    if any(column not in site.columns for column in COMPONENT_COLUMNS):
        site = score_all_sites(site)
    scores = _align_scores(site, site_df, health_scores)

    network_type = str(site["network_type"].iloc[-1])
    latest = site.iloc[-1]
    last_time = site["timestamp"].iloc[-1]

    long_mask = site["timestamp"] >= last_time - pd.Timedelta(hours=TREND_WINDOW_HOURS)
    short_mask = site["timestamp"] >= last_time - pd.Timedelta(hours=SHORT_WINDOW_HOURS)
    long_times = site.loc[long_mask, "timestamp"].tolist()
    short_times = site.loc[short_mask, "timestamp"].tolist()

    trend_long = calculate_trend(scores[long_mask].tolist(), long_times)
    trend_short = calculate_trend(scores[short_mask].tolist(), short_times)
    current_score = float(scores.iloc[-1])

    # a) Health score trend
    decline_long = _decline_rate(trend_long)
    decline_short = _decline_rate(trend_short)
    trend_risk = float(np.interp(
        decline_long,
        [0.0, DECLINE_MEDIUM_PER_HOUR, DECLINE_HIGH_PER_HOUR],
        [RISK_NONE, RISK_MEDIUM, RISK_HIGH],
    ))

    # b) Current health score
    current_risk = float(np.interp(
        current_score,
        [HEALTH_SCORE_DEGRADED, HEALTH_SCORE_WARNING, HEALTH_SCORE_HEALTHY],
        [RISK_HIGH, RISK_MEDIUM, RISK_NONE],
    ))

    # c) KPI threshold proximity
    thresholds = _critical_thresholds(network_type)
    near_critical = [
        kpi for kpi, (warning, critical) in thresholds.items()
        if _is_near_critical(kpi, float(latest[kpi]), warning, critical)
    ]
    proximity_risk = min(len(near_critical) / PROXIMITY_SATURATION_COUNT, RISK_HIGH)

    # d) Anomaly density
    anomaly_cutoff = last_time - pd.Timedelta(hours=ANOMALY_WINDOW_HOURS)
    recent_anomalies = [
        a for a in anomalies
        if a.site_id == site_id and pd.Timestamp(a.timestamp) >= anomaly_cutoff
    ]
    density_risk = min(len(recent_anomalies) / ANOMALY_SATURATION_COUNT, RISK_HIGH)

    # e) Trend acceleration: only counts when the recent decline is steeper
    acceleration = max(0.0, decline_short - max(decline_long, 0.0)) if decline_short > 0 else 0.0
    acceleration_risk = min(acceleration / ACCELERATION_SATURATION, RISK_HIGH)

    contributions = {
        "health_trend": trend_risk,
        "current_health": current_risk,
        "kpi_proximity": proximity_risk,
        "anomaly_density": density_risk,
        "acceleration": acceleration_risk,
    }
    risk_score = sum(SIGNAL_WEIGHTS[name] * value for name, value in contributions.items())
    risk_score = float(min(max(risk_score, 0.0), 1.0))

    # Failure type from per-KPI stress (1 - sub-score / 100)
    stress = {
        kpi: 1.0 - float(latest[component]) / SUBSCORE_GOOD
        for kpi, component in KPI_COMPONENT.items()
    }
    history = site.loc[~long_mask, USER_KPI]
    user_baseline = float(history.mean()) if len(history) else float(site[USER_KPI].iloc[0])
    surge = float(latest[USER_KPI]) / max(user_baseline, MIN_REFERENCE_VALUE) - 1.0
    stress[USER_KPI] = float(np.clip(surge / USER_SURGE_SATURATION, 0.0, 1.0))
    type_scores = _failure_type_scores(stress)
    failure_type = max(type_scores, key=type_scores.get)

    # Failure window by linear extrapolation of the faster recent decline
    decline = max(decline_long, decline_short)
    if current_score < FAILURE_SCORE_LEVEL:
        window = 0.0
    elif decline >= MIN_DECLINE_FOR_EXTRAPOLATION:
        window = min((current_score - FAILURE_SCORE_LEVEL) / decline, MAX_WINDOW_HOURS)
    else:
        window = MAX_WINDOW_HOURS

    # Confidence from fit quality, signal agreement, and data completeness
    agreement = sum(v >= AGREEMENT_MIN_CONTRIBUTION for v in contributions.values()) / len(
        contributions
    )
    completeness = min(int(long_mask.sum()) / FULL_WINDOW_POINTS, 1.0)
    confidence = (
        CONFIDENCE_WEIGHT_AGREEMENT * agreement
        + CONFIDENCE_WEIGHT_FIT * trend_long["r_squared"]
        + CONFIDENCE_WEIGHT_DATA * completeness
    )

    # Supporting KPIs ordered by stress
    supporting = []
    ranked = sorted(stress, key=stress.get, reverse=True)
    for kpi in ranked[:MAX_SUPPORTING_KPIS]:
        if stress[kpi] < SUPPORTING_STRESS_MIN:
            break
        kpi_trend = calculate_trend(site.loc[long_mask, kpi].tolist(), long_times)
        supporting.append({
            "kpi_name": kpi,
            "current_value": round(float(latest[kpi]), ROUND_DECIMALS),
            "trend": kpi_trend["direction"],
            "rate_per_hour": round(kpi_trend["rate_of_change_per_hour"], ROUND_DECIMALS),
            "threshold": thresholds[kpi][1] if kpi in thresholds else None,
        })

    explanation = _build_explanation(
        failure_type, current_score, decline_long,
        near_critical, len(recent_anomalies), acceleration_risk > 0, window,
    )
    action = RECOMMENDED_ACTIONS[failure_type]
    if window <= URGENT_WINDOW_HOURS and risk_score > RISK_REPORT_THRESHOLD:
        action = URGENT_ACTION_PREFIX + action

    return Prediction(
        site_id=site_id,
        predicted_failure_type=failure_type,
        risk_score=round(risk_score, ROUND_DECIMALS),
        confidence=round(float(min(max(confidence, 0.0), 1.0)), ROUND_DECIMALS),
        estimated_failure_window_hours=round(float(window), ROUND_DECIMALS),
        supporting_kpis=supporting,
        explanation=explanation,
        recommended_action=action,
    )


def predict_all_sites(scored_df: pd.DataFrame, anomalies: list[Anomaly]) -> list[Prediction]:
    """Predict for every site and return those above the risk threshold."""
    if "health_score" not in scored_df.columns:
        scored_df = score_all_sites(scored_df)

    anomalies_by_site: dict[str, list[Anomaly]] = {}
    for anomaly in anomalies:
        anomalies_by_site.setdefault(anomaly.site_id, []).append(anomaly)

    predictions = []
    for site_id, site_df in scored_df.groupby("site_id"):
        prediction = predict_failure(
            str(site_id), site_df, anomalies_by_site.get(str(site_id), []),
            site_df["health_score"],
        )
        if prediction is not None and prediction.risk_score > RISK_REPORT_THRESHOLD:
            predictions.append(prediction)
    return sorted(predictions, key=lambda p: p.risk_score, reverse=True)


def get_prediction_summary(predictions: list[Prediction]) -> dict:
    """Summarize predictions by failure type, risk level, and top sites."""
    by_type = Counter(p.predicted_failure_type for p in predictions)

    def level(risk: float) -> str:
        if risk >= RISK_LEVEL_HIGH:
            return "high"
        if risk >= RISK_LEVEL_MEDIUM:
            return "medium"
        return "low"

    by_level = Counter(level(p.risk_score) for p in predictions)
    ranked = sorted(predictions, key=lambda p: p.risk_score, reverse=True)
    return {
        "total_at_risk": len(predictions),
        "by_failure_type": dict(by_type.most_common()),
        "by_risk_level": {name: by_level.get(name, 0) for name in RISK_LEVELS},
        "top_10_riskiest": [
            {
                "site_id": p.site_id,
                "risk_score": p.risk_score,
                "predicted_failure_type": p.predicted_failure_type,
                "confidence": p.confidence,
                "estimated_failure_window_hours": p.estimated_failure_window_hours,
            }
            for p in ranked[:MAX_RESULTS]
        ],
    }
