"""Unit and integration tests for the network health monitoring pipeline.

Run from the project root with: python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# Make `app` importable no matter where pytest is launched from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.anomaly_detector import detect_statistical_anomalies  # noqa: E402
from app.config import HOURS_OF_DATA, SITE_COUNT  # noqa: E402
from app.data_generator import (  # noqa: E402
    COLUMNS,
    INTERVAL_MINUTES,
    POINTS_PER_HOUR,
    SCENARIO_COUNTS,
    generate_dataset,
    generate_site_telemetry,
)
from app.health_scorer import (  # noqa: E402
    KPI_WEIGHTS,
    calculate_health_score,
    score_all_sites,
)
from app.models import (  # noqa: E402
    AIExplanation,
    Anomaly,
    HealthScore,
    Prediction,
    SiteMetric,
)
from app.predictor import (  # noqa: E402
    RISK_REPORT_THRESHOLD,
    SIGNAL_WEIGHTS,
    predict_failure,
)

# --- Test constants --------------------------------------------------------

TEST_SEED = 7
TEST_END_TIME = pd.Timestamp("2025-01-02 00:00:00")
EXPECTED_ROWS = SITE_COUNT * HOURS_OF_DATA * POINTS_PER_HOUR
NON_NEGATIVE_COLUMNS = [
    "cpu_utilization", "memory_utilization", "latency_ms", "packet_loss_pct",
    "throughput_mbps", "connected_users", "alarm_count",
]
HEALTHY_SCORE_FLOOR = 80
CRITICAL_SCORE_CEILING = 40
HIGH_RISK_FLOOR = 0.5
SPIKE_LATENCY_MS = 500.0
SPIKE_ROW = 60
SITE_ID = "TEST_5G_RAN_0001"


# --- Fixtures --------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """The full 50-site synthetic dataset."""
    return generate_dataset()


@pytest.fixture(scope="module")
def timestamps() -> pd.DatetimeIndex:
    """A fixed 24-hour, 15-minute timestamp grid."""
    return pd.date_range(
        end=TEST_END_TIME, periods=HOURS_OF_DATA * POINTS_PER_HOUR,
        freq=f"{INTERVAL_MINUTES}min",
    )


@pytest.fixture
def make_site(timestamps):
    """Factory that builds a scored single-site DataFrame for a scenario."""

    def _make(scenario: str, network_type: str = "5G", seed: int = TEST_SEED) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        records = generate_site_telemetry(SITE_ID, network_type, scenario, timestamps, rng)
        return score_all_sites(pd.DataFrame.from_records(records, columns=COLUMNS))

    return _make


@pytest.fixture
def good_row() -> pd.Series:
    """A 5G reading with every KPI comfortably healthy."""
    return pd.Series({
        "site_id": SITE_ID, "timestamp": TEST_END_TIME, "network_type": "5G",
        "cpu_utilization": 40.0, "memory_utilization": 45.0, "latency_ms": 8.0,
        "packet_loss_pct": 0.1, "throughput_mbps": 1000.0, "connected_users": 120,
        "signal_quality_sinr": 20.0, "alarm_count": 0, "site_status": "Healthy",
    })


@pytest.fixture
def bad_row(good_row) -> pd.Series:
    """The same site with every KPI far beyond its critical threshold."""
    row = good_row.copy()
    row.update({
        "cpu_utilization": 99.0, "memory_utilization": 98.0, "latency_ms": 300.0,
        "packet_loss_pct": 15.0, "throughput_mbps": 20.0, "signal_quality_sinr": -10.0,
        "alarm_count": 10, "site_status": "Failed",
    })
    return row


# --- Data generation -------------------------------------------------------

def test_data_generation(dataset):
    """Dataset has the right shape, sites, scenarios, and no negative values."""
    assert dataset.shape == (EXPECTED_ROWS, len(COLUMNS))
    assert list(dataset.columns) == COLUMNS
    assert dataset["site_id"].nunique() == SITE_COUNT
    assert dataset.groupby("network_type")["site_id"].nunique().to_dict() == {"5G": 30, "LTE": 20}

    scenarios = pd.Series(dataset.attrs["site_scenarios"]).value_counts().to_dict()
    assert scenarios == SCENARIO_COUNTS

    assert not dataset.isna().any().any()
    for column in NON_NEGATIVE_COLUMNS:
        assert (dataset[column] >= 0).all(), f"negative values in {column}"
    assert (dataset[["cpu_utilization", "memory_utilization", "packet_loss_pct"]] <= 100).all().all()


def test_data_generation_is_reproducible(timestamps):
    """The same seed produces identical KPI values."""
    first = generate_site_telemetry(SITE_ID, "5G", "DEGRADING", timestamps,
                                    np.random.default_rng(TEST_SEED))
    second = generate_site_telemetry(SITE_ID, "5G", "DEGRADING", timestamps,
                                     np.random.default_rng(TEST_SEED))
    assert first == second


# --- Health scoring --------------------------------------------------------

def test_health_scoring_healthy_site(good_row):
    """A row with all-good KPIs scores above 80 and is Healthy."""
    result = calculate_health_score(good_row, "5G")
    assert result.score > HEALTHY_SCORE_FLOOR
    assert result.status == "Healthy"


def test_health_scoring_critical_site(bad_row):
    """A row with all-bad KPIs scores below 40 and is Critical."""
    result = calculate_health_score(bad_row, "5G")
    assert result.score < CRITICAL_SCORE_CEILING
    assert result.status == "Critical"
    assert set(result.component_scores) == set(KPI_WEIGHTS)


def test_health_scoring_weights_sum_to_one():
    """Health scorer and predictor weights each sum to exactly 1.0."""
    assert sum(KPI_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)


# --- Anomaly detection -----------------------------------------------------

def test_anomaly_detection_finds_spike(make_site):
    """A 500 ms latency reading injected into normal data is flagged critical."""
    site = make_site("HEALTHY")
    spike_time = site["timestamp"].iloc[SPIKE_ROW]
    site.loc[site.index[SPIKE_ROW], "latency_ms"] = SPIKE_LATENCY_MS

    anomalies = detect_statistical_anomalies(site, SITE_ID)
    hits = [a for a in anomalies if a.metric_name == "latency_ms" and a.timestamp == spike_time]
    assert hits, "injected latency spike was not detected"
    assert hits[0].severity == "critical"
    assert hits[0].current_value == SPIKE_LATENCY_MS


def test_anomaly_detection_quiet_on_healthy_data(make_site):
    """Normal noise alone produces no critical anomalies."""
    site = make_site("HEALTHY")
    anomalies = detect_statistical_anomalies(site, SITE_ID)
    assert not [a for a in anomalies if a.severity == "critical"]


# --- Prediction ------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["DEGRADING", "FAILING", "SUDDEN_FAILURE"])
def test_prediction_failing_site(make_site, scenario):
    """A site on a failure trajectory gets a risk score above 0.5."""
    site = make_site(scenario)
    anomalies = detect_statistical_anomalies(site, SITE_ID)
    prediction = predict_failure(SITE_ID, site, anomalies, site["health_score"])

    assert prediction is not None
    assert prediction.risk_score > HIGH_RISK_FLOOR
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.supporting_kpis


@pytest.mark.parametrize("network_type", ["5G", "LTE"])
def test_prediction_healthy_site(make_site, network_type):
    """A healthy site has no prediction or a risk below the reporting threshold."""
    site = make_site("HEALTHY", network_type)
    anomalies = detect_statistical_anomalies(site, SITE_ID)
    prediction = predict_failure(SITE_ID, site, anomalies, site["health_score"])

    assert prediction is None or prediction.risk_score < RISK_REPORT_THRESHOLD


# --- Models ----------------------------------------------------------------

def test_models_validation():
    """Pydantic models accept valid data and reject invalid data."""
    metric = {
        "site_id": SITE_ID, "timestamp": TEST_END_TIME, "network_type": "5G",
        "cpu_utilization": 40.0, "memory_utilization": 45.0, "latency_ms": 8.0,
        "packet_loss_pct": 0.1, "throughput_mbps": 1000.0, "connected_users": 120,
        "signal_quality_sinr": 20.0, "alarm_count": 0, "site_status": "Healthy",
    }
    assert SiteMetric(**metric).network_type == "5G"
    for field, bad_value in [
        ("network_type", "6G"), ("cpu_utilization", 150.0), ("latency_ms", -1.0),
        ("site_status", "Unknown"), ("connected_users", -5),
    ]:
        with pytest.raises(ValidationError):
            SiteMetric(**{**metric, field: bad_value})

    health = {"site_id": SITE_ID, "timestamp": TEST_END_TIME, "score": 85,
              "status": "Healthy", "component_scores": {"cpu_score": 90.0}}
    assert HealthScore(**health).score == 85
    with pytest.raises(ValidationError):
        HealthScore(**{**health, "score": 120})
    with pytest.raises(ValidationError):
        HealthScore(**{**health, "status": "Fine"})

    anomaly = {"site_id": SITE_ID, "timestamp": TEST_END_TIME, "metric_name": "latency_ms",
               "current_value": 500.0, "expected_range_low": 5.0, "expected_range_high": 15.0,
               "severity": "critical", "description": "spike"}
    assert Anomaly(**anomaly).severity == "critical"
    with pytest.raises(ValidationError):
        Anomaly(**{**anomaly, "severity": "info"})

    prediction = {"site_id": SITE_ID, "predicted_failure_type": "congestion",
                  "risk_score": 0.8, "confidence": 0.7, "estimated_failure_window_hours": 2.0,
                  "supporting_kpis": [{"kpi_name": "latency_ms"}], "explanation": "x",
                  "recommended_action": "y"}
    assert Prediction(**prediction).risk_score == 0.8
    with pytest.raises(ValidationError):
        Prediction(**{**prediction, "risk_score": 1.5})
    with pytest.raises(ValidationError):
        Prediction(**{**prediction, "confidence": -0.1})

    explanation = {"site_id": SITE_ID, "summary": "s", "evidence": ["e"],
                   "recommended_actions": ["a"], "priority": "P1-Critical"}
    assert AIExplanation(**explanation).priority == "P1-Critical"
    with pytest.raises(ValidationError):
        AIExplanation(site_id=SITE_ID, summary="s")
