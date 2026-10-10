"""Pydantic data models for the network health monitoring system.

Defines the shared schemas for raw site metrics, health scores,
detected anomalies, failure predictions, and AI-generated explanations.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


MetricName = Literal[
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
ComponentName = Literal[
    "cpu_score",
    "memory_score",
    "latency_score",
    "packet_loss_score",
    "throughput_score",
    "sinr_score",
    "handover_score",
    "alarm_score",
]


class SiteMetric(BaseModel):
    """A single KPI snapshot for one cell site at one point in time."""

    site_id: str
    timestamp: datetime
    network_type: Literal["5G", "LTE"]
    cpu_utilization: float = Field(ge=0, le=100)
    memory_utilization: float = Field(ge=0, le=100)
    latency_ms: float = Field(ge=0)
    packet_loss_pct: float = Field(ge=0, le=100)
    throughput_mbps: float = Field(ge=0)
    connected_users: int = Field(ge=0)
    signal_quality_sinr: float
    handover_failure_rate: float = Field(ge=0, le=100)
    alarm_count: int = Field(ge=0)
    site_status: Literal["Healthy", "Degraded", "Critical", "Failed"]


class HealthScore(BaseModel):
    """Composite health score for a site with per-KPI sub-scores."""

    site_id: str
    timestamp: datetime
    score: float = Field(ge=0, le=100)
    status: Literal["Healthy", "Warning", "Degraded", "Critical"]
    component_scores: dict[ComponentName, float]


class Anomaly(BaseModel):
    """A KPI reading that falls outside its expected range."""

    site_id: str
    timestamp: datetime
    metric_name: MetricName
    current_value: float
    expected_range_low: float
    expected_range_high: float
    severity: Literal["warning", "critical"]
    description: str


class Prediction(BaseModel):
    """A forecast of a likely site failure within a time window."""

    site_id: str
    predicted_failure_type: str
    risk_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    estimated_failure_window_hours: float = Field(ge=0)
    supporting_kpis: list[dict[str, Any]]
    explanation: str
    recommended_action: str


class AIExplanation(BaseModel):
    """A language-model explanation of a site's condition and next steps."""

    site_id: str
    summary: str
    evidence: list[str]
    recommended_actions: list[str]
    priority: str
