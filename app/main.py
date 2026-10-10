"""FastAPI backend for the network health monitoring system.

On startup the service loads the telemetry dataset once and runs the full
pipeline: health scoring, anomaly detection, and failure prediction. Results
are held in memory on app.state, so every read endpoint is a cheap lookup.
Only the three AI endpoints call OpenAI.

Run with: uvicorn app.main:app --reload
"""

import functools
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ai_agent import (
    answer_question,
    create_openai_client,
    explain_site_risk,
    generate_network_summary,
)
from app.anomaly_detector import (
    KPI_COLUMNS,
    detect_all_anomalies,
    get_anomaly_summary,
)
from app.config import DATA_DIR, FALLBACK_MESSAGE, OPENAI_API_KEY
from app.data_generator import load_dataset
from app.health_scorer import (
    COMPONENT_COLUMNS,
    get_current_health_summary,
    score_all_sites,
)
from app.models import AIExplanation, Anomaly, Prediction
from app.predictor import (
    get_prediction_summary,
    predict_all_sites,
    predict_failure,
)

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

# --- Constants -------------------------------------------------------------

DATA_FILENAME = "network_telemetry.csv"
DATA_FILE = DATA_DIR / DATA_FILENAME
HEALTH_STATUSES = ("Healthy", "Warning", "Degraded", "Critical")
# Critical anomalies sort before warnings
SEVERITY_ORDER = {"critical": 0, "warning": 1}
# The dataset spans 24 hours, so a longer history window adds nothing
MAX_HISTORY_HOURS = 24
# Largest page of anomalies one request may return
MAX_ANOMALY_LIMIT = 5000
MAX_QUESTION_LENGTH = 1000
INTERNAL_ERROR_MESSAGE = "Internal server error"


# --- Response and request schemas ------------------------------------------

class HealthCheckResponse(BaseModel):
    """Service health and dependency status."""

    status: str
    data_loaded: bool
    model_ready: bool
    openai_connected: bool


class SiteSummary(BaseModel):
    """One row in the site list: latest health and status."""

    site_id: str
    network_type: str
    timestamp: datetime
    health_score: int
    health_status: str
    site_status: str
    risk_score: float | None = None


class SiteDetail(BaseModel):
    """Everything known about one site at its latest timestamp."""

    site_id: str
    network_type: str
    timestamp: datetime
    health_score: int
    health_status: str
    site_status: str
    latest_metrics: dict[str, float]
    component_scores: dict[str, float]
    ml_anomaly: bool
    ml_anomaly_score: float
    anomalies: list[Anomaly]
    prediction: Prediction | None = None


class HistoryPoint(BaseModel):
    """One timestamp of a site's metrics and health, for charting."""

    timestamp: datetime
    health_score: int
    health_status: str
    site_status: str
    is_anomaly: bool
    anomaly_score: float
    metrics: dict[str, float]


class SiteHistory(BaseModel):
    """Time series for one site."""

    site_id: str
    network_type: str
    hours: int | None
    points: list[HistoryPoint]


class WorstSite(BaseModel):
    """A low-scoring site in the network summary."""

    site_id: str
    network_type: str
    health_score: int
    health_status: str
    weakest_kpi: str


class HealthSummaryResponse(BaseModel):
    """Network-wide health counts and averages."""

    total_sites: int
    healthy_count: int
    warning_count: int
    degraded_count: int
    critical_count: int
    average_score: float
    worst_sites: list[WorstSite]


class ExplainRequest(BaseModel):
    """Request body for /api/explain."""

    site_id: str = Field(min_length=1)


class AskRequest(BaseModel):
    """Request body for /api/ask."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class AskResponse(BaseModel):
    """Answer to a natural language question."""

    answer: str


class NetworkSummaryResponse(BaseModel):
    """Executive summary of network health."""

    summary: str


# --- Error handling --------------------------------------------------------

class ServiceNotReadyError(Exception):
    """Raised when a request arrives before startup data is available."""


def handle_errors(func: Callable) -> Callable:
    """Map exceptions to HTTP status codes and log unexpected failures."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except ServiceNotReadyError as error:
            logger.error("Service not ready: %s", error)
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            logger.warning("Bad request in %s: %s", func.__name__, error)
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            logger.exception("Unhandled error in %s", func.__name__)
            raise HTTPException(status_code=500, detail=INTERNAL_ERROR_MESSAGE) from error

    return wrapper


# --- Startup ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Validate prerequisites, load data, and run the analysis pipeline."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env, set "
            "OPENAI_API_KEY to your key, and restart the server."
        )
    if not DATA_FILE.exists():
        raise RuntimeError(
            f"Dataset not found at {DATA_FILE}. Run `python generate_data.py` "
            "to create it, then restart the server."
        )

    raw = load_dataset(DATA_FILE)
    scored = score_all_sites(raw)
    augmented, anomalies = detect_all_anomalies(raw)
    scored["is_anomaly"] = augmented["is_anomaly"]
    scored["anomaly_score"] = augmented["anomaly_score"]
    predictions = predict_all_sites(scored, anomalies)
    health_summary = get_current_health_summary(scored)

    state = application.state
    state.scored_df = scored
    state.latest_df = (
        scored.sort_values("timestamp").groupby("site_id").tail(1).set_index("site_id")
    )
    state.anomalies = anomalies
    state.anomalies_by_site = {}
    for anomaly in anomalies:
        state.anomalies_by_site.setdefault(anomaly.site_id, []).append(anomaly)
    state.predictions = predictions
    state.predictions_by_site = {p.site_id: p for p in predictions}
    state.health_summary = health_summary
    state.openai_client = create_openai_client()
    state.network_summary_cache = None

    logger.info(
        "Loaded %d sites, %d anomalies detected, %d sites at risk",
        scored["site_id"].nunique(), len(anomalies), len(predictions),
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Network Health Monitoring API",
    description="Health scoring, anomaly detection, and failure prediction for 5G/LTE sites.",
    version="1.0.0",
    lifespan=lifespan,
)

# Open CORS is for local development only; restrict origins before deploying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---------------------------------------------------------------

def _ready() -> Any:
    """Return app.state, or raise when the pipeline has not run."""
    state = app.state
    if not hasattr(state, "scored_df"):
        raise ServiceNotReadyError("Data is not loaded yet; the service is starting up")
    return state


def _require_site(site_id: str) -> Any:
    """Return app.state for a known site or raise 404."""
    state = _ready()
    if site_id not in state.latest_df.index:
        raise HTTPException(status_code=404, detail=f"Site '{site_id}' not found")
    return state


def _site_rows(state: Any, site_id: str) -> pd.DataFrame:
    """All scored rows for one site, oldest first."""
    rows = state.scored_df[state.scored_df["site_id"] == site_id]
    return rows.sort_values("timestamp")


def _number(value: Any) -> float:
    """Convert a numpy scalar to a plain float."""
    return float(value)


def _prediction_for(state: Any, site_id: str) -> Prediction | None:
    """Stored prediction, or one computed on demand for a below-threshold site."""
    stored = state.predictions_by_site.get(site_id)
    if stored is not None:
        return stored
    rows = _site_rows(state, site_id)
    return predict_failure(
        site_id, rows, state.anomalies_by_site.get(site_id, []), rows["health_score"]
    )


def _require_ai_text(text: str) -> str:
    """Raise 503 when the AI agent returned its fallback message."""
    if text == FALLBACK_MESSAGE:
        raise HTTPException(
            status_code=503, detail="The AI service is currently unavailable. Try again later."
        )
    return text


# --- Endpoints -------------------------------------------------------------

@app.get("/health", response_model=HealthCheckResponse)
@handle_errors
def health_check() -> HealthCheckResponse:
    """Report service health and whether data, analysis, and OpenAI are ready."""
    state = app.state
    data_loaded = hasattr(state, "scored_df")
    model_ready = data_loaded and hasattr(state, "predictions")
    openai_connected = hasattr(state, "openai_client")
    healthy = data_loaded and model_ready and openai_connected
    return HealthCheckResponse(
        status="ok" if healthy else "degraded",
        data_loaded=data_loaded,
        model_ready=model_ready,
        openai_connected=openai_connected,
    )


@app.get("/api/sites", response_model=list[SiteSummary])
@handle_errors
def list_sites(
    status: str | None = Query(
        default=None, description="Filter by health status: Healthy, Warning, Degraded, Critical"
    ),
) -> list[SiteSummary]:
    """List all sites with latest health score and status, lowest score first."""
    state = _ready()
    latest = state.latest_df
    if status is not None:
        matches = [s for s in HEALTH_STATUSES if s.lower() == status.strip().lower()]
        if not matches:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Use one of: {', '.join(HEALTH_STATUSES)}",
            )
        latest = latest[latest["health_status"] == matches[0]]

    latest = latest.sort_values(["health_score"])
    summaries = []
    for site_id, row in latest.iterrows():
        prediction = state.predictions_by_site.get(site_id)
        summaries.append(
            SiteSummary(
                site_id=site_id,
                network_type=row["network_type"],
                timestamp=row["timestamp"],
                health_score=int(row["health_score"]),
                health_status=row["health_status"],
                site_status=row["site_status"],
                risk_score=prediction.risk_score if prediction else None,
            )
        )
    return summaries


@app.get("/api/sites/{site_id}", response_model=SiteDetail)
@handle_errors
def get_site(site_id: str) -> SiteDetail:
    """Return one site's latest metrics, health breakdown, anomalies, and prediction."""
    state = _require_site(site_id)
    row = state.latest_df.loc[site_id]
    return SiteDetail(
        site_id=site_id,
        network_type=row["network_type"],
        timestamp=row["timestamp"],
        health_score=int(row["health_score"]),
        health_status=row["health_status"],
        site_status=row["site_status"],
        latest_metrics={kpi: _number(row[kpi]) for kpi in KPI_COLUMNS},
        component_scores={name: _number(row[name]) for name in COMPONENT_COLUMNS},
        ml_anomaly=bool(row["is_anomaly"]),
        ml_anomaly_score=_number(row["anomaly_score"]),
        anomalies=state.anomalies_by_site.get(site_id, []),
        prediction=state.predictions_by_site.get(site_id),
    )


@app.get("/api/sites/{site_id}/history", response_model=SiteHistory)
@handle_errors
def get_site_history(
    site_id: str,
    hours: int | None = Query(
        default=None, ge=1, le=MAX_HISTORY_HOURS, description="Limit to the most recent N hours"
    ),
) -> SiteHistory:
    """Return a site's metric and health time series, optionally the last N hours."""
    state = _require_site(site_id)
    rows = _site_rows(state, site_id)
    if hours is not None:
        cutoff = rows["timestamp"].max() - pd.Timedelta(hours=hours)
        rows = rows[rows["timestamp"] > cutoff]

    points = [
        HistoryPoint(
            timestamp=row["timestamp"],
            health_score=int(row["health_score"]),
            health_status=row["health_status"],
            site_status=row["site_status"],
            is_anomaly=bool(row["is_anomaly"]),
            anomaly_score=_number(row["anomaly_score"]),
            metrics={kpi: _number(row[kpi]) for kpi in KPI_COLUMNS},
        )
        for _, row in rows.iterrows()
    ]
    return SiteHistory(
        site_id=site_id,
        network_type=str(state.latest_df.loc[site_id, "network_type"]),
        hours=hours,
        points=points,
    )


@app.get("/api/health-summary", response_model=HealthSummaryResponse)
@handle_errors
def get_health_summary() -> HealthSummaryResponse:
    """Return network-wide health counts, average score, and worst sites."""
    state = _ready()
    return HealthSummaryResponse(**state.health_summary)


@app.get("/api/anomalies", response_model=list[Anomaly])
@handle_errors
def list_anomalies(
    site_id: str | None = Query(default=None, description="Only anomalies for this site"),
    limit: int | None = Query(default=None, ge=1, le=MAX_ANOMALY_LIMIT),
) -> list[Anomaly]:
    """Return anomalies sorted by severity (critical first), then most recent."""
    state = _ready()
    if site_id is not None:
        if site_id not in state.latest_df.index:
            raise HTTPException(status_code=404, detail=f"Site '{site_id}' not found")
        anomalies = list(state.anomalies_by_site.get(site_id, []))
    else:
        anomalies = list(state.anomalies)

    # Newest first, then a stable sort by severity keeps that order within a level
    anomalies.sort(key=lambda a: a.timestamp, reverse=True)
    anomalies.sort(key=lambda a: SEVERITY_ORDER[a.severity])
    return anomalies[:limit] if limit else anomalies


@app.get("/api/predictions", response_model=list[Prediction])
@handle_errors
def list_predictions(
    min_risk: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum risk score"),
) -> list[Prediction]:
    """Return failure predictions by descending risk score, optionally filtered."""
    state = _ready()
    return [p for p in state.predictions if p.risk_score >= min_risk]


@app.get("/api/predictions/{site_id}", response_model=Prediction)
@handle_errors
def get_prediction(site_id: str) -> Prediction:
    """Return the detailed failure prediction for one site, including supporting KPIs."""
    state = _require_site(site_id)
    prediction = _prediction_for(state, site_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"No prediction available for '{site_id}'")
    return prediction


@app.post("/api/explain", response_model=AIExplanation)
@handle_errors
def explain(request: ExplainRequest) -> AIExplanation:
    """Generate an AI explanation of a site's failure risk."""
    state = _require_site(request.site_id)
    prediction = _prediction_for(state, request.site_id)
    if prediction is None:
        raise HTTPException(
            status_code=404, detail=f"No prediction available for '{request.site_id}'"
        )
    rows = _site_rows(state, request.site_id)
    explanation = explain_site_risk(
        state.openai_client,
        prediction,
        rows,
        float(rows["health_score"].iloc[-1]),
        state.anomalies_by_site.get(request.site_id, []),
    )
    _require_ai_text(explanation.summary)
    return explanation


@app.post("/api/ask", response_model=AskResponse)
@handle_errors
def ask(request: AskRequest) -> AskResponse:
    """Answer a natural language question using the current network data."""
    state = _ready()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be blank")
    answer = answer_question(
        state.openai_client, question, state.health_summary, state.predictions, state.scored_df
    )
    return AskResponse(answer=_require_ai_text(answer))


@app.get("/api/network-summary", response_model=NetworkSummaryResponse)
@handle_errors
def network_summary() -> NetworkSummaryResponse:
    """Return an AI-written executive summary of overall network health."""
    state = _ready()
    # The data is static after startup, so one successful summary is reused
    if state.network_summary_cache is None:
        text = generate_network_summary(
            state.openai_client,
            state.health_summary,
            get_prediction_summary(state.predictions),
            get_anomaly_summary(state.anomalies),
        )
        state.network_summary_cache = _require_ai_text(text)
    return NetworkSummaryResponse(summary=state.network_summary_cache)
