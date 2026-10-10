# AI-Powered Network Health Monitoring & Failure Prediction

A system that scores the health of 5G/LTE cell sites from KPI telemetry, detects anomalies, predicts which sites are likely to fail and when, and explains each risk in plain language through a FastAPI backend and a Streamlit dashboard.

## Architecture

The system is a six-stage pipeline. Stages 1-4 are deterministic and run once when the API starts; stage 5 calls OpenAI on demand; stage 6 is the user interface.

1. **Data ingestion** loads 15-minute KPI telemetry for every site from CSV.
2. **Health scoring** reduces each reading to a 0-100 score from seven weighted KPIs.
3. **Anomaly detection** flags unusual readings per KPI (z-score) and per row (Isolation Forest).
4. **Failure prediction** combines trend, level, threshold proximity, anomalies, and acceleration into a risk score, a failure type, and a time-to-failure estimate.
5. **AI explanation** turns the computed findings into business-friendly explanations, executive summaries, and answers.
6. **Dashboard** presents everything and lets an operator drill into a site or ask questions.

```
                         ┌──────────────────────────┐
  generate_data.py ────► │ data/network_telemetry.csv│
                         │ data/site_scenarios.json  │
                         └────────────┬─────────────┘
                                      │ load_dataset()
                                      ▼
   ┌───────────────────────────── FastAPI (app/main.py) ─────────────────────────────┐
   │                                                                                 │
   │  1 Ingestion      2 Health scoring     3 Anomaly detection   4 Prediction       │
   │  data_generator ─► health_scorer ─┬──► anomaly_detector ───► predictor          │
   │                    score 0-100    │    z-score (per KPI)     risk 0-1           │
   │                    7 sub-scores   │    IsolationForest       type, window,      │
   │                                   │    (multivariate)        confidence         │
   │                                   └─────────────────────────────┘               │
   │                                      │  results held in app.state               │
   │                                      ▼                                          │
   │  5 AI explanation (ai_agent.py) ◄── structured summaries only, never raw data   │
   │      explain_site_risk · generate_network_summary · answer_question ──► OpenAI  │
   └──────────────────────────────────────┬──────────────────────────────────────────┘
                                          │ JSON over HTTP (httpx)
                                          ▼
                               6 Streamlit dashboard (dashboard.py)
                  Overview · Predictions & Risks · Anomalies · Site Deep Dive · AI Assistant
```

### Key design decisions

- **Weighted multi-KPI scoring, not a single threshold.** Faults rarely show up in one KPI. A site with CPU at 70%, latency at 45 ms, and 1.8% packet loss breaks no single threshold but is clearly in trouble. A weighted score captures combined degradation, degrades smoothly instead of flipping at a cliff, and keeps per-KPI sub-scores so the cause of a low score can be explained.
- **Dual anomaly detection (statistical + Isolation Forest).** The z-score method is interpretable: it names the KPI, the value, and the expected range, which is what an engineer needs. It looks at one KPI at a time, though. The Isolation Forest sees all eight KPIs together and catches joint shifts that stay under every individual threshold, but it cannot say which KPI is responsible. Each covers the other's blind spot.
- **Transparent risk model, not black-box ML.** There is no labeled outage history to train on, and operations teams must be able to audit why a site was flagged. A weighted sum of five readable signals can be inspected, argued about, and tuned, and every prediction lists the signals behind it.
- **Trend extrapolation for failure windows.** The question an on-call engineer asks is "how long do I have?". Extrapolating the health score's recent slope to the Critical band answers it with a number that is simple to explain and cheap to compute. Its limits are stated explicitly below.

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone <repo>
cd network-health-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (it is git-ignored). A template is provided in `.env.example`.

```
OPENAI_API_KEY=sk-your-key-here
```

The key must be on a single line. The API refuses to start if it is missing.

### Generate Synthetic Data

```bash
python generate_data.py
```

This produces `data/network_telemetry.csv` and `data/site_scenarios.json`:

- **50 sites** (30 on 5G, 20 on LTE) across 15 US cities, with IDs such as `DALLAS_5G_RAN_1000`.
- **24 hours at 15-minute intervals**, which is 96 readings per site, and **50 × 96 = 4,800 rows**.
- **12 columns**: `site_id`, `timestamp`, `network_type`, `cpu_utilization`, `memory_utilization`, `latency_ms`, `packet_loss_pct`, `throughput_mbps`, `connected_users`, `signal_quality_sinr`, `alarm_count`, `site_status`.
- **Planted scenarios**, one per site, recorded in `site_scenarios.json` for evaluation:

| Scenario | Sites | Behavior |
|---|---|---|
| HEALTHY | 25 | KPIs stay at baseline with small noise |
| DEGRADING | 8 | Gradual, accelerating worsening; Degraded around hour 16-18 |
| CONGESTION | 7 | Normal for 18 hours, then a user surge drags CPU, latency, loss |
| FAILING | 5 | KPIs cross critical thresholds in the last 2-4 hours |
| INTERMITTENT | 3 | 3-4 spike-and-recover cycles (flapping) |
| SUDDEN_FAILURE | 2 | Normal for ~20 hours, then everything crashes within about an hour |

Generation is seeded (`seed=42`), so KPI values are identical on every run. Timestamps are anchored to the current UTC time.

### Run the Application

Terminal 1 — API server:

```bash
uvicorn app.main:app --reload --port 8000
```

Terminal 2 — Dashboard:

```bash
streamlit run dashboard.py
```

The API is at http://localhost:8000 (interactive docs at `/docs`) and the dashboard at http://localhost:8501. On startup the API logs `Loaded 50 sites, 696 anomalies detected, 22 sites at risk`.

## API Reference

Base URL: `http://localhost:8000`. All responses are JSON. Errors use the form `{"detail": "..."}` with status 400 (bad input), 404 (unknown site), 422 (schema validation), 503 (service or AI unavailable), or 500 (unexpected, details are logged server-side only).

### `GET /health`

Service health and dependency status.

```json
{
  "status": "ok",
  "data_loaded": true,
  "model_ready": true,
  "openai_connected": true
}
```

`openai_connected` means the client was initialized, not that a live call succeeded.

### `GET /api/sites`

All sites with latest health, lowest score first. Query: `status` = `Healthy` | `Warning` | `Degraded` | `Critical` (case-insensitive; anything else returns 400).

```json
[
  {
    "site_id": "MINNEAPOLIS_LTE_RAN_1731",
    "network_type": "LTE",
    "timestamp": "2026-10-10T17:45:00",
    "health_score": 0,
    "health_status": "Critical",
    "site_status": "Failed",
    "risk_score": 0.85
  }
]
```

`risk_score` is `null` for sites with no prediction above the reporting threshold.

### `GET /api/sites/{site_id}`

Detailed view of one site: latest metrics, component sub-scores, Isolation Forest flag, anomalies, and prediction (anomaly and prediction lists shortened here).

```json
{
  "site_id": "MINNEAPOLIS_LTE_RAN_1731",
  "network_type": "LTE",
  "timestamp": "2026-10-10T17:45:00",
  "health_score": 0,
  "health_status": "Critical",
  "site_status": "Failed",
  "latest_metrics": {
    "cpu_utilization": 98.28,
    "memory_utilization": 97.64,
    "latency_ms": 253.62,
    "packet_loss_pct": 10.35,
    "throughput_mbps": 5.65,
    "connected_users": 166.0,
    "signal_quality_sinr": -6.77,
    "alarm_count": 9.0
  },
  "component_scores": {
    "latency_score": 0.0, "packet_loss_score": 0.0, "cpu_score": 0.0,
    "throughput_score": 0.0, "memory_score": 0.0, "sinr_score": 0.0, "alarm_score": 0.0
  },
  "ml_anomaly": true,
  "ml_anomaly_score": 0.784,
  "anomalies": [
    {
      "site_id": "MINNEAPOLIS_LTE_RAN_1731",
      "timestamp": "2026-10-10T15:45:00",
      "metric_name": "cpu_utilization",
      "current_value": 52.75,
      "expected_range_low": 36.88,
      "expected_range_high": 47.42,
      "severity": "critical",
      "description": "cpu_utilization is 52.75, 5.0 standard deviations above its rolling mean (expected 36.88 to 47.42)"
    }
  ],
  "prediction": { "...": "same shape as /api/predictions/{site_id}" }
}
```

### `GET /api/sites/{site_id}/history`

Time series for charts. Query: `hours` (1-24, optional; relative to the site's latest reading).

```json
{
  "site_id": "MINNEAPOLIS_LTE_RAN_1731",
  "network_type": "LTE",
  "hours": 1,
  "points": [
    {
      "timestamp": "2026-10-10T17:00:00",
      "health_score": 1,
      "health_status": "Critical",
      "site_status": "Failed",
      "is_anomaly": true,
      "anomaly_score": 0.786,
      "metrics": {
        "cpu_utilization": 99.15, "memory_utilization": 95.18, "latency_ms": 254.04,
        "packet_loss_pct": 10.41, "throughput_mbps": 0.0, "connected_users": 166.0,
        "signal_quality_sinr": -6.2, "alarm_count": 10.0
      }
    }
  ]
}
```

### `GET /api/health-summary`

Network-wide counts, average score, and the 10 lowest-scoring sites.

```json
{
  "total_sites": 50,
  "healthy_count": 28,
  "warning_count": 1,
  "degraded_count": 12,
  "critical_count": 9,
  "average_score": 70.7,
  "worst_sites": [
    {
      "site_id": "MINNEAPOLIS_LTE_RAN_1731",
      "network_type": "LTE",
      "health_score": 0,
      "health_status": "Critical",
      "weakest_kpi": "latency_score"
    }
  ]
}
```

### `GET /api/anomalies`

Statistical anomalies, critical first then newest first. Query: `site_id` (404 if unknown), `limit` (1-5000).

```json
[
  {
    "site_id": "DENVER_5G_RAN_1085",
    "timestamp": "2026-10-10T17:00:00",
    "metric_name": "signal_quality_sinr",
    "current_value": 5.2,
    "expected_range_low": 6.8,
    "expected_range_high": 9.3,
    "severity": "critical",
    "description": "signal_quality_sinr is 5.2, 5.7 standard deviations below its rolling mean (expected 6.8 to 9.3)"
  }
]
```

### `GET /api/predictions`

Predictions above the 0.3 risk threshold, highest risk first. Query: `min_risk` (0-1).

```json
[
  {
    "site_id": "HOUSTON_LTE_RAN_1527",
    "predicted_failure_type": "signal_degradation",
    "risk_score": 1.0,
    "confidence": 0.956,
    "estimated_failure_window_hours": 0.0,
    "supporting_kpis": [
      {
        "kpi_name": "latency_ms",
        "current_value": 231.92,
        "trend": "increasing",
        "rate_per_hour": 50.777,
        "threshold": 100
      }
    ],
    "explanation": "Likely signal degradation: Health score is 1 and falling 27.5 points/hour; 7 KPI(s) near critical (...); 37 anomalies in the last 2 hours; the decline is accelerating; already in the critical band.",
    "recommended_action": "URGENT: Inspect antennas, feeders, and interference sources; schedule a radio optimization."
  }
]
```

### `GET /api/predictions/{site_id}`

One site's prediction, same shape as above. For a site below the reporting threshold the prediction is computed on demand, so it returns 200 with a low risk score. Unknown site returns 404.

### `POST /api/explain`

AI explanation of one site's risk. Returns 503 if OpenAI is unavailable.

Request:

```json
{ "site_id": "MINNEAPOLIS_LTE_RAN_1731" }
```

Response:

```json
{
  "site_id": "MINNEAPOLIS_LTE_RAN_1731",
  "summary": "The Minneapolis LTE site is experiencing severe performance issues, with a health score of 0 and multiple critical KPIs indicating imminent failure. ...",
  "evidence": [
    "Health score dropped from 99 to 0 in 4 hours.",
    "CPU utilization increased from 43.2% to 98.3%.",
    "Latency increased from 37.9 ms to 253.6 ms."
  ],
  "recommended_actions": [
    "Investigate and resolve high CPU and memory utilization immediately.",
    "Check for any hardware failures or network congestion causing high latency and packet loss."
  ],
  "priority": "P1-Critical"
}
```

### `POST /api/ask`

Natural language question answered from the current data. Question is 1-1000 characters.

Request / response:

```json
{ "question": "How many sites are critical?" }
```

```json
{ "answer": "There are 9 critical sites." }
```

### `GET /api/network-summary`

One-paragraph executive briefing. Cached after the first successful call because the data is static.

```json
{
  "summary": "The network's overall health score is 70.7, with 9 critical and 12 degraded sites out of 50 total. ..."
}
```

## Evaluation

```bash
python evaluate.py
pytest tests/ -v
```

`pytest` is not in `requirements.txt`; install it with `pip install pytest`.

`evaluate.py` runs the real scoring, anomaly, and prediction modules (no API) against three purpose-built datasets and exits non-zero if any check fails:

| Scenario | Data | Checks |
|---|---|---|
| 1. Healthy network baseline | 12 healthy sites | All health scores above 80; zero critical anomalies; zero failure predictions (no false alarms) |
| 2. Degrading site detection | 8 degrading sites | Final health score below 60 at every site; anomalies detected at every site; a prediction with risk above 0.3 at every site |
| 3. Sudden failure detection | 4 sudden-failure sites | Every failure caught; risk above 0.7 at every site; estimated failure window under 4 hours |

`tests/test_pipeline.py` holds 13 unit and integration tests: dataset shape, scenarios and non-negative values; health scoring of healthy and critical rows; weights summing to 1.0; an injected 500 ms latency spike being flagged; prediction risk on failing versus healthy series; and Pydantic validation of valid and invalid data.

## Project Structure

```
.
├── app
│   ├── __init__.py
│   ├── ai_agent.py
│   ├── anomaly_detector.py
│   ├── config.py
│   ├── data_generator.py
│   ├── health_scorer.py
│   ├── main.py
│   ├── models.py
│   └── predictor.py
├── data
│   ├── network_telemetry.csv
│   └── site_scenarios.json
├── tests
│   └── test_pipeline.py
├── apikey
├── dashboard.py
├── evaluate.py
├── generate_data.py
└── requirements.txt

4 directories, 17 files
```

Output of `tree -I '__pycache__|venv|.git|*.pyc' --dirsfirst`. `tree` hides dotfiles, so `.env`, `.env.example`, and `.gitignore` are present but not shown.

## Prediction Methodology

All thresholds live in `app/config.py` (with a comment on each) so there is one place to calibrate.

### Health scoring

Each of seven KPIs is mapped to a 0-100 sub-score with a piecewise linear curve through three anchors, then combined with fixed weights into an integer score.

| Sub-score | Value |
|---|---|
| At or better than "good" | 100 |
| At the warning threshold | 60 |
| At the critical threshold | 20 |
| Beyond critical | continues on the same slope to 0, then clamped |

The curve is continuous, so small KPI changes never cause large score jumps.

| KPI | Weight | Good | Warning | Critical | Why |
|---|---|---|---|---|---|
| Latency | 0.20 | 5G 15 ms / LTE 40 ms | 5G 30 / LTE 60 | 5G 50 / LTE 100 | Maps to the 3GPP 5QI packet delay budget; strongest driver of user experience. 5G SLA is typically under 10 ms, so 30 ms is an early signal. LTE normally runs 30-50 ms. |
| Packet loss | 0.20 | 1.0% | 2.0% | 5.0% | Maps to the 5QI packet error rate; breaks voice, video, and TCP quickly. Carrier targets are usually under 1%. |
| CPU | 0.15 | 60% | 75% | 90% | Best leading indicator of overload. 75% leaves 15 points of headroom; above 90% bursts cause failures. |
| Throughput | 0.15 | 5G 900 / LTE 150 Mbps | 70% of baseline (630 / 105) | 50% of baseline (450 / 75) | Delivered capacity. Baselines sit at the lower quartile of each healthy range so low-end healthy sites are not penalized. |
| Memory | 0.10 | 60% | 75% | 90% | Changes slowly; failures also show up in CPU and latency. |
| SINR | 0.10 | 10 dB | 5 dB | 0 dB | Already reflected in throughput and loss. At 0 dB, noise equals signal. |
| Alarms | 0.10 | 0 | 2 | 5 | Corroborates KPI problems rather than leading them. |

Per-network-type adjustment: latency and throughput use separate 5G and LTE anchors (above). The other KPIs share thresholds across technologies.

The integer score maps to a status:

| Score | Status |
|---|---|
| 80-100 | Healthy |
| 60-79 | Warning |
| 40-59 | Degraded |
| 0-39 | Critical |

### Anomaly detection

- **Statistical (z-score).** For each site and KPI, each value is compared with the mean and standard deviation of the **previous 12 readings** (a 3-hour rolling window, which excludes the current value). Absolute z above **2.5** is a `warning`, above **3.5** a `critical`. Both directions are flagged. The standard deviation is floored at 5% of the mean plus a small per-KPI absolute minimum, so a flat window cannot turn a tiny move into an enormous z-score. The expected range reported is mean ± 2.5 standard deviations.
- **Isolation Forest.** One model is fitted on all eight KPIs (`n_estimators=100`, `max_samples="auto"`, `contamination=0.1`, `random_state=42`). `contamination=0.1` makes it flag the 10% most unusual rows, matching the share of planted faults. `anomaly_score` is the negated `score_samples` value, so higher is more anomalous.
- **Combined use.** The z-score output feeds explanations and the anomaly-density risk signal. The Isolation Forest output is attached to each row for dashboards and downstream boosting.

### Risk scoring

The risk score is a weighted sum of five signals, each normalized to 0-1:

| Signal | Weight | Normalization |
|---|---|---|
| Health score trend (last 4 h slope) | 0.30 | 0 at no decline, 0.5 at 2 points/hour, 1.0 at 5 points/hour or more |
| Current health score | 0.20 | 1.0 at 40 or below, 0.5 at 60, 0 at 80 or above |
| KPI threshold proximity | 0.20 | Count of KPIs within 15% of critical (or beyond), saturating at 3 |
| Anomaly density (last 2 h) | 0.15 | Statistical anomaly count, saturating at 10 |
| Trend acceleration | 0.15 | 2-hour decline minus 4-hour decline, saturating at 5 points/hour |

Weight rationale: the trend is the most direct evidence of movement toward failure; the current level matters but a low stable score is less urgent than a falling one; proximity shows margin running out on several fronts; anomalies catch jumps that trends smooth over but are noisier; acceleration shortens the time left to react.

Sites above **0.3** risk are reported. Risk levels in the summary are high (0.7 or above), medium (0.5 or above), and low.

**Failure type** comes from per-KPI stress (1 minus the sub-score divided by 100) weighted by how specifically each KPI points to a failure mode: congestion (user surge 1.0, latency and loss 0.5), capacity exhaustion (CPU and memory 0.8), signal degradation (SINR 1.0, throughput 0.5), hardware degradation (alarms and loss 0.5). The type with the highest weighted stress wins. This is a heuristic: when many KPIs collapse at once, the root cause is genuinely ambiguous.

### Failure window estimation

The window is the time for the health score to fall below 40 (the Critical band) if it keeps falling at its recent rate:

```
window = (current_score - 40) / max(decline_over_4h, decline_over_2h)
```

- A site already below 40 has a window of 0.
- A site declining slower than 0.5 points/hour is capped at 24 hours, meaning no crossing is expected.
- The faster of the two slopes is used deliberately: it errs toward warning earlier.

Limitations: this is linear extrapolation, which assumes a constant rate. Real failures often accelerate (the acceleration signal partly compensates) or stop abruptly. Treat the window as an estimate with wide error, not a forecast.

### Confidence calibration

```
confidence = 0.5 × signal_agreement + 0.3 × R²(4-hour health trend) + 0.2 × data_completeness
```

- **Signal agreement** is the fraction of the five risk signals at or above the medium level (0.5), so several independent signals pointing the same way raise confidence.
- **R²** measures how well a straight line fits the 4-hour health trend; a steady decline is more trustworthy than a noisy one.
- **Data completeness** is the share of the expected 4-hour window actually present.

The weights are expert-set, not learned, so confidence is an ordinal guide rather than a calibrated probability.

## Design Decisions

- **Weighted scoring over single-threshold:** captures multi-dimensional degradation and gives smooth, explainable scores with per-KPI sub-scores.
- **Dual anomaly detection:** statistical (interpretable) + ML (multivariate patterns that single-KPI z-scores miss).
- **Transparent risk model:** network ops teams need to trust and audit predictions; no labeled outage history exists to train a supervised model.
- **Pydantic models:** type safety, validation, serialization. The same schemas are used by the generator, scorer, predictor, and API responses.
- **FastAPI + Streamlit:** rapid development, async support, interactive dashboards. The analysis runs once at startup and is held in memory, so read endpoints are cheap lookups.
- **Grounded AI explanations:** the language model receives small, labeled, structured summaries computed by the deterministic engines, never raw data dumps. It is instructed to cite only provided values, which keeps every statement traceable to a metric. All OpenAI errors are caught and degrade to a fallback message instead of crashing the app.

## AI Tools Used

- **Claude Code** — scaffolding, implementation, testing, and documentation of this project.
- **ChatGPT / Claude** — code generation and review.
- **OpenAI GPT-4o** — runtime model that writes the explanations, executive summaries, and question answers inside the application.

## Limitations & Future Improvements

- Linear trend extrapolation (could use ARIMA/Prophet)
- No real-time streaming (batch processing on static dataset)
- Single-node deployment (no HA)
- No authentication/authorization (CORS is open for local development)
- Threshold values would need calibration with real network data
- Could add LSTM-based prediction with sufficient training data
- Evaluation uses data from the project's own generator, so it shows internal consistency with the designed scenarios, not performance on real telemetry
- Failure type is a heuristic and is ambiguous when many KPIs collapse together
- Sudden failures cannot be predicted ahead of time by any trend-based method; they are recognized once the crash begins
