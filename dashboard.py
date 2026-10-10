"""Streamlit dashboard for the AI-powered network health monitoring system.

Talks to the FastAPI backend over HTTP (start it with
`uvicorn app.main:app --reload`) and renders five tabs: Overview,
Predictions & Risks, Anomalies, Site Deep Dive, and AI Assistant.

Run with: streamlit run dashboard.py
"""

import time
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

API_BASE_URL = "http://localhost:8000"

# --- Settings --------------------------------------------------------------

# Data refreshes at most once a minute; the backend data changes slowly
CACHE_TTL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 10.0
# AI calls can take several seconds
AI_TIMEOUT_SECONDS = 60.0
REFRESH_INTERVALS = [60, 120, 300]
TOP_N_RISKS = 10
# Compare today's status counts with this many hours earlier
DELTA_HOURS = 4
DELTA_HISTORY_HOURS = DELTA_HOURS + 1
HISTORY_HOURS_DEFAULT = 24

STATUS_ORDER = ["Healthy", "Warning", "Degraded", "Critical"]
STATUS_COLORS = {
    "Healthy": "#2ecc71",
    "Warning": "#f39c12",
    "Degraded": "#e67e22",
    "Critical": "#e74c3c",
}
SEVERITY_COLORS = {"critical": "#e74c3c", "warning": "#f39c12"}
# Risk color bands from the requirements: red > 0.7, orange > 0.4, yellow > 0.3
RISK_BANDS = [(0.7, "#e74c3c"), (0.4, "#e67e22"), (0.3, "#f1c40f")]
GAUGE_STEPS = [
    (0, 40, STATUS_COLORS["Critical"]),
    (40, 60, STATUS_COLORS["Degraded"]),
    (60, 80, STATUS_COLORS["Warning"]),
    (80, 100, STATUS_COLORS["Healthy"]),
]
COMPONENT_LABELS = {
    "cpu_score": "CPU",
    "memory_score": "Memory",
    "latency_score": "Latency",
    "packet_loss_score": "Packet loss",
    "throughput_score": "Throughput",
    "sinr_score": "SINR",
    "alarm_score": "Alarms",
}
DEEP_DIVE_KPIS = [
    ("latency_ms", "Latency (ms)"),
    ("packet_loss_pct", "Packet loss (%)"),
    ("cpu_utilization", "CPU utilization (%)"),
    ("throughput_mbps", "Throughput (Mbps)"),
]
EXAMPLE_QUESTIONS = [
    "What are the biggest risks right now?",
    "Which sites should engineers prioritize?",
    "Why is the overall health score dropping?",
    "Summarize the 5G network status",
]
BACKEND_DOWN_MESSAGE = (
    f"Backend not running. Start it with `uvicorn app.main:app --reload` "
    f"and make sure it is reachable at {API_BASE_URL}."
)


# --- API access ------------------------------------------------------------

class ApiError(Exception):
    """Raised when the backend cannot be reached or returns an error."""

    def __init__(self, message: str, backend_down: bool = False):
        """Store the message and whether the backend was unreachable."""
        super().__init__(message)
        self.backend_down = backend_down


def _request(method: str, path: str, timeout: float, **kwargs: Any) -> Any:
    """Send one request and return decoded JSON, raising ApiError on failure."""
    try:
        response = httpx.request(
            method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as error:
        raise ApiError(BACKEND_DOWN_MESSAGE, backend_down=True) from error
    except httpx.RequestError as error:
        raise ApiError(f"Request to the backend failed: {error}") from error

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"Backend returned {response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def api_get(path: str, params: tuple = ()) -> Any:
    """GET a backend path with caching; params is a tuple of (key, value) pairs."""
    return _request("GET", path, REQUEST_TIMEOUT_SECONDS, params=dict(params))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def api_explain(site_id: str) -> dict:
    """Request an AI explanation for a site, cached to avoid repeated AI cost."""
    return _request("POST", "/api/explain", AI_TIMEOUT_SECONDS, json={"site_id": site_id})


def api_ask(question: str) -> dict:
    """Ask the AI assistant a question (not cached)."""
    return _request("POST", "/api/ask", AI_TIMEOUT_SECONDS, json={"question": question})


def get_json(path: str, **params: Any) -> Any:
    """GET with keyword query parameters; None values are dropped."""
    pairs = tuple(sorted((k, v) for k, v in params.items() if v is not None))
    return api_get(path, pairs)


# --- Data helpers ----------------------------------------------------------

def sidebar_filters() -> dict:
    """Render the sidebar and return the chosen filters and refresh settings."""
    st.sidebar.title("🌐 Network Health Monitor")

    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    interval = st.sidebar.selectbox(
        "Refresh interval (seconds)", REFRESH_INTERVALS, disabled=not auto_refresh
    )
    network_type = st.sidebar.selectbox("Network type", ["All", "5G", "LTE"])
    status = st.sidebar.selectbox("Health status", ["All", *STATUS_ORDER])

    st.sidebar.divider()
    if st.sidebar.button("Generate AI Summary", width="stretch"):
        try:
            with st.spinner("Generating summary..."):
                result = get_json("/api/network-summary")
            st.session_state["network_summary"] = result["summary"]
        except ApiError as error:
            st.session_state["network_summary"] = None
            st.sidebar.error(str(error))
    if st.session_state.get("network_summary"):
        st.sidebar.info(st.session_state["network_summary"])

    return {
        "auto_refresh": auto_refresh,
        "interval": interval,
        "network_type": network_type,
        "status": status,
    }


def apply_filters(sites: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter the site table by the sidebar network type and status."""
    result = sites
    if filters["network_type"] != "All":
        result = result[result["network_type"] == filters["network_type"]]
    if filters["status"] != "All":
        result = result[result["health_status"] == filters["status"]]
    return result


def status_counts_hours_ago(site_ids: list[str], latest_time: pd.Timestamp) -> dict | None:
    """Count sites per health status DELTA_HOURS before the latest reading."""
    target = latest_time - pd.Timedelta(hours=DELTA_HOURS)
    counts = {status: 0 for status in STATUS_ORDER}
    try:
        for site_id in site_ids:
            history = get_json(f"/api/sites/{site_id}/history", hours=DELTA_HISTORY_HOURS)
            points = pd.DataFrame(history["points"])
            if points.empty:
                continue
            times = pd.to_datetime(points["timestamp"])
            nearest = (times - target).abs().idxmin()
            counts[points.loc[nearest, "health_status"]] += 1
    except (ApiError, KeyError):
        return None
    return counts


def risk_color(value: float) -> str:
    """Return a CSS background style for a risk score."""
    for threshold, color in RISK_BANDS:
        if value > threshold:
            return f"background-color: {color}; color: white"
    return ""


def show_api_error(error: ApiError) -> None:
    """Display a backend error in the main area."""
    st.error(str(error))


# --- Tabs ------------------------------------------------------------------

def render_overview(sites: pd.DataFrame, all_sites: pd.DataFrame) -> None:
    """Render the Overview tab: KPI cards, health gauge, status pie, histogram."""
    if sites.empty:
        st.info("No sites match the current filters.")
        return

    counts = sites["health_status"].value_counts().to_dict()
    latest_time = pd.to_datetime(all_sites["timestamp"]).max()
    earlier = status_counts_hours_ago(sites["site_id"].tolist(), latest_time)

    def delta(value: int, statuses: list[str]) -> int | None:
        """Change in the summed status counts versus DELTA_HOURS ago."""
        if earlier is None:
            return None
        return value - sum(earlier[s] for s in statuses)

    healthy = counts.get("Healthy", 0)
    watch = counts.get("Degraded", 0) + counts.get("Warning", 0)
    critical = counts.get("Critical", 0)
    suffix = f"vs {DELTA_HOURS}h ago"

    cols = st.columns(4)
    cols[0].metric("Total Sites", len(sites))
    cols[1].metric(
        "Healthy", healthy, delta(healthy, ["Healthy"]), help=suffix
    )
    cols[2].metric(
        "Degraded / Warning", watch, delta(watch, ["Degraded", "Warning"]),
        delta_color="inverse", help=suffix,
    )
    cols[3].metric(
        "Critical", critical, delta(critical, ["Critical"]),
        delta_color="inverse", help=suffix,
    )

    left, right = st.columns(2)
    average = float(sites["health_score"].mean())
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=average,
        number={"suffix": " / 100"},
        title={"text": "Average health score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#34495e"},
            "steps": [{"range": [lo, hi], "color": color} for lo, hi, color in GAUGE_STEPS],
        },
    ))
    gauge.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    left.plotly_chart(gauge, width="stretch")

    pie_data = (
        sites["health_status"].value_counts().reindex(STATUS_ORDER).dropna().reset_index()
    )
    pie_data.columns = ["status", "sites"]
    pie = px.pie(
        pie_data, names="status", values="sites", color="status",
        color_discrete_map=STATUS_COLORS, title="Sites by status", hole=0.4,
    )
    pie.update_layout(height=320, margin=dict(t=60, b=10, l=10, r=10))
    right.plotly_chart(pie, width="stretch")

    histogram = px.histogram(
        sites, x="health_score", nbins=20, range_x=[0, 100],
        title="Health score distribution (latest scores)",
        labels={"health_score": "Health score", "count": "Sites"},
        color_discrete_sequence=["#3498db"],
    )
    histogram.update_layout(height=320, yaxis_title="Sites", bargap=0.05)
    st.plotly_chart(histogram, width="stretch")


def render_explanation(site_id: str) -> None:
    """Fetch and show the AI explanation for one site."""
    try:
        with st.spinner("Generating AI explanation..."):
            explanation = api_explain(site_id)
    except ApiError as error:
        st.warning(f"AI explanation unavailable: {error}")
        return

    st.markdown(f"**Priority:** {explanation['priority']}")
    st.info(explanation["summary"])
    evidence_col, action_col = st.columns(2)
    evidence_col.markdown("**Evidence**")
    for item in explanation["evidence"]:
        evidence_col.markdown(f"- {item}")
    action_col.markdown("**Recommended actions**")
    for item in explanation["recommended_actions"]:
        action_col.markdown(f"- {item}")


def render_predictions(sites: pd.DataFrame) -> None:
    """Render the Predictions & Risks tab with a selectable top-risk table."""
    try:
        predictions = get_json("/api/predictions")
    except ApiError as error:
        show_api_error(error)
        return

    table = pd.DataFrame(predictions)
    if table.empty:
        st.info("No sites are currently at risk.")
        return

    table = table.merge(
        sites[["site_id", "network_type", "health_score"]], on="site_id", how="inner"
    )
    table = table.sort_values("risk_score", ascending=False).head(TOP_N_RISKS)
    if table.empty:
        st.info("No at-risk sites match the current filters.")
        return

    display = table.rename(columns={
        "site_id": "Site ID",
        "network_type": "Network Type",
        "health_score": "Health Score",
        "risk_score": "Risk Score",
        "predicted_failure_type": "Predicted Issue",
        "estimated_failure_window_hours": "Failure Window (h)",
        "confidence": "Confidence",
    })[["Site ID", "Network Type", "Health Score", "Risk Score",
        "Predicted Issue", "Failure Window (h)", "Confidence"]].reset_index(drop=True)

    st.subheader(f"Top {len(display)} at-risk sites")
    styled = display.style.map(risk_color, subset=["Risk Score"]).format(
        {"Risk Score": "{:.2f}", "Confidence": "{:.2f}", "Failure Window (h)": "{:.1f}"}
    )
    selected_site = None
    try:
        event = st.dataframe(
            styled, width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row",
        )
        rows = event.selection.rows
        if rows:
            selected_site = display.loc[rows[0], "Site ID"]
        st.caption("Click a row to see the detailed prediction and AI explanation.")
    except TypeError:
        st.dataframe(styled, width="stretch", hide_index=True)
        selected_site = st.selectbox("Select a site", display["Site ID"].tolist())

    if not selected_site:
        return

    st.divider()
    st.subheader(selected_site)
    try:
        prediction = get_json(f"/api/predictions/{selected_site}")
    except ApiError as error:
        show_api_error(error)
        return

    cols = st.columns(4)
    cols[0].metric("Risk score", f"{prediction['risk_score']:.2f}")
    cols[1].metric("Confidence", f"{prediction['confidence']:.2f}")
    cols[2].metric("Failure window", f"{prediction['estimated_failure_window_hours']:.1f} h")
    cols[3].metric("Predicted issue", prediction["predicted_failure_type"].replace("_", " "))
    st.write(prediction["explanation"])
    st.markdown(f"**Recommended action:** {prediction['recommended_action']}")
    if prediction["supporting_kpis"]:
        st.markdown("**Supporting KPIs**")
        st.dataframe(
            pd.DataFrame(prediction["supporting_kpis"]),
            width="stretch", hide_index=True,
        )

    st.markdown("#### AI explanation")
    render_explanation(selected_site)


def render_anomalies(sites: pd.DataFrame) -> None:
    """Render the Anomalies tab: summary metrics, table, and count by metric."""
    try:
        anomalies = get_json("/api/anomalies")
    except ApiError as error:
        show_api_error(error)
        return

    table = pd.DataFrame(anomalies)
    if table.empty:
        st.info("No anomalies detected.")
        return
    table = table[table["site_id"].isin(sites["site_id"])]
    if table.empty:
        st.info("No anomalies for the sites matching the current filters.")
        return

    top_metric = table["metric_name"].value_counts().idxmax()
    cols = st.columns(3)
    cols[0].metric("Total anomalies", len(table))
    cols[1].metric("Critical", int((table["severity"] == "critical").sum()))
    cols[2].metric("Most affected KPI", top_metric)

    by_metric = table.groupby(["metric_name", "severity"]).size().reset_index(name="count")
    chart = px.bar(
        by_metric, x="metric_name", y="count", color="severity",
        color_discrete_map=SEVERITY_COLORS, title="Anomalies by metric",
        labels={"metric_name": "Metric", "count": "Anomalies"},
    )
    st.plotly_chart(chart, width="stretch")

    shown = table[[
        "timestamp", "site_id", "metric_name", "current_value",
        "expected_range_low", "expected_range_high", "severity", "description",
    ]].copy()
    shown["timestamp"] = pd.to_datetime(shown["timestamp"])

    def severity_style(row: pd.Series) -> list[str]:
        """Color a table row by anomaly severity."""
        color = SEVERITY_COLORS[row["severity"]]
        return [f"background-color: {color}33"] * len(row)

    styled = shown.style.apply(severity_style, axis=1).format(
        {"current_value": "{:.2f}", "expected_range_low": "{:.2f}",
         "expected_range_high": "{:.2f}"}
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def build_deep_dive_figure(history: dict, anomalies: list[dict]) -> go.Figure:
    """Build KPI time series with health score overlay and anomaly markers."""
    points = pd.DataFrame(history["points"])
    points["timestamp"] = pd.to_datetime(points["timestamp"])
    metrics = pd.DataFrame(list(points["metrics"]))
    anomaly_frame = pd.DataFrame(anomalies)
    if not anomaly_frame.empty:
        anomaly_frame["timestamp"] = pd.to_datetime(anomaly_frame["timestamp"])

    figure = make_subplots(
        rows=2, cols=2, subplot_titles=[label for _, label in DEEP_DIVE_KPIS],
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]],
        horizontal_spacing=0.1, vertical_spacing=0.15,
    )
    for index, (kpi, label) in enumerate(DEEP_DIVE_KPIS):
        row, col = divmod(index, 2)
        row, col = row + 1, col + 1
        first = index == 0
        figure.add_trace(
            go.Scatter(x=points["timestamp"], y=metrics[kpi], name=label,
                       mode="lines", line=dict(color="#2980b9"), showlegend=False),
            row=row, col=col, secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(x=points["timestamp"], y=points["health_score"],
                       name="Health score", mode="lines",
                       line=dict(color="#7f8c8d", dash="dot"), showlegend=first),
            row=row, col=col, secondary_y=True,
        )
        if not anomaly_frame.empty:
            marks = anomaly_frame[anomaly_frame["metric_name"] == kpi]
            if not marks.empty:
                figure.add_trace(
                    go.Scatter(
                        x=marks["timestamp"], y=marks["current_value"], mode="markers",
                        name="Anomaly", showlegend=first,
                        marker=dict(color="#e74c3c", size=9, symbol="x"),
                        hovertext=marks["description"],
                    ),
                    row=row, col=col, secondary_y=False,
                )
        figure.update_yaxes(title_text=label, row=row, col=col, secondary_y=False)
        figure.update_yaxes(range=[0, 100], title_text="Health", row=row, col=col,
                            secondary_y=True, showgrid=False)
    figure.update_layout(height=620, margin=dict(t=60, b=20),
                         legend=dict(orientation="h", y=1.1))
    return figure


def render_deep_dive(sites: pd.DataFrame) -> None:
    """Render the Site Deep Dive tab for a selected site."""
    if sites.empty:
        st.info("No sites match the current filters.")
        return

    ordered = sites.sort_values("health_score")["site_id"].tolist()
    site_id = st.selectbox("Select a site", ordered)
    try:
        detail = get_json(f"/api/sites/{site_id}")
        history = get_json(f"/api/sites/{site_id}/history", hours=HISTORY_HOURS_DEFAULT)
    except ApiError as error:
        show_api_error(error)
        return

    head = st.columns(4)
    head[0].metric("Health score", detail["health_score"])
    head[1].metric("Health status", detail["health_status"])
    head[2].metric("Site status", detail["site_status"])
    head[3].metric("Network type", detail["network_type"])

    left, right = st.columns([1, 2])
    labels = [COMPONENT_LABELS.get(k, k) for k in detail["component_scores"]]
    values = list(detail["component_scores"].values())
    radar = go.Figure(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1], fill="toself",
        line=dict(color=STATUS_COLORS.get(detail["health_status"], "#3498db")),
    ))
    radar.update_layout(
        title="Component scores (higher is better)", height=380,
        polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
        showlegend=False, margin=dict(t=60, b=20),
    )
    left.plotly_chart(radar, width="stretch")

    right.markdown("**Latest metrics**")
    right.dataframe(
        pd.DataFrame([detail["latest_metrics"]]).round(2),
        width="stretch", hide_index=True,
    )
    if detail.get("prediction"):
        prediction = detail["prediction"]
        right.warning(
            f"Risk {prediction['risk_score']:.2f}: "
            f"{prediction['predicted_failure_type'].replace('_', ' ')}. "
            f"{prediction['explanation']}"
        )

    st.plotly_chart(
        build_deep_dive_figure(history, detail["anomalies"]), width="stretch"
    )
    st.caption(
        f"{len(detail['anomalies'])} statistical anomalies for this site. "
        "Red markers show anomalies on the KPI they affect; the dotted line is the health score."
    )


def run_question(question: str) -> None:
    """Send a question to the AI assistant and store the answer in session state."""
    try:
        with st.spinner("Thinking..."):
            st.session_state["answer"] = api_ask(question)["answer"]
        st.session_state["answered_question"] = question
        st.session_state["answer_error"] = None
    except ApiError as error:
        st.session_state["answer"] = None
        st.session_state["answer_error"] = str(error)


def use_example(question: str) -> None:
    """Callback: load an example question and queue it for sending."""
    st.session_state["question_text"] = question
    st.session_state["pending_question"] = question


def render_assistant() -> None:
    """Render the AI Assistant tab."""
    st.subheader("Ask about network health")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(cols, EXAMPLE_QUESTIONS):
        column.button(
            example, key=f"example_{example}", on_click=use_example, args=(example,),
            width="stretch",
        )

    question = st.text_input("Ask about network health...", key="question_text")
    send = st.button("Send", type="primary")

    pending = st.session_state.pop("pending_question", None)
    if pending:
        run_question(pending)
    elif send:
        if question.strip():
            run_question(question.strip())
        else:
            st.warning("Please enter a question.")

    if st.session_state.get("answer_error"):
        st.error(st.session_state["answer_error"])
    elif st.session_state.get("answer"):
        st.caption(f"Q: {st.session_state.get('answered_question', '')}")
        st.info(st.session_state["answer"])


# --- Main ------------------------------------------------------------------

def main() -> None:
    """Configure the page, load site data, and render all tabs."""
    st.set_page_config(page_title="Network Health Monitor", layout="wide")
    filters = sidebar_filters()
    st.title("Network Health Monitor")

    try:
        all_sites = pd.DataFrame(get_json("/api/sites"))
    except ApiError as error:
        st.error(str(error))
        st.stop()

    sites = apply_filters(all_sites, filters)

    overview, predictions, anomalies, deep_dive, assistant = st.tabs([
        "Overview", "Predictions & Risks", "Anomalies", "Site Deep Dive", "AI Assistant",
    ])
    with overview:
        render_overview(sites, all_sites)
    with predictions:
        render_predictions(sites)
    with anomalies:
        render_anomalies(sites)
    with deep_dive:
        render_deep_dive(sites)
    with assistant:
        render_assistant()

    st.caption(
        "AI-powered predictions are advisory. Always verify with network operations team."
    )

    if filters["auto_refresh"]:
        time.sleep(filters["interval"])
        st.rerun()


main()
