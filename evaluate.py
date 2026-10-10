"""End-to-end validation scenarios for the network health monitoring pipeline.

Generates targeted synthetic data, runs it through the real health scorer,
anomaly detector, and predictor (no API involved), and checks the results
against what each scenario should produce.

Usage: python evaluate.py
Exit code is 0 when every check passes and 1 otherwise.
"""

import sys

try:
    import numpy as np
    import pandas as pd

    from app.anomaly_detector import detect_all_anomalies
    from app.config import HOURS_OF_DATA, PREDICTION_WINDOW_HOURS
    from app.data_generator import (
        COLUMNS,
        INTERVAL_MINUTES,
        POINTS_PER_HOUR,
        SEED,
        generate_site_telemetry,
    )
    from app.health_scorer import score_all_sites
    from app.predictor import predict_all_sites
except ImportError as error:
    print(f"Missing dependency or module: {error}")
    print("Install the requirements first:  pip install -r requirements.txt")
    print("Then run this script from the project root:  python evaluate.py")
    sys.exit(1)

# --- Evaluation settings ---------------------------------------------------

# Fixed end time so results do not depend on when the script runs
EVAL_END_TIME = pd.Timestamp("2025-01-02 00:00:00")
HEALTHY_SITES = 12
DEGRADING_SITES = 8
SUDDEN_FAILURE_SITES = 4
# Share of 5G sites in each generated group; the rest are LTE
FIVE_G_SHARE = 0.6

# Scenario 1: a healthy site must stay well inside the Healthy band
HEALTHY_SCORE_MIN = 80
# Scenario 2: a degrading site must end below the Warning band
DEGRADED_SCORE_MAX = 60
DEGRADING_RISK_MIN = 0.3
# Scenario 3: sudden failures must be high risk with a short window
SUDDEN_RISK_MIN = 0.7
SUDDEN_WINDOW_MAX_HOURS = PREDICTION_WINDOW_HOURS

PASS = "PASS"
FAIL = "FAIL"


# --- Helpers ---------------------------------------------------------------

def build_dataset(scenario: str, n_sites: int, seed_offset: int) -> pd.DataFrame:
    """Generate n_sites of a single scenario, ready for the pipeline."""
    timestamps = pd.date_range(
        end=EVAL_END_TIME, periods=HOURS_OF_DATA * POINTS_PER_HOUR,
        freq=f"{INTERVAL_MINUTES}min",
    )
    n_five_g = round(n_sites * FIVE_G_SHARE)
    records = []
    for index in range(n_sites):
        network_type = "5G" if index < n_five_g else "LTE"
        site_id = f"EVAL_{scenario}_{network_type}_{index:02d}"
        rng = np.random.default_rng(SEED + seed_offset + index)
        records.extend(generate_site_telemetry(site_id, network_type, scenario, timestamps, rng))
    return pd.DataFrame.from_records(records, columns=COLUMNS)


def run_pipeline(df: pd.DataFrame) -> dict:
    """Run scoring, anomaly detection, and prediction on a dataset."""
    scored = score_all_sites(df)
    _, anomalies = detect_all_anomalies(df)
    predictions = predict_all_sites(scored, anomalies)
    latest = scored.sort_values("timestamp").groupby("site_id").tail(1).set_index("site_id")
    return {
        "scored": scored,
        "latest": latest,
        "anomalies": anomalies,
        "predictions": {p.site_id: p for p in predictions},
    }


def check(scenario: str, expected: str, actual: str, passed: bool) -> dict:
    """Build one check record and print it."""
    result = PASS if passed else FAIL
    print(f"  [{result}] {expected}  ->  {actual}")
    return {"scenario": scenario, "expected": expected, "actual": actual, "result": result}


# --- Scenarios -------------------------------------------------------------

def scenario_healthy_baseline() -> list[dict]:
    """Scenario 1: an all-healthy network raises no alarms and no predictions."""
    name = "1. Healthy baseline"
    print(f"\n{name}")
    run = run_pipeline(build_dataset("HEALTHY", HEALTHY_SITES, seed_offset=0))

    lowest = int(run["scored"]["health_score"].min())
    critical = [a for a in run["anomalies"] if a.severity == "critical"]
    return [
        check(name, f"all health scores > {HEALTHY_SCORE_MIN}",
              f"minimum score {lowest}", lowest > HEALTHY_SCORE_MIN),
        check(name, "0 critical anomalies",
              f"{len(critical)} critical of {len(run['anomalies'])} total", not critical),
        check(name, "0 failure predictions",
              f"{len(run['predictions'])} predictions", not run["predictions"]),
    ]


def scenario_degrading_detection() -> list[dict]:
    """Scenario 2: degrading sites end unhealthy, show anomalies, and are predicted."""
    name = "2. Degrading detection"
    print(f"\n{name}")
    run = run_pipeline(build_dataset("DEGRADING", DEGRADING_SITES, seed_offset=1000))
    sites = list(run["latest"].index)

    final_scores = run["latest"]["health_score"]
    below = int((final_scores < DEGRADED_SCORE_MAX).sum())

    with_anomalies = {a.site_id for a in run["anomalies"]}
    detected = sum(site in with_anomalies for site in sites)

    risks = {s: run["predictions"][s].risk_score for s in sites if s in run["predictions"]}
    above = sum(risk > DEGRADING_RISK_MIN for risk in risks.values())
    risk_range = f"{min(risks.values()):.2f}-{max(risks.values()):.2f}" if risks else "none"

    return [
        check(name, f"final health score < {DEGRADED_SCORE_MAX} at {len(sites)}/{len(sites)} sites",
              f"{below}/{len(sites)} sites (scores {int(final_scores.min())}-{int(final_scores.max())})",
              below == len(sites)),
        check(name, "anomalies detected at every site",
              f"{detected}/{len(sites)} sites, {len(run['anomalies'])} anomalies",
              detected == len(sites)),
        check(name, f"prediction with risk > {DEGRADING_RISK_MIN} at every site",
              f"{above}/{len(sites)} sites (risk {risk_range})", above == len(sites)),
    ]


def scenario_sudden_failure() -> list[dict]:
    """Scenario 3: sudden failures are caught with high risk and a short window."""
    name = "3. Sudden failure detection"
    print(f"\n{name}")
    run = run_pipeline(build_dataset("SUDDEN_FAILURE", SUDDEN_FAILURE_SITES, seed_offset=2000))
    sites = list(run["latest"].index)
    caught = [run["predictions"][s] for s in sites if s in run["predictions"]]

    high_risk = [p for p in caught if p.risk_score > SUDDEN_RISK_MIN]
    short_window = [p for p in caught if p.estimated_failure_window_hours < SUDDEN_WINDOW_MAX_HOURS]
    risks = ", ".join(f"{p.risk_score:.2f}" for p in caught) or "none"
    windows = ", ".join(f"{p.estimated_failure_window_hours:.1f}h" for p in caught) or "none"

    return [
        check(name, f"all {len(sites)} failures caught",
              f"{len(caught)}/{len(sites)} predicted", len(caught) == len(sites)),
        check(name, f"risk score > {SUDDEN_RISK_MIN} at every site",
              f"{len(high_risk)}/{len(sites)} sites (risk {risks})", len(high_risk) == len(sites)),
        check(name, f"failure window < {SUDDEN_WINDOW_MAX_HOURS} hours at every site",
              f"{len(short_window)}/{len(sites)} sites (windows {windows})",
              len(short_window) == len(sites)),
    ]


# --- Reporting -------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print the final markdown-style results table."""
    scenario_width = max(len("Scenario"), *(len(r["scenario"]) for r in results))
    expected_width = max(len("Expected"), *(len(r["expected"]) for r in results))
    actual_width = max(len("Actual"), *(len(r["actual"]) for r in results))

    def line(scenario: str, expected: str, actual: str, result: str) -> str:
        """Format one table row."""
        return (f"| {scenario:<{scenario_width}} | {expected:<{expected_width}} | "
                f"{actual:<{actual_width}} | {result:<6} |")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(line("Scenario", "Expected", "Actual", "Result"))
    print(f"|{'-' * (scenario_width + 2)}|{'-' * (expected_width + 2)}|"
          f"{'-' * (actual_width + 2)}|{'-' * 8}|")
    for r in results:
        print(line(r["scenario"], r["expected"], r["actual"], r["result"]))


def main() -> int:
    """Run all scenarios, print the summary, and return the exit code."""
    results = []
    for scenario in (scenario_healthy_baseline, scenario_degrading_detection,
                     scenario_sudden_failure):
        results.extend(scenario())

    print_summary(results)
    failed = [r for r in results if r["result"] == FAIL]
    verdict = "ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED"
    print(f"\n{verdict} ({len(results) - len(failed)}/{len(results)} passed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
