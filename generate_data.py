"""Generate the synthetic telemetry dataset and print a verification summary.

Usage: python generate_data.py
Writes data/network_telemetry.csv and data/site_scenarios.json.
"""

import sys

import pandas as pd

from app.config import DATA_DIR, HOURS_OF_DATA, SITE_COUNT
from app.data_generator import (
    COLUMNS,
    INTERVAL_MINUTES,
    POINTS_PER_HOUR,
    generate_dataset,
    save_dataset,
)

OUTPUT_FILE = DATA_DIR / "network_telemetry.csv"
NON_NEGATIVE_COLUMNS = [
    "cpu_utilization",
    "memory_utilization",
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "connected_users",
    "handover_failure_rate",
    "alarm_count",
]
PERCENT_COLUMNS = [
    "cpu_utilization",
    "memory_utilization",
    "packet_loss_pct",
    "handover_failure_rate",
]
SAMPLE_ROWS = 5
EXPECTED_ROWS = SITE_COUNT * HOURS_OF_DATA * POINTS_PER_HOUR
SEPARATOR = "-" * 60


def verify(df: pd.DataFrame) -> list[str]:
    """Return a list of data quality problems; empty means the data is valid."""
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append(f"Unexpected columns: {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        problems.append(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")
    null_counts = df.isna().sum()
    for column, count in null_counts[null_counts > 0].items():
        problems.append(f"{count} null values in {column}")
    for column in NON_NEGATIVE_COLUMNS:
        negatives = int((df[column] < 0).sum())
        if negatives:
            problems.append(f"{negatives} negative values in {column}")
    for column in PERCENT_COLUMNS:
        too_high = int((df[column] > 100).sum())
        if too_high:
            problems.append(f"{too_high} values above 100 in {column}")
    return problems


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset size, site breakdowns, date range, and a failing site sample."""
    scenarios = df.attrs["site_scenarios"]
    site_info = df.drop_duplicates("site_id").set_index("site_id")["network_type"]
    site_table = pd.DataFrame({"network_type": site_info, "scenario": pd.Series(scenarios)})

    print(SEPARATOR)
    print(f"Total rows:  {len(df)}")
    print(f"Total sites: {df['site_id'].nunique()}")
    print(f"Date range:  {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(SEPARATOR)
    print("Sites per network type:")
    print(site_table["network_type"].value_counts().to_string())
    print(SEPARATOR)
    print("Sites per scenario:")
    print(site_table["scenario"].value_counts().to_string())
    print(SEPARATOR)

    last = df.sort_values("timestamp").groupby("site_id").tail(1).set_index("site_id")
    final = pd.crosstab(site_table["scenario"], last["site_status"].reindex(site_table.index))
    print("Final site_status by scenario:")
    print(final.to_string())
    print(SEPARATOR)

    failing = [site for site, scenario in scenarios.items() if scenario == "FAILING"][0]
    print(f"Last {SAMPLE_ROWS} timestamps of FAILING site {failing}:")
    sample = df[df["site_id"] == failing].tail(SAMPLE_ROWS).drop(columns=["site_id"])
    print(sample.to_string(index=False))
    print(SEPARATOR)


def main() -> int:
    """Generate, save, verify, and report. Returns a process exit code."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    save_dataset(df, OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE} ({INTERVAL_MINUTES}-minute intervals)")

    print_summary(df)

    problems = verify(df)
    if problems:
        print("Data validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Dataset generated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
