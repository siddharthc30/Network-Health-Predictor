"""Central configuration for the network health monitoring system.

All paths, model settings, and KPI thresholds live here so that every
other module imports its constants from a single place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# --- Environment and model -----------------------------------------------

load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Strong reasoning and structured-output support at reasonable latency
MODEL_NAME = "gpt-4o"

# --- Health score bands (0-100) ------------------------------------------

# 80+ means all KPIs are comfortably within normal operating range
HEALTH_SCORE_HEALTHY = 80
# 60-79 signals early drift that deserves watching but not yet action
HEALTH_SCORE_WARNING = 60
# 40-59 means service quality is likely already impacted for some users
HEALTH_SCORE_DEGRADED = 40
# Anything below 40 is critical; 0 is the floor of the scale
HEALTH_SCORE_CRITICAL = 0

# --- Resource utilization (percent) --------------------------------------

# Above 90% CPU there is no headroom for traffic bursts and failures follow
CPU_CRITICAL = 90
# 75% leaves about 15 points of headroom, enough time to react
CPU_WARNING = 75

# Memory leaks and exhaustion cause abrupt crashes, so match the CPU limits
MEMORY_CRITICAL = 90
# 75% gives early notice of a leak before the site becomes unstable
MEMORY_WARNING = 75

# --- Latency (milliseconds) ----------------------------------------------

# 5G user-plane SLA is typically under 10 ms, so 50 ms is a clear breach
LATENCY_CRITICAL_5G = 50
# 30 ms is three times the SLA target and gives an early signal
LATENCY_WARNING_5G = 30

# LTE typically delivers 30-50 ms, so 100 ms means serious degradation
LATENCY_CRITICAL_LTE = 100
# 60 ms is above normal LTE behavior but still usable, which makes it a warning
LATENCY_WARNING_LTE = 60

# --- Packet loss (percent) -----------------------------------------------

# Above 5% voice and video sessions become unusable and TCP throughput collapses
PACKET_LOSS_CRITICAL = 5.0
# Carrier targets are usually under 1%, so 2% flags a developing problem
PACKET_LOSS_WARNING = 2.0

# --- Throughput (ratio of current value to baseline) ---------------------

# Losing half of baseline throughput is a severe capacity or radio problem
THROUGHPUT_DROP_CRITICAL = 0.5
# A 30% drop is beyond normal daily variation and worth investigating
THROUGHPUT_DROP_WARNING = 0.7

# --- Signal quality (SINR in dB) -----------------------------------------

# At or below 0 dB, noise equals signal and links are barely decodable
SINR_CRITICAL = 0
# Below 5 dB, modulation falls back to low rates and cell edge users suffer
SINR_WARNING = 5

# --- Alarms (count per interval) -----------------------------------------

# Five or more simultaneous alarms usually indicate a correlated fault
ALARM_COUNT_CRITICAL = 5
# Two alarms are common in isolation but suggest a fault when persistent
ALARM_COUNT_WARNING = 2

# --- Handover failure rate (percent of failed handovers between cells) ----

# Above 15% handover failures causes dropped calls and session resets
HANDOVER_FAILURE_RATE_CRITICAL = 15.0
# Above 8% indicates mobility issues, roughly twice a healthy 5G cell's worst case
HANDOVER_FAILURE_RATE_WARNING = 8.0

# --- "Good" reference points (KPI sub-score is 100 at or better) ---------

# Healthy sites run at 30-60% CPU, so 60% is the top of normal operation
CPU_GOOD = 60
# Healthy sites hold memory at 35-60%, so 60% is the top of normal operation
MEMORY_GOOD = 60
# Top of the normal 5G latency range (5-15 ms)
LATENCY_GOOD_5G = 15
# Top of the normal LTE latency range (20-40 ms)
LATENCY_GOOD_LTE = 40
# Top of the normal packet loss range (0.01-1.0%) across both technologies
PACKET_LOSS_GOOD = 1.0
# SINR of 10 dB or more supports higher-order modulation and is widely treated as good
SINR_GOOD = 10
# A healthy site raises no alarms
ALARM_COUNT_GOOD = 0
# Top of the normal handover failure range (0.5-4%) across both technologies
HANDOVER_FAILURE_RATE_GOOD = 4.0

# Expected healthy throughput per technology (Mbps). Set at the lower quartile
# of the normal range (5G 800-1200, LTE 100-300) so a healthy site that is
# naturally at the low end is not penalized.
THROUGHPUT_BASELINE_5G_MBPS = 900
THROUGHPUT_BASELINE_LTE_MBPS = 150

# Absolute throughput levels derived from the baseline and the drop ratios above
THROUGHPUT_GOOD_5G_MBPS = THROUGHPUT_BASELINE_5G_MBPS
THROUGHPUT_WARNING_5G_MBPS = THROUGHPUT_BASELINE_5G_MBPS * THROUGHPUT_DROP_WARNING
THROUGHPUT_CRITICAL_5G_MBPS = THROUGHPUT_BASELINE_5G_MBPS * THROUGHPUT_DROP_CRITICAL
THROUGHPUT_GOOD_LTE_MBPS = THROUGHPUT_BASELINE_LTE_MBPS
THROUGHPUT_WARNING_LTE_MBPS = THROUGHPUT_BASELINE_LTE_MBPS * THROUGHPUT_DROP_WARNING
THROUGHPUT_CRITICAL_LTE_MBPS = THROUGHPUT_BASELINE_LTE_MBPS * THROUGHPUT_DROP_CRITICAL

# --- KPI sub-score anchors (0-100 scale) ---------------------------------

# A KPI at or better than its good reference is fully healthy
SUBSCORE_GOOD = 100
# A KPI at its warning threshold scores 60, a clearly reduced but usable level
SUBSCORE_WARNING = 60
# A KPI at its critical threshold scores 20, a severe but not yet total loss
SUBSCORE_CRITICAL = 20
# The sub-score never goes below zero
SUBSCORE_FLOOR = 0

# --- Prediction and output settings --------------------------------------

# Four hours gives field teams enough lead time to dispatch and intervene
PREDICTION_WINDOW_HOURS = 4
# Keeps dashboards and AI prompts focused on the highest-risk sites
MAX_RESULTS = 10
FALLBACK_MESSAGE = "Unable to generate analysis. Please check the data and try again."

# --- Synthetic data generation -------------------------------------------

# Large enough for fleet-level patterns, small enough to run locally
SITE_COUNT = 50
# One full daily traffic cycle captures peak and off-peak behavior
HOURS_OF_DATA = 24
