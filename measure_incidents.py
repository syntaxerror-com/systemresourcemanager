"""
Measures the real effect of the automatic scheduler.

Workflow (do both runs under a comparable workload, e.g. 30-60 min each):
  1. python anomaly_detector.py --no-actions   -> logs Mode=monitor_only to check_log.csv
  2. python anomaly_detector.py                -> logs Mode=scheduler    to check_log.csv
  3. python measure_incidents.py

An "incident" = a check where at least one process was CRITICAL.
Only quote a percentage on your resume if THIS script reports it.
"""
import os
import sys
import pandas as pd

path = os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "check_log.csv")
df = pd.read_csv(path)
df["incident"] = df["Critical"] > 0
summary = df.groupby("Mode").agg(checks=("incident", "size"), incidents=("incident", "sum"))
summary["incident_rate_%"] = (100 * summary.incidents / summary.checks).round(2)
print(summary, "\n")
if {"monitor_only", "scheduler"} <= set(summary.index):
    base = summary.loc["monitor_only", "incident_rate_%"]
    sched = summary.loc["scheduler", "incident_rate_%"]
    if base > 0:
        print(f"Critical-incident rate reduction with scheduler: {100 * (base - sched) / base:.1f}%")
    else:
        print("Baseline had no incidents; cannot compute a reduction.")
else:
    print("Need runs in both modes (monitor_only and scheduler) to compare.")
