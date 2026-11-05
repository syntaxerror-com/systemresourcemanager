"""
Predictive anomaly detector for SystemResourceMonitor.

Reads process_log.csv written by the C++ monitor and flags heavy / abnormal
processes using TWO independent methods:

  1. Z-score   - per-process memory/CPU compared to that process's own history.
  2. KNN       - k-nearest-neighbour distance in a scaled feature space
                 (Memory, process CPU, system CPU, system RAM, battery drain rate).

Result per process in the latest snapshot:
  CRITICAL = flagged by BOTH z-score and KNN
  WARNING  = flagged by ONE of them
  OK       = neither

CRITICAL processes are handed to the C++ monitor as THROTTLE actions
(automatic task scheduling: priority is lowered, and restored once the
process behaves normally again).

Usage (run from the same folder the C++ exe runs in, or pass --data-dir):
    python anomaly_detector.py                  # live loop, scheduler ON
    python anomaly_detector.py --no-actions     # monitor-only (for baseline measurement)
    python anomaly_detector.py --once           # single check, then exit
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = ["Memory_MB", "CPU_Percent", "Sys_CPU", "Sys_Mem", "Battery_Drain"]
REQUIRED = ["Timestamp", "PID", "ProcessName", "Memory_MB"]

Z_THRESHOLD = 3.0          # z-score above this => statistical outlier
RATIO_THRESHOLD = 3.0      # or memory > 3x the process's normal
MIN_FLAG_MB = 100          # ignore tiny processes for the ratio rule
MIN_PROC_HISTORY = 5       # samples needed before a process z-score is trusted
MIN_KNN_HISTORY = 30       # rows needed before KNN is trusted
KNN_K = 5
KNN_PERCENTILE = 99        # training-score percentile used as KNN threshold
KNN_MARGIN = 1.25          # safety margin: system-wide features are shared by every row in a snapshot
MAX_HISTORY_ROWS = 5000

PROTECTED = {
    "system", "system idle process", "registry", "smss.exe", "csrss.exe",
    "wininit.exe", "services.exe", "lsass.exe", "winlogon.exe", "svchost.exe",
    "dwm.exe", "explorer.exe", "memcompression", "fontdrvhost.exe",
    "python.exe", "systemresourcemonitor.exe",
}


# ----------------------------------------------------------------- data ----
def load_data(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    if any(c not in df.columns for c in REQUIRED):
        return None
    for col in FEATURES:                      # older logs may lack some columns
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["PID"] = pd.to_numeric(df["PID"], errors="coerce")
    df = df.dropna(subset=["Memory_MB", "PID"]).fillna({c: 0.0 for c in FEATURES})
    return df.tail(MAX_HISTORY_ROWS * 4).reset_index(drop=True)


def split_history_latest(df):
    """Latest snapshot = rows with the newest timestamp; everything before = history."""
    latest_ts = df["Timestamp"].max()
    latest = df[df["Timestamp"] == latest_ts].copy()
    history = df[df["Timestamp"] != latest_ts]
    return history.tail(MAX_HISTORY_ROWS), latest, latest_ts


# ------------------------------------------------------------- baselines ---
def build_baselines(history):
    g = history.groupby("ProcessName")
    b = g.agg(mean=("Memory_MB", "mean"), std=("Memory_MB", "std"),
              cpu_mean=("CPU_Percent", "mean"), cpu_std=("CPU_Percent", "std"),
              n=("Memory_MB", "count")).reset_index()
    b[["std", "cpu_std"]] = b[["std", "cpu_std"]].fillna(0)
    if len(b):
        hi, lo = b["mean"].quantile(0.85), b["mean"].quantile(0.50)
        b["Tier"] = np.where(b["mean"] >= hi, "HIGH", np.where(b["mean"] >= lo, "MEDIUM", "LOW"))
    else:
        b["Tier"] = []
    return b


def _safe_z(value, mean, std):
    floor = np.maximum(std, np.maximum(1.0, 0.05 * np.abs(mean)))  # avoid /0 on flat processes
    return (value - mean) / floor


def zscore_flags(latest, baselines):
    m = latest.merge(baselines, on="ProcessName", how="left")
    known = m["n"].fillna(0) >= MIN_PROC_HISTORY
    z_mem = np.where(known, _safe_z(m["Memory_MB"], m["mean"], m["std"]), 0.0)
    z_cpu = np.where(known, _safe_z(m["CPU_Percent"], m["cpu_mean"], m["cpu_std"]), 0.0)
    z = np.maximum(z_mem, z_cpu)
    ratio = known & (m["Memory_MB"] > RATIO_THRESHOLD * m["mean"]) & (m["Memory_MB"] > MIN_FLAG_MB)
    m["Z_Score"] = np.round(z, 2)
    m["Z_Flag"] = (z > Z_THRESHOLD) | ratio
    m["Tier"] = m["Tier"].fillna("NEW")
    m["Avg_MB"] = m["mean"].fillna(m["Memory_MB"]).round(1)
    return m


# ------------------------------------------------------------------- KNN ---
def fit_knn(history, k=KNN_K, percentile=KNN_PERCENTILE):
    """Fit scaler + neighbour index on history; return model or None if too little data."""
    if len(history) < MIN_KNN_HISTORY:
        return None
    X = history[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    d, _ = nn.kneighbors(Xs)
    train_scores = d[:, 1:].mean(axis=1)            # drop self-distance
    return {"scaler": scaler, "nn": nn, "k": k,
            "threshold": float(np.percentile(train_scores, percentile)) * KNN_MARGIN}


def knn_scores(model, latest):
    if model is None:
        return np.zeros(len(latest)), np.zeros(len(latest), dtype=bool)
    Xs = model["scaler"].transform(latest[FEATURES].to_numpy(dtype=float))
    d, _ = model["nn"].kneighbors(Xs, n_neighbors=model["k"])
    scores = d.mean(axis=1)
    return scores, scores > model["threshold"]


# ---------------------------------------------------------------- detect ---
def detect(df):
    """Run z-score + KNN on the latest snapshot. Returns (results_df, timestamp)."""
    history, latest, ts = split_history_latest(df)
    baselines = build_baselines(history)
    res = zscore_flags(latest, baselines)
    model = fit_knn(history)
    scores, flags = knn_scores(model, res)
    res["KNN_Score"] = np.round(scores, 4)
    res["KNN_Flag"] = flags
    res["Alert"] = np.where(res.Z_Flag & res.KNN_Flag, "CRITICAL",
                    np.where(res.Z_Flag | res.KNN_Flag, "WARNING", "OK"))
    return res.sort_values(["Alert", "Memory_MB"], ascending=[True, False]), ts


# --------------------------------------------------------------- outputs ---
def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_alerts(res, ts, path):
    lines = [f"TIMESTAMP:{ts}"]
    flagged = res[res.Alert != "OK"]
    if flagged.empty:
        lines.append("STATUS:OK")
    for _, r in flagged.iterrows():
        # LEVEL:proc:mem:avg:tier:z:knn
        lines.append(f"{r.Alert}:{r.ProcessName}:{r.Memory_MB:.0f}:{r.Avg_MB:.1f}:"
                     f"{r.Tier}:{r.Z_Score:.2f}:{r.KNN_Score:.3f}")
    _atomic_write(path, "\n".join(lines) + "\n")


def plan_scheduler_actions(res, state):
    """
    Simple automatic task scheduling:
      CRITICAL (z-score AND KNN agree) -> THROTTLE (lower priority class)
      previously throttled and now OK  -> RESTORE
    `state` maps pid(str) -> process name for currently throttled processes.
    """
    actions, new_state = [], dict(state)
    for _, r in res.iterrows():
        pid = str(int(r.PID))
        name = r.ProcessName
        if r.Alert == "CRITICAL" and name.lower() not in PROTECTED and pid not in new_state:
            actions.append(("THROTTLE", pid, name))
            new_state[pid] = name
        elif r.Alert == "OK" and pid in new_state:
            actions.append(("RESTORE", pid, name))
            del new_state[pid]
    return actions, new_state


def write_actions(actions, path):
    if actions:
        _atomic_write(path, "\n".join(f"{a}:{p}:{n}" for a, p, n in actions) + "\n")


def log_check(path, ts, mode, res, n_actions):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("Timestamp,Mode,Critical,Warning,Throttled\n")
        f.write(f"{ts},{mode},{(res.Alert == 'CRITICAL').sum()},"
                f"{(res.Alert == 'WARNING').sum()},{n_actions}\n")


def save_flagged(res, ts, path):
    flagged = res[res.Alert != "OK"][["ProcessName", "Memory_MB", "CPU_Percent", "Avg_MB",
                                      "Z_Score", "Tier", "Alert", "KNN_Score", "KNN_Flag"]].copy()
    flagged.insert(0, "Timestamp", ts)
    flagged.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ------------------------------------------------------------------ main ---
def check_once(data_dir, scheduler=True, state=None, last_ts=None):
    state = state or {}
    df = load_data(os.path.join(data_dir, "process_log.csv"))
    if df is None or df["Timestamp"].nunique() < 3:
        return None, state, last_ts
    res, ts = detect(df)
    if ts == last_ts:                      # no new snapshot yet
        return None, state, last_ts
    write_alerts(res, ts, os.path.join(data_dir, "alerts.txt"))
    save_flagged(res, ts, os.path.join(data_dir, "process_anomaly_flagged.csv"))
    actions = []
    if scheduler:
        actions, state = plan_scheduler_actions(res, state)
        write_actions(actions, os.path.join(data_dir, "scheduler_actions.txt"))
    log_check(os.path.join(data_dir, "check_log.csv"), ts,
              "scheduler" if scheduler else "monitor_only", res, len(actions))
    return res, state, ts


def interactive_kill(res, kill_mb=1500):
    for _, r in res[res.Memory_MB > kill_mb].iterrows():
        if input(f"  Kill {r.ProcessName} ({r.Memory_MB:.0f} MB)? (y/n): ").strip().lower() == "y":
            os.system(f"taskkill /F /PID {int(r.PID)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SRM_DATA_DIR", "."))
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--no-actions", action="store_true", help="monitor only, no auto-scheduling")
    ap.add_argument("--interactive-kill", action="store_true")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    state_path = os.path.join(a.data_dir, "scheduler_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    last_ts = None
    print(f"=== Predictive Anomaly Monitor (z-score + KNN) | scheduler "
          f"{'OFF' if a.no_actions else 'ON'} ===")
    while True:
        res, state, last_ts = check_once(a.data_dir, not a.no_actions, state, last_ts)
        now = datetime.now().strftime("%H:%M:%S")
        if res is None:
            print(f"[{now}] Waiting for new data...")
        else:
            bad = res[res.Alert != "OK"]
            print(f"[{now}] {len(bad)} flagged "
                  f"({(res.Alert == 'CRITICAL').sum()} critical, {(res.Alert == 'WARNING').sum()} warning)")
            for _, r in bad.head(10).iterrows():
                print(f"   [{r.Alert:<8}] {r.ProcessName:<30} {r.Memory_MB:>6.0f} MB  "
                      f"CPU {r.CPU_Percent:>5.1f}%  z={r.Z_Score:>5.2f}  knn={r.KNN_Score:.2f}")
            if a.interactive_kill:
                interactive_kill(res)
            _atomic_write(state_path, json.dumps(state))
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Monitor shut down.")
