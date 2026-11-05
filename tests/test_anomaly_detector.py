import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import anomaly_detector as ad  # noqa: E402


def make_log(spike=True, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n + 1):
        ts = f"2026-01-01 10:{i // 60:02d}:{i % 60:02d}"
        sys_cpu, sys_mem, drain = rng.normal(40, 3), rng.normal(70, 2), rng.normal(0.3, 0.02)
        for pid, name, mem, cpu in [(100, "chrome.exe", 400, 10), (200, "code.exe", 300, 6),
                                    (300, "notepad.exe", 40, 0.5), (400, "hog.exe", 200, 5)]:
            m = mem + rng.normal(0, mem * 0.03)
            c = max(0, cpu + rng.normal(0, 0.5))
            if spike and i == n and name == "hog.exe":
                m, c = 1800, 70
            rows.append((ts, pid, name, m, c, sys_cpu, sys_mem, drain))
    return pd.DataFrame(rows, columns=["Timestamp", "PID", "ProcessName", "Memory_MB",
                                       "CPU_Percent", "Sys_CPU", "Sys_Mem", "Battery_Drain"])


def test_spike_flagged_by_both_methods_is_critical():
    res, _ = ad.detect(make_log(spike=True))
    hog = res[res.ProcessName == "hog.exe"].iloc[0]
    assert hog.Z_Flag and hog.KNN_Flag and hog.Alert == "CRITICAL"


def test_normal_snapshot_has_no_critical():
    res, _ = ad.detect(make_log(spike=False))
    assert (res.Alert == "CRITICAL").sum() == 0


def test_zscore_alone_when_history_too_short_for_knn():
    df = make_log(spike=True, n=5)           # 20 history rows < MIN_KNN_HISTORY
    res, _ = ad.detect(df)
    hog = res[res.ProcessName == "hog.exe"].iloc[0]
    assert hog.Z_Flag and not hog.KNN_Flag and hog.Alert == "WARNING"


def test_scheduler_throttles_then_restores():
    res, _ = ad.detect(make_log(spike=True))
    actions, state = ad.plan_scheduler_actions(res, {})
    assert ("THROTTLE", "400", "hog.exe") in actions and "400" in state
    ok = res.copy()
    ok["Alert"] = "OK"
    actions2, state2 = ad.plan_scheduler_actions(ok, state)
    assert ("RESTORE", "400", "hog.exe") in actions2 and "400" not in state2


def test_protected_processes_never_throttled():
    res, _ = ad.detect(make_log(spike=True))
    res.loc[res.ProcessName == "hog.exe", "ProcessName"] = "csrss.exe"
    actions, _ = ad.plan_scheduler_actions(res, {})
    assert actions == []


def test_end_to_end_writes_alert_and_action_files(tmp_path):
    make_log(spike=True).to_csv(tmp_path / "process_log.csv", index=False)
    res, state, ts = ad.check_once(str(tmp_path), scheduler=True)
    assert "CRITICAL:hog.exe" in (tmp_path / "alerts.txt").read_text()
    assert "THROTTLE:400:hog.exe" in (tmp_path / "scheduler_actions.txt").read_text()
    assert (tmp_path / "check_log.csv").exists()
