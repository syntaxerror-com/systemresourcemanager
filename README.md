# Predictive System Resource Manager

C++ telemetry collector + Python anomaly detection (z-score **and** KNN) + automatic task scheduling, for Windows.

## How it works
1. `SystemResourceMonitor.exe` (C++) logs system metrics and per-process memory/CPU every 3 s
   (PDH, ToolHelp32, PSAPI, `GetProcessTimes`, `GetSystemPowerStatus`), including **battery drain rate (%/min)**.
2. `anomaly_detector.py` loads `process_log.csv` and scores the newest snapshot two ways:
   - **Z-score** of each process vs its own history (memory and CPU)
   - **KNN** distance in a scaled feature space (memory, process CPU, system CPU, system RAM, battery drain)
   - `CRITICAL` = both methods flag it, `WARNING` = one method flags it
3. **Automatic task scheduling:** CRITICAL processes get `THROTTLE` actions; the C++ monitor lowers their
   priority class (`SetPriorityClass`) and restores it when they return to normal. Protected system
   processes are never touched. Press `T` in the monitor to toggle it.

## Run
```
pip install -r requirements.txt
# build SystemResourceMonitor.cpp in Visual Studio (x64) or:  cl /EHsc /std:c++17 SystemResourceMonitor\SystemResourceMonitor.cpp
# terminal 1 (in the exe's folder):  SystemResourceMonitor.exe
# terminal 2 (same folder):          python <path>\anomaly_detector.py
python -m pytest tests
```

## Measuring the incident reduction (do this before quoting a %)
```
python anomaly_detector.py --no-actions     # 30-60 min, normal workload  -> baseline
python anomaly_detector.py                  # 30-60 min, same workload    -> scheduler ON
python measure_incidents.py
```
