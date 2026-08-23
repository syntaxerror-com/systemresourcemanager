import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import subprocess
import time
from datetime import datetime

PROCESS_CSV  = r"D:\PROJECTS\SystemResourceMonitor\x64\Debug\process_log.csv"
ALERT_FILE   = r"D:\PROJECTS\SystemResourceMonitor\x64\Debug\alerts.txt"
CHECK_EVERY  = 15  # seconds between each check

def load_data():
    if not os.path.exists(PROCESS_CSV):
        return None
    df = pd.read_csv(PROCESS_CSV)
    df.columns = df.columns.str.strip()
    df['Memory_MB'] = pd.to_numeric(df['Memory_MB'], errors='coerce')
    df = df.dropna()
    return df

def build_baselines(df):
    baselines = df.groupby('ProcessName')['Memory_MB'].agg(
        mean='mean',
        std='std',
        max='max'
    ).reset_index()
    baselines['std'] = baselines['std'].fillna(0)
    high_thresh   = baselines['mean'].quantile(0.85)
    low_thresh    = baselines['mean'].quantile(0.50)
    high_tier_avg = baselines[baselines['mean'] >= high_thresh]['mean'].mean()

    def classify(m):
        if m >= high_thresh: return 'HIGH'
        elif m >= low_thresh: return 'MEDIUM'
        else: return 'LOW'

    baselines['Tier'] = baselines['mean'].apply(classify)
    return baselines, high_tier_avg

def check_anomalies(df, baselines, high_tier_avg):
    df = df.merge(baselines[['ProcessName', 'mean', 'std', 'Tier']], on='ProcessName', how='left')

    # Only look at the LATEST snapshot (last timestamp)
    latest_ts = df['Timestamp'].max()
    latest    = df[df['Timestamp'] == latest_ts]

    alerts = []
    for _, row in latest.iterrows():
        mem     = row['Memory_MB']
        avg     = row['mean']
        std     = row['std']
        tier    = row['Tier']
        proc    = row['ProcessName']
        z_score = (mem - avg) / std if std > 0 else 0

        if mem > high_tier_avg and tier != 'HIGH':
            alerts.append(('CRITICAL', proc, mem, avg, tier, z_score))
        elif mem > high_tier_avg and tier == 'HIGH':
            alerts.append(('CRITICAL', proc, mem, avg, tier, z_score))
        elif z_score > 3 or mem > avg * 3:
            alerts.append(('WARNING', proc, mem, avg, tier, z_score))

    return alerts, latest_ts

def write_alerts(alerts, timestamp):
    with open(ALERT_FILE, 'w') as f:
        f.write(f"TIMESTAMP:{timestamp}\n")
        if not alerts:
            f.write("STATUS:OK\n")
        else:
            for level, proc, mem, avg, tier, z in alerts:
                f.write(f"{level}:{proc}:{mem:.0f}:{avg:.1f}:{tier}:{z:.2f}\n")

KILL_THRESHOLD = 1500  # MB — extreme for 12GB RAM

def check_and_kill(df):
    latest_ts = df['Timestamp'].max()
    latest    = df[df['Timestamp'] == latest_ts]
    heavy     = latest[latest['Memory_MB'] > KILL_THRESHOLD]

    for _, row in heavy.iterrows():
        proc = row['ProcessName']
        mem  = row['Memory_MB']
        pid  = int(row['PID'])

        print(f"\n  [!!] {proc} is using {mem:.0f} MB (PID: {pid})")
        print(f"  This is extremely high for a 12GB RAM system.")
        answer = input(f"  Kill {proc}? (y/n): ").strip().lower()

        if answer == 'y':
            os.system(f"taskkill /F /PID {pid}")
            print(f"  [OK] {proc} (PID {pid}) has been killed.\n")
        else:
            print(f"  [SKIPPED] {proc} left running.\n")

def run_realtime():
    print("\n=== Real-Time Process Anomaly Monitor ===")
    print(f"Checking every {CHECK_EVERY} seconds...")
    print("Press Ctrl+C to stop.\n")

    prev_alert_count = 0

    while True:
        df = load_data()

        if df is None or len(df) < 10:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for data...")
            time.sleep(CHECK_EVERY)
            continue

        baselines, high_avg = build_baselines(df)
        alerts, ts          = check_anomalies(df, baselines, high_avg)
        check_and_kill(df)
        write_alerts(alerts, ts)

        now = datetime.now().strftime('%H:%M:%S')

        if not alerts:
            print(f"[{now}] All processes normal.")
        else:
            print(f"\n[{now}] === {len(alerts)} ALERT(S) DETECTED ===")
            for level, proc, mem, avg, tier, z in alerts:
                tag = "[CRITICAL]" if level == "CRITICAL" else "[WARNING] "
                print(f"  {tag} {proc:<35} "
                      f"Using: {mem:.0f} MB | "
                      f"Normal: {avg:.1f} MB | "
                      f"Tier: {tier} | "
                      f"Z: {z:.2f}")
            print()

        prev_alert_count = len(alerts)
        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    try:
        run_realtime()
    except KeyboardInterrupt:
        print("\n[STOPPED] Monitor shut down.")