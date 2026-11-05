"""
Timeline graphs for the anomaly detector (replacement for the old
"Process Memory - KNN Anomaly Detection" chart).

What was wrong with the old chart
  * x-axis was "Sample" (row number), so rows from DIFFERENT processes were
    stitched together and every line jumped up and down -> the vertical spikes.
  * Red dots were drawn at the raw sample index, not on a process's own line.
  * Only memory, no way to see when alerts happened or which process caused them.

What this script does
  * Uses REAL TIME on the x-axis and one clean line per process.
  * Replays the real detector (z-score + KNN) over the whole log, so every
    marker is an alert the detector would actually have raised at that moment.
  * Breaks lines where the monitor was stopped, so gaps are not drawn as data.

Outputs (written into --data-dir)
  timeline_report.png  - dashboard: memory + alerts + system load + worst offenders
  process_detail.png   - one panel per process: memory, its own normal band, alerts

Usage (same folder as the other scripts):
    python plot_timeline.py --data-dir SystemResourceMonitor
    python plot_timeline.py --data-dir SystemResourceMonitor --top 8 --last-minutes 20
    python plot_timeline.py --data-dir SystemResourceMonitor --step 3   # faster replay
    python plot_timeline.py --data-dir SystemResourceMonitor --show
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

import anomaly_detector as ad

RANK = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
COLORS = {"WARNING": "#f0a020", "CRITICAL": "#d93a3a"}
PALETTE = ["#4c78a8", "#59a14f", "#b07aa1", "#76b7b2", "#9c755f", "#7f7f7f",
           "#bab0ac", "#8cd17d", "#a0cbe8", "#d4a6c8"]


# ---------------------------------------------------------------- replay ----
def replay(df, warmup=10, step=1):
    """
    Run the real detector on every snapshot (after `warmup` snapshots of history)
    and return one row per (snapshot, process) with its alert level.
    """
    stamps = sorted(df["Timestamp"].unique())
    if len(stamps) <= warmup:
        raise SystemExit(f"Need more than {warmup} snapshots, found {len(stamps)}. "
                         "Let the C++ monitor run a few more minutes.")
    todo = stamps[warmup::step]
    parts = []
    for i, ts in enumerate(todo, 1):
        res, _ = ad.detect(df[df["Timestamp"] <= ts])
        parts.append(res[["Timestamp", "ProcessName", "PID", "Memory_MB", "Alert"]])
        if i % 50 == 0 or i == len(todo):
            print(f"  replayed {i}/{len(todo)} snapshots", end="\r")
    print()
    rep = pd.concat(parts, ignore_index=True)
    rep["T"] = pd.to_datetime(rep["Timestamp"])
    rep["Rank"] = rep["Alert"].map(RANK)
    return rep


# --------------------------------------------------------------- helpers ----
def memory_wide(df, names):
    """Time x process table of memory (a process with many PIDs -> its biggest one)."""
    d = df[df.ProcessName.isin(names)].copy()
    d["T"] = pd.to_datetime(d["Timestamp"])
    return d.groupby(["T", "ProcessName"])["Memory_MB"].max().unstack("ProcessName")[names]


def break_gaps(wide, factor=5):
    """Insert NaN rows where the monitor was stopped, so lines are not drawn across gaps."""
    idx = wide.index.to_series()
    dt = idx.diff().dt.total_seconds()
    limit = max(10.0, factor * dt.median())
    holes = idx[dt > limit] - pd.Timedelta(seconds=1)
    if holes.empty:
        return wide
    gap_rows = pd.DataFrame(np.nan, index=holes.values, columns=wide.columns)
    return pd.concat([wide, gap_rows]).sort_index()


def worst_alert(rep, names):
    """Worst alert level per (time, process), only for flagged rows."""
    f = rep[(rep.Rank > 0) & rep.ProcessName.isin(names)]
    return (f.groupby(["T", "ProcessName"], as_index=False)
             .agg(Rank=("Rank", "max"), Memory_MB=("Memory_MB", "max")))


def snap_to_line(flagged, wide):
    """A process can have several PIDs; the line shows its biggest one. Put each marker
    on that line (at the alert's time) so markers never float off the curve."""
    long = wide.stack().rename("Line_MB").reset_index()
    long.columns = ["T", "ProcessName", "Line_MB"]
    m = flagged.merge(long, on=["T", "ProcessName"], how="left")
    m["Memory_MB"] = m["Line_MB"].fillna(m["Memory_MB"])
    return m.drop(columns="Line_MB")


def draw_markers(ax, flagged, name, label=False):
    for rank, level, mk, size in ((1, "WARNING", "o", 38), (2, "CRITICAL", "X", 70)):
        p = flagged[(flagged.ProcessName == name) & (flagged.Rank == rank)]
        if not p.empty:
            ax.scatter(p["T"], p["Memory_MB"], s=size, marker=mk, c=COLORS[level],
                       edgecolor="black", linewidth=0.5, zorder=4,
                       label=level.title() if label else None)


def style_time_axis(ax):
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.grid(alpha=0.25)


# ------------------------------------------------------------- dashboard ----
def plot_dashboard(df, rep, top, out_path, show):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    seen = df[pd.to_datetime(df.Timestamp) >= rep["T"].min()]
    names = (seen.groupby("ProcessName")["Memory_MB"].max()
                 .sort_values(ascending=False).head(top).index.tolist())
    raw = memory_wide(seen, names)
    flagged = snap_to_line(worst_alert(rep, names), raw)
    wide = break_gaps(raw)

    per_snap = rep.groupby("T").agg(
        critical=("Alert", lambda s: (s == "CRITICAL").sum()),
        warning=("Alert", lambda s: (s == "WARNING").sum()))
    incident_pct = 100 * (per_snap.critical > 0).mean()
    step_s = max(1.0, pd.Series(per_snap.index).diff().dt.total_seconds().median())

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.4, 1], height_ratios=[2.3, 1, 1],
                          hspace=0.28, wspace=0.18)
    fig.suptitle("Process anomaly timeline  |  z-score + KNN replayed over the whole log",
                 fontsize=15, fontweight="bold")

    # 1) memory per process with alert markers ON the lines
    ax1 = fig.add_subplot(gs[0, 0])
    for i, n in enumerate(names):
        ax1.plot(wide.index, wide[n], lw=1.3, color=PALETTE[i % len(PALETTE)], label=n)
        draw_markers(ax1, flagged, n, label=(i == 0))
    ax1.set_yscale("log")
    ax1.set_ylabel("memory (MB, log scale)")
    ax1.set_title(f"1) Memory of the {len(names)} heaviest processes - markers = alerts raised for that process (any PID)")
    style_time_axis(ax1)
    h, l = ax1.get_legend_handles_labels()
    h += [Line2D([], [], marker="X", ls="", mfc=COLORS["CRITICAL"], mec="black", ms=9)]
    l += ["Critical (z-score AND KNN)"]
    keep = [(a, b) for a, b in zip(h, l) if b not in ("Warning", "Critical")]
    if (flagged.Rank == 1).any():
        keep.append((Line2D([], [], marker="o", ls="", mfc=COLORS["WARNING"], mec="black", ms=7),
                     "Warning (one method)"))
    ax1.legend(*zip(*keep), ncol=3, fontsize=8, loc="upper left")

    # summary box
    axs = fig.add_subplot(gs[0, 1])
    axs.axis("off")
    span = (rep["T"].max() - rep["T"].min()).total_seconds() / 60
    top_critical = (rep[rep.Alert == "CRITICAL"].ProcessName.value_counts().head(1))
    worst = f"{top_critical.index[0]} ({top_critical.iloc[0]}x)" if len(top_critical) else "none"
    axs.text(0, 1, "Summary\n", fontsize=13, fontweight="bold", va="top")
    axs.text(0, 0.88,
             f"Snapshots analysed:   {len(per_snap)}\n"
             f"Time span:            {span:.0f} min\n"
             f"Critical alerts:      {int(per_snap.critical.sum())}\n"
             f"Warning alerts:       {int(per_snap.warning.sum())}\n"
             f"Snapshots with a\n"
             f"critical incident:    {incident_pct:.1f}%\n"
             f"Most critical:        {worst}\n",
             fontsize=10.5, va="top", family="monospace")

    # 2) alerts per snapshot
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    w = 0.9 * step_s / 86400
    ax2.bar(per_snap.index, per_snap.critical, width=w, color=COLORS["CRITICAL"], label="Critical")
    ax2.bar(per_snap.index, per_snap.warning, width=w, bottom=per_snap.critical,
            color=COLORS["WARNING"], label="Warning")
    ax2.set_ylabel("flagged processes")
    ax2.set_title("2) Alerts per snapshot - tall or dense bars = trouble")
    ax2.legend(fontsize=8, loc="upper left")
    style_time_axis(ax2)

    # 3) system load
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    sysd = (seen.assign(T=pd.to_datetime(seen.Timestamp))
                .groupby("T")[["Sys_CPU", "Sys_Mem"]].first())
    sysd = break_gaps(sysd)
    ax3.plot(sysd.index, sysd.Sys_CPU, color="#4c78a8", lw=1.2, label="System CPU %")
    ax3.plot(sysd.index, sysd.Sys_Mem, color="#b07aa1", lw=1.2, label="System RAM %")
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("percent")
    ax3.set_title("3) Whole-machine load - do alerts line up with CPU or RAM pressure?")
    ax3.legend(fontsize=8, loc="lower left", ncol=2)
    style_time_axis(ax3)

    # 4) who triggers the most alerts
    ax4 = fig.add_subplot(gs[1:, 1])
    counts = (rep[rep.Rank > 0].groupby(["ProcessName", "Alert"]).size().unstack(fill_value=0)
              .reindex(columns=["CRITICAL", "WARNING"], fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False).head(10).iloc[::-1]
    if counts.empty:
        ax4.text(0.5, 0.5, "No alerts raised", ha="center", va="center")
        ax4.axis("off")
    else:
        ax4.barh(counts.index, counts["CRITICAL"], color=COLORS["CRITICAL"], label="Critical")
        ax4.barh(counts.index, counts["WARNING"], left=counts["CRITICAL"],
                 color=COLORS["WARNING"], label="Warning")
        ax4.legend(fontsize=8, loc="lower right")
        ax4.tick_params(axis="y", labelsize=8)
    ax4.set_xlabel("alerts")
    ax4.set_title("4) Top alert sources")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------- per-process -----
def plot_detail(df, rep, top, out_path, show, window=60):
    """One panel per process: memory, its rolling normal band (mean +/- 3 std), alerts."""
    import matplotlib.pyplot as plt

    alerts = rep[rep.Rank > 0].ProcessName.value_counts()
    names = alerts.head(top).index.tolist()
    if len(names) < top:   # pad with the heaviest processes so the grid is always full
        heavy = df.groupby("ProcessName")["Memory_MB"].max().sort_values(ascending=False).index
        names += [n for n in heavy if n not in names][:top - len(names)]
    wide = memory_wide(df[pd.to_datetime(df.Timestamp) >= rep["T"].min() - pd.Timedelta(minutes=5)], names)
    flagged = snap_to_line(worst_alert(rep, names), wide)

    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.3 * rows), sharex=True, squeeze=False)
    fig.suptitle("Per-process view - shaded band = that process's own normal (rolling mean +/- 3 std)",
                 fontsize=14, fontweight="bold")
    for ax, n in zip(axes.ravel(), names):
        s = wide[n].dropna()
        base = s.shift(1).rolling(window, min_periods=10)        # history only, like the detector
        mean, std = base.mean(), base.std().fillna(0)
        floor = np.maximum(std, np.maximum(1.0, 0.05 * mean.abs()))
        gapped = break_gaps(pd.DataFrame({"mem": s}))["mem"]
        ax.fill_between(s.index, (mean - 3 * floor).clip(lower=0), mean + 3 * floor,
                        color="#4c78a8", alpha=0.15, label="normal band")
        ax.plot(gapped.index, gapped.values, color="#2b4a6f", lw=1.3, label="memory")
        draw_markers(ax, flagged, n, label=True)
        ax.set_title(f"{n}  -  {int(alerts.get(n, 0))} alert(s)", fontsize=10)
        ax.set_ylabel("MB")
        style_time_axis(ax)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SRM_DATA_DIR", "."))
    ap.add_argument("--top", type=int, default=6, help="processes in the dashboard / detail panels")
    ap.add_argument("--last-minutes", type=float, help="only plot the most recent N minutes")
    ap.add_argument("--warmup", type=int, default=10, help="snapshots of history before alerting starts")
    ap.add_argument("--step", type=int, default=1, help="analyse every Nth snapshot (faster)")
    ap.add_argument("--show", action="store_true", help="open interactive windows")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")

    df = ad.load_data(os.path.join(a.data_dir, "process_log.csv"))
    if df is None or df["Timestamp"].nunique() < a.warmup + 2:
        raise SystemExit("process_log.csv missing or too short - check --data-dir, "
                         "or run the C++ monitor for a few more minutes.")
    if a.last_minutes:
        cutoff = pd.to_datetime(df["Timestamp"]).max() - pd.Timedelta(minutes=a.last_minutes)
        df = df[pd.to_datetime(df["Timestamp"]) >= cutoff]

    print("Replaying the detector over the log...")
    rep = replay(df, warmup=a.warmup, step=a.step)
    plot_dashboard(df, rep, a.top, os.path.join(a.data_dir, "timeline_report.png"), a.show)
    plot_detail(df, rep, a.top, os.path.join(a.data_dir, "process_detail.png"), a.show)


if __name__ == "__main__":
    main()