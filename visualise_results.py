"""
Visualise what the detector sees.

Builds ONE report image (viz_report.png) with 5 panels:
  1. KNN feature space (PCA 2-D): grey = history, coloured = latest snapshot.
     Lines connect every flagged process to its k nearest historical neighbours,
     so you can see "this point is far from where normal points live".
  2. KNN score per process vs the anomaly threshold.
  3. Z-score per process vs the +/-3 line (how far from its OWN normal).
  4. Top memory-hungry processes (bars) vs their normal average (black ticks),
     coloured by alert level.
  5. Memory over time for the heaviest processes.

Usage (from the folder that holds process_log.csv, or pass --data-dir):
    python visualize_results.py
    python visualize_results.py --data-dir x64\\Debug --top 12
    python visualize_results.py --at "2026-06-25 20:30:00"   # inspect a past moment
    python visualize_results.py --show                        # open an interactive window
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

import anomaly_detector as ad

COLORS = {"OK": "#2e9e5b", "WARNING": "#f0a020", "CRITICAL": "#d93a3a"}


def build_report(df, top=10, k=ad.KNN_K):
    history, latest, ts = ad.split_history_latest(df)
    baselines = ad.build_baselines(history)
    res = ad.zscore_flags(latest, baselines)
    model = ad.fit_knn(history, k=k)
    scores, flags = ad.knn_scores(model, res)
    res["KNN_Score"], res["KNN_Flag"] = scores, flags
    res["Alert"] = np.where(res.Z_Flag & res.KNN_Flag, "CRITICAL",
                    np.where(res.Z_Flag | res.KNN_Flag, "WARNING", "OK"))
    return history, res, model, ts


def plot_all(df, top, out_path, show):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    history, res, model, ts = build_report(df, top)
    res = res.sort_values("Memory_MB", ascending=False).reset_index(drop=True)
    res["Label"] = res.ProcessName + " (" + res.PID.astype(int).astype(str) + ")"
    topres = res.head(top)

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.38, wspace=0.22)
    fig.suptitle(f"Process anomaly report  |  snapshot {ts}  |  "
                 f"{(res.Alert == 'CRITICAL').sum()} critical, {(res.Alert == 'WARNING').sum()} warning, "
                 f"{(res.Alert == 'OK').sum()} ok", fontsize=15, fontweight="bold")

    # ---- 1. KNN feature space (PCA) -----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    if model is None:
        ax.text(0.5, 0.5, f"Not enough history for KNN yet\n(need {ad.MIN_KNN_HISTORY}+ rows, have {len(history)})",
                ha="center", va="center"); ax.axis("off")
    else:
        Xh = model["scaler"].transform(history[ad.FEATURES].to_numpy(float))
        Xl = model["scaler"].transform(res[ad.FEATURES].to_numpy(float))
        pca = PCA(n_components=2).fit(Xh)
        Ph, Pl = pca.transform(Xh), pca.transform(Xl)
        ax.scatter(Ph[:, 0], Ph[:, 1], s=10, c="#b8c4d6", alpha=0.5, label="history (normal behaviour)")
        # neighbour lines for flagged points
        _, idx = model["nn"].kneighbors(Xl, n_neighbors=model["k"])
        for i, r in res.iterrows():
            if r.Alert != "OK":
                for j in idx[i]:
                    ax.plot([Pl[i, 0], Ph[j, 0]], [Pl[i, 1], Ph[j, 1]], color=COLORS[r.Alert], alpha=0.25, lw=0.8)
        for level in ("OK", "WARNING", "CRITICAL"):
            m = (res.Alert == level).to_numpy()
            if m.any():
                ax.scatter(Pl[m, 0], Pl[m, 1], s=70 if level == "OK" else 130, c=COLORS[level],
                           edgecolor="black", linewidth=0.6, label=f"latest: {level}", zorder=3)
        for i, r in res.iterrows():
            if r.Alert != "OK" or i < 5:
                ax.annotate(r.Label, (Pl[i, 0], Pl[i, 1]), fontsize=8, xytext=(5, 5), textcoords="offset points")
        ev = np.nan_to_num(pca.explained_variance_ratio_) * 100
        ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.0f}% of variance)")
        ax.legend(fontsize=8, loc="best")
    ax.set_title("1) KNN feature space - flagged points sit far from their k nearest neighbours")
    ax.grid(alpha=0.25)

    # ---- 2. KNN score vs threshold ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    t = topres.iloc[::-1]
    ax.barh(t.Label, t.KNN_Score, color=[COLORS[a] for a in t.Alert])
    if model is not None:
        ax.axvline(model["threshold"], color="black", ls="--", label=f"threshold = {model['threshold']:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("mean distance to k nearest neighbours")
    ax.set_title("2) KNN anomaly score (bigger = more unusual)")

    # ---- 3. Z-scores --------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.barh(t.Label, t.Z_Score, color=[COLORS[a] for a in t.Alert])
    ax.axvline(ad.Z_THRESHOLD, color="black", ls="--", label=f"z = {ad.Z_THRESHOLD:g}")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xscale("symlog", linthresh=3)
    ax.set_xlabel("z-score  =  (current - process's own average) / its std   (log scale beyond 3)")
    ax.set_title("3) Z-score vs the process's OWN normal")
    ax.legend(fontsize=8)

    # ---- 4. Top memory consumers ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.barh(t.Label, t.Memory_MB, color=[COLORS[a] for a in t.Alert])
    ax.scatter(t.Avg_MB, t.Label, marker="|", s=350, c="black", zorder=3, label="normal average")
    ax.set_xlabel("memory (MB)")
    ax.set_title(f"4) Top {len(t)} memory consumers right now")
    from matplotlib.patches import Patch
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color=COLORS[l], label=l) for l in ("CRITICAL", "WARNING", "OK")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    # ---- 5. Memory over time ---------------------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    full = pd.concat([history, res[history.columns.intersection(res.columns)]], ignore_index=True)
    heavy = list(dict.fromkeys(res.ProcessName.tolist()))[:6]
    full["T"] = pd.to_datetime(full["Timestamp"], errors="coerce")
    for name in heavy:
        s = full[full.ProcessName == name].groupby("T")["Memory_MB"].max()
        ax.plot(s.index, s.values, label=name, lw=1.4)
    ax.axvline(pd.to_datetime(ts), color="black", ls=":", lw=1)
    ax.set_ylabel("memory (MB)")
    ax.set_title("5) Memory over time for the heaviest processes (dotted line = analysed snapshot)")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.25)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")
    if show:
        plt.show()

    print("\nTop processes at this snapshot:")
    cols = ["Label", "Memory_MB", "Avg_MB", "Z_Score", "KNN_Score", "Alert"]
    print(topres[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SRM_DATA_DIR", "."))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--at", help="analyse the state at this timestamp instead of the latest one")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")

    df = ad.load_data(os.path.join(a.data_dir, "process_log.csv"))
    if df is None or df["Timestamp"].nunique() < 3:
        raise SystemExit("process_log.csv missing or too short - run the C++ monitor for a few minutes first.")
    if a.at:
        df = df[df["Timestamp"] <= a.at]
    plot_all(df, a.top, os.path.join(a.data_dir, "viz_report.png"), a.show)


if __name__ == "__main__":
    main()