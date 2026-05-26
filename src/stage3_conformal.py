import argparse
import json
import sys
from collections import deque
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_utils import (
    features_path, horizon_dir, horizon_outputs_dir, horizon_plots_dir,
    DAYTIME_THRESH,
)

S1_THRESHOLD = 0.65
CONFORMAL_YEAR = 2023
HOLDOUT_YEAR = 2024

ALPHA = 0.10
GAMMAS = {"A": 0.001, "B": 0.005, "C": 0.020}
GAMMA_LABELS = {
    "A": "gamma=0.001 (slow)",
    "B": "gamma=0.005 (baseline)",
    "C": "gamma=0.020 (fast)",
}
WINDOW_SIZE = 200

CLOUD_TYPE_NAMES = {
    0: "Clear", 1: "Prob Clear", 2: "Fog", 3: "Water",
    4: "SC Water", 5: "Mixed", 6: "Opaque Ice", 7: "Cirrus",
    8: "Overlapping", 9: "Overshooting",
}

SEP = "=" * 68


def load_and_prepare(h: int) -> tuple:
    fp = features_path(ROOT_DIR, h)
    if not fp.exists():
        raise FileNotFoundError(
            f"Features file not found: {fp}\n"
            f"Run  python src/features.py --horizon {h}  first."
        )
    print(f"\n[1] Loading {fp.name} ...")
    feat = pd.read_parquet(fp)
    feat.sort_index(inplace=True)
    print(f"    Full frame : {feat.shape}  "
          f"({feat.index.min()} -> {feat.index.max()})")
    day = feat[feat["clearsky_ghi"] >= DAYTIME_THRESH].copy()
    print(f"    Daytime rows : {len(day):,}")
    return day, feat


def load_models(h: int) -> tuple:
    mdir = horizon_dir(ROOT_DIR, h)
    s1_pkl = mdir / "stage1_classifier.pkl"
    s2_pkl = mdir / "stage2_regressor.pkl"
    s1_feat_json = mdir / "stage1_feature_list.json"
    s2_feat_json = mdir / "stage2_feature_list.json"
    for p in [s1_pkl, s2_pkl, s1_feat_json, s2_feat_json]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {p}\n"
                f"Run stage1_finalize.py and stage2_regressor.py "
                f"with --horizon {h} first."
            )
    s1_model = joblib.load(s1_pkl)
    s2_model = joblib.load(s2_pkl)
    with open(s1_feat_json) as f:
        s1_cols = json.load(f)
    with open(s2_feat_json) as f:
        s2_cols = json.load(f)
    print(f"    Stage 1 : {s1_pkl.name}  ({s1_pkl.stat().st_size/1024:.0f} KB)")
    print(f"    Stage 2 : {s2_pkl.name}  ({s2_pkl.stat().st_size/1024:.0f} KB)")
    print(f"    S1 features: {len(s1_cols)}  S2 features: {len(s2_cols)}")
    return s1_model, s2_model, s1_cols, s2_cols


def predict_twostage_ghi(
    rows: pd.DataFrame, s1_model, s2_model,
    s1_cols: list, s2_cols: list,
) -> np.ndarray:
    n = len(rows)
    prob_clear = s1_model.predict_proba(rows[s1_cols])[:, 1]
    pred_clear = prob_clear >= S1_THRESHOLD
    kt_pred = np.ones(n, dtype=float)
    cloudy_idx = np.where(~pred_clear)[0]
    if len(cloudy_idx) > 0:
        kt_pred[cloudy_idx] = np.clip(
            s2_model.predict(rows.iloc[cloudy_idx][s2_cols]), 0.0, 1.0
        )
    return kt_pred * rows["clearsky_ghi"].values


def compute_ramp_mask(day: pd.DataFrame, feat_full: pd.DataFrame, year: int) -> np.ndarray:
    f = feat_full.copy()
    f["daytime"] = f["clearsky_ghi"] >= DAYTIME_THRESH
    f["prev_daytime"] = f["daytime"].shift(1).fillna(False)
    f["ghi_diff"] = f["ghi"].diff()
    ramp_full = (
        f["daytime"] & f["prev_daytime"] & (f["ghi_diff"].abs() > 100.0)
    )
    subset = day[day.index.year == year]
    return ramp_full.reindex(subset.index, fill_value=False).values.astype(bool)


def adaptive_conformal_loop(
    ghi_pred: np.ndarray, ghi_actual: np.ndarray, initial_scores: list,
    alpha: float = ALPHA, gamma: float = 0.005, window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    n = len(ghi_pred)
    window = deque(initial_scores, maxlen=window_size)
    alpha_t = alpha
    covered = np.zeros(n, dtype=np.int8)
    lower = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)
    alpha_t_series = np.zeros(n, dtype=float)
    q_t_series = np.zeros(n, dtype=float)
    for t in range(n):
        w_arr = np.asarray(window, dtype=float)
        q_t = (float(np.quantile(w_arr, np.clip(1.0 - alpha_t, 0.0, 1.0)))
               if len(w_arr) > 0 else np.inf)
        lo = max(0.0, ghi_pred[t] - q_t)
        hi = ghi_pred[t] + q_t
        cov = int(lo <= ghi_actual[t] <= hi)
        covered[t] = cov
        lower[t] = lo
        upper[t] = hi
        alpha_t_series[t] = alpha_t
        q_t_series[t] = q_t
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - (1 - cov)), 0.01, 0.99))
        window.append(abs(ghi_actual[t] - ghi_pred[t]))
    return pd.DataFrame({
        "covered": covered, "lower": lower, "upper": upper,
        "alpha_t": alpha_t_series, "q_t": q_t_series,
        "interval_width": upper - lower,
    })


def static_conformal(ghi_pred, ghi_actual, calib_scores, alpha=ALPHA):
    q_stat = float(np.quantile(calib_scores, 1.0 - alpha))
    lo = np.maximum(0.0, ghi_pred - q_stat)
    hi = ghi_pred + q_stat
    covered = ((ghi_actual >= lo) & (ghi_actual <= hi)).astype(np.int8)
    return pd.DataFrame({
        "covered": covered, "lower": lo, "upper": hi,
        "q_t": np.full(len(ghi_pred), q_stat),
        "interval_width": hi - lo,
    })


def naive_gaussian(ghi_pred, ghi_actual, calib_scores, alpha=ALPHA):
    q_gauss = 1.96 * float(np.std(calib_scores))
    lo = np.maximum(0.0, ghi_pred - q_gauss)
    hi = ghi_pred + q_gauss
    covered = ((ghi_actual >= lo) & (ghi_actual <= hi)).astype(np.int8)
    return pd.DataFrame({
        "covered": covered, "lower": lo, "upper": hi,
        "interval_width": hi - lo,
    })


def winkler_score(lower, upper, y_actual, alpha=ALPHA):
    width = upper - lower
    penalty = np.where(
        y_actual < lower, (2.0 / alpha) * (lower - y_actual),
        np.where(y_actual > upper, (2.0 / alpha) * (y_actual - upper), 0.0)
    )
    return float(np.mean(width + penalty))


def cov_stats(covered, widths, mask=None):
    if mask is not None:
        c, w = covered[mask].astype(float), widths[mask]
    else:
        c, w = covered.astype(float), widths
    n = len(c)
    if n == 0:
        return dict(coverage=float("nan"), mean_width=float("nan"), n=0)
    return dict(coverage=float(c.mean()), mean_width=float(w.mean()), n=int(n))


def flag_str(cov):
    if np.isnan(cov):
        return ""
    if cov < 0.85:
        return " *** BELOW 0.85"
    if cov > 0.97:
        return " *** ABOVE 0.97"
    return ""


def find_transition_week(feat_full: pd.DataFrame, year: int) -> tuple:
    f = feat_full[feat_full.index.year == year].copy()
    f["daytime"] = f["clearsky_ghi"] >= DAYTIME_THRESH
    f["prev_daytime"] = f["daytime"].shift(1).fillna(False)
    f["transition"] = (
        f["daytime"] & f["prev_daytime"] &
        (f["is_clear_sky"] != f["is_clear_sky"].shift(1))
    ).astype(int)
    weekly = f["transition"].resample("W-MON").sum()
    if weekly.empty:
        start = pd.Timestamp(f"{year}-07-01", tz="UTC")
        return start, start + pd.Timedelta(days=6)
    best_end = weekly.idxmax()
    best_start = best_end - pd.Timedelta(days=6)
    return best_start, best_end


def plot_timeseries(d24: pd.DataFrame, adap_df: pd.DataFrame,
                    static_df: pd.DataFrame, feat_full: pd.DataFrame,
                    plot_dir: Path, h: int) -> None:
    w_start, w_end = find_transition_week(feat_full, HOLDOUT_YEAR)
    week_mask = (d24.index >= w_start) & (d24.index <= w_end)

    wk_ts = d24.index[week_mask]
    wk_act = d24.loc[week_mask, "ghi"].values
    wk_lo = adap_df.loc[week_mask, "lower"].values
    wk_hi = adap_df.loc[week_mask, "upper"].values
    wk_cov = adap_df.loc[week_mask, "covered"].values
    wk_pred = d24.loc[week_mask, "ghi_pred"].values

    cov_series = pd.Series(adap_df["covered"].values.astype(float), index=d24.index)
    roll_cov = cov_series.rolling("7D", min_periods=24).mean()
    alpha_ts = pd.Series(adap_df["alpha_t"].values, index=d24.index)

    width_adap_daily = pd.Series(adap_df["interval_width"].values,
                                 index=d24.index).resample("D").mean()
    static_mean_width = float(static_df["interval_width"].mean())

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[2, 1.2, 1.2])

    ax1 = fig.add_subplot(gs[0])
    for i in range(len(wk_ts)):
        color = "#66BB6A" if wk_cov[i] == 1 else "#EF5350"
        ax1.fill_between([wk_ts[i], wk_ts[i] + pd.Timedelta(hours=1)],
                         [wk_lo[i], wk_lo[i]], [wk_hi[i], wk_hi[i]],
                         alpha=0.35, color=color, linewidth=0)
    ax1.plot(wk_ts, wk_act, color="black", lw=1.5, label="GHI actual", zorder=5)
    ax1.plot(wk_ts, wk_pred, color="#1565C0", lw=1.0, ls="--",
             label="Two-stage forecast", zorder=4, alpha=0.8)
    ax1.plot(wk_ts, wk_lo, color="#2E7D32", lw=0.6, alpha=0.5)
    ax1.plot(wk_ts, wk_hi, color="#2E7D32", lw=0.6, alpha=0.5)

    from matplotlib.patches import Patch
    legend_els = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="GHI actual"),
        plt.Line2D([0], [0], color="#1565C0", lw=1.0, ls="--", label="Two-stage forecast"),
        Patch(facecolor="#66BB6A", alpha=0.6, label="Interval -- covered"),
        Patch(facecolor="#EF5350", alpha=0.6, label="Interval -- missed"),
    ]
    ax1.legend(handles=legend_els, fontsize=8, loc="upper left", ncol=2)
    ax1.set_ylabel("GHI  (W/m2)", fontsize=10)
    ax1.set_title(
        f"Adaptive 90% Prediction Interval -- Highest Cloud-Transition Week  h={h}\n"
        f"{w_start.date()} -> {w_end.date()}",
        fontsize=10, fontweight="bold"
    )
    ax1.grid(alpha=0.25)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(roll_cov.index, roll_cov.values, color="black", lw=1.4,
             label="Rolling 7-day coverage")
    ax2.axhline(0.90, color="red", lw=1.2, ls="--", alpha=0.8, label="Target 0.90")
    ax2.axhline(0.85, color="orange", lw=0.8, ls=":", alpha=0.7, label="Floor 0.85")
    ax2.set_ylabel("Coverage rate", fontsize=10)
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.25)

    ax2b = ax2.twinx()
    ax2b.plot(alpha_ts.index, alpha_ts.values, color="#1565C0", lw=0.9,
              alpha=0.7, label="alpha_t")
    ax2b.axhline(ALPHA, color="#1565C0", lw=0.7, ls="--", alpha=0.5)
    ax2b.set_ylabel("alpha_t", fontsize=9, color="#1565C0")
    ax2b.tick_params(axis="y", labelcolor="#1565C0")
    ax2b.set_ylim(0.0, 0.35)
    ax2b.legend(fontsize=8, loc="upper right")
    ax2.set_title(f"Rolling 7-Day Coverage Rate and Adaptive alpha_t  ({HOLDOUT_YEAR})  h={h}",
                  fontsize=10, fontweight="bold")

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(width_adap_daily.index, width_adap_daily.values,
             color="black", lw=1.2, label="Adaptive (daily mean)")
    ax3.axhline(static_mean_width, color="steelblue", lw=1.3, ls="--",
                label=f"Static conformal  ({static_mean_width:.0f} W/m2)")
    ax3.set_ylabel("Interval width  (W/m2)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    ax3.set_ylim(bottom=0)
    ax3.set_title(f"Daily Mean Prediction Interval Width  ({HOLDOUT_YEAR})  h={h}",
                  fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Stage 3 -- Adaptive Conformal Prediction Intervals  h={h}  |  Babcock Ranch Solar",
        fontsize=12, fontweight="bold", y=1.01
    )
    out_path = plot_dir / f"conformal_coverage_timeseries_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot A saved -> {out_path}")


def plot_conditional_heatmap(d24: pd.DataFrame, adap_df: pd.DataFrame,
                              plot_dir: Path, h: int) -> None:
    months = list(range(1, 13))
    hours = list(range(6, 20))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid = np.full((12, len(hours)), np.nan)
    n_grid = np.zeros((12, len(hours)), dtype=int)
    covered = adap_df["covered"].values
    mo = d24.index.month
    hr = d24.index.hour
    for mi, m in enumerate(months):
        for hi2, hh in enumerate(hours):
            mask = (mo == m) & (hr == hh)
            if mask.sum() >= 5:
                grid[mi, hi2] = float(covered[mask].mean())
                n_grid[mi, hi2] = int(mask.sum())
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cov_cmap",
        [(0.0, "#D32F2F"), (0.25, "#FF7043"),
         (0.5, "#FFFFFF"),
         (0.75, "#42A5F5"), (1.0, "#1565C0")],
        N=256
    )
    vmin, vmax = 0.80, 1.00
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="upper", interpolation="nearest")
    for mi in range(12):
        for hi2 in range(len(hours)):
            v = grid[mi, hi2]
            n = n_grid[mi, hi2]
            if np.isfinite(v):
                tc = "white" if (v < 0.83 or v > 0.97) else "black"
                ax.text(hi2, mi, f"{v:.2f}\n(n={n})", ha="center", va="center",
                        fontsize=6.5, color=tc, fontweight="bold")
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{hh:02d}:30" for hh in hours], fontsize=8)
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_labels, fontsize=9)
    ax.set_xlabel("Hour of day (UTC, NSRDB midpoint)", fontsize=10)
    ax.set_ylabel("Month", fontsize=10)
    ax.set_title(
        f"Adaptive Conformal -- Coverage Rate by Month x Hour  h={h}  ({HOLDOUT_YEAR})\n"
        "Target: 90%  |  Red < 0.85  |  White = 0.90  |  Blue > 0.95",
        fontsize=11, fontweight="bold"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Coverage rate", fontsize=9)
    cbar.set_ticks([0.80, 0.85, 0.90, 0.95, 1.00])
    plt.tight_layout()
    out_path = plot_dir / f"conformal_conditional_coverage_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot B saved -> {out_path}")


def plot_gamma_sensitivity(d24: pd.DataFrame, sens_results: dict,
                            plot_dir: Path, h: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharey=True, sharex=True)
    idx = d24.index
    for ax, (key, res) in zip(axes, sens_results.items()):
        loop_df = res["loop_df"]
        cov_mar = res["marginal_coverage"]
        gamma = res["gamma"]
        alpha_ts = pd.Series(loop_df["alpha_t"].values, index=idx)
        ax.plot(idx, alpha_ts.values, color="#1565C0", lw=0.8, alpha=0.85)
        ax.axhline(ALPHA, color="red", lw=1.2, ls="--", alpha=0.7,
                   label=f"Target alpha = {ALPHA}")
        ax.fill_between(idx, alpha_ts.values, ALPHA,
                        where=alpha_ts.values > ALPHA,
                        color="#EF9A9A", alpha=0.3, label="Over-shooting (wide)")
        ax.fill_between(idx, alpha_ts.values, ALPHA,
                        where=alpha_ts.values < ALPHA,
                        color="#90CAF9", alpha=0.3, label="Under-shooting (narrow)")
        ax.set_ylabel("alpha_t", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 0.35)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(
            f"gamma = {gamma}  ({GAMMA_LABELS[key]})   |   "
            f"Marginal coverage = {cov_mar:.4f}   |   "
            f"Mean width = {res['mean_width']:.1f} W/m2",
            fontsize=9, fontweight="bold"
        )
    axes[-1].set_xlabel(f"Date ({HOLDOUT_YEAR})", fontsize=10)
    fig.suptitle(
        f"Stage 3 -- Adaptive alpha_t Evolution: Gamma Sensitivity  h={h}\n"
        "Babcock Ranch Solar  |  2024 Daytime Holdout",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = plot_dir / f"conformal_gamma_sensitivity_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot C saved -> {out_path}")


def plot_adaptive_vs_static(d24: pd.DataFrame, adap_df: pd.DataFrame,
                             stat_df: pd.DataFrame,
                             plot_dir: Path, h: int) -> None:
    cloud_types = sorted(d24["cloud_type"].dropna().unique().astype(int))
    cloud_types = [ct for ct in cloud_types if (d24["cloud_type"] == ct).sum() >= 10]

    adap_cov = adap_df["covered"].values
    stat_cov = stat_df["covered"].values
    ct_vals = d24["cloud_type"].values.astype(int)

    adap_rates, stat_rates, ns, labels = [], [], [], []
    for ct in cloud_types:
        mask = ct_vals == ct
        n = mask.sum()
        if n < 10:
            continue
        adap_rates.append(float(adap_cov[mask].mean()))
        stat_rates.append(float(stat_cov[mask].mean()))
        ns.append(n)
        name = CLOUD_TYPE_NAMES.get(ct, f"Type {ct}")
        labels.append(f"Type {ct}\n{name}\n(n={n:,})")

    k = len(labels)
    x = np.arange(k)
    w = 0.36

    fig, ax = plt.subplots(figsize=(max(10, k * 1.5 + 2), 6))
    b1 = ax.bar(x - w/2, adap_rates, width=w, color="#1565C0", alpha=0.85,
                label="Adaptive conformal", edgecolor="white", lw=0.5)
    b2 = ax.bar(x + w/2, stat_rates, width=w, color="#EF5350", alpha=0.75,
                label="Static conformal", edgecolor="white", lw=0.5)
    ax.axhline(0.90, color="black", lw=1.3, ls="--", label="Target 0.90")
    ax.axhline(0.85, color="orange", lw=0.9, ls=":", label="Floor 0.85")

    for bar, val in zip(b1, adap_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                color="#0D47A1", fontweight="bold")
    for bar, val in zip(b2, stat_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                color="#B71C1C", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Coverage rate", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(
        f"Adaptive vs Static Conformal -- Coverage by Cloud Type  h={h}  ({HOLDOUT_YEAR})\n"
        "Babcock Ranch Solar  |  Target = 90%",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_path = plot_dir / f"conformal_adaptive_vs_static_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot D saved -> {out_path}")


def write_report(
    warmup_scores, sens_results, selected_key,
    adap_df, stat_df, gauss_df, d24, ramp_mask,
    report_path: Path, h: int,
) -> None:
    import calendar as _cal
    lines = []

    def ln(s=""):
        lines.append(s)

    ln("=" * 68)
    ln(f"Stage 3 Report -- Adaptive Conformal Prediction Intervals  h={h}")
    ln("Babcock Ranch Solar Energy Center  |  2024 Holdout")
    ln("=" * 68)
    ln()
    ln(f"  horizon           : h={h}")
    ln(f"  conformal year    : {CONFORMAL_YEAR}")
    ln(f"  test year         : {HOLDOUT_YEAR}")
    ln(f"  alpha             : {ALPHA}")
    ln(f"  window_size       : {WINDOW_SIZE}")
    ln()

    ln("--- WARM-UP WINDOW ({} NONCONFORMITY SCORES) ---".format(CONFORMAL_YEAR))
    ln()
    ln(f"  {CONFORMAL_YEAR} daytime hours           : {len(warmup_scores):,}")
    ln(f"  Mean  |e| (W/m2)             : {np.mean(warmup_scores):.2f}")
    ln(f"  Median |e|                   : {np.median(warmup_scores):.2f}")
    ln(f"  90th percentile |e|          : {np.quantile(warmup_scores, 0.90):.2f}")
    ln(f"  Rolling window size (maxlen) : {WINDOW_SIZE}")
    ln(f"  Initialised with last {min(WINDOW_SIZE, len(warmup_scores)):,} {CONFORMAL_YEAR} entries")

    ln()
    ln("--- GAMMA SENSITIVITY ---")
    ln()
    hdr = f"  {'Gamma':<8}  {'Marginal Cov':>13}  {'Mean Width':>12}  "
    hdr += f"{'Cov Cloudy':>11}  {'Cov Ramp':>10}  {'Winkler':>9}"
    ln(hdr)
    ln("  " + "-" * (len(hdr) - 2))
    for key, res in sens_results.items():
        ln(f"  {res['gamma']:<8.3f}  {res['marginal_coverage']:>13.4f}  "
           f"{res['mean_width']:>12.1f}  "
           f"{res['cov_cloudy']:>11.4f}  "
           f"{res['cov_ramp']:>10.4f}  "
           f"{res['winkler']:>9.1f}")

    sel = sens_results[selected_key]
    ln()
    ln("--- SELECTED GAMMA ---")
    ln()
    ln(f"  Selected: gamma_{selected_key} = {sel['gamma']}")
    ln()
    for key, res in sens_results.items():
        dev = abs(res["marginal_coverage"] - 0.90)
        ln(f"  gamma_{key} = {res['gamma']:.3f}:  "
           f"coverage = {res['marginal_coverage']:.4f}  "
           f"(|dev| = {dev:.4f})  "
           f"mean width = {res['mean_width']:.1f} W/m2")

    adap_cov = adap_df["covered"].values.astype(float)
    stat_cov = stat_df["covered"].values.astype(float)
    gauss_cov = gauss_df["covered"].values.astype(float)
    is_clear = d24["is_clear_sky"].values.astype(bool)
    ct_vals = d24["cloud_type"].values.astype(int)
    mo_vals = d24.index.month
    hr_vals = d24.index.hour
    adap_w = adap_df["interval_width"].values
    ghi_act = d24["ghi"].values

    ln()
    ln("--- MARGINAL COVERAGE ---")
    ln()
    ln(f"  {'Method':<28}  {'Coverage':>9}  {'Mean Width':>11}")
    ln("  " + "-" * 52)
    ln(f"  {'Adaptive conformal':<28}  {adap_cov.mean():>9.4f}  "
       f"{adap_df['interval_width'].mean():>11.1f}")
    ln(f"  {'Static conformal':<28}  {stat_cov.mean():>9.4f}  "
       f"{stat_df['interval_width'].mean():>11.1f}")
    ln(f"  {'Naive Gaussian (1.96 sigma)':<28}  {gauss_cov.mean():>9.4f}  "
       f"{gauss_df['interval_width'].mean():>11.1f}")

    for label, cov_arr in [("Adaptive", adap_cov), ("Static", stat_cov),
                            ("Naive Gaussian", gauss_cov)]:
        dev = cov_arr.mean() - 0.90
        if abs(dev) > 0.03:
            ln(f"\n  *** WARNING: {label} marginal coverage = {cov_arr.mean():.4f}; "
               f"deviation from 0.90 = {dev:+.4f} > 0.03 ***")

    ln()
    ln("--- CONDITIONAL COVERAGE ---")
    ln()
    ln("  a) Cloud regime:")
    ln(f"  {'Regime':<16}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>6}")
    ln("  " + "-" * 47)
    for label, mask in [("Clear-sky", is_clear), ("Cloudy", ~is_clear)]:
        s = cov_stats(adap_df["covered"].values, adap_w, mask)
        ln(f"  {label:<16}  {s['coverage']:>9.4f}  {s['mean_width']:>11.1f}  "
           f"{s['n']:>6,}{flag_str(s['coverage'])}")

    ln()
    ln("  b) Cloud type:")
    ln(f"  {'Type':<22}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>6}")
    ln("  " + "-" * 53)
    for ct in sorted(set(ct_vals)):
        mask = ct_vals == ct
        s = cov_stats(adap_df["covered"].values, adap_w, mask)
        if s["n"] < 10:
            continue
        name = CLOUD_TYPE_NAMES.get(int(ct), f"Type {ct}")
        ln(f"  {f'Type {ct} ({name})':<22}  {s['coverage']:>9.4f}  "
           f"{s['mean_width']:>11.1f}  {s['n']:>6,}{flag_str(s['coverage'])}")

    ln()
    ln("  c) Month:")
    ln(f"  {'Month':<8}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>6}")
    ln("  " + "-" * 40)
    for m in range(1, 13):
        mask = mo_vals == m
        s = cov_stats(adap_df["covered"].values, adap_w, mask)
        if s["n"] == 0:
            continue
        ln(f"  {_cal.month_abbr[m]:<8}  {s['coverage']:>9.4f}  "
           f"{s['mean_width']:>11.1f}  {s['n']:>6,}{flag_str(s['coverage'])}")

    ln()
    ln("  d) Hour of day:")
    ln(f"  {'Hour':>5}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>6}")
    ln("  " + "-" * 38)
    for hh in range(6, 20):
        mask = hr_vals == hh
        s = cov_stats(adap_df["covered"].values, adap_w, mask)
        if s["n"] == 0:
            continue
        ln(f"  {hh:>5}  {s['coverage']:>9.4f}  {s['mean_width']:>11.1f}  "
           f"{s['n']:>6,}{flag_str(s['coverage'])}")

    ln()
    ln("  e) Ramp events:")
    for label, mask in [("Ramp events", ramp_mask), ("Non-ramp", ~ramp_mask)]:
        s = cov_stats(adap_df["covered"].values, adap_w, mask)
        ln(f"  {label:<14}  coverage={s['coverage']:.4f}  "
           f"mean_width={s['mean_width']:.1f} W/m2  n={s['n']:,}{flag_str(s['coverage'])}")

    ws_adap = winkler_score(adap_df["lower"].values, adap_df["upper"].values, ghi_act)
    ws_stat = winkler_score(stat_df["lower"].values, stat_df["upper"].values, ghi_act)
    ws_gauss = winkler_score(gauss_df["lower"].values, gauss_df["upper"].values, ghi_act)

    ln()
    ln("--- WINKLER SCORES ---")
    ln()
    ln(f"  {'Method':<28}  {'Winkler Score':>14}")
    ln("  " + "-" * 44)
    ln(f"  {'Adaptive conformal':<28}  {ws_adap:>14.2f}")
    ln(f"  {'Static conformal':<28}  {ws_stat:>14.2f}")
    ln(f"  {'Naive Gaussian (1.96 sigma)':<28}  {ws_gauss:>14.2f}")

    ln()
    ln("--- ADAPTIVE VS STATIC COMPARISON ---")
    ln()

    def _seg_compare(label, mask):
        a = cov_stats(adap_df["covered"].values, adap_df["interval_width"].values, mask)
        s = cov_stats(stat_df["covered"].values, stat_df["interval_width"].values, mask)
        ln(f"  {label:<18}  adap cov={a['coverage']:.4f}  w={a['mean_width']:.0f}  "
           f"  stat cov={s['coverage']:.4f}  w={s['mean_width']:.0f}")

    ln(f"  {'Segment':<18}  {'Adaptive':>30}  {'Static':>28}")
    ln("  " + "-" * 79)
    _seg_compare("All daytime", None)
    _seg_compare("Clear-sky", is_clear)
    _seg_compare("Cloudy", ~is_clear)
    _seg_compare("Ramp events", ramp_mask)
    _seg_compare("Non-ramp", ~ramp_mask)

    ln()
    ln("  Coverage variance across months:")
    adap_monthly, stat_monthly = [], []
    for m in range(1, 13):
        mask = mo_vals == m
        if mask.sum() < 5:
            continue
        adap_monthly.append(float(adap_df["covered"].values[mask].mean()))
        stat_monthly.append(float(stat_df["covered"].values[mask].mean()))
    ln(f"    Adaptive std(monthly coverage) = {np.std(adap_monthly):.4f}")
    ln(f"    Static   std(monthly coverage) = {np.std(stat_monthly):.4f}")

    ln()
    ln("=" * 68)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n    Report saved -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True,
                        help="Forecast horizon in hours (e.g. 1, 3, 6, 24).")
    args = parser.parse_args()
    h = args.horizon

    odir = horizon_outputs_dir(ROOT_DIR, h)
    pdir = horizon_plots_dir(ROOT_DIR, h)
    conf_parquet = odir / f"conformal_predictions_{HOLDOUT_YEAR}_h{h:03d}.parquet"
    report_path = odir / "stage3_conformal_report.txt"

    print(SEP)
    print(f"Stage 3 -- Adaptive Conformal Prediction Intervals  h={h}")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)
    print(f"\n  Horizon          : h={h}")
    print(f"  Conformal year   : {CONFORMAL_YEAR}")
    print(f"  Test year        : {HOLDOUT_YEAR}")
    print(f"  Alpha            : {ALPHA}")
    print(f"  Window size      : {WINDOW_SIZE}")

    day, feat_full = load_and_prepare(h)
    print(f"\n[2] Loading models ...")
    s1_model, s2_model, s1_cols, s2_cols = load_models(h)

    print(f"\n{SEP}")
    print(f"STEP 1 -- {CONFORMAL_YEAR} WARM-UP WINDOW")
    print(SEP)
    d_conf = day[day.index.year == CONFORMAL_YEAR].copy()
    print(f"\n  Generating two-stage forecasts for {CONFORMAL_YEAR} ...")
    ghi_pred_conf = predict_twostage_ghi(d_conf, s1_model, s2_model, s1_cols, s2_cols)
    ghi_actual_conf = d_conf["ghi"].values
    nc_scores_conf = np.abs(ghi_actual_conf - ghi_pred_conf)

    print(f"\n  {CONFORMAL_YEAR} daytime hours   : {len(nc_scores_conf):,}")
    print(f"  Mean  |e| (W/m2) : {np.mean(nc_scores_conf):.2f}")
    print(f"  Median |e|       : {np.median(nc_scores_conf):.2f}")
    print(f"  90th percentile  : {np.quantile(nc_scores_conf, 0.90):.2f}")
    print(f"  Max              : {np.max(nc_scores_conf):.2f}")

    print(f"\n{SEP}")
    print(f"STEP 2 -- {HOLDOUT_YEAR} TWO-STAGE GHI PREDICTIONS")
    print(SEP)
    d24 = day[day.index.year == HOLDOUT_YEAR].copy()
    print(f"\n  Generating two-stage forecasts for {HOLDOUT_YEAR} ...")
    ghi_pred_2024 = predict_twostage_ghi(d24, s1_model, s2_model, s1_cols, s2_cols)
    ghi_actual_2024 = d24["ghi"].values
    d24["ghi_pred"] = ghi_pred_2024
    ramp_mask_2024 = compute_ramp_mask(d24, feat_full, HOLDOUT_YEAR)
    print(f"  {HOLDOUT_YEAR} daytime hours   : {len(d24):,}")
    print(f"  Ramp events          : {ramp_mask_2024.sum():,} "
          f"({ramp_mask_2024.mean()*100:.1f}%)")

    print(f"\n{SEP}")
    print("STEP 3 -- GAMMA SENSITIVITY ANALYSIS")
    print(SEP)
    is_clear_2024 = d24["is_clear_sky"].values.astype(bool)
    sens_results = {}

    for key, gamma in GAMMAS.items():
        print(f"\n  Running adaptive loop: gamma_{key} = {gamma} ...")
        loop_df = adaptive_conformal_loop(
            ghi_pred_2024, ghi_actual_2024, list(nc_scores_conf),
            alpha=ALPHA, gamma=gamma, window_size=WINDOW_SIZE
        )
        cov_all = float(loop_df["covered"].mean())
        mean_wid = float(loop_df["interval_width"].mean())
        cov_cloud = float(loop_df["covered"].values[~is_clear_2024].mean())
        cov_ramp = (float(loop_df["covered"].values[ramp_mask_2024].mean())
                    if ramp_mask_2024.sum() > 0 else float("nan"))
        ws = winkler_score(loop_df["lower"].values, loop_df["upper"].values, ghi_actual_2024)
        sens_results[key] = {
            "gamma": gamma, "loop_df": loop_df,
            "marginal_coverage": cov_all, "mean_width": mean_wid,
            "cov_cloudy": cov_cloud, "cov_ramp": cov_ramp, "winkler": ws,
        }
        print(f"    Marginal coverage : {cov_all:.4f}  (target 0.90)")
        print(f"    Mean width        : {mean_wid:.1f} W/m2")
        print(f"    Coverage (cloudy) : {cov_cloud:.4f}")
        print(f"    Coverage (ramp)   : {cov_ramp:.4f}")
        print(f"    Winkler score     : {ws:.2f}")
        if abs(cov_all - 0.90) > 0.03:
            print(f"    *** FLAG: coverage {cov_all:.4f} deviates from 0.90 by "
                  f"{cov_all-0.90:+.4f} > 0.03 ***")

    print(f"\n{SEP}")
    print("GAMMA SELECTION")
    print(SEP)
    devs = {k: abs(v["marginal_coverage"] - 0.90) for k, v in sens_results.items()}
    min_dev = min(devs.values())
    candidates = {k: v for k, v in sens_results.items()
                  if abs(devs[k] - min_dev) < 1e-4}
    selected_key = min(candidates, key=lambda k: candidates[k]["mean_width"])
    sel = sens_results[selected_key]
    print(f"\n  Selected: gamma_{selected_key} = {sel['gamma']}")
    print(f"    Marginal coverage : {sel['marginal_coverage']:.4f}")
    print(f"    Mean width        : {sel['mean_width']:.1f} W/m2")

    print(f"\n{SEP}")
    print(f"STEP 4 -- FULL ADAPTIVE ANALYSIS  (gamma = {sel['gamma']})")
    print(SEP)
    adap_df = sens_results[selected_key]["loop_df"].copy()
    adap_df.index = d24.index

    print(f"\n{SEP}")
    print("STEP 5 -- STATIC CONFORMAL")
    print(SEP)
    stat_df = static_conformal(ghi_pred_2024, ghi_actual_2024, nc_scores_conf)
    stat_df.index = d24.index
    q_static = float(stat_df["q_t"].iloc[0])
    print(f"\n  Fixed quantile q_static = {q_static:.2f} W/m2")
    print(f"  Static marginal coverage = {stat_df['covered'].mean():.4f}")
    print(f"  Static mean width        = {stat_df['interval_width'].mean():.1f} W/m2")

    print(f"\n{SEP}")
    print("STEP 6 -- NAIVE GAUSSIAN BASELINE")
    print(SEP)
    gauss_df = naive_gaussian(ghi_pred_2024, ghi_actual_2024, nc_scores_conf)
    gauss_df.index = d24.index
    sigma = float(np.std(nc_scores_conf))
    print(f"\n  sigma = {sigma:.2f} W/m2  q_gaussian = {1.96*sigma:.2f} W/m2")
    print(f"  Gaussian marginal coverage = {gauss_df['covered'].mean():.4f}")
    print(f"  Gaussian mean width        = {gauss_df['interval_width'].mean():.1f} W/m2")

    ws_adap = winkler_score(adap_df["lower"].values, adap_df["upper"].values, ghi_actual_2024)
    ws_stat = winkler_score(stat_df["lower"].values, stat_df["upper"].values, ghi_actual_2024)
    ws_gauss = winkler_score(gauss_df["lower"].values, gauss_df["upper"].values, ghi_actual_2024)
    print(f"\n  {'Method':<28}  {'Winkler Score':>14}")
    print(f"  {'-'*44}")
    print(f"  {'Adaptive conformal':<28}  {ws_adap:>14.2f}")
    print(f"  {'Static conformal':<28}  {ws_stat:>14.2f}")
    print(f"  {'Naive Gaussian (1.96 sigma)':<28}  {ws_gauss:>14.2f}")

    print(f"\n{SEP}")
    print("STEP 7 -- GENERATING PLOTS")
    print(SEP)
    print()
    plot_timeseries(d24, adap_df, stat_df, feat_full, pdir, h)
    plot_conditional_heatmap(d24, adap_df, pdir, h)
    plot_gamma_sensitivity(d24, sens_results, pdir, h)
    plot_adaptive_vs_static(d24, adap_df, stat_df, pdir, h)

    print(f"\n{SEP}")
    print("STEP 8 -- SAVING RESULTS")
    print(SEP)
    out = pd.DataFrame({
        "GHI_actual": ghi_actual_2024,
        "GHI_pred": ghi_pred_2024,
        "lower": adap_df["lower"].values,
        "upper": adap_df["upper"].values,
        "covered": adap_df["covered"].values,
        "alpha_t": adap_df["alpha_t"].values,
        "q_t": adap_df["q_t"].values,
        "interval_width": adap_df["interval_width"].values,
        "is_clear_sky": d24["is_clear_sky"].values.astype(int),
        "cloud_type": d24["cloud_type"].values.astype(int),
        "month": d24.index.month,
        "hour_of_day": d24.index.hour,
        "is_ramp_event": ramp_mask_2024.astype(int),
    }, index=d24.index)
    out.to_parquet(conf_parquet)
    print(f"\n    Parquet saved -> {conf_parquet}  "
          f"({conf_parquet.stat().st_size/1024:.1f} KB)")

    write_report(
        warmup_scores=nc_scores_conf,
        sens_results=sens_results,
        selected_key=selected_key,
        adap_df=adap_df,
        stat_df=stat_df,
        gauss_df=gauss_df,
        d24=d24,
        ramp_mask=ramp_mask_2024,
        report_path=report_path,
        h=h,
    )

    print(f"\n{SEP}")
    print(f"STAGE 3 COMPLETE -- FILE SUMMARY  h={h}")
    print(SEP)
    for p in [conf_parquet, report_path]:
        size = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {p}  ({size:.1f} KB)")


if __name__ == "__main__":
    main()
