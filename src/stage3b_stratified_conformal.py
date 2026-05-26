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
    DAYTIME_THRESH, predict_ghi_for_mode, KT_CLEAR_STRATEGY,
)

S1_THRESHOLD_DEFAULT = 0.65
CONFORMAL_YEAR = 2023
HOLDOUT_YEAR = 2024
ALPHA = 0.10
GAMMA = 0.005
WINDOW_SIZE = 200

CLOUD_TYPE_NAMES = {
    0: "Clear", 1: "Prob Clear", 2: "Fog", 3: "Water",
    4: "SC Water", 5: "Mixed", 6: "Opaque Ice", 7: "Cirrus",
    8: "Overlapping", 9: "Overshooting",
}

SEP = "=" * 68


def load_mode_selection(h: int) -> tuple:
    p = horizon_dir(ROOT_DIR, h) / "prediction_mode_selection.json"
    if p.exists():
        with open(p) as f:
            sel = json.load(f)
        return sel["selected_prediction_mode"], float(sel.get("selected_threshold", S1_THRESHOLD_DEFAULT))
    return "hard_fixed", S1_THRESHOLD_DEFAULT


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


def predict_for_conformal(
    rows: pd.DataFrame, s1_model, s2_model,
    s1_cols: list, s2_cols: list,
    selected_mode: str, selected_threshold: float,
) -> tuple:
    ghi_pred, _, pred_clear = predict_ghi_for_mode(
        rows, s1_model, s2_model, s1_cols, s2_cols,
        selected_mode, selected_threshold,
    )
    return ghi_pred, pred_clear


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
    alpha: float = ALPHA, gamma: float = GAMMA, window_size: int = WINDOW_SIZE,
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


def adaptive_stratified_loop(
    ghi_pred: np.ndarray, ghi_actual: np.ndarray, regime_pred: np.ndarray,
    init_scores_C: list, init_scores_K: list,
    alpha: float = ALPHA, gamma: float = GAMMA, window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    n = len(ghi_pred)
    window_C = deque(init_scores_C, maxlen=window_size)
    window_K = deque(init_scores_K, maxlen=window_size)
    alpha_C = alpha
    alpha_K = alpha
    covered = np.zeros(n, dtype=np.int8)
    lower = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)
    alpha_C_series = np.zeros(n, dtype=float)
    alpha_K_series = np.zeros(n, dtype=float)
    q_t_series = np.zeros(n, dtype=float)
    for t in range(n):
        clear_pred = (regime_pred[t] == 1)
        if clear_pred:
            w_arr = np.asarray(window_C, dtype=float)
            a_t = alpha_C
        else:
            w_arr = np.asarray(window_K, dtype=float)
            a_t = alpha_K
        q_t = (float(np.quantile(w_arr, np.clip(1.0 - a_t, 0.0, 1.0)))
               if len(w_arr) > 0 else np.inf)
        lo = max(0.0, ghi_pred[t] - q_t)
        hi = ghi_pred[t] + q_t
        cov = int(lo <= ghi_actual[t] <= hi)
        nc = abs(ghi_actual[t] - ghi_pred[t])
        covered[t] = cov
        lower[t] = lo
        upper[t] = hi
        alpha_C_series[t] = alpha_C
        alpha_K_series[t] = alpha_K
        q_t_series[t] = q_t
        if clear_pred:
            alpha_C = float(np.clip(alpha_C + gamma * (alpha - (1 - cov)), 0.01, 0.99))
            window_C.append(nc)
        else:
            alpha_K = float(np.clip(alpha_K + gamma * (alpha - (1 - cov)), 0.01, 0.99))
            window_K.append(nc)
    return pd.DataFrame({
        "covered": covered, "lower": lower, "upper": upper,
        "alpha_C_t": alpha_C_series, "alpha_K_t": alpha_K_series,
        "q_t": q_t_series, "interval_width": upper - lower,
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


def plot_stratified_alpha(d24: pd.DataFrame, strat_df: pd.DataFrame,
                          plot_dir: Path, h: int) -> None:
    idx = d24.index
    alpha_C = pd.Series(strat_df["alpha_C_t"].values, index=idx)
    alpha_K = pd.Series(strat_df["alpha_K_t"].values, index=idx)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, alpha_s, color, label, stream_name in [
        (axes[0], alpha_C, "#1565C0", "alpha_C_t (clear stream)", "Stream C (Clear-Sky)"),
        (axes[1], alpha_K, "#E65100", "alpha_K_t (cloudy stream)", "Stream K (Cloudy)"),
    ]:
        ax.plot(idx, alpha_s.values, color=color, lw=0.8, alpha=0.9, label=label)
        ax.axhline(ALPHA, color="red", lw=1.2, ls="--", alpha=0.7,
                   label=f"Target alpha = {ALPHA}")
        ax.fill_between(idx, alpha_s.values, ALPHA,
                        where=(alpha_s.values < ALPHA),
                        color="#90CAF9", alpha=0.3)
        ax.fill_between(idx, alpha_s.values, ALPHA,
                        where=(alpha_s.values > ALPHA),
                        color="#EF9A9A", alpha=0.3)
        ax.set_ylabel("alpha_t", fontsize=10)
        ax.set_ylim(0.0, 0.40)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        ax.set_title(f"{stream_name} -- Adaptive Alpha Evolution {HOLDOUT_YEAR}  h={h}",
                     fontsize=10, fontweight="bold")
    axes[1].set_xlabel(f"Date ({HOLDOUT_YEAR})", fontsize=10)
    fig.suptitle(
        f"Stage 3b -- Regime-Stratified Conformal: Independent Alpha Streams  h={h}\n"
        "Babcock Ranch Solar  |  2024 Daytime Holdout",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = plot_dir / f"conformal_stratified_alpha_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved -> {out_path}")


def plot_stratified_widths(d24: pd.DataFrame, strat_df: pd.DataFrame,
                           plot_dir: Path, h: int) -> None:
    stream_C = strat_df["stream"].values == 1
    stream_K = strat_df["stream"].values == 0
    widths_C = strat_df["interval_width"].values[stream_C]
    widths_K = strat_df["interval_width"].values[stream_K]
    errors_C = np.abs(d24["ghi"].values[stream_C] - d24["ghi_pred"].values[stream_C])
    errors_K = np.abs(d24["ghi"].values[stream_K] - d24["ghi_pred"].values[stream_K])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    max_w = max(widths_C.max() if len(widths_C) > 0 else 1,
                widths_K.max() if len(widths_K) > 0 else 1)
    bins = np.linspace(0, max_w * 1.05, 60)
    if len(widths_C) > 0:
        ax1.hist(widths_C, bins=bins, color="#1565C0", alpha=0.6, density=True,
                 label=f"Stream C (clear)  n={stream_C.sum():,}")
        ax1.axvline(widths_C.mean(), color="#1565C0", lw=1.8, ls="--",
                    label=f"Mean C = {widths_C.mean():.1f}")
    if len(widths_K) > 0:
        ax1.hist(widths_K, bins=bins, color="#E65100", alpha=0.6, density=True,
                 label=f"Stream K (cloudy)  n={stream_K.sum():,}")
        ax1.axvline(widths_K.mean(), color="#E65100", lw=1.8, ls="--",
                    label=f"Mean K = {widths_K.mean():.1f}")
    ax1.set_xlabel("Interval width (W/m2)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Interval Width Distribution by Stream", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)
    ax2 = axes[1]
    if len(errors_C) > 0:
        ax2.scatter(errors_C, widths_C, color="#1565C0", alpha=0.12, s=3, rasterized=True,
                    label=f"Stream C (clear)")
    if len(errors_K) > 0:
        ax2.scatter(errors_K, widths_K, color="#E65100", alpha=0.12, s=3, rasterized=True,
                    label=f"Stream K (cloudy)")
    max_err = max(
        errors_C.max() if len(errors_C) > 0 else 0,
        errors_K.max() if len(errors_K) > 0 else 0,
    )
    x_ref = np.linspace(0, max_err, 200)
    ax2.plot(x_ref, 2.0 * x_ref, color="black", lw=1.2, ls="--", alpha=0.5,
             label="width = 2x|error|")
    ax2.set_xlabel("|GHI_actual - GHI_pred|  (W/m2)", fontsize=11)
    ax2.set_ylabel("Interval width  (W/m2)", fontsize=11)
    ax2.set_title("Interval Width vs Forecast Error by Stream",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    fig.suptitle(
        f"Stage 3b -- Regime-Stratified Conformal: Interval Width Analysis  h={h}\n"
        "Babcock Ranch Solar  |  2024 Daytime Holdout",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = plot_dir / f"conformal_stratified_widths_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved -> {out_path}")


def _heatmap_grid(covered_arr, index, months, hours):
    mo = index.month
    hr = index.hour
    grid = np.full((len(months), len(hours)), np.nan)
    n_grid = np.zeros((len(months), len(hours)), dtype=int)
    for mi, m in enumerate(months):
        for hi, h in enumerate(hours):
            mask = (mo == m) & (hr == h)
            if mask.sum() >= 5:
                grid[mi, hi] = float(covered_arr[mask].mean())
                n_grid[mi, hi] = int(mask.sum())
    return grid, n_grid


def plot_side_by_side_heatmap(d24: pd.DataFrame, adap_df_s3: pd.DataFrame,
                               strat_df: pd.DataFrame,
                               plot_dir: Path, h: int) -> None:
    months = list(range(1, 13))
    hours = list(range(6, 20))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid_s3, n_s3 = _heatmap_grid(adap_df_s3["covered"].values,
                                   d24.index, months, hours)
    grid_strat, n_strat = _heatmap_grid(strat_df["covered"].values,
                                        d24.index, months, hours)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cov_cmap",
        [(0.0, "#D32F2F"), (0.25, "#FF7043"),
         (0.5, "#FFFFFF"),
         (0.75, "#42A5F5"), (1.0, "#1565C0")],
        N=256
    )
    vmin, vmax = 0.80, 1.00
    fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharey=True)
    panel_data = [
        (grid_s3, n_s3,
         "Single-Window Adaptive (Stage 3)\ngamma=0.005, shared rolling window"),
        (grid_strat, n_strat,
         "Regime-Stratified Adaptive (Stage 3b)\ngamma=0.005, Stream C + Stream K"),
    ]
    im = None
    for ax, (grid, n_grid, title) in zip(axes, panel_data):
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", origin="upper", interpolation="nearest")
        for mi in range(12):
            for hi in range(len(hours)):
                v = grid[mi, hi]
                n = n_grid[mi, hi]
                if np.isfinite(v):
                    tc = "white" if (v < 0.83 or v > 0.97) else "black"
                    ax.text(hi, mi, f"{v:.2f}\n(n={n})", ha="center", va="center",
                            fontsize=5.5, color=tc, fontweight="bold")
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels([f"{h2:02d}:30" for h2 in hours], fontsize=8)
        ax.set_yticks(range(12))
        ax.set_yticklabels(month_labels, fontsize=9)
        ax.set_xlabel("Hour of day (UTC)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Month", fontsize=11)
    if im is not None:
        cb = plt.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.02)
        cb.set_label("Coverage rate", fontsize=9)
        cb.set_ticks([0.80, 0.85, 0.90, 0.95, 1.00])
    fig.suptitle(
        f"Conformal Coverage Rate: Month x Hour  h={h}  |  Babcock Ranch Solar {HOLDOUT_YEAR}\n"
        "Target: 90%  |  Red < 0.85  |  White = 0.90  |  Blue > 0.95",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = plot_dir / f"conformal_conditional_coverage_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved -> {out_path}")


def write_combined_report(
    nc_scores_2023, nc_C_2023, nc_K_2023, n_C_2023, n_K_2023,
    adap_df_s3, stat_df, gauss_df,
    strat_df, regime_pred_2024, regime_true_2024,
    d24, ramp_mask_2024, report_path: Path, h: int,
    selected_mode: str = "hard_fixed",
    selected_threshold: float = S1_THRESHOLD_DEFAULT,
):
    import calendar as _cal
    lines = []

    def ln(s=""):
        lines.append(s)

    is_clear_true = regime_true_2024.astype(bool)
    ct_vals = d24["cloud_type"].values.astype(int)
    mo_vals = d24.index.month
    hr_vals = d24.index.hour
    ghi_act = d24["ghi"].values

    adap_s3_cov = adap_df_s3["covered"].values
    adap_s3_w = adap_df_s3["interval_width"].values
    stat_cov = stat_df["covered"].values
    gauss_cov = gauss_df["covered"].values
    strat_cov = strat_df["covered"].values
    strat_w = strat_df["interval_width"].values
    mismatch = (regime_pred_2024 != regime_true_2024)
    n_mismatch = int(mismatch.sum())
    mismatch_rate = n_mismatch / max(len(mismatch), 1)

    ws_s3 = winkler_score(adap_df_s3["lower"].values, adap_df_s3["upper"].values, ghi_act)
    ws_stat = winkler_score(stat_df["lower"].values, stat_df["upper"].values, ghi_act)
    ws_gauss = winkler_score(gauss_df["lower"].values, gauss_df["upper"].values, ghi_act)
    ws_strat = winkler_score(strat_df["lower"].values, strat_df["upper"].values, ghi_act)

    thr_display = (f"{selected_threshold:.2f} (routing only)"
                   if selected_mode == "soft" else f"{selected_threshold:.2f}")

    ln("=" * 68)
    ln(f"Stage 3 / Stage 3b Report -- Conformal Prediction Intervals  h={h}")
    ln("Babcock Ranch Solar Energy Center  |  2024 Holdout")
    ln("=" * 68)
    ln()
    ln(f"  horizon           : h={h}")
    ln(f"  conformal year    : {CONFORMAL_YEAR}")
    ln(f"  test year         : {HOLDOUT_YEAR}")
    ln(f"  alpha             : {ALPHA}")
    ln(f"  gamma             : {GAMMA}")
    ln(f"  window_size       : {WINDOW_SIZE}")
    ln(f"  prediction_mode   : {selected_mode}")
    ln(f"  threshold_used    : {thr_display}")
    ln(f"  clear_kt_strategy : {KT_CLEAR_STRATEGY}")
    ln()

    ln("--- WARM-UP WINDOW STATISTICS (2023 NONCONFORMITY SCORES) ---")
    ln()
    ln("  Combined (all 2023 daytime hours):")
    ln(f"    n = {len(nc_scores_2023):,}  "
       f"mean = {np.mean(nc_scores_2023):.2f} W/m2  "
       f"median = {np.median(nc_scores_2023):.2f}  "
       f"90th pctile = {np.quantile(nc_scores_2023, 0.90):.2f}")
    ln()
    ln(f"  Stream C (Stage 1 predicted clear, prob >= {selected_threshold:.2f}):")
    ln(f"    n = {n_C_2023:,}  ({100*n_C_2023/max(len(nc_scores_2023),1):.1f}% of daytime)")
    if n_C_2023 > 0:
        ln(f"    mean = {np.mean(nc_C_2023):.2f} W/m2  "
           f"median = {np.median(nc_C_2023):.2f}  "
           f"90th pctile = {np.quantile(nc_C_2023, 0.90):.2f}")
    ln(f"    Window initialized with last {min(WINDOW_SIZE, n_C_2023):,} entries")
    ln()
    ln(f"  Stream K (Stage 1 predicted cloudy):")
    ln(f"    n = {n_K_2023:,}  ({100*n_K_2023/max(len(nc_scores_2023),1):.1f}% of daytime)")
    if n_K_2023 > 0:
        ln(f"    mean = {np.mean(nc_K_2023):.2f} W/m2  "
           f"median = {np.median(nc_K_2023):.2f}  "
           f"90th pctile = {np.quantile(nc_K_2023, 0.90):.2f}")
    ln(f"    Window initialized with last {min(WINDOW_SIZE, n_K_2023):,} entries")

    ln()
    ln("--- SINGLE-WINDOW RESULTS (Stage 3, gamma=0.005) ---")
    ln()
    ln(f"  {'Method':<30}  {'Coverage':>9}  {'Mean Width':>11}")
    ln("  " + "-" * 54)
    ln(f"  {'Adaptive conformal (Stage 3)':<30}  "
       f"{adap_s3_cov.mean():>9.4f}  {adap_s3_w.mean():>11.1f}")
    ln(f"  {'Static conformal':<30}  "
       f"{stat_cov.mean():>9.4f}  {stat_df['interval_width'].mean():>11.1f}")
    ln(f"  {'Naive Gaussian (1.96 sigma)':<30}  "
       f"{gauss_cov.mean():>9.4f}  {gauss_df['interval_width'].mean():>11.1f}")
    ln()
    ln("  Cloud regime coverage (single-window, true label):")
    ln(f"  {'Regime':<18}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>7}")
    ln("  " + "-" * 49)
    for label, mask in [("Clear-sky (true)", is_clear_true), ("Cloudy (true)", ~is_clear_true)]:
        s = cov_stats(adap_s3_cov, adap_s3_w, mask)
        ln(f"  {label:<18}  {s['coverage']:>9.4f}  {s['mean_width']:>11.1f}  "
           f"{s['n']:>7,}{flag_str(s['coverage'])}")
    ln()
    ln(f"  Winkler: Adaptive={ws_s3:.2f}  Static={ws_stat:.2f}  Gaussian={ws_gauss:.2f}")

    ln()
    ln("--- STRATIFIED ADAPTIVE RESULTS (Stage 3b) ---")
    ln()

    mask_C_pred = regime_pred_2024 == 1
    mask_K_pred = regime_pred_2024 == 0
    s_all = cov_stats(strat_cov, strat_w)
    s_C = cov_stats(strat_cov, strat_w, mask_C_pred)
    s_K = cov_stats(strat_cov, strat_w, mask_K_pred)

    ln("  a) Overall and per-stream marginal coverage:")
    ln(f"  {'Segment':<28}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>7}")
    ln("  " + "-" * 59)
    ln(f"  {'Overall (all daytime)':<28}  {s_all['coverage']:>9.4f}  "
       f"{s_all['mean_width']:>11.1f}  {s_all['n']:>7,}{flag_str(s_all['coverage'])}")
    ln(f"  {'Stream C (predicted clear)':<28}  {s_C['coverage']:>9.4f}  "
       f"{s_C['mean_width']:>11.1f}  {s_C['n']:>7,}{flag_str(s_C['coverage'])}")
    ln(f"  {'Stream K (predicted cloudy)':<28}  {s_K['coverage']:>9.4f}  "
       f"{s_K['mean_width']:>11.1f}  {s_K['n']:>7,}{flag_str(s_K['coverage'])}")
    if n_mismatch > 0:
        s_mis = cov_stats(strat_cov, strat_w, mismatch)
        ln(f"  {'Mismatch hours':<28}  {s_mis['coverage']:>9.4f}  "
           f"{s_mis['mean_width']:>11.1f}  {s_mis['n']:>7,}  "
           f"(rate={mismatch_rate:.3f})")

    ln()
    ln("  b) Cloud regime (true label):")
    ln(f"  {'Regime':<18}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>7}")
    ln("  " + "-" * 49)
    for label, mask in [("Clear-sky (true)", is_clear_true), ("Cloudy (true)", ~is_clear_true)]:
        s = cov_stats(strat_cov, strat_w, mask)
        ln(f"  {label:<18}  {s['coverage']:>9.4f}  {s['mean_width']:>11.1f}  "
           f"{s['n']:>7,}{flag_str(s['coverage'])}")

    ln()
    ln("  c) Cloud type:")
    ln(f"  {'Type':<22}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>7}")
    ln("  " + "-" * 53)
    for ct in sorted(set(ct_vals)):
        mask = ct_vals == ct
        s = cov_stats(strat_cov, strat_w, mask)
        if s["n"] < 10:
            continue
        name = CLOUD_TYPE_NAMES.get(int(ct), f"Type {ct}")
        ln(f"  {f'Type {ct} ({name})':<22}  {s['coverage']:>9.4f}  "
           f"{s['mean_width']:>11.1f}  {s['n']:>7,}{flag_str(s['coverage'])}")

    ln()
    ln("  d) Month:")
    ln(f"  {'Month':<8}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>7}")
    ln("  " + "-" * 40)
    for m in range(1, 13):
        mask = mo_vals == m
        s = cov_stats(strat_cov, strat_w, mask)
        if s["n"] == 0:
            continue
        ln(f"  {_cal.month_abbr[m]:<8}  {s['coverage']:>9.4f}  "
           f"{s['mean_width']:>11.1f}  {s['n']:>7,}{flag_str(s['coverage'])}")

    ln()
    ln("  e) Hour of day:")
    ln(f"  {'Hour':>5}  {'Coverage':>9}  {'Mean Width':>11}  {'n':>7}")
    ln("  " + "-" * 38)
    for hh in range(6, 20):
        mask = hr_vals == hh
        s = cov_stats(strat_cov, strat_w, mask)
        if s["n"] == 0:
            continue
        ln(f"  {hh:>5}  {s['coverage']:>9.4f}  {s['mean_width']:>11.1f}  "
           f"{s['n']:>7,}{flag_str(s['coverage'])}")

    ln()
    ln("  f) Ramp events:")
    for label, mask in [("Ramp events", ramp_mask_2024), ("Non-ramp", ~ramp_mask_2024)]:
        s = cov_stats(strat_cov, strat_w, mask)
        ln(f"  {label:<14}  coverage={s['coverage']:.4f}  "
           f"mean_width={s['mean_width']:.1f} W/m2  n={s['n']:,}{flag_str(s['coverage'])}")

    ln()
    ln("  g) Stream mismatch:")
    mis_true_clear = mismatch & is_clear_true
    mis_true_cloudy = mismatch & ~is_clear_true
    ln(f"    Total mismatch hours     : {n_mismatch:,} ({mismatch_rate*100:.1f}% of daytime)")
    ln(f"    Pred cloudy / true clear : {mis_true_clear.sum():,}")
    ln(f"    Pred clear  / true cloudy: {mis_true_cloudy.sum():,}")

    ln()
    ln("--- WINKLER SCORES ---")
    ln()
    ln(f"  {'Method':<34}  {'Winkler Score':>14}")
    ln("  " + "-" * 50)
    ln(f"  {'Stratified adaptive (Stage 3b)':<34}  {ws_strat:>14.2f}")
    ln(f"  {'Single-window adaptive (Stage 3)':<34}  {ws_s3:>14.2f}")
    ln(f"  {'Static conformal':<34}  {ws_stat:>14.2f}")
    ln(f"  {'Naive Gaussian (1.96 sigma)':<34}  {ws_gauss:>14.2f}")

    ln()
    ln("--- HEAD-TO-HEAD COMPARISON ---")
    ln()
    ln(f"  {'Metric':<32}  {'Strat Adap':>11}  {'Single Adap':>12}  "
       f"{'Static':>8}  {'Gaussian':>10}")
    ln("  " + "-" * 78)

    cov_cloud_strat = float(strat_cov[~is_clear_true].mean())
    cov_cloud_s3 = float(adap_s3_cov[~is_clear_true].mean())
    cov_cloud_stat = float(stat_cov[~is_clear_true].mean())
    cov_cloud_gauss = float(gauss_cov[~is_clear_true].mean())
    cov_ramp_strat = float(strat_cov[ramp_mask_2024].mean()) if ramp_mask_2024.sum() > 0 else float("nan")
    cov_ramp_s3 = float(adap_s3_cov[ramp_mask_2024].mean()) if ramp_mask_2024.sum() > 0 else float("nan")
    cov_ramp_stat = float(stat_cov[ramp_mask_2024].mean()) if ramp_mask_2024.sum() > 0 else float("nan")
    cov_ramp_gauss = float(gauss_cov[ramp_mask_2024].mean()) if ramp_mask_2024.sum() > 0 else float("nan")

    strat_monthly, s3_monthly, stat_monthly, gauss_monthly = [], [], [], []
    for m in range(1, 13):
        mask = mo_vals == m
        if mask.sum() >= 5:
            strat_monthly.append(float(strat_cov[mask].mean()))
            s3_monthly.append(float(adap_s3_cov[mask].mean()))
            stat_monthly.append(float(stat_cov[mask].mean()))
            gauss_monthly.append(float(gauss_cov[mask].mean()))

    def _row4(label, a, b, c, d, fmt=".4f"):
        fs = f"{{:{fmt}}}"
        ln(f"  {label:<32}  {fs.format(a):>11}  {fs.format(b):>12}  "
           f"{fs.format(c):>8}  {fs.format(d):>10}")

    _row4("Marginal coverage",
          s_all["coverage"], adap_s3_cov.mean(), stat_cov.mean(), gauss_cov.mean())
    _row4("Cloudy coverage (true label)",
          cov_cloud_strat, cov_cloud_s3, cov_cloud_stat, cov_cloud_gauss)
    _row4("Ramp coverage",
          cov_ramp_strat, cov_ramp_s3, cov_ramp_stat, cov_ramp_gauss)
    _row4("Monthly std(coverage)",
          np.std(strat_monthly), np.std(s3_monthly),
          np.std(stat_monthly), np.std(gauss_monthly))
    _row4("Winkler score", ws_strat, ws_s3, ws_stat, ws_gauss, fmt=".2f")
    _row4("Mean interval width (W/m2)",
          s_all["mean_width"], adap_s3_w.mean(),
          stat_df["interval_width"].mean(), gauss_df["interval_width"].mean(),
          fmt=".1f")

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

    strat_parquet = odir / f"conformal_stratified_predictions_{HOLDOUT_YEAR}.parquet"
    report_path = odir / "stage3_conformal_report.txt"

    print(SEP)
    print(f"Stage 3b -- Regime-Stratified Conformal Prediction Intervals  h={h}")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)
    print(f"\n  Horizon          : h={h}")
    print(f"  Conformal year   : {CONFORMAL_YEAR}")
    print(f"  Test year        : {HOLDOUT_YEAR}")
    print(f"  Alpha            : {ALPHA}")
    print(f"  Gamma            : {GAMMA}")
    print(f"  Window size      : {WINDOW_SIZE}")

    day, feat_full = load_and_prepare(h)
    print(f"\n[2] Loading models ...")
    s1_model, s2_model, s1_cols, s2_cols = load_models(h)

    selected_mode, selected_threshold = load_mode_selection(h)
    print(f"\n  Prediction mode  : {selected_mode}")
    print(f"  Threshold        : {selected_threshold:.2f}")

    print(f"\n{SEP}")
    print(f"STEP 1 -- {CONFORMAL_YEAR} WARM-UP (per-stream initialization)")
    print(SEP)
    d_conf = day[day.index.year == CONFORMAL_YEAR].copy()
    ghi_pred_conf, pred_clear_conf = predict_for_conformal(
        d_conf, s1_model, s2_model, s1_cols, s2_cols, selected_mode, selected_threshold
    )
    ghi_actual_conf = d_conf["ghi"].values
    nc_scores_conf = np.abs(ghi_actual_conf - ghi_pred_conf)
    nc_C_conf = nc_scores_conf[pred_clear_conf == 1]
    nc_K_conf = nc_scores_conf[pred_clear_conf == 0]
    n_C_conf = len(nc_C_conf)
    n_K_conf = len(nc_K_conf)
    print(f"\n  {CONFORMAL_YEAR} daytime hours  : {len(nc_scores_conf):,}")
    print(f"  Combined:  mean={np.mean(nc_scores_conf):.2f}  "
          f"median={np.median(nc_scores_conf):.2f}  "
          f"90th={np.quantile(nc_scores_conf, 0.90):.2f}  W/m2")
    print(f"\n  Stream C (pred clear) : {n_C_conf:,} "
          f"({100*n_C_conf/max(len(nc_scores_conf),1):.1f}%)")
    print(f"\n  Stream K (pred cloudy): {n_K_conf:,} "
          f"({100*n_K_conf/max(len(nc_scores_conf),1):.1f}%)")

    print(f"\n{SEP}")
    print(f"STEP 2 -- {HOLDOUT_YEAR} PREDICTIONS AND ROUTING")
    print(SEP)
    d24 = day[day.index.year == HOLDOUT_YEAR].copy()
    ghi_pred_2024, pred_clear_2024 = predict_for_conformal(
        d24, s1_model, s2_model, s1_cols, s2_cols, selected_mode, selected_threshold
    )
    ghi_actual_2024 = d24["ghi"].values
    d24["ghi_pred"] = ghi_pred_2024
    ramp_mask_2024 = compute_ramp_mask(d24, feat_full, HOLDOUT_YEAR)
    regime_true_2024 = d24["is_clear_sky"].values.astype(int)
    n_mismatch = int((pred_clear_2024 != regime_true_2024).sum())
    print(f"\n  {HOLDOUT_YEAR} daytime hours   : {len(d24):,}")
    print(f"  Stage1 pred clear    : {pred_clear_2024.sum():,}")
    print(f"  True clear-sky hours : {regime_true_2024.sum():,}")
    print(f"  Stream mismatch      : {n_mismatch:,} "
          f"({100*n_mismatch/max(len(d24),1):.1f}%)")
    print(f"  Ramp events          : {ramp_mask_2024.sum():,}")

    print(f"\n{SEP}")
    print("STEP 3 -- SINGLE-WINDOW ADAPTIVE LOOP (Stage 3 replication)")
    print(SEP)
    adap_df_s3 = adaptive_conformal_loop(
        ghi_pred_2024, ghi_actual_2024, list(nc_scores_conf),
        alpha=ALPHA, gamma=GAMMA, window_size=WINDOW_SIZE
    )
    adap_df_s3.index = d24.index
    print(f"\n  Marginal coverage : {adap_df_s3['covered'].mean():.4f}")
    print(f"  Mean width        : {adap_df_s3['interval_width'].mean():.1f} W/m2")

    print(f"\n{SEP}")
    print("STEP 4 -- STRATIFIED ADAPTIVE LOOP (Stage 3b)")
    print(SEP)
    strat_df = adaptive_stratified_loop(
        ghi_pred_2024, ghi_actual_2024, pred_clear_2024,
        list(nc_C_conf), list(nc_K_conf),
        alpha=ALPHA, gamma=GAMMA, window_size=WINDOW_SIZE
    )
    strat_df.index = d24.index
    strat_df["stream"] = pred_clear_2024
    mask_C = pred_clear_2024 == 1
    mask_K = pred_clear_2024 == 0
    is_clear_true = regime_true_2024.astype(bool)
    print(f"\n  Overall coverage  : {strat_df['covered'].mean():.4f}")
    print(f"  Mean width        : {strat_df['interval_width'].mean():.1f} W/m2")
    print(f"  Stream C coverage : {strat_df['covered'].values[mask_C].mean():.4f}  "
          f"(n={mask_C.sum():,})")
    print(f"  Stream K coverage : {strat_df['covered'].values[mask_K].mean():.4f}  "
          f"(n={mask_K.sum():,})")
    cov_cloud_true = float(strat_df["covered"].values[~is_clear_true].mean())
    print(f"  Cloudy cov (true) : {cov_cloud_true:.4f}")
    if cov_cloud_true < 0.85:
        print(f"  *** BELOW 0.85 -- check mismatch routing")

    print(f"\n{SEP}")
    print("STEP 5 -- STATIC CONFORMAL AND NAIVE GAUSSIAN")
    print(SEP)
    stat_df = static_conformal(ghi_pred_2024, ghi_actual_2024, nc_scores_conf)
    gauss_df = naive_gaussian(ghi_pred_2024, ghi_actual_2024, nc_scores_conf)
    stat_df.index = d24.index
    gauss_df.index = d24.index
    print(f"\n  Static q = {stat_df['q_t'].iloc[0]:.2f} W/m2  "
          f"coverage = {stat_df['covered'].mean():.4f}")
    print(f"  Gaussian coverage = {gauss_df['covered'].mean():.4f}")

    print(f"\n{SEP}")
    print("STEP 6 -- PLOTS")
    print(SEP)
    print()
    plot_stratified_alpha(d24, strat_df, pdir, h)
    plot_stratified_widths(d24, strat_df, pdir, h)
    plot_side_by_side_heatmap(d24, adap_df_s3, strat_df, pdir, h)

    print(f"\n{SEP}")
    print("STEP 7 -- SAVING RESULTS")
    print(SEP)
    out = pd.DataFrame({
        "GHI_actual": ghi_actual_2024,
        "GHI_pred": ghi_pred_2024,
        "lower": strat_df["lower"].values,
        "upper": strat_df["upper"].values,
        "covered": strat_df["covered"].values,
        "regime_pred": pred_clear_2024,
        "regime_true": regime_true_2024,
        "stream_mismatch": (pred_clear_2024 != regime_true_2024).astype(int),
        "alpha_C_t": strat_df["alpha_C_t"].values,
        "alpha_K_t": strat_df["alpha_K_t"].values,
        "q_t": strat_df["q_t"].values,
        "interval_width": strat_df["interval_width"].values,
        "cloud_type": d24["cloud_type"].values.astype(int),
        "is_ramp_event": ramp_mask_2024.astype(int),
        "month": d24.index.month,
        "hour_of_day": d24.index.hour,
    }, index=d24.index)
    out.to_parquet(strat_parquet)
    print(f"\n    Parquet saved -> {strat_parquet}  "
          f"({strat_parquet.stat().st_size/1024:.1f} KB)")

    write_combined_report(
        nc_scores_2023=nc_scores_conf,
        nc_C_2023=nc_C_conf,
        nc_K_2023=nc_K_conf,
        n_C_2023=n_C_conf,
        n_K_2023=n_K_conf,
        adap_df_s3=adap_df_s3,
        stat_df=stat_df,
        gauss_df=gauss_df,
        strat_df=strat_df,
        regime_pred_2024=pred_clear_2024,
        regime_true_2024=regime_true_2024,
        d24=d24,
        ramp_mask_2024=ramp_mask_2024,
        report_path=report_path,
        h=h,
        selected_mode=selected_mode,
        selected_threshold=selected_threshold,
    )

    print(f"\n{SEP}")
    print(f"STAGE 3b COMPLETE -- FILE SUMMARY  h={h}")
    print(SEP)
    for p in [strat_parquet, report_path]:
        size = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {p}  ({size:.1f} KB)")


if __name__ == "__main__":
    main()
