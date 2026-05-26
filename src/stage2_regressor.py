import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_utils import (
    features_path, horizon_dir, horizon_outputs_dir, horizon_plots_dir,
    s1_feature_cols, s2_feature_cols, DAYTIME_THRESH,
    predict_ghi_for_mode, predict_ghi_hard, KT_CLEAR_STRATEGY,
)

TARGET_COL = "kt"
CLOUDY_COL = "is_clear_sky"
S1_THRESHOLD_FIXED = 0.65

TRAIN_YEARS = list(range(2018, 2022))
EARLY_STOP_YEAR = 2022
HOLDOUT_YEAR = 2024
VAL_YEAR = EARLY_STOP_YEAR

THRESHOLD_GRID = [round(v, 2) for v in np.arange(0.20, 0.86, 0.05)]

FOLDS = [
    (list(range(2018, 2019)), 2019),
    (list(range(2018, 2020)), 2020),
    (list(range(2018, 2021)), 2021),
    (list(range(2018, 2022)), 2022),
]

CLOUD_TYPE_NAMES = {
    2: "Fog", 3: "Water", 4: "SC Water", 5: "Mixed",
    6: "Opaque Ice", 7: "Cirrus", 8: "Overlapping",
    9: "Overshooting", 10: "Unknown", 11: "Dusty",
}

S2_LGB_PARAMS = dict(
    objective="regression",
    metric=["rmse", "mae"],
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

SEP = "=" * 68
SEP2 = "-" * 68


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
    print(f"    Full dataset : {feat.shape}")
    day = feat[feat["clearsky_ghi"] >= DAYTIME_THRESH].copy()
    n_cloudy = (day[CLOUDY_COL] == 0).sum()
    n_clear = (day[CLOUDY_COL] == 1).sum()
    print(f"    Daytime rows : {len(day):,}  (cloudy={n_cloudy:,}, clear={n_clear:,})")
    return day, feat


def print_kt_distribution(day: pd.DataFrame) -> None:
    print(f"\n[2] kt distribution -- cloudy training rows ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}) ...")
    mask = day.index.year.isin(TRAIN_YEARS) & (day[CLOUDY_COL] == 0)
    kt = day.loc[mask, TARGET_COL].dropna()
    print(f"    Rows  : {len(kt):,}")
    print(f"    Min   : {kt.min():.4f}")
    print(f"    Max   : {kt.max():.4f}")
    print(f"    Mean  : {kt.mean():.4f}")
    print(f"    Median: {kt.median():.4f}")
    print(f"    Std   : {kt.std():.4f}")
    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(kt, bins=edges)
    total = len(kt)
    print(f"\n    10-bin histogram (equal-width, [0, 1]):")
    print(f"    {'Bin':<14}  {'Count':>7}  {'%':>6}")
    for i, cnt in enumerate(counts):
        print(f"    [{edges[i]:.1f} - {edges[i+1]:.1f})  "
              f"{cnt:>7,}  {cnt/total*100:>5.1f}%")


def _fit_s2(X_tr, y_tr, X_val, y_val, n_est=None):
    params = dict(S2_LGB_PARAMS)
    if n_est is not None:
        params["n_estimators"] = n_est
    model = lgb.LGBMRegressor(**params)
    callbacks = []
    if n_est is None:
        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    kwargs = dict(eval_set=[(X_val, y_val)]) if n_est is None else {}
    model.fit(X_tr, y_tr, callbacks=callbacks, **kwargs)
    best = getattr(model, "best_iteration_", model.n_estimators_)
    return model, best


def walk_forward_cv(day: pd.DataFrame, s2_cols: list) -> list:
    print(f"\n[3] Walk-forward CV ({len(FOLDS)} folds, cloudy hours only) ...")
    print(f"\n    {'Fold':<5} {'Train':<16} {'Val':>6} {'Best iter':>10}  "
          f"{'RMSE':>7} {'MAE':>7} {'R2':>7}")
    print(f"    {'-'*5} {'-'*16} {'-'*6} {'-'*10}  {'-'*7} {'-'*7} {'-'*7}")
    results = []
    for idx, (train_yrs, val_yr) in enumerate(FOLDS, 1):
        tr_mask = day.index.year.isin(train_yrs) & (day[CLOUDY_COL] == 0)
        val_mask = (day.index.year == val_yr) & (day[CLOUDY_COL] == 0)
        X_tr = day.loc[tr_mask, s2_cols]
        y_tr = day.loc[tr_mask, TARGET_COL]
        X_val = day.loc[val_mask, s2_cols]
        y_val = day.loc[val_mask, TARGET_COL]
        model, best = _fit_s2(X_tr, y_tr, X_val, y_val)
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        train_str = f"{train_yrs[0]}-{train_yrs[-1]}"
        results.append(dict(fold=idx, train=train_str, val=val_yr,
                            best_iter=best, rmse=rmse, mae=mae, r2=r2))
        print(f"    {idx:<5} {train_str:<16} {val_yr:>6} {best:>10,}  "
              f"{rmse:>7.4f} {mae:>7.4f} {r2:>7.4f}")
    print(f"\n    {'-'*5} {'-'*16} {'-'*6} {'-'*10}  {'-'*7} {'-'*7} {'-'*7}")
    for metric in ["rmse", "mae", "r2"]:
        vals = [r[metric] for r in results]
        print(f"    {'Mean':>43}  {np.mean(vals):>7.4f}  ({np.std(vals):.4f} std)"
              if metric == "rmse"
              else f"    {'':>43}  {np.mean(vals):>7.4f}  ({np.std(vals):.4f} std)")
    return results


def train_final_s2_model(day: pd.DataFrame, cv_results: list, s2_cols: list):
    print(f"\n[4] Training final Stage 2 model "
          f"({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} cloudy) ...")
    mask = day.index.year.isin(TRAIN_YEARS) & (day[CLOUDY_COL] == 0)
    X = day.loc[mask, s2_cols]
    y = day.loc[mask, TARGET_COL]
    print(f"    Training rows: {len(y):,}")
    mean_iter = int(round(np.mean([r["best_iter"] for r in cv_results])))
    final_iter = min(1000, int(mean_iter * 1.10))
    print(f"    CV mean best iter: {mean_iter}  -> final n_estimators: {final_iter}")
    model, _ = _fit_s2(X, y, X, y, n_est=final_iter)
    return model


def build_climatological_baseline(day: pd.DataFrame) -> dict:
    mask = day.index.year.isin(TRAIN_YEARS) & (day[CLOUDY_COL] == 0)
    train = day.loc[mask].copy()
    clim = train.groupby(["month", "hour_of_day"])["kt"].mean().to_dict()
    print(f"    Climatological baseline: {len(clim)} (month, hour) buckets")
    return clim


def evaluate_s2_holdout(model, day: pd.DataFrame, clim: dict, s2_cols: list):
    print(f"\n[5] Evaluating Stage 2 on {HOLDOUT_YEAR} cloudy holdout ...")
    mask = (day.index.year == HOLDOUT_YEAR) & (day[CLOUDY_COL] == 0)
    hold = day.loc[mask].copy()
    print(f"    Holdout cloudy rows: {len(hold):,}")
    X = hold[s2_cols]
    y_arr = np.array(hold[TARGET_COL])
    y_pred_s2 = np.clip(model.predict(X), 0.0, 1.0)

    lag_col = [c for c in s2_cols if c.startswith("kt_lag_")][0]
    y_pers = np.array(hold[lag_col].fillna(hold[lag_col].median()))

    y_clim = np.array([
        clim.get((int(r["month"]), int(r["hour_of_day"])), float(hold["kt"].mean()))
        for _, r in hold.iterrows()
    ])

    def _metrics(y_t, y_p):
        return dict(
            rmse=float(np.sqrt(mean_squared_error(y_t, y_p))),
            mae=float(mean_absolute_error(y_t, y_p)),
            r2=float(r2_score(y_t, y_p)),
        )

    m_s2 = _metrics(y_arr, y_pred_s2)
    m_pers = _metrics(y_arr, y_pers)
    m_clim = _metrics(y_arr, y_clim)

    print(f"\n    {'Model':<24}  {'RMSE':>7}  {'MAE':>7}  {'R2':>7}")
    print(f"    {'-'*24}  {'-'*7}  {'-'*7}  {'-'*7}")
    for label, m in [("Stage 2 model", m_s2),
                     (f"Persistence ({lag_col})", m_pers),
                     ("Climatological mean", m_clim)]:
        print(f"    {label:<24}  {m['rmse']:>7.4f}  {m['mae']:>7.4f}  {m['r2']:>7.4f}")

    cloud_types = np.array(hold["cloud_type"])
    unique_ct = sorted(set(cloud_types.astype(int)))
    ct_metrics = {}
    print(f"\n    RMSE by cloud type:")
    for ct in unique_ct:
        ct_mask = cloud_types == ct
        if ct_mask.sum() < 5:
            continue
        ct_rmse = float(np.sqrt(mean_squared_error(y_arr[ct_mask], y_pred_s2[ct_mask])))
        ct_n = ct_mask.sum()
        name = CLOUD_TYPE_NAMES.get(ct, f"Type {ct}")
        ct_metrics[ct] = dict(rmse=ct_rmse, n=int(ct_n), name=name)
        print(f"      cloud_type {ct:>2} ({name:<14}): RMSE={ct_rmse:.4f}  n={ct_n:,}")

    return (dict(s2=m_s2, persistence=m_pers, climatological=m_clim),
            y_arr, y_pred_s2, ct_metrics, hold, lag_col)


def tune_stage1_threshold(
    day: pd.DataFrame, s1_model, s2_model, s1_cols: list, s2_cols: list
) -> tuple:
    val_day = day[day.index.year == VAL_YEAR].copy()
    ghi_actual = val_day["ghi"].values
    results = []
    for thr in THRESHOLD_GRID:
        ghi_pred, _, _ = predict_ghi_hard(
            val_day, s1_model, s2_model, s1_cols, s2_cols, thr
        )
        valid = np.isfinite(ghi_pred) & np.isfinite(ghi_actual)
        rmse = float(np.sqrt(np.mean((ghi_pred[valid] - ghi_actual[valid]) ** 2)))
        results.append({"threshold": thr, "val_rmse": rmse})
    best = min(results, key=lambda x: x["val_rmse"])
    return best["threshold"], results


def select_prediction_mode(
    day: pd.DataFrame, s1_model, s2_model,
    s1_cols: list, s2_cols: list, best_thresh: float
) -> tuple:
    val_day = day[day.index.year == VAL_YEAR].copy()
    ghi_actual = val_day["ghi"].values

    mode_results = {}
    for mode, thresh in [
        ("hard_fixed", S1_THRESHOLD_FIXED),
        ("hard_tuned", best_thresh),
        ("soft", best_thresh),
    ]:
        ghi_pred, _, _ = predict_ghi_for_mode(
            val_day, s1_model, s2_model, s1_cols, s2_cols, mode, thresh
        )
        valid = np.isfinite(ghi_pred) & np.isfinite(ghi_actual)
        rmse = float(np.sqrt(np.mean((ghi_pred[valid] - ghi_actual[valid]) ** 2)))
        mode_results[mode] = {
            "val_rmse": rmse,
            "threshold": thresh if mode != "soft" else None,
        }

    selected_mode = min(mode_results, key=lambda k: mode_results[k]["val_rmse"])
    return selected_mode, mode_results


def _rmse_mae_mbe(y_t, y_p):
    y_t = np.asarray(y_t, dtype=float)
    y_p = np.asarray(y_p, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if valid.sum() < 2:
        return dict(rmse=float("nan"), mae=float("nan"), mbe=float("nan"), n=0)
    yt, yp = y_t[valid], y_p[valid]
    return dict(
        rmse=float(np.sqrt(np.mean((yp - yt) ** 2))),
        mae=float(np.mean(np.abs(yp - yt))),
        mbe=float(np.mean(yp - yt)),
        n=int(valid.sum()),
    )


def _seg_metrics_np(
    ghi_act: np.ndarray, ghi_pred: np.ndarray,
    ghi_yest: np.ndarray, is_clr: np.ndarray, ramp: np.ndarray,
) -> dict:
    n = len(ghi_act)
    segments = {
        "All daytime": np.ones(n, dtype=bool),
        "Clear-sky": is_clr.astype(bool),
        "Cloudy": ~is_clr.astype(bool),
        "Ramp events": ramp.astype(bool),
    }
    results = {}
    for seg_name, seg_mask in segments.items():
        m_mod = _rmse_mae_mbe(ghi_act[seg_mask], ghi_pred[seg_mask])
        m_yest = _rmse_mae_mbe(ghi_act[seg_mask], ghi_yest[seg_mask])
        ss = (1.0 - m_mod["rmse"] / m_yest["rmse"]
              if m_yest["rmse"] > 0 and np.isfinite(m_mod["rmse"]) else float("nan"))
        results[seg_name] = dict(model=m_mod, yesterday=m_yest, skill=ss)
    return results


def build_seg_metrics(
    day: pd.DataFrame, ghi_2stage: pd.Series,
    ghi_yest: pd.Series, ramp: pd.Series,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n[11] Computing segment metrics ...")
    d24 = day[day.index.year == HOLDOUT_YEAR].copy()
    ghi_act = d24["ghi"]
    is_clr = d24[CLOUDY_COL].astype(bool)
    segments = {
        "All daytime": np.ones(len(d24), dtype=bool),
        "Clear-sky": is_clr.values,
        "Cloudy": (~is_clr).values,
        "Ramp events": ramp.values.astype(bool),
    }
    results = {}
    if verbose:
        print(f"\n    {'Segment':<14} {'RMSE_mod':>9}  {'MAE_mod':>8}  {'MBE_mod':>8}  "
              f"{'RMSE_yest':>10}  {'SkillScore':>11}")
        print(f"    {'-'*14} {'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*11}")
    for seg_name, seg_mask in segments.items():
        act = ghi_act.values[seg_mask]
        pred2 = ghi_2stage.values[seg_mask]
        predy = ghi_yest.values[seg_mask]
        m_mod = _rmse_mae_mbe(act, pred2)
        m_yest = _rmse_mae_mbe(act, predy)
        ss = (1.0 - m_mod["rmse"] / m_yest["rmse"]
              if m_yest["rmse"] > 0 and np.isfinite(m_mod["rmse"]) else float("nan"))
        results[seg_name] = dict(model=m_mod, yesterday=m_yest, skill=ss)
        if verbose:
            ss_s = f"{ss:+.4f}" if np.isfinite(ss) else "     N/A"
            print(f"    {seg_name:<14} {m_mod['rmse']:>9.2f}  {m_mod['mae']:>8.2f}  "
                  f"{m_mod['mbe']:>+8.2f}  {m_yest['rmse']:>10.2f}  {ss_s:>11}")
    return results


def plot_residuals(y_true, y_pred, save_path: Path) -> None:
    residuals = y_pred - y_true
    bias = np.mean(residuals)
    std = np.std(residuals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(residuals, bins=40, range=(-1, 1), color="#2196F3",
             alpha=0.75, edgecolor="white", lw=0.4)
    ax1.axvline(0, color="black", lw=1.5, ls="--", alpha=0.8, label="Zero bias")
    ax1.axvline(bias, color="red", lw=1.5, ls="-",
                label=f"Mean bias = {bias:+.4f}")
    ax1.set_xlabel("Residual  (predicted kt - actual kt)", fontsize=11)
    ax1.set_ylabel("Hour count", fontsize=11)
    ax1.set_title("Stage 2 -- Residual Distribution\n(2024 cloudy holdout)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.text(0.97, 0.97, f"std = {std:.4f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax1.grid(alpha=0.3, axis="y")
    ax2.scatter(y_true, y_pred, s=6, alpha=0.35, color="#2196F3", lw=0)
    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="1:1 line")
    ax2.set_xlabel("Actual kt", fontsize=11)
    ax2.set_ylabel("Predicted kt", fontsize=11)
    ax2.set_title("Stage 2 -- Predicted vs Actual kt\n(2024 cloudy holdout)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.suptitle("Stage 2 LightGBM Regressor -- Babcock Ranch",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[6] Residual plot saved -> {save_path}")


def plot_error_by_cloudtype(ct_metrics: dict, save_path: Path) -> None:
    cts = sorted(ct_metrics.keys())
    names = [f"Type {ct}\n{ct_metrics[ct]['name']}" for ct in cts]
    rmses = [ct_metrics[ct]["rmse"] for ct in cts]
    ns = [ct_metrics[ct]["n"] for ct in cts]
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(min(rmses), max(rmses))
    colors = [cmap(norm(r)) for r in rmses]
    fig, ax = plt.subplots(figsize=(max(8, len(cts) * 1.3), 5))
    bars = ax.bar(names, rmses, color=colors, edgecolor="white", lw=0.5)
    for bar, rmse, n in zip(bars, rmses, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{rmse:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE (kt units)", fontsize=11)
    ax.set_title("Stage 2 -- RMSE by Cloud Type\n2024 Cloudy Holdout  |  Babcock Ranch",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(rmses) * 1.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[7] Cloud-type error plot saved -> {save_path}")


def plot_feature_importance(model: lgb.LGBMRegressor, save_path: Path) -> None:
    booster = model.booster_
    imp = booster.feature_importance(importance_type="gain")
    names = np.array(booster.feature_name())
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, max(8, len(names) * 0.3)))
    bars = ax.barh(names[order], imp[order], color="#90CAF9",
                   edgecolor="white", lw=0.4)
    for bar, val in zip(bars, imp[order]):
        ax.text(val + imp[order].max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=7.5)
    ax.set_xlabel("Feature importance (gain)", fontsize=11)
    ax.set_title("Stage 2 LightGBM -- Feature Importance (Gain)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[8] Feature importance plot saved -> {save_path}")


def yesterday_ghi_forecast(day: pd.DataFrame) -> pd.Series:
    d24 = day[day.index.year == HOLDOUT_YEAR].copy()
    lag_col = "kt_lag_24"
    ghi_yest = d24[lag_col] * d24["clearsky_ghi"]
    n_nan = ghi_yest.isna().sum()
    print(f"    Yesterday's weather: {len(ghi_yest):,} rows  ({n_nan} NaN)")
    return ghi_yest.rename("ghi_pred_yest")


def compute_ramp_mask(day: pd.DataFrame, feat_full: pd.DataFrame) -> pd.Series:
    f = feat_full.copy()
    f["daytime"] = f["clearsky_ghi"] >= DAYTIME_THRESH
    f["prev_daytime"] = f["daytime"].shift(1).fillna(False)
    f["ghi_diff"] = f["ghi"].diff()
    ramp_full = (
        f["daytime"] & f["prev_daytime"] & (f["ghi_diff"].abs() > 100.0)
    )
    d24 = day[day.index.year == HOLDOUT_YEAR]
    ramp = ramp_full.reindex(d24.index, fill_value=False)
    print(f"    Ramp events in {HOLDOUT_YEAR} daytime: {ramp.sum():,}  "
          f"({ramp.sum()/len(ramp)*100:.1f}% of daytime hours)")
    return ramp


def plot_model_vs_yesterday(
    day: pd.DataFrame, feat_full: pd.DataFrame,
    ghi_2stage: pd.Series, ghi_yest: pd.Series,
    seg_metrics: dict, plot_dir: Path, h: int,
    selected_mode: str,
) -> None:
    print(f"\n[12] Generating 3-panel comparison plot ...")
    d24 = day[day.index.year == HOLDOUT_YEAR].copy()
    ghi_act = d24["ghi"]
    is_clr = d24[CLOUDY_COL].astype(bool)

    f = feat_full[feat_full.index.year == HOLDOUT_YEAR].copy()
    f["daytime"] = f["clearsky_ghi"] >= DAYTIME_THRESH
    f["prev_daytime"] = f["daytime"].shift(1).fillna(False)
    f["transition"] = (
        f["daytime"] & f["prev_daytime"] &
        (f["is_clear_sky"] != f["is_clear_sky"].shift(1))
    ).astype(int)
    weekly = f["transition"].resample("W-MON").sum()
    if weekly.empty:
        wk_start = pd.Timestamp(f"{HOLDOUT_YEAR}-07-01", tz="UTC")
    else:
        best_end = weekly.idxmax()
        wk_start = best_end - pd.Timedelta(days=6)
    wk_end = wk_start + pd.Timedelta(days=7)

    wk_mask = (d24.index >= wk_start) & (d24.index <= wk_end)
    wk_data = d24.loc[wk_mask]
    wk_2st = ghi_2stage.reindex(wk_data.index).fillna(0)
    wk_yest = ghi_yest.reindex(wk_data.index)

    valid_both = ghi_2stage.index.intersection(ghi_yest.dropna().index)
    act_v = ghi_act.reindex(valid_both).values
    st2_v = ghi_2stage.reindex(valid_both).values
    yst_v = ghi_yest.reindex(valid_both).values
    clr_v = is_clr.reindex(valid_both).values

    fig = plt.figure(figsize=(18, 14))
    gs = mgridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30,
                            height_ratios=[1.2, 1.0, 0.9])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_sc1 = fig.add_subplot(gs[1, 0])
    ax_sc2 = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[2, :])

    ax_ts.plot(wk_data.index, wk_data["ghi"], color="black", lw=1.8, label="Actual GHI")
    ax_ts.plot(wk_data.index, wk_2st.values, color="#1565C0", lw=1.6,
               label=f"Two-stage model ({selected_mode})")
    ax_ts.plot(wk_data.index, wk_yest.values, color="#E65100", lw=1.4,
               ls="--", label="Yesterday's weather")
    ax_ts.set_xlabel("Date (UTC)", fontsize=10)
    ax_ts.set_ylabel("GHI  (W/m2)", fontsize=10)
    ax_ts.set_title(
        f"Panel 1 -- Representative Week  "
        f"({wk_start.date()} - {(wk_end - pd.Timedelta(days=1)).date()})",
        fontsize=11, fontweight="bold")
    ax_ts.legend(fontsize=9, ncol=3)
    ax_ts.grid(alpha=0.3)

    import matplotlib.patches as mpatches
    c_pts = np.where(clr_v, "#4CAF50", "#FF9800")
    ax_sc1.scatter(act_v, st2_v, s=5, c=c_pts, alpha=0.3, lw=0)
    ax_sc1.plot([0, 1200], [0, 1200], "k--", lw=1.2, alpha=0.6)
    ax_sc1.set_xlabel("Actual GHI  (W/m2)", fontsize=10)
    ax_sc1.set_ylabel("Predicted GHI  (W/m2)", fontsize=10)
    ax_sc1.set_title("Panel 2a -- Two-Stage Model vs Actual", fontsize=10, fontweight="bold")
    ax_sc1.set_xlim(-20, 1300)
    ax_sc1.set_ylim(-20, 1300)
    ax_sc1.grid(alpha=0.3)
    rmse_2st = seg_metrics["All daytime"]["model"]["rmse"]
    ax_sc1.text(0.04, 0.96, f"RMSE = {rmse_2st:.1f} W/m2",
                transform=ax_sc1.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_sc1.legend(handles=[
        mpatches.Patch(color="#4CAF50", label="Clear-sky"),
        mpatches.Patch(color="#FF9800", label="Cloudy"),
    ], fontsize=8, loc="lower right")

    ax_sc2.scatter(act_v, yst_v, s=5, c=c_pts, alpha=0.3, lw=0)
    ax_sc2.plot([0, 1200], [0, 1200], "k--", lw=1.2, alpha=0.6)
    ax_sc2.set_xlabel("Actual GHI  (W/m2)", fontsize=10)
    ax_sc2.set_ylabel("Predicted GHI  (W/m2)", fontsize=10)
    ax_sc2.set_title("Panel 2b -- Yesterday's Weather vs Actual",
                     fontsize=10, fontweight="bold")
    ax_sc2.set_xlim(-20, 1300)
    ax_sc2.set_ylim(-20, 1300)
    ax_sc2.grid(alpha=0.3)
    rmse_yest = seg_metrics["All daytime"]["yesterday"]["rmse"]
    ax_sc2.text(0.04, 0.96, f"RMSE = {rmse_yest:.1f} W/m2",
                transform=ax_sc2.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax_sc2.legend(handles=[
        mpatches.Patch(color="#4CAF50", label="Clear-sky"),
        mpatches.Patch(color="#FF9800", label="Cloudy"),
    ], fontsize=8, loc="lower right")

    seg_order = ["All daytime", "Clear-sky", "Cloudy", "Ramp events"]
    x = np.arange(len(seg_order))
    width = 0.35
    rmse_2 = [seg_metrics[s]["model"]["rmse"] for s in seg_order]
    rmse_y = [seg_metrics[s]["yesterday"]["rmse"] for s in seg_order]
    skills = [seg_metrics[s]["skill"] for s in seg_order]
    bars2 = ax_bar.bar(x - width / 2, rmse_2, width, label=f"Two-stage ({selected_mode})",
                       color="#1565C0", alpha=0.85)
    barsy = ax_bar.bar(x + width / 2, rmse_y, width, label="Yesterday's weather",
                       color="#E65100", alpha=0.85)
    for bar, val, ss in zip(bars2, rmse_2, skills):
        ss_str = f"SS={ss:+.3f}" if np.isfinite(ss) else ""
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}\n{ss_str}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(barsy, rmse_y):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(seg_order, fontsize=10)
    ax_bar.set_ylabel("RMSE  (W/m2)", fontsize=10)
    ax_bar.set_title("Panel 3 -- RMSE by Segment: Two-Stage vs Yesterday's Weather",
                     fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(alpha=0.3, axis="y")
    ax_bar.set_ylim(0, max(max(rmse_2), max(rmse_y)) * 1.30)

    fig.suptitle(
        f"Two-Stage Solar Forecast vs Yesterday's Weather -- {HOLDOUT_YEAR} Holdout  h={h}\n"
        "Babcock Ranch Solar Energy Center",
        fontsize=13, fontweight="bold", y=1.005,
    )
    out_path = plot_dir / f"model_vs_yesterday_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def write_report(
    cv_results: list, s2_eval_metrics: dict, ct_metrics: dict,
    thresh_results: list, best_thresh: float,
    mode_val_results: dict, selected_mode: str,
    all_mode_seg: dict, seg_metrics_selected: dict,
    report_path: Path, h: int,
) -> None:
    L = []

    def ln(s=""):
        L.append(s)

    ln(SEP)
    ln(f"Stage 2 Full Report -- Babcock Ranch Solar Energy Center  h={h}")
    ln(SEP)
    ln()
    ln(f"  horizon             : h={h}")
    ln(f"  train years         : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    ln(f"  early-stop year     : {EARLY_STOP_YEAR}")
    ln(f"  test year           : {HOLDOUT_YEAR}")
    ln(f"  selected_mode       : {selected_mode}")
    thr_display = f"{best_thresh:.2f}" if selected_mode != "hard_fixed" else f"{S1_THRESHOLD_FIXED:.2f}"
    if selected_mode == "hard_fixed":
        thr_display = f"{S1_THRESHOLD_FIXED:.2f}"
    elif selected_mode == "hard_tuned":
        thr_display = f"{best_thresh:.2f}"
    else:
        thr_display = f"{best_thresh:.2f} (routing only)"
    ln(f"  threshold_used      : {thr_display}")
    ln(f"  clear_kt_strategy   : {KT_CLEAR_STRATEGY}")
    ln()

    ln("--- STAGE 2 CV RESULTS ---")
    ln()
    ln(f"  {'Fold':<5} {'Train':<16} {'Val':>6} {'Best iter':>10}  "
       f"{'RMSE':>7}  {'MAE':>7}  {'R2':>7}")
    ln(f"  {'-'*5} {'-'*16} {'-'*6} {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}")
    for r in cv_results:
        ln(f"  {r['fold']:<5} {r['train']:<16} {r['val']:>6} "
           f"{r['best_iter']:>10,}  {r['rmse']:>7.4f}  {r['mae']:>7.4f}  {r['r2']:>7.4f}")
    for key, label in [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2")]:
        vals = [r[key] for r in cv_results]
        ln(f"  {label} mean +/- std: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    ln()

    ln("--- STAGE 2 HOLDOUT RESULTS ---")
    ln()
    ln(f"  {'Model':<24}  {'RMSE':>7}  {'MAE':>7}  {'R2':>7}")
    ln(f"  {'-'*24}  {'-'*7}  {'-'*7}  {'-'*7}")
    for label, m in [("Stage 2 model", s2_eval_metrics["s2"]),
                     ("Persistence", s2_eval_metrics["persistence"]),
                     ("Climatological mean", s2_eval_metrics["climatological"])]:
        ln(f"  {label:<24}  {m['rmse']:>7.4f}  {m['mae']:>7.4f}  {m['r2']:>7.4f}")
    ln()
    ln(f"  RMSE by cloud type:")
    for ct, v in sorted(ct_metrics.items()):
        ln(f"    cloud_type {ct:>2} ({v['name']:<14})  RMSE={v['rmse']:.4f}  n={v['n']:,}")
    ln()

    ln("--- STAGE 1 THRESHOLD TUNING ---")
    ln()
    ln(f"  validation_year  : {VAL_YEAR}")
    ln(f"  selection_metric : all-daytime GHI RMSE")
    ln(f"  fixed_threshold  : {S1_THRESHOLD_FIXED:.2f}")
    ln(f"  selected_threshold : {best_thresh:.2f}")
    ln()
    ln(f"  {'Threshold':>10}  {'Val_RMSE':>10}")
    ln(f"  {'-'*10}  {'-'*10}")
    for r in thresh_results:
        marker = " *" if abs(r["threshold"] - best_thresh) < 1e-9 else \
                 " (fixed)" if abs(r["threshold"] - S1_THRESHOLD_FIXED) < 1e-9 else ""
        ln(f"  {r['threshold']:>10.2f}  {r['val_rmse']:>10.2f}{marker}")
    ln()

    ln("--- PREDICTION MODE COMPARISON ---")
    ln()
    ln(f"  validation_year : {VAL_YEAR}")
    ln()
    ln(f"  {'mode':<14}  {'threshold':>10}  {'val_rmse':>10}")
    ln(f"  {'-'*14}  {'-'*10}  {'-'*10}")
    for mode in ["hard_fixed", "hard_tuned", "soft"]:
        m = mode_val_results[mode]
        thr_s = f"{m['threshold']:.2f}" if m["threshold"] is not None else "N/A"
        marker = " *selected*" if mode == selected_mode else ""
        ln(f"  {mode:<14}  {thr_s:>10}  {m['val_rmse']:>10.2f}{marker}")
    ln()
    ln(f"  2024 holdout -- all modes:")
    ln()
    hdr = (f"  {'mode':<14}  {'threshold':>10}  "
           f"{'all_rmse':>9}  {'clear_rmse':>10}  {'cloudy_rmse':>11}  "
           f"{'ramp_rmse':>9}  {'all_skill':>10}")
    ln(hdr)
    ln("  " + "-" * (len(hdr) - 2))
    yest_all = all_mode_seg["hard_fixed"]["All daytime"]["yesterday"]["rmse"]
    for mode in ["hard_fixed", "hard_tuned", "soft"]:
        sm = all_mode_seg[mode]
        m_all = sm["All daytime"]["model"]
        m_clr = sm["Clear-sky"]["model"]
        m_cld = sm["Cloudy"]["model"]
        m_rmp = sm["Ramp events"]["model"]
        ss_all = sm["All daytime"]["skill"]
        marker = " *selected*" if mode == selected_mode else ""
        thr_v = mode_val_results[mode]["threshold"]
        thr_s = f"{thr_v:.2f}" if thr_v is not None else "N/A"
        ln(f"  {mode:<14}  {thr_s:>10}  "
           f"{m_all['rmse']:>9.2f}  {m_clr['rmse']:>10.2f}  "
           f"{m_cld['rmse']:>11.2f}  {m_rmp['rmse']:>9.2f}  "
           f"{ss_all:>+10.4f}{marker}")
    m_y = all_mode_seg["hard_fixed"]["All daytime"]["yesterday"]
    m_y_clr = all_mode_seg["hard_fixed"]["Clear-sky"]["yesterday"]
    m_y_cld = all_mode_seg["hard_fixed"]["Cloudy"]["yesterday"]
    m_y_rmp = all_mode_seg["hard_fixed"]["Ramp events"]["yesterday"]
    ln(f"  {'yesterday':<14}  {'N/A':>10}  "
       f"{m_y['rmse']:>9.2f}  {m_y_clr['rmse']:>10.2f}  "
       f"{m_y_cld['rmse']:>11.2f}  {m_y_rmp['rmse']:>9.2f}  "
       f"{'---':>10}")
    ln()

    ln("--- YESTERDAY'S WEATHER COMPARISON ---")
    ln()
    ln(f"  Segment          RMSE_mod   MAE_mod   MBE_mod   RMSE_yest   SkillScore")
    ln(f"  {'-'*14}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*11}")
    for seg, v in seg_metrics_selected.items():
        m, y = v["model"], v["yesterday"]
        ss = v["skill"]
        ss_s = f"{ss:+.4f}" if np.isfinite(ss) else "     N/A"
        ln(f"  {seg:<14}  {m['rmse']:>9.2f}  {m['mae']:>8.2f}  {m['mbe']:>+8.2f}  "
           f"{y['rmse']:>10.2f}  {ss_s:>11}")
    ln()
    ln(SEP)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[13] Report saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True,
                        help="Forecast horizon in hours (e.g. 1, 3, 6, 24).")
    args = parser.parse_args()
    h = args.horizon

    mdir = horizon_dir(ROOT_DIR, h)
    odir = horizon_outputs_dir(ROOT_DIR, h)
    pdir = horizon_plots_dir(ROOT_DIR, h)
    s1_cols = s1_feature_cols(h)
    s2_cols = s2_feature_cols(h)

    print(SEP)
    print(f"Stage 2 Regressor + Yesterday's Weather Comparison  h={h}")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)
    print(f"\n  Horizon         : h={h}")
    print(f"  Train years     : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print(f"  Early-stop year : {EARLY_STOP_YEAR}")
    print(f"  Test year       : {HOLDOUT_YEAR}")
    print(f"  Model dir       : {mdir}")
    print(f"  S1 features     : {len(s1_cols)}")
    print(f"  S2 features     : {len(s2_cols)}")

    day, feat_full = load_and_prepare(h)
    print_kt_distribution(day)
    cv_results = walk_forward_cv(day, s2_cols)
    s2_model = train_final_s2_model(day, cv_results, s2_cols)

    print(f"\n[4b] Building climatological baseline ...")
    clim = build_climatological_baseline(day)

    s2_metrics, y_true, y_pred_s2, ct_metrics, hold, lag_col = evaluate_s2_holdout(
        s2_model, day, clim, s2_cols
    )

    plot_residuals(y_true, y_pred_s2, pdir / f"stage2_residuals_h{h:03d}.png")
    plot_error_by_cloudtype(ct_metrics, pdir / f"stage2_error_by_cloudtype_h{h:03d}.png")
    plot_feature_importance(s2_model, pdir / f"stage2_feature_importance_h{h:03d}.png")

    s2_pkl = mdir / "stage2_regressor.pkl"
    s2_feat_json = mdir / "stage2_feature_list.json"
    print(f"\n[9] Saving Stage 2 artifacts ...")
    joblib.dump(s2_model, s2_pkl)
    print(f"    {s2_pkl}  ({s2_pkl.stat().st_size/1024:.1f} KB)")
    with open(s2_feat_json, "w") as f:
        json.dump(s2_cols, f, indent=2)
    print(f"    {s2_feat_json}")

    s1_pkl = mdir / "stage1_classifier.pkl"
    if not s1_pkl.exists():
        raise FileNotFoundError(
            f"Stage 1 pkl not found: {s1_pkl}\n"
            f"Run  python src/stage1_finalize.py --horizon {h}  first."
        )
    s1_model = joblib.load(s1_pkl)
    print(f"\n    Loaded Stage 1 model: {s1_pkl.name}")

    print(f"\n[9b] Stage 1 threshold tuning (val year={VAL_YEAR}) ...")
    best_thresh, thresh_results = tune_stage1_threshold(
        day, s1_model, s2_model, s1_cols, s2_cols
    )
    print(f"    Fixed threshold : {S1_THRESHOLD_FIXED:.2f}")
    print(f"    Selected threshold: {best_thresh:.2f}  "
          f"(val RMSE = {next(r['val_rmse'] for r in thresh_results if abs(r['threshold']-best_thresh)<1e-9):.2f})")

    thresh_json = mdir / "stage1_threshold.json"
    with open(thresh_json, "w") as f:
        json.dump({
            "horizon": h,
            "selected_threshold": best_thresh,
            "selection_metric": "all-daytime GHI RMSE",
            "validation_year": VAL_YEAR,
            "fixed_threshold": S1_THRESHOLD_FIXED,
            "candidate_thresholds": THRESHOLD_GRID,
            "validation_metrics_by_threshold": thresh_results,
        }, f, indent=2)
    print(f"    Saved: {thresh_json}")

    print(f"\n[9c] Prediction mode selection (val year={VAL_YEAR}) ...")
    selected_mode, mode_val_results = select_prediction_mode(
        day, s1_model, s2_model, s1_cols, s2_cols, best_thresh
    )
    sel_thresh = best_thresh if selected_mode != "hard_fixed" else S1_THRESHOLD_FIXED
    for mode, res in mode_val_results.items():
        marker = " *selected*" if mode == selected_mode else ""
        thr_s = f"{res['threshold']:.2f}" if res["threshold"] is not None else "N/A"
        print(f"    {mode:<14}  thresh={thr_s}  val_rmse={res['val_rmse']:.2f}{marker}")

    mode_json = mdir / "prediction_mode_selection.json"
    with open(mode_json, "w") as f:
        json.dump({
            "horizon": h,
            "selected_prediction_mode": selected_mode,
            "selected_threshold": sel_thresh,
            "validation_year": VAL_YEAR,
            "validation_metrics_by_mode": mode_val_results,
            "clear_kt_strategy": KT_CLEAR_STRATEGY,
        }, f, indent=2)
    print(f"    Saved: {mode_json}")

    print(f"\n[10] Computing segment metrics for all modes ({HOLDOUT_YEAR}) ...")
    d24 = day[day.index.year == HOLDOUT_YEAR].copy()
    ghi_act_arr = d24["ghi"].values
    is_clr_arr = d24[CLOUDY_COL].values.astype(bool)

    print(f"\n[10b] Computing yesterday's weather baseline ...")
    ghi_yest_ser = yesterday_ghi_forecast(day)
    ghi_yest_arr = ghi_yest_ser.reindex(d24.index).values

    print(f"\n[10c] Computing ramp event mask ...")
    ramp_ser = compute_ramp_mask(day, feat_full)
    ramp_arr = ramp_ser.values

    all_mode_seg = {}
    all_mode_ghi = {}
    all_mode_pred_clear = {}
    for mode, thresh in [
        ("hard_fixed", S1_THRESHOLD_FIXED),
        ("hard_tuned", best_thresh),
        ("soft", best_thresh),
    ]:
        ghi_pred_arr, _, pred_c = predict_ghi_for_mode(
            d24, s1_model, s2_model, s1_cols, s2_cols, mode, thresh
        )
        all_mode_ghi[mode] = ghi_pred_arr
        all_mode_pred_clear[mode] = pred_c
        all_mode_seg[mode] = _seg_metrics_np(
            ghi_act_arr, ghi_pred_arr, ghi_yest_arr, is_clr_arr, ramp_arr
        )

    print(f"\n[11] Segment metrics summary (selected: {selected_mode}):")
    print(f"\n    {'Segment':<14} {'RMSE_mod':>9}  {'MAE_mod':>8}  {'MBE_mod':>8}  "
          f"{'RMSE_yest':>10}  {'SkillScore':>11}")
    print(f"    {'-'*14} {'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*11}")
    for seg, v in all_mode_seg[selected_mode].items():
        m, y, ss = v["model"], v["yesterday"], v["skill"]
        ss_s = f"{ss:+.4f}" if np.isfinite(ss) else "     N/A"
        print(f"    {seg:<14} {m['rmse']:>9.2f}  {m['mae']:>8.2f}  "
              f"{m['mbe']:>+8.2f}  {y['rmse']:>10.2f}  {ss_s:>11}")

    ghi_selected_ser = pd.Series(all_mode_ghi[selected_mode], index=d24.index,
                                 name="ghi_pred_2stage")
    plot_model_vs_yesterday(
        day, feat_full, ghi_selected_ser, ghi_yest_ser,
        all_mode_seg[selected_mode], pdir, h, selected_mode
    )

    write_report(
        cv_results=cv_results,
        s2_eval_metrics=s2_metrics,
        ct_metrics=ct_metrics,
        thresh_results=thresh_results,
        best_thresh=best_thresh,
        mode_val_results=mode_val_results,
        selected_mode=selected_mode,
        all_mode_seg=all_mode_seg,
        seg_metrics_selected=all_mode_seg[selected_mode],
        report_path=odir / "stage2_full_report.txt",
        h=h,
    )

    print(f"\n{SEP}")
    print("FILES SAVED")
    print(SEP)
    for p in [s2_pkl, s2_feat_json, thresh_json, mode_json]:
        if p.exists():
            print(f"  {p}  ({p.stat().st_size/1024:.1f} KB)")
        else:
            print(f"  {p}  (MISSING)")

    print(f"\n{SEP}")
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
