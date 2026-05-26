import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_utils import (
    features_path, horizon_dir, horizon_plots_dir,
    s1_feature_cols, DAYTIME_THRESH,
)

DAYTIME_THRESH = 50.0
TARGET_COL = "is_clear_sky"
THRESHOLD = 0.5

TRAIN_YEARS = list(range(2018, 2022))
EARLY_STOP_YEAR = 2022
HOLDOUT_YEAR = 2024

FOLDS = [
    (list(range(2018, 2019)), 2019),
    (list(range(2018, 2020)), 2020),
    (list(range(2018, 2021)), 2021),
    (list(range(2018, 2022)), 2022),
]

LGB_PARAMS = dict(
    objective="binary",
    metric=["binary_logloss", "auc"],
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
EARLY_STOP_PATIENCE = 50

SEP = "=" * 68
SEP2 = "-" * 68


def load_and_prepare(h: int) -> pd.DataFrame:
    fp = features_path(ROOT_DIR, h)
    if not fp.exists():
        raise FileNotFoundError(
            f"Features file not found: {fp}\n"
            f"Run  python src/features.py --horizon {h}  first."
        )
    print(f"\n[1] Loading {fp.name} ...")
    feat = pd.read_parquet(fp)
    feat.sort_index(inplace=True)
    print(f"    Full dataset: {feat.shape}")
    day = feat[feat["clearsky_ghi"] >= DAYTIME_THRESH].copy()
    print(f"    Daytime rows (clearsky_ghi >= {DAYTIME_THRESH} W/m2): {len(day):,}")
    n_clear = day[TARGET_COL].sum()
    n_cloudy = len(day) - n_clear
    print(f"\n    Class balance (daytime rows):")
    print(f"      is_clear_sky = 1 (Clear)  : {n_clear:6,}  ({n_clear/len(day)*100:.1f}%)")
    print(f"      is_clear_sky = 0 (Cloudy) : {n_cloudy:6,}  ({n_cloudy/len(day)*100:.1f}%)")
    return day


def _fit_fold(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
) -> tuple:
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP_PATIENCE, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = getattr(model, "best_iteration_", model.n_estimators_)
    return model, best_iter


def _fold_metrics(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return dict(
        auc=roc_auc_score(y_val, prob),
        f1=f1_score(y_val, pred, zero_division=0),
        precision=precision_score(y_val, pred, zero_division=0),
        recall=recall_score(y_val, pred, zero_division=0),
        brier=brier_score_loss(y_val, prob),
    )


def walk_forward_cv(day: pd.DataFrame, feature_cols: list) -> list:
    print(f"\n[2] Walk-forward cross-validation ({len(FOLDS)} folds) ...")
    print(f"\n    {'Fold':<5} {'Train years':<16} {'Val yr':<8} "
          f"{'Best iter':>10} {'AUC-ROC':>8} {'F1':>7} "
          f"{'Prec':>7} {'Recall':>7} {'Brier':>7}")
    print(f"    {'-'*5} {'-'*16} {'-'*8} {'-'*10} {'-'*8} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*7}")
    results = []
    for fold_idx, (train_yrs, val_yr) in enumerate(FOLDS, 1):
        train_mask = day.index.year.isin(train_yrs)
        val_mask = day.index.year == val_yr
        X_tr = day.loc[train_mask, feature_cols]
        y_tr = day.loc[train_mask, TARGET_COL]
        X_val = day.loc[val_mask, feature_cols]
        y_val = day.loc[val_mask, TARGET_COL]
        model, best_iter = _fit_fold(X_tr, y_tr, X_val, y_val)
        m = _fold_metrics(model, X_val, y_val)
        m["fold"] = fold_idx
        m["train_yrs"] = f"{train_yrs[0]}-{train_yrs[-1]}"
        m["val_yr"] = val_yr
        m["best_iter"] = best_iter
        results.append(m)
        print(f"    {fold_idx:<5} {m['train_yrs']:<16} {val_yr:<8} "
              f"{best_iter:>10,} {m['auc']:>8.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['brier']:>7.4f}")
    metric_keys = ["auc", "f1", "precision", "recall", "brier"]
    print(f"    {'':5} {'':16} {'':8} {'':10}", end="")
    for k in metric_keys:
        vals = [r[k] for r in results]
        print(f"  {np.mean(vals):>6.4f}", end="")
    print("  <- mean")
    return results


def train_final_model(day: pd.DataFrame, feature_cols: list) -> tuple:
    print(f"\n[3] Training final base model ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}) ...")
    base_mask = day.index.year.isin(TRAIN_YEARS)
    val_mask = day.index.year == EARLY_STOP_YEAR
    X_base = day.loc[base_mask, feature_cols]
    y_base = day.loc[base_mask, TARGET_COL]
    X_val = day.loc[val_mask, feature_cols]
    y_val = day.loc[val_mask, TARGET_COL]
    print(f"    Base train : {base_mask.sum():,} rows  "
          f"({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})")
    print(f"    Early-stop : {val_mask.sum():,} rows  ({EARLY_STOP_YEAR})")
    base_model, best_iter = _fit_fold(X_base, y_base, X_val, y_val)
    print(f"    Best iteration: {best_iter:,}")
    print(f"\n[4] Applying isotonic calibration (cv='prefit', on {EARLY_STOP_YEAR}) ...")
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="isotonic",
    )
    calibrated.fit(X_val, y_val)
    print("    Calibration fit complete.")
    return base_model, calibrated


def evaluate_on_holdout(
    calibrated, base_model, day: pd.DataFrame, feature_cols: list
) -> tuple:
    print(f"\n[5] Evaluating calibrated model on {HOLDOUT_YEAR} holdout ...")
    hold_mask = day.index.year == HOLDOUT_YEAR
    X_hold = day.loc[hold_mask, feature_cols]
    y_hold = day.loc[hold_mask, TARGET_COL]
    print(f"    Holdout rows: {len(y_hold):,}  "
          f"(clear={y_hold.sum():,}, cloudy={(~y_hold.astype(bool)).sum():,})")
    prob = calibrated.predict_proba(X_hold)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    metrics = dict(
        auc=roc_auc_score(y_hold, prob),
        f1=f1_score(y_hold, pred, zero_division=0),
        precision=precision_score(y_hold, pred, zero_division=0),
        recall=recall_score(y_hold, pred, zero_division=0),
        brier=brier_score_loss(y_hold, prob),
    )
    cm = confusion_matrix(y_hold, pred)
    print(f"\n    Confusion matrix (threshold={THRESHOLD}):")
    print(f"    {'':20}  Pred: Cloudy  Pred: Clear")
    print(f"    {'Actual: Cloudy':<20}  {cm[0,0]:>12,}  {cm[0,1]:>11,}")
    print(f"    {'Actual: Clear':<20}  {cm[1,0]:>12,}  {cm[1,1]:>11,}")
    print(f"\n    Classification report:")
    rpt = classification_report(
        y_hold, pred, target_names=["Cloudy (0)", "Clear (1)"], digits=4
    )
    for line in rpt.split("\n"):
        print(f"    {line}")
    return metrics, prob, y_hold


def naive_baseline(day: pd.DataFrame, h: int) -> dict:
    print(f"\n[6] Evaluating naive persistence baseline on {HOLDOUT_YEAR} holdout ...")
    hold_mask = day.index.year == HOLDOUT_YEAR
    lag_col = f"is_clear_sky_lag_{h}"
    sub = day.loc[hold_mask].dropna(subset=[lag_col])
    y_true = sub[TARGET_COL]
    y_prob = sub[lag_col].astype(float)
    y_pred = y_prob.astype(int)
    metrics = dict(
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        brier=brier_score_loss(y_true, y_prob),
    )
    print(f"    Naive AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
          f"Brier={metrics['brier']:.4f}")
    return metrics


def plot_calibration(
    calibrated, base_model, day: pd.DataFrame,
    feature_cols: list, plot_dir: Path, h: int,
) -> None:
    hold_mask = day.index.year == HOLDOUT_YEAR
    X_hold = day.loc[hold_mask, feature_cols]
    y_hold = day.loc[hold_mask, TARGET_COL]
    prob_cal = calibrated.predict_proba(X_hold)[:, 1]
    prob_raw = base_model.predict_proba(X_hold)[:, 1]
    fig, (ax_cal, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for prob, label, color, ls in [
        (prob_cal, "Calibrated (isotonic)", "#4CAF50", "-"),
        (prob_raw, "Raw LightGBM", "#FF9800", "--"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_hold, prob, n_bins=10, strategy="uniform")
        ax_cal.plot(mean_pred, frac_pos, "o", label=label,
                    color=color, lw=2, ms=6, ls=ls)
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")
    ax_cal.set_xlabel("Mean predicted probability", fontsize=12)
    ax_cal.set_ylabel("Fraction of positives (is_clear_sky=1)", fontsize=12)
    ax_cal.set_title(f"Reliability Diagram -- {HOLDOUT_YEAR} Holdout\n"
                     "(10 equal-width bins)", fontsize=12, fontweight="bold")
    ax_cal.legend(fontsize=10)
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.grid(alpha=0.3)
    bins = np.linspace(0, 1, 26)
    ax_hist.hist(prob_cal, bins=bins, color="#4CAF50", alpha=0.7,
                 edgecolor="white", lw=0.4, label="Calibrated")
    ax_hist.hist(prob_raw, bins=bins, color="#FF9800", alpha=0.5,
                 edgecolor="white", lw=0.4, label="Raw LightGBM")
    ax_hist.set_xlabel("Predicted probability", fontsize=12)
    ax_hist.set_ylabel("Hour count", fontsize=12)
    ax_hist.set_title("Distribution of Predicted\nProbabilities",
                      fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=10)
    ax_hist.grid(alpha=0.3, axis="y")
    plt.suptitle(
        f"Stage 1 Classifier -- Calibration Assessment  h={h}\n"
        "Babcock Ranch Solar Energy Center",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = plot_dir / f"stage1_calibration_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[7] Calibration plot saved -> {out_path}")


def plot_feature_importance(
    base_model: lgb.LGBMClassifier, plot_dir: Path, h: int
) -> None:
    booster = base_model.booster_
    importance = booster.feature_importance(importance_type="gain")
    feat_names = booster.feature_name()
    order = np.argsort(importance)
    imp_sort = importance[order]
    nam_sort = np.array(feat_names)[order]
    fig, ax = plt.subplots(figsize=(9, max(7, len(nam_sort) * 0.3)))
    bars = ax.barh(nam_sort, imp_sort, color="#90CAF9", edgecolor="white", lw=0.4)
    ax.set_xlabel("Feature importance (gain)", fontsize=12)
    ax.set_title(
        f"Stage 1 LightGBM -- Feature Importance (Gain)  h={h}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.3, axis="x")
    for bar, val in zip(bars, imp_sort):
        ax.text(val + imp_sort.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=7.5)
    plt.tight_layout()
    out_path = plot_dir / f"stage1_feature_importance_h{h:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[8] Feature importance plot saved -> {out_path}")


def save_artifacts(
    calibrated, feature_cols: list, mdir: Path
) -> None:
    print(f"\n[9] Saving artifacts ...")
    pkl_path = mdir / "stage1_classifier.pkl"
    feat_path = mdir / "stage1_feature_list.json"
    joblib.dump(calibrated, pkl_path)
    print(f"    {pkl_path}  ({pkl_path.stat().st_size/1024:.1f} KB)")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"    {feat_path}")


def print_summary(
    cv_results: list, hold_metrics: dict, naive_metrics: dict, h: int
) -> None:
    print(f"\n{SEP}")
    print(f"FINAL SUMMARY -- Stage 1 Clear-Sky Classifier  h={h}")
    print(SEP)
    print(f"\n  Temporal split:")
    print(f"    Train years      : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print(f"    Early-stop year  : {EARLY_STOP_YEAR}")
    print(f"    Test year        : {HOLDOUT_YEAR}")
    print(f"\n  Walk-Forward Cross-Validation (Folds 1-{len(cv_results)})")
    print(f"  {SEP2}")
    hdr = (f"  {'Fold':<5} {'Train':>12} {'Val yr':>7} {'Best iter':>10}  "
           f"{'AUC-ROC':>8} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Brier':>7}")
    print(hdr)
    print(f"  {'-'*5} {'-'*12} {'-'*7} {'-'*10}  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in cv_results:
        print(f"  {r['fold']:<5} {r['train_yrs']:>12} {r['val_yr']:>7} "
              f"{r['best_iter']:>10,}  "
              f"{r['auc']:>8.4f} {r['f1']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['brier']:>7.4f}")
    metric_keys = ["auc", "f1", "precision", "recall", "brier"]
    means = {k: np.mean([r[k] for r in cv_results]) for k in metric_keys}
    stds = {k: np.std([r[k] for r in cv_results]) for k in metric_keys}
    print(f"  {'-'*5} {'-'*12} {'-'*7} {'-'*10}  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'Mean':<5} {'':>12} {'':>7} {'':>10}  "
          + "  ".join(f"{means[k]:>6.4f}" for k in metric_keys))
    print(f"  {'Std':<5} {'':>12} {'':>7} {'':>10}  "
          + "  ".join(f"{stds[k]:>6.4f}" for k in metric_keys))
    print(f"\n  {HOLDOUT_YEAR} Holdout  (calibrated model vs naive persistence lag_{h})")
    print(f"  {SEP2}")
    print(f"  {'Model':<24}  {'AUC-ROC':>8}  {'F1':>7}  {'Prec':>7}  "
          f"{'Recall':>7}  {'Brier':>7}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    def _fmt(label, m):
        return (f"  {label:<24}  {m['auc']:>8.4f}  {m['f1']:>7.4f}  "
                f"{m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['brier']:>7.4f}")

    print(_fmt("Stage 1 (calibrated)", hold_metrics))
    print(_fmt(f"Naive persistence lag_{h}", naive_metrics))
    lift_auc = hold_metrics["auc"] - naive_metrics["auc"]
    lift_f1 = hold_metrics["f1"] - naive_metrics["f1"]
    lift_brier = naive_metrics["brier"] - hold_metrics["brier"]
    print(f"\n  Lift over naive:  dAUC={lift_auc:+.4f}  dF1={lift_f1:+.4f}  "
          f"dBrier={lift_brier:+.4f}  (positive Brier delta = better)")
    print(f"\n{SEP}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True,
                        help="Forecast horizon in hours (e.g. 1, 3, 6, 24).")
    args = parser.parse_args()
    h = args.horizon

    mdir = horizon_dir(ROOT_DIR, h)
    pdir = horizon_plots_dir(ROOT_DIR, h)
    feature_cols = s1_feature_cols(h)

    print(SEP)
    print(f"Stage 1 -- Clear-Sky Binary Classifier  h={h}")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)
    print(f"\n  Horizon         : h={h}")
    print(f"  Train years     : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print(f"  Early-stop year : {EARLY_STOP_YEAR}")
    print(f"  Test year       : {HOLDOUT_YEAR}")
    print(f"  Model dir       : {mdir}")
    print(f"  Features        : {len(feature_cols)} columns")

    day = load_and_prepare(h)
    cv_results = walk_forward_cv(day, feature_cols)
    base_model, calibrated = train_final_model(day, feature_cols)
    hold_metrics, prob_hold, y_hold = evaluate_on_holdout(
        calibrated, base_model, day, feature_cols
    )
    naive_metrics = naive_baseline(day, h)
    plot_calibration(calibrated, base_model, day, feature_cols, pdir, h)
    plot_feature_importance(base_model, pdir, h)
    save_artifacts(calibrated, feature_cols, mdir)
    print_summary(cv_results, hold_metrics, naive_metrics, h)


if __name__ == "__main__":
    main()
