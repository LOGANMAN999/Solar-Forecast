import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_utils import (
    features_path, horizon_dir, horizon_outputs_dir,
    s1_feature_cols, DAYTIME_THRESH,
)

TARGET_COL = "is_clear_sky"
TRAIN_YEARS = list(range(2018, 2022))
EARLY_STOP_YEAR = 2022
HOLDOUT_YEAR = 2024
OP_THRESHOLD = 0.65

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

SEP = "=" * 68


def load_and_prepare(h: int) -> pd.DataFrame:
    fp = features_path(ROOT_DIR, h)
    feat = pd.read_parquet(fp)
    feat.sort_index(inplace=True)
    return feat[feat["clearsky_ghi"] >= DAYTIME_THRESH].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True,
                        help="Forecast horizon in hours (e.g. 1, 3, 6, 24).")
    args = parser.parse_args()
    h = args.horizon

    mdir = horizon_dir(ROOT_DIR, h)
    odir = horizon_outputs_dir(ROOT_DIR, h)
    feature_cols = s1_feature_cols(h)

    report_path = odir / "stage1_calibration_report.txt"

    print(SEP)
    print(f"Stage 1 Finalize -- Raw LightGBM, threshold = {OP_THRESHOLD}  h={h}")
    print(SEP)
    print(f"\n  Horizon         : h={h}")
    print(f"  Train years     : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print(f"  Early-stop year : {EARLY_STOP_YEAR}")
    print(f"  Test year       : {HOLDOUT_YEAR}")
    print(f"  Model dir       : {mdir}")

    day = load_and_prepare(h)
    print(f"Daytime rows: {len(day):,}")

    base_mask = day.index.year.isin(TRAIN_YEARS)
    val_mask = day.index.year == EARLY_STOP_YEAR
    hold_mask = day.index.year == HOLDOUT_YEAR

    X_base = day.loc[base_mask, feature_cols]
    y_base = day.loc[base_mask, TARGET_COL]
    X_val = day.loc[val_mask, feature_cols]
    y_val = day.loc[val_mask, TARGET_COL]
    X_hold = day.loc[hold_mask, feature_cols]
    y_hold = day.loc[hold_mask, TARGET_COL]

    print(f"\nTraining raw LightGBM on {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} "
          f"(early stop on {EARLY_STOP_YEAR}) ...")
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_base, y_base,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = getattr(model, "best_iteration_", model.n_estimators_)
    print(f"Best iteration: {best_iter:,}")

    probs = model.predict_proba(X_hold)[:, 1]
    pred = (probs >= OP_THRESHOLD).astype(int)
    ya = np.array(y_hold)

    cm = confusion_matrix(ya, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = dict(
        auc=roc_auc_score(ya, probs),
        f1=f1_score(ya, pred, zero_division=0),
        precision=precision_score(ya, pred, zero_division=0),
        recall=recall_score(ya, pred, zero_division=0),
        brier=brier_score_loss(ya, probs),
    )

    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0

    print(f"\n{HOLDOUT_YEAR} Holdout -- Raw LightGBM @ threshold = {OP_THRESHOLD}:")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Brier     : {metrics['brier']:.4f}")
    print(f"\n  Confusion matrix (threshold = {OP_THRESHOLD}):")
    print(f"  {'':18}  Pred: Cloudy  Pred: Clear")
    print(f"  {'Actual: Cloudy':<18}  {tn:>12,}  {fp:>11,}")
    print(f"  {'Actual: Clear':<18}  {fn:>12,}  {tp:>11,}")
    print(f"\n  False clear-sky rate (FPR): {fpr:.4f}  ({fp} FP out of {tn+fp} actual cloudy)")
    print(f"  False cloudy rate   (FNR): {fnr:.4f}  ({fn} FN out of {tp+fn} actual clear)")

    pkl_path = mdir / "stage1_classifier.pkl"
    joblib.dump(model, pkl_path)
    print(f"\nSaved: {pkl_path}  ({pkl_path.stat().st_size/1024:.1f} KB)")

    feat_path = mdir / "stage1_feature_list.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved: {feat_path}")

    meta = {
        "method": "raw",
        "horizon": h,
        "train_years": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "early_stop_year": EARLY_STOP_YEAR,
        "holdout_year": HOLDOUT_YEAR,
        "best_iteration": best_iter,
        "operational_threshold": OP_THRESHOLD,
        "auc": round(metrics["auc"], 6),
        "f1": round(metrics["f1"], 6),
        "brier": round(metrics["brier"], 6),
    }
    meta_path = mdir / "stage1_calibration_method.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    section_lines = [
        "",
        "--- STAGE 1 FINAL ---",
        "",
        f"Horizon                 : h={h}",
        f"Method                  : Raw LightGBM (no calibration wrapper)",
        f"Operational threshold   : {OP_THRESHOLD}",
        f"Training years          : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        f"Early-stop eval year    : {EARLY_STOP_YEAR}",
        f"Test year               : {HOLDOUT_YEAR}",
        f"Best iteration          : {best_iter:,}",
        f"Feature list            : {feat_path}",
        f"Model path              : {pkl_path}",
        "",
        f"{HOLDOUT_YEAR} Holdout metrics @ threshold = {OP_THRESHOLD}:",
        f"  AUC-ROC   : {metrics['auc']:.4f}",
        f"  F1        : {metrics['f1']:.4f}",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  Brier     : {metrics['brier']:.4f}",
        "",
        f"  Confusion matrix:",
        f"  {'':18}  Pred: Cloudy  Pred: Clear",
        f"  {'Actual: Cloudy':<18}  {tn:>12,}  {fp:>11,}",
        f"  {'Actual: Clear':<18}  {fn:>12,}  {tp:>11,}",
        "",
        f"  False clear-sky rate (FPR): {fpr:.4f}  ({fp} FP / {tn+fp} actual cloudy)",
        f"  False cloudy rate   (FNR): {fnr:.4f}  ({fn} FN / {tp+fn} actual clear)",
        "",
        SEP,
    ]
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(section_lines))
    print(f"Appended: {report_path}")

    print(f"\n{SEP}")
    print("Stage 1 finalization complete.")


if __name__ == "__main__":
    main()
