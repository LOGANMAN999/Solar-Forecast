import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from horizon_utils import (
    features_path, horizon_dir, s1_feature_cols, s2_feature_cols,
    DAYTIME_THRESH,
)

SEP = "=" * 68
FORBIDDEN_FEATURES = {"kt", "ghi", "cloud_type", "is_clear_sky", "surface_albedo"}


def check_forbidden_in_list(feature_list: list, label: str) -> list:
    violations = []
    for col in feature_list:
        if col in FORBIDDEN_FEATURES:
            violations.append(f"LEAK [{label}]: forbidden raw column '{col}' in feature list")
    return violations


def check_lag_safety(feature_list: list, h: int, label: str) -> list:
    violations = []
    unsafe_patterns = [
        (r"^kt_lag_(\d+)$", "kt_lag"),
        (r"^is_clear_sky_lag_(\d+)$", "is_clear_sky_lag"),
        (r"^air_temperature_lag_(\d+)$", "air_temperature_lag"),
        (r"^wind_speed_lag_(\d+)$", "wind_speed_lag"),
        (r"^relative_humidity_lag_(\d+)$", "relative_humidity_lag"),
        (r"^surface_albedo_lag_(\d+)$", "surface_albedo_lag"),
    ]
    for col in feature_list:
        for pattern, kind in unsafe_patterns:
            m = re.match(pattern, col)
            if m:
                lag = int(m.group(1))
                if lag < h:
                    violations.append(
                        f"LEAK [{label}]: '{col}' has lag {lag} < horizon {h}"
                    )
    return violations


def check_rolling_not_current_inclusive(
    feat: pd.DataFrame, h: int, n_check: int = 500
) -> list:
    violations = []
    if "kt_roll3_mean" not in feat.columns or "kt" not in feat.columns:
        return violations

    kt = feat["kt"]
    roll_col = feat["kt_roll3_mean"]
    freq = pd.Timedelta(hours=1)

    valid_idx = roll_col.dropna().index
    if len(valid_idx) < n_check:
        check_idx = valid_idx
    else:
        rng = np.random.default_rng(42)
        check_idx = valid_idx[rng.choice(len(valid_idx), size=n_check, replace=False)]

    expected_vals = []
    actual_vals = []
    curr_kt_vals = []
    kt_h_vals = []
    kt_h1_vals = []

    for ts in check_idx:
        ts_h = ts - h * freq
        ts_h1 = ts - (h + 1) * freq
        ts_h2 = ts - (h + 2) * freq
        v_h = kt.get(ts_h, np.nan) if ts_h in kt.index else np.nan
        v_h1 = kt.get(ts_h1, np.nan) if ts_h1 in kt.index else np.nan
        v_h2 = kt.get(ts_h2, np.nan) if ts_h2 in kt.index else np.nan
        v_cur = kt.get(ts, np.nan) if ts in kt.index else np.nan
        actual = float(roll_col.loc[ts])
        if np.isfinite(v_h) and np.isfinite(v_h1) and np.isfinite(v_h2):
            expected_vals.append((v_h + v_h1 + v_h2) / 3.0)
            actual_vals.append(actual)
            curr_kt_vals.append(v_cur)
            kt_h_vals.append(v_h)
            kt_h1_vals.append(v_h1)

    if len(expected_vals) < 10:
        return violations

    expected_arr = np.array(expected_vals)
    actual_arr = np.array(actual_vals)
    valid = np.isfinite(expected_arr) & np.isfinite(actual_arr)
    if valid.sum() < 10:
        return violations

    diff = np.abs(actual_arr[valid] - expected_arr[valid])
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    if max_diff > 0.02:
        violations.append(
            f"ROLLING CHECK: kt_roll3_mean deviates from expected (h={h} shift) "
            f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
            f"(>0.02 may indicate current-inclusive rolling)"
        )

    curr_arr = np.array(curr_kt_vals)
    kt_h_arr = np.array(kt_h_vals)
    kt_h1_arr = np.array(kt_h1_vals)
    valid2 = np.isfinite(curr_arr) & np.isfinite(actual_arr)
    if valid2.sum() >= 10:
        ci_expected = (curr_arr[valid2] + kt_h_arr[valid2] + kt_h1_arr[valid2]) / 3.0
        mean_diff_ci = float(np.abs(actual_arr[valid2] - ci_expected).mean())
        if mean_diff_ci < 1e-6:
            violations.append(
                f"LEAK: kt_roll3_mean appears current-inclusive "
                f"(matches (kt[t]+kt[t-h]+kt[t-h-1])/3 with mean_diff={mean_diff_ci:.2e})"
            )

    return violations


def check_no_unshifted_surface_albedo(feature_list: list, label: str) -> list:
    violations = []
    if "surface_albedo" in feature_list:
        violations.append(
            f"LEAK [{label}]: unshifted 'surface_albedo' in feature list"
        )
    return violations


def check_feature_list_matches_parquet(
    feature_list: list, feat: pd.DataFrame, label: str
) -> list:
    violations = []
    missing = [c for c in feature_list if c not in feat.columns]
    if missing:
        violations.append(
            f"MISMATCH [{label}]: {len(missing)} feature cols missing from parquet: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return violations


def check_horizon_dir_artifacts(h: int) -> list:
    violations = []
    mdir = horizon_dir(ROOT_DIR, h)
    for fname in [
        "stage1_classifier.pkl",
        "stage1_feature_list.json",
        "stage2_regressor.pkl",
        "stage2_feature_list.json",
    ]:
        p = mdir / fname
        if not p.exists():
            violations.append(f"MISSING: {p}")
    return violations


def validate_saved_feature_lists(h: int, feat: pd.DataFrame) -> list:
    violations = []
    mdir = horizon_dir(ROOT_DIR, h)
    for fname, label in [
        ("stage1_feature_list.json", "stage1"),
        ("stage2_feature_list.json", "stage2"),
    ]:
        p = mdir / fname
        if not p.exists():
            violations.append(f"MISSING: {p}")
            continue
        with open(p) as f:
            saved_list = json.load(f)
        violations += check_forbidden_in_list(saved_list, label)
        violations += check_lag_safety(saved_list, h, label)
        violations += check_no_unshifted_surface_albedo(saved_list, label)
        violations += check_feature_list_matches_parquet(saved_list, feat, label)
    return violations


def validate_expected_cols(h: int, feat: pd.DataFrame) -> list:
    violations = []
    s1_cols = s1_feature_cols(h)
    s2_cols = s2_feature_cols(h)
    violations += check_forbidden_in_list(s1_cols, "s1_expected")
    violations += check_forbidden_in_list(s2_cols, "s2_expected")
    violations += check_lag_safety(s1_cols, h, "s1_expected")
    violations += check_lag_safety(s2_cols, h, "s2_expected")
    violations += check_no_unshifted_surface_albedo(s1_cols, "s1_expected")
    violations += check_no_unshifted_surface_albedo(s2_cols, "s2_expected")
    violations += check_feature_list_matches_parquet(s1_cols, feat, "s1_expected")
    violations += check_feature_list_matches_parquet(s2_cols, feat, "s2_expected")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True,
                        help="Forecast horizon in hours (e.g. 1, 3, 6, 24).")
    args = parser.parse_args()
    h = args.horizon

    print(SEP)
    print(f"Leakage Validation -- horizon h={h}")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)

    fp = features_path(ROOT_DIR, h)
    if not fp.exists():
        print(f"\nERROR: Features file not found: {fp}")
        print(f"Run  python src/features.py --horizon {h}  first.")
        sys.exit(1)

    print(f"\n[1] Loading {fp.name} ...")
    feat = pd.read_parquet(fp)
    feat.sort_index(inplace=True)
    print(f"    Shape: {feat.shape}")

    all_violations = []

    print(f"\n[2] Checking expected feature lists for h={h} ...")
    v = validate_expected_cols(h, feat)
    all_violations += v
    status = "PASS" if not v else f"FAIL ({len(v)} issues)"
    print(f"    Expected cols check: {status}")

    print(f"\n[3] Checking rolling feature leakage (numerical) ...")
    v = check_rolling_not_current_inclusive(feat, h)
    all_violations += v
    status = "PASS" if not v else f"FAIL ({len(v)} issues)"
    print(f"    Rolling check: {status}")

    print(f"\n[4] Checking saved model artifact feature lists ...")
    mdir = horizon_dir(ROOT_DIR, h)
    artifact_missing = [
        f for f in ["stage1_feature_list.json", "stage2_feature_list.json"]
        if not (mdir / f).exists()
    ]
    if artifact_missing:
        print(f"    SKIP: Artifacts not yet trained for h={h}")
        print(f"      Missing: {artifact_missing}")
    else:
        v = validate_saved_feature_lists(h, feat)
        all_violations += v
        status = "PASS" if not v else f"FAIL ({len(v)} issues)"
        print(f"    Saved feature list check: {status}")

    print(f"\n[5] Feature column summary for h={h}:")
    s1_cols = s1_feature_cols(h)
    s2_cols = s2_feature_cols(h)
    s2_extra = [c for c in s2_cols if c not in s1_cols]
    print(f"    S1 feature count : {len(s1_cols)}")
    print(f"    S2 feature count : {len(s2_cols)}")
    print(f"    S2 extra vs S1   : {s2_extra}")
    print(f"\n    S1 features:")
    for c in s1_cols:
        in_feat = c in feat.columns
        print(f"      {'OK' if in_feat else 'MISSING':<8} {c}")
    print(f"\n    S2-only features:")
    for c in s2_extra:
        in_feat = c in feat.columns
        print(f"      {'OK' if in_feat else 'MISSING':<8} {c}")

    print(f"\n{SEP}")
    if all_violations:
        print(f"RESULT: {len(all_violations)} VIOLATION(S) FOUND")
        print(SEP)
        for v in all_violations:
            print(f"  {v}")
    else:
        print("RESULT: ALL CHECKS PASSED")
    print(SEP)

    if all_violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
