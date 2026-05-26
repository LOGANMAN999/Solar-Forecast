import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "nsrdb_babcock_2018_2024.parquet"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEARSKY_THRESHOLD = 50.0
MET_COLS = [
    "air_temperature", "wind_speed", "relative_humidity",
    "surface_albedo", "cloud_type", "solar_zenith_angle",
]

SEP = "=" * 65


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw parquet not found: {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    df.sort_index(inplace=True)
    print(f"Loaded raw data: {df.shape}  ({df.index.min()} -> {df.index.max()})")
    return df


def report_clear_sky_balance(feat: pd.DataFrame) -> None:
    MONTH_NAMES = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    day = feat[feat["clearsky_ghi"] >= CLEARSKY_THRESHOLD].dropna(subset=["kt"])
    print(f"\n{SEP}")
    print("is_clear_sky Monthly Class Balance  (daytime: clearsky_ghi >= 50 W/m2)")
    print(SEP)
    print(f"\n  {'Month':<6}  {'n_day':>7}  {'n_clear':>8}  {'n_cloudy':>9}  "
          f"{'%_clear':>8}  {'%_cloudy':>9}  bar")
    print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*30}")
    bar_width = 30
    for m in range(1, 13):
        sub = day[day["month"] == m]
        n_day = len(sub)
        n_clear = int(sub["is_clear_sky"].sum())
        n_cloud = n_day - n_clear
        pct_c = n_clear / n_day * 100 if n_day else 0
        pct_cl = n_cloud / n_day * 100 if n_day else 0
        n_bar_clear = round(pct_c / 100 * bar_width)
        bar = chr(9619) * n_bar_clear + chr(9617) * (bar_width - n_bar_clear)
        print(f"  {MONTH_NAMES[m]:<6}  {n_day:>7,}  {n_clear:>8,}  {n_cloud:>9,}  "
              f"{pct_c:>7.1f}%  {pct_cl:>8.1f}%  {bar}")
    n_day_tot = len(day)
    n_clear_tot = int(day["is_clear_sky"].sum())
    print(f"  {'TOTAL':<6}  {n_day_tot:>7,}  {n_clear_tot:>8,}  "
          f"{n_day_tot - n_clear_tot:>9,}  "
          f"{n_clear_tot/n_day_tot*100:>7.1f}%  "
          f"{(n_day_tot - n_clear_tot)/n_day_tot*100:>8.1f}%")


def main_horizon(raw: pd.DataFrame, h: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from horizon_utils import make_horizon_features, features_path
    feat = make_horizon_features(raw, h)
    out_path = features_path(ROOT_DIR, h)
    print(f"\nHorizon h={h} feature matrix shape: {feat.shape}")
    feat.to_parquet(out_path, engine="pyarrow", compression="snappy", index=True)
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"Saved: {out_path}  ({size_mb:.2f} MB)")
    print(f"\nColumns ({len(feat.columns)}):")
    for col in feat.columns:
        n_nan = feat[col].isna().sum()
        tag = f"  ({n_nan:,} NaN)" if n_nan else ""
        print(f"  {col:<40}  {str(feat[col].dtype):<10}{tag}")
    report_clear_sky_balance(feat)


def main_legacy(raw: pd.DataFrame) -> None:
    keep = ["ghi", "dni", "dhi", "clearsky_ghi", "clearsky_dni", "clearsky_dhi"]
    keep += [c for c in MET_COLS if c in raw.columns]
    feat = raw[keep].copy()

    cs = raw["clearsky_ghi"].copy()
    kt_vals = np.where(cs >= CLEARSKY_THRESHOLD, raw["ghi"] / cs, np.nan)
    kt = pd.Series(kt_vals, index=raw.index, name="kt")
    feat["kt"] = kt
    feat["is_clear_sky"] = (raw["cloud_type"] <= 1).astype(int)

    for lag in [1, 2, 3, 24, 48]:
        feat[f"kt_lag{lag}"] = kt.shift(lag)

    kt_shifted = kt.shift(1)
    for w in [3, 6]:
        feat[f"kt_roll{w}_mean"] = kt_shifted.rolling(window=w, min_periods=1).mean()
        feat[f"kt_roll{w}_std"] = kt_shifted.rolling(window=w, min_periods=1).std(ddof=0)

    idx = feat.index
    feat["hour_of_day"] = idx.hour
    feat["month"] = idx.month
    feat["day_of_year"] = idx.dayofyear
    feat["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    out_path = PROCESSED_DIR / "features.parquet"
    print(f"\nLegacy feature matrix shape: {feat.shape}")
    feat.to_parquet(out_path, engine="pyarrow", compression="snappy", index=True)
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"Saved: {out_path}  ({size_mb:.2f} MB)")
    print("\nkt descriptive stats (daytime only):")
    print(feat["kt"].dropna().describe().round(4).to_string())
    report_clear_sky_balance(feat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Forecast horizon in hours (1, 3, 6, 24). "
             "If omitted, generates legacy features.parquet.",
    )
    args = parser.parse_args()

    print(SEP)
    if args.horizon is not None:
        print(f"Feature Engineering -- horizon h={args.horizon}")
    else:
        print("Feature Engineering -- legacy mode (features.parquet)")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)

    raw = load_raw()

    if args.horizon is not None:
        main_horizon(raw, args.horizon)
    else:
        main_legacy(raw)

    print(f"\n{SEP}")
    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
