import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DAYTIME_THRESH = 50.0
HORIZONS = [1, 3, 6, 24]


def horizon_dir(root: Path, h: int) -> Path:
    d = root / "models" / f"h{h:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def horizon_outputs_dir(root: Path, h: int) -> Path:
    d = root / "outputs" / f"h{h:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def horizon_plots_dir(root: Path, h: int) -> Path:
    d = root / "notebooks" / f"h{h:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def features_path(root: Path, h: int) -> Path:
    return root / "data" / "processed" / f"features_h{h:03d}.parquet"


def _unique_shifts(base_shifts: list) -> list:
    seen = set()
    out = []
    for s in base_shifts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def make_horizon_features(raw: pd.DataFrame, h: int) -> pd.DataFrame:
    needed = [
        "ghi", "dni", "dhi",
        "clearsky_ghi", "clearsky_dni", "clearsky_dhi",
        "air_temperature", "wind_speed", "relative_humidity",
        "surface_albedo", "cloud_type", "solar_zenith_angle",
    ]
    existing = [c for c in needed if c in raw.columns]
    f = raw[existing].copy()

    cs = raw["clearsky_ghi"].copy()
    kt_vals = np.where(cs >= DAYTIME_THRESH, raw["ghi"] / cs, np.nan)
    kt = pd.Series(kt_vals, index=raw.index, name="kt")
    f["kt"] = kt
    f["is_clear_sky"] = (raw["cloud_type"] <= 1).astype(int)

    ghi = raw["ghi"]
    is_cs = f["is_clear_sky"]

    for s in _unique_shifts([h, h + 1, h + 2, h + 3, 24, 48, 168]):
        f[f"kt_lag_{s}"] = kt.shift(s)

    kt_past = kt.shift(h)
    f["kt_roll3_mean"] = kt_past.rolling(3, min_periods=1).mean()
    f["kt_roll3_std"] = kt_past.rolling(3, min_periods=1).std(ddof=0)
    f["kt_roll6_mean"] = kt_past.rolling(6, min_periods=1).mean()
    f["kt_roll6_std"] = kt_past.rolling(6, min_periods=1).std(ddof=0)
    f["kt_roll24_mean"] = kt_past.rolling(24, min_periods=1).mean()
    f["kt_roll24_std"] = kt_past.rolling(24, min_periods=1).std(ddof=0)

    f["kt_delta_1"] = kt.shift(h) - kt.shift(h + 1)
    f["kt_delta_3"] = kt.shift(h) - kt.shift(h + 3)
    f["ghi_delta_1"] = ghi.shift(h) - ghi.shift(h + 1)
    f["ghi_delta_3"] = ghi.shift(h) - ghi.shift(h + 3)

    for s in _unique_shifts([h, h + 1, h + 2, 24, 48]):
        f[f"is_clear_sky_lag_{s}"] = is_cs.shift(s)

    f[f"air_temperature_lag_{h}"] = raw["air_temperature"].shift(h)
    f[f"wind_speed_lag_{h}"] = raw["wind_speed"].shift(h)
    f[f"relative_humidity_lag_{h}"] = raw["relative_humidity"].shift(h)

    for s in _unique_shifts([h, 24, 48]):
        f[f"surface_albedo_lag_{s}"] = raw["surface_albedo"].shift(s)

    idx = f.index
    f["hour_of_day"] = idx.hour
    f["month"] = idx.month
    f["day_of_year"] = idx.dayofyear

    hour_rad = 2 * np.pi * idx.hour / 24
    f["hour_sin"] = np.sin(hour_rad)
    f["hour_cos"] = np.cos(hour_rad)

    doy_rad = 2 * np.pi * idx.dayofyear / 365.25
    f["doy_sin"] = np.sin(doy_rad)
    f["doy_cos"] = np.cos(doy_rad)

    month_rad = 2 * np.pi * (idx.month - 1) / 12
    f["month_sin"] = np.sin(month_rad)
    f["month_cos"] = np.cos(month_rad)

    sza_rad = raw["solar_zenith_angle"] * (np.pi / 180.0)
    f["sza_cos"] = np.cos(sza_rad)
    f["solar_zenith_angle"] = raw["solar_zenith_angle"]

    f["kt_sza_interaction"] = f[f"kt_lag_{h}"] * f["sza_cos"]

    return f


def s1_feature_cols(h: int) -> list:
    cols = []
    for s in _unique_shifts([h, h + 1, h + 2, h + 3, 24, 48, 168]):
        cols.append(f"kt_lag_{s}")
    cols += [
        "kt_roll3_mean", "kt_roll3_std",
        "kt_roll6_mean", "kt_roll6_std",
        "kt_roll24_mean", "kt_roll24_std",
        "kt_delta_1", "kt_delta_3",
    ]
    for s in _unique_shifts([h, h + 1, h + 2, 24, 48]):
        cols.append(f"is_clear_sky_lag_{s}")
    cols += [
        f"air_temperature_lag_{h}",
        f"wind_speed_lag_{h}",
        f"relative_humidity_lag_{h}",
        "hour_of_day", "month", "day_of_year",
        "hour_sin", "hour_cos",
        "doy_sin", "doy_cos",
        "month_sin", "month_cos",
        "solar_zenith_angle",
        "sza_cos",
    ]
    return cols


def s2_feature_cols(h: int) -> list:
    s1 = s1_feature_cols(h)
    extra = []
    for s in _unique_shifts([h, 24, 48]):
        extra.append(f"surface_albedo_lag_{s}")
    extra += ["kt_sza_interaction", "ghi_delta_1", "ghi_delta_3"]
    return s1 + extra


KT_CLEAR_VALUE = 1.0
KT_CLEAR_STRATEGY = "constant_1.0"


def predict_ghi_hard(rows, s1_model, s2_model, s1_cols, s2_cols, threshold):
    prob_clear = s1_model.predict_proba(rows[s1_cols])[:, 1]
    pred_clear = (prob_clear >= threshold).astype(int)
    n = len(rows)
    kt_pred = np.ones(n, dtype=float)
    cloudy_idx = np.where(pred_clear == 0)[0]
    if len(cloudy_idx) > 0:
        kt_pred[cloudy_idx] = np.clip(
            s2_model.predict(rows.iloc[cloudy_idx][s2_cols]), 0.0, 1.0
        )
    ghi_pred = kt_pred * rows["clearsky_ghi"].values
    return ghi_pred, prob_clear, pred_clear


def predict_ghi_soft(rows, s1_model, s2_model, s1_cols, s2_cols, routing_threshold):
    prob_clear = s1_model.predict_proba(rows[s1_cols])[:, 1]
    kt_cloudy = np.clip(s2_model.predict(rows[s2_cols]), 0.0, 1.0)
    kt_pred = prob_clear * KT_CLEAR_VALUE + (1.0 - prob_clear) * kt_cloudy
    ghi_pred = kt_pred * rows["clearsky_ghi"].values
    pred_clear = (prob_clear >= routing_threshold).astype(int)
    return ghi_pred, prob_clear, pred_clear


def predict_ghi_for_mode(rows, s1_model, s2_model, s1_cols, s2_cols, mode, threshold):
    if mode == "soft":
        return predict_ghi_soft(rows, s1_model, s2_model, s1_cols, s2_cols, threshold)
    return predict_ghi_hard(rows, s1_model, s2_model, s1_cols, s2_cols, threshold)
