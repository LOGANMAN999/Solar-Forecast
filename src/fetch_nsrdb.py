import io
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import requests

API_KEY  = "*********"
BASE_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"

LAT  =  26.7736
LON  = -81.5242
WKT  = f"POINT({LON} {LAT})"

ATTRIBUTES = (
    "ghi,dni,dhi,"
    "clearsky_ghi,clearsky_dni,clearsky_dhi,"
    "air_temperature,wind_speed,relative_humidity,"
    "surface_albedo,cloud_type,solar_zenith_angle"
)

YEARS = list(range(2018, 2025))

REQUEST_DELAY_S = 1.1
MAX_RETRIES     = 4
BACKOFF_BASE_S  = 2.0

LEAP_YEARS = {2020, 2024}

COLUMN_RENAME = {
    "Year"               : "year",
    "Month"              : "month",
    "Day"                : "day",
    "Hour"               : "hour",
    "Minute"             : "minute",
    "GHI"                : "ghi",
    "DNI"                : "dni",
    "DHI"                : "dhi",
    "Clearsky GHI"       : "clearsky_ghi",
    "Clearsky DNI"       : "clearsky_dni",
    "Clearsky DHI"       : "clearsky_dhi",
    "Temperature"        : "air_temperature",
    "Air Temperature"    : "air_temperature",
    "Wind Speed"         : "wind_speed",
    "Relative Humidity"  : "relative_humidity",
    "Surface Albedo"     : "surface_albedo",
    "Cloud Type"         : "cloud_type",
    "Solar Zenith Angle" : "solar_zenith_angle",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RAW_DIR / "nsrdb_babcock_2018_2024.parquet"


def _expected_rows(year: int) -> int:
    return 8784 if year in LEAP_YEARS else 8760


def _fetch_year(year: int) -> tuple[dict, pd.DataFrame]:
    params = dict(
        wkt        = WKT,
        attributes = ATTRIBUTES,
        names      = str(year),
        interval   = "60",
        utc        = "true",
        leap_day   = "true",
        email      = "logansingerman@gmail.com",
        api_key    = API_KEY,
    )

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=120)

            if resp.status_code == 429 or resp.status_code >= 500:
                wait = BACKOFF_BASE_S ** attempt
                print(f"    [{year}] HTTP {resp.status_code} on attempt {attempt}; "
                      f"retrying in {wait:.0f}s …")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP {resp.status_code} for year {year}.\n"
                    f"Body: {resp.text[:1000]}"
                )

            ct = resp.headers.get("Content-Type", "")
            if "json" in ct.lower():
                raise RuntimeError(
                    f"Unexpected JSON response for year {year} "
                    f"(async email workflow triggered?): {resp.text[:500]}"
                )
            if len(resp.content) < 1000:
                raise RuntimeError(
                    f"Suspiciously small response ({len(resp.content)} bytes) "
                    f"for year {year}: {resp.text[:500]}"
                )

            text  = resp.text
            lines = text.splitlines()

            meta_keys   = [k.strip() for k in lines[0].split(",")]
            meta_values = [v.strip() for v in lines[1].split(",")]
            metadata    = dict(zip(meta_keys, meta_values))
            metadata["year_requested"] = year

            df = pd.read_csv(io.StringIO(text), skiprows=2)

            df.rename(columns=COLUMN_RENAME, inplace=True)

            dt_cols = df[["year", "month", "day", "hour", "minute"]].rename(
                columns={"year": "year", "month": "month", "day": "day",
                         "hour": "hour", "minute": "minute"}
            )
            df["timestamp_utc"] = pd.to_datetime(dt_cols)
            df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
            df.set_index("timestamp_utc", inplace=True)

            df.drop(columns=["year", "month", "day", "hour", "minute"],
                    inplace=True)

            return metadata, df

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            wait = BACKOFF_BASE_S ** attempt
            print(f"    [{year}] Network error on attempt {attempt}: {exc!r}; "
                  f"retrying in {wait:.0f}s …")
            time.sleep(wait)
            continue

    raise RuntimeError(
        f"All {MAX_RETRIES} attempts failed for year {year}."
    ) from last_exc


def _validate_year(year: int, df: pd.DataFrame) -> None:
    expected = _expected_rows(year)
    actual   = len(df)
    leap_tag = " (leap)" if year in LEAP_YEARS else ""
    row_ok   = "✓" if actual == expected else f"⚠ expected {expected}"
    print(f"  rows: {actual:,} {row_ok}{leap_tag}")

    dupes = df.index.duplicated().sum()
    if dupes:
        print(f"  ⚠ {dupes} duplicate timestamps found!")
    else:
        print(f"  duplicates: none ✓")

    core = ["ghi", "clearsky_ghi", "solar_zenith_angle"]
    for col in core:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan:
                print(f"  ⚠ {col}: {n_nan} NaN values")

    if "ghi" in df.columns:
        neg = (df["ghi"] < 0).sum()
        if neg:
            print(f"  ⚠ ghi: {neg} negative values")


def main() -> None:
    print("=" * 65)
    print("NSRDB GOES Conus PSM v4 — Babcock Ranch Solar Energy Center")
    print(f"Coordinates : lat={LAT}, lon={LON}")
    print(f"WKT (API)   : {WKT}")
    print(f"Years       : {YEARS[0]}–{YEARS[-1]}")
    print(f"Output      : {OUTPUT_PATH}")
    print("=" * 65)

    all_dfs: list[pd.DataFrame] = []
    all_meta: list[dict]        = []

    for i, year in enumerate(YEARS):
        print(f"\n[{i+1}/{len(YEARS)}] Fetching year {year} …")
        try:
            metadata, df = _fetch_year(year)
            _validate_year(year, df)
            all_dfs.append(df)
            all_meta.append(metadata)
            print(f"  metadata: Location ID={metadata.get('Location ID','?')}, "
                  f"snapped to lat={metadata.get('Latitude','?')}, "
                  f"lon={metadata.get('Longitude','?')}")
        except RuntimeError as exc:
            print(f"  FATAL: {exc}", file=sys.stderr)
            sys.exit(1)

        if i < len(YEARS) - 1:
            time.sleep(REQUEST_DELAY_S)

    print(f"\nConcatenating {len(all_dfs)} years …")
    combined = pd.concat(all_dfs, axis=0)
    combined.sort_index(inplace=True)

    total_expected = sum(_expected_rows(y) for y in YEARS)
    print(f"Combined shape : {combined.shape}  (expected {total_expected:,} rows)")

    full_range = pd.date_range(
        start = combined.index.min(),
        end   = combined.index.max(),
        freq  = "60min",
    )
    idx_diff   = combined.index.to_series().diff().dropna()
    unexpected = idx_diff[idx_diff != pd.Timedelta("60min")]
    if unexpected.empty:
        print("Timestamp gaps : none ✓")
    else:
        print(f"⚠ {len(unexpected)} unexpected timestamp gaps found:")
        print(unexpected.head(10))

    site_meta = {k: v for k, v in all_meta[0].items()
                 if k != "year_requested"}
    print(f"\nSite metadata  : {site_meta}")

    print(f"\nSaving to {OUTPUT_PATH} …")
    combined.to_parquet(
        OUTPUT_PATH,
        engine       = "pyarrow",
        compression  = "snappy",
        index        = True,
    )

    meta_path = RAW_DIR / "nsrdb_babcock_site_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(site_meta, f, indent=2)
    print(f"Site metadata  → {meta_path}")

    size_mb = OUTPUT_PATH.stat().st_size / 1024**2
    print(f"\nDone!  Parquet size: {size_mb:.2f} MB")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
