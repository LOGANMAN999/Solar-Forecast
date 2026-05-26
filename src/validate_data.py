import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR    = Path(__file__).resolve().parent.parent
RAW_PATH    = ROOT_DIR / "data" / "raw"  / "nsrdb_babcock_2018_2024.parquet"
PLOT_DIR    = ROOT_DIR / "notebooks"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_OUT = PLOT_DIR / "ghi_calendar_heatmap.png"

GHI_CLEARSKY_FACTOR = 1.2


def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw parquet not found: {RAW_PATH}\n"
            "Run  python src/fetch_nsrdb.py  first."
        )
    df = pd.read_parquet(RAW_PATH)
    print(f"Loaded {RAW_PATH.name}: shape={df.shape}, "
          f"index={df.index.min()} → {df.index.max()}")
    return df


def check_missing_timestamps(df: pd.DataFrame) -> None:
    print("\n── Check 1: Missing timestamps ──────────────────────────────────")

    expected_idx = pd.date_range(
        start = df.index.min(),
        end   = df.index.max(),
        freq  = "60min",
    )
    missing_ts = expected_idx.difference(df.index)

    if missing_ts.empty:
        print("  No missing timestamps ✓")
    else:
        print(f"  ⚠ {len(missing_ts)} missing timestamps:")
        for ts in missing_ts[:20]:
            print(f"    {ts}")
        if len(missing_ts) > 20:
            print(f"    … and {len(missing_ts) - 20} more.")

    dupes = df.index[df.index.duplicated()].unique()
    if dupes.empty:
        print("  No duplicate timestamps ✓")
    else:
        print(f"  ⚠ {len(dupes)} duplicate timestamps:")
        print(f"    {dupes[:10].tolist()}")

    years      = sorted(df.index.year.unique())
    leap_years = {y for y in years if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))}
    expected_n = sum(8784 if y in leap_years else 8760 for y in years)
    actual_n   = len(df)
    tag        = "✓" if actual_n == expected_n else f"⚠ expected {expected_n:,}"
    print(f"  Row count: {actual_n:,}  {tag}  (years {years[0]}–{years[-1]})")


def check_missing_values(df: pd.DataFrame) -> None:
    print("\n── Check 2: Missing values per column ───────────────────────────")
    n = len(df)
    missing = df.isna().sum()
    max_col_len = max(len(c) for c in df.columns)
    any_missing = False
    for col, cnt in missing.items():
        pct = cnt / n * 100
        flag = "  ✓" if cnt == 0 else f"  ⚠"
        if cnt:
            any_missing = True
        print(f"  {col:<{max_col_len}}  {cnt:6,}  ({pct:5.2f}%){flag}")
    if not any_missing:
        print("  All columns complete — no NaNs ✓")


def check_physics(df: pd.DataFrame) -> None:
    print("\n── Check 3: Physical sanity checks ──────────────────────────────")
    results = []

    if "ghi" in df.columns:
        neg = (df["ghi"] < 0).sum()
        results.append(("GHI < 0",
                        f"{neg:,} rows",
                        "✓" if neg == 0 else "⚠"))

    if "ghi" in df.columns and "clearsky_ghi" in df.columns:
        day = df["clearsky_ghi"] > 0
        exceeded = (df.loc[day, "ghi"] > df.loc[day, "clearsky_ghi"] * GHI_CLEARSKY_FACTOR).sum()
        pct_day  = exceeded / day.sum() * 100 if day.sum() > 0 else 0
        results.append((f"GHI > clearsky_GHI × {GHI_CLEARSKY_FACTOR}",
                        f"{exceeded:,} daytime rows ({pct_day:.2f}%)",
                        "✓" if exceeded == 0 else "ℹ (cloud enhancement)"))

    if "solar_zenith_angle" in df.columns:
        oob = ((df["solar_zenith_angle"] < 0) | (df["solar_zenith_angle"] > 180)).sum()
        results.append(("SZA ∉ [0, 180]",
                        f"{oob:,} rows",
                        "✓" if oob == 0 else "⚠"))

    if "air_temperature" in df.columns:
        oob_t = ((df["air_temperature"] < -5) | (df["air_temperature"] > 45)).sum()
        results.append(("Temperature ∉ [−5, 45 °C]",
                        f"{oob_t:,} rows",
                        "✓" if oob_t == 0 else "⚠"))

    if "relative_humidity" in df.columns:
        oob_rh = ((df["relative_humidity"] < 0) | (df["relative_humidity"] > 100)).sum()
        results.append(("RH ∉ [0, 100%]",
                        f"{oob_rh:,} rows",
                        "✓" if oob_rh == 0 else "⚠"))

    max_check_len = max(len(r[0]) for r in results)
    for check, detail, flag in results:
        print(f"  {check:<{max_check_len}}  {detail:<30}  {flag}")


def plot_ghi_heatmap(df: pd.DataFrame) -> None:
    print("\n── Check 4: Calendar heatmap (daily mean GHI) ───────────────────")

    if "ghi" not in df.columns:
        print("  ghi column not found — skipping plot.")
        return

    daily_ghi = df["ghi"].resample("D").mean()
    daily_df  = daily_ghi.reset_index()
    daily_df.columns = ["date", "ghi_mean"]
    daily_df["year"]  = daily_df["date"].dt.year
    daily_df["doy"]   = daily_df["date"].dt.dayofyear
    daily_df["month"] = daily_df["date"].dt.month

    years   = sorted(daily_df["year"].unique())
    n_years = len(years)
    max_doy = 366

    mat = np.full((n_years, max_doy), np.nan)
    for row_idx, yr in enumerate(years):
        sub = daily_df[daily_df["year"] == yr]
        for _, record in sub.iterrows():
            doy_idx = int(record["doy"]) - 1
            mat[row_idx, doy_idx] = record["ghi_mean"]

    fig, ax = plt.subplots(figsize=(18, n_years * 0.9 + 1.5))
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="#e0e0e0")

    im = ax.imshow(
        mat,
        aspect    = "auto",
        cmap      = cmap,
        vmin      = 0,
        vmax      = daily_df["ghi_mean"].quantile(0.99),
        interpolation = "nearest",
    )

    ax.set_yticks(range(n_years))
    ax.set_yticklabels(years, fontsize=11)

    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_xlim(0, max_doy - 1)

    cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("Daily mean GHI  (W/m²)", fontsize=11)

    ax.set_title(
        "Babcock Ranch Solar Energy Center — Daily Mean GHI\n"
        "NSRDB GOES Conus PSM v4  (2018–2024, UTC)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Year", fontsize=11)

    plt.tight_layout()
    plt.savefig(HEATMAP_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {HEATMAP_OUT}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n── Summary statistics (numeric columns) ─────────────────────────")
    print(df.describe().round(2).to_string())


def main() -> None:
    print("=" * 65)
    print("NSRDB Data Validation — Babcock Ranch Solar Energy Center")
    print("=" * 65)

    df = load_data()

    check_missing_timestamps(df)
    check_missing_values(df)
    check_physics(df)
    plot_ghi_heatmap(df)
    print_summary(df)

    print("\n" + "=" * 65)
    print("Validation complete.")


if __name__ == "__main__":
    main()
