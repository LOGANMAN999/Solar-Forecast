import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR     = Path(__file__).resolve().parent.parent
RAW_PATH     = ROOT_DIR / "data" / "raw"       / "nsrdb_babcock_2018_2024.parquet"
FEAT_PATH    = ROOT_DIR / "data" / "processed" / "features.parquet"
OUT_PARQUET  = ROOT_DIR / "data" / "processed" / "nsrdb_with_pvlib.parquet"
OUT_PLOT     = ROOT_DIR / "notebooks"          / "clearsky_comparison.png"

LAT      =  26.78
LON      = -81.53
ALTITUDE =  12
TZ       = "UTC"

DAYTIME_THRESH = 50.0

MAX_BIAS_WM2  = 20.0
MAX_RMSE_WM2  = 50.0
MIN_PEARSON_R = 0.98

SEP = "=" * 65


def compute_pvlib_clearsky(df: pd.DataFrame) -> pd.DataFrame:
    print("  Building pvlib Location object …")
    location = pvlib.location.Location(
        latitude  = LAT,
        longitude = LON,
        altitude  = ALTITUDE,
        tz        = TZ,
    )

    times = df.index

    print(f"  Computing Ineichen clear-sky for {len(times):,} timestamps …")
    cs = location.get_clearsky(times, model="ineichen")

    df = df.copy()
    df["pvlib_clearsky_ghi"] = cs["ghi"].values
    print(f"  pvlib_clearsky_ghi range: "
          f"{df['pvlib_clearsky_ghi'].min():.1f} – "
          f"{df['pvlib_clearsky_ghi'].max():.1f} W/m²")
    return df


def print_summary(df: pd.DataFrame) -> dict:
    day  = df[df["clearsky_ghi"] >= DAYTIME_THRESH].copy()
    clr  = df[(df["clearsky_ghi"] >= DAYTIME_THRESH) & (df["cloud_type"] == 0)].copy()

    def stats(sub, label):
        diff  = sub["clearsky_ghi"] - sub["pvlib_clearsky_ghi"]
        bias  = diff.mean()
        rmse  = np.sqrt((diff ** 2).mean())
        r     = sub["clearsky_ghi"].corr(sub["pvlib_clearsky_ghi"])
        print(f"\n  [{label}]  n={len(sub):,}")
        print(f"    Mean bias (NSRDB − pvlib) : {bias:+.2f} W/m²")
        print(f"    RMSE                      : {rmse:.2f} W/m²")
        print(f"    Pearson r                 : {r:.5f}")
        return dict(bias=bias, rmse=rmse, r=r)

    print(f"\n{SEP}")
    print("Numerical Comparison — NSRDB clearsky_GHI vs pvlib Ineichen")
    print(SEP)
    d_stats = stats(day, f"All daytime (clearsky_ghi ≥ {DAYTIME_THRESH} W/m²)")
    c_stats = stats(clr, "Cloud-type = 0 (Clear) only")

    print(f"\n  {'─'*50}")
    print("  Acceptance thresholds (daytime all-sky):")
    checks = [
        ("Mean |bias| < 20 W/m²", abs(d_stats["bias"]) < MAX_BIAS_WM2),
        ("RMSE       < 50 W/m²", d_stats["rmse"]       < MAX_RMSE_WM2),
        ("Pearson r  > 0.98",     d_stats["r"]          > MIN_PEARSON_R),
    ]
    all_pass = True
    for label, passed in checks:
        status = "✓ PASS" if passed else "⚠ FAIL"
        print(f"    {label:<30} {status}")
        if not passed:
            all_pass = False

    if not all_pass:
        print("\n    One or more acceptance thresholds violated.")
        print("     Inspect the scatter plot (Panel 2) and residual histogram")
        print("     (Panel 4) for the primary source of divergence.")
    else:
        print("\n    All acceptance thresholds passed.")

    return d_stats


def find_clearest_week_2022(df: pd.DataFrame) -> pd.Timestamp:
    y2022 = df[(df.index.year == 2022) &
               (df["clearsky_ghi"] >= DAYTIME_THRESH)].copy()
    y2022["is_clear_sky"] = (y2022["cloud_type"] <= 1).astype(float)

    y2022["iso_week"] = y2022.index.isocalendar().week.values
    weekly = y2022.groupby("iso_week")["is_clear_sky"].mean()
    best_week = int(weekly.idxmax())

    jan_4_2022 = pd.Timestamp("2022-01-03", tz="UTC")
    week_start = jan_4_2022 + pd.Timedelta(weeks=best_week - 1)

    pct = weekly.max() * 100
    print(f"\n  Clearest 2022 week: ISO week {best_week}  "
          f"(starts {week_start.date()}, {pct:.1f}% clear daytime hours)")
    return week_start


def make_comparison_figure(df: pd.DataFrame, week_start: pd.Timestamp,
                           d_stats: dict) -> None:
    day = df[df["clearsky_ghi"] >= DAYTIME_THRESH].copy()

    C_GHI   = "#2196F3"
    C_NSRDB = "#FF9800"
    C_PVLIB = "#4CAF50"
    C_REF   = "#E53935"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "pvlib Ineichen vs NSRDB Clear-Sky GHI\n"
        "Babcock Ranch Solar Energy Center — NSRDB GOES Conus PSM v4  (2018–2024)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    ax1 = axes[0, 0]
    daily = df[["ghi", "clearsky_ghi", "pvlib_clearsky_ghi"]].resample("D").mean()
    ax1.plot(daily.index, daily["ghi"],             lw=0.7, color=C_GHI,
             alpha=0.7, label="GHI (measured)")
    ax1.plot(daily.index, daily["clearsky_ghi"],    lw=1.1, color=C_NSRDB,
             alpha=0.85, label="NSRDB clearsky_GHI")
    ax1.plot(daily.index, daily["pvlib_clearsky_ghi"], lw=1.1, color=C_PVLIB,
             alpha=0.85, label="pvlib Ineichen GHI", ls="--")
    ax1.set_title("Panel 1 — Daily Mean GHI: Full Time Series (2018–2024)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("W/m²")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    n_pts = len(day)
    if n_pts > 15_000:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(n_pts, size=15_000, replace=False)
        plot_day = day.iloc[idx]
    else:
        plot_day = day
    ax2.scatter(plot_day["pvlib_clearsky_ghi"], plot_day["clearsky_ghi"],
                s=3, alpha=0.25, color=C_NSRDB, rasterized=True,
                label=f"Hourly daytime (n={n_pts:,})")
    lim = max(day["pvlib_clearsky_ghi"].max(), day["clearsky_ghi"].max()) * 1.03
    ax2.plot([0, lim], [0, lim], color=C_REF, lw=1.5, ls="--", label="1:1 line")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_aspect("equal")
    metrics_txt = (f"RMSE  = {d_stats['rmse']:.1f} W/m²\n"
                   f"MAE   = {day['clearsky_ghi'].sub(day['pvlib_clearsky_ghi']).abs().mean():.1f} W/m²\n"
                   f"r     = {d_stats['r']:.4f}\n"
                   f"bias  = {d_stats['bias']:+.1f} W/m²")
    ax2.text(0.04, 0.96, metrics_txt, transform=ax2.transAxes,
             fontsize=9, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax2.set_title("Panel 2 — Scatter: NSRDB clearsky_GHI vs pvlib Ineichen\n"
                  f"(daytime: clearsky_GHI ≥ {DAYTIME_THRESH} W/m²)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("pvlib clearsky GHI  (W/m²)")
    ax2.set_ylabel("NSRDB clearsky GHI  (W/m²)")
    ax2.legend(fontsize=9, markerscale=3)
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    week_end  = week_start + pd.Timedelta(days=7)
    week_df   = df.loc[week_start:week_end]
    ax3.plot(week_df.index, week_df["ghi"],
             lw=1.4, color=C_GHI,   alpha=0.85, label="GHI (measured)")
    ax3.plot(week_df.index, week_df["clearsky_ghi"],
             lw=1.8, color=C_NSRDB, alpha=0.9,  label="NSRDB clearsky_GHI")
    ax3.plot(week_df.index, week_df["pvlib_clearsky_ghi"],
             lw=1.8, color=C_PVLIB, alpha=0.9,  label="pvlib Ineichen GHI",
             ls="--")
    ax3.set_title(
        f"Panel 3 — Clearest Week in 2022\n"
        f"({week_start.strftime('%b %d')} – {(week_end - pd.Timedelta(days=1)).strftime('%b %d, %Y')}, UTC)",
        fontsize=11, fontweight="bold",
    )
    ax3.set_xlabel("Date (UTC)")
    ax3.set_ylabel("W/m²")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%m/%d"))
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    residual = day["clearsky_ghi"] - day["pvlib_clearsky_ghi"]
    bias_val = residual.mean()
    std_val  = residual.std()
    max_abs = max(abs(residual.quantile(0.002)), abs(residual.quantile(0.998)))
    max_abs = min(max_abs * 1.1, 300)
    bins = np.linspace(-max_abs, max_abs, 80)
    ax4.hist(residual.clip(-max_abs, max_abs), bins=bins,
             color=C_NSRDB, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax4.axvline(0,        color=C_REF,   lw=1.5, ls="--", label="Zero")
    ax4.axvline(bias_val, color="#222",  lw=1.8, ls="-",
                label=f"Mean bias = {bias_val:+.1f} W/m²")
    ax4.axvline(bias_val + std_val, color="#666", lw=1.2, ls=":",
                label=f"±1σ  (σ = {std_val:.1f} W/m²)")
    ax4.axvline(bias_val - std_val, color="#666", lw=1.2, ls=":")
    ax4.set_title(
        "Panel 4 — Residual Distribution\n"
        "NSRDB clearsky_GHI − pvlib Ineichen GHI  (daytime)",
        fontsize=11, fontweight="bold",
    )
    ax4.set_xlabel("Residual  (W/m²)")
    ax4.set_ylabel("Hour count")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {OUT_PLOT}")


def main() -> None:
    print(SEP)
    print("pvlib Clear-Sky Comparison — Babcock Ranch Solar Energy Center")
    print(f"Site: lat={LAT}, lon={LON}, alt={ALTITUDE}m, tz={TZ}")
    print(f"Model: Ineichen with monthly Linke turbidity (auto-lookup)")
    print(SEP)

    print("\n[1] Loading NSRDB parquet …")
    df = pd.read_parquet(RAW_PATH)
    df.sort_index(inplace=True)
    print(f"    {df.shape}  ({df.index.min()} → {df.index.max()})")

    print("\n[2] Computing pvlib Ineichen clear-sky …")
    df = compute_pvlib_clearsky(df)

    print(f"\n[3] Saving → {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET, engine="pyarrow", compression="snappy", index=True)
    size_mb = OUT_PARQUET.stat().st_size / 1024**2
    print(f"    Saved  ({size_mb:.2f} MB, {len(df.columns)} columns)")
    print(f"    Columns: {list(df.columns)}")

    d_stats = print_summary(df)

    print(f"\n[4] Finding clearest week in 2022 …")
    week_start = find_clearest_week_2022(df)

    print(f"\n[5] Generating comparison figure …")
    make_comparison_figure(df, week_start, d_stats)

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()
