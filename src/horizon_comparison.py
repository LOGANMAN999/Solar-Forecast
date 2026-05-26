import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
HORIZONS = [1, 3, 6, 24]

OUT_DIR = ROOT_DIR / "outputs"
PLOT_DIR = ROOT_DIR / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 68


def _outputs(h: int) -> Path:
    return ROOT_DIR / "outputs" / f"h{h:03d}"


def _models(h: int) -> Path:
    return ROOT_DIR / "models" / f"h{h:03d}"


def parse_mode_selection(h: int) -> dict:
    p = _models(h) / "prediction_mode_selection.json"
    if not p.exists():
        return {}
    with open(p) as f:
        d = json.load(f)
    return {
        "selected_prediction_mode": d.get("selected_prediction_mode", ""),
        "selected_threshold": d.get("selected_threshold", float("nan")),
        "clear_kt_strategy": d.get("clear_kt_strategy", ""),
    }


def parse_stage1_report(h: int) -> dict:
    path = _outputs(h) / "stage1_calibration_report.txt"
    d = {}
    if not path.exists():
        return d
    text = path.read_text(encoding="utf-8")
    m = re.search(r"Training years\s+:\s+([\d\-]+)", text)
    if m:
        d["train_years"] = m.group(1)
    m = re.search(r"Early-stop eval year\s+:\s+(\d+)", text)
    if m:
        d["early_stop_year"] = int(m.group(1))
    m = re.search(r"Test year\s+:\s+(\d+)", text)
    if m:
        d["test_year"] = int(m.group(1))
    m = re.search(r"AUC-ROC\s+:\s+([\d.]+)", text)
    if m:
        d["stage1_auc"] = float(m.group(1))
    m = re.search(r"\bF1\s+:\s+([\d.]+)", text)
    if m:
        d["stage1_f1"] = float(m.group(1))
    m = re.search(r"Precision\s+:\s+([\d.]+)", text)
    if m:
        d["stage1_precision"] = float(m.group(1))
    m = re.search(r"Recall\s+:\s+([\d.]+)", text)
    if m:
        d["stage1_recall"] = float(m.group(1))
    m = re.search(r"Brier\s+:\s+([\d.]+)", text)
    if m:
        d["stage1_brier"] = float(m.group(1))
    m = re.search(r"False clear-sky rate \(FPR\):\s+([\d.]+)", text)
    if m:
        d["stage1_false_clear_rate"] = float(m.group(1))
    m = re.search(r"False cloudy rate\s+\(FNR\):\s+([\d.]+)", text)
    if m:
        d["stage1_false_cloudy_rate"] = float(m.group(1))
    return d


def parse_stage2_report(h: int) -> dict:
    path = _outputs(h) / "stage2_full_report.txt"
    d = {}
    if not path.exists():
        return d
    text = path.read_text(encoding="utf-8")

    m = re.search(r"Stage 2 model\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)", text)
    if m:
        d["stage2_cloudy_kt_rmse"] = float(m.group(1))
        d["stage2_cloudy_kt_mae"] = float(m.group(2))
        d["stage2_cloudy_kt_r2"] = float(m.group(3))

    m = re.search(r"Persistence\s*(?:\([^)]+\))?\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)", text)
    if m:
        d["persistence_kt_rmse"] = float(m.group(1))

    m = re.search(r"Climatological mean\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)", text)
    if m:
        d["climatology_kt_rmse"] = float(m.group(1))

    m = re.search(
        r"All daytime\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)",
        text,
    )
    if m:
        d["all_daytime_rmse_model"] = float(m.group(1))
        d["all_daytime_mae_model"] = float(m.group(2))
        d["all_daytime_mbe_model"] = float(m.group(3))
        d["all_daytime_rmse_yesterday"] = float(m.group(4))
        d["all_daytime_skill_score"] = float(m.group(5))

    m = re.search(
        r"Clear-sky\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)",
        text,
    )
    if m:
        d["clear_sky_rmse_model"] = float(m.group(1))
        d["clear_sky_rmse_yesterday"] = float(m.group(4))
        d["clear_sky_skill_score"] = float(m.group(5))

    m = re.search(
        r"Cloudy\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)",
        text,
    )
    if m:
        d["cloudy_rmse_model"] = float(m.group(1))
        d["cloudy_rmse_yesterday"] = float(m.group(4))
        d["cloudy_skill_score"] = float(m.group(5))

    m = re.search(
        r"Ramp events\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)",
        text,
    )
    if m:
        d["ramp_rmse_model"] = float(m.group(1))
        d["ramp_rmse_yesterday"] = float(m.group(4))
        d["ramp_skill_score"] = float(m.group(5))

    mode_rows = {}
    for mode in ["hard_fixed", "hard_tuned", "soft"]:
        pat = re.compile(
            rf"{mode}\s+([\d.NA/]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)"
        )
        mp = pat.search(text)
        if mp:
            mode_rows[mode] = {
                "all_rmse": float(mp.group(2)),
                "clear_rmse": float(mp.group(3)),
                "cloudy_rmse": float(mp.group(4)),
                "ramp_rmse": float(mp.group(5)),
                "skill": float(mp.group(6)),
            }
    d["mode_comparison"] = mode_rows

    return d


def parse_stage3_report(h: int) -> dict:
    path = _outputs(h) / "stage3_conformal_report.txt"
    d = {}
    if not path.exists():
        return d
    text = path.read_text(encoding="utf-8")

    m = re.search(r"conformal year\s+:\s+(\d+)", text)
    if m:
        d["conformal_year"] = int(m.group(1))

    m = re.search(
        r"Marginal coverage\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text
    )
    if m:
        d["stratified_marginal_coverage"] = float(m.group(1))
        d["single_window_marginal_coverage"] = float(m.group(2))
        d["static_marginal_coverage"] = float(m.group(3))
        d["gaussian_marginal_coverage"] = float(m.group(4))

    m = re.search(
        r"Winkler score\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text
    )
    if m:
        d["stratified_winkler_score"] = float(m.group(1))
        d["single_window_winkler_score"] = float(m.group(2))
        d["static_winkler_score"] = float(m.group(3))
        d["gaussian_winkler_score"] = float(m.group(4))

    m = re.search(
        r"Mean interval width \(W/m2\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        text,
    )
    if m:
        d["stratified_mean_width"] = float(m.group(1))
        d["single_window_mean_width"] = float(m.group(2))
        d["static_mean_width"] = float(m.group(3))
        d["gaussian_mean_width"] = float(m.group(4))

    m = re.search(r"Cloudy coverage \(true label\)\s+([\d.]+)", text)
    if m:
        d["stratified_cloudy_coverage"] = float(m.group(1))

    m = re.search(r"Ramp coverage\s+([\d.]+)", text)
    if m:
        d["stratified_ramp_coverage"] = float(m.group(1))

    m = re.search(r"Monthly std\(coverage\)\s+([\d.]+)", text)
    if m:
        d["monthly_coverage_std"] = float(m.group(1))

    m = re.search(r"Total mismatch hours\s+:\s+[\d,]+\s+\(([\d.]+)%", text)
    if m:
        d["mismatch_rate"] = float(m.group(1)) / 100.0

    m = re.search(r"Mismatch hours\s+([\d.]+)\s+[\d.]+\s+[\d,]+", text)
    if m:
        d["mismatch_coverage"] = float(m.group(1))

    return d


def build_row(h: int) -> dict:
    row = {"horizon": h}
    row.update(parse_mode_selection(h))
    row.update(parse_stage1_report(h))
    s2 = parse_stage2_report(h)
    mode_comp = s2.pop("mode_comparison", {})
    row.update(s2)
    row["mode_comparison"] = mode_comp
    row.update(parse_stage3_report(h))
    return row


COLUMNS = [
    "horizon",
    "selected_prediction_mode",
    "selected_threshold",
    "clear_kt_strategy",
    "train_years",
    "early_stop_year",
    "conformal_year",
    "test_year",
    "stage1_auc",
    "stage1_f1",
    "stage1_precision",
    "stage1_recall",
    "stage1_brier",
    "stage1_false_clear_rate",
    "stage1_false_cloudy_rate",
    "stage2_cloudy_kt_rmse",
    "stage2_cloudy_kt_mae",
    "stage2_cloudy_kt_r2",
    "persistence_kt_rmse",
    "climatology_kt_rmse",
    "all_daytime_rmse_model",
    "all_daytime_mae_model",
    "all_daytime_mbe_model",
    "all_daytime_rmse_yesterday",
    "all_daytime_skill_score",
    "clear_sky_rmse_model",
    "clear_sky_rmse_yesterday",
    "clear_sky_skill_score",
    "cloudy_rmse_model",
    "cloudy_rmse_yesterday",
    "cloudy_skill_score",
    "ramp_rmse_model",
    "ramp_rmse_yesterday",
    "ramp_skill_score",
    "stratified_marginal_coverage",
    "stratified_mean_width",
    "stratified_winkler_score",
    "single_window_marginal_coverage",
    "single_window_mean_width",
    "static_marginal_coverage",
    "static_mean_width",
    "mismatch_rate",
    "mismatch_coverage",
    "monthly_coverage_std",
    "stratified_cloudy_coverage",
    "stratified_ramp_coverage",
]


def build_dataframe() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        rows.append(build_row(h))
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[COLUMNS]


def _fmt(v, decimals=4):
    try:
        f = float(v)
        if np.isnan(f):
            return ""
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else ""


def _fmts(v, decimals=4):
    try:
        f = float(v)
        if np.isnan(f):
            return ""
        return f"{f:+.{decimals}f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else ""


def _fmt_int(v):
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return ""


def save_csv(df: pd.DataFrame) -> Path:
    p = OUT_DIR / "horizon_comparison.csv"
    df.to_csv(p, index=False, float_format="%.4f")
    return p


def save_txt(df: pd.DataFrame, rows_raw: list) -> Path:
    p = OUT_DIR / "horizon_comparison.txt"
    lines = []
    lines.append(SEP)
    lines.append("Horizon Comparison -- Babcock Ranch Solar Energy Center")
    lines.append("Test year: 2024  |  Horizons: h=1, 3, 6, 24")
    lines.append(SEP)
    lines.append("")

    lines.append("--- CONFIGURATION ---")
    lines.append("")
    hdr = (f"  {'h':>4}  {'mode':<14}  {'threshold':>10}  {'clear_kt':<14}  "
           f"{'train_years':>12}  {'early_stop':>10}  {'conf_year':>9}  {'test_year':>9}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for _, row in df.iterrows():
        thr_s = _fmt(row.get("selected_threshold"), 2)
        lines.append(
            f"  {int(row['horizon']):>4}  "
            f"{str(row.get('selected_prediction_mode', '')):.<14}  "
            f"{thr_s:>10}  "
            f"{str(row.get('clear_kt_strategy', '')):.<14}  "
            f"{str(row.get('train_years', '')):>12}  "
            f"{_fmt_int(row.get('early_stop_year')):>10}  "
            f"{_fmt_int(row.get('conformal_year')):>9}  "
            f"{_fmt_int(row.get('test_year')):>9}"
        )
    lines.append("")

    lines.append("--- PREDICTION MODE COMPARISON (2024 holdout, all modes) ---")
    lines.append("")
    hdr = (f"  {'h':>4}  {'mode':<14}  {'all_rmse':>9}  "
           f"{'clear_rmse':>10}  {'cloudy_rmse':>11}  {'ramp_rmse':>9}  {'all_skill':>10}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for raw_row in rows_raw:
        h = raw_row["horizon"]
        mc = raw_row.get("mode_comparison", {})
        yest_rmse = raw_row.get("all_daytime_rmse_yesterday", float("nan"))
        for mode in ["hard_fixed", "hard_tuned", "soft"]:
            if mode not in mc:
                continue
            mv = mc[mode]
            sel = raw_row.get("selected_prediction_mode", "")
            marker = " *" if mode == sel else "  "
            lines.append(
                f"  {h:>4}  {mode:<14}  {mv['all_rmse']:>9.2f}  "
                f"{mv['clear_rmse']:>10.2f}  {mv['cloudy_rmse']:>11.2f}  "
                f"{mv['ramp_rmse']:>9.2f}  {mv['skill']:>+10.4f}{marker}"
            )
        lines.append(
            f"  {h:>4}  {'yesterday':<14}  {_fmt(yest_rmse, 2):>9}  "
            f"{'':>10}  {'':>11}  {'':>9}  {'---':>10}"
        )
        lines.append("")

    lines.append("--- STAGE 1 METRICS ---")
    lines.append("")
    hdr = (
        f"  {'h':>4}  {'AUC':>6}  {'F1':>6}  {'Prec':>6}  "
        f"{'Recall':>6}  {'Brier':>6}  {'FPR':>6}  {'FNR':>6}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for _, row in df.iterrows():
        lines.append(
            f"  {int(row['horizon']):>4}  "
            f"{_fmt(row.get('stage1_auc')):>6}  "
            f"{_fmt(row.get('stage1_f1')):>6}  "
            f"{_fmt(row.get('stage1_precision')):>6}  "
            f"{_fmt(row.get('stage1_recall')):>6}  "
            f"{_fmt(row.get('stage1_brier')):>6}  "
            f"{_fmt(row.get('stage1_false_clear_rate')):>6}  "
            f"{_fmt(row.get('stage1_false_cloudy_rate')):>6}"
        )
    lines.append("")

    lines.append("--- TWO-STAGE GHI METRICS (selected mode, W/m2) ---")
    lines.append("")
    segs = [
        ("All daytime", "all_daytime"),
        ("Clear-sky", "clear_sky"),
        ("Cloudy", "cloudy"),
        ("Ramp events", "ramp"),
    ]
    hdr = (f"  {'h':>4}  {'Segment':<12}  "
           f"{'RMSE_mod':>8}  {'RMSE_yest':>9}  {'Skill':>7}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for _, row in df.iterrows():
        h = int(row["horizon"])
        for seg_label, seg_key in segs:
            lines.append(
                f"  {h:>4}  {seg_label:<12}  "
                f"{_fmt(row.get(f'{seg_key}_rmse_model'), 2):>8}  "
                f"{_fmt(row.get(f'{seg_key}_rmse_yesterday'), 2):>9}  "
                f"{_fmts(row.get(f'{seg_key}_skill_score'), 4):>7}"
            )
        lines.append("")

    lines.append("--- CONFORMAL PREDICTION METRICS ---")
    lines.append("")
    hdr = (f"  {'h':>4}  {'Method':<18}  "
           f"{'Coverage':>8}  {'Width':>7}  {'Winkler':>8}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    methods = [
        ("Stratified", "stratified"),
        ("Single-window", "single_window"),
        ("Static", "static"),
    ]
    for _, row in df.iterrows():
        h = int(row["horizon"])
        for mlabel, mkey in methods:
            lines.append(
                f"  {h:>4}  {mlabel:<18}  "
                f"{_fmt(row.get(f'{mkey}_marginal_coverage')):>8}  "
                f"{_fmt(row.get(f'{mkey}_mean_width'), 1):>7}  "
                f"{_fmt(row.get(f'{mkey}_winkler_score', float('nan')), 2):>8}"
            )
        lines.append("")

    lines.append(SEP)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def save_md(df: pd.DataFrame) -> Path:
    p = OUT_DIR / "horizon_comparison.md"
    lines = []
    lines.append("# Horizon Comparison -- Babcock Ranch Solar Energy Center")
    lines.append("")
    lines.append("**Test year:** 2024  ")
    lines.append("**Horizons:** h=1, h=3, h=6, h=24  ")
    lines.append("**Model:** Two-stage LightGBM (Stage 1 binary classifier + Stage 2 kt regressor)  ")
    lines.append("**Conformal:** Regime-stratified adaptive (Stage 3b)  ")
    lines.append("")

    lines.append("## Configuration and Selected Prediction Mode")
    lines.append("")
    lines.append("| h | mode | threshold | clear_kt_strategy | train_years | early_stop | conf_year | test_year |")
    lines.append("|--:|:-----|----------:|:------------------|:-----------:|:----------:|:---------:|:---------:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['horizon'])} "
            f"| {row.get('selected_prediction_mode', '')} "
            f"| {_fmt(row.get('selected_threshold'), 2)} "
            f"| {row.get('clear_kt_strategy', '')} "
            f"| {row.get('train_years', '')} "
            f"| {_fmt_int(row.get('early_stop_year'))} "
            f"| {_fmt_int(row.get('conformal_year'))} "
            f"| {_fmt_int(row.get('test_year'))} |"
        )
    lines.append("")

    lines.append("## Stage 1 Metrics (2024 holdout, threshold=0.65)")
    lines.append("")
    lines.append("| h | AUC | F1 | Precision | Recall | Brier | FPR | FNR |")
    lines.append("|--:|----:|---:|----------:|-------:|------:|----:|----:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['horizon'])} "
            f"| {_fmt(row.get('stage1_auc'))} "
            f"| {_fmt(row.get('stage1_f1'))} "
            f"| {_fmt(row.get('stage1_precision'))} "
            f"| {_fmt(row.get('stage1_recall'))} "
            f"| {_fmt(row.get('stage1_brier'))} "
            f"| {_fmt(row.get('stage1_false_clear_rate'))} "
            f"| {_fmt(row.get('stage1_false_cloudy_rate'))} |"
        )
    lines.append("")

    lines.append("## Two-Stage GHI Metrics (selected mode, 2024 holdout)")
    lines.append("")
    lines.append("| h | Segment | RMSE_model | RMSE_yesterday | Skill |")
    lines.append("|--:|:--------|----------:|---------------:|------:|")
    segs = [
        ("All daytime", "all_daytime"),
        ("Clear-sky", "clear_sky"),
        ("Cloudy", "cloudy"),
        ("Ramp events", "ramp"),
    ]
    for _, row in df.iterrows():
        h = int(row["horizon"])
        for seg_label, seg_key in segs:
            lines.append(
                f"| {h} | {seg_label} "
                f"| {_fmt(row.get(f'{seg_key}_rmse_model'), 2)} "
                f"| {_fmt(row.get(f'{seg_key}_rmse_yesterday'), 2)} "
                f"| {_fmts(row.get(f'{seg_key}_skill_score'), 4)} |"
            )
    lines.append("")

    lines.append("## Conformal Prediction Metrics (2024 holdout)")
    lines.append("")
    lines.append("| h | Method | Coverage | Mean Width (W/m²) | Winkler |")
    lines.append("|--:|:-------|---------:|------------------:|--------:|")
    methods = [
        ("Stratified", "stratified"),
        ("Single-window", "single_window"),
        ("Static", "static"),
    ]
    for _, row in df.iterrows():
        h = int(row["horizon"])
        for mlabel, mkey in methods:
            lines.append(
                f"| {h} | {mlabel} "
                f"| {_fmt(row.get(f'{mkey}_marginal_coverage'))} "
                f"| {_fmt(row.get(f'{mkey}_mean_width'), 1)} "
                f"| {_fmt(row.get(f'{mkey}_winkler_score', float('nan')), 2)} |"
            )
    lines.append("")

    lines.append("## Output Artifacts")
    lines.append("")
    lines.append("| Artifact | Path |")
    lines.append("|:---------|:-----|")
    lines.append("| Comparison CSV | `outputs/horizon_comparison.csv` |")
    lines.append("| Comparison TXT | `outputs/horizon_comparison.txt` |")
    lines.append("| Degradation plot | `notebooks/horizon_degradation_curve.png` |")
    lines.append("| Mode comparison plot | `notebooks/prediction_mode_comparison.png` |")
    for h in HORIZONS:
        tag = f"h{h:03d}"
        lines.append(f"| h={h} Stage 2 report | `outputs/{tag}/stage2_full_report.txt` |")
        lines.append(f"| h={h} Conformal report | `outputs/{tag}/stage3_conformal_report.txt` |")
        lines.append(f"| h={h} Threshold JSON | `models/{tag}/stage1_threshold.json` |")
        lines.append(f"| h={h} Mode JSON | `models/{tag}/prediction_mode_selection.json` |")
    lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def plot_degradation(df: pd.DataFrame) -> Path:
    hs = df["horizon"].astype(int).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Horizon Degradation -- Babcock Ranch Solar Energy Center  (2024 holdout, selected mode)",
        fontsize=11,
    )

    ax = axes[0, 0]
    rmse_mod = df["all_daytime_rmse_model"].tolist()
    rmse_yest = df["all_daytime_rmse_yesterday"].tolist()
    ax.plot(hs, rmse_mod, "o-", color="steelblue", label="Two-stage (selected)")
    ax.plot(hs, rmse_yest, "s--", color="gray", label="Yesterday's weather")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("GHI RMSE (W/m²)")
    ax.set_title("All-daytime GHI RMSE")
    ax.set_xticks(hs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    skill_all = df["all_daytime_skill_score"].tolist()
    skill_cld = df["cloudy_skill_score"].tolist()
    ax.plot(hs, skill_all, "o-", color="steelblue", label="All daytime")
    ax.plot(hs, skill_cld, "^-", color="darkorange", label="Cloudy")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Skill score (vs yesterday)")
    ax.set_title("GHI Skill Score")
    ax.set_xticks(hs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    auc = df["stage1_auc"].tolist()
    ax.plot(hs, auc, "o-", color="mediumpurple")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Stage 1 Classifier AUC")
    ax.set_xticks(hs)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    strat_w = df["stratified_mean_width"].tolist()
    sw_w = df["single_window_mean_width"].tolist()
    static_w = df["static_mean_width"].tolist()
    ax.plot(hs, strat_w, "o-", color="steelblue", label="Stratified")
    ax.plot(hs, sw_w, "s--", color="darkorange", label="Single-window")
    ax.plot(hs, static_w, "^:", color="gray", label="Static")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Mean interval width (W/m²)")
    ax.set_title("Conformal Interval Width")
    ax.set_xticks(hs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / "horizon_degradation_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_mode_comparison(rows_raw: list) -> Path:
    hs = [r["horizon"] for r in rows_raw]
    modes = ["hard_fixed", "hard_tuned", "soft"]
    mode_labels = {"hard_fixed": "Hard fixed (0.65)", "hard_tuned": "Hard tuned", "soft": "Soft mixture"}
    colors = {"hard_fixed": "#E65100", "hard_tuned": "#1565C0", "soft": "#2E7D32"}
    markers = {"hard_fixed": "s", "hard_tuned": "^", "soft": "o"}

    def _extract(rows, key1, key2):
        return [r.get("mode_comparison", {}).get(key1, {}).get(key2, float("nan"))
                for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Prediction Mode Comparison -- Babcock Ranch Solar Energy Center  (2024 holdout)",
        fontsize=11,
    )

    ax = axes[0, 0]
    for mode in modes:
        vals = _extract(rows_raw, mode, "all_rmse")
        ax.plot(hs, vals, f"{markers[mode]}-", color=colors[mode],
                label=mode_labels[mode], lw=1.8)
    yest = [r.get("all_daytime_rmse_yesterday", float("nan")) for r in rows_raw]
    ax.plot(hs, yest, "D--", color="gray", label="Yesterday", lw=1.4, alpha=0.7)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("GHI RMSE (W/m²)")
    ax.set_title("All-Daytime GHI RMSE")
    ax.set_xticks(hs)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for mode in modes:
        vals = _extract(rows_raw, mode, "clear_rmse")
        ax.plot(hs, vals, f"{markers[mode]}-", color=colors[mode],
                label=mode_labels[mode], lw=1.8)
    clr_yest = [r.get("clear_sky_rmse_yesterday", float("nan")) for r in rows_raw]
    ax.plot(hs, clr_yest, "D--", color="gray", label="Yesterday", lw=1.4, alpha=0.7)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("GHI RMSE (W/m²)")
    ax.set_title("Clear-Sky GHI RMSE")
    ax.set_xticks(hs)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for mode in modes:
        vals = _extract(rows_raw, mode, "cloudy_rmse")
        ax.plot(hs, vals, f"{markers[mode]}-", color=colors[mode],
                label=mode_labels[mode], lw=1.8)
    cld_yest = [r.get("cloudy_rmse_yesterday", float("nan")) for r in rows_raw]
    ax.plot(hs, cld_yest, "D--", color="gray", label="Yesterday", lw=1.4, alpha=0.7)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("GHI RMSE (W/m²)")
    ax.set_title("Cloudy GHI RMSE")
    ax.set_xticks(hs)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for mode in modes:
        vals = _extract(rows_raw, mode, "skill")
        ax.plot(hs, vals, f"{markers[mode]}-", color=colors[mode],
                label=mode_labels[mode], lw=1.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Skill score (vs yesterday)")
    ax.set_title("All-Daytime Skill Score")
    ax.set_xticks(hs)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / "prediction_mode_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def verify_artifacts() -> list:
    issues = []
    for h in HORIZONS:
        mdir = ROOT_DIR / "models" / f"h{h:03d}"
        odir = ROOT_DIR / "outputs" / f"h{h:03d}"
        for fname in [
            "stage1_classifier.pkl",
            "stage1_feature_list.json",
            "stage2_regressor.pkl",
            "stage2_feature_list.json",
            "stage1_threshold.json",
            "prediction_mode_selection.json",
        ]:
            p = mdir / fname
            if not p.exists():
                issues.append(f"MISSING: {p}")
        for fname in [
            "stage1_calibration_report.txt",
            "stage2_full_report.txt",
            "stage3_conformal_report.txt",
        ]:
            p = odir / fname
            if not p.exists():
                issues.append(f"MISSING: {p}")
    return issues


def main() -> None:
    print(SEP)
    print("Horizon Comparison Script")
    print("Babcock Ranch Solar Energy Center")
    print(SEP)

    print("\n[1] Verifying horizon artifacts ...")
    art_issues = verify_artifacts()
    if art_issues:
        for iss in art_issues:
            print(f"  {iss}")
    else:
        print("  All horizon artifacts present.")

    print("\n[2] Parsing report files ...")
    rows_raw = [build_row(h) for h in HORIZONS]
    df = build_dataframe()
    print(f"  Rows: {len(df)}  Columns: {len(df.columns)}")
    for h in HORIZONS:
        row = df[df["horizon"] == h].iloc[0]
        n_filled = row.notna().sum()
        mode = row.get("selected_prediction_mode", "?")
        thr = _fmt(row.get("selected_threshold"), 2)
        print(f"  h={h:>2}: {n_filled} / {len(COLUMNS)} fields  mode={mode}  threshold={thr}")

    print("\n[3] Saving CSV ...")
    csv_path = save_csv(df)
    print(f"  {csv_path}  ({csv_path.stat().st_size / 1024:.1f} KB)")

    print("\n[4] Saving TXT report ...")
    txt_path = save_txt(df, rows_raw)
    print(f"  {txt_path}  ({txt_path.stat().st_size / 1024:.1f} KB)")

    print("\n[5] Saving Markdown summary ...")
    md_path = save_md(df)
    print(f"  {md_path}  ({md_path.stat().st_size / 1024:.1f} KB)")

    print("\n[6] Generating horizon degradation curve ...")
    deg_path = plot_degradation(df)
    print(f"  {deg_path}")

    print("\n[7] Generating prediction mode comparison plot ...")
    mode_plot_path = plot_mode_comparison(rows_raw)
    print(f"  {mode_plot_path}")

    print(f"\n{SEP}")
    print("KEY METRICS SUMMARY")
    print(SEP)
    print(f"\n  {'h':>4}  {'mode':<14}  {'thr':>4}  {'S1_AUC':>7}  "
          f"{'S2_RMSE':>8}  {'GHI_RMSE':>9}  {'Skill':>7}  {'Strat_Width':>11}")
    print(f"  {'----':>4}  {'----':.<14}  {'---':>4}  {'-------':>7}  "
          f"{'--------':>8}  {'---------':>9}  {'-------':>7}  {'-----------':>11}")
    for _, row in df.iterrows():
        thr_s = _fmt(row.get("selected_threshold"), 2)
        print(
            f"  {int(row['horizon']):>4}  "
            f"{str(row.get('selected_prediction_mode', '')):.<14}  "
            f"{thr_s:>4}  "
            f"{_fmt(row.get('stage1_auc')):>7}  "
            f"{_fmt(row.get('stage2_cloudy_kt_rmse')):>8}  "
            f"{_fmt(row.get('all_daytime_rmse_model'), 2):>9}  "
            f"{_fmts(row.get('all_daytime_skill_score'), 4):>7}  "
            f"{_fmt(row.get('stratified_mean_width'), 1):>11}"
        )

    print(f"\n{SEP}")
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {md_path}")
    print(f"  {deg_path}")
    print(f"  {mode_plot_path}")
    print(f"\n{SEP}")
    print("Horizon comparison complete.")


if __name__ == "__main__":
    main()
