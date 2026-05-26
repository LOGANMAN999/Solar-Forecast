# Horizon Comparison -- Babcock Ranch Solar Energy Center

**Test year:** 2024  
**Horizons:** h=1, h=3, h=6, h=24  
**Model:** Two-stage LightGBM (Stage 1 binary classifier + Stage 2 kt regressor)  
**Conformal:** Regime-stratified adaptive (Stage 3b)  

## Configuration and Selected Prediction Mode

| h | mode | threshold | clear_kt_strategy | train_years | early_stop | conf_year | test_year |
|--:|:-----|----------:|:------------------|:-----------:|:----------:|:---------:|:---------:|
| 1 | soft | 0.35 | constant_1.0 | 2018-2021 | 2022 | 2023 | 2024 |
| 3 | soft | 0.45 | constant_1.0 | 2018-2021 | 2022 | 2023 | 2024 |
| 6 | soft | 0.35 | constant_1.0 | 2018-2021 | 2022 | 2023 | 2024 |
| 24 | soft | 0.30 | constant_1.0 | 2018-2021 | 2022 | 2023 | 2024 |

## Stage 1 Metrics (2024 holdout, threshold=0.65)

| h | AUC | F1 | Precision | Recall | Brier | FPR | FNR |
|--:|----:|---:|----------:|-------:|------:|----:|----:|
| 1 | 0.8895 | 0.7927 | 0.8874 | 0.7163 | 0.1321 | 0.1240 | 0.2837 |
| 3 | 0.8224 | 0.6894 | 0.8524 | 0.5787 | 0.1734 | 0.1366 | 0.4213 |
| 6 | 0.7843 | 0.6417 | 0.8384 | 0.5198 | 0.1900 | 0.1366 | 0.4802 |
| 24 | 0.7037 | 0.5619 | 0.7740 | 0.4411 | 0.2162 | 0.1757 | 0.5589 |

## Two-Stage GHI Metrics (selected mode, 2024 holdout)

| h | Segment | RMSE_model | RMSE_yesterday | Skill |
|--:|:--------|----------:|---------------:|------:|
| 1 | All daytime | 116.00 | 204.18 | +0.4319 |
| 1 | Clear-sky | 92.29 | 166.25 | +0.4449 |
| 1 | Cloudy | 142.09 | 246.64 | +0.4239 |
| 1 | Ramp events | 130.10 | 200.97 | +0.3526 |
| 3 | All daytime | 140.00 | 204.18 | +0.3143 |
| 3 | Clear-sky | 118.73 | 166.25 | +0.2858 |
| 3 | Cloudy | 164.65 | 246.64 | +0.3325 |
| 3 | Ramp events | 137.34 | 200.97 | +0.3166 |
| 6 | All daytime | 150.91 | 204.18 | +0.2609 |
| 6 | Clear-sky | 130.39 | 166.25 | +0.2157 |
| 6 | Cloudy | 175.05 | 246.64 | +0.2903 |
| 6 | Ramp events | 142.13 | 200.97 | +0.2928 |
| 24 | All daytime | 160.47 | 204.18 | +0.2141 |
| 24 | Clear-sky | 134.94 | 166.25 | +0.1883 |
| 24 | Cloudy | 189.83 | 246.64 | +0.2303 |
| 24 | Ramp events | 148.84 | 200.97 | +0.2594 |

## Conformal Prediction Metrics (2024 holdout)

| h | Method | Coverage | Mean Width (W/m²) | Winkler |
|--:|:-------|---------:|------------------:|--------:|
| 1 | Stratified | 0.8980 | 334.3 | 497.74 |
| 1 | Single-window | 0.8980 | 364.2 |  |
| 1 | Static | 0.9203 | 395.7 |  |
| 3 | Stratified | 0.8992 | 394.1 | 555.05 |
| 3 | Single-window | 0.8980 | 438.1 |  |
| 3 | Static | 0.8956 | 441.3 |  |
| 6 | Stratified | 0.8985 | 441.0 | 599.87 |
| 6 | Single-window | 0.8980 | 466.1 |  |
| 6 | Static | 0.8939 | 469.8 |  |
| 24 | Stratified | 0.8985 | 473.8 | 651.05 |
| 24 | Single-window | 0.8975 | 479.7 |  |
| 24 | Static | 0.8902 | 479.3 |  |

## Output Artifacts

| Artifact | Path |
|:---------|:-----|
| Comparison CSV | `outputs/horizon_comparison.csv` |
| Comparison TXT | `outputs/horizon_comparison.txt` |
| Degradation plot | `notebooks/horizon_degradation_curve.png` |
| Mode comparison plot | `notebooks/prediction_mode_comparison.png` |
| h=1 Stage 2 report | `outputs/h001/stage2_full_report.txt` |
| h=1 Conformal report | `outputs/h001/stage3_conformal_report.txt` |
| h=1 Threshold JSON | `models/h001/stage1_threshold.json` |
| h=1 Mode JSON | `models/h001/prediction_mode_selection.json` |
| h=3 Stage 2 report | `outputs/h003/stage2_full_report.txt` |
| h=3 Conformal report | `outputs/h003/stage3_conformal_report.txt` |
| h=3 Threshold JSON | `models/h003/stage1_threshold.json` |
| h=3 Mode JSON | `models/h003/prediction_mode_selection.json` |
| h=6 Stage 2 report | `outputs/h006/stage2_full_report.txt` |
| h=6 Conformal report | `outputs/h006/stage3_conformal_report.txt` |
| h=6 Threshold JSON | `models/h006/stage1_threshold.json` |
| h=6 Mode JSON | `models/h006/prediction_mode_selection.json` |
| h=24 Stage 2 report | `outputs/h024/stage2_full_report.txt` |
| h=24 Conformal report | `outputs/h024/stage3_conformal_report.txt` |
| h=24 Threshold JSON | `models/h024/stage1_threshold.json` |
| h=24 Mode JSON | `models/h024/prediction_mode_selection.json` |
