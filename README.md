# HRRR Wind Forecast Correction — Idaho Sensor

## Goal
Given an HRRR wind forecast, can a supervised machine-learning model produce a corrected wind forecast that is closer to what the site sensor measured?

## Problem framing
I treated this as a **bias-correction / post-processing problem**, not as a standalone weather model. The target is the error in HRRR's 10 m wind vector:

- `delta_u = observed_u - HRRR_u10`
- `delta_v = observed_v - HRRR_v10`

Two LightGBM regressors predict those corrections. The predicted corrections are added back to the raw HRRR U and V components, then converted to corrected wind speed and direction.

The sensor reports more frequently than HRRR, so observations were aggregated to **hourly U/V wind components** and matched to HRRR by valid time. The model uses 16 HRRR predictors, including 10 m and 80 m winds, gust, temperature, dew point, pressure, boundary-layer height, friction velocity, roughness length, shortwave radiation, precipitation rate, and cloud cover.

## Train / test split
Rows with missing required predictors or targets were removed, leaving **202,462 usable rows**. To avoid leakage, I used a **chronological split by unique valid time** rather than a random split:

- Training rows: **165,582**
- Held-out test rows: **36,880**
- Held-out period begins: **2026-01-14 20:00:00+00:00**

The held-out period was not used to fit the evaluated model.

## Baseline vs corrected model
The baseline is raw HRRR 10 m vector wind speed.

| Metric | Raw HRRR | Corrected model | Change |
|---|---:|---:|---:|
| RMSE | 1.583 m/s | 1.624 m/s | **-2.58%** |
| MAE | 1.214 m/s | 1.197 m/s | **1.40%** |

**Headline result: -2.58% RMSE improvement over raw HRRR.** Because this value is negative, the model **did not beat HRRR on wind-speed RMSE**. It slightly improved wind-speed MAE, and it improved both U- and V-component MAE, but the larger errors became worse enough to increase RMSE.

## Data issues / handling
The paired dataset contains missing values in some HRRR predictors, so only complete rows were used for model fitting and evaluation. Wind was modeled in U/V components rather than direction directly to avoid the 0°/360° discontinuity. One provenance limitation remains: the roughness-length feature is present in the paired dataset, but the exact original raw-data extraction path for that field was not preserved. This repository therefore reproduces the supervised experiment from the paired dataset, not from raw HRRR GRIB files.

## Visual evidence
- `figures/idaho_holdout_timeseries.png` compares observed sensor wind speed, raw HRRR, and corrected wind speed on the held-out period.
- `figures/idaho_holdout_error_distribution.png` compares the raw and corrected wind-speed error distributions.
- `figures/WireWarrior_Idaho_GIS_3D_with_speed_difference.html` is an optional interactive GIS visualization of Idaho wind-speed error by direction sector.

## Reproduce
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 src/evaluate_idaho_holdout.py \
  data/wirewarrior_hrrr_paired_2025-04-01_2026-04-01.csv.gz \
  results/idaho_holdout_metrics_reproduced.csv \
  results/idaho_holdout_scored_reproduced.csv.gz
```

To retrain and save the final fitted model:
```bash
python3 src/train_idaho_model.py \
  data/wirewarrior_hrrr_paired_2025-04-01_2026-04-01.csv.gz \
  model/idaho_hrrr_uv_correction_model_retrained.joblib \
  model/idaho_hrrr_uv_correction_model_retrained_metadata.json
```
