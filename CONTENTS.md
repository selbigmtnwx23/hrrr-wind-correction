# Repository contents

- `README.md` — assignment-focused problem framing, split, results, limitations, and run instructions.
- `requirements.txt` — Python dependencies.
- `data/wirewarrior_hrrr_paired_2025-04-01_2026-04-01.csv.gz` — paired Idaho HRRR predictors and hourly sensor truth; this is the source dataset for the supervised experiment.
- `src/evaluate_idaho_holdout.py` — reproduces the chronological 80/20 held-out test and writes metrics plus row-level predictions.
- `src/train_idaho_model.py` — trains the documented model and refits a final frozen model on all complete Idaho rows after evaluation.
- `src/run_idaho_model.py` — inference utility for applying the frozen model to any compatible HRRR feature file.
- `model/idaho_hrrr_uv_correction_model.joblib` — frozen final model artifact.
- `model/idaho_hrrr_uv_correction_model_metadata.json` — features, target definition, training period, metrics, and model hash.
- `results/idaho_holdout_metrics.csv` — held-out raw-HRRR vs corrected RMSE/MAE results.
- `results/idaho_holdout_scored.csv.gz` — row-level held-out predictions and observations.
- `figures/idaho_holdout_timeseries.png` — observed vs raw HRRR vs corrected held-out time series.
- `figures/idaho_holdout_error_distribution.png` — raw vs corrected error distribution on the held-out set.
- `figures/WireWarrior_Idaho_GIS_3D_with_speed_difference.html` — optional interactive GIS visual.
