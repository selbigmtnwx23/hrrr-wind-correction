#!/usr/bin/env python3
"""
Train the Idaho HRRR wind-correction model from the paired WireWarrior dataset.

Usage:
    python3 train_idaho_model.py \
        wirewarrior_hrrr_paired_2025-04-01_2026-04-01.csv.gz \
        idaho_hrrr_uv_correction_model.joblib \
        idaho_hrrr_uv_correction_model_metadata.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


FEATURES = [
    "hrrr_u10_ms",
    "hrrr_v10_ms",
    "hrrr_wind10_ms",
    "hrrr_gust_ms",
    "hrrr_u80_ms",
    "hrrr_v80_ms",
    "hrrr_80m_vector_speed_ms",
    "hrrr_t2m_k",
    "hrrr_d2m_k",
    "hrrr_pressure_sfc_pa",
    "hrrr_pbl_height_m",
    "hrrr_friction_velocity_ms",
    "hrrr_roughness_length_m",
    "hrrr_dswrf_wm2",
    "hrrr_prate_kgm2s",
    "hrrr_total_cloud_pct",
]

PARAMS = dict(
    n_estimators=700,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d * d)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paired_csv")
    p.add_argument("model_output")
    p.add_argument("metadata_output")
    args = p.parse_args()

    df = pd.read_csv(args.paired_csv)
    df["valid_hour_utc"] = pd.to_datetime(df["valid_hour_utc"], utc=True)

    df["target_delta_u_ms"] = df["obs_u_mean_ms"] - df["hrrr_u10_ms"]
    df["target_delta_v_ms"] = df["obs_v_mean_ms"] - df["hrrr_v10_ms"]

    needed = FEATURES + [
        "target_delta_u_ms",
        "target_delta_v_ms",
        "obs_u_mean_ms",
        "obs_v_mean_ms",
    ]
    work = df.dropna(subset=needed).sort_values("valid_hour_utc").copy()

    unique_times = np.sort(work["valid_hour_utc"].unique())
    split_time = unique_times[int(len(unique_times) * 0.80)]

    train = work[work["valid_hour_utc"] < split_time]
    valid = work[work["valid_hour_utc"] >= split_time]

    u_model = LGBMRegressor(**PARAMS).fit(train[FEATURES], train["target_delta_u_ms"])
    v_model = LGBMRegressor(**PARAMS).fit(train[FEATURES], train["target_delta_v_ms"])

    pred_u = valid["hrrr_u10_ms"].to_numpy() + u_model.predict(valid[FEATURES])
    pred_v = valid["hrrr_v10_ms"].to_numpy() + v_model.predict(valid[FEATURES])

    raw_speed = np.hypot(valid["hrrr_u10_ms"], valid["hrrr_v10_ms"])
    corrected_speed = np.hypot(pred_u, pred_v)
    obs_speed = np.hypot(valid["obs_u_mean_ms"], valid["obs_v_mean_ms"])

    metrics = {
        "validation_start_utc": str(split_time),
        "n_total_rows_used": int(len(work)),
        "n_train_rows": int(len(train)),
        "n_validation_rows": int(len(valid)),
        "raw_u_mae_ms": mae(valid["hrrr_u10_ms"], valid["obs_u_mean_ms"]),
        "corrected_u_mae_ms": mae(pred_u, valid["obs_u_mean_ms"]),
        "raw_v_mae_ms": mae(valid["hrrr_v10_ms"], valid["obs_v_mean_ms"]),
        "corrected_v_mae_ms": mae(pred_v, valid["obs_v_mean_ms"]),
        "raw_vector_speed_mae_ms": mae(raw_speed, obs_speed),
        "corrected_vector_speed_mae_ms": mae(corrected_speed, obs_speed),
        "raw_vector_speed_rmse_ms": rmse(raw_speed, obs_speed),
        "corrected_vector_speed_rmse_ms": rmse(corrected_speed, obs_speed),
    }

    # Refit frozen transfer model on all complete Idaho rows.
    final_u_model = LGBMRegressor(**PARAMS).fit(work[FEATURES], work["target_delta_u_ms"])
    final_v_model = LGBMRegressor(**PARAMS).fit(work[FEATURES], work["target_delta_v_ms"])

    bundle = {
        "model_type": "LightGBM dual-regressor HRRR 10m U/V additive correction",
        "feature_columns": FEATURES,
        "u_model": final_u_model,
        "v_model": final_v_model,
    }
    joblib.dump(bundle, args.model_output)

    metadata = {
        "model_file": Path(args.model_output).name,
        "sha256": sha256_file(args.model_output),
        "source_file": Path(args.paired_csv).name,
        "feature_columns": FEATURES,
        "training_rows": int(len(work)),
        "training_period_start": str(work["valid_hour_utc"].min()),
        "training_period_end": str(work["valid_hour_utc"].max()),
        "validation_metrics_before_final_refit": metrics,
        "target_definition": (
            "Predict obs_u - HRRR_u10 and obs_v - HRRR_v10; "
            "add predictions to HRRR 10m U/V."
        ),
        "lightgbm_params": PARAMS,
    }
    Path(args.metadata_output).write_text(json.dumps(metadata, indent=2))

    print("DONE")
    print(f"Rows used: {len(work):,}")
    print(f"Validation start: {split_time}")
    print(f"Model: {args.model_output}")
    print(f"Metadata: {args.metadata_output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
