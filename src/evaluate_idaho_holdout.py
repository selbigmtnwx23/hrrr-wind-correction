#!/usr/bin/env python3
"""
Reproduce the Idaho held-out evaluation used for the assignment.

The supervised problem is:
  target_delta_u = observed_u - HRRR_u10
  target_delta_v = observed_v - HRRR_v10

Two LightGBM regressors are trained on the first 80% of unique valid times.
The final 20% of unique valid times is held out chronologically.

Usage:
    python3 src/evaluate_idaho_holdout.py \
      data/wirewarrior_hrrr_paired_2025-04-01_2026-04-01.csv.gz \
      results/idaho_holdout_metrics.csv \
      results/idaho_holdout_scored.csv.gz
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

FEATURES = [
    "hrrr_u10_ms", "hrrr_v10_ms", "hrrr_wind10_ms", "hrrr_gust_ms",
    "hrrr_u80_ms", "hrrr_v80_ms", "hrrr_80m_vector_speed_ms",
    "hrrr_t2m_k", "hrrr_d2m_k", "hrrr_pressure_sfc_pa",
    "hrrr_pbl_height_m", "hrrr_friction_velocity_ms",
    "hrrr_roughness_length_m", "hrrr_dswrf_wm2",
    "hrrr_prate_kgm2s", "hrrr_total_cloud_pct",
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

def mae(a,b):
    return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def rmse(a,b):
    d=np.asarray(a)-np.asarray(b)
    return float(np.sqrt(np.mean(d*d)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("paired_csv")
    p.add_argument("metrics_output")
    p.add_argument("scored_output")
    args=p.parse_args()

    df=pd.read_csv(args.paired_csv)
    df["valid_hour_utc"]=pd.to_datetime(df["valid_hour_utc"], utc=True)

    df["target_delta_u_ms"]=df["obs_u_mean_ms"]-df["hrrr_u10_ms"]
    df["target_delta_v_ms"]=df["obs_v_mean_ms"]-df["hrrr_v10_ms"]

    needed=FEATURES+["target_delta_u_ms","target_delta_v_ms","obs_u_mean_ms","obs_v_mean_ms"]
    work=df.dropna(subset=needed).sort_values("valid_hour_utc").copy()

    unique_times=np.sort(work["valid_hour_utc"].unique())
    split_time=unique_times[int(len(unique_times)*0.80)]

    train=work[work["valid_hour_utc"] < split_time].copy()
    test=work[work["valid_hour_utc"] >= split_time].copy()

    u_model=LGBMRegressor(**PARAMS).fit(train[FEATURES], train["target_delta_u_ms"])
    v_model=LGBMRegressor(**PARAMS).fit(train[FEATURES], train["target_delta_v_ms"])

    test["pred_delta_u_ms"]=u_model.predict(test[FEATURES])
    test["pred_delta_v_ms"]=v_model.predict(test[FEATURES])
    test["corrected_u_ms"]=test["hrrr_u10_ms"]+test["pred_delta_u_ms"]
    test["corrected_v_ms"]=test["hrrr_v10_ms"]+test["pred_delta_v_ms"]

    test["obs_vector_speed_ms"]=np.hypot(test["obs_u_mean_ms"],test["obs_v_mean_ms"])
    test["raw_hrrr_vector_speed_ms"]=np.hypot(test["hrrr_u10_ms"],test["hrrr_v10_ms"])
    test["corrected_vector_speed_ms"]=np.hypot(test["corrected_u_ms"],test["corrected_v_ms"])

    raw_rmse=rmse(test["raw_hrrr_vector_speed_ms"],test["obs_vector_speed_ms"])
    corr_rmse=rmse(test["corrected_vector_speed_ms"],test["obs_vector_speed_ms"])
    raw_mae=mae(test["raw_hrrr_vector_speed_ms"],test["obs_vector_speed_ms"])
    corr_mae=mae(test["corrected_vector_speed_ms"],test["obs_vector_speed_ms"])

    metrics=pd.DataFrame([
        {"metric":"Wind-speed RMSE (m/s)","Raw HRRR":raw_rmse,"Corrected":corr_rmse,
         "Improvement %":100*(raw_rmse-corr_rmse)/raw_rmse},
        {"metric":"Wind-speed MAE (m/s)","Raw HRRR":raw_mae,"Corrected":corr_mae,
         "Improvement %":100*(raw_mae-corr_mae)/raw_mae},
        {"metric":"U-component MAE (m/s)",
         "Raw HRRR":mae(test["hrrr_u10_ms"],test["obs_u_mean_ms"]),
         "Corrected":mae(test["corrected_u_ms"],test["obs_u_mean_ms"])},
        {"metric":"V-component MAE (m/s)",
         "Raw HRRR":mae(test["hrrr_v10_ms"],test["obs_v_mean_ms"]),
         "Corrected":mae(test["corrected_v_ms"],test["obs_v_mean_ms"])},
    ])
    for i in [2,3]:
        raw=metrics.loc[i,"Raw HRRR"]; cor=metrics.loc[i,"Corrected"]
        metrics.loc[i,"Improvement %"]=100*(raw-cor)/raw

    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output,index=False)
    compression="gzip" if str(args.scored_output).endswith(".gz") else None
    test.to_csv(args.scored_output,index=False,compression=compression)

    print(f"Train rows: {len(train):,}")
    print(f"Held-out test rows: {len(test):,}")
    print(f"Validation/test start: {split_time}")
    print(metrics.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"\nHeadline RMSE improvement: {100*(raw_rmse-corr_rmse)/raw_rmse:.2f}%")

if __name__=="__main__":
    main()
