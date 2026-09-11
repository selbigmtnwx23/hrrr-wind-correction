#!/usr/bin/env python3
"""
Run the frozen Idaho HRRR wind-correction model on a CSV/CSV.GZ file.

Usage:
    python run_idaho_model.py input.csv output.csv
    python run_idaho_model.py input.csv.gz output.csv.gz

Default model file:
    idaho_hrrr_uv_correction_model.joblib

The model predicts additive corrections to HRRR 10 m U and V wind components:
    corrected_u = hrrr_u10_ms + predicted_delta_u
    corrected_v = hrrr_v10_ms + predicted_delta_v

It then derives corrected wind speed and meteorological wind direction.

Required input columns are read directly from the trained model bundle's
feature_columns list, so the runner stays synchronized with the model artifact.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib


def meteorological_direction_from_uv(u, v):
    """
    Convert U/V components to meteorological wind direction in degrees,
    where 0/360 = wind FROM north, 90 = FROM east.
    """
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return direction


def load_model(model_path):
    bundle = joblib.load(model_path)

    required_keys = {"feature_columns", "u_model", "v_model"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(
            f"Model bundle is missing required keys: {sorted(missing_keys)}"
        )

    return bundle


def validate_input(df, feature_columns):
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file is missing required model feature columns:\n  - "
            + "\n  - ".join(missing)
        )

    nonnumeric = []
    for c in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                nonnumeric.append(c)

    if nonnumeric:
        raise ValueError(
            "These required feature columns could not be converted to numeric:\n  - "
            + "\n  - ".join(nonnumeric)
        )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Apply the frozen Idaho LightGBM HRRR wind correction model."
    )
    parser.add_argument("input_csv", help="Input CSV or CSV.GZ containing HRRR features")
    parser.add_argument("output_csv", help="Output CSV or CSV.GZ with corrected wind")
    parser.add_argument(
        "--model",
        default="idaho_hrrr_uv_correction_model.joblib",
        help="Path to frozen model joblib file "
             "(default: idaho_hrrr_uv_correction_model.joblib)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Score only rows with complete required features; preserve other rows with NaN outputs.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    model_path = Path(args.model)

    if not input_path.exists():
        sys.exit(f"ERROR: Input file not found: {input_path}")

    if not model_path.exists():
        sys.exit(f"ERROR: Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    bundle = load_model(model_path)

    feature_columns = bundle["feature_columns"]
    u_model = bundle["u_model"]
    v_model = bundle["v_model"]

    print(f"Model features: {len(feature_columns)}")
    for c in feature_columns:
        print(f"  - {c}")

    print(f"\nReading input: {input_path}")
    df = pd.read_csv(input_path)
    df = validate_input(df, feature_columns)

    complete_mask = df[feature_columns].notna().all(axis=1)
    n_complete = int(complete_mask.sum())
    n_total = len(df)

    if not args.allow_missing and n_complete != n_total:
        missing_rows = n_total - n_complete
        sys.exit(
            f"ERROR: {missing_rows:,} of {n_total:,} rows have missing required "
            "features. Re-run with --allow-missing to score only complete rows."
        )

    # Prepare output columns first so incomplete rows remain clearly marked.
    df["pred_delta_u_ms"] = np.nan
    df["pred_delta_v_ms"] = np.nan
    df["corrected_u_ms"] = np.nan
    df["corrected_v_ms"] = np.nan
    df["corrected_wind_speed_ms"] = np.nan
    df["corrected_wind_direction_deg"] = np.nan

    if n_complete:
        X = df.loc[complete_mask, feature_columns]

        pred_du = u_model.predict(X)
        pred_dv = v_model.predict(X)

        raw_u = df.loc[complete_mask, "hrrr_u10_ms"].to_numpy(dtype=float)
        raw_v = df.loc[complete_mask, "hrrr_v10_ms"].to_numpy(dtype=float)

        corrected_u = raw_u + pred_du
        corrected_v = raw_v + pred_dv
        corrected_speed = np.hypot(corrected_u, corrected_v)
        corrected_dir = meteorological_direction_from_uv(corrected_u, corrected_v)

        df.loc[complete_mask, "pred_delta_u_ms"] = pred_du
        df.loc[complete_mask, "pred_delta_v_ms"] = pred_dv
        df.loc[complete_mask, "corrected_u_ms"] = corrected_u
        df.loc[complete_mask, "corrected_v_ms"] = corrected_v
        df.loc[complete_mask, "corrected_wind_speed_ms"] = corrected_speed
        df.loc[complete_mask, "corrected_wind_direction_deg"] = corrected_dir

    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression = "gzip" if output_path.suffix.lower() == ".gz" else None
    df.to_csv(output_path, index=False, compression=compression)

    print("\nDONE")
    print(f"Input rows:   {n_total:,}")
    print(f"Scored rows:  {n_complete:,}")
    print(f"Skipped rows: {n_total - n_complete:,}")
    print(f"Output:       {output_path}")

    if "model_type" in bundle:
        print(f"Model type:   {bundle['model_type']}")


if __name__ == "__main__":
    main()
