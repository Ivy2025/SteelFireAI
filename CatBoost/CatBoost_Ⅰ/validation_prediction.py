from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS

# 直接在本文件定义NUMERIC_FEATURE_COLUMNS，保证独立性
PROCESS_COLUMNS = ["N", "HR", "TMCP", "TMCP_T", "QT", "CF"]
CHEMICAL_COLUMNS = [
    "C", "Mn", "Si", "Cr", "Mo", "Nb", "V", "Ti", "Ni", "Al", "Cu", "P", "S"
]
TEMPERATURE_COLUMN = "temperature"
NUMERIC_FEATURE_COLUMNS = PROCESS_COLUMNS + CHEMICAL_COLUMNS + [TEMPERATURE_COLUMN]


def read_validation_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed, errors="ignore")

    return df


def build_process_type_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "process_type" in df.columns:
        return df

    process_cols = [c for c in ["N", "HR", "TMCP", "TMCP_T", "QT", "CF"] if c in df.columns]
    if not process_cols:
        df["process_type"] = "UNKNOWN"
        return df

    flags = df[process_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    onehot_sum = flags.sum(axis=1)

    process_type = pd.Series("UNKNOWN", index=df.index, dtype="string")
    one_mask = onehot_sum.eq(1)
    process_type.loc[one_mask] = flags.loc[one_mask].idxmax(axis=1).astype("string")
    process_type.loc[onehot_sum.eq(0)] = "NONE"
    process_type.loc[onehot_sum.gt(1)] = "MULTI"

    df["process_type"] = process_type
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in NUMERIC_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["process_type"] = df["process_type"].fillna("UNKNOWN").astype(str)
    return df[FEATURE_COLUMNS].copy()


def model_paths(model_dir: str, prefix: str, plan: str) -> Dict[str, str]:
    return {
        target: os.path.join(model_dir, f"{prefix}_{plan}_{target}.pkl")
        for target in TARGET_COLUMNS
    }


def build_prediction_output(df: pd.DataFrame, preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    if "diff_steel" in df.columns:
        diff_steel = df["diff_steel"]
    elif "source_ref" in df.columns:
        diff_steel = df["source_ref"]
    elif "sourcr_ref" in df.columns:
        diff_steel = df["sourcr_ref"]
    else:
        diff_steel = ""

    output = pd.DataFrame(
        {
            "steel_ID": df["steel_ID"] if "steel_ID" in df.columns else pd.Series(range(1, len(df) + 1)),
            "diff_steel": diff_steel,
            "temperature": pd.to_numeric(df["temperature"], errors="coerce") if "temperature" in df.columns else np.nan,
        }
    )

    for target in TARGET_COLUMNS:
        short_name = target.replace("_reduction", "")
        y_true = pd.to_numeric(df[target], errors="coerce") if target in df.columns else pd.Series(np.nan, index=df.index)
        y_pred = pd.Series(preds[target], index=df.index, dtype=float)

        # Signed relative error based on original values.
        rel = (y_pred - y_true) / y_true
        rel = rel.where(y_true != 0, np.nan)

        output[f"{short_name}_true"] = y_true
        output[f"{short_name}_pred"] = y_pred
        output[f"{short_name}_relative_error"] = rel

    return output


def build_statistics(pred_df: pd.DataFrame) -> pd.DataFrame:
    specs: List[Tuple[str, str]] = [
        ("fy", "fy_relative_error"),
        ("fu", "fu_relative_error"),
        ("E", "E_relative_error"),
    ]

    temperature = pd.to_numeric(pred_df["temperature"], errors="coerce")
    rows = []

    for target, err_col in specs:
        rel_all = pd.to_numeric(pred_df[err_col], errors="coerce")
        valid_all = rel_all.notna()
        n_all = int(valid_all.sum())
        gt_all = int((rel_all[valid_all] > 0.2).sum())
        pct_all = (gt_all / n_all * 100.0) if n_all else float("nan")

        mask_lt800 = temperature < 800
        rel_lt800 = pd.to_numeric(pred_df.loc[mask_lt800, err_col], errors="coerce")
        valid_lt800 = rel_lt800.notna()
        n_lt800 = int(valid_lt800.sum())
        gt_lt800 = int((rel_lt800[valid_lt800] > 0.2).sum())
        pct_lt800 = (gt_lt800 / n_lt800 * 100.0) if n_lt800 else float("nan")

        rows.append(
            {
                "target": target,
                "threshold": "relative_error > 0.2 (signed)",
                "all_total_samples": n_all,
                "all_count_gt_0_2": gt_all,
                "all_percent_gt_0_2": pct_all,
                "lt800_total_samples": n_lt800,
                "lt800_count_gt_0_2": gt_lt800,
                "lt800_percent_gt_0_2": pct_lt800,
            }
        )

    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    validation_df = read_validation_csv(args.input)
    validation_df = build_process_type_if_needed(validation_df)
    X = prepare_feature_matrix(validation_df)

    path_map = model_paths(args.model_dir, args.prefix, args.plan)
    missing = [f for f in path_map.values() if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    preds: Dict[str, np.ndarray] = {}
    for target in TARGET_COLUMNS:
        model = joblib.load(path_map[target])
        preds[target] = model.predict(X)

    pred_df = build_prediction_output(validation_df, preds)
    stat_df = build_statistics(pred_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)

    pred_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    stat_df.to_csv(args.stats_output, index=False, encoding="utf-8-sig")

    print(f"[OK] validation predictions saved: {args.output}")
    print(f"[OK] validation statistics saved: {args.stats_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict validation dataset with trained CatBoost plan models.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "validation_data.CSV"),
        help="Validation CSV path.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join("results", "models"),
        help="Directory containing trained model files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="catboost",
        help="Model filename prefix.",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="plan2",
        help="Plan name in model filenames, e.g. plan2 or plan2.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("results", "validation_prediction.csv"),
        help="Prediction output CSV path.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=os.path.join("results", "val_data_statistics.csv"),
        help="Statistics output CSV path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
