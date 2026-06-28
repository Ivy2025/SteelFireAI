from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS


PLAN_TO_MODEL_PREFIX: Dict[str, str] = {
    "plan3": "rf_random",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict validation_data with trained RF models")
    parser.add_argument(
        "--validation-path",
        type=str,
        default="data/validation_data.CSV",
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="results/models",
        help="Directory containing trained model pkl files",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="plan3",
        help="Model plan name. plan3 maps to rf_random model prefix.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/validation_prediction.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def resolve_model_prefix(plan: str) -> str:
    return PLAN_TO_MODEL_PREFIX.get(plan, plan)


def relative_error(y_true: pd.Series, y_pred: np.ndarray) -> pd.Series:
    """Return signed relative error (pred - true) / true.

    For entries where true is NaN or near-zero, return NaN.
    """
    y_true_arr = pd.to_numeric(y_true, errors="coerce").astype(float).to_numpy()
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rel = np.full_like(y_true_arr, np.nan, dtype=float)
    eps = 1e-12
    valid = (~np.isnan(y_true_arr)) & (np.abs(y_true_arr) > eps)
    rel[valid] = (y_pred_arr[valid] - y_true_arr[valid]) / y_true_arr[valid]
    return pd.Series(rel, index=y_true.index)


def load_and_clean_validation(validation_path: str) -> pd.DataFrame:
    df = pd.read_csv(validation_path)

    # Remove noisy trailing unnamed/comment columns from the source file.
    keep_cols = [
        c
        for c in df.columns
        if not str(c).startswith("Unnamed:")
        and str(c).strip().lower() not in {"h1", "h2", "h3"}
    ]
    df = df[keep_cols].copy()

    numeric_columns: List[str] = FEATURE_COLUMNS + TARGET_COLUMNS
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_meta = ["steel_ID", "diff_steel", "temperature"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing required columns in validation data: {missing_meta}")

    missing_feat = [c for c in FEATURE_COLUMNS if c not in df.columns]
    for col in missing_feat:
        # Leave missing features as NaN and let the pipeline imputer handle them.
        df[col] = np.nan

    return df


def predict_validation(df: pd.DataFrame, model_prefix: str, models_dir: str) -> pd.DataFrame:
    model_dir = Path(models_dir)
    result = pd.DataFrame({
        "steel_ID": df["steel_ID"],
        "diff_steel": df["diff_steel"],
        "temperature": df["temperature"]
    })

    for target in TARGET_COLUMNS:
        model_path = model_dir / f"{model_prefix}_{target}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        x = df[FEATURE_COLUMNS].copy()
        y_pred = model.predict(x)
        y_true = df[target] if target in df.columns else pd.Series(np.nan, index=df.index)
        rel_err = relative_error(y_true=y_true, y_pred=y_pred)
        result[f"{target}_true"] = y_true.values
        result[f"{target}_pred"] = y_pred
        result[f"{target}_rel_error"] = rel_err.values

    # Arrange columns in the requested order
    ordered_cols = [
        "steel_ID", "diff_steel", "temperature",
        "fy_reduction_true", "fy_reduction_pred", "fy_reduction_rel_error",
        "fu_reduction_true", "fu_reduction_pred", "fu_reduction_rel_error",
        "E_reduction_true", "E_reduction_pred", "E_reduction_rel_error"
    ]
    result = result[ordered_cols]
    return result


def summarize_over20(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for target in TARGET_COLUMNS:
        true_col = f"{target}_true"
        rel_col = f"{target}_rel_error"
        valid_all = pred_df[rel_col].notna()
        n_all = int(valid_all.sum())
        over_all = int((pred_df.loc[valid_all, rel_col] > 0.2).sum())
        pct_all = float(over_all / n_all * 100.0) if n_all > 0 else float("nan")

        mask_t800 = valid_all & (pred_df["temperature"] < 800)
        n_t800 = int(mask_t800.sum())
        over_t800 = int((pred_df.loc[mask_t800, rel_col] > 0.2).sum())
        pct_t800 = float(over_t800 / n_t800 * 100.0) if n_t800 > 0 else float("nan")

        rows.append(
            {
                "target": target,
                "over20_pct_all": pct_all,
                "over20_count_all": over_all,
                "valid_count_all": n_all,
                "over20_pct_temp_lt_800": pct_t800,
                "over20_count_temp_lt_800": over_t800,
                "valid_count_temp_lt_800": n_t800,
            }
        )
    return pd.DataFrame(rows)


def summarize_over20_by_original_count(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Use only samples with non-NaN and non-zero y_true as denominator."""
    rows: List[Dict[str, float]] = []
    for target in TARGET_COLUMNS:
        rel_col = f"{target}_rel_error"
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"

        def rmse_on_mask(mask: pd.Series) -> float:
            if int(mask.sum()) == 0:
                return float("nan")
            y_true = pd.to_numeric(pred_df.loc[mask, true_col], errors="coerce").to_numpy(dtype=float)
            y_pred = pd.to_numeric(pred_df.loc[mask, pred_col], errors="coerce").to_numpy(dtype=float)
            valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
            if not np.any(valid):
                return float("nan")
            err = y_pred[valid] - y_true[valid]
            return float(np.sqrt(np.mean(err ** 2)))

        # 只统计真实值非空且不为0的样本
        valid_all = pred_df[true_col].notna() & (pred_df[true_col] != 0)
        total_all = int(valid_all.sum())
        over_all = int((pred_df.loc[valid_all, rel_col] > 0.2).sum())
        pct_all = float(over_all / total_all * 100.0) if total_all > 0 else float("nan")
        rmse_all = rmse_on_mask(valid_all)

        mask_t800 = valid_all & (pred_df["temperature"] < 800)
        total_t800 = int(mask_t800.sum())
        over_t800 = int((pred_df.loc[mask_t800, rel_col] > 0.2).sum())
        pct_t800 = float(over_t800 / total_t800 * 100.0) if total_t800 > 0 else float("nan")
        rmse_t800 = rmse_on_mask(mask_t800)

        rows.append(
            {
                "target": target,
                "threshold": 0.2,
                "denominator_rule": "original_row_count",
                "over20_count_all": over_all,
                "total_count_all": total_all,
                "over20_pct_all": pct_all,
                "rmse_all": rmse_all,
                "over20_count_temp_lt_800": over_t800,
                "total_count_temp_lt_800": total_t800,
                "over20_pct_temp_lt_800": pct_t800,
                "rmse_temp_lt_800": rmse_t800,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    model_prefix = resolve_model_prefix(args.plan)

    df = load_and_clean_validation(args.validation_path)
    pred_df = predict_validation(df=df, model_prefix=model_prefix, models_dir=args.models_dir)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    summary_df = summarize_over20(pred_df)
    summary_original_df = summarize_over20_by_original_count(pred_df)
    summary_original_path = output_path.parent / "val_data_statistics.csv"
    summary_original_df.to_csv(summary_original_path, index=False)

    print(f"[INFO] Plan: {args.plan}")
    print(f"[INFO] Model prefix used: {model_prefix}")
    print(f"[INFO] Validation predictions saved to: {output_path}")
    print(f"[INFO] Original-count statistics saved to: {summary_original_path}")
    print("[INFO] Relative error > 0.2 statistics (%):")
    for _, row in summary_df.iterrows():
        print(
            f"  - {row['target']}: "
            f"all={row['over20_pct_all']:.2f}% ({int(row['over20_count_all'])}/{int(row['valid_count_all'])}), "
            f"temp<800={row['over20_pct_temp_lt_800']:.2f}% "
            f"({int(row['over20_count_temp_lt_800'])}/{int(row['valid_count_temp_lt_800'])})"
        )


if __name__ == "__main__":
    main()
