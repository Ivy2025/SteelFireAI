from __future__ import annotations

import argparse
import os
from datetime import datetime
from time import perf_counter
from typing import Dict, List

import pandas as pd

from data_loader import NUMERIC_FEATURE_COLUMNS, TARGET_COLUMNS, load_dataset, prepare_target_data
from train_catboost import train_one_target_one_plan


def make_output_dirs(root: str = "results") -> Dict[str, str]:
    dirs = {
        "metrics": os.path.join(root, "metrics"),
        "predictions": os.path.join(root, "predictions"),
        "models": os.path.join(root, "models"),
        "tuning": os.path.join(root, "tuning"),
        "figures": os.path.join(root, "figures"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def build_comparison(metrics_all: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "target",
        "test_R2",
        "test_MAE",
        "test_P20",
        "train_R2",
        "train_MAE",
        "train_P20",
        "cv_R2_mean",
        "cv_R2_std",
        "n_splits_used",
    ]

    plan3 = metrics_all.loc[metrics_all["plan"] == "plan3", base_cols].copy()
    plan3 = plan3.rename(columns={c: f"plan3_{c}" for c in base_cols if c != "target"})
    return plan3


def build_planA_median_imputed_df(df: pd.DataFrame) -> pd.DataFrame:
    df_planA = df.copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in df_planA.columns:
            continue
        numeric_col = pd.to_numeric(df_planA[col], errors="coerce")
        median_val = numeric_col.median(skipna=True)
        if pd.isna(median_val):
            median_val = 0.0
        df_planA[col] = numeric_col.fillna(median_val)
    return df_planA


def save_planA_dataset(df_planA: pd.DataFrame, source_data_path: str) -> str:
    base, ext = os.path.splitext(source_data_path)
    output_path = f"{base}_planA{ext or '.csv'}"
    save_df = df_planA.drop(columns=["sample_id"], errors="ignore")
    save_df.to_csv(output_path, index=False)
    return output_path


def run(args: argparse.Namespace) -> None:
    output_dirs = make_output_dirs("results")
    df = load_dataset(args.data_path)

    plans = [p.strip() for p in args.plans.split(",") if p.strip()]
    if not plans:
        raise ValueError("No valid plans provided. Example: --plans plan3 or --plans plan3,planB")

    df_planA = None
    if "planA" in plans:
        df_planA = build_planA_median_imputed_df(df)
        planA_data_path = save_planA_dataset(df_planA, args.data_path)
        print(f"[INFO] Saved planA median-imputed dataset: {planA_data_path}")

    for plan in plans:
        metrics_rows: List[Dict[str, object]] = []
        oof_frames: List[pd.DataFrame] = []
        test_frames: List[pd.DataFrame] = []
        current_df = df_planA if plan == "planA" and df_planA is not None else df

        for target in TARGET_COLUMNS:
            start_time = datetime.now()
            start_counter = perf_counter()
            status = "SUCCESS"
            error_message = ""

            print(
                f"[{start_time:%Y-%m-%d %H:%M:%S}] [TARGET_START] "
                f"plan={plan} target={target}"
            )
            try:
                target_data = prepare_target_data(current_df, target)
                result = train_one_target_one_plan(
                    target_data=target_data,
                    target=target,
                    plan=plan,
                    output_prefix=args.prefix,
                    output_dirs=output_dirs,
                    random_state=args.random_state,
                )
                metrics_rows.append(result["metrics_row"])
                oof_frames.append(result["oof_df"])
                test_frames.append(result["test_df"])
            except Exception as exc:
                status = "FAILED"
                error_message = str(exc)
                metrics_rows.append(
                    {
                        "plan": plan,
                        "target": target,
                        "best_params": "",
                        "train_sample_count": 0,
                        "test_sample_count": 0,
                        "train_group_count": 0,
                        "test_group_count": 0,
                        "n_splits_used": 0,
                        "cv_R2_mean": float("nan"),
                        "cv_R2_std": float("nan"),
                        "train_R2": float("nan"),
                        "train_MAE": float("nan"),
                        "train_P20": float("nan"),
                        "test_R2": float("nan"),
                        "test_MAE": float("nan"),
                        "test_P20": float("nan"),
                        "error": str(exc),
                    }
                )
            finally:
                end_time = datetime.now()
                elapsed = perf_counter() - start_counter
                if status == "SUCCESS":
                    print(
                        f"[{end_time:%Y-%m-%d %H:%M:%S}] [TARGET_END] "
                        f"plan={plan} target={target} status={status} elapsed={elapsed:.2f}s"
                    )
                else:
                    print(
                        f"[{end_time:%Y-%m-%d %H:%M:%S}] [TARGET_END] "
                        f"plan={plan} target={target} status={status} elapsed={elapsed:.2f}s "
                        f"error={error_message}"
                    )

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(
            output_dirs["metrics"],
            f"{args.prefix}_{plan}_metrics.csv",
        )
        metrics_df.to_csv(metrics_path, index=False)

        oof_all = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame(
            columns=["sample_id", "diff_steel", "plan", "target", "fold", "y_true", "y_pred"]
        )
        oof_path = os.path.join(
            output_dirs["predictions"],
            f"{args.prefix}_{plan}_oof_predictions.csv",
        )
        oof_all.to_csv(oof_path, index=False)

        test_all = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(
            columns=["sample_id", "diff_steel", "plan", "target", "y_true", "y_pred", "split"]
        )
        test_path = os.path.join(
            output_dirs["predictions"],
            f"{args.prefix}_{plan}_test_predictions.csv",
        )
        test_all.to_csv(test_path, index=False)

    if "plan3" in plans:
        metrics_plan3 = pd.read_csv(os.path.join(output_dirs["metrics"], f"{args.prefix}_plan3_metrics.csv"))
        merged_metrics = pd.concat([metrics_plan3], ignore_index=True)

        comparison_df = build_comparison(merged_metrics)
        comparison_path = os.path.join(
            output_dirs["metrics"],
            f"{args.prefix}_plan3_comparison.csv",
        )
        comparison_df.to_csv(comparison_path, index=False)

    print("All runs completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost grouped regression for steel degradation factors")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("data", "processed_dataset.csv"),
        help="Path to source csv dataset.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="catboost",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used in group split and model training.",
    )
    parser.add_argument(
        "--plans",
        type=str,
        default="plan3",
        help="Comma-separated plans to run, e.g. plan3, planB, or planA.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
