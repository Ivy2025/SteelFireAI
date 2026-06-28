from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from data_loader import TARGET_COLUMNS, load_dataset, prepare_target_data
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
        "test_RMSE",
        "test_P20",
        "train_R2",
        "train_MAE",
        "train_RMSE",
        "train_P20",
        "cv_R2_mean",
        "cv_R2_std",
        "n_splits_used",
    ]

    plan1 = metrics_all.loc[metrics_all["plan"] == "plan1", base_cols].copy()
    plan2 = metrics_all.loc[metrics_all["plan"] == "plan2", base_cols].copy()

    plan1 = plan1.rename(columns={c: f"plan1_{c}" for c in base_cols if c != "target"})
    plan2 = plan2.rename(columns={c: f"plan2_{c}" for c in base_cols if c != "target"})

    merged = pd.merge(plan1, plan2, on="target", how="outer")
    return merged


def run(args: argparse.Namespace) -> None:
    output_dirs = make_output_dirs("results")
    df = load_dataset(args.data_path)

    plans = ["plan1", "plan2"]

    for plan in plans:
        metrics_rows: List[Dict[str, object]] = []
        oof_frames: List[pd.DataFrame] = []
        test_frames: List[pd.DataFrame] = []

        for target in TARGET_COLUMNS:
            try:
                target_data = prepare_target_data(df, target)
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
                        "train_RMSE": float("nan"),
                        "train_P20": float("nan"),
                        "test_R2": float("nan"),
                        "test_MAE": float("nan"),
                        "test_RMSE": float("nan"),
                        "test_P20": float("nan"),
                        "error": str(exc),
                    }
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

    metrics_plan1 = pd.read_csv(os.path.join(output_dirs["metrics"], f"{args.prefix}_plan1_metrics.csv"))
    metrics_plan2 = pd.read_csv(os.path.join(output_dirs["metrics"], f"{args.prefix}_plan2_metrics.csv"))
    merged_metrics = pd.concat([metrics_plan1, metrics_plan2], ignore_index=True)

    comparison_df = build_comparison(merged_metrics)
    comparison_path = os.path.join(
        output_dirs["metrics"],
        f"{args.prefix}_plan1_vs_plan2_comparison.csv",
    )
    comparison_df.to_csv(comparison_path, index=False)

    print("All runs completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost grouped regression for steel degradation factors")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("data", "steel_v2_cleaned_r2.csv"),
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
