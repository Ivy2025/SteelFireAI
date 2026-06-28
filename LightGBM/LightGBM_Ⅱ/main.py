from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import pandas as pd

from data_loader import FEATURE_COLUMNS, TARGET_COLUMNS, load_raw_data, validate_columns
from train_lightgbm import make_random_split_for_target, train_one_target_one_plan


VALID_PLANS = ["plan3", "planB"]


def resolve_data_path(data_path: str) -> str:
    if os.path.exists(data_path):
        return data_path
    fallback = os.path.join("data", os.path.basename(data_path))
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Dataset not found: {data_path} (fallback checked: {fallback})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM regression models for steel degradation factors.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/steel_v2_cleaned_r2.csv",
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="lightgbm",
        help="Prefix used in saved file names.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV. Use 1 for maximum stability on Windows.",
    )
    parser.add_argument(
        "--plans",
        nargs="+",
        default=VALID_PLANS,
        choices=VALID_PLANS,
        help="Plans to run. Example: --plans plan2",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGET_COLUMNS,
        choices=TARGET_COLUMNS,
        help="Targets to run. Example: --targets E_reduction",
    )
    return parser.parse_args()


def save_plan_outputs(
    result_dir: str,
    output_prefix: str,
    plan: str,
    metrics_rows: List[dict],
    oof_list: List[pd.DataFrame],
    test_pred_list: List[pd.DataFrame],
) -> None:
    metrics_dir = os.path.join(result_dir, "metrics")
    pred_dir = os.path.join(result_dir, "predictions")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(metrics_dir, f"{output_prefix}_{plan}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if oof_list:
        oof_df = pd.concat(oof_list, axis=0, ignore_index=True)
    else:
        oof_df = pd.DataFrame(columns=["sample_id", "diff_steel", "plan", "target", "fold", "y_true", "y_pred"])
    oof_path = os.path.join(pred_dir, f"{output_prefix}_{plan}_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    if test_pred_list:
        test_df = pd.concat(test_pred_list, axis=0, ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=["sample_id", "diff_steel", "plan", "target", "y_true", "y_pred", "split"])
    test_path = os.path.join(pred_dir, f"{output_prefix}_{plan}_test_predictions.csv")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")


def save_comparison(result_dir: str, output_prefix: str) -> None:
    metrics_dir = os.path.join(result_dir, "metrics")
    p1 = os.path.join(metrics_dir, f"{output_prefix}_plan1_metrics.csv")
    p2 = os.path.join(metrics_dir, f"{output_prefix}_plan2_metrics.csv")

    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)

    key_cols = ["target"]
    keep_cols = ["test_R2", "test_MAE", "test_P20", "cv_R2_mean", "cv_R2_std", "train_R2", "train_MAE", "train_P20"]

    left = df1[key_cols + keep_cols].copy()
    right = df2[key_cols + keep_cols].copy()

    left = left.rename(columns={c: f"plan1_{c}" for c in keep_cols})
    right = right.rename(columns={c: f"plan2_{c}" for c in keep_cols})

    comp = left.merge(right, on="target", how="outer")
    comp_path = os.path.join(metrics_dir, f"{output_prefix}_plan1_vs_plan2_comparison.csv")
    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()

    data_path = resolve_data_path(args.data_path)
    df = load_raw_data(data_path)
    validate_columns(df)

    plans = args.plans
    targets = args.targets
    split_by_target = {}

    # Same outer split for all plans under the same target.
    for target in targets:
        split_by_target[target] = make_random_split_for_target(
            data_df=df,
            target=target,
            test_size=0.2,
            random_state=args.random_state,
        )

    for plan in plans:
        metrics_rows = []
        oof_frames = []
        test_frames = []

        for target in targets:
            train_sample_ids, test_sample_ids = split_by_target[target]

            start_time = datetime.now()
            print(
                f"[START] plan={plan}, target={target}, time={start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            result = train_one_target_one_plan(
                data_df=df,
                feature_cols=FEATURE_COLUMNS,
                target=target,
                plan=plan,
                train_sample_ids=train_sample_ids,
                test_sample_ids=test_sample_ids,
                output_prefix=args.output_prefix,
                result_dir=args.result_dir,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )

            end_time = datetime.now()
            elapsed_sec = (end_time - start_time).total_seconds()
            print(
                f"[END] plan={plan}, target={target}, time={end_time.strftime('%Y-%m-%d %H:%M:%S')}, elapsed_sec={elapsed_sec:.1f}"
            )

            metrics_rows.append(result.metric_row)
            if not result.oof_df.empty:
                oof_frames.append(result.oof_df)
            if not result.test_pred_df.empty:
                test_frames.append(result.test_pred_df)

            if result.error is not None:
                print(f"[WARN] {result.error}")

        save_plan_outputs(
            result_dir=args.result_dir,
            output_prefix=args.output_prefix,
            plan=plan,
            metrics_rows=metrics_rows,
            oof_list=oof_frames,
            test_pred_list=test_frames,
        )

    print("[INFO] Comparison file skipped for plan3-only run.")
    print("Training completed.")


if __name__ == "__main__":
    main()
