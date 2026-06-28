from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

from data_loader import TARGET_COLUMNS, load_data, prepare_target_data
from train_dt import train_single_target
from visualize import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision Tree regression for steel high-temperature degradation factors")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/steel_v2_cleaned_r2.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="dt",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save all outputs",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for sub in ["metrics", "predictions", "models", "tuning", "figures"]:
        ensure_dir(os.path.join(args.results_dir, sub))

    df = load_data(args.data_path)

    metrics_rows: List[dict] = []
    oof_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []
    errors: List[dict] = []

    for target in TARGET_COLUMNS:
        prepared = prepare_target_data(df, target)
        result = train_single_target(
            prepared=prepared,
            target=target,
            prefix=args.prefix,
            random_state=args.random_state,
            results_root=args.results_dir,
        )

        if result.success:
            metrics_rows.append(result.metrics)
            oof_frames.append(result.oof_df)
            test_frames.append(result.test_pred_df)
            print(f"[OK] {target} finished.")
        else:
            errors.append({"target": target, "error": result.error})
            print(f"[FAILED] {target}: {result.error}")

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(args.results_dir, "metrics", f"{args.prefix}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"Saved metrics: {metrics_path}")

    if oof_frames:
        all_oof = pd.concat(oof_frames, axis=0, ignore_index=True)
        oof_path = os.path.join(args.results_dir, "predictions", f"{args.prefix}_oof_predictions.csv")
        all_oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
        print(f"Saved OOF predictions: {oof_path}")

    if test_frames:
        all_test = pd.concat(test_frames, axis=0, ignore_index=True)
        test_path = os.path.join(args.results_dir, "predictions", f"{args.prefix}_test_predictions.csv")
        all_test.to_csv(test_path, index=False, encoding="utf-8-sig")
        print(f"Saved test predictions: {test_path}")

    if errors:
        error_df = pd.DataFrame(errors)
        error_path = os.path.join(args.results_dir, "metrics", f"{args.prefix}_errors.csv")
        error_df.to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"Some targets failed. Error report: {error_path}")
    else:
        print("All targets completed successfully.")


if __name__ == "__main__":
    main()
