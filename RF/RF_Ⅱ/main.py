from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from data_loader import load_dataset
from train_rf import train_single_target
from visualize import plot_feature_importance, plot_scatter


def ensure_output_dirs(base_dir: str = "results") -> Dict[str, Path]:
    base = Path(base_dir)
    dirs = {
        "metrics": base / "metrics",
        "predictions": base / "predictions",
        "models": base / "models",
        "tuning": base / "tuning",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF regression for steel high-temperature degradation factors")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/steel_v2_cleaned_r2.csv",
        help="Path to input CSV data",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="rf_random",
        help="Prefix used in all output file names",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(base_dir="results")

    df, feature_columns, targets = load_dataset(args.data_path)

    metrics_rows: List[Dict[str, object]] = []
    oof_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    for target in targets:
        print(f"[INFO] Training target: {target}")
        output = train_single_target(
            df=df,
            feature_columns=feature_columns,
            target=target,
            random_state=args.random_state,
            test_size=0.2,
        )

        metrics_rows.append(output["result"])

        if not output["success"]:
            print(f"[ERROR] Target {target} failed: {output['result'].get('error', 'Unknown error')}")
            continue

        try:
            model_path = dirs["models"] / f"{args.output_prefix}_{target}.pkl"
            joblib.dump(output["model"], model_path)

            params_path = dirs["tuning"] / f"{args.output_prefix}_{target}_best_params.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(output["result"]["best_params"], f, ensure_ascii=False, indent=2)

            oof_df = output["oof_predictions"]
            test_df = output["test_predictions"]
            oof_frames.append(oof_df)
            test_frames.append(test_df)

            row = output["result"]
            plot_scatter(
                y_true=test_df["y_true"].values,
                y_pred=test_df["y_pred"].values,
                target=target,
                metrics={"R2": row["test_R2"], "MAE": row["test_MAE"], "RMSE": row["test_RMSE"], "P20": row["test_P20"]},
                output_path=str(dirs["figures"] / f"{args.output_prefix}_{target}_test_scatter.png"),
                split_name="test",
            )

            oof_metrics = output["oof_metrics"]
            plot_scatter(
                y_true=oof_df["y_true"].values,
                y_pred=oof_df["y_pred"].values,
                target=target,
                metrics=oof_metrics,
                output_path=str(dirs["figures"] / f"{args.output_prefix}_{target}_oof_scatter.png"),
                split_name="oof",
            )

            plot_feature_importance(
                feature_importance=output["feature_importance"],
                target=target,
                output_path=str(dirs["figures"] / f"{args.output_prefix}_{target}_feature_importance.png"),
            )
        except Exception as exc:
            print(f"[ERROR] Target {target} post-processing failed: {exc}")
            continue

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = dirs["metrics"] / f"{args.output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    oof_path = dirs["predictions"] / f"{args.output_prefix}_oof_predictions.csv"
    test_path = dirs["predictions"] / f"{args.output_prefix}_test_predictions.csv"

    if oof_frames:
        all_oof = pd.concat(oof_frames, axis=0, ignore_index=True)
    else:
        all_oof = pd.DataFrame(columns=["sample_id", "diff_steel", "target", "y_true", "y_pred", "split", "fold"])
    all_oof.to_csv(oof_path, index=False)

    if test_frames:
        all_test = pd.concat(test_frames, axis=0, ignore_index=True)
    else:
        all_test = pd.DataFrame(columns=["sample_id", "diff_steel", "target", "y_true", "y_pred", "split"])
    all_test.to_csv(test_path, index=False)

    print(f"[INFO] Metrics saved to: {metrics_path}")
    print(f"[INFO] OOF predictions saved to: {oof_path}")
    print(f"[INFO] Test predictions saved to: {test_path}")
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
