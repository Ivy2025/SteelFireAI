import argparse
from pathlib import Path

import pandas as pd

from data_loader import load_data
from train_knn import train_one_target
from visualize import save_test_plots


def parse_args():
    parser = argparse.ArgumentParser(description="KNN regression for steel reduction factors")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/steel_v2_cleaned_r2.csv",
        help="CSV 数据路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于外层分组切分）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例（按分组切分）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="结果输出根目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="knn",
        help="输出文件前缀，可自定义；设为空字符串可不加前缀",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path):
    dirs = {
        "metrics": output_root / "metrics",
        "predictions": output_root / "predictions",
        "models": output_root / "models",
        "tuning": output_root / "tuning",
        "figures": output_root / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def build_file_name(prefix: str, suffix: str) -> str:
    if prefix:
        return f"{prefix}_{suffix}"
    return suffix


def main():
    args = parse_args()

    output_root = Path(args.output_dir)
    dirs = ensure_dirs(output_root)

    df = load_data(args.data_path)

    targets = ["fy_reduction", "fu_reduction", "E_reduction"]

    all_metrics = []
    all_test_preds = []

    for target in targets:
        result = train_one_target(
            df=df,
            target=target,
            seed=args.seed,
            test_size=args.test_size,
            output_dirs=dirs,
            prefix=args.prefix,
        )
        all_metrics.append(result["metrics_row"])
        all_test_preds.append(result["test_predictions"])

    metrics_df = pd.DataFrame(all_metrics)
    test_pred_df = pd.concat(all_test_preds, axis=0, ignore_index=True)

    metrics_file = dirs["metrics"] / build_file_name(args.prefix, "metrics.csv")
    test_pred_file = dirs["predictions"] / build_file_name(args.prefix, "test_predictions.csv")

    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(test_pred_file, index=False, encoding="utf-8-sig")

    save_test_plots(
        test_predictions=test_pred_df,
        figure_dir=dirs["figures"],
        prefix=args.prefix,
    )

    print("训练完成，结果文件如下：")
    print(f"- 指标: {metrics_file}")
    print(f"- 测试预测: {test_pred_file}")
    print(f"- 模型目录: {dirs['models']}")
    print(f"- 调参目录: {dirs['tuning']}")
    print(f"- 图像目录: {dirs['figures']}")


if __name__ == "__main__":
    main()
