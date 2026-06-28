import argparse

from src.train import DEFAULT_TARGETS, run_svr_experiment, run_svr_planb_pca_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Steel high-temperature reduction factor regression")
    parser.add_argument(
        "--model-type",
        type=str,
        default="svr",
        choices=["svr", "svr_planB_pca"],
        help="支持 svr(基线) 与 svr_planB_pca(PCA策略)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/steel_v2_cleaned_r2.csv",
        help="数据文件路径",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="结果输出根目录",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="要训练的目标列，默认 fy_reduction fu_reduction E_reduction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_type == "svr":
        result = run_svr_experiment(
            data_path=args.data_path,
            output_root=args.output_root,
            targets=args.targets,
            random_state=args.random_state,
        )
    elif args.model_type == "svr_planB_pca":
        result = run_svr_planb_pca_experiment(
            data_path=args.data_path,
            output_root=args.output_root,
            targets=args.targets,
            random_state=args.random_state,
        )
    else:
        raise ValueError(f"不支持的 model_type: {args.model_type}")

    print("训练完成。")
    print(f"metrics: {result['metrics_path']}")
    print(f"predictions: {result['predictions_path']}")
    print(result["metrics_df"])


if __name__ == "__main__":
    main()
