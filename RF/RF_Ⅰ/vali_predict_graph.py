from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    # 以脚本所在目录为基准
    base_dir = Path(__file__).resolve().parent
    data_path = Path("results/validation_prediction.csv")
    output_dir = Path("results/figures/validation_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
   

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    # 读取数据
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="gbk")

    # 清理列名
    df.columns = df.columns.str.strip()

    required_cols = [
        "diff_steel",
        "temperature",
        "fy_reduction_true", "fy_reduction_pred",
        "fu_reduction_true", "fu_reduction_pred",
        "E_reduction_true", "E_reduction_pred",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    # 转换数值列
    numeric_cols = [
        "temperature",
        "fy_reduction_true", "fy_reduction_pred",
        "fu_reduction_true", "fu_reduction_pred",
        "E_reduction_true", "E_reduction_pred",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["diff_steel"] = df["diff_steel"].astype(str).str.strip()

    # 只保留温度 100~800 的数据
    df = df[(df["temperature"] >= 100) & (df["temperature"] <= 800)].copy()

    # 若你的数据温度不是整数，统一到整数便于排序显示
    df["temperature"] = df["temperature"].round().astype("Int64")

    # 每个 steel_ID 生成 3 张图：fy、fu、E
    targets = [
        ("fy_reduction_true", "fy_reduction_pred", "fy"),
        ("fu_reduction_true", "fu_reduction_pred", "fu"),
        ("E_reduction_true", "E_reduction_pred", "E"),
    ]

    diff_steels = df["diff_steel"].dropna().unique().tolist()

    if not diff_steels:
        raise ValueError("没有可用于绘图的 diff_steel 数据。")

    for diff_steel in diff_steels:
        steel_df = df[df["diff_steel"] == diff_steel].copy()
        steel_df = steel_df.sort_values("temperature")

        for true_col, pred_col, target_name in targets:
            plot_df = steel_df[["temperature", true_col, pred_col]].dropna().copy()

            if plot_df.empty:
                print(f"跳过 diff_steel={diff_steel}, target={target_name}: 无有效数据")
                continue

            plt.figure(figsize=(8, 5))

            plt.scatter(
                plot_df["temperature"],
                plot_df[true_col],
                marker="o",
                linewidth=0.7,
                label="True",
            )
            plt.scatter(
                plot_df["temperature"],
                plot_df[pred_col],
                marker="s",
                linewidth=0.7,
                label="Pred",
            )

            plt.xticks(range(100, 801, 100))
            # y轴：至少显示到1；若最大值超过1，则按0.1向上取整扩展
            y_max_data = max(plot_df[true_col].max(), plot_df[pred_col].max())
            y_upper = max(1.0, ((y_max_data * 10) // 1 + 1) / 10)
            plt.ylim(0, y_upper)
            plt.yticks([i / 10 for i in range(0, int(y_upper * 10) + 1)])
            plt.xlabel("Temperature (°C)", fontsize=12)
            plt.ylabel(target_name, fontsize=12)
            plt.title(f"diff_steel={diff_steel} | {target_name}: True vs Pred", fontsize=13)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            output_file = output_dir / f"diff_steel_{diff_steel}_{target_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close("all")

            print(f"已保存: {output_file}")

    print(f"全部完成，图片保存在: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
