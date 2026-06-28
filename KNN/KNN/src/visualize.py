from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_plot_name(prefix: str, target: str) -> str:
    if prefix:
        return f"{prefix}_{target}_test_scatter.png"
    return f"{target}_test_scatter.png"


def plot_test_scatter_with_reference_lines(
    df_target_pred: pd.DataFrame,
    target: str,
    out_file: Path,
) -> None:
    y_true = df_target_pred["y_true"].to_numpy(dtype=float)
    y_pred = df_target_pred["y_pred"].to_numpy(dtype=float)

    vmin = float(np.nanmin(np.concatenate([y_true, y_pred])))
    vmax = float(np.nanmax(np.concatenate([y_true, y_pred])))

    # 给坐标轴一点边距，避免线条贴边
    pad = max((vmax - vmin) * 0.05, 1e-6)
    x_min, x_max = vmin - pad, vmax + pad
    x_line = np.linspace(x_min, x_max, 200)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, s=28)

    # 45度线与 ±20% 参考线
    plt.plot(x_line, x_line, linestyle="--", linewidth=1.8, label="y = x")
    plt.plot(x_line, 1.2 * x_line, linestyle=":", linewidth=1.4, label="+20%")
    plt.plot(x_line, 0.8 * x_line, linestyle=":", linewidth=1.4, label="-20%")

    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"KNN Test Scatter - {target}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def save_test_plots(
    test_predictions: pd.DataFrame,
    figure_dir: Path,
    prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for target, g in test_predictions.groupby("target"):
        out_file = figure_dir / _build_plot_name(prefix, target)
        plot_test_scatter_with_reference_lines(g, target=target, out_file=out_file)
