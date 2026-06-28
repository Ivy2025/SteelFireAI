from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    metrics: Dict[str, float],
    output_path: str,
    split_name: str,
) -> None:
    _ensure_parent(output_path)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    min_val = float(np.min(np.concatenate([y_true, y_pred])))
    max_val = float(np.max(np.concatenate([y_true, y_pred])))
    x_line = np.linspace(min_val, max_val, 200)

    plt.figure(figsize=(7.5, 6.5))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolors="k", linewidths=0.3)
    plt.plot(x_line, x_line, "r-", linewidth=1.5, label="y = x")
    plt.plot(x_line, 1.2 * x_line, "--", color="gray", linewidth=1.2, label="+20%")
    plt.plot(x_line, 0.8 * x_line, "--", color="gray", linewidth=1.2, label="-20%")

    title = f"{target} ({split_name})\nTrue vs Predicted"
    plt.title(title)
    plt.xlabel("True value")
    plt.ylabel("Predicted value")

    text_lines = [
        f"R2: {metrics.get('R2', float('nan')):.4f}",
        f"MAE: {metrics.get('MAE', float('nan')):.4f}",
        f"RMSE: {metrics.get('RMSE', float('nan')):.4f}",
        f"P20: {metrics.get('P20', float('nan')):.4f}",
    ]
    plt.text(
        0.04,
        0.96,
        "\n".join(text_lines),
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    feature_importance: pd.Series,
    target: str,
    output_path: str,
) -> None:
    _ensure_parent(output_path)

    importance_sorted = feature_importance.sort_values(ascending=False)

    plt.figure(figsize=(8, 7))
    plt.barh(importance_sorted.index, importance_sorted.values, color="#4C72B0")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"{target} Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
