from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent_dir(file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def plot_scatter_with_bands(
    y_true,
    y_pred,
    title: str,
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    _ensure_parent_dir(save_path)

    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    span = max_v - min_v
    pad = max(0.02 * span, 1e-3)
    x = np.linspace(min_v - pad, max_v + pad, 300)

    plt.figure(figsize=(7.5, 6.0))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")

    plt.plot(x, x, linestyle="-", linewidth=1.5, color="black", label="y=x")
    plt.plot(x, 1.2 * x, linestyle="--", linewidth=1.0, color="red", label="+20%")
    plt.plot(x, 0.8 * x, linestyle="--", linewidth=1.0, color="blue", label="-20%")

    txt = (
        f"R2: {metrics.get('R2', float('nan')):.4f}\n"
        f"MAE: {metrics.get('MAE', float('nan')):.4f}\n"
        f"P20: {metrics.get('P20', float('nan')):.4f}"
    )
    plt.text(
        0.03,
        0.97,
        txt,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_feature_importance(feature_names, importances, title: str, save_path: str) -> None:
    _ensure_parent_dir(save_path)

    importances = np.asarray(importances, dtype=float)
    feature_names = np.asarray(feature_names)

    order = np.argsort(importances)[::-1]
    sorted_names = feature_names[order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8.5, 6.5))
    y_pos = np.arange(len(sorted_names))
    plt.barh(y_pos, sorted_importances)
    plt.yticks(y_pos, sorted_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
