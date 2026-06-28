from __future__ import annotations

import os
from typing import Sequence

import matplotlib

# Use a non-GUI backend to avoid tkinter/thread cleanup warnings in CLI runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    metrics_text: str,
    save_path: str,
) -> None:
    """Plot and save y_true vs y_pred scatter with y=x and +/-20% bands."""
    ensure_dir(os.path.dirname(save_path))

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_min = float(min(np.min(y_true), np.min(y_pred)))
    y_max = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (y_max - y_min + 1e-12)
    low, high = y_min - pad, y_max + pad

    x_line = np.linspace(low, high, 200)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolors="k", linewidths=0.3)
    plt.plot(x_line, x_line, linestyle="-", linewidth=1.8, label="y=x")
    plt.plot(x_line, 1.2 * x_line, linestyle="--", linewidth=1.2, label="+20%")
    plt.plot(x_line, 0.8 * x_line, linestyle="--", linewidth=1.2, label="-20%")

    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.legend(loc="best")

    plt.text(
        0.03,
        0.97,
        metrics_text,
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Plot and save sorted feature importance chart."""
    ensure_dir(os.path.dirname(save_path))

    feature_names = np.asarray(feature_names)
    importances = np.asarray(importances, dtype=float)

    order = np.argsort(importances)[::-1]
    sorted_features = feature_names[order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1])
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_decision_tree_structure(
    fitted_pipeline,
    feature_names: Sequence[str],
    title: str,
    save_path: str,
    max_depth: int = 3,
) -> None:
    """Plot and save decision tree structure from fitted pipeline."""
    ensure_dir(os.path.dirname(save_path))

    dt_model = fitted_pipeline.named_steps["dt"]

    plt.figure(figsize=(16, 9))
    plot_tree(
        dt_model,
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=9,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
