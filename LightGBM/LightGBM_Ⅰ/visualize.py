from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _draw_reference_lines(ax, y_true: np.ndarray) -> None:
    y_min = float(np.nanmin(y_true))
    y_max = float(np.nanmax(y_true))
    low, high = min(y_min, y_max), max(y_min, y_max)

    x_line = np.linspace(low, high, 200)
    ax.plot(x_line, x_line, color="black", linestyle="-", linewidth=1.2, label="y=x")
    ax.plot(x_line, 1.2 * x_line, color="gray", linestyle="--", linewidth=1.0, label="+20%")
    ax.plot(x_line, 0.8 * x_line, color="gray", linestyle="--", linewidth=1.0, label="-20%")


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    title: str,
    save_path: str,
) -> None:
    ensure_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=28, alpha=0.75, color="tab:blue", edgecolor="none")
    _draw_reference_lines(ax, np.asarray(y_true, dtype=float))

    text = (
        f"R2={metrics.get('R2', float('nan')):.4f}\n"
        f"MAE={metrics.get('MAE', float('nan')):.4f}\n"
        f"P20={metrics.get('P20', float('nan')):.4f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, title: str, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))

    order = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[order]
    sorted_importance = np.array(importances)[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_features[::-1], sorted_importance[::-1], color="tab:green", alpha=0.85)
    ax.set_xlabel("importance")
    ax.set_ylabel("feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
