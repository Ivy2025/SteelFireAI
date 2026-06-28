# src/evaluate.py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def p20_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.where(np.abs(y_true) < 1e-12, 1.0, np.abs(y_true))
    rel_err = np.abs((y_pred - y_true) / denom)
    return float(np.mean(rel_err <= 0.20))


def regression_metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        f"{prefix}_P20": p20_score(y_true, y_pred),
    }


def save_test_scatter_plot(y_true, y_pred, target: str, save_path, add_p20_band: bool = False) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    min_v = float(np.nanmin([y_true.min(), y_pred.min()]))
    max_v = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = (max_v - min_v) * 0.05 if max_v > min_v else 0.05

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.scatter(y_true, y_pred, s=26, alpha=0.75)
    ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], linestyle="--", linewidth=1.2)

    if add_p20_band:
        x = np.linspace(min_v - pad, max_v + pad, 200)
        ax.plot(x, 1.2 * x, linestyle=":", linewidth=1.2)
        ax.plot(x, 0.8 * x, linestyle=":", linewidth=1.2)

    ax.set_xlim(min_v - pad, max_v + pad)
    ax.set_ylim(min_v - pad, max_v + pad)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    suffix = " (+/-20%)" if add_p20_band else " (Perfect Line Only)"
    ax.set_title(f"SVR Test Scatter - {target}{suffix}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)