from pathlib import Path

import joblib
import numpy as np
import pandas as pd


TARGETS = [
    "fy_reduction",
    "fu_reduction",
    "E_reduction",
]


def load_model(models_dir: Path, target: str):
    model_path = models_dir / f"{target}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def get_feature_names(model, target: str):
    if not hasattr(model, "feature_names_"):
        raise ValueError(
            f"Model for {target} does not have feature_names_. "
            "Please retrain model with current training pipeline."
        )
    return list(model.feature_names_)


def prepare_feature_matrix(df: pd.DataFrame, feature_names):
    # Reindex to model features. Missing features are created as NaN and then filled with 0.
    x = df.reindex(columns=feature_names)
    x = x.apply(pd.to_numeric, errors="coerce")
    missing_after_numeric = x.isna().sum()
    if (missing_after_numeric > 0).any():
        x = x.fillna(0.0)
    return x.astype(float)


def compute_rel_error(y_true: pd.Series, y_pred: pd.Series):
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    rel = (y_pred_num - y_true_num) / y_true_num
    # Avoid inf caused by division by zero.
    rel = rel.replace([np.inf, -np.inf], np.nan)
    return rel


def build_statistics(pred_df: pd.DataFrame):
    rows = []
    has_temperature = "temperature" in pred_df.columns

    for target in TARGETS:
        rel_col = f"{target}_rel_error"
        if rel_col not in pred_df.columns:
            continue

        rel_all = pd.to_numeric(pred_df[rel_col], errors="coerce")
        valid_all = rel_all.dropna()
        all_n = int(valid_all.shape[0])
        all_bad = int((valid_all > 0.2).sum())
        all_pct = round(100.0 * all_bad / all_n, 4) if all_n > 0 else np.nan

        if has_temperature:
            temp = pd.to_numeric(pred_df["temperature"], errors="coerce")
            mask_lt800 = temp < 800
            rel_lt800 = rel_all[mask_lt800].dropna()
            lt800_n = int(rel_lt800.shape[0])
            lt800_bad = int((rel_lt800 > 0.2).sum())
            lt800_pct = round(100.0 * lt800_bad / lt800_n, 4) if lt800_n > 0 else np.nan
        else:
            lt800_n = np.nan
            lt800_bad = np.nan
            lt800_pct = np.nan

        rows.append(
            {
                "target": rel_col,
                "all_n": all_n,
                "all_gt_0p2_n": all_bad,
                "all_gt_0p2_pct": all_pct,
                "lt800_n": lt800_n,
                "lt800_gt_0p2_n": lt800_bad,
                "lt800_gt_0p2_pct": lt800_pct,
            }
        )

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent
    input_path = root / "data" / "validation_data.CSV"
    models_dir = root / "results" / "models"
    pred_output = root / "results" / "validation_prediction.csv"
    stats_output = root / "results" / "val_data_statistics.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Validation data not found: {input_path}")

    df = pd.read_csv(input_path)

    # Drop accidental unnamed helper columns from manual annotations.
    drop_cols = [c for c in df.columns if str(c).strip() == "" or str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    id_cols = [c for c in ["steel_ID", "diff_steel", "temperature"] if c in df.columns]
    pred_df = df[id_cols].copy()

    for target in TARGETS:
        model = load_model(models_dir, target)
        feature_names = get_feature_names(model, target)
        x = prepare_feature_matrix(df, feature_names)

        y_pred = model.predict(x)
        pred_col = f"{target}_pred"
        true_col = f"{target}_true"
        rel_col = f"{target}_rel_error"

        pred_df[pred_col] = pd.Series(y_pred, index=df.index).astype(float)

        if target in df.columns:
            pred_df[true_col] = pd.to_numeric(df[target], errors="coerce")
            pred_df[rel_col] = compute_rel_error(pred_df[true_col], pred_df[pred_col])
        else:
            pred_df[true_col] = np.nan
            pred_df[rel_col] = np.nan

    # Keep output column order consistent with existing file style.
    ordered_cols = [c for c in ["steel_ID", "diff_steel", "temperature"] if c in pred_df.columns]
    for t in TARGETS:
        ordered_cols.extend([f"{t}_true", f"{t}_pred", f"{t}_rel_error"])
    pred_df = pred_df.reindex(columns=ordered_cols)

    pred_output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_output, index=False, encoding="utf-8-sig")

    stats_df = build_statistics(pred_df)
    stats_df.to_csv(stats_output, index=False, encoding="utf-8-sig")

    print(f"Saved prediction file: {pred_output}")
    print(f"Saved statistics file: {stats_output}")


if __name__ == "__main__":
    main()
