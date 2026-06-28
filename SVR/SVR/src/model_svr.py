# src/model_svr.py
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR


def build_svr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR()),
        ]
    )


def get_svr_param_grid() -> dict:
    return {
        "svr__kernel": ["rbf"],
        "svr__C": [1, 10, 50, 100],
        "svr__epsilon": [0.01, 0.03, 0.05, 0.1],
        "svr__gamma": ["scale", 0.01, 0.05, 0.1],
    }


def get_svr_linear_param_grid() -> dict:
    # 可选扩展，不影响主实验
    return {
        "svr__kernel": ["linear"],
        "svr__C": [0.1, 1, 10, 50, 100],
        "svr__epsilon": [0.01, 0.03, 0.05, 0.1],
    }


def build_svr_pca_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("svr", SVR()),
        ]
    )


def get_svr_pca_param_grid() -> dict:
    return {
        "pca__n_components": [0.95, 0.99, 10, 15],
        "svr__kernel": ["rbf"],
        "svr__C": [1, 10, 50, 100],
        "svr__epsilon": [0.01, 0.03, 0.05, 0.1],
        "svr__gamma": ["scale", 0.01, 0.05, 0.1],
    }