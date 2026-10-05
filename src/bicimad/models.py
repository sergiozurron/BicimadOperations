"""Temporal split, metrics, and the three model families."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED
from .features import feature_matrix

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None
    XGBClassifier = None


PEAK_HOURS = {7, 8, 9, 10, 17, 18, 19, 20}


def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2022-08-31",
    val_end: str = "2022-10-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["hour"])
    if getattr(ts.dt, "tz", None) is not None:
        dates = ts.dt.tz_convert("Europe/Madrid").dt.date
    else:
        dates = ts.dt.date
    train_end_d = pd.Timestamp(train_end).date()
    val_end_d = pd.Timestamp(val_end).date()
    train = df.loc[dates <= train_end_d].copy()
    val = df.loc[(dates > train_end_d) & (dates <= val_end_d)].copy()
    test = df.loc[dates > val_end_d].copy()
    return train, val, test


def demo_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """For a single-month demo: split by time, never shuffle."""
    hours = np.sort(df["hour"].unique())
    n = len(hours)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train_h, val_h, test_h = set(hours[:i_train]), set(hours[i_train:i_val]), set(hours[i_val:])
    return (
        df[df["hour"].isin(train_h)].copy(),
        df[df["hour"].isin(val_h)].copy(),
        df[df["hour"].isin(test_h)].copy(),
    )


def peak_weights(df: pd.DataFrame) -> np.ndarray:
    hod = df["hour_of_day"] if "hour_of_day" in df.columns else pd.to_datetime(df["hour"]).dt.hour
    return np.where(hod.isin(PEAK_HOURS), 2.0, 1.0).astype(float)


def _xy(df: pd.DataFrame, target: str, cols: list[str] | None = None):
    cols = cols or feature_matrix(df)
    mask = df[target].notna()
    X = df.loc[mask, cols]
    y = df.loc[mask, target]
    return X, y, cols, mask


def regression_metrics(y_true, y_pred, sample_weight=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    wmae = (
        mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        if sample_weight is not None
        else mae
    )
    return {"MAE": float(mae), "RMSE": float(rmse), "weighted_MAE": float(wmae)}


def classification_metrics(y_true, y_pred, y_proba=None, average="macro") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    out = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    if y_proba is not None:
        try:
            # binary AP if 2-class proba
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                out["pr_auc"] = float(average_precision_score(y_true, y_proba.ravel()))
            elif y_proba.shape[1] == 2:
                out["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                out["pr_auc"] = float(
                    average_precision_score(y_true, y_proba, average="macro")
                )
        except Exception:
            out["pr_auc"] = np.nan
    return out


def make_rf_regressor(seed: int = RANDOM_SEED, n_estimators: int = 80) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )


def make_rf_classifier(seed: int = RANDOM_SEED, n_estimators: int = 80) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )


def make_xgb_regressor(seed: int = RANDOM_SEED):
    return XGBRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=seed,
        early_stopping_rounds=20,
    )


def make_xgb_classifier(seed: int = RANDOM_SEED, scale_pos_weight: float = 1.0):
    return XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20,
    )


def _pipe(estimator, scale: bool = True) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_ridge_regressor(seed: int = RANDOM_SEED) -> Pipeline:
    return _pipe(Ridge(alpha=1.0, random_state=seed))


def make_elasticnet_regressor(seed: int = RANDOM_SEED) -> Pipeline:
    return _pipe(
        SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=0.15,
            alpha=1e-4,
            max_iter=500,
            random_state=seed,
            early_stopping=False,
        )
    )


def make_extratrees_regressor(seed: int = RANDOM_SEED, n_estimators: int = 80) -> Pipeline:
    return _pipe(
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
        scale=False,
    )


def make_hgb_regressor(seed: int = RANDOM_SEED) -> Pipeline:
    # No internal early-stopping: sklearn's HGB would shuffle a validation split.
    return _pipe(
        HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.08,
            max_iter=150,
            l2_regularization=0.1,
            random_state=seed,
            early_stopping=False,
        ),
        scale=False,
    )


def make_linearsvr_regressor(seed: int = RANDOM_SEED) -> Pipeline:
    # Primal SGD is the scalable stand-in for LinearSVR on 100k+ rows.
    return _pipe(
        SGDRegressor(
            loss="epsilon_insensitive",
            penalty="l2",
            alpha=1e-4,
            max_iter=400,
            random_state=seed,
            early_stopping=False,
        )
    )


def make_logreg_classifier(seed: int = RANDOM_SEED) -> Pipeline:
    return _pipe(LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed))


def make_extratrees_classifier(seed: int = RANDOM_SEED, n_estimators: int = 80) -> Pipeline:
    return _pipe(
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        ),
        scale=False,
    )


def make_hgb_classifier(seed: int = RANDOM_SEED) -> Pipeline:
    return _pipe(
        HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=150,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
        ),
        scale=False,
    )


def make_linearsvc_classifier(seed: int = RANDOM_SEED) -> Pipeline:
    return _pipe(
        SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            max_iter=400,
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
        )
    )


def make_adaboost_classifier(seed: int = RANDOM_SEED) -> Pipeline:
    return _pipe(
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=40,
            class_weight="balanced",
            random_state=seed,
        ),
        scale=False,
    )


def scores_to_proba(scores) -> np.ndarray:
    """Map LinearSVC decision scores to a two-column pseudo-probability."""
    z = np.asarray(scores, dtype=float).ravel()
    p = 1.0 / (1.0 + np.exp(np.clip(-z, -50, 50)))
    return np.column_stack([1.0 - p, p])


def make_mlp_regressor(seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=40,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_mlp_classifier(seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=40,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def save_joblib(obj, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
