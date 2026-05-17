"""Model utilities for training and inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

DEFAULT_RISK_SCALE = 100.0


@dataclass
class ModelArtifacts:
    model_path: Path
    feature_columns_path: Path
    metadata_path: Path


class HeuristicCrashModel:
    """Simple rule-based fallback when a trained model is unavailable."""

    def __init__(self) -> None:
        logger.warning("Using heuristic crash model. Train a model for better accuracy.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base = 2.0 + 0.002 * df.get("length", 0)
        highway_bonus = df.get("is_primary", 0) * 3.0 + df.get("is_secondary", 0) * 1.5
        curvature_penalty = np.clip(df.get("curvature_ratio", 0) - 1, 0, None) * 4
        maxspeed_factor = 0.015 * df.get("maxspeed", 50)
        rain_factor = 0.4 * df.get("mean_rain_mm", 0)
        visibility_penalty = np.clip(15 - df.get("mean_visibility_km", 15), 0, None) * 0.6
        temperature_factor = 0.05 * np.clip(df.get("mean_temp_c", 30) - 32, -20, 20)

        weather_condition = df.get("dominant_weather", "clear")
        if isinstance(weather_condition, pd.Series):
            condition_adjustment = weather_condition.fillna("clear").map(
                {
                    "clear": 0.0,
                    "rainy": 6.0,
                    "foggy": 4.0,
                    "dusty": 5.0,
                    "storm": 7.0,
                }
            ).fillna(2.0)
        else:
            mapping = {
                "clear": 0.0,
                "rainy": 6.0,
                "foggy": 4.0,
                "dusty": 5.0,
                "storm": 7.0,
            }
            condition_adjustment = mapping.get(str(weather_condition), 2.0)

        risk = (
            base
            + highway_bonus
            + curvature_penalty
            + maxspeed_factor
            + rain_factor
            + visibility_penalty
            + temperature_factor
            + condition_adjustment
        )
        return np.clip(risk, 0, 100)


class CrashRiskModel:
    def __init__(self, pipeline: Pipeline, feature_columns: Iterable[str]) -> None:
        self.pipeline = pipeline
        self.feature_columns = list(feature_columns)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[self.feature_columns])

    def save(self, artifacts: ModelArtifacts) -> None:
        artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, artifacts.model_path)
        artifacts.feature_columns_path.write_text(json.dumps(self.feature_columns, indent=2), encoding="utf-8")
        logger.info("Saved model to {}", artifacts.model_path)

    @classmethod
    def load(cls, artifacts: ModelArtifacts) -> "CrashRiskModel":
        pipeline = joblib.load(artifacts.model_path)
        feature_columns = json.loads(artifacts.feature_columns_path.read_text(encoding="utf-8"))
        return cls(pipeline=pipeline, feature_columns=feature_columns)


def build_training_pipeline(categorical_cols: Iterable[str], numeric_cols: Iterable[str]) -> Pipeline:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, list(categorical_cols)),
            ("numeric", numeric_transformer, list(numeric_cols)),
        ]
    )

    model = GradientBoostingRegressor(random_state=config.DEFAULT_RANDOM_SEED)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def train_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    test_size: float = 0.2,
    random_state: int = config.DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_training_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(((y_test - y_pred) ** 2).mean())),
        "test_size": len(y_test),
    }
    logger.info("Model evaluation: {}", metrics)

    model = CrashRiskModel(pipeline=pipeline, feature_columns=list(features.columns))

    percentile_99 = float(np.percentile(target, 99)) if len(target) else DEFAULT_RISK_SCALE
    risk_scale = max(percentile_99, 1.0)
    logger.info("Derived risk scaling factor (99th percentile of target): {:.2f}", risk_scale)

    artifacts = ModelArtifacts(
        model_path=config.MODEL_ARTIFACT,
        feature_columns_path=config.MODEL_FEATURE_COLUMNS,
        metadata_path=config.MODEL_METADATA,
    )
    model.save(artifacts)
    metadata = {
        "metrics": metrics,
        "risk_scale": risk_scale,
    }
    config.MODEL_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model metadata to {}", config.MODEL_METADATA)

    return metrics


def load_model_or_heuristic() -> Any:
    artifacts = ModelArtifacts(
        model_path=config.MODEL_ARTIFACT,
        feature_columns_path=config.MODEL_FEATURE_COLUMNS,
        metadata_path=config.MODEL_METADATA,
    )
    if artifacts.model_path.exists() and artifacts.feature_columns_path.exists():
        logger.info("Loading trained model from disk")
    logger.warning("Model artifacts missing; falling back to heuristic model")
    return HeuristicCrashModel()


def load_metadata(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if default is None:
        default = {"metrics": {}, "risk_scale": DEFAULT_RISK_SCALE}
    if not config.MODEL_METADATA.exists():
        return default
    try:
        return json.loads(config.MODEL_METADATA.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read model metadata: {}", exc)
        return default


def rate_to_percent(values: Any, risk_scale: float) -> Any:
    scale = max(float(risk_scale or DEFAULT_RISK_SCALE), 1.0)
    array = np.asarray(values, dtype=float)
    percent = np.clip((array / scale) * 100.0, 0.0, 100.0)
    if np.isscalar(values):
        return float(percent)
    return percent
