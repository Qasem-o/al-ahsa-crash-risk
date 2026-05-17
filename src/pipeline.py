"""CLI-friendly pipeline orchestration."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from . import config, data_io, features, models, validation


def _select_model_features(feature_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    feature_columns = getattr(model, "feature_columns", None)
    if feature_columns is None:
        excluded = {features.TARGET_COLUMN, "segment_id"}
        feature_columns = [col for col in feature_df.columns if col not in excluded]
    return feature_df[feature_columns]


def build_feature_matrix() -> pd.DataFrame:
    config.ensure_directories()
    roads = data_io.load_roads()
    validation.validate_roads_schema(roads)

    crash_history = data_io.load_crash_history()
    validation.validate_crash_history_schema(crash_history)

    crash_summary = data_io.summarise_crash_history(crash_history)
    table = features.engineer_segment_table(roads, crash_summary)
    X, y = features.prepare_ml_dataset(table)

    training_df = X.copy()
    training_df[features.TARGET_COLUMN] = y
    training_df["segment_id"] = table["segment_id"].values
    data_io.save_segment_features(training_df)
    return training_df


def train_model() -> None:
    training_df = build_feature_matrix()
    X, y = data_io.split_features_and_target(training_df, features.TARGET_COLUMN)

    metrics = models.train_and_evaluate(
        features=X,
        target=y,
        categorical_cols=features.MODEL_CATEGORICAL_COLUMNS,
        numeric_cols=features.MODEL_NUMERIC_COLUMNS,
    )
    logger.success("Training complete. Metrics: {}", metrics)


def score_to_geojson() -> gpd.GeoDataFrame:
    roads = data_io.load_roads()
    feature_df = data_io.load_segment_features()
    model = models.load_model_or_heuristic()
    metadata = models.load_metadata()
    risk_scale = float(metadata.get("risk_scale", models.DEFAULT_RISK_SCALE))

    prediction_input = _select_model_features(feature_df, model)
    predictions = model.predict(prediction_input)

    result = roads.merge(
        feature_df[["segment_id"]],
        on="segment_id",
        how="left",
    )
    result["predicted_crash_rate_per_km"] = predictions
    result["predicted_crash_percent"] = models.rate_to_percent(predictions, risk_scale)
    data_io.save_geojson(result, config.PREDICTIONS_GEOJSON)
    return result


def run(action: Literal["train", "score"]) -> None:
    if action == "train":
        train_model()
    elif action == "score":
        score_to_geojson()
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m src.pipeline [train|score]")
        sys.exit(1)
    run(sys.argv[1])
