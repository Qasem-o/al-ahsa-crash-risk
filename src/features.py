"""Feature engineering utilities."""
from __future__ import annotations

import math
from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from loguru import logger

from . import config, data_io


ROAD_CATEGORICAL_COLUMNS = ["highway", "oneway", "lanes", "bridge", "tunnel"]
WEATHER_CATEGORICAL_COLUMNS = ["dominant_weather"]
CATEGORICAL_COLUMNS = ROAD_CATEGORICAL_COLUMNS + WEATHER_CATEGORICAL_COLUMNS

ROAD_NUMERIC_COLUMNS = [
    "curvature_ratio",
    "is_primary",
    "is_secondary",
    "is_tertiary",
    "length",
    "maxspeed",
]
WEATHER_NUMERIC_COLUMNS = ["mean_rain_mm", "mean_temp_c", "mean_visibility_km"]
NUMERIC_COLUMNS = ROAD_NUMERIC_COLUMNS + WEATHER_NUMERIC_COLUMNS
TARGET_COLUMN = "crash_rate_per_km"

MODEL_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS
MODEL_NUMERIC_COLUMNS = NUMERIC_COLUMNS
MODEL_FEATURE_COLUMNS = MODEL_CATEGORICAL_COLUMNS + MODEL_NUMERIC_COLUMNS


def _normalise_maxspeed(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.fillna(cleaned.median())


def _categorical_fill(series: pd.Series) -> pd.Series:
    return series.fillna("unknown")


def compute_curvature(geometry: gpd.GeoSeries) -> pd.Series:
    """Approximate curvature as length divided by chord length."""
    curvature_values = []
    for line in geometry:
        if line.length == 0:
            curvature_values.append(0.0)
            continue
        start, end = line.coords[0], line.coords[-1]
        chord = math.dist(start, end)
        if chord == 0:
            curvature_values.append(0.0)
        else:
            curvature_values.append(float(line.length / chord))
    return pd.Series(curvature_values, index=geometry.index)


def engineer_segment_table(
    roads: gpd.GeoDataFrame,
    crash_summary: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Create a modelling table by merging road attributes with crash statistics."""
    logger.debug("Engineering feature table with {} road segments", len(roads))

    gdf = roads.copy()
    gdf["segment_id"] = gdf["segment_id"].astype(int)

    for col in ROAD_CATEGORICAL_COLUMNS:
        if col in gdf.columns:
            gdf[col] = _categorical_fill(gdf[col])
        else:
            gdf[col] = "unknown"

    if "maxspeed" in gdf.columns:
        gdf["maxspeed"] = _normalise_maxspeed(gdf["maxspeed"])
    else:
        gdf["maxspeed"] = gdf["length"].clip(lower=20, upper=120)  # heuristic fallback

    gdf["curvature_ratio"] = compute_curvature(gdf.geometry)
    gdf["is_primary"] = (gdf["highway"] == "primary").astype(int)
    gdf["is_secondary"] = (gdf["highway"] == "secondary").astype(int)
    gdf["is_tertiary"] = (gdf["highway"] == "tertiary").astype(int)

    if "length" not in gdf.columns:
        gdf["length"] = gdf.geometry.length

    merged = gdf.merge(crash_summary, on="segment_id", how="left")
    merged["total_crashes"] = merged["total_crashes"].fillna(0.0)

    for col in WEATHER_NUMERIC_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(merged[col].median() if not merged[col].dropna().empty else 0.0)

    for col in WEATHER_CATEGORICAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = "clear"
        merged[col] = merged[col].fillna("clear")

    # Convert crash counts to a rate per kilometre as the modelling target.
    merged[TARGET_COLUMN] = merged["total_crashes"] / merged["length"].replace({0: 1}) * 1000
    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].fillna(0.0)

    return merged


def prepare_ml_dataset(table: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the modelling table into ML-ready features and target."""
    feature_cols: List[str] = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

    df = table[feature_cols + [TARGET_COLUMN, "segment_id"]].drop_duplicates("segment_id")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN, "segment_id"])
    return X, y
