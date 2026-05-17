"""Data loading and saving utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import geopandas as gpd
import pandas as pd
from loguru import logger
import json
from shapely.geometry import shape

from . import config


def load_roads(filepath: Path | None = None) -> gpd.GeoDataFrame:
    """Load the road network GeoJSON as a GeoDataFrame."""
    path = filepath or config.ROADS_FILE
    logger.debug("Loading roads from {}", path)
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        # Some Fiona/OGR builds fail parsing certain GeoJSON properties
        # (JSONField -> json.loads errors). Fall back to a safe loader
        # using the stdlib json + shapely geometry construction.
        logger.warning("gpd.read_file failed (%s); falling back to json loader", exc)
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        features = doc.get("features", [])
        props = [f.get("properties", {}) for f in features]
        geoms = [shape(f.get("geometry")) for f in features]
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
    if "segment_id" not in gdf.columns:
        # Guarantee a stable segment identifier.
        gdf["segment_id"] = gdf.index.astype(int)
    return gdf


def load_crash_history(filepath: Path | None = None) -> pd.DataFrame:
    """Load the crash history table."""
    path = filepath or config.RAW_CRASH_HISTORY_FILE
    logger.debug("Loading crash history from {}", path)
    df = pd.read_csv(path)
    expected_columns = {"segment_id", "year", "month", "crash_count"}
    missing = expected_columns.difference(df.columns)
    if missing:
        msg = f"Crash history missing columns: {sorted(missing)}"
        raise ValueError(msg)
    return df


def load_segment_features(filepath: Path | None = None) -> pd.DataFrame:
    """Load the engineered feature matrix for modelling."""
    path = filepath or config.PROCESSED_SEGMENT_FEATURES
    logger.debug("Loading processed features from {}", path)
    return pd.read_parquet(path)


def save_segment_features(df: pd.DataFrame, filepath: Path | None = None) -> None:
    """Persist engineered features to parquet."""
    path = filepath or config.PROCESSED_SEGMENT_FEATURES
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved segment features to {}", path)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved JSON to {}", filepath)


def load_json(filepath: Path) -> Dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarise_crash_history(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crash, weather, and traffic surrogates per segment."""
    grouped = df.groupby("segment_id")

    crash_stats = grouped["crash_count"].sum().rename("total_crashes")
    mean_weather = grouped[
        ["avg_daily_rain_mm", "avg_temp_c", "avg_visibility_km"]
    ].mean().rename(
        columns={
            "avg_daily_rain_mm": "mean_rain_mm",
            "avg_temp_c": "mean_temp_c",
            "avg_visibility_km": "mean_visibility_km",
        }
    )

    def _mode(series: pd.Series) -> Any:
        return series.mode().iloc[0] if not series.mode().empty else "clear"

    dominant_weather = grouped["weather_condition"].agg(_mode).rename("dominant_weather")

    summary = (
        crash_stats
        .to_frame()
        .join(mean_weather, how="left")
        .join(dominant_weather, how="left"))
    summary = summary.reset_index()
    return summary


def split_features_and_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def save_geojson(gdf: gpd.GeoDataFrame, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(filepath, driver="GeoJSON")
    logger.info("Saved GeoJSON to {}", filepath)
