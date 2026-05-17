"""Lightweight validation utilities."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger


def validate_roads_schema(roads: gpd.GeoDataFrame) -> None:
    required_columns = {"segment_id", "geometry"}
    missing = required_columns.difference(roads.columns)
    if missing:
        raise ValueError(f"Roads GeoDataFrame missing columns: {sorted(missing)}")
    if roads.geometry.is_empty.any():
        raise ValueError("Road geometries contain empty shapes.")
    logger.debug("Roads schema validation passed for {} segments", len(roads))


def validate_crash_history_schema(crashes: pd.DataFrame) -> None:
    required_columns = {
        "segment_id",
        "year",
        "month",
        "crash_count",
        "avg_daily_rain_mm",
        "avg_temp_c",
        "avg_visibility_km",
        "weather_condition",
    }
    missing = required_columns.difference(crashes.columns)
    if missing:
        raise ValueError(f"Crash history missing columns: {sorted(missing)}")
    if (crashes["crash_count"] < 0).any():
        raise ValueError("Crash counts must be non-negative.")
    logger.debug("Crash history validation passed for {} records", len(crashes))
