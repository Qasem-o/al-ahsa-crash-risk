"""FastAPI service exposing crash predictions and serving the web map."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from . import config, data_io, features, models

app = FastAPI(title="Al Ahsa Crash Risk Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://0.0.0.0:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    config.ensure_directories()
    _load_assets()


_cached_roads: Optional[gpd.GeoDataFrame] = None
_cached_features: Optional[pd.DataFrame] = None
_model: Any | None = None
_risk_scale: float = models.DEFAULT_RISK_SCALE


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for col in features.MODEL_CATEGORICAL_COLUMNS:
        default_value = "clear" if col == "dominant_weather" else "unknown"
        if col not in enriched.columns:
            enriched[col] = default_value
        enriched[col] = enriched[col].fillna(default_value)

    for col in features.MODEL_NUMERIC_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = 0.0
        enriched[col] = enriched[col].fillna(0.0)

    return enriched


def _apply_weather_overrides(
    df: pd.DataFrame,
    weather_condition: Optional[str],
    rain_mm: Optional[float],
    visibility_km: Optional[float],
    temperature_c: Optional[float],
) -> pd.DataFrame:
    adjusted = df.copy()
    if weather_condition:
        adjusted["dominant_weather"] = weather_condition.lower()
    if rain_mm is not None:
        adjusted["mean_rain_mm"] = rain_mm
    if visibility_km is not None:
        adjusted["mean_visibility_km"] = visibility_km
    if temperature_c is not None:
        adjusted["mean_temp_c"] = temperature_c
    return adjusted


def _select_model_features(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    feature_columns = getattr(model, "feature_columns", None)
    excluded = {features.TARGET_COLUMN, "segment_id"}
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in excluded]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        df = df.copy()
        for col in missing:
            if col in features.MODEL_CATEGORICAL_COLUMNS:
                default_value = "clear" if col == "dominant_weather" else "unknown"
                df[col] = default_value
            else:
                df[col] = 0.0
    return df[feature_columns]


def _load_assets() -> None:
    global _cached_roads, _cached_features, _model, _risk_scale
    if _cached_roads is None:
        _cached_roads = data_io.load_roads()
        if "segment_id" not in _cached_roads.columns:
            _cached_roads["segment_id"] = _cached_roads.index.astype(int)
    try:
        _cached_features = data_io.load_segment_features()
    except FileNotFoundError:
        logger.warning("Processed features missing. The API will synthesise heuristics on the fly.")
        roads = _cached_roads.copy()
        roads["curvature_ratio"] = features.compute_curvature(roads.geometry)
        roads["is_primary"] = (roads.get("highway") == "primary").astype(int)
        roads["is_secondary"] = (roads.get("highway") == "secondary").astype(int)
        roads["is_tertiary"] = (roads.get("highway") == "tertiary").astype(int)
        roads["length"] = roads.get("length", roads.geometry.length)
        roads["maxspeed"] = pd.to_numeric(roads.get("maxspeed"), errors="coerce").fillna(60)
        roads["highway"] = roads.get("highway", "unknown").fillna("unknown")
        roads["oneway"] = roads.get("oneway", "unknown").fillna("unknown")
        roads["lanes"] = roads.get("lanes", "unknown").fillna("unknown")
        roads["bridge"] = roads.get("bridge", "unknown").fillna("unknown")
        roads["tunnel"] = roads.get("tunnel", "unknown").fillna("unknown")
        roads["dominant_weather"] = "clear"
        roads["mean_rain_mm"] = 0.0
        roads["mean_temp_c"] = 34.0
        roads["mean_visibility_km"] = 12.0
        roads = roads[
            [
                "segment_id",
                "highway",
                "oneway",
                "lanes",
                "bridge",
                "tunnel",
                "curvature_ratio",
                "is_primary",
                "is_secondary",
                "is_tertiary",
                "length",
                "maxspeed",
                "dominant_weather",
                "mean_rain_mm",
                "mean_temp_c",
                "mean_visibility_km",
            ]
        ]
        _cached_features = roads
    _cached_features = _ensure_feature_columns(_cached_features)
    _model = models.load_model_or_heuristic()
    metadata = models.load_metadata()
    _risk_scale = float(metadata.get("risk_scale", models.DEFAULT_RISK_SCALE))


@app.get("/api/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/segments.geojson")
def get_segments(
    min_percent: float = Query(0.0, ge=0.0, le=100.0),
    weather_condition: Optional[str] = Query(None, description="Override dominant weather condition"),
    rain_mm: Optional[float] = Query(None, ge=0.0, description="Average daily rainfall in mm"),
    visibility_km: Optional[float] = Query(None, ge=0.0, description="Average visibility in km"),
    temperature_c: Optional[float] = Query(None, description="Average temperature in °C"),
) -> JSONResponse:
    assert _cached_roads is not None
    assert _cached_features is not None
    working_features = _apply_weather_overrides(
        _cached_features,
        weather_condition,
        rain_mm,
        visibility_km,
        temperature_c,
    )
    working_features = _ensure_feature_columns(working_features)

    prediction_input = _select_model_features(working_features, _model)
    predictions = _model.predict(prediction_input)
    predicted_percent = models.rate_to_percent(predictions, _risk_scale)

    working_features = working_features.assign(predicted_crash_percent=predicted_percent)

    geo = _cached_roads.merge(
        working_features[["segment_id", "predicted_crash_percent"]],
        on="segment_id",
        how="left",
    )
    geo = geo[geo["predicted_crash_percent"] >= min_percent]

    payload = json.loads(geo.to_json())
    return JSONResponse(content=payload)


@app.get("/api/segments/{segment_id}")
def get_segment(
    segment_id: int,
    weather_condition: Optional[str] = Query(None, description="Override dominant weather condition"),
    rain_mm: Optional[float] = Query(None, ge=0.0, description="Average daily rainfall in mm"),
    visibility_km: Optional[float] = Query(None, ge=0.0, description="Average visibility in km"),
    temperature_c: Optional[float] = Query(None, description="Average temperature in °C"),
) -> Dict[str, Any]:
    assert _cached_roads is not None
    assert _cached_features is not None

    row = _cached_features[_cached_features["segment_id"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Segment not found")

    row = _apply_weather_overrides(row, weather_condition, rain_mm, visibility_km, temperature_c)
    row = _ensure_feature_columns(row)
    prediction_input = _select_model_features(row, _model)
    prediction_rate = float(_model.predict(prediction_input)[0])
    prediction_percent = models.rate_to_percent(prediction_rate, _risk_scale)

    road_row = _cached_roads[_cached_roads["segment_id"] == segment_id].iloc[0]
    return {
        "segment_id": int(segment_id),
        "predicted_crash_percent": float(prediction_percent),
        "predicted_crash_rate_per_km": float(prediction_rate),
        "highway": road_row.get("highway"),
        "name": road_row.get("name"),
        "length_m": float(road_row.get("length", 0.0)),
        "maxspeed": road_row.get("maxspeed"),
    }


# Serve the static Leaflet application.
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="static")
