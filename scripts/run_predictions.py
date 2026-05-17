"""Run predictions using the trained model, bypassing fiona/geopandas read_file.
This script loads the GeoJSON with the stdlib json module and builds geometries with shapely.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape

from src import config, data_io, models, features


def main() -> None:
    config.ensure_directories()

    # Load roads via json + shapely to avoid fiona parsing issues
    with config.ROADS_FILE.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    feats = doc.get("features", [])
    if not feats:
        raise SystemExit("No features found in roads geojson")

    props = [f.get("properties", {}) for f in feats]
    geoms = [shape(f.get("geometry")) for f in feats]

    roads = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
    if "segment_id" not in roads.columns:
        roads["segment_id"] = roads.index.astype(int)

    feature_df = data_io.load_segment_features()

    model = models.load_model_or_heuristic()
    metadata = models.load_metadata()
    risk_scale = float(metadata.get("risk_scale", models.DEFAULT_RISK_SCALE))

    feature_columns = getattr(model, "feature_columns", None)
    if feature_columns is None:
        excluded = {features.TARGET_COLUMN, "segment_id"}
        feature_columns = [col for col in feature_df.columns if col not in excluded]

    predictions = model.predict(feature_df[feature_columns])

    result = roads.merge(
        feature_df[["segment_id"]],
        on="segment_id",
        how="left",
    )
    result["predicted_crash_rate_per_km"] = predictions
    result["predicted_crash_percent"] = models.rate_to_percent(predictions, risk_scale)

    data_io.save_geojson(result, config.PREDICTIONS_GEOJSON)
    print("Exported", len(result), "segments to", config.PREDICTIONS_GEOJSON)


if __name__ == "__main__":
    main()
