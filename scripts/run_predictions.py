"""Run predictions using the trained model, bypassing fiona/geopandas read_file.
This script loads the GeoJSON with the stdlib json module and builds geometries with shapely.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

# Allow running this file directly (python scripts/<name>.py) by putting the
# repository root on sys.path before importing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    risk_scale = models.resolve_risk_scale(model, metadata)

    feature_columns = getattr(model, "feature_columns", None)
    if feature_columns is None:
        excluded = {features.TARGET_COLUMN, "segment_id"}
        feature_columns = [col for col in feature_df.columns if col not in excluded]

    predictions = model.predict(feature_df[feature_columns])

    # Join on segment_id, not row position - see pipeline.score_to_geojson.
    scored = pd.DataFrame(
        {
            "segment_id": feature_df["segment_id"].to_numpy(),
            "predicted_crash_rate_per_km": np.asarray(predictions, dtype=float),
        }
    )
    scored["predicted_crash_percent"] = models.rate_to_percent(
        scored["predicted_crash_rate_per_km"].to_numpy(), risk_scale
    )
    result = roads.merge(scored, on="segment_id", how="left")

    data_io.save_geojson(result, config.PREDICTIONS_GEOJSON)
    print("Exported", len(result), "segments to", config.PREDICTIONS_GEOJSON)


if __name__ == "__main__":
    main()
