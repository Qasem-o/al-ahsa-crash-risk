"""Central configuration for file paths and environment settings."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"
WEB_DATA_DIR = WEB_DIR / "data"

ROADS_FILE = DATA_DIR / "roads_small.geojson"
RAW_CRASH_HISTORY_FILE = RAW_DATA_DIR / "al_ahsa_crash_history.csv"
PROCESSED_SEGMENT_FEATURES = PROCESSED_DATA_DIR / "segment_crash_features.parquet"
MODEL_ARTIFACT = MODELS_DIR / "crash_risk_model.joblib"
MODEL_FEATURE_COLUMNS = MODELS_DIR / "feature_columns.json"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"
PREDICTIONS_GEOJSON = WEB_DATA_DIR / "road_predictions.geojson"

# Default random seed for reproducibility when synthesising data or training.
DEFAULT_RANDOM_SEED = 42


def ensure_directories() -> None:
    """Create expected directories if they are missing."""
    for directory in (
        RAW_DATA_DIR,
        INTERMEDIATE_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        WEB_DATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
