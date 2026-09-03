"""Score road segments and export a GeoJSON ready for the web map."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Allow running this file directly (python scripts/<name>.py) by putting the
# repository root on sys.path before importing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, pipeline


def main() -> None:
    config.ensure_directories()
    gdf = pipeline.score_to_geojson()
    logger.success("Exported {} segments to {}", len(gdf), config.PREDICTIONS_GEOJSON)


if __name__ == "__main__":
    main()
