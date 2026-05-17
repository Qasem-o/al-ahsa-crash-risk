"""Score road segments and export a GeoJSON ready for the web map."""
from __future__ import annotations

from loguru import logger

from src import config, pipeline


def main() -> None:
    config.ensure_directories()
    gdf = pipeline.score_to_geojson()
    logger.success("Exported {} segments to {}", len(gdf), config.PREDICTIONS_GEOJSON)


if __name__ == "__main__":
    main()
