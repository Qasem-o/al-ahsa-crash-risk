"""Generate deterministic synthetic crash data for Al Ahsa road segments."""
from __future__ import annotations

import hashlib
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import LineString

# Allow running this file directly (python scripts/<name>.py) by putting the
# repository root on sys.path before importing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io


def _seed_for_segment(segment_id: int) -> int:
    digest = hashlib.md5(str(segment_id).encode("ascii")).hexdigest()
    return int(digest[:8], 16)


def _sample_weather(month: int, base_risk: float) -> dict[str, float | str]:
    wet_months = {1, 2, 3, 11, 12}
    rain_bias = 1.3 if month in wet_months else 0.7
    rain_shape = 1.0 + base_risk * 0.4
    avg_daily_rain_mm = float(np.random.gamma(shape=rain_shape, scale=0.8) * rain_bias)

    visibility_base = 14 - avg_daily_rain_mm * 0.35 + np.random.normal(0, 1.5)
    avg_visibility_km = float(np.clip(visibility_base, 2.5, 18.0))

    temp_base = 38 - abs(month - 7) * 1.6 + np.random.normal(0, 1.8)
    avg_temp_c = float(np.clip(temp_base + base_risk * 0.2, 22, 46))

    if avg_daily_rain_mm > 6:
        weather_condition = "storm"
    elif avg_daily_rain_mm > 3:
        weather_condition = "rainy"
    elif avg_visibility_km < 6:
        weather_condition = "foggy"
    elif avg_daily_rain_mm < 0.8 and avg_visibility_km < 10:
        weather_condition = "dusty"
    else:
        weather_condition = "clear"

    return {
        "avg_daily_rain_mm": avg_daily_rain_mm,
        "avg_visibility_km": avg_visibility_km,
        "avg_temp_c": avg_temp_c,
        "weather_condition": weather_condition,
    }


def _simulate_monthly_crashes(segment_id: int, base_risk: float) -> pd.DataFrame:
    records = []
    years = range(2020, 2025)
    for year in years:
        for month in range(1, 13):
            seasonality = 1.0 + 0.2 * np.sin((month / 12) * 2 * np.pi)
            weather = _sample_weather(month, base_risk)
            visibility_penalty = max(0.0, 15 - weather["avg_visibility_km"]) * 0.04
            rain_multiplier = 1 + weather["avg_daily_rain_mm"] * 0.07
            weather_multiplier = 1 + visibility_penalty + rain_multiplier - 1
            lam = max(base_risk * seasonality * weather_multiplier, 0.05)
            crashes = np.random.poisson(lam)
            records.append(
                {
                    "segment_id": segment_id,
                    "year": year,
                    "month": month,
                    "crash_count": int(crashes),
                    "avg_daily_rain_mm": weather["avg_daily_rain_mm"],
                    "avg_temp_c": weather["avg_temp_c"],
                    "avg_visibility_km": weather["avg_visibility_km"],
                    "weather_condition": weather["weather_condition"],
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    config.ensure_directories()
    roads = data_io.load_roads()
    logger.info("Loaded {} road segments", len(roads))

    rows = []
    for _, road in roads.iterrows():
        segment_id = int(road["segment_id"])
        seed = _seed_for_segment(segment_id)
        random.seed(seed)
        np.random.seed(seed)

        highway = road.get("highway", "unknown") or "unknown"
        base = 0.2
        if highway == "primary":
            base += 1.2
        elif highway == "secondary":
            base += 0.8
        elif highway == "tertiary":
            base += 0.5
        else:
            base += 0.3

        geom: LineString = road.geometry  # type: ignore[assignment]
        length = float(road.get("length", geom.length))
        if geom.length == 0 or len(geom.coords) < 2:
            curvature_ratio = 1.0
        else:
            start, end = geom.coords[0], geom.coords[-1]
            chord = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
            curvature_ratio = geom.length / chord if chord else 1.0
        curvature_bonus = min(curvature_ratio, 5.0) * 0.1
        lanes = road.get("lanes")
        if lanes and str(lanes).isdigit():
            base += int(lanes) * 0.1

        risk = base + curvature_bonus + length / 1000
        rows.append((segment_id, risk))

    crash_frames = []
    for segment_id, risk in rows:
        crash_frames.append(_simulate_monthly_crashes(segment_id, risk))

    crash_history = pd.concat(crash_frames, ignore_index=True)
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    crash_history.to_csv(config.RAW_CRASH_HISTORY_FILE, index=False)
    logger.success("Wrote synthetic crash history to {}", config.RAW_CRASH_HISTORY_FILE)

    summary = data_io.summarise_crash_history(crash_history)
    summary.to_csv(config.INTERMEDIATE_DATA_DIR / "segment_crash_summary.csv", index=False)
    logger.info("Wrote crash summary to {}", config.INTERMEDIATE_DATA_DIR / "segment_crash_summary.csv")


if __name__ == "__main__":
    main()
