# Al Ahsa Traffic Crash Risk Prediction

End-to-end reference implementation for predicting crash likelihood on road segments in Al Ahsa, Saudi Arabia. The project covers data ingestion, feature engineering, model training, and an interactive web map that displays predicted crash percentages when a user clicks a road segment.

## Project Layout

```
├── data/
│   ├── roads_small.geojson            # Base road network (provided)
│   ├── raw/                           # External datasets (created on demand)
│   ├── intermediate/                  # Curated GIS/ML intermediate outputs
│   └── processed/                     # Model-ready datasets
├── models/                            # Serialized models and inference assets
├── notebooks/                         # Optional exploratory notebooks (empty by default)
├── scripts/
│   ├── generate_synthetic_crash_data.py
│   └── build_predictions_geojson.py
├── src/
│   ├── api.py                         # FastAPI app serving the map + REST endpoints
│   ├── config.py                      # Centralised path configuration
│   ├── data_io.py                     # IO helpers for roads, crash data, and models
│   ├── features.py                    # Feature engineering utilities
│   ├── models.py                      # Model loading/training helpers
│   ├── pipeline.py                    # End-to-end training + scoring pipeline
│   └── validation.py                  # Data validation helpers
├── web/
│   ├── index.html                     # Leaflet-based interactive map
│   ├── css/styles.css
│   ├── js/app.js
│   └── data/                          # Static data exports (optional)
└── requirements.txt
```

> **Note**: The repository ships with deterministic synthetic crash data generation utilities. Replace these with official crashes once you obtain access to authoritative datasets (e.g., GASTAT, Saudi Open Data Portal, Ministry of Transport).

## Quick Start

Run every command from the repository root.

1. **Create and activate a Python environment** (Python 3.10+ recommended):

   macOS / Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Generate synthetic crash history** (skip if you have real data):

   ```bash
   python scripts/generate_synthetic_crash_data.py
   ```

   This script reads `data/roads_small.geojson`, produces a reproducible crash history CSV under `data/raw/`, and materialises a segment-level training table under `data/processed/segment_crash_features.parquet`.

4. **Train the model & export predictions**:

   ```bash
   python -m src.pipeline train
   python scripts/build_predictions_geojson.py
   ```

   - `python -m src.pipeline train` fits a Gradient Boosting Regressor on engineered features and saves the model under `models/`. Run it as a module (`-m`), not as `python src/pipeline.py` - the package uses relative imports.
   - `build_predictions_geojson.py` uses the trained model to create a GeoJSON with crash percentages for each road segment (stored under `web/data/road_predictions.geojson`).

   > The trained model (`models/*.joblib`) and the exported predictions are gitignored, so a fresh clone must run these two steps. Until you do, the API falls back to a rule-based heuristic and logs `Model artifacts missing`. When the trained model is picked up you will see `Loaded trained model from ...` at startup.

5. **Launch the interactive map** (serves FastAPI backend + static Leaflet app):

   ```bash
   uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
   ```

   Visit `http://localhost:8000` to explore the map. The page is served by the API itself and calls it with relative URLs, so any port works. Clicking on a road segment reveals the predicted crash percentage and key metadata.
   Use the **Weather Overrides** panel to apply live rainfall, visibility, temperature, or categorical weather conditions (clear, rainy, dusty, foggy, storm). Adjustments are sent to the API as query parameters so the crash model re-scores segments instantly under the selected scenario.

## Working with Real Crash Data

- Replace the synthetic CSV placed in `data/raw/` with an official crash dataset (e.g., per-segment or per-intersection crash counts for Al Ahsa).
- Update or extend `scripts/generate_synthetic_crash_data.py` to handle your schema.
- Re-run the training pipeline to incorporate the new data.

## Re-training & Experimentation

The project is modular:

- **Feature Engineering**: `src/features.py` encapsulates feature derivations (road length, curvature, classification, historical crash density, traffic proxies, etc.).
- **Modelling**: `src/models.py` contains helper classes to train / load scikit-learn regressors.
- **Validation**: `src/validation.py` provides schema checks to ensure data integrity before training/prediction.

Use these modules to iterate quickly or integrate with ML experiment tracking solutions.

## Testing

Basic unit tests can be plugged via `pytest` (not included by default). Suggested future additions:

- Model evaluation assertions (R^2, RMSE thresholds).
- API contract tests (ensuring `/api/segments` returns valid GeoJSON).
- Front-end integration tests using Playwright.

## Deployment Notes

- Package the FastAPI service with Docker or deploy on Azure App Service, AWS Fargate, etc.
- Store large datasets & models in object storage (Azure Blob, S3) instead of git.
- Harden the API with caching/ratelimiting if exposed publicly.

## License & Data Sourcing

All synthetic assets in this repository are released under the MIT License. Replace synthetic data with officially licensed datasets aligned with your organisational policies.
