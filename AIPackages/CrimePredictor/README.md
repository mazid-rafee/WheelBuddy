# CrimePredictor local inference API

## Setup

```bash
cd /Users/zaimazarnaz/Desktop/IOSApps/WheelBuddy/AIPackages/CrimePredictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Required artifacts (already expected under this package):

- `src/saved_weights/best.pt`
- `data/crime_rate_metadata/{city_to_index,h3_cell_to_index,index_to_city,index_to_h3_cell}.json`
- `data/crime_rate_dataset.parquet` (used once to build/cache `h3_cell_to_city.json`)

Optional env overrides:

- `CRIME_MODEL_PATH` — checkpoint file
- `CRIME_METADATA_DIR` — vocabulary directory
- `CRIME_PARQUET_PATH` — training parquet for H3→city mapping

## Launch server

```bash
cd /Users/zaimazarnaz/Desktop/IOSApps/WheelBuddy/AIPackages/CrimePredictor
source .venv/bin/activate
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Health check

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

## Predict routes

```bash
curl -s http://127.0.0.1:8000/predict-routes \
  -H 'Content-Type: application/json' \
  -d @api/fixtures/mock_predict_routes.json | python -m json.tool
```

Generate the fixture (and run loader smoke) with:

```bash
python scripts/smoke_test_api.py --mode local
# with server running:
python scripts/smoke_test_api.py --mode http
```

## Notes

- Model outputs nonnegative **per-hour crime rates** via `softplus + eps` (not probabilities).
- Route summaries aggregate `severity_weighted_rate` (weights `[4.0, 1.5, 2.0, 1.0]`).
- H3 resolution is **9**; temporal bins are **3-hour** starts `{0,3,6,9,12,15,18,21}`.
- Weekday uses Monday=0; month is 1..12; city names are lowercase with spaces (`new york`).
