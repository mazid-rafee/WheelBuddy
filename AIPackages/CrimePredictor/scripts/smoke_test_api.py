"""Smoke checks for CrimePredictor local inference API."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def encode_polyline(coordinates: list[tuple[float, float]]) -> str:
    """Encode (lat, lng) points as a Google polyline string."""

    def write(value: int, output: list[str]) -> None:
        value = ~(value << 1) if value < 0 else (value << 1)
        while value >= 0x20:
            output.append(chr((0x20 | (value & 0x1F)) + 63))
            value >>= 5
        output.append(chr(value + 63))

    result: list[str] = []
    prev_lat = 0
    prev_lng = 0
    for lat, lng in coordinates:
        lat_i = int(round(lat * 1e5))
        lng_i = int(round(lng * 1e5))
        write(lat_i - prev_lat, result)
        write(lng_i - prev_lng, result)
        prev_lat = lat_i
        prev_lng = lng_i
    return "".join(result)


def austin_mock_request() -> dict:
    """Short mock route that stays inside a known training H3 cell in Austin."""
    try:
        import h3
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("h3 is required for the smoke-test fixture") from exc

    cell = "8948985a4d3ffff"
    if hasattr(h3, "cell_to_latlng"):
        lat, lng = h3.cell_to_latlng(cell)
    else:
        lat, lng = h3.h3_to_geo(cell)

    # Keep offsets tiny so densification does not leave the training cell.
    points = [
        (lat, lng),
        (lat + 0.00004, lng + 0.00003),
        (lat + 0.00008, lng + 0.00005),
    ]
    for point in points:
        if hasattr(h3, "latlng_to_cell"):
            resolved = str(h3.latlng_to_cell(point[0], point[1], 9))
        else:
            resolved = str(h3.geo_to_h3(point[0], point[1], 9))
        if resolved != cell:
            raise RuntimeError(
                f"smoke fixture point left training cell {cell} → {resolved}"
            )

    polyline = encode_polyline(points)
    return {
        "request_id": "11111111-1111-1111-1111-111111111111",
        "routes": [
            {
                "route_id": "route_0",
                "departure_time_utc": "2026-09-20T00:35:00Z",
                "duration_seconds": 600,
                "steps": [
                    {
                        "encoded_polyline": polyline,
                        "distance_meters": 20,
                        "static_duration_seconds": 600,
                    }
                ],
            }
        ],
    }


def run_local_model_smoke() -> dict:
    from api.service import load_runtime_artifacts, smoke_validation_example

    artifacts = load_runtime_artifacts(device="cpu")
    result = smoke_validation_example(artifacts)
    for key in (
        "person_rate",
        "property_rate",
        "society_rate",
        "other_rate",
        "total_rate",
        "severity_weighted_rate",
    ):
        value = float(result[key])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite {key}: {value}")
    print("local model smoke OK")
    print(json.dumps(result, indent=2))
    return result


def run_http_smoke(base_url: str) -> None:
    health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=30.0)
    print("GET /health", health.status_code, health.text)
    health.raise_for_status()

    payload = austin_mock_request()
    fixture_path = PACKAGE_ROOT / "api" / "fixtures" / "mock_predict_routes.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    response = httpx.post(
        f"{base_url.rstrip('/')}/predict-routes",
        json=payload,
        timeout=120.0,
    )
    print("POST /predict-routes", response.status_code)
    print(response.text)
    response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="CrimePredictor API smoke tests")
    parser.add_argument(
        "--mode",
        choices=("local", "http", "all"),
        default="local",
        help="local=shared loader only; http=requires a running server",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for HTTP smoke checks",
    )
    args = parser.parse_args()

    if args.mode in {"local", "all"}:
        t0 = time.perf_counter()
        run_local_model_smoke()
        print(f"local smoke seconds={time.perf_counter() - t0:.1f}")
    if args.mode in {"http", "all"}:
        run_http_smoke(args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
