"""Run CrimeRateMLP inference from a saved best.pt checkpoint.

Accepts either already-encoded indices or raw geographic/temporal fields
(city name, H3 cell or lat/lng, month, weekday, hour), then returns the
four nonnegative per-hour crime rates produced by softplus + eps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from model.model import (
    RATE_NAMES,
    CrimeRateMLP,
    build_model,
    resolve_torch_device,
)
from time_bins import HOUR_BIN_STARTS, TIME_BIN_HOURS

try:
    import h3
except ImportError:  # pragma: no cover
    h3 = None  # type: ignore[assignment]


_SRC_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _SRC_DIR.parent

DEFAULT_CHECKPOINT = _SRC_DIR / "saved_weights" / "best.pt"
DEFAULT_METADATA_DIR = _PACKAGE_DIR / "data" / "crime_rate_metadata"
H3_RESOLUTION = 9
UNK_TOKEN = "<UNK>"
VALID_HOUR_BINS: frozenset[int] = frozenset(HOUR_BIN_STARTS)


@dataclass(frozen=True)
class RatePrediction:
    """Per-category hourly rates plus optional severity-weighted total."""

    person_rate: float
    property_rate: float
    society_rate: float
    other_rate: float
    total_rate: float
    severity_weighted_rate: float
    h3_cell: str | None
    h3_cell_index: int
    city_name: str | None
    city_index: int
    month: int
    day_of_week: int
    hour_bin_start: int
    used_unk_h3: bool
    used_unk_city: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "person_rate": self.person_rate,
            "property_rate": self.property_rate,
            "society_rate": self.society_rate,
            "other_rate": self.other_rate,
            "total_rate": self.total_rate,
            "severity_weighted_rate": self.severity_weighted_rate,
            "h3_cell": self.h3_cell,
            "h3_cell_index": self.h3_cell_index,
            "city_name": self.city_name,
            "city_index": self.city_index,
            "month": self.month,
            "day_of_week": self.day_of_week,
            "hour_bin_start": self.hour_bin_start,
            "used_unk_h3": self.used_unk_h3,
            "used_unk_city": self.used_unk_city,
            "rate_names": list(RATE_NAMES),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in raw.items()}


def normalize_city_name(city_name: str) -> str:
    return str(city_name).strip().lower()


def hour_to_bin_start(hour: int) -> int:
    hour_int = int(hour)
    if hour_int < 0 or hour_int > 23:
        raise ValueError(f"hour must be in 0..23, got {hour_int}")
    return (hour_int // TIME_BIN_HOURS) * TIME_BIN_HOURS


def latlng_to_h3_cell(latitude: float, longitude: float, resolution: int = H3_RESOLUTION) -> str:
    if h3 is None:
        raise ImportError("The 'h3' package is required for lat/lng inputs. Install with: pip install h3")
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(float(latitude), float(longitude), int(resolution)))
    return str(h3.geo_to_h3(float(latitude), float(longitude), int(resolution)))


def lookup_index(mapping: Mapping[str, int], key: str, *, kind: str) -> tuple[int, bool]:
    if key in mapping:
        return int(mapping[key]), False
    if UNK_TOKEN not in mapping:
        raise KeyError(f"{kind} vocabulary is missing {UNK_TOKEN!r}")
    return int(mapping[UNK_TOKEN]), True


class CrimeRateInference:
    """Load best.pt + vocabularies and score spatiotemporal queries."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        metadata_dir: str | Path = DEFAULT_METADATA_DIR,
        *,
        device: str | torch.device | None = None,
        verify_checksums: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.metadata_dir = Path(metadata_dir).expanduser().resolve()
        self.device = resolve_torch_device(device)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.metadata_dir.is_dir():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"{self.checkpoint_path} is not a CrimeRateMLP training checkpoint dictionary"
            )
        if "model_config" not in checkpoint:
            raise ValueError(
                f"{self.checkpoint_path} is missing model_config; expected a rate-model checkpoint"
            )

        self.checkpoint = checkpoint
        self.model_config: dict[str, Any] = dict(checkpoint["model_config"])
        self.category_names: tuple[str, ...] = tuple(
            checkpoint.get("category_names", ["person", "property", "society", "other"])
        )
        self.time_bin_hours = int(checkpoint.get("time_bin_hours", TIME_BIN_HOURS))
        if self.time_bin_hours != TIME_BIN_HOURS:
            raise ValueError(
                f"Checkpoint time_bin_hours={self.time_bin_hours} does not match "
                f"code TIME_BIN_HOURS={TIME_BIN_HOURS}"
            )

        self.city_to_index = load_json_mapping(self.metadata_dir / "city_to_index.json")
        self.h3_cell_to_index = load_json_mapping(self.metadata_dir / "h3_cell_to_index.json")
        self._validate_vocabularies()
        if verify_checksums:
            self._verify_checksums()

        expected_h3 = int(checkpoint.get("h3_vocabulary_size", self.model_config["num_h3_embeddings"]))
        expected_city = int(
            checkpoint.get("city_vocabulary_size", self.model_config["num_city_embeddings"])
        )
        if max(self.h3_cell_to_index.values()) + 1 != expected_h3:
            raise ValueError(
                "H3 vocabulary size does not match checkpoint: "
                f"{max(self.h3_cell_to_index.values()) + 1} vs {expected_h3}"
            )
        if max(self.city_to_index.values()) + 1 != expected_city:
            raise ValueError(
                "City vocabulary size does not match checkpoint: "
                f"{max(self.city_to_index.values()) + 1} vs {expected_city}"
            )

        self.model: CrimeRateMLP = build_model(
            num_h3_embeddings=int(self.model_config["num_h3_embeddings"]),
            num_city_embeddings=int(self.model_config["num_city_embeddings"]),
            h3_embedding_dim=int(self.model_config.get("h3_embedding_dim", 24)),
            city_embedding_dim=int(self.model_config.get("city_embedding_dim", 8)),
            month_embedding_dim=int(self.model_config.get("month_embedding_dim", 4)),
            weekday_embedding_dim=int(self.model_config.get("weekday_embedding_dim", 3)),
            time_bin_embedding_dim=int(self.model_config.get("time_bin_embedding_dim", 3)),
            hidden_dim=int(self.model_config.get("hidden_dim", 128)),
            num_blocks=int(self.model_config.get("num_blocks", 2)),
            dropout=float(self.model_config.get("dropout", 0.15)),
            device=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    def _validate_vocabularies(self) -> None:
        known_cities = sorted(name for name in self.city_to_index if name != UNK_TOKEN)
        known_h3 = sorted(name for name in self.h3_cell_to_index if name != UNK_TOKEN)
        rebuilt_cities = {UNK_TOKEN: 0, **{name: i for i, name in enumerate(known_cities, 1)}}
        rebuilt_h3 = {UNK_TOKEN: 0, **{name: i for i, name in enumerate(known_h3, 1)}}
        if rebuilt_cities != self.city_to_index:
            raise RuntimeError("city_to_index.json is not a deterministic sorted vocabulary")
        if rebuilt_h3 != self.h3_cell_to_index:
            raise RuntimeError("h3_cell_to_index.json is not a deterministic sorted vocabulary")

    def _verify_checksums(self) -> None:
        recorded = self.checkpoint.get("vocabulary_checksums")
        if not recorded:
            return
        files = {
            "city_to_index": self.metadata_dir / "city_to_index.json",
            "h3_cell_to_index": self.metadata_dir / "h3_cell_to_index.json",
            "index_to_city": self.metadata_dir / "index_to_city.json",
            "index_to_h3_cell": self.metadata_dir / "index_to_h3_cell.json",
        }
        for key, path in files.items():
            expected = recorded.get(key)
            if expected is None:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Missing vocabulary file required for checksum: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {key}: got {actual}, expected {expected}. "
                    f"File: {path}"
                )

    def encode(
        self,
        *,
        city_name: str | None = None,
        city_index: int | None = None,
        h3_cell: str | None = None,
        h3_cell_index: int | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        month: int,
        day_of_week: int,
        hour: int | None = None,
        hour_bin_start: int | None = None,
    ) -> dict[str, Any]:
        """Map a raw or partially encoded query to model indices."""
        month_int = int(month)
        dow_int = int(day_of_week)
        if month_int < 1 or month_int > 12:
            raise ValueError(f"month must be in 1..12, got {month_int}")
        if dow_int < 0 or dow_int > 6:
            raise ValueError(f"day_of_week must be in 0..6 (Mon=0), got {dow_int}")

        if hour_bin_start is None:
            if hour is None:
                raise ValueError("Provide hour (0..23) or hour_bin_start")
            bin_start = hour_to_bin_start(hour)
        else:
            bin_start = int(hour_bin_start)
            if bin_start not in VALID_HOUR_BINS:
                raise ValueError(
                    f"hour_bin_start must be one of {sorted(VALID_HOUR_BINS)}, got {bin_start}"
                )

        resolved_city_name: str | None = None
        used_unk_city = False
        if city_index is not None:
            city_idx = int(city_index)
            if city_idx < 0 or city_idx >= len(self.city_to_index):
                raise ValueError(f"city_index out of range: {city_idx}")
            used_unk_city = city_idx == int(self.city_to_index[UNK_TOKEN])
            if city_name is not None:
                resolved_city_name = normalize_city_name(city_name)
        elif city_name is not None:
            resolved_city_name = normalize_city_name(city_name)
            city_idx, used_unk_city = lookup_index(
                self.city_to_index, resolved_city_name, kind="city"
            )
        else:
            raise ValueError("Provide city_name or city_index")

        resolved_h3: str | None = None
        used_unk_h3 = False
        if h3_cell_index is not None:
            h3_idx = int(h3_cell_index)
            if h3_idx < 0 or h3_idx >= len(self.h3_cell_to_index):
                raise ValueError(f"h3_cell_index out of range: {h3_idx}")
            used_unk_h3 = h3_idx == int(self.h3_cell_to_index[UNK_TOKEN])
            if h3_cell is not None:
                resolved_h3 = str(h3_cell).strip()
        elif h3_cell is not None:
            resolved_h3 = str(h3_cell).strip()
            h3_idx, used_unk_h3 = lookup_index(
                self.h3_cell_to_index, resolved_h3, kind="h3"
            )
        elif latitude is not None and longitude is not None:
            resolved_h3 = latlng_to_h3_cell(latitude, longitude, H3_RESOLUTION)
            h3_idx, used_unk_h3 = lookup_index(
                self.h3_cell_to_index, resolved_h3, kind="h3"
            )
        else:
            raise ValueError("Provide h3_cell, h3_cell_index, or latitude+longitude")

        return {
            "h3_cell": resolved_h3,
            "h3_cell_index": h3_idx,
            "city_name": resolved_city_name,
            "city_index": city_idx,
            "month": month_int,
            "day_of_week": dow_int,
            "hour_bin_start": bin_start,
            "used_unk_h3": used_unk_h3,
            "used_unk_city": used_unk_city,
        }

    @torch.inference_mode()
    def predict_encoded(
        self,
        h3_cell_index: int | Sequence[int],
        city_index: int | Sequence[int],
        month: int | Sequence[int],
        day_of_week: int | Sequence[int],
        hour_bin_start: int | Sequence[int],
    ) -> torch.Tensor:
        """Forward pass on encoded tensors/scalars. Returns `[B, 4]` rates."""

        def as_1d(value: int | Sequence[int], name: str) -> torch.Tensor:
            if isinstance(value, (int,)):
                tensor = torch.tensor([int(value)], dtype=torch.long, device=self.device)
            else:
                tensor = torch.as_tensor(value, dtype=torch.long, device=self.device)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                if tensor.ndim != 1:
                    raise ValueError(f"{name} must be a scalar or 1-D sequence")
            return tensor

        h3_t = as_1d(h3_cell_index, "h3_cell_index")
        city_t = as_1d(city_index, "city_index")
        month_t = as_1d(month, "month")
        dow_t = as_1d(day_of_week, "day_of_week")
        hour_t = as_1d(hour_bin_start, "hour_bin_start")
        return self.model(h3_t, city_t, month_t, dow_t, hour_t)

    def predict(
        self,
        *,
        city_name: str | None = None,
        city_index: int | None = None,
        h3_cell: str | None = None,
        h3_cell_index: int | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        month: int,
        day_of_week: int,
        hour: int | None = None,
        hour_bin_start: int | None = None,
    ) -> RatePrediction:
        """Encode one query and return category rates plus totals."""
        encoded = self.encode(
            city_name=city_name,
            city_index=city_index,
            h3_cell=h3_cell,
            h3_cell_index=h3_cell_index,
            latitude=latitude,
            longitude=longitude,
            month=month,
            day_of_week=day_of_week,
            hour=hour,
            hour_bin_start=hour_bin_start,
        )
        rates = self.predict_encoded(
            encoded["h3_cell_index"],
            encoded["city_index"],
            encoded["month"],
            encoded["day_of_week"],
            encoded["hour_bin_start"],
        )
        row = rates[0]
        total = float(row.sum().item())
        weights = self.model.severity_weights.to(device=row.device, dtype=row.dtype)
        severity = float((row * weights).sum().item())
        return RatePrediction(
            person_rate=float(row[0].item()),
            property_rate=float(row[1].item()),
            society_rate=float(row[2].item()),
            other_rate=float(row[3].item()),
            total_rate=total,
            severity_weighted_rate=severity,
            h3_cell=encoded["h3_cell"],
            h3_cell_index=int(encoded["h3_cell_index"]),
            city_name=encoded["city_name"],
            city_index=int(encoded["city_index"]),
            month=int(encoded["month"]),
            day_of_week=int(encoded["day_of_week"]),
            hour_bin_start=int(encoded["hour_bin_start"]),
            used_unk_h3=bool(encoded["used_unk_h3"]),
            used_unk_city=bool(encoded["used_unk_city"]),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrimeRateMLP inference from best.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Vocabulary directory (default: {DEFAULT_METADATA_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Do not verify vocabulary SHA-256 against the checkpoint",
    )
    parser.add_argument("--city-name", type=str, default=None)
    parser.add_argument("--city-index", type=int, default=None)
    parser.add_argument("--h3-cell", type=str, default=None)
    parser.add_argument("--h3-cell-index", type=int, default=None)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--month", type=int, required=True, help="Calendar month 1..12")
    parser.add_argument(
        "--day-of-week",
        type=int,
        required=True,
        help="Weekday 0..6 with Monday=0",
    )
    parser.add_argument("--hour", type=int, default=None, help="Clock hour 0..23")
    parser.add_argument(
        "--hour-bin-start",
        type=int,
        default=None,
        help=f"One of {list(HOUR_BIN_STARTS)}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the prediction as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engine = CrimeRateInference(
        checkpoint_path=args.checkpoint,
        metadata_dir=args.metadata_dir,
        device=args.device,
        verify_checksums=not args.skip_checksums,
    )
    prediction = engine.predict(
        city_name=args.city_name,
        city_index=args.city_index,
        h3_cell=args.h3_cell,
        h3_cell_index=args.h3_cell_index,
        latitude=args.latitude,
        longitude=args.longitude,
        month=args.month,
        day_of_week=args.day_of_week,
        hour=args.hour,
        hour_bin_start=args.hour_bin_start,
    )
    if args.json:
        print(json.dumps(prediction.as_dict(), indent=2))
    else:
        print(
            f"city={prediction.city_name!r} ({prediction.city_index}) "
            f"h3={prediction.h3_cell!r} ({prediction.h3_cell_index}) "
            f"month={prediction.month} dow={prediction.day_of_week} "
            f"hour_bin={prediction.hour_bin_start}"
        )
        if prediction.used_unk_city or prediction.used_unk_h3:
            print(
                f"warning: used UNK "
                f"(city={prediction.used_unk_city}, h3={prediction.used_unk_h3})"
            )
        for name, value in zip(
            RATE_NAMES,
            (
                prediction.person_rate,
                prediction.property_rate,
                prediction.society_rate,
                prediction.other_rate,
            ),
        ):
            print(f"  {name}: {value:.8f}")
        print(f"  total_rate: {prediction.total_rate:.8f}")
        print(f"  severity_weighted_rate: {prediction.severity_weighted_rate:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
