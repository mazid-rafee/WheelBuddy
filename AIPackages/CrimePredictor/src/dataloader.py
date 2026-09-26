"""Streaming PyTorch DataLoader for categorical crime-rate Parquet data."""

from __future__ import annotations

import os
import random
import tempfile
import zlib
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


from time_bins import HOUR_BIN_STARTS, TIME_BIN_HOURS

DEFAULT_PARQUET_PATH = Path(
    "/data/quantization/zaima/crimecode/Data/crime_rate_dataset.parquet"
)

COUNT_COLUMNS: tuple[str, ...] = (
    "person_count",
    "property_count",
    "society_count",
    "other_count",
)

VALID_HOUR_BINS: frozenset[int] = frozenset(HOUR_BIN_STARTS)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "city_name",
    "h3_cell",
    "city_index",
    "h3_cell_index",
    "month",
    "day_of_week",
    "hour_bin_start",
    "exposure_hours",
    *COUNT_COLUMNS,
    "total_count",
)

SplitName = Literal["train", "val"] | None


def make_record_id(
    city_index: int,
    h3_cell_index: int,
    month: int,
    day_of_week: int,
    hour_bin_start: int,
) -> str:
    """Stable cross-process record identifier."""
    return (
        f"{int(city_index)}|{int(h3_cell_index)}|{int(month)}|"
        f"{int(day_of_week)}|{int(hour_bin_start)}"
    )


def validation_assignment(record_id: str, seed: int, val_fraction: float) -> bool:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    key = f"{int(seed)}:{record_id}".encode("utf-8")
    random_key = zlib.crc32(key) & 0xFFFFFFFF
    return random_key < int(val_fraction * (2**32))


def _valid_mask(
    city_index: np.ndarray,
    h3_index: np.ndarray,
    month: np.ndarray,
    day_of_week: np.ndarray,
    hour_bin: np.ndarray,
    exposure: np.ndarray,
    counts: np.ndarray,
    total: np.ndarray,
) -> np.ndarray:
    keep = np.ones(city_index.shape[0], dtype=bool)
    keep &= city_index >= 1
    keep &= h3_index >= 1
    keep &= (month >= 1) & (month <= 12)
    keep &= (day_of_week >= 0) & (day_of_week <= 6)
    keep &= np.isin(hour_bin, list(VALID_HOUR_BINS))
    keep &= np.isfinite(exposure) & (exposure > 0)
    keep &= np.isfinite(counts).all(axis=1) & (counts >= 0).all(axis=1)
    keep &= np.isfinite(total)
    keep &= np.isclose(total, counts.sum(axis=1), rtol=0.0, atol=0.0)
    return keep


class CrimeRateIterableDataset(IterableDataset):
    """Stream categorical crime-rate Parquet batches with CRC32 train/val split."""

    def __init__(
        self,
        parquet_path: str | os.PathLike[str] = DEFAULT_PARQUET_PATH,
        *,
        split: SplitName = None,
        val_fraction: float = 0.20,
        seed: int = 42,
        batch_rows: int = 65_536,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.parquet_path = Path(parquet_path).expanduser()
        self.split = split
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.batch_rows = int(batch_rows)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"Crime-rate Parquet not found: {self.parquet_path}")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be greater than zero")
        if self.split not in {None, "train", "val"}:
            raise ValueError("split must be None, 'train', or 'val'")
        if self.split is not None and not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")

        schema_names = set(pq.read_schema(self.parquet_path).names)
        missing = [name for name in REQUIRED_COLUMNS if name not in schema_names]
        if missing:
            raise ValueError(
                f"{self.parquet_path} is missing required columns: {missing}"
            )
        if self.split == "val":
            self.shuffle = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers

        parquet_file = pq.ParquetFile(self.parquet_path)
        for batch_index, record_batch in enumerate(
            parquet_file.iter_batches(
                batch_size=self.batch_rows,
                columns=list(REQUIRED_COLUMNS),
            )
        ):
            if batch_index % worker_count != worker_id:
                continue
            yield from self._iter_record_batch(record_batch, batch_index)

    def _iter_record_batch(
        self,
        record_batch: pa.RecordBatch,
        batch_index: int,
    ) -> Iterator[dict[str, object]]:
        table = pa.Table.from_batches([record_batch])
        city_index = np.asarray(
            table["city_index"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        h3_index = np.asarray(
            table["h3_cell_index"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        month = np.asarray(table["month"].to_numpy(zero_copy_only=False), dtype=np.int64)
        day_of_week = np.asarray(
            table["day_of_week"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        hour_bin = np.asarray(
            table["hour_bin_start"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        exposure = np.asarray(
            table["exposure_hours"].to_numpy(zero_copy_only=False), dtype=np.float32
        )
        counts = np.column_stack(
            [
                np.asarray(
                    table[name].to_numpy(zero_copy_only=False), dtype=np.float32
                )
                for name in COUNT_COLUMNS
            ]
        )
        total = np.asarray(
            table["total_count"].to_numpy(zero_copy_only=False), dtype=np.float64
        )
        cities = [
            "" if value is None else str(value)
            for value in table["city_name"].to_pylist()
        ]
        cells = [
            "" if value is None else str(value)
            for value in table["h3_cell"].to_pylist()
        ]

        keep = _valid_mask(
            city_index, h3_index, month, day_of_week, hour_bin, exposure, counts, total
        )
        if not np.any(keep):
            return
        indices = np.flatnonzero(keep)

        record_ids = [
            make_record_id(
                int(city_index[i]),
                int(h3_index[i]),
                int(month[i]),
                int(day_of_week[i]),
                int(hour_bin[i]),
            )
            for i in indices
        ]

        if self.split is not None:
            split_keep = []
            for local_pos, record_id in enumerate(record_ids):
                is_val = validation_assignment(
                    record_id, self.seed, self.val_fraction
                )
                if is_val == (self.split == "val"):
                    split_keep.append(local_pos)
            if not split_keep:
                return
            selected = indices[np.asarray(split_keep, dtype=np.int64)]
            record_ids = [record_ids[pos] for pos in split_keep]
        else:
            selected = indices

        if self.shuffle and selected.size > 1:
            rng = np.random.default_rng(
                self.seed + self.epoch * 1_000_003 + batch_index * 9176
            )
            order = rng.permutation(selected.size)
            selected = selected[order]
            record_ids = [record_ids[pos] for pos in order]

        for row_pos, row_index in enumerate(selected.tolist()):
            yield {
                "h3_cell_index": torch.tensor(
                    int(h3_index[row_index]), dtype=torch.long
                ),
                "city_index": torch.tensor(
                    int(city_index[row_index]), dtype=torch.long
                ),
                "month": torch.tensor(int(month[row_index]), dtype=torch.long),
                "day_of_week": torch.tensor(
                    int(day_of_week[row_index]), dtype=torch.long
                ),
                "hour_bin_start": torch.tensor(
                    int(hour_bin[row_index]), dtype=torch.long
                ),
                "counts": torch.from_numpy(counts[row_index].copy()),
                "exposure_hours": torch.tensor(
                    float(exposure[row_index]), dtype=torch.float32
                ),
                "record_id": record_ids[row_pos],
                "h3_cell": cells[row_index],
                "city_name": cities[row_index],
            }


def create_crime_rate_dataloader(
    parquet_path: str | os.PathLike[str] = DEFAULT_PARQUET_PATH,
    split: SplitName = None,
    batch_size: int = 1024,
    val_fraction: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool = False,
    *,
    batch_rows: int = 65_536,
    pin_memory: bool | None = None,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    effective_shuffle = bool(shuffle) and split != "val"
    dataset = CrimeRateIterableDataset(
        parquet_path,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
        batch_rows=batch_rows,
        shuffle=effective_shuffle,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def seed_worker(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _write_synthetic_parquet(path: Path) -> list[str]:
    rows = []
    record_ids: list[str] = []
    city_map = {"alpha_city": 1, "beta_city": 2}
    for city, city_index in city_map.items():
        for h3_i in range(1, 4):
            h3_cell = f"h3_{city_index}_{h3_i}"
            h3_index = (city_index - 1) * 3 + h3_i
            for month in (1, 6):
                for dow in (0, 5):
                    for hour_bin in (0, 3, 9, 15, 21):
                        person = float((h3_i + month) % 3)
                        property_c = float((dow + hour_bin) % 2)
                        society = 0.0
                        other = 1.0 if (h3_i + dow) % 4 == 0 else 0.0
                        total = person + property_c + society + other
                        rid = make_record_id(
                            city_index, h3_index, month, dow, hour_bin
                        )
                        record_ids.append(rid)
                        rows.append(
                            {
                                "city_name": city,
                                "h3_cell": h3_cell,
                                "city_index": city_index,
                                "h3_cell_index": h3_index,
                                "month": month,
                                "day_of_week": dow,
                                "hour_bin_start": hour_bin,
                                "exposure_hours": 168.0,
                                "person_count": person,
                                "property_count": property_c,
                                "society_count": society,
                                "other_count": other,
                                "total_count": total,
                            }
                        )
    # Invalid row (unknown indices) must be rejected.
    rows.append(
        {
            "city_name": "bad",
            "h3_cell": "bad",
            "city_index": 0,
            "h3_cell_index": 0,
            "month": 1,
            "day_of_week": 0,
            "hour_bin_start": 0,
            "exposure_hours": 4.0,
            "person_count": 0.0,
            "property_count": 0.0,
            "society_count": 0.0,
            "other_count": 0.0,
            "total_count": 0.0,
        }
    )
    pq.write_table(pa.Table.from_pylist(rows), path)
    return record_ids


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "synthetic.parquet"
        all_ids = set(_write_synthetic_parquet(parquet_path))
        seed = 42
        val_fraction = 0.20

        full = next(
            iter(
                create_crime_rate_dataloader(
                    parquet_path,
                    split=None,
                    batch_size=8,
                    val_fraction=val_fraction,
                    seed=seed,
                    num_workers=0,
                    shuffle=False,
                    batch_rows=16,
                )
            )
        )
        assert full["h3_cell_index"].dtype == torch.long
        assert full["counts"].shape[1] == 4
        assert torch.all(full["exposure_hours"] > 0)
        assert torch.all(full["h3_cell_index"] >= 1)
        assert torch.all(full["city_index"] >= 1)

        def collect(split: str) -> tuple[set[str], set[str], bool]:
            loader = create_crime_rate_dataloader(
                parquet_path,
                split=split,  # type: ignore[arg-type]
                batch_size=16,
                val_fraction=val_fraction,
                seed=seed,
                num_workers=0,
                shuffle=(split == "train"),
                batch_rows=32,
            )
            ids: set[str] = set()
            cities: set[str] = set()
            saw_zero = False
            for batch in loader:
                assert batch["counts"].shape[1] == 4
                assert torch.all(batch["exposure_hours"] > 0)
                if torch.any(batch["counts"].sum(dim=1) == 0):
                    saw_zero = True
                ids.update(str(v) for v in batch["record_id"])
                cities.update(str(v) for v in batch["city_name"])
            return ids, cities, saw_zero

        train_a, train_cities, z1 = collect("train")
        val_a, val_cities, z2 = collect("val")
        train_b, _, _ = collect("train")
        val_b, _, _ = collect("val")

        assert z1 or z2
        assert train_a.isdisjoint(val_a)
        assert train_a == train_b and val_a == val_b
        assert train_cities == {"alpha_city", "beta_city"}
        assert val_cities == {"alpha_city", "beta_city"}
        assert len(train_a) + len(val_a) == len(all_ids)
        print(
            f"train={len(train_a)} val={len(val_a)} overlap=0 "
            f"cities={sorted(train_cities)}"
        )
        print("self-test passed")


if __name__ == "__main__":
    _self_test()
