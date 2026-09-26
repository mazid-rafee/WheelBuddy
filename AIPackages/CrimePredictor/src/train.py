"""Train categorical CrimeRateMLP for exposure-aware crime-rate prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from build_rate_dataset import DEFAULT_METADATA_DIR
from time_bins import NUM_TIME_BINS, TIME_BIN_HOURS
from dataloader import (
    COUNT_COLUMNS,
    DEFAULT_PARQUET_PATH,
    create_crime_rate_dataloader,
    make_record_id,
    validation_assignment,
)
from loss import build_loss
from metrics import (
    CATEGORY_NAMES,
    CrimeRateMetrics,
    format_metrics,
    format_per_category_metrics,
)
from model import build_model, resolve_torch_device


def set_seed(seed: int, *, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            item = value.item()
            return float(item) if isinstance(item, float) else int(item)
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def estimate_num_batches(num_records: int, batch_size: int) -> int:
    if num_records <= 0 or batch_size <= 0:
        return 0
    return int(math.ceil(num_records / float(batch_size)))


def load_json_mapping(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_vocabularies(metadata_dir: Path) -> tuple[dict[str, int], dict[str, int], dict[str, object]]:
    city_path = metadata_dir / "city_to_index.json"
    h3_path = metadata_dir / "h3_cell_to_index.json"
    city_to_index = {str(k): int(v) for k, v in load_json_mapping(city_path).items()}
    h3_to_index = {str(k): int(v) for k, v in load_json_mapping(h3_path).items()}

    # Determinism check: rebuild sorted assignment for known keys.
    known_cities = sorted(name for name in city_to_index if name != "<UNK>")
    known_h3 = sorted(name for name in h3_to_index if name != "<UNK>")
    rebuilt_cities = {"<UNK>": 0, **{name: i for i, name in enumerate(known_cities, 1)}}
    rebuilt_h3 = {"<UNK>": 0, **{name: i for i, name in enumerate(known_h3, 1)}}
    if rebuilt_cities != city_to_index:
        raise RuntimeError("city_to_index.json is not a deterministic sorted vocabulary")
    if rebuilt_h3 != h3_to_index:
        raise RuntimeError("h3_cell_to_index.json is not a deterministic sorted vocabulary")

    meta = {
        "paths": {
            "city_to_index": str(city_path),
            "h3_cell_to_index": str(h3_path),
            "index_to_city": str(metadata_dir / "index_to_city.json"),
            "index_to_h3_cell": str(metadata_dir / "index_to_h3_cell.json"),
        },
        "checksums": {
            "city_to_index": sha256_file(city_path),
            "h3_cell_to_index": sha256_file(h3_path),
            "index_to_city": sha256_file(metadata_dir / "index_to_city.json"),
            "index_to_h3_cell": sha256_file(metadata_dir / "index_to_h3_cell.json"),
        },
        "num_city_embeddings": max(city_to_index.values()) + 1,
        "num_h3_embeddings": max(h3_to_index.values()) + 1,
        "num_cities": len(known_cities),
        "num_h3_cells": len(known_h3),
    }
    print(
        f"Loaded vocabularies: cities={meta['num_cities']:,} "
        f"h3={meta['num_h3_cells']:,} "
        f"(embed sizes city={meta['num_city_embeddings']}, "
        f"h3={meta['num_h3_embeddings']})"
    )
    return city_to_index, h3_to_index, meta


def scan_parquet_split_statistics(
    parquet_path: Path,
    *,
    val_fraction: float,
    seed: int,
    batch_rows: int = 65_536,
) -> tuple[Counter[str], Counter[str], dict[str, float]]:
    train_city_counts: Counter[str] = Counter()
    val_city_counts: Counter[str] = Counter()
    train_count_sums = np.zeros(len(COUNT_COLUMNS), dtype=np.float64)
    train_exposure_sum = 0.0

    parquet_file = pq.ParquetFile(parquet_path)
    print("\nParquet schema columns:")
    for name in parquet_file.schema_arrow.names:
        print(f"  - {name}")

    columns = [
        "city_name",
        "city_index",
        "h3_cell_index",
        "month",
        "day_of_week",
        "hour_bin_start",
        "exposure_hours",
        *COUNT_COLUMNS,
    ]
    for batch_index, batch in enumerate(
        parquet_file.iter_batches(batch_size=batch_rows, columns=columns)
    ):
        cities = [
            "" if v is None else str(v) for v in batch.column("city_name").to_pylist()
        ]
        city_index = np.asarray(
            batch.column("city_index").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        h3_index = np.asarray(
            batch.column("h3_cell_index").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        month = np.asarray(
            batch.column("month").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        dow = np.asarray(
            batch.column("day_of_week").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        hour = np.asarray(
            batch.column("hour_bin_start").to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        exposure = np.asarray(
            batch.column("exposure_hours").to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        counts = np.column_stack(
            [
                np.asarray(
                    batch.column(name).to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
                for name in COUNT_COLUMNS
            ]
        )

        for row in range(len(cities)):
            city = cities[row]
            if not city or city_index[row] < 1 or h3_index[row] < 1:
                continue
            if not np.isfinite(exposure[row]) or exposure[row] <= 0:
                continue
            record_id = make_record_id(
                int(city_index[row]),
                int(h3_index[row]),
                int(month[row]),
                int(dow[row]),
                int(hour[row]),
            )
            if validation_assignment(record_id, seed, val_fraction):
                val_city_counts[city] += 1
            else:
                train_city_counts[city] += 1
                train_count_sums += counts[row]
                train_exposure_sum += float(exposure[row])

        if (batch_index + 1) % 20 == 0:
            print(
                f"Split scan batches={batch_index + 1}: "
                f"train={sum(train_city_counts.values()):,} "
                f"val={sum(val_city_counts.values()):,}",
                end="\r",
                flush=True,
            )
    print()

    if train_exposure_sum <= 0:
        raise RuntimeError("Training split has non-positive total exposure")

    baseline = {
        name: float(train_count_sums[i] / train_exposure_sum)
        for i, name in enumerate(CATEGORY_NAMES)
    }
    baseline["total"] = float(train_count_sums.sum() / train_exposure_sum)
    baseline["train_exposure_hours"] = float(train_exposure_sum)
    return train_city_counts, val_city_counts, baseline


def print_city_split_counts(
    train_counts: Counter[str], val_counts: Counter[str]
) -> None:
    cities = sorted(set(train_counts) | set(val_counts))
    print("\nPer-city split counts")
    print(f"{'city':<24} {'train':>14} {'validation':>14} {'total':>14}")
    print("-" * 70)
    for city in cities:
        tr = train_counts[city]
        va = val_counts[city]
        print(f"{city:<24} {tr:>14,} {va:>14,} {tr + va:>14,}")
    print("-" * 70)
    print(
        f"{'TOTAL':<24} {sum(train_counts.values()):>14,} "
        f"{sum(val_counts.values()):>14,} "
        f"{sum(train_counts.values()) + sum(val_counts.values()):>14,}"
    )
    held = [c for c in cities if train_counts[c] == 0 or val_counts[c] == 0]
    if held:
        raise RuntimeError(f"Cities missing from a split: {held}")


def verify_split_id_nonoverlap(train_loader, val_loader, *, max_batches: int = 5) -> None:
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        train_ids.update(str(v) for v in batch["record_id"])
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        val_ids.update(str(v) for v in batch["record_id"])
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"Train/val record_id overlap: {sorted(overlap)[:5]}")
    print(
        f"\nSplit ID check (bounded): train_sample={len(train_ids):,} "
        f"val_sample={len(val_ids):,} overlap=0"
    )


def run_epoch(
    *,
    model,
    loader,
    criterion,
    device: torch.device,
    optimizer: AdamW | None,
    max_batches: int | None,
    desc: str,
    total_batches: int | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    non_blocking = device.type == "cuda"
    total_loss = 0.0
    total_records = 0
    metric_tracker = CrimeRateMetrics(category_names=CATEGORY_NAMES)

    expected = total_batches
    if max_batches is not None:
        expected = max_batches if expected is None else min(expected, max_batches)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(
            enumerate(loader),
            total=expected,
            desc=desc,
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )
        for batch_index, batch in progress:
            if max_batches is not None and batch_index >= max_batches:
                break

            h3 = batch["h3_cell_index"].to(device, non_blocking=non_blocking)
            city = batch["city_index"].to(device, non_blocking=non_blocking)
            month = batch["month"].to(device, non_blocking=non_blocking)
            dow = batch["day_of_week"].to(device, non_blocking=non_blocking)
            hour = batch["hour_bin_start"].to(device, non_blocking=non_blocking)
            counts = batch["counts"].to(device, non_blocking=non_blocking)
            exposure = batch["exposure_hours"].to(device, non_blocking=non_blocking)

            if training:
                optimizer.zero_grad(set_to_none=True)
            rates = model(h3, city, month, dow, hour)
            loss = criterion(rates, counts, exposure)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            n = int(h3.shape[0])
            total_loss += float(loss.detach()) * n
            total_records += n
            metric_tracker.update(rates, counts, exposure)
            progress.set_postfix(
                loss=f"{total_loss / max(total_records, 1):.4f}",
                n=f"{total_records:,}",
                refresh=False,
            )

    if total_records == 0:
        raise RuntimeError("DataLoader produced no records")
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    metrics = metric_tracker.compute()
    return {"loss": total_loss / total_records, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train categorical crime-rate model")
    parser.add_argument("--data", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crime_rate_model"))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume from a checkpoint path, or 'last'/'best' under --output-dir. "
            "When set, --epochs is the number of additional epochs to run."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs to train from scratch, or additional epochs when --resume is set.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--h3-embedding-dim", type=int, default=24)
    parser.add_argument("--city-embedding-dim", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def resolve_resume_path(resume: Path, output_dir: Path) -> Path:
    """Resolve --resume to a concrete checkpoint file."""
    token = str(resume)
    if token in {"last", "best"}:
        path = output_dir / f"{token}.pt"
    else:
        path = resume
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def load_resume_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: AdamW,
    device: torch.device,
    learning_rate: float,
    expected_time_bin_hours: int,
    expected_h3_vocab: int,
    expected_city_vocab: int,
) -> tuple[int, float]:
    """Restore model/optimizer. Return (next_epoch, best_val_poisson)."""
    print(f"\nResuming from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    ckpt_time_bin = int(checkpoint.get("time_bin_hours", -1))
    if ckpt_time_bin != expected_time_bin_hours:
        raise ValueError(
            f"Checkpoint time_bin_hours={ckpt_time_bin} does not match "
            f"current TIME_BIN_HOURS={expected_time_bin_hours}"
        )
    if int(checkpoint.get("h3_vocabulary_size", -1)) != expected_h3_vocab:
        raise ValueError(
            "Checkpoint H3 vocabulary size does not match current metadata"
        )
    if int(checkpoint.get("city_vocabulary_size", -1)) != expected_city_vocab:
        raise ValueError(
            "Checkpoint city vocabulary size does not match current metadata"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Restart base LR for a fresh cosine cycle over the additional epochs.
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)

    completed_epoch = int(checkpoint.get("epoch", 0))
    best_val = float(checkpoint.get("best_val_poisson_deviance", float("inf")))
    print(
        f"  restored epoch={completed_epoch} "
        f"best_val_poisson_deviance={best_val:.6f} "
        f"lr_reset={learning_rate:.6e}"
    )
    return completed_epoch + 1, best_val


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be strictly between 0 and 1")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    device = resolve_torch_device(args.device)
    set_seed(args.seed, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.json"
    history: list[dict[str, object]] = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []

    _, _, vocab_meta = load_vocabularies(args.metadata_dir)
    train_city_counts, val_city_counts, baseline = scan_parquet_split_statistics(
        args.data,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_rows=args.batch_rows,
    )
    print_city_split_counts(train_city_counts, val_city_counts)

    print("\nTraining-reference baseline rates (per hour):")
    for name in (*CATEGORY_NAMES, "total"):
        print(f"  {name:<10} {baseline[name]:.8f}")

    split_summary = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "device": str(device),
        "parquet_path": str(args.data),
        "time_bin_hours": TIME_BIN_HOURS,
        "category_names": list(CATEGORY_NAMES),
        "train_city_counts": dict(sorted(train_city_counts.items())),
        "val_city_counts": dict(sorted(val_city_counts.items())),
        "vocabulary": vocab_meta,
        "baseline_rate_per_hour": {
            key: baseline[key] for key in (*CATEGORY_NAMES, "total")
        },
        "note": (
            "baseline_rate_per_hour is the training-split reference average, "
            "not a national average."
        ),
    }
    with (args.output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(split_summary), handle, indent=2)

    if device.type == "cuda":
        print(f"\nDevice: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"\nDevice: {device}")

    model_config = {
        "num_h3_embeddings": vocab_meta["num_h3_embeddings"],
        "num_city_embeddings": vocab_meta["num_city_embeddings"],
        "h3_embedding_dim": args.h3_embedding_dim,
        "city_embedding_dim": args.city_embedding_dim,
        "month_embedding_dim": 4,
        "weekday_embedding_dim": 3,
        "time_bin_embedding_dim": 3,
        "num_time_bins": NUM_TIME_BINS,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "num_outputs": 4,
        "time_bin_hours": TIME_BIN_HOURS,
        "inputs": [
            "h3_cell_index",
            "city_index",
            "month",
            "day_of_week",
            "hour_bin_start",
        ],
    }
    model = build_model(
        num_h3_embeddings=int(vocab_meta["num_h3_embeddings"]),
        num_city_embeddings=int(vocab_meta["num_city_embeddings"]),
        h3_embedding_dim=args.h3_embedding_dim,
        city_embedding_dim=args.city_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = build_loss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_poisson = float("inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        resume_path = resolve_resume_path(args.resume, args.output_dir)
        start_epoch, best_val_poisson = load_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            learning_rate=args.learning_rate,
            expected_time_bin_hours=TIME_BIN_HOURS,
            expected_h3_vocab=int(vocab_meta["num_h3_embeddings"]),
            expected_city_vocab=int(vocab_meta["num_city_embeddings"]),
        )

    # Fresh cosine over the upcoming epoch budget (scratch or additional).
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    end_epoch = start_epoch + args.epochs - 1
    print(
        f"\nTraining plan: epochs {start_epoch}..{end_epoch} "
        f"({'resume' if args.resume is not None else 'from scratch'}, "
        f"{args.epochs} epoch(s))"
    )

    train_loader = create_crime_rate_dataloader(
        args.data,
        split="train",
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=True,
        batch_rows=args.batch_rows,
        pin_memory=device.type == "cuda",
    )
    val_loader = create_crime_rate_dataloader(
        args.data,
        split="val",
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        batch_rows=args.batch_rows,
        pin_memory=device.type == "cuda",
    )
    verify_split_id_nonoverlap(train_loader, val_loader, max_batches=5)

    train_batches = estimate_num_batches(
        sum(train_city_counts.values()), args.batch_size
    )
    val_batches = estimate_num_batches(sum(val_city_counts.values()), args.batch_size)

    for epoch in range(start_epoch, end_epoch + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            desc=f"epoch {epoch}/{end_epoch} train",
            total_batches=train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
            desc=f"epoch {epoch}/{end_epoch} val",
            total_batches=val_batches,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        print(f"\nEpoch {epoch:02d}/{end_epoch} | lr={current_lr:.6e}")
        print(
            "  train: "
            f"loss={float(train_metrics['loss']):.5f} | "
            f"pois_dev={float(train_metrics['poisson_deviance']):.5f} | "
            f"total_rate_mae={float(train_metrics['total_rate_mae']):.6f} | "
            f"count_mae={float(train_metrics['count_mae']):.4f} | "
            f"calib={train_metrics['calibration_ratio']} | "
            f"records={train_metrics['num_records']}"
        )
        print(
            "  val:   "
            f"loss={float(val_metrics['loss']):.5f} | "
            f"pois_dev={float(val_metrics['poisson_deviance']):.5f} | "
            f"total_rate_mae={float(val_metrics['total_rate_mae']):.6f} | "
            f"count_mae={float(val_metrics['count_mae']):.4f} | "
            f"calib={val_metrics['calibration_ratio']} | "
            f"records={val_metrics['num_records']}"
        )
        print(" ", format_metrics(val_metrics))

        val_poisson = float(val_metrics["poisson_deviance"])
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config,
            "embedding_dimensions": {
                "h3": args.h3_embedding_dim,
                "city": args.city_embedding_dim,
                "month": 4,
                "weekday": 3,
                "time_bin": 3,
                "num_time_bins": NUM_TIME_BINS,
                "hidden": args.hidden_dim,
            },
            "h3_vocabulary_size": vocab_meta["num_h3_embeddings"],
            "city_vocabulary_size": vocab_meta["num_city_embeddings"],
            "vocabulary_file_paths": vocab_meta["paths"],
            "vocabulary_checksums": vocab_meta["checksums"],
            "category_names": list(CATEGORY_NAMES),
            "time_bin_hours": TIME_BIN_HOURS,
            "train_metrics": to_jsonable(train_metrics),
            "val_metrics": to_jsonable(val_metrics),
            "val_poisson_deviance": val_poisson,
            "best_val_poisson_deviance": min(best_val_poisson, val_poisson)
            if best_val_poisson < float("inf")
            else val_poisson,
            "baseline_rate_per_hour": {
                key: baseline[key] for key in (*CATEGORY_NAMES, "total")
            },
            "trainable_parameter_count": n_params,
            "args": to_jsonable(vars(args)),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")

        if val_poisson < best_val_poisson:
            best_val_poisson = val_poisson
            epochs_without_improvement = 0
            checkpoint["best_val_poisson_deviance"] = best_val_poisson
            torch.save(checkpoint, args.output_dir / "best.pt")
            print("  new best validation Poisson deviance")
            print(format_per_category_metrics(val_metrics))
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train": to_jsonable(train_metrics),
                "validation": to_jsonable(val_metrics),
            }
        )
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    print(f"\nBest validation Poisson deviance: {best_val_poisson:.6f}")
    print(f"Checkpoints: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
