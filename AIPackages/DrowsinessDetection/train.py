#!/usr/bin/env python3
"""Train the DMD drowsiness causal TCN with session-level 80/20 validation.

Checkpoints are written under ``saved_weights/``:
  - ``best_accuracy.pt`` when validation accuracy improves
  - ``best_loss.pt`` when validation loss improves

Splits are performed by complete recording session (not by random frames) to
avoid leakage between adjacent video frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataloader import (  # noqa: E402
    ANNS_DIR,
    DEFAULT_CLOSED_SAMPLE_FRACTION,
    DEFAULT_SAMPLING_RATE_HZ,
    DEFAULT_WINDOW_SIZE,
    DROWSINESS_FEATURE_NAMES,
    LANDMARKS_DIR,
    ClosedAugmentationConfig,
    DMDGazeFrameDataset,
    build_train_val_window_datasets,
    make_train_val_loaders,
    seed_everything,
)
from feature_contract import FEATURE_COUNT, FEATURE_SCHEMA_VERSION  # noqa: E402
from label_contract import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES  # noqa: E402
from device import (  # noqa: E402
    DEFAULT_GPU_ID,
    configure_cuda,
    resolve_device,
    to_device,
)
from loss import build_loss  # noqa: E402
from metrics import MulticlassMetricMeter  # noqa: E402
from model.model import GazeZoneTCN, build_model  # noqa: E402

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "saved_weights"

# Explicit experimental inverse-frequency weights (unsafe with oversampling).
EXPERIMENTAL_CLASS_WEIGHTS = [
    2.414783496136463,
    0.38792861157953906,
    123.6044776119403,
]


def set_seed(seed: int) -> None:
    seed_everything(seed)


def save_checkpoint(
    path: Path,
    *,
    model: GazeZoneTCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    class_to_idx: Dict[str, int],
    split_info: Dict[str, object],
    args: argparse.Namespace,
    standardizer_payload: Dict[str, object],
    augmentation_config: Dict[str, object],
    sampling_config: Dict[str, object],
    model_config: Dict[str, object],
    best_val_loss: float,
    best_val_accuracy: float,
    best_val_macro_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "arch": "tcn",
        "model_config": dict(model_config),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "window_size": int(getattr(args, "window_size", DEFAULT_WINDOW_SIZE)),
        "window_frames": int(getattr(args, "window_size", DEFAULT_WINDOW_SIZE)),
        "sampling_rate_hz": float(DEFAULT_SAMPLING_RATE_HZ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_macro_f1": float(best_val_macro_f1),
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": {int(k): v for k, v in IDX_TO_CLASS.items()},
        "feature_names": list(DROWSINESS_FEATURE_NAMES),
        "feature_mean": standardizer_payload["feature_mean"],
        "feature_std": standardizer_payload["feature_std"],
        "standardized_feature_mask": standardizer_payload["standardized_feature_mask"],
        "augmentation_config": dict(augmentation_config),
        "sampling_config": dict(sampling_config),
        "split_info": {
            "train_sessions": list(split_info["train_sessions"]),
            "val_sessions": list(split_info["val_sessions"]),
            "train_samples": int(
                split_info.get("train_windows", split_info.get("train_samples", 0))
            ),
            "val_samples": int(
                split_info.get("val_windows", split_info.get("val_samples", 0))
            ),
            "val_ratio": float(split_info["val_ratio"]),
            "seed": int(split_info["seed"]),
            "counts_before_filter": split_info.get("counts_before_filter"),
            "counts_after_filter": split_info.get("counts_after_filter"),
            "removed_transition_windows": split_info.get(
                "removed_transition_windows"
            ),
            "class_weights": split_info.get("class_weights"),
            "sampler_enabled": split_info.get("sampler_enabled"),
            "sampler_target_distribution": split_info.get(
                "sampler_target_distribution"
            ),
            "closed_sample_fraction": split_info.get("closed_sample_fraction"),
            "closed_augmentation": split_info.get("closed_augmentation"),
            "augmentation_seed": split_info.get("augmentation_seed"),
            "loss_mode": split_info.get("loss_mode"),
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def compute_class_weights(
    class_counts: Dict[str, int],
    *,
    imbalance_ratio_threshold: float = 1.5,
) -> Optional[torch.Tensor]:
    """Inverse-frequency weights when max/min support exceeds ``threshold``."""
    ordered = [int(class_counts.get(IDX_TO_CLASS[i], 0)) for i in range(NUM_CLASSES)]
    positive = [c for c in ordered if c > 0]
    if len(positive) < 2:
        return None
    ratio = float(max(positive)) / float(min(positive))
    if ratio < float(imbalance_ratio_threshold):
        print(
            f"class imbalance ratio={ratio:.2f} < {imbalance_ratio_threshold}; "
            "using unweighted CE"
        )
        return None
    total = float(sum(ordered))
    weights = [
        (total / (NUM_CLASSES * float(count))) if count > 0 else 0.0
        for count in ordered
    ]
    print(f"class imbalance ratio={ratio:.2f}; using class weights={weights}")
    return torch.tensor(weights, dtype=torch.float32)


def format_detailed_metrics(detailed: Dict[str, object]) -> str:
    lines = [
        f"loss={detailed['loss']:.6f} accuracy={detailed['accuracy']:.6f} "
        f"macro_f1={detailed['macro_f1']:.6f}"
    ]
    confmat = detailed.get("confusion_matrix")
    per_class = detailed.get("per_class") or {}
    names = list(detailed.get("class_names") or [
        IDX_TO_CLASS[i] for i in range(NUM_CLASSES)
    ])
    if per_class:
        precision = per_class["precision"]
        recall = per_class["recall"]
        f1 = per_class["f1"]
        support = per_class["support"]
        lines.append("per-class precision / recall / f1 / support:")
        for index, name in enumerate(names):
            lines.append(
                f"  {name:10s}  "
                f"p={float(precision[index]):.4f}  "
                f"r={float(recall[index]):.4f}  "
                f"f1={float(f1[index]):.4f}  "
                f"n={int(support[index])}"
            )
    if confmat is not None:
        lines.append("confusion matrix (rows=true, cols=pred):")
        lines.append("           " + " ".join(f"{n:>10s}" for n in names))
        for row_index, name in enumerate(names):
            row = confmat[row_index]
            lines.append(
                f"  {name:10s}" + " ".join(f"{int(row[c]):10d}" for c in range(len(names)))
            )
    return "\n".join(lines)

def train_one_epoch(
    model: GazeZoneTCN,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    meter = MulticlassMetricMeter(num_classes)
    sampled_counts = {IDX_TO_CLASS[i]: 0 for i in range(num_classes)}
    progress = tqdm(
        loader,
        desc=f"train {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for features, labels in progress:
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        meter.update(logits, labels, loss=loss)
        for label_index in labels.detach().cpu().tolist():
            sampled_counts[IDX_TO_CLASS[int(label_index)]] += 1
        stats = meter.compute()
        progress.set_postfix(
            loss=f"{stats['loss']:.4f}",
            acc=f"{stats['accuracy']:.4f}",
        )
    result = meter.compute()
    result["sampled_class_counts"] = sampled_counts
    total = max(sum(sampled_counts.values()), 1)
    result["sampled_class_fractions"] = {
        name: float(count) / float(total) for name, count in sampled_counts.items()
    }
    return result

@torch.no_grad()
def validate_one_epoch(
    model: GazeZoneTCN,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    meter = MulticlassMetricMeter(num_classes)
    progress = tqdm(
        loader,
        desc=f"val   {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for features, labels in progress:
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)
        logits = model(features)
        loss = criterion(logits, labels)
        meter.update(logits, labels, loss=loss)
        stats = meter.compute()
        progress.set_postfix(
            loss=f"{stats['loss']:.4f}",
            acc=f"{stats['accuracy']:.4f}",
        )
    return meter.compute()


@torch.no_grad()
def evaluate_loader_detailed(
    model: GazeZoneTCN,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    model.eval()
    meter = MulticlassMetricMeter(num_classes)
    for features, labels in loader:
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)
        logits = model(features)
        loss = criterion(logits, labels)
        meter.update(logits, labels, loss=loss)
    return meter.compute_detailed(
        class_names=[IDX_TO_CLASS[i] for i in range(num_classes)]
    )


def plot_val_curves(
    history: Sequence[Dict[str, float]],
    output_path: Path,
    *,
    title: str = "Validation loss and accuracy",
) -> Path:
    """Draw validation loss and accuracy curves and save to ``output_path``."""
    if not history:
        raise ValueError("history is empty; nothing to plot")

    epochs = [int(row["epoch"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    val_acc = [float(row["val_accuracy"]) for row in history]

    best_loss_epoch = epochs[min(range(len(val_loss)), key=lambda i: val_loss[i])]
    best_acc_epoch = epochs[max(range(len(val_acc)), key=lambda i: val_acc[i])]
    best_loss = min(val_loss)
    best_acc = max(val_acc)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_acc = ax_loss.twinx()

    loss_line = ax_loss.plot(
        epochs, val_loss, color="#1f77b4", marker="o", linewidth=2, label="Val loss"
    )
    acc_line = ax_acc.plot(
        epochs,
        val_acc,
        color="#d62728",
        marker="s",
        linewidth=2,
        label="Val accuracy",
    )

    ax_loss.axvline(
        best_loss_epoch,
        color="#1f77b4",
        linestyle="--",
        alpha=0.5,
        label=f"Best loss @ {best_loss_epoch}",
    )
    ax_acc.axvline(
        best_acc_epoch,
        color="#d62728",
        linestyle=":",
        alpha=0.5,
        label=f"Best acc @ {best_acc_epoch}",
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation loss", color="#1f77b4")
    ax_acc.set_ylabel("Validation accuracy", color="#d62728")
    ax_loss.tick_params(axis="y", labelcolor="#1f77b4")
    ax_acc.tick_params(axis="y", labelcolor="#d62728")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xticks(epochs)

    lines = loss_line + acc_line
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # print(
    #     f"wrote val curves: {output_path} "
    #     f"(best loss={best_loss:.4f} @ epoch {best_loss_epoch}, "
    #     f"best acc={best_acc:.4f} @ epoch {best_acc_epoch})"
    # )
    return output_path


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DMD drowsiness TCN (multiclass, session 80/20 split)."
    )
    parser.add_argument("--anns-dir", type=Path, default=ANNS_DIR)
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_DIR)
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Causal input window length (default: 20).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Stop after this many epochs without val-loss improvement.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for v3 checkpoints (best_accuracy_v3.pt / best_loss_v3.pt).",
    )
    parser.add_argument(
        "--best-accuracy-name",
        type=str,
        default="best_accuracy_v3.pt",
        help="Filename for the best-accuracy checkpoint (never overwrites v1/v2).",
    )
    parser.add_argument(
        "--best-loss-name",
        type=str,
        default="best_loss_v3.pt",
        help="Filename for the best-loss checkpoint (never overwrites v1/v2).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU_ID,
        help=f"CUDA device index (default: {DEFAULT_GPU_ID} -> cuda:{DEFAULT_GPU_ID})",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable (default: require CUDA).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional CrossEntropy label smoothing in [0, 1).",
    )
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=None,
        help="Path for the validation loss/accuracy curve PNG "
        "(default: <checkpoint-dir>/val_loss_accuracy_curves.png).",
    )
    parser.add_argument(
        "--closed-augmentation",
        dest="closed_augmentation",
        action="store_true",
        default=True,
        help="Enable closed-only on-the-fly augmentation (default: on).",
    )
    parser.add_argument(
        "--no-closed-augmentation",
        dest="closed_augmentation",
        action="store_false",
        help="Disable closed-only augmentation.",
    )
    parser.add_argument(
        "--closed-sample-fraction",
        type=float,
        default=DEFAULT_CLOSED_SAMPLE_FRACTION,
        help="Desired closed fraction among open+closed train samples "
        f"(default: {DEFAULT_CLOSED_SAMPLE_FRACTION}, range 0.0–0.50).",
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=None,
        help="Seed for closed-window augmenter RNG (default: --seed).",
    )
    parser.add_argument(
        "--morphology-probability",
        type=float,
        default=0.70,
        help="Probability of eye-morphology scaling (default: 0.70).",
    )
    parser.add_argument(
        "--measurement-noise-probability",
        type=float,
        default=0.50,
        help="Probability of temporally correlated measurement noise (default: 0.50).",
    )
    parser.add_argument(
        "--calibration-bias-probability",
        type=float,
        default=0.50,
        help="Probability of per-window calibration bias (default: 0.50).",
    )
    parser.add_argument(
        "--pupil-jitter-probability",
        type=float,
        default=0.30,
        help="Probability of coherent pupil jitter when pupil features exist "
        "(default: 0.30).",
    )
    parser.add_argument(
        "--landmark-dropout-probability",
        type=float,
        default=0.03,
        help="Probability of eye-landmark dropout when validity masks exist "
        "(default: 0.03).",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Experimental: use fixed inverse-frequency class weights. "
        "Incompatible with closed oversampling (closed-sample-fraction > 0).",
    )
    return parser.parse_args(argv)

def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= float(args.closed_sample_fraction) <= 0.50:
        raise ValueError(
            f"--closed-sample-fraction must be in [0.0, 0.50], "
            f"got {args.closed_sample_fraction}"
        )
    sampler_enabled = float(args.closed_sample_fraction) > 0.0
    if sampler_enabled and bool(args.use_class_weights):
        raise ValueError(
            "Cannot enable --use-class-weights together with closed-window "
            "oversampling (--closed-sample-fraction > 0). Disable one of them."
        )

    set_seed(args.seed)
    device = resolve_device(args.gpu, require_cuda=not args.allow_cpu)
    configure_cuda(device)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=== frozen feature schema ===")
    print(f"feature_schema_version: {FEATURE_SCHEMA_VERSION}")
    print(f"feature_count: {FEATURE_COUNT}")
    print(f"feature_names: {list(DROWSINESS_FEATURE_NAMES)}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"random seed: {args.seed}")

    frame_dataset = DMDGazeFrameDataset(
        anns_dir=args.anns_dir,
        landmarks_dir=args.landmarks_dir,
        verbose=True,
    )
    if args.window_size < 1:
        raise ValueError(f"window_size must be positive, got {args.window_size}")

    aug_config = ClosedAugmentationConfig(
        morphology_p=float(args.morphology_probability),
        noise_p=float(args.measurement_noise_probability),
        bias_p=float(args.calibration_bias_probability),
        pupil_jitter_p=float(args.pupil_jitter_probability),
        dropout_p=float(args.landmark_dropout_probability),
    )
    aug_seed = (
        int(args.seed)
        if args.augmentation_seed is None
        else int(args.augmentation_seed)
    )

    train_dataset, val_dataset, standardizer, split_info = (
        build_train_val_window_datasets(
            frame_dataset,
            window_size=args.window_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            aug_config=aug_config,
            closed_augmentation=bool(args.closed_augmentation),
            augmentation_seed=aug_seed,
            verbose=True,
        )
    )
    train_loader, val_loader, split_info = make_train_val_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        split_info=split_info,
        closed_sample_fraction=float(args.closed_sample_fraction),
    )

    train_counts = dict(getattr(train_dataset, "class_counts", {}))
    val_counts = dict(getattr(val_dataset, "class_counts", {}))
    train_total = max(sum(int(v) for v in train_counts.values()), 1)
    val_total = max(sum(int(v) for v in val_counts.values()), 1)
    majority_val = max(val_counts, key=lambda k: val_counts[k]) if val_counts else "n/a"
    majority_val_frac = (
        float(val_counts[majority_val]) / float(val_total) if val_counts else 0.0
    )

    print("=== label distributions ===")
    print(f"train counts: {train_counts}")
    print(
        "train fractions: "
        + str({k: float(v) / float(train_total) for k, v in train_counts.items()})
    )
    print(f"val counts:   {val_counts}")
    print(
        "val fractions: "
        + str({k: float(v) / float(val_total) for k, v in val_counts.items()})
    )
    print(
        f"majority-class validation baseline: {majority_val} "
        f"({majority_val_frac:.4f})"
    )
    print("=== augmentation configuration ===")
    print(split_info.get("augmentation_config"))
    print(f"closed_augmentation: {args.closed_augmentation}")
    print(f"augmentation_seed: {aug_seed}")
    print("=== sampler configuration ===")
    print(f"sampler_enabled: {split_info.get('sampler_enabled')}")
    print(f"closed_sample_fraction: {args.closed_sample_fraction}")
    print(f"sampler_target_distribution: {split_info.get('sampler_target_distribution')}")
    print(f"TCN window_size: {args.window_size}")
    print(f"DataLoader pin_memory={split_info['pin_memory']}")
    print(
        f"session split: train={split_info['train_windows']} "
        f"({len(split_info['train_sessions'])} sessions) | "
        f"val={split_info['val_windows']} "
        f"({len(split_info['val_sessions'])} sessions)"
    )
    print(f"train sessions: {split_info['train_sessions']}")
    print(f"val sessions:   {split_info['val_sessions']}")

    standardizer_payload = standardizer.to_checkpoint_dict()
    augmentation_config = dict(split_info.get("augmentation_config") or {})
    sampling_config = {
        "closed_sample_fraction": float(args.closed_sample_fraction),
        "sampler_enabled": bool(split_info.get("sampler_enabled")),
        "sampler_target_distribution": split_info.get("sampler_target_distribution"),
        "closed_augmentation": bool(args.closed_augmentation),
        "augmentation_seed": aug_seed,
        "use_class_weights": bool(args.use_class_weights),
    }

    num_classes = NUM_CLASSES
    assert num_classes == len(CLASS_TO_IDX)
    assert FEATURE_COUNT == len(DROWSINESS_FEATURE_NAMES)
    model_config = {
        "arch": "tcn",
        "num_classes": num_classes,
        "input_dim": FEATURE_COUNT,
        "channels": 64,
        "kernel_size": 3,
        "dilations": (1, 2, 4),
        "dropout": float(args.dropout),
        "window_frames": int(args.window_size),
    }
    model = build_model(
        num_classes=num_classes,
        input_dim=FEATURE_COUNT,
        dropout=args.dropout,
    ).to(device)
    print(f"model device: {next(model.parameters()).device}")
    print(f"trainable parameters: {model.num_trainable_parameters:,}")
    print(f"num_classes={num_classes} from CLASS_TO_IDX={CLASS_TO_IDX}")
    print(f"model input_dim={FEATURE_COUNT} (frozen schema)")

    if sampler_enabled:
        class_weights = None
        loss_mode = "unweighted_ce_with_oversampling"
        print(
            "loss: unweighted CrossEntropyLoss "
            "(oversampling active; class weights disabled)"
        )
    elif args.use_class_weights:
        class_weights = torch.tensor(EXPERIMENTAL_CLASS_WEIGHTS, dtype=torch.float32)
        if class_weights.numel() != num_classes:
            raise ValueError(
                f"EXPERIMENTAL_CLASS_WEIGHTS length {class_weights.numel()} "
                f"!= num_classes {num_classes}"
            )
        loss_mode = "experimental_fixed_class_weights"
        print(f"loss: weighted CE with experimental weights={class_weights.tolist()}")
        class_weights = class_weights.to(device)
    else:
        class_weights = compute_class_weights(dict(train_dataset.class_counts))
        if class_weights is not None:
            loss_mode = "auto_inverse_frequency_weights"
            class_weights = class_weights.to(device)
        else:
            loss_mode = "unweighted_ce"
    split_info["class_weights"] = (
        class_weights.detach().cpu().tolist() if class_weights is not None else None
    )
    split_info["loss_mode"] = loss_mode
    sampling_config["loss_mode"] = loss_mode
    print(f"active loss configuration: {loss_mode}")

    criterion = build_loss(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_val_accuracy = -float("inf")
    best_val_macro_f1 = -float("inf")
    epochs_without_loss_improve = 0
    history = []

    best_loss_path = checkpoint_dir / args.best_loss_name
    best_acc_path = checkpoint_dir / args.best_accuracy_name
    # Refuse to overwrite legacy v1 filenames.
    for path in (best_loss_path, best_acc_path):
        if path.name in {"best_loss.pt", "best_accuracy.pt"}:
            raise ValueError(
                f"refusing to overwrite legacy checkpoint name {path.name}; "
                "use *_v3.pt"
            )
    curve_path = (
        args.curve_path.expanduser().resolve()
        if args.curve_path is not None
        else checkpoint_dir / "val_loss_accuracy_curves_v3.png"
    )

    def _checkpoint_kwargs(metrics: Dict[str, float]) -> Dict[str, object]:
        return {
            "model": model,
            "optimizer": optimizer,
            "metrics": metrics,
            "class_to_idx": CLASS_TO_IDX,
            "split_info": split_info,
            "args": args,
            "standardizer_payload": standardizer_payload,
            "augmentation_config": augmentation_config,
            "sampling_config": sampling_config,
            "model_config": model_config,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
        }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            epoch,
            args.epochs,
        )
        if epoch == 1:
            print("=== actual sampled class distribution (epoch 1) ===")
            print(f"counts: {train_metrics.get('sampled_class_counts')}")
            print(f"fractions: {train_metrics.get('sampled_class_fractions')}")
            split_info["epoch1_sampled_class_counts"] = train_metrics.get(
                "sampled_class_counts"
            )
            split_info["epoch1_sampled_class_fractions"] = train_metrics.get(
                "sampled_class_fractions"
            )
            sampling_config["epoch1_sampled_class_counts"] = train_metrics.get(
                "sampled_class_counts"
            )
            sampling_config["epoch1_sampled_class_fractions"] = train_metrics.get(
                "sampled_class_fractions"
            )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            epoch,
            args.epochs,
        )
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
            f"f1={row['train_macro_f1']:.4f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_accuracy']:.4f} "
            f"f1={row['val_macro_f1']:.4f} | "
            f"lr={row['lr']:.2e}"
        )

        plot_val_curves(history, curve_path)

        improved_loss = val_metrics["loss"] < best_val_loss
        improved_acc = val_metrics["accuracy"] > best_val_accuracy

        if improved_loss:
            best_val_loss = val_metrics["loss"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            epochs_without_loss_improve = 0
            save_checkpoint(
                best_loss_path,
                epoch=epoch,
                **_checkpoint_kwargs(val_metrics),  # type: ignore[arg-type]
            )
        else:
            epochs_without_loss_improve += 1

        if improved_acc:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_macro_f1 = max(best_val_macro_f1, val_metrics["macro_f1"])
            save_checkpoint(
                best_acc_path,
                epoch=epoch,
                **_checkpoint_kwargs(val_metrics),  # type: ignore[arg-type]
            )

        if epochs_without_loss_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val-loss improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    # Final detailed train/val report (reload best-loss weights for val metrics).
    if best_loss_path.is_file():
        ckpt = torch.load(best_loss_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

    print("=== final TRAIN metrics (best-loss weights) ===")
    train_detailed = evaluate_loader_detailed(
        model, train_loader, criterion, device, num_classes
    )
    print(format_detailed_metrics(train_detailed))
    print("=== final VAL metrics (best-loss weights) ===")
    val_detailed = evaluate_loader_detailed(
        model, val_loader, criterion, device, num_classes
    )
    print(format_detailed_metrics(val_detailed))

    # Refresh best_* fields on accuracy checkpoint after final eval.
    best_val_loss = float(val_detailed["loss"])
    best_val_accuracy = float(val_detailed["accuracy"])
    best_val_macro_f1 = float(val_detailed["macro_f1"])
    if best_acc_path.is_file():
        # Keep accuracy checkpoint as the named deliverable; update metrics block.
        acc_ckpt = torch.load(best_acc_path, map_location="cpu", weights_only=False)
        model.load_state_dict(acc_ckpt["model_state_dict"], strict=True)
        model.to(device)
        acc_val = evaluate_loader_detailed(
            model, val_loader, criterion, device, num_classes
        )
        save_checkpoint(
            best_acc_path,
            epoch=int(acc_ckpt.get("epoch", 0)),
            model=model,
            optimizer=optimizer,
            metrics={
                "loss": float(acc_val["loss"]),
                "accuracy": float(acc_val["accuracy"]),
                "macro_f1": float(acc_val["macro_f1"]),
            },
            class_to_idx=CLASS_TO_IDX,
            split_info=split_info,
            args=args,
            standardizer_payload=standardizer_payload,
            augmentation_config=augmentation_config,
            sampling_config=sampling_config,
            model_config=model_config,
            best_val_loss=float(acc_val["loss"]),
            best_val_accuracy=float(acc_val["accuracy"]),
            best_val_macro_f1=float(acc_val["macro_f1"]),
        )
        # Also dump confusion matrix beside the checkpoint for the report.
        report = {
            "checkpoint": str(best_acc_path),
            "best_val_loss": float(acc_val["loss"]),
            "best_val_accuracy": float(acc_val["accuracy"]),
            "best_val_macro_f1": float(acc_val["macro_f1"]),
            "class_to_idx": CLASS_TO_IDX,
            "confusion_matrix": [
                [int(x) for x in row]
                for row in acc_val["confusion_matrix"].tolist()
            ],
            "per_class": {
                name: {
                    "precision": float(acc_val["per_class"]["precision"][i]),
                    "recall": float(acc_val["per_class"]["recall"][i]),
                    "f1": float(acc_val["per_class"]["f1"][i]),
                    "support": int(acc_val["per_class"]["support"][i]),
                }
                for i, name in enumerate(
                    [IDX_TO_CLASS[j] for j in range(num_classes)]
                )
            },
            "sampling_config": sampling_config,
            "augmentation_config": augmentation_config,
        }
        report_path = checkpoint_dir / "best_accuracy_v3_metrics.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote metrics report: {report_path}")

    history_path = checkpoint_dir / "train_history_v3.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"wrote history: {history_path}")
    plot_val_curves(history, curve_path)
    print(
        f"best val loss={best_val_loss:.6f} | "
        f"best val accuracy={best_val_accuracy:.6f} | "
        f"best val macro_f1={best_val_macro_f1:.6f}"
    )
    print(f"v3 accuracy checkpoint: {best_acc_path}")
    print(f"v3 loss checkpoint:     {best_loss_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
