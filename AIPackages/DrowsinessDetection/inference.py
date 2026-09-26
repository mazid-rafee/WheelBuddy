#!/usr/bin/env python3
"""DMD drowsiness TCN inference helpers (importable) and optional CLI.

Importable API surface used by ``api.inference_service``:

* ``load_checkpoint`` — construct ``GazeZoneTCN``, load weights strictly
* ``predict_logits`` / ``predict_proba`` — run a ``[batch, T, F]`` tensor
* ``normalize_windows`` — apply checkpoint mean/std + validity remask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataloader import (  # noqa: E402
    ANNS_DIR,
    DEFAULT_WINDOW_SIZE,
    DROWSINESS_FEATURE_NAMES,
    LANDMARKS_DIR,
    DMDGazeFrameDataset,
    FeatureStandardizer,
    build_train_val_window_datasets,
    enforce_feature_invariants,
    feature_index_map,
)
from feature_contract import (  # noqa: E402
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSIONS,
)
from label_contract import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES  # noqa: E402
from device import (  # noqa: E402
    DEFAULT_GPU_ID,
    configure_cuda,
    resolve_device,
    to_device,
)
from model.model import GazeZoneTCN, build_model  # noqa: E402

DEFAULT_CHECKPOINT = _SRC_DIR / "saved_weights" / "best_loss_v3.pt"


class SchemaContractError(ValueError):
    """Raised when a checkpoint or feature packet violates schema v3."""


def _require_v3_checkpoint(checkpoint: Dict[str, Any]) -> None:
    schema = checkpoint.get("feature_schema_version") or checkpoint.get(
        "schema_version"
    )
    if schema is None:
        raise SchemaContractError(
            "checkpoint missing feature_schema_version; rejecting as non-v3"
        )
    if schema in LEGACY_SCHEMA_VERSIONS or str(schema) != FEATURE_SCHEMA_VERSION:
        raise SchemaContractError(
            f"rejected schema {schema!r}; required {FEATURE_SCHEMA_VERSION!r}"
        )

    feature_names = list(checkpoint.get("feature_names") or [])
    if feature_names != list(DROWSINESS_FEATURE_NAMES):
        raise SchemaContractError(
            "checkpoint feature_names must exactly match "
            f"DROWSINESS_FEATURE_NAMES ({FEATURE_COUNT}); "
            f"got count={len(feature_names)}"
        )

    class_to_idx = dict(checkpoint["class_to_idx"])
    if class_to_idx != dict(CLASS_TO_IDX):
        raise SchemaContractError(
            f"checkpoint class_to_idx must equal {CLASS_TO_IDX}, got {class_to_idx}"
        )
    if len(class_to_idx) != NUM_CLASSES:
        raise SchemaContractError(
            f"checkpoint must have {NUM_CLASSES} classes, got {len(class_to_idx)}"
        )

    for key in ("feature_mean", "feature_std", "standardized_feature_mask"):
        if key not in checkpoint:
            raise SchemaContractError(f"checkpoint missing required key {key!r}")


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[GazeZoneTCN, Dict[str, Any], int]:
    """Load a v3 checkpoint and return ``(model, ckpt, window)``.

    Uses ``load_state_dict(..., strict=True)``. Model is moved to ``device`` and
    set to ``eval()``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint missing required key 'model_state_dict'")
    if "class_to_idx" not in checkpoint:
        raise KeyError("checkpoint missing required key 'class_to_idx'")

    _require_v3_checkpoint(checkpoint)

    feature_names: List[str] = list(checkpoint["feature_names"])
    class_to_idx: Dict[str, int] = dict(checkpoint["class_to_idx"])
    idx_to_class = {
        int(k): str(v) for k, v in dict(checkpoint.get("idx_to_class") or IDX_TO_CLASS).items()
    }
    if idx_to_class != dict(IDX_TO_CLASS):
        # Prefer authoritative contract when checkpoint is slightly off on types.
        idx_to_class = dict(IDX_TO_CLASS)
    checkpoint["idx_to_class"] = idx_to_class
    checkpoint["class_to_idx"] = dict(CLASS_TO_IDX)

    standardizer = FeatureStandardizer.from_checkpoint_dict(
        checkpoint, feature_names=feature_names
    )
    checkpoint["_standardizer"] = standardizer

    args = checkpoint.get("args") or {}
    window_size = int(
        checkpoint.get("window_frames")
        or checkpoint.get("window_size")
        or args.get("window_size")
        or DEFAULT_WINDOW_SIZE
    )
    if window_size < 1:
        raise ValueError(f"invalid window_size={window_size}")

    model_config = dict(checkpoint.get("model_config") or {})
    model = build_model(
        num_classes=NUM_CLASSES,
        input_dim=FEATURE_COUNT,
        channels=int(model_config.get("channels", 64)),
        kernel_size=int(model_config.get("kernel_size", 3)),
        dilations=tuple(model_config.get("dilations", (1, 2, 4))),
        dropout=float(model_config.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, window_size


def normalize_windows(
    windows: np.ndarray | torch.Tensor,
    standardizer: FeatureStandardizer,
) -> torch.Tensor:
    """Apply training standardization and re-apply validity masks.

    Accepts ``[T, F]`` or ``[B, T, F]`` raw geometry features.
    """
    if isinstance(windows, torch.Tensor):
        array = windows.detach().cpu().numpy()
    else:
        array = np.asarray(windows, dtype=np.float64)

    single = array.ndim == 2
    if single:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != FEATURE_COUNT:
        raise SchemaContractError(
            f"expected windows [B, T, {FEATURE_COUNT}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise SchemaContractError("feature windows contain non-finite values")

    out = np.empty_like(array, dtype=np.float32)
    for index in range(array.shape[0]):
        out[index] = standardizer.transform(array[index])
    tensor = torch.from_numpy(out)
    return tensor[0] if single else tensor


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return raw logits for ``features`` shaped ``[batch, T, F]``."""
    model.eval()
    features = to_device(features, device, dtype=torch.float32)
    if features.ndim != 3 or features.shape[-1] != FEATURE_COUNT:
        raise SchemaContractError(
            f"expected features [B, T, {FEATURE_COUNT}], got {tuple(features.shape)}"
        )
    return model(features)


@torch.inference_mode()
def predict_proba(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return softmax class probabilities for ``features`` ``[batch, T, F]``."""
    logits = predict_logits(model, features, device)
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def predict_label(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Return label / confidence / probabilities for a single window or batch."""
    probs = predict_proba(model, features, device)
    if probs.shape[-1] != NUM_CLASSES:
        raise SchemaContractError(
            f"model produced {probs.shape[-1]} classes; expected {NUM_CLASSES}"
        )
    row = probs[0]
    pred_index = int(row.argmax().item())
    return {
        "label": IDX_TO_CLASS[pred_index],
        "label_index": pred_index,
        "confidence": float(row[pred_index].item()),
        "probabilities": {
            IDX_TO_CLASS[i]: float(row[i].item()) for i in range(NUM_CLASSES)
        },
    }


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Score a DataLoader of ``(features, labels)`` batches (CLI / eval)."""
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    for features, labels in tqdm(loader, desc="infer", leave=False):
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)
        logits = model(features)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
    return {
        "preds": torch.cat(all_preds, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "probs": torch.cat(all_probs, dim=0),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--anns-dir", type=Path, default=ANNS_DIR)
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU_ID,
        help=f"CUDA device index (default: {DEFAULT_GPU_ID})",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write prediction summary JSON.",
    )
    parser.add_argument(
        "--smoke-val-window",
        action="store_true",
        help="Run a single real validation-window smoke prediction and exit.",
    )
    return parser.parse_args(argv)


def smoke_val_window(
    checkpoint_path: Path,
    *,
    anns_dir: Path,
    landmarks_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint and score one real validation window."""
    model, checkpoint, window_size = load_checkpoint(checkpoint_path, device)
    frame_dataset = DMDGazeFrameDataset(
        anns_dir=anns_dir,
        landmarks_dir=landmarks_dir,
        verbose=False,
    )
    _, val_dataset, _, info = build_train_val_window_datasets(
        frame_dataset,
        window_size=window_size,
        val_ratio=0.2,
        seed=int((checkpoint.get("split_info") or {}).get("seed", 42)),
        verbose=False,
    )
    if len(val_dataset) < 1:
        raise RuntimeError("validation split is empty; cannot smoke-test")
    # Pull raw (pre-standardizer) window by temporarily disabling standardizer.
    raw_standardizer = val_dataset.standardizer
    val_dataset.standardizer = None
    val_dataset.augment = False
    features, label = val_dataset[0]
    val_dataset.standardizer = raw_standardizer
    meta = val_dataset.get_metadata(0)
    standardizer: FeatureStandardizer = checkpoint["_standardizer"]
    normalized = normalize_windows(features.numpy(), standardizer).unsqueeze(0)
    prediction = predict_label(model, normalized, device)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "window_frames": window_size,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "session_key": meta["session_key"],
        "frame_index": meta["frame_index"],
        "true_label": meta["label_name"],
        "true_label_index": int(label.item()),
        **prediction,
        "val_sessions": info["val_sessions"],
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.gpu, require_cuda=not args.allow_cpu)
    configure_cuda(device)

    if args.smoke_val_window:
        smoke_val_window(
            args.checkpoint,
            anns_dir=args.anns_dir,
            landmarks_dir=args.landmarks_dir,
            device=device,
        )
        return 0

    model, checkpoint, window_size = load_checkpoint(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"feature_schema_version: {FEATURE_SCHEMA_VERSION}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"TCN window_size: {window_size}")
    print(f"num_classes: {NUM_CLASSES} {list(CLASS_TO_IDX)}")
    print(f"model device: {next(model.parameters()).device}")

    frame_dataset = DMDGazeFrameDataset(
        anns_dir=args.anns_dir,
        landmarks_dir=args.landmarks_dir,
        verbose=False,
    )
    _, val_dataset, _, _ = build_train_val_window_datasets(
        frame_dataset,
        window_size=window_size,
        val_ratio=0.2,
        seed=int((checkpoint.get("split_info") or {}).get("seed", 42)),
        verbose=False,
    )
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    outputs = run_inference(model, loader, device)
    preds = outputs["preds"]
    labels = outputs["labels"]
    accuracy = float((preds == labels).float().mean().item())
    print(f"val windows: {len(val_dataset)}")
    print(f"accuracy: {accuracy:.6f}")

    if args.output_json is not None:
        payload = {
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "arch": "tcn",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "window_size": window_size,
            "num_samples": int(len(val_dataset)),
            "accuracy": accuracy,
            "class_to_idx": dict(CLASS_TO_IDX),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
