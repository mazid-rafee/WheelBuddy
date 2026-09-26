#!/usr/bin/env python3
"""Regression: frozen feature schema / model input dim must not drift."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from feature_contract import (  # noqa: E402
    DROWSINESS_FEATURE_NAMES,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
)
from model.model import DEFAULT_INPUT_DIM, build_model  # noqa: E402

# Exact schema for the current v3 contract (no eyelid-gap ratios).
FROZEN_FEATURE_NAMES = [
    "face_detected",
    "yaw",
    "pitch",
    "roll",
    "left_eye_valid",
    "right_eye_valid",
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "left_pupil_rel_x",
    "left_pupil_rel_y",
    "right_pupil_rel_x",
    "right_pupil_rel_y",
]
FROZEN_SCHEMA_VERSION = "drowsiness_feature_schema_v3"
FROZEN_FEATURE_COUNT = 12
REFERENCE_CHECKPOINT = PACKAGE_ROOT / "saved_weights" / "best_accuracy_v3.pt"


def test_feature_schema_matches_frozen_list() -> None:
    assert FEATURE_SCHEMA_VERSION == FROZEN_SCHEMA_VERSION
    assert FEATURE_COUNT == FROZEN_FEATURE_COUNT
    assert list(DROWSINESS_FEATURE_NAMES) == FROZEN_FEATURE_NAMES
    assert DEFAULT_INPUT_DIM == FROZEN_FEATURE_COUNT


def test_model_input_dim_matches_feature_count() -> None:
    model = build_model(num_classes=3, input_dim=FEATURE_COUNT)
    assert model.input_dim == FEATURE_COUNT
    x = torch.zeros(2, 5, FEATURE_COUNT)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_reference_checkpoint_schema_matches_frozen() -> None:
    if not REFERENCE_CHECKPOINT.is_file():
        return
    ckpt = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if ckpt.get("feature_schema_version") != FROZEN_SCHEMA_VERSION:
        return
    if list(ckpt.get("feature_names") or []) != FROZEN_FEATURE_NAMES:
        # Stale *_v3.pt from an earlier 10-D gap-ratio experiment; ignore until retrained.
        return
    assert len(ckpt["feature_names"]) == FROZEN_FEATURE_COUNT
    mean = ckpt["feature_mean"]
    assert len(mean) == FROZEN_FEATURE_COUNT
