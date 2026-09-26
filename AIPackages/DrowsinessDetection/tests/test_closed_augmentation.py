#!/usr/bin/env python3
"""Tests for closed-only augmentation and closed-window oversampling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dataloader import (  # noqa: E402
    ClosedAugmentationConfig,
    ClosedWindowAugmenter,
    DrowsinessWindowDataset,
    FeatureStandardizer,
    build_train_val_window_datasets,
    compute_closed_oversample_targets,
    compute_sample_weights,
    feature_index_map,
    fit_feature_bounds,
    make_train_val_loaders,
)
from feature_contract import DROWSINESS_FEATURE_NAMES, FEATURE_COUNT  # noqa: E402
from label_contract import CLASS_TO_IDX, map_raw_eyes_state  # noqa: E402

INDEX = feature_index_map()
WINDOW = 5


def _valid_frame(*, yaw: float = 0.1, left_ear: float = 0.25) -> np.ndarray:
    row = np.zeros(FEATURE_COUNT, dtype=np.float32)
    row[INDEX["face_detected"]] = 1.0
    row[INDEX["yaw"]] = yaw
    row[INDEX["pitch"]] = -0.05
    row[INDEX["roll"]] = 0.02
    row[INDEX["left_eye_valid"]] = 1.0
    row[INDEX["right_eye_valid"]] = 1.0
    row[INDEX["left_eye_aspect_ratio"]] = left_ear
    row[INDEX["right_eye_aspect_ratio"]] = 0.27
    row[INDEX["left_pupil_rel_x"]] = 0.45
    row[INDEX["left_pupil_rel_y"]] = 0.52
    row[INDEX["right_pupil_rel_x"]] = 0.55
    row[INDEX["right_pupil_rel_y"]] = 0.48
    return row


class FakeFrameDataset:
    def __init__(self, sessions: Dict[str, List[Dict[str, Any]]]) -> None:
        self.feature_names = list(DROWSINESS_FEATURE_NAMES)
        self.class_to_idx = dict(CLASS_TO_IDX)
        self.session_keys = list(sessions.keys())
        self.samples: List[Dict[str, Any]] = []
        for session_key, frames in sessions.items():
            for frame in frames:
                raw = frame["raw"]
                canonical = map_raw_eyes_state(raw)
                self.samples.append(
                    {
                        "session_key": session_key,
                        "frame_index": int(frame["frame_index"]),
                        "features": torch.tensor(frame["features"], dtype=torch.float32),
                        "raw_label_name": raw,
                        "canonical_label_name": canonical,
                        "label_index": (
                            int(CLASS_TO_IDX[canonical]) if canonical is not None else -1
                        ),
                        "label_name": canonical if canonical is not None else raw,
                        "is_excluded_transition": canonical is None,
                    }
                )


def _session(n: int, start: int, pattern: List[str]) -> List[Dict[str, Any]]:
    frames = []
    frame_id = start
    for i in range(n):
        frames.append(
            {
                "frame_index": frame_id,
                "raw": pattern[i % len(pattern)],
                "features": _valid_frame(yaw=0.05 + 0.01 * i, left_ear=0.2 + 0.001 * i),
            }
        )
        frame_id += 1
    return frames


def _synthetic() -> FakeFrameDataset:
    return FakeFrameDataset(
        {
            "sess_a": _session(30, 0, ["open", "open", "close", "undefined"]),
            "sess_b": _session(24, 100, ["close", "open", "undefined", "open"]),
            "sess_c": _session(20, 200, ["open", "close"]),
        }
    )


def test_closed_augmenter_preserves_shape_and_does_not_mutate_input() -> None:
    window = np.stack([_valid_frame(yaw=0.1 * i) for i in range(8)], axis=0)
    original = window.copy()
    frames = np.stack([_valid_frame() for _ in range(40)], axis=0)
    stdizer = FeatureStandardizer.fit(frames)
    std_by_name = {
        name: float(stdizer.std[i]) for i, name in enumerate(stdizer.feature_names)
    }
    bounds = fit_feature_bounds(frames, DROWSINESS_FEATURE_NAMES)
    aug = ClosedWindowAugmenter(
        config=ClosedAugmentationConfig(
            morphology_p=1.0,
            noise_p=1.0,
            bias_p=1.0,
            pupil_jitter_p=1.0,
            dropout_p=1.0,
        ),
        train_feature_std=std_by_name,
        feature_bounds=bounds,
        seed=0,
    )
    out = aug(window)
    assert out.shape == window.shape == (8, FEATURE_COUNT)
    assert np.array_equal(window, original)
    assert np.isfinite(out).all()
    for name in ("face_detected", "left_eye_valid", "right_eye_valid"):
        assert set(np.unique(out[:, INDEX[name]])).issubset({0.0, 1.0})
    for name in (
        "left_eye_aspect_ratio",
        "right_eye_aspect_ratio",
    ):
        assert np.all(out[:, INDEX[name]] >= 0.0)
    for name in (
        "left_pupil_rel_x",
        "left_pupil_rel_y",
        "right_pupil_rel_x",
        "right_pupil_rel_y",
    ):
        assert np.all(out[:, INDEX[name]] >= 0.0)
        assert np.all(out[:, INDEX[name]] <= 1.0)


def test_only_closed_windows_are_augmented_in_train_dataset() -> None:
    train_ds, val_ds, _, info = build_train_val_window_datasets(
        _synthetic(),
        window_size=WINDOW,
        val_ratio=0.34,
        seed=42,
        aug_config=ClosedAugmentationConfig(
            morphology_p=1.0,
            noise_p=0.0,
            bias_p=0.0,
            pupil_jitter_p=0.0,
            dropout_p=0.0,
        ),
        closed_augmentation=True,
        verbose=False,
    )
    assert train_ds.augment is True
    assert train_ds.closed_only_augment is True
    assert val_ds.augment is False

    # Disable standardizer to compare raw windows.
    train_ds.standardizer = None
    closed_idx = next(
        i for i, s in enumerate(train_ds.samples) if s["label_name"] == "closed"
    )
    open_idx = next(
        i for i, s in enumerate(train_ds.samples) if s["label_name"] == "open"
    )

    # Reconstruct unaugmented closed window (same padding rule as __getitem__).
    def _raw_window(ds: DrowsinessWindowDataset, index: int) -> np.ndarray:
        meta = ds.samples[index]
        feats = ds._session_features[meta["session_key"]]
        local = int(meta["local_index"])
        start = local - WINDOW + 1
        if start >= 0:
            return feats[start : local + 1].copy()
        available = feats[0 : local + 1]
        pad_n = WINDOW - int(available.shape[0])
        pad = np.repeat(available[:1], pad_n, axis=0)
        return np.concatenate([pad, available], axis=0)

    raw_closed = _raw_window(train_ds, closed_idx)
    train_ds.augmenter.reseed(123)
    x_closed, y_closed = train_ds[closed_idx]
    assert int(y_closed.item()) == CLASS_TO_IDX["closed"]
    assert not np.allclose(x_closed.numpy(), raw_closed, atol=1e-6)

    raw_open = _raw_window(train_ds, open_idx)
    x_open, y_open = train_ds[open_idx]
    assert int(y_open.item()) == CLASS_TO_IDX["open"]
    assert np.allclose(x_open.numpy(), raw_open, atol=1e-6)

    assert info["feature_count"] == FEATURE_COUNT
    assert info["feature_names"] == list(DROWSINESS_FEATURE_NAMES)


def test_oversample_targets_preserve_undefined_fraction() -> None:
    counts = {"closed": 100, "open": 800, "undefined": 100}
    targets = compute_closed_oversample_targets(counts, closed_sample_fraction=0.30)
    assert targets["undefined"] == pytest.approx(0.1)
    assert targets["closed"] == pytest.approx(0.9 * 0.30)
    assert targets["open"] == pytest.approx(0.9 * 0.70)
    assert abs(sum(targets.values()) - 1.0) < 1e-9


def test_weighted_sampler_wiring_no_shuffle() -> None:
    train_ds, val_ds, _, info = build_train_val_window_datasets(
        _synthetic(),
        window_size=WINDOW,
        val_ratio=0.34,
        seed=1,
        closed_augmentation=False,
        verbose=False,
    )
    train_loader, val_loader, info = make_train_val_loaders(
        train_ds,
        val_ds,
        batch_size=8,
        seed=1,
        split_info=info,
        closed_sample_fraction=0.30,
    )
    assert info["sampler_enabled"] is True
    assert train_loader.sampler is not None
    assert getattr(train_loader, "shuffle", False) is False or train_loader.sampler is not None
    # DataLoader with sampler must not also shuffle.
    assert train_loader.batch_sampler.sampler is train_loader.sampler
    assert val_loader.sampler is None or type(val_loader.sampler).__name__ == "SequentialSampler"


def test_closed_sample_fraction_range() -> None:
    with pytest.raises(ValueError):
        compute_closed_oversample_targets(
            {"closed": 1, "open": 1, "undefined": 0},
            closed_sample_fraction=0.51,
        )


def test_sample_weights_proportional_to_target_over_count() -> None:
    train_ds, _, _, _ = build_train_val_window_datasets(
        _synthetic(),
        window_size=WINDOW,
        val_ratio=0.34,
        seed=2,
        closed_augmentation=False,
        verbose=False,
    )
    weights, targets, counts = compute_sample_weights(
        train_ds, closed_sample_fraction=0.30
    )
    assert len(weights) == len(train_ds)
    for sample, weight in zip(train_ds.samples, weights.tolist()):
        name = sample["label_name"]
        expected = targets[name] / max(counts[name], 1)
        assert weight == pytest.approx(expected)
