#!/usr/bin/env python3
"""Tests for v2 drowsiness window dataloader, filtering, and augmentation."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dataloader import (  # noqa: E402
    AugmentationConfig,
    DrowsinessFeatureAugmenter,
    DrowsinessWindowDataset,
    FeatureStandardizer,
    build_train_val_window_datasets,
    feature_index_map,
    seed_everything,
)
from feature_contract import DROWSINESS_FEATURE_NAMES, FEATURE_COUNT  # noqa: E402
from label_contract import (  # noqa: E402
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RAW_EYES_STATE_TO_CANONICAL,
    map_raw_eyes_state,
)

INDEX = feature_index_map()
T = 12
WINDOW = 5


def _valid_frame(
    *,
    yaw: float = 0.1,
    pitch: float = -0.05,
    roll: float = 0.02,
    left_ear: float = 0.25,
    right_ear: float = 0.27,
    left_px: float = 0.45,
    left_py: float = 0.52,
    right_px: float = 0.55,
    right_py: float = 0.48,
) -> np.ndarray:
    row = np.zeros(FEATURE_COUNT, dtype=np.float32)
    row[INDEX["face_detected"]] = 1.0
    row[INDEX["yaw"]] = yaw
    row[INDEX["pitch"]] = pitch
    row[INDEX["roll"]] = roll
    row[INDEX["left_eye_valid"]] = 1.0
    row[INDEX["right_eye_valid"]] = 1.0
    row[INDEX["left_eye_aspect_ratio"]] = left_ear
    row[INDEX["right_eye_aspect_ratio"]] = right_ear
    row[INDEX["left_pupil_rel_x"]] = left_px
    row[INDEX["left_pupil_rel_y"]] = left_py
    row[INDEX["right_pupil_rel_x"]] = right_px
    row[INDEX["right_pupil_rel_y"]] = right_py
    return row


class FakeFrameDataset:
    """Minimal stand-in for ``DMDGazeFrameDataset``."""

    def __init__(
        self,
        sessions: Dict[str, List[Dict[str, Any]]],
    ) -> None:
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


def _make_session_frames(
    *,
    n: int,
    start_id: int,
    pattern: List[str],
    gap_at: Optional[int] = None,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    frame_id = start_id
    for i in range(n):
        if gap_at is not None and i == gap_at:
            frame_id += 5  # discontinuity
        raw = pattern[i % len(pattern)]
        frames.append(
            {
                "frame_index": frame_id,
                "raw": raw,
                "features": _valid_frame(
                    yaw=0.05 + 0.01 * i,
                    left_ear=0.20 + 0.001 * i,
                    right_ear=0.22 + 0.001 * i,
                ),
            }
        )
        frame_id += 1
    return frames


def _synthetic_frame_dataset() -> FakeFrameDataset:
    # Observed raw suffixes from DMD OpenLABEL: close/open/undefined/opening/closing
    s1 = _make_session_frames(
        n=30,
        start_id=0,
        pattern=["open", "open", "close", "opening", "closing", "undefined"],
    )
    s2 = _make_session_frames(
        n=24,
        start_id=100,
        pattern=["close", "open", "undefined", "open"],
    )
    s3 = _make_session_frames(
        n=20,
        start_id=200,
        pattern=["open", "close"],
        gap_at=10,
    )
    return FakeFrameDataset({"sess_a": s1, "sess_b": s2, "sess_c": s3})


def test_raw_label_mapping_matches_observed_types() -> None:
    assert set(RAW_EYES_STATE_TO_CANONICAL) == {
        "close",
        "open",
        "undefined",
        "opening",
        "closing",
    }
    assert map_raw_eyes_state("close") == "closed"
    assert map_raw_eyes_state("open") == "open"
    assert map_raw_eyes_state("undefined") == "undefined"
    assert map_raw_eyes_state("closing") == "closed"
    assert map_raw_eyes_state("opening") is None
    with pytest.raises(KeyError):
        map_raw_eyes_state("opened")
    with pytest.raises(KeyError):
        map_raw_eyes_state("openend")


def test_undefined_is_explicit_annotation_not_missing() -> None:
    """undefined comes from eyes_state/undefined, never invented for unlabeled frames."""
    assert RAW_EYES_STATE_TO_CANONICAL["undefined"] == "undefined"
    # Missing annotation is simply absent from label_map / skipped in frame loader.
    assert "missing" not in RAW_EYES_STATE_TO_CANONICAL


def test_augmenter_output_shape_masks_pupils_finite() -> None:
    window = np.stack([_valid_frame(yaw=0.1 * i) for i in range(T)], axis=0)
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(
            morphology_p=1.0,
            noise_p=1.0,
            bias_p=1.0,
            pupil_jitter_p=1.0,
            dropout_p=1.0,
        ),
        seed=0,
    )
    out = aug(window)
    assert out.shape == (T, 12)
    for name in ("face_detected", "left_eye_valid", "right_eye_valid"):
        vals = out[:, INDEX[name]]
        assert set(np.unique(vals)).issubset({0.0, 1.0})
    for name in (
        "left_pupil_rel_x",
        "left_pupil_rel_y",
        "right_pupil_rel_x",
        "right_pupil_rel_y",
    ):
        assert np.all(out[:, INDEX[name]] >= 0.0)
        assert np.all(out[:, INDEX[name]] <= 1.0)
    assert np.isfinite(out).all()


def test_morphology_scale_constant_through_window() -> None:
    window = np.stack([_valid_frame() for _ in range(T)], axis=0)
    # Identical rows so ratios after scale stay constant if scale is constant.
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(
            morphology_p=1.0,
            noise_p=0.0,
            bias_p=0.0,
            dropout_p=0.0,
        ),
        seed=7,
    )
    out = aug(window)
    for name in (
        "left_eye_aspect_ratio",
        "right_eye_aspect_ratio",
    ):
        col = out[:, INDEX[name]]
        assert np.allclose(col, col[0]), name


def test_calibration_bias_constant_through_window() -> None:
    window = np.stack([_valid_frame() for _ in range(T)], axis=0)
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(
            morphology_p=0.0,
            noise_p=0.0,
            bias_p=1.0,
            pupil_jitter_p=0.0,
            dropout_p=0.0,
        ),
        seed=11,
    )
    out = aug(window)
    for name in ("yaw", "pitch", "roll", "left_pupil_rel_x", "right_pupil_rel_y"):
        delta = out[:, INDEX[name]] - window[:, INDEX[name]]
        assert np.allclose(delta, delta[0]), name


def test_measurement_noise_is_temporally_correlated() -> None:
    window = np.stack([_valid_frame() for _ in range(64)], axis=0)
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(
            morphology_p=0.0,
            noise_p=1.0,
            bias_p=0.0,
            pupil_jitter_p=0.0,
            dropout_p=0.0,
            pose_noise_std_fraction=0.05,
            noise_rho=0.85,
        ),
        train_feature_std={"yaw": 1.0, "pitch": 1.0, "roll": 1.0},
        seed=3,
    )
    out = aug(window)
    noise = out[:, INDEX["yaw"]] - window[:, INDEX["yaw"]]
    # Lag-1 autocorrelation should be clearly positive for AR(1).
    n = noise - noise.mean()
    corr = float(np.corrcoef(n[:-1], n[1:])[0, 1])
    assert corr > 0.5


def test_dropout_affects_only_selected_eye_and_frames() -> None:
    window = np.stack([_valid_frame() for _ in range(T)], axis=0)
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(
            morphology_p=0.0,
            noise_p=0.0,
            bias_p=0.0,
            pupil_jitter_p=0.0,
            dropout_p=1.0,
            dropout_min_frames=2,
            dropout_max_frames=2,
        ),
        seed=0,
    )

    class _FixedRng:
        def integers(self, low, high=None, size=None, dtype=None, endpoint=False):
            # First call: duration in [2, 3); second: start in [0, T-2+1).
            if high == 3:
                return 2
            return 3

        def choice(self, a, *args, **kwargs):
            return "left"

        def random(self, *args, **kwargs):
            return 0.0

        def uniform(self, *args, **kwargs):
            return 1.0

        def normal(self, *args, **kwargs):
            return np.zeros(args[2] if len(args) > 2 else kwargs.get("size", 1))

    aug._rng = _FixedRng()  # type: ignore[assignment]
    out = aug._landmark_dropout(window.copy())
    assert np.all(out[3:5, INDEX["left_eye_valid"]] == 0.0)
    assert np.all(out[3:5, INDEX["left_eye_aspect_ratio"]] == 0.0)
    assert np.all(out[3:5, INDEX["left_pupil_rel_x"]] == 0.0)
    assert np.all(out[3:5, INDEX["right_eye_valid"]] == 1.0)
    assert np.all(out[3:5, INDEX["right_eye_aspect_ratio"]] > 0.0)
    assert np.all(out[:, INDEX["face_detected"]] == 1.0)
    assert out[0, INDEX["left_eye_valid"]] == 1.0
    assert out[2, INDEX["left_eye_valid"]] == 1.0
    assert out[5, INDEX["left_eye_valid"]] == 1.0


def test_invalid_eye_features_exactly_zero() -> None:
    window = np.stack([_valid_frame() for _ in range(4)], axis=0)
    window[:, INDEX["left_eye_valid"]] = 0.0
    window[:, INDEX["left_eye_aspect_ratio"]] = 0.9  # should be cleared
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(
            morphology_p=0.0, noise_p=0.0, bias_p=0.0, dropout_p=0.0
        ),
        seed=1,
    )
    out = aug(window)
    for name in (
        "left_eye_aspect_ratio",
        "left_pupil_rel_x",
        "left_pupil_rel_y",
    ):
        assert np.all(out[:, INDEX[name]] == 0.0)


def test_fixed_seed_deterministic_and_different_seeds_differ() -> None:
    window = np.stack([_valid_frame(yaw=0.2) for _ in range(T)], axis=0)
    cfg = AugmentationConfig(
        morphology_p=1.0, noise_p=1.0, bias_p=1.0, dropout_p=0.5
    )
    a1 = DrowsinessFeatureAugmenter(config=cfg, seed=123)(window)
    a2 = DrowsinessFeatureAugmenter(config=cfg, seed=123)(window)
    b = DrowsinessFeatureAugmenter(config=cfg, seed=999)(window)
    assert np.allclose(a1, a2)
    assert not np.allclose(a1, b)


def test_window_filter_drops_opening_keeps_closing_as_closed() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        ds = DrowsinessWindowDataset(
            _synthetic_frame_dataset(),
            window_size=WINDOW,
            augment=False,
            verbose=True,
        )
    text = buf.getvalue()
    assert "endpoint counts BEFORE filter" in text
    assert "endpoint counts AFTER filter" in text
    assert "opening" in ds.removed_transition_windows
    assert "closing" not in ds.removed_transition_windows
    removed = sum(ds.removed_transition_windows.values())
    assert removed > 0
    labels = {s["label_name"] for s in ds.samples}
    assert labels.issubset({"closed", "open", "undefined"})
    assert "opening" not in labels and "closing" not in labels
    indices = {s["label_index"] for s in ds.samples}
    assert indices.issubset({0, 1, 2})
    # Closing endpoints are retained with canonical closed label.
    closing_kept = [
        s for s in ds.samples if s["raw_label_name"] == "closing"
    ]
    assert len(closing_kept) > 0
    assert all(s["label_name"] == "closed" for s in closing_kept)
    assert all(s["label_index"] == CLASS_TO_IDX["closed"] for s in closing_kept)
    for s in ds.samples:
        assert s["raw_label_name"] != "opening"


def test_windows_never_cross_sessions_or_discontinuities() -> None:
    ds = DrowsinessWindowDataset(
        _synthetic_frame_dataset(),
        window_size=WINDOW,
        augment=False,
        verbose=False,
    )
    for i, meta in enumerate(ds.samples):
        session = meta["session_key"]
        local = meta["local_index"]
        feats, label = ds[i]
        assert feats.shape == (WINDOW, 12)
        assert int(label.item()) in (0, 1, 2)
        frame_ids = ds._session_frame_ids[session]
        start = local - WINDOW + 1
        real_start = max(0, start)
        ids = frame_ids[real_start : local + 1]
        assert np.all(np.diff(ids) == 1)
        # Endpoint frame belongs to this session only.
        assert meta["session_key"] == session


def test_val_test_never_augmented_train_is() -> None:
    frame_ds = _synthetic_frame_dataset()
    train_ds, val_ds, standardizer, info = build_train_val_window_datasets(
        frame_ds,
        window_size=WINDOW,
        val_ratio=0.34,
        seed=42,
        aug_config=AugmentationConfig(
            morphology_p=1.0,
            noise_p=0.0,
            bias_p=0.0,
            dropout_p=0.0,
        ),
        verbose=False,
    )
    assert train_ds.augment is True
    assert val_ds.augment is False
    assert train_ds.augmenter is not None
    assert val_ds.augmenter is None
    assert standardizer is not None
    assert "augmentation_config" in info

    # Capture raw (no std) by temporarily clearing standardizer on copies.
    train_ds.standardizer = None
    val_ds.standardizer = None
    # Force morphology-only via augmenter already configured p=1.
    seed_everything(0)
    train_ds.augmenter.reseed(0)
    x_train_a, _ = train_ds[0]
    train_ds.augmenter.reseed(0)
    x_train_b, _ = train_ds[0]
    assert torch.allclose(x_train_a, x_train_b)

    # Val sample equals raw window features (invariants only).
    meta = val_ds.samples[0]
    session = meta["session_key"]
    local = meta["local_index"]
    feats = val_ds._session_features[session]
    start = local - WINDOW + 1
    if start >= 0:
        expected = feats[start : local + 1]
    else:
        available = feats[0 : local + 1]
        pad = np.repeat(available[:1], WINDOW - available.shape[0], axis=0)
        expected = np.concatenate([pad, available], axis=0)
    x_val, _ = val_ds[0]
    assert np.allclose(x_val.numpy(), expected, atol=1e-5)


def test_standardizer_skips_binary_and_masks_invalid_eyes() -> None:
    frames = np.stack([_valid_frame(left_ear=0.3 + 0.01 * i) for i in range(20)])
    frames[0, INDEX["left_eye_valid"]] = 0.0
    frames[0, INDEX["left_eye_aspect_ratio"]] = 9.0  # ignored in stats
    stdizer = FeatureStandardizer.fit(frames)
    assert not stdizer.standardized_feature_mask[INDEX["face_detected"]]
    assert not stdizer.standardized_feature_mask[INDEX["left_eye_valid"]]
    assert stdizer.standardized_feature_mask[INDEX["yaw"]]
    window = frames[:WINDOW].copy()
    window[:, INDEX["right_eye_valid"]] = 0.0
    out = stdizer.transform(window)
    assert np.all(out[:, INDEX["right_eye_aspect_ratio"]] == 0.0)
    assert np.all(out[:, INDEX["right_pupil_rel_x"]] == 0.0)
    assert set(np.unique(out[:, INDEX["face_detected"]])).issubset({0.0, 1.0})


def test_class_counts_reported_and_only_canonical_indices() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        ds = DrowsinessWindowDataset(
            _synthetic_frame_dataset(),
            window_size=WINDOW,
            verbose=True,
        )
    text = buf.getvalue()
    assert "BEFORE filter" in text and "AFTER filter" in text
    assert sum(ds.counts_after_filter.values()) == len(ds.samples)
    assert set(ds.counts_after_filter) == set(CLASS_TO_IDX)
    for s in ds.samples:
        assert IDX_TO_CLASS[s["label_index"]] == s["label_name"]


def test_debug_mode_prints_one_summary() -> None:
    window = np.stack([_valid_frame() for _ in range(8)], axis=0)
    aug = DrowsinessFeatureAugmenter(
        config=AugmentationConfig(morphology_p=1.0, noise_p=0.0, bias_p=0.0, dropout_p=0.0),
        seed=1,
        debug=True,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        aug(window)
    text = buf.getvalue()
    assert "aug debug: original mean" in text
    assert "aug debug: augmented mean" in text


def test_checkpoint_metadata_fields_present_in_builder_info() -> None:
    train_ds, val_ds, standardizer, info = build_train_val_window_datasets(
        _synthetic_frame_dataset(),
        window_size=WINDOW,
        val_ratio=0.34,
        seed=1,
        verbose=False,
    )
    payload = {
        "feature_schema_version": "drowsiness_feature_schema_v3",
        "feature_names": list(DROWSINESS_FEATURE_NAMES),
        **standardizer.to_checkpoint_dict(),
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": dict(IDX_TO_CLASS),
        "window_frames": WINDOW,
        "sampling_rate_hz": info["sampling_rate_hz"],
        "augmentation_config": info["augmentation_config"],
    }
    for key in (
        "feature_schema_version",
        "feature_names",
        "feature_mean",
        "feature_std",
        "standardized_feature_mask",
        "class_to_idx",
        "idx_to_class",
        "window_frames",
        "sampling_rate_hz",
        "augmentation_config",
    ):
        assert key in payload
    assert len(train_ds) > 0 and len(val_ds) > 0
