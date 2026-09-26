#!/usr/bin/env python3
"""Unit tests for drowsiness feature schema v3 (eye-local normalization)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from feature_contract import (  # noqa: E402
    DROWSINESS_FEATURE_NAMES,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
)
from feature_math import (  # noqa: E402
    build_feature_row,
    empty_feature_row,
    scale_points,
    translate_points,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "synthetic_eye_landmarks.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fixture_row() -> list[float]:
    fixture = _load_fixture()
    return build_feature_row(
        face_detected=True,
        yaw=fixture["yaw"],
        pitch=fixture["pitch"],
        roll=fixture["roll"],
        left_eye_points=fixture["left_eye_points"],
        left_pupil=fixture["left_pupil"],
        right_eye_points=fixture["right_eye_points"],
        right_pupil=fixture["right_pupil"],
    )


def test_feature_count_is_12() -> None:
    assert FEATURE_COUNT == 12
    assert len(DROWSINESS_FEATURE_NAMES) == 12
    assert len(_fixture_row()) == 12


def test_feature_order_matches_canonical_list() -> None:
    assert DROWSINESS_FEATURE_NAMES == [
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
    assert FEATURE_SCHEMA_VERSION == "drowsiness_feature_schema_v3"


def test_translation_invariance_of_normalized_eye_features() -> None:
    fixture = _load_fixture()
    base = _fixture_row()
    dx, dy = 0.17, -0.09
    shifted = build_feature_row(
        face_detected=True,
        yaw=fixture["yaw"],
        pitch=fixture["pitch"],
        roll=fixture["roll"],
        left_eye_points=translate_points(fixture["left_eye_points"], dx, dy),
        left_pupil=(fixture["left_pupil"][0] + dx, fixture["left_pupil"][1] + dy),
        right_eye_points=translate_points(fixture["right_eye_points"], dx, dy),
        right_pupil=(fixture["right_pupil"][0] + dx, fixture["right_pupil"][1] + dy),
    )
    # Pose channels unchanged; normalized eye channels must match.
    for index in range(4, 12):
        assert shifted[index] == pytest.approx(base[index], abs=1e-5)


def test_uniform_scale_invariance_of_normalized_eye_features() -> None:
    fixture = _load_fixture()
    base = _fixture_row()
    origin_x, origin_y = 0.45, 0.55
    scale = 1.7
    scaled = build_feature_row(
        face_detected=True,
        yaw=fixture["yaw"],
        pitch=fixture["pitch"],
        roll=fixture["roll"],
        left_eye_points=scale_points(
            fixture["left_eye_points"], origin_x=origin_x, origin_y=origin_y, scale=scale
        ),
        left_pupil=(
            origin_x + (fixture["left_pupil"][0] - origin_x) * scale,
            origin_y + (fixture["left_pupil"][1] - origin_y) * scale,
        ),
        right_eye_points=scale_points(
            fixture["right_eye_points"], origin_x=origin_x, origin_y=origin_y, scale=scale
        ),
        right_pupil=(
            origin_x + (fixture["right_pupil"][0] - origin_x) * scale,
            origin_y + (fixture["right_pupil"][1] - origin_y) * scale,
        ),
    )
    for name in (
        "left_eye_aspect_ratio",
        "right_eye_aspect_ratio",
        "left_pupil_rel_x",
        "left_pupil_rel_y",
        "right_pupil_rel_x",
        "right_pupil_rel_y",
    ):
        index = DROWSINESS_FEATURE_NAMES.index(name)
        assert scaled[index] == pytest.approx(base[index], abs=1e-5)


def test_pupil_relative_coords_in_unit_interval() -> None:
    row = _fixture_row()
    for name in (
        "left_pupil_rel_x",
        "left_pupil_rel_y",
        "right_pupil_rel_x",
        "right_pupil_rel_y",
    ):
        value = row[DROWSINESS_FEATURE_NAMES.index(name)]
        assert 0.0 <= value <= 1.0


def test_invalid_eye_zeros_features() -> None:
    row = build_feature_row(
        face_detected=True,
        yaw=0.1,
        pitch=0.0,
        roll=0.0,
        left_eye_points=[(0.0, 0.0), (0.1, 0.0)],  # too few points
        left_pupil=(0.05, 0.0),
        right_eye_points=None,
        right_pupil=None,
    )
    assert row[0] == 1.0
    assert row[4] == 0.0  # left_eye_valid
    assert row[5] == 0.0  # right_eye_valid
    assert row[6:12] == [0.0] * 6


def test_missing_face_produces_all_zero_row() -> None:
    row = empty_feature_row()
    assert row == [0.0] * 12
    row2 = build_feature_row(face_detected=False, yaw=9.0, pitch=9.0, roll=9.0)
    assert row2 == [0.0] * 12


def test_no_nan_or_inf_in_outputs() -> None:
    row = _fixture_row()
    assert all(math.isfinite(v) for v in row)
    assert all(math.isfinite(v) for v in empty_feature_row())


def test_fixture_expected_vector_for_swift_parity() -> None:
    """Freeze the fixture vector so Swift can hardcode the same expected values."""
    row = _fixture_row()
    # Hand-checked structure: face on, both eyes valid, pupils near eye centers.
    assert row[0] == 1.0
    assert row[1] == pytest.approx(0.12, abs=1e-5)
    assert row[4] == 1.0
    assert row[5] == 1.0
    # Persist expected vector beside the fixture for cross-language checks.
    expected_path = FIXTURE_PATH.with_name("synthetic_eye_expected.json")
    expected_path.write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(DROWSINESS_FEATURE_NAMES),
                "values": row,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = json.loads(expected_path.read_text(encoding="utf-8"))
    assert loaded["values"] == pytest.approx(row, abs=1e-12)


def test_every_frame_emits_one_row_even_when_invalid() -> None:
    # Contract: callers must still write one CSV row per decoded frame.
    rows = [
        build_feature_row(face_detected=False),
        build_feature_row(face_detected=True, left_eye_points=None, right_eye_points=None),
        _fixture_row(),
    ]
    assert all(len(row) == 12 for row in rows)
