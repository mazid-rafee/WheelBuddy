#!/usr/bin/env python3
"""Pure eye-local feature math for drowsiness schema v3.

Intentionally free of Apple frameworks so tests run on Linux CI. Formulas must
stay bit-compatible with ``DrowsinessFeatureExtractor.swift``.

v3 keeps EAR + pupil-relative coordinates and drops eyelid-gap-ratio channels.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from feature_contract import (
    DROWSINESS_FEATURE_NAMES,
    EPS,
    FEATURE_COUNT,
    MIN_EYE_LANDMARK_POINTS,
)

Point = Sequence[float]
PointList = Sequence[Point]


def points_are_finite(points: Iterable[Point]) -> bool:
    for point in points:
        if len(point) < 2:
            return False
        if not math.isfinite(float(point[0])) or not math.isfinite(float(point[1])):
            return False
    return True


def eye_bounds(
    eye_points: PointList,
) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(min_x, max_x, min_y, max_y)`` or ``None`` if invalid."""
    if len(eye_points) < MIN_EYE_LANDMARK_POINTS or not points_are_finite(eye_points):
        return None
    xs = [float(p[0]) for p in eye_points]
    ys = [float(p[1]) for p in eye_points]
    return min(xs), max(xs), min(ys), max(ys)


def bounding_box_ear(
    eye_min_x: float,
    eye_max_x: float,
    eye_min_y: float,
    eye_max_y: float,
) -> float:
    """Existing EAR convention: bbox height / width."""
    width = eye_max_x - eye_min_x
    height = eye_max_y - eye_min_y
    if width <= EPS or height <= EPS:
        return float("nan")
    if not all(math.isfinite(v) for v in (width, height)):
        return float("nan")
    return height / width


def pupil_relative(
    pupil_x: float,
    pupil_y: float,
    eye_min_x: float,
    eye_max_x: float,
    eye_min_y: float,
    eye_max_y: float,
) -> Tuple[float, float, bool]:
    width = eye_max_x - eye_min_x
    height = eye_max_y - eye_min_y
    if width <= EPS or height <= EPS:
        return 0.0, 0.0, False
    if not all(
        math.isfinite(v)
        for v in (pupil_x, pupil_y, eye_min_x, eye_max_x, eye_min_y, eye_max_y)
    ):
        return 0.0, 0.0, False
    rel_x = (pupil_x - eye_min_x) / max(width, EPS)
    rel_y = (pupil_y - eye_min_y) / max(height, EPS)
    if not math.isfinite(rel_x) or not math.isfinite(rel_y):
        return 0.0, 0.0, False
    rel_x = min(1.0, max(0.0, rel_x))
    rel_y = min(1.0, max(0.0, rel_y))
    return rel_x, rel_y, True


def compute_one_eye(
    eye_points: Optional[PointList],
    pupil_point: Optional[Point],
) -> Dict[str, float]:
    """Return eye_valid, EAR, and pupil_rel_x/y for one eye."""
    invalid = {
        "eye_valid": 0.0,
        "eye_aspect_ratio": 0.0,
        "pupil_rel_x": 0.0,
        "pupil_rel_y": 0.0,
    }
    if eye_points is None:
        return invalid
    bounds = eye_bounds(eye_points)
    if bounds is None:
        return invalid
    eye_min_x, eye_max_x, eye_min_y, eye_max_y = bounds
    eye_width = eye_max_x - eye_min_x
    eye_height = eye_max_y - eye_min_y
    if eye_width <= EPS or eye_height <= EPS:
        return invalid

    ear = bounding_box_ear(eye_min_x, eye_max_x, eye_min_y, eye_max_y)
    if not math.isfinite(ear):
        return invalid

    if pupil_point is None or len(pupil_point) < 2:
        return invalid
    raw_x = float(pupil_point[0])
    raw_y = float(pupil_point[1])
    rel_x, rel_y, ok = pupil_relative(
        raw_x, raw_y, eye_min_x, eye_max_x, eye_min_y, eye_max_y
    )
    if not ok:
        return invalid

    return {
        "eye_valid": 1.0,
        "eye_aspect_ratio": float(ear),
        "pupil_rel_x": float(rel_x),
        "pupil_rel_y": float(rel_y),
    }


def empty_feature_row() -> List[float]:
    """No-face / fully masked row: face_detected=0 and all other features 0."""
    return [0.0] * FEATURE_COUNT


def build_feature_row(
    *,
    face_detected: bool,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    left_eye_points: Optional[PointList] = None,
    left_pupil: Optional[Point] = None,
    right_eye_points: Optional[PointList] = None,
    right_pupil: Optional[Point] = None,
) -> List[float]:
    """Assemble the canonical 12-D model input vector."""
    if not face_detected:
        return empty_feature_row()

    left = compute_one_eye(left_eye_points, left_pupil)
    right = compute_one_eye(right_eye_points, right_pupil)

    values = {
        "face_detected": 1.0,
        "yaw": float(yaw) if math.isfinite(float(yaw)) else 0.0,
        "pitch": float(pitch) if math.isfinite(float(pitch)) else 0.0,
        "roll": float(roll) if math.isfinite(float(roll)) else 0.0,
        "left_eye_valid": left["eye_valid"],
        "right_eye_valid": right["eye_valid"],
        "left_eye_aspect_ratio": left["eye_aspect_ratio"],
        "right_eye_aspect_ratio": right["eye_aspect_ratio"],
        "left_pupil_rel_x": left["pupil_rel_x"],
        "left_pupil_rel_y": left["pupil_rel_y"],
        "right_pupil_rel_x": right["pupil_rel_x"],
        "right_pupil_rel_y": right["pupil_rel_y"],
    }
    row = [float(values[name]) for name in DROWSINESS_FEATURE_NAMES]
    for index, value in enumerate(row):
        if not math.isfinite(value):
            row[index] = 0.0
    assert len(row) == FEATURE_COUNT
    return row


def translate_points(points: PointList, dx: float, dy: float) -> List[Tuple[float, float]]:
    return [(float(p[0]) + dx, float(p[1]) + dy) for p in points]


def scale_points(
    points: PointList,
    *,
    origin_x: float,
    origin_y: float,
    scale: float,
) -> List[Tuple[float, float]]:
    return [
        (origin_x + (float(p[0]) - origin_x) * scale,
         origin_y + (float(p[1]) - origin_y) * scale)
        for p in points
    ]
