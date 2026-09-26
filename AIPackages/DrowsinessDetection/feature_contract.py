#!/usr/bin/env python3
"""Authoritative drowsiness feature schema (v3).

Import ``DROWSINESS_FEATURE_NAMES`` / ``FEATURE_SCHEMA_VERSION`` from this module
everywhere. Do not copy the feature list by hand into other Python files.

v3 drops eyelid-gap-ratio channels and keeps pupil-relative coordinates.
"""

from __future__ import annotations

from typing import List, Sequence

# Explicit schema id shared with the iOS client and the inference API.
FEATURE_SCHEMA_VERSION = "drowsiness_feature_schema_v3"

# Reject legacy clients / checkpoints that still advertise older schemas.
LEGACY_SCHEMA_VERSIONS: frozenset[str | int] = frozenset(
    {1, "1", "v1", "schema_v1", "drowsiness_feature_schema_v2"}
)

EPS: float = 1e-6

# Minimum landmark points required for a usable eye region contour.
MIN_EYE_LANDMARK_POINTS: int = 4

DROWSINESS_FEATURE_NAMES: List[str] = [
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

FEATURE_COUNT: int = len(DROWSINESS_FEATURE_NAMES)

# Backward-compatible alias.
FEATURE_NAMES: Sequence[str] = DROWSINESS_FEATURE_NAMES

_BINARY_FEATURE_NAMES = frozenset(
    {
        "face_detected",
        "left_eye_valid",
        "right_eye_valid",
    }
)

# Landmark CSV suffix for v3 extractions (do not overwrite v1/v2 files).
CSV_SUFFIX = "_rgb_face.apple_drowsiness_v3.csv"
CSV_META_SUFFIX = "_rgb_face.apple_drowsiness_v3.meta.json"

# Bounding-box EAR (unchanged convention from the previous extractor / iOS app).
EAR_DEFINITION = (
    "bounding_box_aspect_ratio: "
    "(eye_max_y - eye_min_y) / max(eye_max_x - eye_min_x, eps)"
)

PUPIL_REL_DEFINITION = (
    "pupil_rel_x = (pupil_x - eye_min_x) / max(eye_width, eps); "
    "pupil_rel_y = (pupil_y - eye_min_y) / max(eye_height, eps); "
    "clamped to [0, 1] when valid"
)

assert FEATURE_COUNT == 12, FEATURE_COUNT
