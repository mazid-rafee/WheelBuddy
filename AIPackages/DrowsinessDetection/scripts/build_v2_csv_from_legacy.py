#!/usr/bin/env python3
"""Build interim ``*_v2.csv`` files from legacy Apple Vision CSVs.

True eye-local ``*_gap_ratio`` / ``*_pupil_rel_*`` require re-extraction with
landmark points. Until that lands, this writes v2-named columns by aliasing:

* ``left_eyelid_gap``  → ``left_eyelid_gap_ratio``  (legacy absolute gap)
* ``left_pupil_x``     → ``left_pupil_rel_x``       (legacy image-normalized)

Do **not** treat these as production-parity features for iPhone live inference
until Mac-side v2 re-extraction replaces them.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from feature_contract import CSV_SUFFIX, DROWSINESS_FEATURE_NAMES  # noqa: E402

LEGACY_SUFFIX = "_rgb_face.apple_drowsiness.csv"

# Legacy column → v2 column (identity or rename).
ALIASES = {
    "face_detected": "face_detected",
    "yaw": "yaw",
    "pitch": "pitch",
    "roll": "roll",
    "left_eye_valid": "left_eye_valid",
    "right_eye_valid": "right_eye_valid",
    "left_eye_aspect_ratio": "left_eye_aspect_ratio",
    "right_eye_aspect_ratio": "right_eye_aspect_ratio",
    "left_eyelid_gap": "left_eyelid_gap_ratio",
    "right_eyelid_gap": "right_eyelid_gap_ratio",
    "left_pupil_x": "left_pupil_rel_x",
    "left_pupil_y": "left_pupil_rel_y",
    "right_pupil_x": "right_pupil_rel_x",
    "right_pupil_y": "right_pupil_rel_y",
}


def _finite(raw: str | None, default: float = 0.0) -> float:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return value


def convert_file(src: Path, dst: Path) -> int:
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{src.name}: empty header")
        rows_out = []
        for row in reader:
            out = {
                "frame_index": int(str(row["frame_index"]).strip()),
            }
            for legacy, v2_name in ALIASES.items():
                value = _finite(row.get(legacy))
                if v2_name.endswith("_rel_x") or v2_name.endswith("_rel_y"):
                    value = min(1.0, max(0.0, value))
                if v2_name.endswith("_ratio") or "aspect_ratio" in v2_name:
                    value = max(0.0, value)
                out[v2_name] = value
            # Enforce binary + missing-face zeroing.
            face = 1.0 if out["face_detected"] >= 0.5 else 0.0
            out["face_detected"] = face
            if face < 0.5:
                for name in DROWSINESS_FEATURE_NAMES:
                    out[name] = 0.0
            else:
                for side in ("left", "right"):
                    valid_key = f"{side}_eye_valid"
                    valid = 1.0 if out[valid_key] >= 0.5 else 0.0
                    out[valid_key] = valid
                    if valid < 0.5:
                        for suffix in (
                            "eye_aspect_ratio",
                            "eyelid_gap_ratio",
                            "pupil_rel_x",
                            "pupil_rel_y",
                        ):
                            out[f"{side}_{suffix}"] = 0.0
            rows_out.append(out)

    fieldnames = ["frame_index", *DROWSINESS_FEATURE_NAMES]
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=Path("/data/quantization/zaima/DMD/drowsiness/landmarks"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing v2 CSVs.",
    )
    args = parser.parse_args()
    landmarks_dir = args.landmarks_dir.expanduser().resolve()
    converted = 0
    for src in sorted(landmarks_dir.glob(f"*{LEGACY_SUFFIX}")):
        stem = src.name[: -len(LEGACY_SUFFIX)]
        dst = landmarks_dir / f"{stem}{CSV_SUFFIX}"
        if dst.exists() and not args.force:
            print(f"skip existing {dst.name}")
            continue
        n = convert_file(src, dst)
        print(f"wrote {dst.name} rows={n} (legacy alias)")
        converted += 1
    print(f"converted={converted}")
    print(
        "WARNING: eyelid_gap_ratio / pupil_rel_* are aliased from legacy columns; "
        "re-extract with Apple Vision for production parity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
