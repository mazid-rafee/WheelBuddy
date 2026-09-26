#!/usr/bin/env python3
"""Match DMD drowsiness annotation JSONs to landmark CSVs and validate alignment.

Ann files look like:
  gA_5_s5_2019-03-13T09;06;49+01;00_rgb_ann_drowsiness.json

Landmark CSVs look like:
  gA_5_s5_2019-03-13T09;06;49+01;00_rgb_face.apple_drowsiness.csv

Pairs are matched on the shared session stem
(everything before ``_rgb_ann_drowsiness`` / ``_rgb_face.apple_drowsiness``).

For each matched pair, compare:
  len(openlabel["frames"]) == landmark CSV data rows
and also validate that the exact ``frame_index`` sequences match.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ANNS_DIR = Path("/data/quantization/zaima/DMD/drowsiness/anns")
LANDMARKS_DIR = Path("/data/quantization/zaima/DMD/drowsiness/landmarks")

ANN_SUFFIX = "_rgb_ann_drowsiness.json"
CSV_SUFFIX = "_rgb_face.apple_drowsiness.csv"
ID_PREVIEW_LIMIT = 30


def session_key_from_ann(path: Path) -> Optional[str]:
    name = path.name
    if not name.endswith(ANN_SUFFIX):
        return None
    return name[: -len(ANN_SUFFIX)]


def session_key_from_csv(path: Path) -> Optional[str]:
    name = path.name
    if not name.endswith(CSV_SUFFIX):
        return None
    return name[: -len(CSV_SUFFIX)]


def format_id_list(ids: Sequence[int], limit: int = ID_PREVIEW_LIMIT) -> str:
    """Format an ID list, truncating long sequences after ``limit`` entries."""
    values = list(ids)
    if not values:
        return "(none)"
    if len(values) <= limit:
        return ", ".join(str(v) for v in values)
    shown = ", ".join(str(v) for v in values[:limit])
    remaining = len(values) - limit
    return f"{shown} ... and {remaining} more"


def get_annotation_frame_ids(ann_path: Path) -> List[int]:
    """Return sorted integer frame IDs from ``openlabel["frames"]``."""
    with ann_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        frames: Any = payload["openlabel"]["frames"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{ann_path.name}: missing openlabel.frames ({exc})"
        ) from exc

    if isinstance(frames, dict):
        frame_ids: List[int] = []
        for key in frames.keys():
            try:
                frame_ids.append(int(key))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{ann_path.name}: non-integer frames key {key!r}"
                ) from exc
        return sorted(frame_ids)

    if isinstance(frames, list):
        return list(range(len(frames)))

    raise ValueError(
        f"{ann_path.name}: openlabel.frames must be a JSON object or array, "
        f"got {type(frames).__name__}"
    )


def get_csv_frame_ids(csv_path: Path) -> List[int]:
    """Return ``frame_index`` values from a landmark CSV, preserving row order."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "frame_index" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path.name}: missing required column 'frame_index'"
            )

        frame_ids: List[int] = []
        for row_number, row in enumerate(reader, start=1):
            raw = row.get("frame_index")
            if raw is None or str(raw).strip() == "":
                raise ValueError(
                    f"{csv_path.name}: empty or missing frame_index "
                    f"at data-row {row_number}"
                )
            try:
                frame_ids.append(int(str(raw).strip()))
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path.name}: non-integer frame_index {raw!r} "
                    f"at data-row {row_number}"
                ) from exc
        return frame_ids


def count_frames_objects(ann_path: Path) -> int:
    """Return ``len(openlabel["frames"])`` (dict or list)."""
    return len(get_annotation_frame_ids(ann_path))


def count_csv_rows(csv_path: Path) -> int:
    """Return the number of CSV data rows via ``frame_index`` parsing."""
    return len(get_csv_frame_ids(csv_path))


def is_contiguous(ids: Sequence[int]) -> bool:
    """True if ``ids`` is empty or equals ``min..max`` ascending with no gaps/dups."""
    if not ids:
        return True
    return list(ids) == list(range(ids[0], ids[-1] + 1))


def duplicate_ids(ids: Sequence[int]) -> List[int]:
    """Return sorted IDs that appear more than once."""
    counts = Counter(ids)
    return sorted(value for value, count in counts.items() if count > 1)


def classify_missing_csv_ids(
    missing_ids: Sequence[int],
    csv_ids: Sequence[int],
) -> str:
    """Classify where annotation IDs missing from the CSV fall relative to CSV span.

    Returns one of: none, leading, trailing, internal, mixed.
    """
    if not missing_ids:
        return "none"
    if not csv_ids:
        # No CSV frames: all missing IDs are outside any CSV span.
        # Treat as mixed when multiple IDs exist without a CSV reference range;
        # a single missing block with empty CSV is reported as leading.
        return "leading" if missing_ids else "none"

    csv_min = min(csv_ids)
    csv_max = max(csv_ids)
    leading = False
    trailing = False
    internal = False
    for frame_id in missing_ids:
        if frame_id < csv_min:
            leading = True
        elif frame_id > csv_max:
            trailing = True
        else:
            internal = True

    region_count = int(leading) + int(trailing) + int(internal)
    if region_count > 1:
        return "mixed"
    if leading:
        return "leading"
    if trailing:
        return "trailing"
    return "internal"


def index_anns(anns_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(anns_dir.iterdir()):
        if not path.is_file():
            continue
        key = session_key_from_ann(path)
        if key is None:
            continue
        if key in index:
            raise ValueError(f"duplicate ann session key {key!r}")
        index[key] = path
    return index


def index_csvs(landmarks_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(landmarks_dir.iterdir()):
        if not path.is_file():
            continue
        key = session_key_from_csv(path)
        if key is None:
            continue
        if key in index:
            raise ValueError(f"duplicate landmark session key {key!r}")
        index[key] = path
    return index


def analyze_pair(
    ann_path: Path,
    csv_path: Path,
) -> Dict[str, Any]:
    ann_ids = get_annotation_frame_ids(ann_path)
    csv_ids = get_csv_frame_ids(csv_path)
    ann_unique_ids = sorted(set(ann_ids))
    csv_unique_ids = sorted(set(csv_ids))
    duplicate_ann_ids = duplicate_ids(ann_ids)
    duplicate_csv_ids = duplicate_ids(csv_ids)

    ann_set = set(ann_ids)
    csv_set = set(csv_ids)
    missing_csv_ids = sorted(ann_set - csv_set)
    extra_csv_ids = sorted(csv_set - ann_set)

    ann_contiguous = is_contiguous(ann_ids)
    csv_contiguous = is_contiguous(csv_ids)
    exact_sequence_match = ann_ids == csv_ids
    missing_classification = classify_missing_csv_ids(missing_csv_ids, csv_ids)

    counts_equal = len(ann_ids) == len(csv_ids)
    ok = (
        counts_equal
        and not duplicate_ann_ids
        and not duplicate_csv_ids
        and ann_contiguous
        and csv_contiguous
        and exact_sequence_match
    )

    return {
        "ann_ids": ann_ids,
        "csv_ids": csv_ids,
        "ann_unique_ids": ann_unique_ids,
        "csv_unique_ids": csv_unique_ids,
        "duplicate_ann_ids": duplicate_ann_ids,
        "duplicate_csv_ids": duplicate_csv_ids,
        "missing_csv_ids": missing_csv_ids,
        "extra_csv_ids": extra_csv_ids,
        "ann_contiguous": ann_contiguous,
        "csv_contiguous": csv_contiguous,
        "exact_sequence_match": exact_sequence_match,
        "missing_classification": missing_classification,
        "ok": ok,
    }


def _first_last(ids: Sequence[int]) -> Tuple[str, str]:
    if not ids:
        return ("n/a", "n/a")
    return (str(ids[0]), str(ids[-1]))


def main(
    anns_dir: Path = ANNS_DIR,
    landmarks_dir: Path = LANDMARKS_DIR,
) -> int:
    if not anns_dir.is_dir():
        print(f"ERROR: anns_dir does not exist: {anns_dir}", file=sys.stderr)
        return 2
    if not landmarks_dir.is_dir():
        print(
            f"ERROR: landmarks_dir does not exist: {landmarks_dir}",
            file=sys.stderr,
        )
        return 2

    anns = index_anns(anns_dir)
    csvs = index_csvs(landmarks_dir)

    matched_keys = sorted(set(anns) & set(csvs))
    ann_only = sorted(set(anns) - set(csvs))
    csv_only = sorted(set(csvs) - set(anns))

    print(f"anns_dir:       {anns_dir}")
    print(f"landmarks_dir:  {landmarks_dir}")
    print(f"ann files:      {len(anns)}")
    print(f"landmark csvs:  {len(csvs)}")
    print(f"matched pairs:  {len(matched_keys)}")
    print(f"ann-only:       {len(ann_only)}")
    print(f"csv-only:       {len(csv_only)}")
    print(
        "compare:        len(openlabel['frames']) == landmark CSV data rows "
        "AND exact frame_index sequence"
    )
    print()

    if ann_only:
        print("Ann files with no matching landmark CSV:")
        for key in ann_only:
            print(f"  - {anns[key].name}")
        print()
    if csv_only:
        print("Landmark CSVs with no matching ann JSON:")
        for key in csv_only:
            print(f"  - {csvs[key].name}")
        print()

    matches: List[str] = []
    mismatches: List[Tuple[str, Dict[str, Any]]] = []
    errors: List[str] = []

    for key in matched_keys:
        ann_path = anns[key]
        csv_path = csvs[key]
        try:
            result = analyze_pair(ann_path, csv_path)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            print(f"[ERROR] {key}: {exc}")
            continue

        ann_ids = result["ann_ids"]
        csv_ids = result["csv_ids"]
        ann_first, ann_last = _first_last(ann_ids)
        csv_first, csv_last = _first_last(csv_ids)
        status = "OK" if result["ok"] else "MISMATCH"

        print(f"[{status}] {key}")
        print(
            f"         ann: count={len(ann_ids)} first={ann_first} "
            f"last={ann_last} contiguous={result['ann_contiguous']}"
        )
        print(
            f"         csv: count={len(csv_ids)} first={csv_first} "
            f"last={csv_last} contiguous={result['csv_contiguous']}"
        )
        print(
            f"         exact frame-ID sequence match: "
            f"{result['exact_sequence_match']}"
        )

        if not result["ok"]:
            print(
                f"         missing from CSV: "
                f"count={len(result['missing_csv_ids'])} "
                f"classification={result['missing_classification']}"
            )
            print(
                f"         missing IDs: "
                f"{format_id_list(result['missing_csv_ids'])}"
            )
            print(
                f"         extra CSV IDs: "
                f"{format_id_list(result['extra_csv_ids'])}"
            )
            print(
                f"         duplicate ann IDs: "
                f"{format_id_list(result['duplicate_ann_ids'])}"
            )
            print(
                f"         duplicate CSV IDs: "
                f"{format_id_list(result['duplicate_csv_ids'])}"
            )
            mismatches.append((key, result))
        else:
            matches.append(key)

    print()
    print("=== summary ===")
    print(f"compared:   {len(matches) + len(mismatches)}")
    print(f"equal:      {len(matches)}")
    print(f"mismatch:   {len(mismatches)}")
    print(f"errors:     {len(errors)}")
    print(f"unmatched:  {len(ann_only) + len(csv_only)}")

    if mismatches:
        print("\nMismatched pairs:")
        for key, result in mismatches:
            ann_count = len(result["ann_ids"])
            csv_count = len(result["csv_ids"])
            print(f"  {key}:")
            print(f"    ann_count={ann_count}")
            print(f"    csv_count={csv_count}")
            print(f"    row_delta={csv_count - ann_count}")
            print(
                f"    exact_sequence_match={result['exact_sequence_match']}"
            )
            print(
                f"    missing_csv_count={len(result['missing_csv_ids'])}"
            )
            print(
                f"    missing_classification={result['missing_classification']}"
            )
            print(
                f"    missing_csv_ids={format_id_list(result['missing_csv_ids'])}"
            )
            print(
                f"    extra_csv_ids={format_id_list(result['extra_csv_ids'])}"
            )
            print(
                f"    duplicate_csv_ids="
                f"{format_id_list(result['duplicate_csv_ids'])}"
            )

    if errors:
        print("\nErrors:")
        for msg in errors:
            print(f"  - {msg}")

    if mismatches or errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
