#!/usr/bin/env python3
"""PyTorch frame-level DMD drowsiness dataset.

Matches OpenLABEL drowsiness annotations to Apple Vision landmark CSVs using
the same session-key strategy as ``check_anns.py``, then joins rows to labels
by explicit ``frame_index`` (never by row position alone).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

PathLike = Union[Path, str]

ANNS_DIR = Path("/data/quantization/zaima/DMD/drowsiness/anns")
# Prefer package-local schema-v3 CSVs (legacy Apple Vision dumps use a different suffix).
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_V3_LANDMARKS = _PACKAGE_ROOT / "data" / "landmarks_v3"
_DEFAULT_V2_LANDMARKS = _PACKAGE_ROOT / "data" / "landmarks_v2"
_LEGACY_LANDMARKS = Path("/data/quantization/zaima/DMD/drowsiness/landmarks")
LANDMARKS_DIR = (
    _DEFAULT_V3_LANDMARKS
    if _DEFAULT_V3_LANDMARKS.is_dir()
    else (
        _DEFAULT_V2_LANDMARKS
        if _DEFAULT_V2_LANDMARKS.is_dir()
        else _LEGACY_LANDMARKS
    )
)

ANN_SUFFIX = "_rgb_ann_drowsiness.json"

# Frame-level multiclass target comes from eyes_state/* annotations.
EYES_STATE_PREFIX = "eyes_state/"

# Allow importing when this file is loaded as ``pre_process.dataloader``.
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from feature_contract import (  # noqa: E402
    CSV_SUFFIX,
    DROWSINESS_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    _BINARY_FEATURE_NAMES,
)
from label_contract import (  # noqa: E402
    CLASS_TO_IDX,
    EXCLUDED_RAW_EYES_STATES,
    IDX_TO_CLASS,
    map_raw_eyes_state,
)

_BINARY_TRUE = frozenset({"1", "true", "yes"})
_BINARY_FALSE = frozenset({"0", "false", "no"})


def session_key_from_ann(path: Path) -> Optional[str]:
    """Return the shared session key for an annotation JSON, or None."""
    name = path.name
    if not name.endswith(ANN_SUFFIX):
        return None
    return name[: -len(ANN_SUFFIX)]


def session_key_from_csv(path: Path) -> Optional[str]:
    """Return the shared session key for a landmark CSV, or None."""
    name = path.name
    if not name.endswith(CSV_SUFFIX):
        return None
    return name[: -len(CSV_SUFFIX)]


def index_anns(anns_dir: Path) -> Dict[str, Path]:
    """Index annotation JSONs by session key (recursive)."""
    index: Dict[str, Path] = {}
    for path in sorted(anns_dir.rglob("*")):
        if not path.is_file():
            continue
        key = session_key_from_ann(path)
        if key is None:
            continue
        if key in index:
            raise ValueError(
                f"duplicate annotation session key {key!r}: "
                f"{index[key]} and {path}"
            )
        index[key] = path
    return index


def index_csvs(landmarks_dir: Path) -> Dict[str, Path]:
    """Index landmark CSVs by session key (recursive)."""
    index: Dict[str, Path] = {}
    for path in sorted(landmarks_dir.rglob("*")):
        if not path.is_file():
            continue
        key = session_key_from_csv(path)
        if key is None:
            continue
        if key in index:
            raise ValueError(
                f"duplicate landmark session key {key!r}: "
                f"{index[key]} and {path}"
            )
        index[key] = path
    return index


def _iter_actions(actions: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(actions, dict):
        for action in actions.values():
            if not isinstance(action, dict):
                raise ValueError(
                    f"action entry must be an object, got {type(action).__name__}"
                )
            yield action
        return
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(
                    f"action entry must be an object, got {type(action).__name__}"
                )
            yield action
        return
    raise ValueError(
        f"openlabel.actions must be a JSON object or array, "
        f"got {type(actions).__name__}"
    )


def build_drowsiness_label_map(
    ann_path: Path,
    *,
    session_key: str,
) -> Dict[int, str]:
    """Map inclusive frame indices to ``eyes_state/*`` class names.

    Raises if the same frame is assigned two different eyes-state labels.
    Identical duplicate assignments are allowed. Other action families
    (``blinks/*``, ``yawning/*``) are ignored for this multiclass target.
    """
    with ann_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        actions = payload["openlabel"]["actions"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{session_key}: missing openlabel.actions ({exc})"
        ) from exc

    label_map: Dict[int, str] = {}
    for action in _iter_actions(actions):
        action_type = action.get("type")
        if not isinstance(action_type, str) or not action_type.startswith(
            EYES_STATE_PREFIX
        ):
            continue
        class_name = action_type[len(EYES_STATE_PREFIX) :]
        if not class_name:
            raise ValueError(
                f"{session_key}: empty eyes-state class in type {action_type!r}"
            )

        intervals = action.get("frame_intervals")
        if intervals is None:
            continue
        if not isinstance(intervals, list):
            raise ValueError(
                f"{session_key}: frame_intervals for {action_type!r} "
                f"must be a list, got {type(intervals).__name__}"
            )

        for interval in intervals:
            if not isinstance(interval, dict):
                raise ValueError(
                    f"{session_key}: frame interval for {action_type!r} "
                    f"must be an object, got {type(interval).__name__}"
                )
            try:
                start = int(interval["frame_start"])
                end = int(interval["frame_end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{session_key}: invalid frame_start/frame_end in "
                    f"{action_type!r} interval {interval!r}: {exc}"
                ) from exc
            if start > end:
                raise ValueError(
                    f"{session_key}: frame_start ({start}) > frame_end ({end}) "
                    f"for {action_type!r}"
                )

            for frame_idx in range(start, end + 1):
                existing = label_map.get(frame_idx)
                if existing is not None and existing != class_name:
                    raise ValueError(
                        f"{session_key}: conflicting eyes-state labels at "
                        f"frame_index={frame_idx}: {existing!r} vs {class_name!r}"
                    )
                label_map[frame_idx] = class_name

    return label_map


# Backward-compatible alias.
build_gaze_zone_label_map = build_drowsiness_label_map


def _parse_binary_float(raw: str) -> Optional[float]:
    token = raw.strip().lower()
    if token in _BINARY_TRUE:
        return 1.0
    if token in _BINARY_FALSE:
        return 0.0
    return None


def parse_feature_value(
    *,
    column: str,
    raw: Optional[str],
    csv_name: str,
    row_number: int,
    frame_index: Optional[int],
) -> float:
    """Parse one feature cell to a finite float (empty/NaN -> 0.0)."""
    frame_msg = (
        f" frame_index={frame_index}" if frame_index is not None else ""
    )
    if raw is None or str(raw).strip() == "":
        return 0.0

    text = str(raw).strip()
    if column in _BINARY_FEATURE_NAMES:
        parsed = _parse_binary_float(text)
        if parsed is not None:
            return parsed
        # Fall through to numeric parse for values like "1.0".

    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{csv_name}: malformed numeric value at data-row {row_number}"
            f"{frame_msg} column={column!r} value={raw!r}"
        ) from exc

    if math.isnan(value):
        return 0.0
    if not math.isfinite(value):
        raise ValueError(
            f"{csv_name}: non-finite numeric value at data-row {row_number}"
            f"{frame_msg} column={column!r} value={raw!r}"
        )
    if column in _BINARY_FEATURE_NAMES and value not in (0.0, 1.0):
        raise ValueError(
            f"{csv_name}: binary column {column!r} must be 0/1 "
            f"(or true/false/yes/no) at data-row {row_number}"
            f"{frame_msg} value={raw!r}"
        )
    return float(value)


def read_csv_feature_rows(
    csv_path: Path,
) -> Tuple[List[int], List[List[float]], Dict[str, int]]:
    """Read landmark CSV features keyed by explicit ``frame_index``.

    Returns ``(frame_indices, feature_rows, diagnostics)`` where
    ``feature_rows[i]`` corresponds to ``frame_indices[i]``.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path.name}: CSV has no header")
        fieldnames = list(reader.fieldnames)
        if "frame_index" not in fieldnames:
            raise ValueError(
                f"{csv_path.name}: missing required column 'frame_index'"
            )
        missing_cols = [
            c for c in DROWSINESS_FEATURE_NAMES if c not in fieldnames
        ]
        if missing_cols:
            raise ValueError(
                f"{csv_path.name}: missing required feature columns: "
                f"{missing_cols}"
            )

        frame_indices: List[int] = []
        feature_rows: List[List[float]] = []
        seen: Dict[int, int] = {}

        for row_number, row in enumerate(reader, start=1):
            raw_idx = row.get("frame_index")
            if raw_idx is None or str(raw_idx).strip() == "":
                raise ValueError(
                    f"{csv_path.name}: empty or missing frame_index "
                    f"at data-row {row_number}"
                )
            try:
                frame_index = int(str(raw_idx).strip())
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path.name}: non-integer frame_index {raw_idx!r} "
                    f"at data-row {row_number}"
                ) from exc

            if frame_index in seen:
                raise ValueError(
                    f"{csv_path.name}: duplicate frame_index={frame_index} "
                    f"at data-rows {seen[frame_index]} and {row_number}"
                )
            seen[frame_index] = row_number

            values: List[float] = []
            for column in DROWSINESS_FEATURE_NAMES:
                values.append(
                    parse_feature_value(
                        column=column,
                        raw=row.get(column),
                        csv_name=csv_path.name,
                        row_number=row_number,
                        frame_index=frame_index,
                    )
                )
            frame_indices.append(frame_index)
            feature_rows.append(values)

    diagnostics = {
        "csv_rows": len(frame_indices),
    }
    return frame_indices, feature_rows, diagnostics


def discover_drowsiness_classes(
    ann_paths: Sequence[Path],
) -> List[str]:
    """Return sorted unique **raw** eyes-state suffixes across annotation files."""
    classes = set()
    for ann_path in ann_paths:
        label_map = build_drowsiness_label_map(
            ann_path, session_key=ann_path.stem
        )
        classes.update(label_map.values())
    return sorted(classes)


# Backward-compatible alias.
discover_gaze_zone_classes = discover_drowsiness_classes


class DMDGazeFrameDataset(Dataset):
    """In-memory frame-level DMD drowsiness dataset (schema v3 features).

    Stores every frame that has an explicit ``eyes_state/*`` annotation.
    ``closing`` is mapped to canonical ``closed`` and kept as a training
    target. ``opening`` remains excluded as a window endpoint but can still
    appear as temporal context inside kept windows.
    """

    def __init__(
        self,
        anns_dir: PathLike = ANNS_DIR,
        landmarks_dir: PathLike = LANDMARKS_DIR,
        class_to_idx: Optional[Dict[str, int]] = None,
        verbose: bool = True,
    ) -> None:
        self.anns_dir = Path(anns_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.feature_names = list(DROWSINESS_FEATURE_NAMES)
        self.verbose = bool(verbose)

        if not self.anns_dir.is_dir():
            raise FileNotFoundError(f"anns_dir does not exist: {self.anns_dir}")
        if not self.landmarks_dir.is_dir():
            raise FileNotFoundError(
                f"landmarks_dir does not exist: {self.landmarks_dir}"
            )

        anns = index_anns(self.anns_dir)
        csvs = index_csvs(self.landmarks_dir)
        matched_keys = sorted(set(anns) & set(csvs))
        ann_only = sorted(set(anns) - set(csvs))
        csv_only = sorted(set(csvs) - set(anns))

        if not matched_keys:
            legacy_hint = ""
            legacy_glob = list(self.landmarks_dir.glob("*_rgb_face.apple_drowsiness.csv"))
            if legacy_glob and not list(self.landmarks_dir.glob(f"*{CSV_SUFFIX}")):
                legacy_hint = (
                    f" Found {len(legacy_glob)} legacy "
                    "'*_rgb_face.apple_drowsiness.csv' files but schema v3 requires "
                    f"'{CSV_SUFFIX}'. Convert with: "
                    "python scripts/build_v3_csv_from_legacy.py "
                    f"--landmarks-dir {self.landmarks_dir} "
                    "or pass --landmarks-dir data/landmarks_v3"
                )
            raise FileNotFoundError(
                "No matched ann/CSV sessions. "
                f"anns={len(anns)} csvs={len(csvs)} "
                f"(looking for landmark suffix {CSV_SUFFIX!r} under "
                f"{self.landmarks_dir}).{legacy_hint}"
            )

        self.matched_session_keys = matched_keys
        self.ann_only_session_keys = ann_only
        self.csv_only_session_keys = csv_only
        self.session_keys = list(matched_keys)

        # Canonical 3-class contract (never discover opening as a class;
        # closing maps to closed).
        if class_to_idx is None:
            self.class_to_idx = dict(CLASS_TO_IDX)
        else:
            if dict(class_to_idx) != dict(CLASS_TO_IDX):
                raise ValueError(
                    "class_to_idx must match the canonical 3-class contract "
                    f"{CLASS_TO_IDX}, got {class_to_idx}"
                )
            self.class_to_idx = dict(CLASS_TO_IDX)
        self.idx_to_class = dict(IDX_TO_CLASS)

        self.samples: List[Dict[str, Any]] = []
        self.session_diagnostics: Dict[str, Dict[str, int]] = {}
        raw_counter: Counter = Counter()
        canonical_counter: Counter = Counter()
        excluded_counter: Counter = Counter()

        for key in matched_keys:
            samples, diagnostics = self._load_session(
                session_key=key,
                ann_path=anns[key],
                csv_path=csvs[key],
            )
            self.session_diagnostics[key] = diagnostics
            for sample in samples:
                raw_counter[sample["raw_label_name"]] += 1
                if sample["canonical_label_name"] is None:
                    excluded_counter[sample["raw_label_name"]] += 1
                else:
                    canonical_counter[sample["canonical_label_name"]] += 1
            self.samples.extend(samples)

        self.raw_label_counts = {name: int(raw_counter[name]) for name in sorted(raw_counter)}
        self.excluded_raw_counts = {
            name: int(excluded_counter[name]) for name in sorted(excluded_counter)
        }
        self.class_counts = {
            name: int(canonical_counter.get(name, 0))
            for name in self.class_to_idx.keys()
        }

        if self.verbose:
            self._print_load_summary()

    def _load_session(
        self,
        *,
        session_key: str,
        ann_path: Path,
        csv_path: Path,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        label_map = build_drowsiness_label_map(
            ann_path, session_key=session_key
        )
        frame_indices, feature_rows, csv_diag = read_csv_feature_rows(csv_path)

        csv_index_set = set(frame_indices)
        annotated_frames = set(label_map.keys())
        ann_missing_from_csv = annotated_frames - csv_index_set
        csv_without_label = csv_index_set - annotated_frames

        samples: List[Dict[str, Any]] = []
        for frame_index, features in zip(frame_indices, feature_rows):
            raw_label = label_map.get(frame_index)
            if raw_label is None:
                # Do NOT invent ``undefined`` for missing eyes_state intervals.
                continue
            try:
                canonical = map_raw_eyes_state(raw_label)
            except KeyError as exc:
                raise ValueError(
                    f"{session_key}: {exc} at frame_index={frame_index}"
                ) from exc

            feature_tensor = torch.tensor(features, dtype=torch.float32)
            if not bool(torch.isfinite(feature_tensor).all()):
                raise ValueError(
                    f"{session_key}: non-finite features at "
                    f"frame_index={frame_index} values={features}"
                )
            if canonical is None:
                label_index = -1
                label_name = raw_label  # excluded transition
            else:
                label_index = int(self.class_to_idx[canonical])
                label_name = canonical

            samples.append(
                {
                    "session_key": session_key,
                    "frame_index": int(frame_index),
                    "features": feature_tensor,
                    "raw_label_name": raw_label,
                    "canonical_label_name": canonical,
                    "label_index": int(label_index),
                    "label_name": label_name,
                    "is_excluded_transition": canonical is None,
                }
            )

        diagnostics = {
            "csv_rows": int(csv_diag["csv_rows"]),
            "annotated_eyes_state_frames": len(annotated_frames),
            "matched_labeled_frames": len(samples),
            "annotation_frames_missing_from_csv": len(ann_missing_from_csv),
            "csv_frames_without_eyes_state_labels": len(csv_without_label),
            "excluded_transition_frames": sum(
                1 for s in samples if s["is_excluded_transition"]
            ),
        }
        return samples, diagnostics

    def _print_load_summary(self) -> None:
        print(f"anns_dir:      {self.anns_dir}")
        print(f"landmarks_dir: {self.landmarks_dir}")
        print(f"feature_schema: {FEATURE_SCHEMA_VERSION}")
        print(f"matched sessions:   {len(self.matched_session_keys)}")
        print(f"annotation-only:    {len(self.ann_only_session_keys)}")
        print(f"CSV-only:           {len(self.csv_only_session_keys)}")
        print(f"total eyes_state frames (incl. transitions): {len(self.samples)}")
        print(
            f"DROWSINESS_FEATURE_NAMES ({len(self.feature_names)}): "
            f"{self.feature_names}"
        )
        print(f"raw_label_counts: {self.raw_label_counts}")
        print(f"excluded_raw_counts (opening): {self.excluded_raw_counts}")
        print(f"canonical class_to_idx: {self.class_to_idx}")
        print(f"canonical class_counts (frame-level keep): {self.class_counts}")
        print(
            "undefined source: explicit OpenLABEL action "
            "'eyes_state/undefined' (not missing annotation)"
        )
        if self.ann_only_session_keys:
            print("annotation-only sessions:")
            for key in self.ann_only_session_keys:
                print(f"  - {key}")
        if self.csv_only_session_keys:
            print("CSV-only sessions:")
            for key in self.csv_only_session_keys:
                print(f"  - {key}")
        print("per-session alignment diagnostics:")
        for key in self.matched_session_keys:
            diag = self.session_diagnostics[key]
            print(
                f"  {key}: "
                f"csv_rows={diag['csv_rows']} "
                f"annotated={diag['annotated_eyes_state_frames']} "
                f"matched={diag['matched_labeled_frames']} "
                f"excluded_transitions={diag['excluded_transition_frames']} "
                f"ann_missing_csv={diag['annotation_frames_missing_from_csv']} "
                f"csv_without_label={diag['csv_frames_without_eyes_state_labels']}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        features = sample["features"]
        if not bool(torch.isfinite(features).all()):
            raise ValueError(
                f"non-finite features at dataset index={index} "
                f"session={sample['session_key']} "
                f"frame_index={sample['frame_index']}"
            )
        if sample["label_index"] < 0:
            raise IndexError(
                f"frame index={index} is an excluded transition "
                f"({sample['raw_label_name']!r}); use the window dataset"
            )
        label = torch.tensor(sample["label_index"], dtype=torch.long)
        return features, label

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Return debugging metadata for sample ``index``."""
        sample = self.samples[index]
        return {
            "session_key": sample["session_key"],
            "frame_index": sample["frame_index"],
            "raw_label_name": sample["raw_label_name"],
            "canonical_label_name": sample["canonical_label_name"],
            "label_name": sample["label_name"],
            "label_index": sample["label_index"],
            "is_excluded_transition": sample["is_excluded_transition"],
        }


def create_dataloader(
    dataset: DMDGazeFrameDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Build a DataLoader for ``DMDGazeFrameDataset``.

    ``pin_memory`` defaults to True when CUDA is available to speed host→GPU
    transfers during training/inference.

    Note: future train/val/test splits must be by participant or full session,
    not by randomly splitting adjacent frames.
    """
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=num_workers > 0,
    )


def main() -> int:
    dataset = DMDGazeFrameDataset(verbose=True)
    loader = create_dataloader(
        dataset,
        batch_size=min(8, max(1, len(dataset))),
        shuffle=False,
        num_workers=0,
    )
    features, labels = next(iter(loader))
    print()
    print("=== one batch ===")
    print(f"features shape: {tuple(features.shape)}")
    print(f"labels shape:   {tuple(labels.shape)}")
    print(f"features dtype: {features.dtype}")
    print(f"labels dtype:   {labels.dtype}")
    print(f"all features finite: {bool(torch.isfinite(features).all())}")
    print(f"minimum label index: {int(labels.min().item())}")
    print(f"maximum label index: {int(labels.max().item())}")
    print()
    print("=== first sample metadata ===")
    print(dataset.get_metadata(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
