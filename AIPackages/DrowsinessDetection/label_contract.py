#!/usr/bin/env python3
"""Canonical drowsiness label contract (3-class).

Raw DMD OpenLABEL ``eyes_state/*`` values observed in the dataset::

    close, closing, open, opening, undefined

``undefined`` is an **explicit** action type present in annotation JSONs
(not a placeholder for missing labels). Frames without any ``eyes_state/*``
interval remain unlabeled and are excluded from the dataset.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

CLASS_TO_IDX: Dict[str, int] = {
    "closed": 0,
    "open": 1,
    "undefined": 2,
}

IDX_TO_CLASS: Dict[int, str] = {
    0: "closed",
    1: "open",
    2: "undefined",
}

NUM_CLASSES: int = len(CLASS_TO_IDX)

assert NUM_CLASSES == 3, NUM_CLASSES
assert set(CLASS_TO_IDX) == {"closed", "open", "undefined"}
assert CLASS_TO_IDX["closed"] == 0
assert CLASS_TO_IDX["open"] == 1
assert CLASS_TO_IDX["undefined"] == 2
assert IDX_TO_CLASS == {idx: name for name, idx in CLASS_TO_IDX.items()}

# Only raw suffixes that actually appear in the DMD JSON files.
RAW_EYES_STATE_TO_CANONICAL: Dict[str, Optional[str]] = {
    "close": "closed",
    "open": "open",
    "undefined": "undefined",
    # Transition into closed eyes: kept and trained as closed.
    "closing": "closed",
    # Transition into open eyes: still excluded from training targets.
    "opening": None,
}

EXCLUDED_RAW_EYES_STATES: FrozenSet[str] = frozenset({"opening"})
KEEP_CANONICAL_CLASSES: FrozenSet[str] = frozenset(CLASS_TO_IDX.keys())


def map_raw_eyes_state(raw: str) -> Optional[str]:
    """Map a raw ``eyes_state`` suffix to a canonical class, or ``None`` if excluded.

    Raises
    ------
    KeyError
        If ``raw`` is not in the explicit observed-type mapping.
    """
    if raw not in RAW_EYES_STATE_TO_CANONICAL:
        raise KeyError(
            f"unknown eyes_state raw label {raw!r}; "
            f"known={sorted(RAW_EYES_STATE_TO_CANONICAL)}"
        )
    return RAW_EYES_STATE_TO_CANONICAL[raw]


def canonical_to_index(canonical: str) -> int:
    if canonical not in CLASS_TO_IDX:
        raise KeyError(f"unknown canonical class {canonical!r}")
    return int(CLASS_TO_IDX[canonical])
