"""Clean and feature-engineer crime CSVs into one combined shortened_dataset.csv."""

from __future__ import annotations

import codecs
import math
import os
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import h3
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'h3' package is required. Install with: pip install h3"
    ) from exc


dataset_path = "/data/quantization/zaima/crimecode/Data/"
output_path = "/data/quantization/zaima/crimecode/Data/shortened_dataset.csv"

KEEP_COLUMNS = [
    "uid",
    "city_name",
    "offense_code",
    "offense_type",
    "offense_group",
    "offense_against",
    "date_single",
    "date_start",
    "date_end",
    "longitude",
    "latitude",
    "location_category",
    "census_block",
]

DERIVED_COLUMNS = [
    "h3_cell",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "time_uncertainty_hours",
    "crime_class",
    "year",
]

OUTPUT_COLUMNS = KEEP_COLUMNS + DERIVED_COLUMNS

selected_cities = [
    # Eastern
    "boston",
    "charlotte",
    "detroit",
    "louisville",
    "new york",
    "virginia beach",
    # Southern / South-Central
    "austin",
    "fort worth",
    "houston",
    "memphis",
    "nashville",
]

CSV_ENCODINGS = ("utf-8", "cp1252", "latin-1")
CHUNKSIZE = 250_000
H3_RESOLUTION = 9
TEMP_OUTPUT_NAME = "shortened_dataset.tmp.csv"
FINAL_OUTPUT_NAME = "shortened_dataset.csv"

OFFENSE_FILL_COLUMNS = (
    "offense_type",
    "offense_group",
    "offense_against",
    "offense_code",
    "location_category",
)


def latlng_to_h3_cell(latitude: float, longitude: float, resolution: int) -> str:
    """H3 cell id for (lat, lng); supports h3-py v4 and v3."""
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(latitude, longitude, resolution))
    return str(h3.geo_to_h3(latitude, longitude, resolution))


def _file_is_valid_utf8(path: str, block_size: int = 1 << 20) -> bool:
    """Stream-decode the whole file as UTF-8 without loading it into memory."""
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(block_size)
                if not block:
                    decoder.decode(b"", final=True)
                    break
                decoder.decode(block)
    except UnicodeDecodeError:
        return False
    return True


def detect_csv_encoding(path: str) -> str:
    """Pick an encoding that works for the whole file, not just the header.

    Some Crime Open Database files are valid UTF-8 near the start but contain
    Windows-1252 bytes (e.g. 0x92) later. Prefer UTF-8 only when the entire
    file streams cleanly as UTF-8; otherwise fall back to cp1252 / latin-1.
    """
    if _file_is_valid_utf8(path):
        # Confirm pandas can open a few rows with UTF-8.
        try:
            pd.read_csv(path, encoding="utf-8", nrows=5, low_memory=False)
            return "utf-8"
        except UnicodeDecodeError:
            pass

    last_error: Optional[Exception] = None
    for encoding in ("cp1252", "latin-1"):
        try:
            pd.read_csv(path, encoding=encoding, nrows=5, low_memory=False)
            return encoding
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not decode {path} with {CSV_ENCODINGS}: {last_error}"
    )


def iter_csv_chunks(path: str, chunksize: int = CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks using encoding fallback (utf-8 / cp1252 / latin-1)."""
    encoding = detect_csv_encoding(path)
    print(f"  using encoding={encoding}")
    yield from pd.read_csv(
        path,
        encoding=encoding,
        chunksize=chunksize,
        low_memory=False,
    )


def map_crime_class(offense_against: object) -> str:
    if offense_against is None or (isinstance(offense_against, float) and math.isnan(offense_against)):
        return "other"
    text = str(offense_against).strip().lower()
    if not text or text == "nan":
        return "other"
    if "person" in text:
        return "person"
    if "property" in text:
        return "property"
    if "society" in text:
        return "society"
    return "other"


def compute_time_uncertainty_hours(
    date_start: pd.Series,
    date_end: pd.Series,
) -> Tuple[pd.Series, int]:
    """Return uncertainty hours and count of malformed/partial intervals."""
    both_valid = date_start.notna() & date_end.notna()
    both_missing = date_start.isna() & date_end.isna()
    ordered = both_valid & (date_end >= date_start)
    malformed = (~both_missing) & (~ordered)

    uncertainty = pd.Series(np.nan, index=date_start.index, dtype="float64")
    uncertainty.loc[both_missing] = 0.0
    if ordered.any():
        delta_seconds = (
            date_end.loc[ordered] - date_start.loc[ordered]
        ).dt.total_seconds()
        uncertainty.loc[ordered] = delta_seconds / 3600.0

    return uncertainty.astype("float32"), int(malformed.sum())


def preprocess_chunk(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Filter, clean, and feature-engineer one chunk. Returns (df, malformed_count)."""
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Chunk missing required columns: {missing}")

    df = df[KEEP_COLUMNS].copy()
    print(len(df))

    # City filtering (lowercase in final output).
    df["city_name"] = df["city_name"].astype("string").str.strip().str.lower()
    print(df["city_name"].unique())
    df = df[df["city_name"].isin(selected_cities)]
    if df.empty:
        print("No rows after city filtering")
        return df.iloc[0:0].copy(), 0
    print("After city filtering: ", len(df))

    # Local police-report timestamps; no timezone conversion.
    df["date_single"] = pd.to_datetime(df["date_single"], errors="coerce")
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")
    df = df[df["date_single"].notna()]
    if df.empty:
        return df.iloc[0:0].copy(), 0
    
    print("After date filtering: ", len(df))
    # Coordinates.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df[
        df["latitude"].between(-90.0, 90.0)
        & df["longitude"].between(-180.0, 180.0)
    ]
    if df.empty:
        return df.iloc[0:0].copy(), 0

    print("After coordinate filtering: ", len(df))

    # Calendar features from date_single.
    df["year"] = df["date_single"].dt.year.astype("int64")
    df["hour"] = df["date_single"].dt.hour.astype("int64")
    df["day_of_week"] = df["date_single"].dt.dayofweek.astype("int64")  # Mon=0 .. Sun=6
    df["month"] = df["date_single"].dt.month.astype("int64")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")
    print("After calendar features: ", len(df)) 
    
    # Cyclical time features.
    two_pi = 2.0 * np.pi
    df["hour_sin"] = np.sin(two_pi * df["hour"] / 24.0).astype("float32")
    df["hour_cos"] = np.cos(two_pi * df["hour"] / 24.0).astype("float32")
    df["weekday_sin"] = np.sin(two_pi * df["day_of_week"] / 7.0).astype("float32")
    df["weekday_cos"] = np.cos(two_pi * df["day_of_week"] / 7.0).astype("float32")

    uncertainty, malformed_count = compute_time_uncertainty_hours(
        df["date_start"],
        df["date_end"],
    )
    df["time_uncertainty_hours"] = uncertainty

    print("After time uncertainty: ", len(df))

    # H3 cells (lat first, lng second).
    df["h3_cell"] = [
        latlng_to_h3_cell(float(lat), float(lng), H3_RESOLUTION)
        for lat, lng in zip(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    ]
    df["h3_cell"] = df["h3_cell"].astype("string")

    print("After h3 cells: ", len(df))

    # Crime class from offense_against.
    df["crime_class"] = df["offense_against"].map(map_crime_class)

    print("After crime class: ", len(df))

    # Fill non-essential offense / location text fields.
    for col in OFFENSE_FILL_COLUMNS:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("unknown")
            .replace({"": "unknown", "<NA>": "unknown"})
        )
    print("After offense fill: ", len(df))

    # Preserve census_block as string (leading zeros).
    df["census_block"] = df["census_block"].astype("string")
    print("After census block: ", len(df))

    # Drop rows with unusable essential fields only.
    essential = ["uid", "city_name", "date_single", "longitude", "latitude", "h3_cell"]
    df = df.dropna(subset=essential)
    df = df[
        df["uid"].astype("string").str.len().gt(0)
        & df["city_name"].astype("string").str.len().gt(0)
        & df["h3_cell"].astype("string").str.len().gt(0)
    ]
    print("After essential filtering: ", len(df))
    df = df.drop_duplicates()
    df = df.reindex(columns=OUTPUT_COLUMNS)
    df = df.reset_index(drop=True)
    return df, malformed_count


def _is_output_or_temp_csv(filename: str) -> bool:
    name = filename.lower()
    return name in {
        FINAL_OUTPUT_NAME.lower(),
        TEMP_OUTPUT_NAME.lower(),
    } or name.endswith(".tmp.csv")


def shorten_dataset(dataset_path: str, output_path: str) -> None:
    """Process all source CSVs into one combined shortened_dataset.csv."""
    output_file = Path(output_path)
    if output_file.suffix.lower() != ".csv":
        raise ValueError(
            f"output_path must be a .csv file path, got: {output_path}"
        )

    parent = output_file.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_file = parent / TEMP_OUTPUT_NAME

    if temp_file.exists():
        temp_file.unlink()

    seen_uids: Set[str] = set()
    header_written = False
    cumulative_rows = 0
    source_files = sorted(
        f
        for f in os.listdir(dataset_path)
        if f.endswith(".csv") and not _is_output_or_temp_csv(f)
    )

    if not source_files:
        raise FileNotFoundError(f"No source CSV files found in {dataset_path}")

    try:
        for file in source_files:
            path = os.path.join(dataset_path, file)
            input_rows = 0
            retained_rows = 0
            malformed_total = 0

            print(f"\nProcessing {file} ...")
            for chunk in iter_csv_chunks(path, chunksize=CHUNKSIZE):
                input_rows += len(chunk)
                try:
                    processed, malformed = preprocess_chunk(chunk)
                except KeyError as exc:
                    print(f"  Skipping file due to missing columns: {exc}")
                    retained_rows = 0
                    break

                malformed_total += malformed

                if processed.empty:
                    continue

                # Lightweight cross-file uid dedup after city filtering.
                processed["uid"] = processed["uid"].astype("string")
                mask_new = ~processed["uid"].isin(seen_uids)
                processed = processed.loc[mask_new]
                if processed.empty:
                    continue

                seen_uids.update(processed["uid"].astype(str).tolist())

                processed.to_csv(
                    temp_file,
                    mode="a",
                    index=False,
                    header=not header_written,
                )
                header_written = True
                retained_rows += len(processed)
                cumulative_rows += len(processed)

            print(
                f"  input_rows={input_rows:,} | retained_rows={retained_rows:,} | "
                f"malformed_time_intervals={malformed_total:,} | "
                f"cumulative_output_rows={cumulative_rows:,}"
            )
            if malformed_total:
                print(
                    f"  note: {malformed_total:,} rows had partial or "
                    "end-before-start date intervals (time_uncertainty_hours=NaN)"
                )

        if cumulative_rows == 0 or not temp_file.exists():
            raise RuntimeError(
                "No rows were retained; temporary output was not created."
            )

        # Atomically replace the final output only after full success.
        os.replace(temp_file, output_file)
    except Exception:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise

    preview = pd.read_csv(output_file, nrows=5)
    print("\n=== Done ===")
    print(f"Final output path: {output_file}")
    print(f"Total rows: {cumulative_rows:,}")
    print(f"Final columns ({len(preview.columns)}): {list(preview.columns)}")
    print("Five-row preview:")
    print(preview)


if __name__ == "__main__":
    shorten_dataset(dataset_path, output_path)
