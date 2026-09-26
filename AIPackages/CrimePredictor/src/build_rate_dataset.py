"""Build a categorical crime-rate dataset from incident-level records.

Aggregates incidents by (city, H3, month, weekday, 2-hour bin), builds
deterministic city/H3 vocabularies (index 0 = <UNK>), and writes Parquet plus
metadata JSON. Uses DuckDB so the CSV never needs to fit in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from time_bins import (
    HOUR_BIN_STARTS,
    NUM_TIME_BINS,
    TEMPORAL_COMBINATIONS_PER_CELL,
    TIME_BIN_HOURS,
)

UNK_TOKEN = "<UNK>"

DEFAULT_INPUT = Path(
    "/data/quantization/zaima/crimecode/Data/shortened_dataset.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/quantization/zaima/crimecode/Data/crime_rate_dataset.parquet"
)
DEFAULT_METADATA_DIR = Path(
    "/data/quantization/zaima/crimecode/Data/crime_rate_metadata"
)

REQUIRED_COLUMNS = (
    "city_name",
    "h3_cell",
    "city_index",
    "h3_cell_index",
    "month",
    "day_of_week",
    "hour_bin_start",
    "exposure_hours",
    "person_count",
    "property_count",
    "society_count",
    "other_count",
    "total_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build categorical crime-rate parquet from incident CSV.",
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--min-active-days", type=int, default=300)
    parser.add_argument("--min-cell-incidents", type=int, default=5)
    parser.add_argument("--max-output-rows", type=int, default=50_000_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print active years, eligible cells, and projected rows, then exit.",
    )
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def connect(threads: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    if threads > 0:
        con.execute(f"SET threads TO {int(threads)}")
    con.execute("SET temp_directory='./.duckdb_tmp_crime_rate'")
    return con


def register_incidents(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    csv_literal = str(csv_path).replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE incidents AS
        SELECT
            lower(trim(CAST(city_name AS VARCHAR))) AS city_name,
            trim(CAST(h3_cell AS VARCHAR)) AS h3_cell,
            CAST(year AS INTEGER) AS year,
            CAST(month AS INTEGER) AS month,
            CAST(day_of_week AS INTEGER) AS day_of_week,
            CAST(hour AS INTEGER) AS hour,
            CAST(date_single AS DATE) AS incident_date,
            CASE
                WHEN lower(trim(CAST(crime_class AS VARCHAR))) IN (
                    'person', 'property', 'society', 'other'
                )
                THEN lower(trim(CAST(crime_class AS VARCHAR)))
                ELSE 'other'
            END AS crime_class,
            (CAST(hour AS INTEGER) // {TIME_BIN_HOURS}) * {TIME_BIN_HOURS}
                AS hour_bin_start
        FROM read_csv(
            '{csv_literal}',
            AUTO_DETECT=TRUE,
            SAMPLE_SIZE=-1,
            IGNORE_ERRORS=TRUE,
            parallel=TRUE
        )
        WHERE city_name IS NOT NULL
          AND trim(CAST(city_name AS VARCHAR)) <> ''
          AND h3_cell IS NOT NULL
          AND trim(CAST(h3_cell AS VARCHAR)) <> ''
          AND lower(trim(CAST(h3_cell AS VARCHAR))) <> 'unknown'
          AND year IS NOT NULL
          AND month IS NOT NULL
          AND day_of_week IS NOT NULL
          AND hour IS NOT NULL
          AND date_single IS NOT NULL
          AND month BETWEEN 1 AND 12
          AND day_of_week BETWEEN 0 AND 6
          AND hour BETWEEN 0 AND 23
        """
    )


def build_active_city_years(
    con: duckdb.DuckDBPyConnection,
    min_active_days: int,
) -> int:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE city_year_activity AS
        SELECT
            city_name,
            year,
            COUNT(DISTINCT incident_date) AS n_distinct_dates
        FROM incidents
        GROUP BY city_name, year
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE active_city_years AS
        SELECT city_name, year, n_distinct_dates
        FROM city_year_activity
        WHERE n_distinct_dates >= {int(min_active_days)}
        """
    )
    rows = con.execute(
        """
        SELECT city_name, year, n_distinct_dates,
               n_distinct_dates >= ? AS accepted
        FROM city_year_activity
        ORDER BY city_name, year
        """,
        [min_active_days],
    ).fetchall()

    n_keep = 0
    n_drop = 0
    print("\nCity-year activity filter")
    print(f"{'city':<20} {'year':>6} {'distinct_dates':>16} {'status':>10}")
    print("-" * 56)
    for city_name, year, n_dates, accepted in rows:
        status = "INCLUDE" if accepted else "EXCLUDE"
        n_keep += int(accepted)
        n_drop += int(not accepted)
        print(f"{city_name:<20} {year:>6} {n_dates:>16,} {status:>10}")
    print("-" * 56)
    print(
        f"Included city-years: {n_keep:,} | Excluded: {n_drop:,} "
        f"(threshold={min_active_days})"
    )
    return n_keep


def build_eligible_cells(
    con: duckdb.DuckDBPyConnection,
    min_cell_incidents: int,
) -> int:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE incidents_active AS
        SELECT i.*
        FROM incidents AS i
        INNER JOIN active_city_years AS a
            USING (city_name, year)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE eligible_cells AS
        SELECT
            city_name,
            h3_cell,
            COUNT(*)::BIGINT AS n_incidents
        FROM incidents_active
        GROUP BY city_name, h3_cell
        HAVING COUNT(*) >= {int(min_cell_incidents)}
        """
    )
    n_cells = int(con.execute("SELECT COUNT(*) FROM eligible_cells").fetchone()[0])
    n_cities = int(
        con.execute(
            "SELECT COUNT(DISTINCT city_name) FROM eligible_cells"
        ).fetchone()[0]
    )
    print(
        f"\nEligible H3 cells: {n_cells:,} across {n_cities:,} cities "
        f"(min incidents={min_cell_incidents})"
    )
    return n_cells


def assert_projected_rows_ok(n_cells: int, max_output_rows: int) -> int:
    projected = n_cells * TEMPORAL_COMBINATIONS_PER_CELL
    print("\nProjected output size (before writing cross product)")
    print(f"  eligible_cells              = {n_cells:,}")
    print(
        f"  temporal combinations/cell  = {TEMPORAL_COMBINATIONS_PER_CELL} "
        f"(12 x 7 x {24 // TIME_BIN_HOURS})"
    )
    print(f"  projected rows              = {projected:,}")
    print(f"  max-output-rows             = {max_output_rows:,}")
    if projected > max_output_rows:
        raise SystemExit(
            f"Projected row count {projected:,} exceeds --max-output-rows "
            f"{max_output_rows:,}. Increase --min-cell-incidents or "
            f"TIME_BIN_HOURS ({TIME_BIN_HOURS}), or raise --max-output-rows."
        )
    if projected <= 0:
        raise SystemExit("Projected row count is zero; nothing to write.")
    return projected


def build_vocabularies(
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, int], dict[str, int]]:
    """Deterministic sorted vocabularies with index 0 reserved for <UNK>."""
    cities = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT city_name FROM eligible_cells ORDER BY city_name"
        ).fetchall()
    ]
    h3_cells = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT h3_cell FROM eligible_cells ORDER BY h3_cell"
        ).fetchall()
    ]

    city_to_index: dict[str, int] = {UNK_TOKEN: 0}
    for index, city in enumerate(cities, start=1):
        city_to_index[city] = index

    h3_to_index: dict[str, int] = {UNK_TOKEN: 0}
    for index, cell in enumerate(h3_cells, start=1):
        h3_to_index[cell] = index

    print(
        f"\nVocabularies: cities={len(cities):,} (+UNK) | "
        f"h3_cells={len(h3_cells):,} (+UNK)"
    )
    return city_to_index, h3_to_index


def save_vocabularies(
    metadata_dir: Path,
    city_to_index: dict[str, int],
    h3_to_index: dict[str, int],
) -> dict[str, str]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_to_city = {str(index): name for name, index in city_to_index.items()}
    index_to_h3 = {str(index): name for name, index in h3_to_index.items()}

    paths = {
        "city_to_index": metadata_dir / "city_to_index.json",
        "h3_cell_to_index": metadata_dir / "h3_cell_to_index.json",
        "index_to_city": metadata_dir / "index_to_city.json",
        "index_to_h3_cell": metadata_dir / "index_to_h3_cell.json",
    }
    payloads = {
        "city_to_index": city_to_index,
        "h3_cell_to_index": h3_to_index,
        "index_to_city": index_to_city,
        "index_to_h3_cell": index_to_h3,
    }
    checksums: dict[str, str] = {}
    for key, path in paths.items():
        text = json.dumps(payloads[key], indent=2, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")
        checksums[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f"  wrote {path} sha256={checksums[key][:12]}...")
    return checksums


def register_vocab_tables(
    con: duckdb.DuckDBPyConnection,
    city_to_index: dict[str, int],
    h3_to_index: dict[str, int],
) -> None:
    city_rows = [
        (name, index)
        for name, index in city_to_index.items()
        if name != UNK_TOKEN
    ]
    h3_rows = [
        (name, index)
        for name, index in h3_to_index.items()
        if name != UNK_TOKEN
    ]
    con.execute("CREATE OR REPLACE TEMP TABLE city_vocab (city_name VARCHAR, city_index BIGINT)")
    con.execute("CREATE OR REPLACE TEMP TABLE h3_vocab (h3_cell VARCHAR, h3_cell_index BIGINT)")
    con.executemany("INSERT INTO city_vocab VALUES (?, ?)", city_rows)
    con.executemany("INSERT INTO h3_vocab VALUES (?, ?)", h3_rows)


def build_counts_and_exposure(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE observed_counts AS
        SELECT
            i.city_name,
            i.h3_cell,
            i.month,
            i.day_of_week,
            i.hour_bin_start,
            SUM(CASE WHEN i.crime_class = 'person' THEN 1 ELSE 0 END)::INTEGER
                AS person_count,
            SUM(CASE WHEN i.crime_class = 'property' THEN 1 ELSE 0 END)::INTEGER
                AS property_count,
            SUM(CASE WHEN i.crime_class = 'society' THEN 1 ELSE 0 END)::INTEGER
                AS society_count,
            SUM(CASE WHEN i.crime_class = 'other' THEN 1 ELSE 0 END)::INTEGER
                AS other_count,
            COUNT(*)::INTEGER AS total_count
        FROM incidents_active AS i
        INNER JOIN eligible_cells AS e
            USING (city_name, h3_cell)
        GROUP BY
            i.city_name, i.h3_cell, i.month, i.day_of_week, i.hour_bin_start
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE calendar_days AS
        SELECT
            a.city_name,
            CAST(gs AS DATE) AS cal_date,
            month(CAST(gs AS DATE))::INTEGER AS month,
            (isodow(CAST(gs AS DATE)) - 1)::INTEGER AS day_of_week
        FROM active_city_years AS a
        CROSS JOIN LATERAL (
            SELECT UNNEST(
                generate_series(
                    make_date(a.year, 1, 1),
                    make_date(a.year, 12, 31),
                    INTERVAL 1 DAY
                )
            ) AS gs
        ) AS series
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE hour_bins AS
        SELECT * FROM (VALUES
            {", ".join(f"({value})" for value in HOUR_BIN_STARTS)}
        ) AS t(hour_bin_start)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE exposure AS
        SELECT
            d.city_name,
            d.month,
            d.day_of_week,
            b.hour_bin_start,
            (COUNT(*) * {TIME_BIN_HOURS})::FLOAT AS exposure_hours
        FROM (
            SELECT city_name, month, day_of_week FROM calendar_days
        ) AS d
        CROSS JOIN hour_bins AS b
        GROUP BY d.city_name, d.month, d.day_of_week, b.hour_bin_start
        """
    )


def build_rate_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE rate_dataset AS
        WITH months AS (
            SELECT UNNEST(range(1, 13)) AS month
        ),
        weekdays AS (
            SELECT UNNEST(range(0, 7)) AS day_of_week
        ),
        grid AS (
            SELECT
                e.city_name,
                e.h3_cell,
                cv.city_index,
                hv.h3_cell_index,
                m.month,
                w.day_of_week,
                b.hour_bin_start
            FROM eligible_cells AS e
            INNER JOIN city_vocab AS cv USING (city_name)
            INNER JOIN h3_vocab AS hv USING (h3_cell)
            CROSS JOIN months AS m
            CROSS JOIN weekdays AS w
            CROSS JOIN hour_bins AS b
        )
        SELECT
            g.city_name::VARCHAR AS city_name,
            g.h3_cell::VARCHAR AS h3_cell,
            CAST(g.city_index AS BIGINT) AS city_index,
            CAST(g.h3_cell_index AS BIGINT) AS h3_cell_index,
            CAST(g.month AS TINYINT) AS month,
            CAST(g.day_of_week AS TINYINT) AS day_of_week,
            CAST(g.hour_bin_start AS TINYINT) AS hour_bin_start,
            CAST(x.exposure_hours AS FLOAT) AS exposure_hours,
            CAST(COALESCE(c.person_count, 0) AS INTEGER) AS person_count,
            CAST(COALESCE(c.property_count, 0) AS INTEGER) AS property_count,
            CAST(COALESCE(c.society_count, 0) AS INTEGER) AS society_count,
            CAST(COALESCE(c.other_count, 0) AS INTEGER) AS other_count,
            CAST(COALESCE(c.total_count, 0) AS INTEGER) AS total_count
        FROM grid AS g
        INNER JOIN exposure AS x
            USING (city_name, month, day_of_week, hour_bin_start)
        LEFT JOIN observed_counts AS c
            USING (city_name, h3_cell, month, day_of_week, hour_bin_start)
        """
    )


def write_parquet(con: duckdb.DuckDBPyConnection, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    out_literal = str(output_path).replace("'", "''")
    con.execute(
        f"""
        COPY (
            SELECT
                city_name,
                h3_cell,
                city_index,
                h3_cell_index,
                month,
                day_of_week,
                hour_bin_start,
                exposure_hours,
                person_count,
                property_count,
                society_count,
                other_count,
                total_count
            FROM rate_dataset
            ORDER BY city_index, h3_cell_index, month, day_of_week, hour_bin_start
        )
        TO '{out_literal}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return int(pq.read_metadata(output_path).num_rows)


def validate_output(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    city_to_index: dict[str, int],
    h3_to_index: dict[str, int],
) -> None:
    path_literal = str(parquet_path).replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE rate_check AS
        SELECT * FROM read_parquet('{path_literal}')
        """
    )
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column
        not in {row[0] for row in con.execute("DESCRIBE rate_check").fetchall()}
    ]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    hour_bin_sql = ", ".join(str(value) for value in HOUR_BIN_STARTS)
    checks = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            (
                SELECT COUNT(*) FROM (
                    SELECT 1
                    FROM rate_check
                    GROUP BY city_index, h3_cell_index, month, day_of_week, hour_bin_start
                )
            ) AS n_unique_keys,
            SUM(CASE WHEN exposure_hours IS NULL OR exposure_hours <= 0
                     THEN 1 ELSE 0 END) AS bad_exposure,
            SUM(CASE WHEN person_count < 0 OR property_count < 0
                       OR society_count < 0 OR other_count < 0
                       OR total_count < 0 THEN 1 ELSE 0 END) AS negative_counts,
            SUM(CASE WHEN total_count <> person_count + property_count
                                            + society_count + other_count
                     THEN 1 ELSE 0 END) AS bad_total,
            SUM(CASE WHEN total_count = 0 THEN 1 ELSE 0 END) AS zero_rows,
            SUM(CASE WHEN total_count > 0 THEN 1 ELSE 0 END) AS positive_rows,
            SUM(CASE WHEN city_index <= 0 OR h3_cell_index <= 0
                     THEN 1 ELSE 0 END) AS bad_known_index,
            SUM(CASE WHEN hour_bin_start NOT IN ({hour_bin_sql})
                     THEN 1 ELSE 0 END) AS bad_hour_bin
        FROM rate_check
        """
    ).fetchone()

    (
        n_rows,
        n_unique,
        bad_exposure,
        negative_counts,
        bad_total,
        zero_rows,
        positive_rows,
        bad_known_index,
        bad_hour_bin,
    ) = checks

    print("\nValidation")
    print(f"  rows written              = {n_rows:,}")
    print(f"  unique aggregation keys   = {n_unique:,}")
    print(f"  zero-count rows           = {zero_rows:,}")
    print(f"  positive-count rows       = {positive_rows:,}")
    print(
        f"  zero-count percentage     = "
        f"{(100.0 * zero_rows / n_rows) if n_rows else 0.0:.2f}%"
    )

    errors: list[str] = []
    if n_rows != n_unique:
        errors.append(
            f"duplicate aggregation keys: rows={n_rows:,} unique={n_unique:,}"
        )
    if bad_exposure:
        errors.append(f"non-positive exposure in {bad_exposure:,} rows")
    if negative_counts:
        errors.append(f"negative counts in {negative_counts:,} rows")
    if bad_total:
        errors.append(f"total_count mismatch in {bad_total:,} rows")
    if zero_rows == 0:
        errors.append("no zero-count rows found")
    if bad_known_index:
        errors.append(f"non-positive city/h3 indices in {bad_known_index:,} rows")
    if bad_hour_bin:
        errors.append(f"invalid hour_bin_start in {bad_hour_bin:,} rows")

    # Stable string <-> index mapping.
    mapping_rows = con.execute(
        """
        SELECT city_name, city_index, COUNT(DISTINCT city_index) AS n_idx
        FROM rate_check
        GROUP BY city_name, city_index
        """
    ).fetchall()
    for city_name, city_index, _n in mapping_rows:
        expected = city_to_index.get(city_name)
        if expected != int(city_index):
            errors.append(
                f"city '{city_name}' mapped to {city_index}, expected {expected}"
            )

    h3_map_bad = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT h3_cell, COUNT(DISTINCT h3_cell_index) AS n
            FROM rate_check
            GROUP BY h3_cell
            HAVING COUNT(DISTINCT h3_cell_index) <> 1
        )
        """
    ).fetchone()[0]
    if h3_map_bad:
        errors.append(f"{h3_map_bad} H3 strings map to multiple indices")

    city_stats = con.execute(
        """
        SELECT
            city_name,
            COUNT(*) AS n_rows,
            SUM(CASE WHEN total_count = 0 THEN 1 ELSE 0 END) AS n_zero,
            SUM(CASE WHEN total_count > 0 THEN 1 ELSE 0 END) AS n_pos
        FROM rate_check
        GROUP BY city_name
        ORDER BY city_name
        """
    ).fetchall()
    print("\nPer-city summary")
    print(f"{'city':<20} {'rows':>12} {'zero':>12} {'positive':>12}")
    print("-" * 60)
    for city_name, n_rows_c, n_zero, n_pos in city_stats:
        print(f"{city_name:<20} {n_rows_c:>12,} {n_zero:>12,} {n_pos:>12,}")
        if n_zero == 0 or n_pos == 0:
            errors.append(
                f"city '{city_name}' lacks both positive and zero-count rows"
            )

    if errors:
        raise RuntimeError("Validation failed:\n  - " + "\n  - ".join(errors))
    print("\nAll validation checks passed.")


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if args.min_active_days < 1 or args.min_cell_incidents < 1:
        raise ValueError("thresholds must be >= 1")
    if 24 % TIME_BIN_HOURS != 0:
        raise ValueError(f"TIME_BIN_HOURS={TIME_BIN_HOURS} must divide 24")

    print(f"Input CSV : {args.input_csv}")
    print(f"Output    : {args.output_parquet}")
    print(f"Metadata  : {args.metadata_dir}")
    print(f"TIME_BIN_HOURS={TIME_BIN_HOURS}")

    con = connect(args.threads)
    try:
        print("\nScanning incidents with DuckDB...")
        register_incidents(con, args.input_csv)
        n_incidents = int(con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0])
        print(f"Cleaned incidents retained: {n_incidents:,}")

        n_keep = build_active_city_years(con, args.min_active_days)
        if n_keep == 0:
            raise SystemExit("No city-years passed the active-days filter.")

        n_cells = build_eligible_cells(con, args.min_cell_incidents)
        projected = assert_projected_rows_ok(n_cells, args.max_output_rows)

        city_to_index, h3_to_index = build_vocabularies(con)

        if args.dry_run:
            print("\nDry run complete; skipping aggregation and write.")
            print(
                f"Would write ~{projected:,} rows with "
                f"{len(city_to_index) - 1} cities and {len(h3_to_index) - 1} H3 cells."
            )
            return

        checksums = save_vocabularies(args.metadata_dir, city_to_index, h3_to_index)
        register_vocab_tables(con, city_to_index, h3_to_index)

        print("\nAggregating counts and building exposure calendars...")
        build_counts_and_exposure(con)
        print("Building full spatial-temporal grid (including zero-count rows)...")
        build_rate_table(con)

        actual = int(con.execute("SELECT COUNT(*) FROM rate_dataset").fetchone()[0])
        print(f"Materialized rows: {actual:,} (projected {projected:,})")

        print(f"\nWriting Parquet to {args.output_parquet} ...")
        n_written = write_parquet(con, args.output_parquet)
        print(f"Wrote {n_written:,} rows")
        print(f"Parquet size: {args.output_parquet.stat().st_size:,} bytes")
        print(f"Vocabulary checksums: {checksums}")

        validate_output(con, args.output_parquet, city_to_index, h3_to_index)
    finally:
        con.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
