from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
import io

from .utils import DbConfig


def write_parquet(df: pd.DataFrame, out_dir: str | Path, dataset: Optional[str] = None) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if dataset:
        partition_dir = out_path / f"dataset={dataset}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / "part-0000.parquet"
    else:
        file_path = out_path / "data.parquet"
    # Use fastparquet engine to avoid heavy pyarrow build on newer Python
    df.to_parquet(file_path, index=False, engine="fastparquet")


def _next_part_index(partition_dir: Path) -> int:
    existing = sorted(partition_dir.glob("part-*.parquet"))
    if not existing:
        return 0
    try:
        last = existing[-1].stem  # e.g., part-0005
        idx = int(last.split("-")[-1])
        return idx + 1
    except Exception:
        # Fallback to count
        return len(existing)


def write_parquet_parts(
    df: pd.DataFrame,
    out_dir: str | Path,
    part_counters: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    counters: Dict[str, int] = part_counters.copy() if part_counters else {}

    for dataset, dfp in df.groupby("dataset"):
        partition_dir = out_path / f"dataset={dataset}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        # Initialize counter lazily from disk if not present
        if dataset not in counters:
            counters[dataset] = _next_part_index(partition_dir)
        file_path = partition_dir / f"part-{counters[dataset]:04d}.parquet"
        dfp.to_parquet(file_path, index=False, engine="fastparquet")
        counters[dataset] += 1

    return counters


def _ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))


def _table_exists(engine: Engine, schema: str, table: str) -> bool:
    query = text(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :schema AND table_name = :table
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        row = conn.execute(query, {"schema": schema, "table": table}).fetchone()
    return row is not None


def _table_columns(engine: Engine, schema: str, table: str) -> set[str]:
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(query, {"schema": schema, "table": table}).fetchall()
    return {r[0] for r in rows}


def _add_missing_columns(engine: Engine, schema: str, table: str, missing: Iterable[str]) -> None:
    if not missing:
        return
    with engine.begin() as conn:
        for col in missing:
            # Use TEXT to be permissive for heterogeneous sources
            conn.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS \"{col}\" TEXT"))


def _create_table_with_text_columns(engine: Engine, schema: str, table: str, columns: Iterable[str]) -> None:
    cols_sql = ", ".join([f'"{c}" TEXT' for c in columns])
    ddl = text(f"CREATE TABLE IF NOT EXISTS {schema}.\"{table}\" ({cols_sql})")
    with engine.begin() as conn:
        conn.execute(ddl)


def _copy_postgres(df: pd.DataFrame, engine: Engine, schema: str, table: str) -> None:
    # Ensure table exists and has all needed columns (as TEXT)
    if not _table_exists(engine, schema, table):
        _create_table_with_text_columns(engine, schema, table, df.columns)
    else:
        existing = _table_columns(engine, schema, table)
        missing = [c for c in df.columns if c not in existing]
        _add_missing_columns(engine, schema, table, missing)

    # Write CSV to memory; use empty strings for nulls and declare NULL '' in COPY
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    cols_sql = ", ".join([f'"{c}"' for c in df.columns])
    copy_sql = f"COPY {schema}.\"{table}\" ({cols_sql}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(copy_sql, csv_buffer)
        raw.commit()
    finally:
        raw.close()


def write_postgres(
    df: pd.DataFrame,
    table: str,
    db: DbConfig,
    if_exists: str = "append",
    chunksize: int = 50_000,
) -> None:
    engine = create_engine(db.sqlalchemy_url)
    _ensure_schema(engine, db.schema)

    # Heuristic: for wide tables or large batches, prefer COPY
    num_cols = max(1, len(df.columns))
    use_copy = (num_cols >= 100) or (len(df) >= 1000)

    if use_copy:
        _copy_postgres(df, engine, db.schema, table)
        return

    # Otherwise, use to_sql with conservative multi batching
    existing_cols = _table_columns(engine, db.schema, table)
    if existing_cols:
        to_add = [c for c in df.columns if c not in existing_cols]
        _add_missing_columns(engine, db.schema, table, to_add)

    max_params = 60000
    max_rows_by_params = max(1, (max_params // num_cols) - 1)
    effective_chunksize = max(1, min(chunksize, max_rows_by_params))

    with engine.begin() as conn:
        df.to_sql(
            table,
            con=conn,
            schema=db.schema,
            if_exists=if_exists,
            index=False,
            chunksize=effective_chunksize,
            method="multi",
        )


