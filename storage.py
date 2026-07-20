"""DuckDB market-data storage.

Replaces the SQLite market_data.db. Datasets stay immutable: each collection
run inserts a new tagged dataset row (asset, timeframe, collection time,
source) whose candles are never updated afterwards. DuckDB gives us direct
Parquet export and fast analytical queries for the feature pipeline.

The trade journal (journal.db, SQLite) is unchanged - execution records and
market data are deliberately separate stores.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "market.duckdb"
SOURCE = "iqoptionapi-websocket"

SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS dataset_id_seq;
CREATE TABLE IF NOT EXISTS datasets (
    id BIGINT PRIMARY KEY DEFAULT nextval('dataset_id_seq'),
    asset VARCHAR NOT NULL,
    interval_seconds INTEGER NOT NULL,
    collected_at_utc TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    start_ts BIGINT,
    end_ts BIGINT,
    candle_count INTEGER NOT NULL,
    gap_count INTEGER NOT NULL,
    gaps VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS candles (
    dataset_id BIGINT NOT NULL,
    from_ts BIGINT NOT NULL,
    to_ts BIGINT NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE,
    UNIQUE (dataset_id, from_ts)
);
CREATE SEQUENCE IF NOT EXISTS payout_id_seq;
CREATE TABLE IF NOT EXISTS payout_snapshots (
    id BIGINT PRIMARY KEY DEFAULT nextval('payout_id_seq'),
    ts_utc TIMESTAMPTZ NOT NULL,
    ts_epoch BIGINT NOT NULL,
    asset VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    payout DOUBLE NOT NULL,
    source VARCHAR NOT NULL
);
"""


def open_db(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path))
    conn.execute(SCHEMA)
    return conn


def store_dataset(
    conn: duckdb.DuckDBPyConnection,
    asset: str,
    interval: int,
    candles: list[dict],
    gaps: list[dict],
) -> int:
    dataset_id = conn.execute(
        """INSERT INTO datasets (asset, interval_seconds, collected_at_utc, source,
               start_ts, end_ts, candle_count, gap_count, gaps)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
        (
            asset,
            interval,
            datetime.now(timezone.utc),
            SOURCE,
            candles[0]["from_ts"] if candles else None,
            candles[-1]["to_ts"] if candles else None,
            len(candles),
            len(gaps),
            json.dumps(gaps),
        ),
    ).fetchone()[0]
    if candles:
        frame = pd.DataFrame(candles)
        frame.insert(0, "dataset_id", dataset_id)
        conn.execute(
            """INSERT INTO candles
               SELECT dataset_id, from_ts, to_ts, open, high, low, close, volume
               FROM frame"""
        )
    return dataset_id


def store_payout_snapshot(conn: duckdb.DuckDBPyConnection, profits: dict) -> int:
    now = datetime.now(timezone.utc)
    rows = [
        (now, int(now.timestamp()), asset, kind, float(payout), SOURCE)
        for asset, kinds in profits.items()
        for kind, payout in kinds.items()
        if isinstance(payout, (int, float))
    ]
    conn.executemany(
        """INSERT INTO payout_snapshots (ts_utc, ts_epoch, asset, kind, payout, source)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    ) if rows else None
    return len(rows)


def load_candles(conn: duckdb.DuckDBPyConnection, dataset_id: int) -> pd.DataFrame:
    return conn.execute(
        """SELECT from_ts, to_ts, open, high, low, close, volume
           FROM candles WHERE dataset_id = ? ORDER BY from_ts""",
        (dataset_id,),
    ).df()


def latest_dataset_id(
    conn: duckdb.DuckDBPyConnection, asset: str, interval: int
) -> int | None:
    row = conn.execute(
        """SELECT id FROM datasets WHERE asset = ? AND interval_seconds = ?
           ORDER BY collected_at_utc DESC, id DESC LIMIT 1""",
        (asset, interval),
    ).fetchone()
    return row[0] if row else None


def export_midas_candles(
    conn: duckdb.DuckDBPyConnection, dataset_id: int, out_path: Path
) -> Path:
    """Export a dataset as a MIDAS Candle JSON array (RFC3339 timestamps)."""
    frame = load_candles(conn, dataset_id)
    records = [
        {
            "timestamp": datetime.fromtimestamp(int(row.from_ts), timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }
        for row in frame.itertuples()
    ]
    out_path.write_text(json.dumps(records))
    return out_path


def export_parquet(
    conn: duckdb.DuckDBPyConnection, dataset_id: int, out_path: Path
) -> Path:
    # COPY does not support prepared-statement parameters; inline them.
    escaped = str(out_path).replace("'", "''")
    conn.execute(
        f"""COPY (SELECT from_ts, to_ts, open, high, low, close, volume
                  FROM candles WHERE dataset_id = {int(dataset_id)} ORDER BY from_ts)
            TO '{escaped}' (FORMAT PARQUET)"""
    )
    return out_path
