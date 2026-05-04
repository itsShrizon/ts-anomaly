"""Write inference outputs into the reporting warehouse.

Connection-agnostic: takes any PEP-249 connection (sqlite3, psycopg2, pyodbc).
Inserts use named params with the `?` paramstyle for sqlite or `%s` for libs that
prefer that style — the caller passes `paramstyle` accordingly.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from .dialect import ddl_for, split_statements


@dataclass(frozen=True)
class WindowScore:
    ts: datetime
    window_start: datetime
    window_end: datetime
    score: float
    predicted: int
    label: int | None = None


@dataclass(frozen=True)
class AnomalyEvent:
    started_at: datetime
    ended_at: datetime
    peak_score: float
    mean_score: float
    n_windows: int

    @property
    def duration_sec(self) -> int:
        return int((self.ended_at - self.started_at).total_seconds())


def _q(paramstyle: str) -> str:
    return "?" if paramstyle == "qmark" else "%s"


def init_schema(conn, backend: str = "sqlite") -> None:
    """Create the warehouse tables if they don't exist."""
    cur = conn.cursor()
    for stmt in split_statements(ddl_for(backend)):
        cur.execute(stmt)
    conn.commit()


def upsert_run(
    conn,
    *,
    run_name: str,
    model_version: str,
    dataset: str,
    threshold: float,
    started_at: datetime,
    ended_at: datetime | None = None,
    paramstyle: str = "qmark",
) -> int:
    """Insert (or fetch) a dim_run row and return its run_id."""
    p = _q(paramstyle)
    cur = conn.cursor()
    cur.execute(f"SELECT run_id FROM dim_run WHERE run_name = {p}", (run_name,))
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        f"INSERT INTO dim_run (run_name, model_version, dataset, threshold, started_at, ended_at) "
        f"VALUES ({p}, {p}, {p}, {p}, {p}, {p})",
        (run_name, model_version, dataset, float(threshold),
         started_at.isoformat(), ended_at.isoformat() if ended_at else None),
    )
    conn.commit()
    cur.execute(f"SELECT run_id FROM dim_run WHERE run_name = {p}", (run_name,))
    return int(cur.fetchone()[0])


def upsert_dates(conn, days: Iterable[date], paramstyle: str = "qmark") -> None:
    """Ensure a dim_date row exists for every date in `days`."""
    p = _q(paramstyle)
    cur = conn.cursor()
    for d in {d for d in days}:
        key = d.year * 10000 + d.month * 100 + d.day
        cur.execute(f"SELECT 1 FROM dim_date WHERE date_key = {p}", (key,))
        if cur.fetchone():
            continue
        cur.execute(
            f"INSERT INTO dim_date (date_key, full_date, year, month, day, weekday) "
            f"VALUES ({p}, {p}, {p}, {p}, {p}, {p})",
            (key, d.isoformat(), d.year, d.month, d.day, d.weekday()),
        )
    conn.commit()


def write_window_scores(
    conn,
    *,
    run_id: int,
    rows: Sequence[WindowScore],
    paramstyle: str = "qmark",
) -> int:
    """Bulk-insert per-window scores. Auto-populates dim_date for rows' dates."""
    if not rows:
        return 0
    upsert_dates(conn, (r.ts.date() for r in rows), paramstyle)
    p = _q(paramstyle)
    cur = conn.cursor()
    cur.executemany(
        f"INSERT INTO fact_window_score "
        f"(run_id, date_key, ts, window_start, window_end, score, predicted, label) "
        f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})",
        [
            (
                run_id,
                r.ts.year * 10000 + r.ts.month * 100 + r.ts.day,
                r.ts.isoformat(),
                r.window_start.isoformat(),
                r.window_end.isoformat(),
                float(r.score),
                int(r.predicted),
                None if r.label is None else int(r.label),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def write_events(
    conn,
    *,
    run_id: int,
    events: Sequence[AnomalyEvent],
    paramstyle: str = "qmark",
) -> int:
    if not events:
        return 0
    upsert_dates(conn, (e.started_at.date() for e in events), paramstyle)
    p = _q(paramstyle)
    cur = conn.cursor()
    cur.executemany(
        f"INSERT INTO fact_anomaly_event "
        f"(run_id, date_key, started_at, ended_at, duration_sec, peak_score, mean_score, n_windows) "
        f"VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})",
        [
            (
                run_id,
                e.started_at.year * 10000 + e.started_at.month * 100 + e.started_at.day,
                e.started_at.isoformat(),
                e.ended_at.isoformat(),
                e.duration_sec,
                float(e.peak_score),
                float(e.mean_score),
                int(e.n_windows),
            )
            for e in events
        ],
    )
    conn.commit()
    return len(events)


def events_from_scores(
    timestamps: Sequence[datetime],
    scores: Sequence[float],
    threshold: float,
    *,
    min_run: int = 1,
) -> list[AnomalyEvent]:
    """Group consecutive above-threshold windows into anomaly events.

    `timestamps[i]` is the end-of-window time for `scores[i]`. Adjacent indices
    are treated as contiguous; gaps are not stitched.
    """
    if len(timestamps) != len(scores):
        raise ValueError("timestamps and scores must be the same length")
    above = np.asarray(scores, dtype=float) >= float(threshold)
    out: list[AnomalyEvent] = []
    i = 0
    n = len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        run_len = j - i + 1
        if run_len >= min_run:
            chunk = np.asarray(scores[i : j + 1], dtype=float)
            out.append(
                AnomalyEvent(
                    started_at=timestamps[i],
                    ended_at=timestamps[j],
                    peak_score=float(chunk.max()),
                    mean_score=float(chunk.mean()),
                    n_windows=int(run_len),
                )
            )
        i = j + 1
    return out
