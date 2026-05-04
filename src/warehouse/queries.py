"""Reusable reporting queries against the warehouse star schema.

All queries use ANSI SQL where possible so they run on PostgreSQL, SQL Server,
and SQLite (the test backend). The `?` placeholder works in sqlite3; psycopg2
expects `%s` — call sites pass `paramstyle` to choose.

Returned rows are tuples in the column order documented in each docstring.
"""
from __future__ import annotations

from collections.abc import Sequence

# ---------- query templates ----------------------------------------------------

DAILY_ANOMALY_COUNTS = """
SELECT d.full_date,
       COUNT(*) AS n_windows,
       SUM(f.predicted) AS n_anomaly_windows,
       AVG(f.score) AS mean_score,
       MAX(f.score) AS peak_score
FROM fact_window_score f
JOIN dim_date d ON d.date_key = f.date_key
WHERE f.run_id = {p}
GROUP BY d.full_date
ORDER BY d.full_date
""".strip()

TOP_N_ANOMALY_WINDOWS = """
SELECT f.ts, f.window_start, f.window_end, f.score, f.label
FROM fact_window_score f
WHERE f.run_id = {p} AND f.predicted = 1
ORDER BY f.score DESC
LIMIT {p}
""".strip()

ACTIVE_EVENTS_IN_RANGE = """
SELECT e.event_id, e.started_at, e.ended_at, e.duration_sec,
       e.peak_score, e.mean_score, e.n_windows
FROM fact_anomaly_event e
WHERE e.run_id = {p}
  AND e.started_at <= {p}
  AND e.ended_at   >= {p}
ORDER BY e.started_at
""".strip()

CONFUSION_BY_RUN = """
SELECT
    SUM(CASE WHEN f.predicted = 1 AND f.label = 1 THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN f.predicted = 1 AND f.label = 0 THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN f.predicted = 0 AND f.label = 1 THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN f.predicted = 0 AND f.label = 0 THEN 1 ELSE 0 END) AS tn
FROM fact_window_score f
WHERE f.run_id = {p} AND f.label IS NOT NULL
""".strip()

ROLLING_MEAN_SCORE = """
SELECT d.full_date, AVG(f.score) AS mean_score
FROM fact_window_score f
JOIN dim_date d ON d.date_key = f.date_key
WHERE f.run_id = {p}
GROUP BY d.full_date
ORDER BY d.full_date
""".strip()


# ---------- python-side runners -----------------------------------------------

def _placeholder(paramstyle: str) -> str:
    return "?" if paramstyle == "qmark" else "%s"


def _render(template: str, paramstyle: str) -> str:
    return template.replace("{p}", _placeholder(paramstyle))


def daily_anomaly_counts(conn, run_id: int, paramstyle: str = "qmark") -> list[tuple]:
    """One row per date: (full_date, n_windows, n_anomaly_windows, mean_score, peak_score)."""
    cur = conn.cursor()
    cur.execute(_render(DAILY_ANOMALY_COUNTS, paramstyle), (run_id,))
    return cur.fetchall()


def top_n_anomaly_windows(conn, run_id: int, n: int = 20, paramstyle: str = "qmark") -> list[tuple]:
    """Top-N predicted-anomaly windows: (ts, window_start, window_end, score, label)."""
    cur = conn.cursor()
    cur.execute(_render(TOP_N_ANOMALY_WINDOWS, paramstyle), (run_id, int(n)))
    return cur.fetchall()


def active_events_in_range(conn, run_id: int, start_iso: str, end_iso: str,
                           paramstyle: str = "qmark") -> list[tuple]:
    """Anomaly events overlapping [start_iso, end_iso]: full event row."""
    cur = conn.cursor()
    cur.execute(_render(ACTIVE_EVENTS_IN_RANGE, paramstyle), (run_id, end_iso, start_iso))
    return cur.fetchall()


def confusion(conn, run_id: int, paramstyle: str = "qmark") -> tuple[int, int, int, int]:
    """Per-run confusion matrix over labelled rows: (tp, fp, fn, tn)."""
    cur = conn.cursor()
    cur.execute(_render(CONFUSION_BY_RUN, paramstyle), (run_id,))
    row = cur.fetchone() or (0, 0, 0, 0)
    return tuple(int(x or 0) for x in row)


def precision_recall_f1(conn, run_id: int, paramstyle: str = "qmark") -> tuple[float, float, float]:
    """Convenience: derive precision/recall/F1 from `confusion`."""
    tp, fp, fn, _tn = confusion(conn, run_id, paramstyle)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def rolling_mean_score(conn, run_id: int, paramstyle: str = "qmark") -> Sequence[tuple]:
    """One row per date: (full_date, mean_score)."""
    cur = conn.cursor()
    cur.execute(_render(ROLLING_MEAN_SCORE, paramstyle), (run_id,))
    return cur.fetchall()
