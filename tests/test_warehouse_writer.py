import sqlite3
from datetime import datetime, timedelta

from src.warehouse.dialect import ddl_for, split_statements
from src.warehouse.writer import (
    AnomalyEvent,
    WindowScore,
    events_from_scores,
    init_schema,
    upsert_run,
    write_events,
    write_window_scores,
)


def _conn():
    return sqlite3.connect(":memory:")


def test_split_statements_handles_comments_and_multi_lines():
    sql = ddl_for("sqlite")
    stmts = split_statements(sql)
    assert all("--" not in s for s in stmts)
    assert any(s.upper().startswith("CREATE TABLE") for s in stmts)
    assert any(s.upper().startswith("CREATE INDEX") for s in stmts)


def test_init_schema_creates_all_tables():
    conn = _conn()
    init_schema(conn, "sqlite")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {r[0] for r in cur.fetchall()}
    assert {"dim_run", "dim_sensor", "dim_date",
            "fact_window_score", "fact_anomaly_event"} <= tables


def test_upsert_run_is_idempotent():
    conn = _conn()
    init_schema(conn, "sqlite")
    t = datetime(2026, 1, 15, 10, 0, 0)
    a = upsert_run(conn, run_name="r1", model_version="v1",
                   dataset="skab", threshold=0.5, started_at=t)
    b = upsert_run(conn, run_name="r1", model_version="v1",
                   dataset="skab", threshold=0.5, started_at=t)
    assert a == b


def test_write_window_scores_inserts_and_populates_dim_date():
    conn = _conn()
    init_schema(conn, "sqlite")
    t = datetime(2026, 1, 15, 10, 0, 0)
    run_id = upsert_run(conn, run_name="r1", model_version="v1",
                        dataset="skab", threshold=0.5, started_at=t)
    rows = [
        WindowScore(ts=t + timedelta(seconds=i*30),
                    window_start=t + timedelta(seconds=i*30 - 60),
                    window_end=t + timedelta(seconds=i*30),
                    score=0.1 * i, predicted=int(0.1 * i >= 0.5),
                    label=1 if i in (5, 6, 7) else 0)
        for i in range(10)
    ]
    n = write_window_scores(conn, run_id=run_id, rows=rows)
    assert n == 10

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fact_window_score")
    assert cur.fetchone()[0] == 10
    cur.execute("SELECT COUNT(*) FROM dim_date")
    assert cur.fetchone()[0] == 1  # all on the same day


def test_events_from_scores_groups_runs():
    base = datetime(2026, 2, 1, 12, 0, 0)
    ts = [base + timedelta(seconds=i) for i in range(10)]
    scores = [0.1, 0.2, 0.9, 0.95, 0.92, 0.1, 0.1, 0.8, 0.85, 0.1]
    events = events_from_scores(ts, scores, threshold=0.7, min_run=1)
    assert len(events) == 2
    assert events[0].n_windows == 3
    assert events[0].peak_score == 0.95
    assert events[1].n_windows == 2


def test_events_from_scores_min_run_filters_short_runs():
    base = datetime(2026, 2, 1, 12, 0, 0)
    ts = [base + timedelta(seconds=i) for i in range(5)]
    scores = [0.1, 0.9, 0.1, 0.9, 0.9]
    events = events_from_scores(ts, scores, threshold=0.7, min_run=2)
    assert len(events) == 1
    assert events[0].n_windows == 2


def test_write_events_persists_event_rows():
    conn = _conn()
    init_schema(conn, "sqlite")
    t = datetime(2026, 1, 15, 10, 0, 0)
    run_id = upsert_run(conn, run_name="r1", model_version="v1",
                        dataset="skab", threshold=0.5, started_at=t)
    events = [
        AnomalyEvent(started_at=t, ended_at=t + timedelta(seconds=30),
                     peak_score=0.99, mean_score=0.85, n_windows=3),
    ]
    n = write_events(conn, run_id=run_id, events=events)
    assert n == 1
    cur = conn.cursor()
    cur.execute("SELECT duration_sec, n_windows FROM fact_anomaly_event")
    row = cur.fetchone()
    assert row == (30, 3)
