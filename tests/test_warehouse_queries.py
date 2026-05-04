import sqlite3
from datetime import datetime, timedelta

from src.warehouse import queries
from src.warehouse.writer import (
    AnomalyEvent,
    WindowScore,
    init_schema,
    upsert_run,
    write_events,
    write_window_scores,
)


def _seeded_conn():
    conn = sqlite3.connect(":memory:")
    init_schema(conn, "sqlite")
    t0 = datetime(2026, 1, 15, 10, 0, 0)
    run_id = upsert_run(conn, run_name="r1", model_version="v1",
                        dataset="skab", threshold=0.5, started_at=t0)
    rows = []
    # day 1: 5 windows, 2 anomalies (labels match)
    for i in range(5):
        ts = t0 + timedelta(minutes=i)
        score = 0.9 if i in (2, 3) else 0.1
        label = 1 if i in (2, 3) else 0
        rows.append(WindowScore(ts=ts, window_start=ts - timedelta(minutes=1),
                                window_end=ts, score=score, predicted=int(score >= 0.5),
                                label=label))
    # day 2: 3 windows, 1 false positive (predicted=1, label=0)
    t1 = t0 + timedelta(days=1)
    for i in range(3):
        ts = t1 + timedelta(minutes=i)
        score = 0.8 if i == 0 else 0.05
        label = 0
        rows.append(WindowScore(ts=ts, window_start=ts - timedelta(minutes=1),
                                window_end=ts, score=score, predicted=int(score >= 0.5),
                                label=label))
    write_window_scores(conn, run_id=run_id, rows=rows)
    write_events(conn, run_id=run_id, events=[
        AnomalyEvent(started_at=t0 + timedelta(minutes=2),
                     ended_at=t0 + timedelta(minutes=3),
                     peak_score=0.9, mean_score=0.9, n_windows=2),
    ])
    return conn, run_id, t0


def test_daily_anomaly_counts_returns_one_row_per_day():
    conn, run_id, _ = _seeded_conn()
    rows = queries.daily_anomaly_counts(conn, run_id)
    assert len(rows) == 2
    day1, day2 = rows
    # day1: 5 windows, 2 anomalies. day2: 3 windows, 1 anomaly.
    assert day1[1] == 5 and day1[2] == 2
    assert day2[1] == 3 and day2[2] == 1


def test_top_n_anomaly_windows_orders_by_score_desc():
    conn, run_id, _ = _seeded_conn()
    rows = queries.top_n_anomaly_windows(conn, run_id, n=10)
    assert len(rows) == 3  # 3 predicted=1 across both days
    scores = [r[3] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_confusion_and_metrics():
    conn, run_id, _ = _seeded_conn()
    tp, fp, fn, tn = queries.confusion(conn, run_id)
    # day1: tp=2, day2: fp=1, tn=2 day1 + 2 day2 = 4
    assert (tp, fp, fn, tn) == (2, 1, 0, 5)
    p, r, f1 = queries.precision_recall_f1(conn, run_id)
    assert abs(p - 2 / 3) < 1e-9
    assert r == 1.0
    assert abs(f1 - (2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))) < 1e-9


def test_active_events_in_range_includes_overlap_only():
    conn, run_id, t0 = _seeded_conn()
    s = (t0 + timedelta(minutes=1)).isoformat()
    e = (t0 + timedelta(minutes=4)).isoformat()
    rows = queries.active_events_in_range(conn, run_id, s, e)
    assert len(rows) == 1

    s2 = (t0 + timedelta(days=5)).isoformat()
    e2 = (t0 + timedelta(days=6)).isoformat()
    assert queries.active_events_in_range(conn, run_id, s2, e2) == []
