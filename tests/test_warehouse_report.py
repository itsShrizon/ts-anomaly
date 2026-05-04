import csv
import sqlite3
from datetime import datetime, timedelta

from src.warehouse.report import write_bundle, write_daily_csv, write_html, write_top_n_csv
from src.warehouse.writer import (
    WindowScore,
    init_schema,
    upsert_run,
    write_window_scores,
)


def _seeded_conn():
    conn = sqlite3.connect(":memory:")
    init_schema(conn, "sqlite")
    t0 = datetime(2026, 3, 1, 8, 0, 0)
    run_id = upsert_run(conn, run_name="r1", model_version="v1",
                        dataset="skab", threshold=0.5, started_at=t0)
    rows = [
        WindowScore(ts=t0 + timedelta(minutes=i),
                    window_start=t0 + timedelta(minutes=i - 1),
                    window_end=t0 + timedelta(minutes=i),
                    score=0.9 if i % 5 == 0 else 0.1,
                    predicted=int(0.9 if i % 5 == 0 else 0.1 >= 0.5),
                    label=1 if i % 5 == 0 else 0)
        for i in range(20)
    ]
    write_window_scores(conn, run_id=run_id, rows=rows)
    return conn, run_id


def test_write_daily_csv_emits_header_and_rows(tmp_path):
    conn, run_id = _seeded_conn()
    out = tmp_path / "daily.csv"
    n = write_daily_csv(conn, run_id, out)
    assert n >= 1
    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["full_date", "n_windows", "n_anomaly_windows", "mean_score", "peak_score"]
    assert len(rows) == n + 1


def test_write_top_n_csv_respects_limit(tmp_path):
    conn, run_id = _seeded_conn()
    out = tmp_path / "top.csv"
    write_top_n_csv(conn, run_id, out, n=2)
    with out.open() as f:
        rows = list(csv.reader(f))
    # header + at most 2 rows
    assert len(rows) <= 3
    assert rows[0][0] == "ts"


def test_write_html_writes_self_contained_summary(tmp_path):
    conn, run_id = _seeded_conn()
    out = tmp_path / "summary.html"
    write_html(conn, run_id, out)
    text = out.read_text(encoding="utf-8")
    assert "<table" in text
    assert "Precision" in text
    assert "Recall" in text


def test_write_bundle_produces_three_artifacts(tmp_path):
    conn, run_id = _seeded_conn()
    paths = write_bundle(conn, run_id, tmp_path / "bundle", top_n=10)
    assert set(paths) == {"daily", "top", "summary"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
