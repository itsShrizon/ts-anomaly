#!/usr/bin/env python
"""Generate the per-run report bundle (daily.csv, top.csv, summary.html).

Usage:
    python scripts/run_report.py --db artifacts/warehouse.sqlite --run-name skab_v1 \
        --out-dir artifacts/reports/skab_v1
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from src.warehouse.report import write_bundle


def _resolve_run_id(conn, run_name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT run_id FROM dim_run WHERE run_name = ?", (run_name,))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"no run found with run_name={run_name!r}")
    return int(row[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to a sqlite warehouse file")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")
    out_dir = Path(args.out_dir)

    with sqlite3.connect(db_path) as conn:
        run_id = _resolve_run_id(conn, args.run_name)
        paths = write_bundle(conn, run_id, out_dir, top_n=args.top_n, paramstyle="qmark")
    for label, p in paths.items():
        print(f"{label}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
