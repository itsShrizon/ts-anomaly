# Reporting warehouse

The `src/warehouse/` module is an additive reporting layer on top of the
inference pipeline. It does not participate in training or scoring — it only
consumes outputs and exposes them to MIS/FIS-style scheduled reporting.

## Star schema

```
dim_run --------+
                |
dim_date -----> fact_window_score   (per-window score, prediction, label)
                |
                +--> fact_anomaly_event  (contiguous above-threshold runs)

dim_sensor (optional, for per-channel drift dashboards)
```

DDL ships in three flavours:

| backend     | file                                |
| ----------- | ----------------------------------- |
| PostgreSQL  | `src/warehouse/schema/postgres.sql` |
| SQL Server  | `src/warehouse/schema/sqlserver.sql`|
| SQLite      | `src/warehouse/schema/sqlite.sql`   |

The PostgreSQL and SQL Server schemas are equivalent — same tables, same
columns, same semantics. Type aliases differ (`BIGSERIAL` vs `BIGINT IDENTITY`,
`TIMESTAMPTZ` vs `DATETIME2`), and SQL Server uses object-existence guards
(`IF OBJECT_ID(...) IS NULL`) instead of `IF NOT EXISTS`.

## Writing detections

```python
import sqlite3
from datetime import datetime
from src.warehouse.writer import (
    init_schema, upsert_run, write_window_scores,
    write_events, events_from_scores, WindowScore,
)

conn = sqlite3.connect("artifacts/warehouse.sqlite")
init_schema(conn, backend="sqlite")

run_id = upsert_run(
    conn, run_name="skab_v1", model_version="0.4.0",
    dataset="skab", threshold=0.5,
    started_at=datetime.utcnow(),
)

write_window_scores(conn, run_id=run_id, rows=[
    WindowScore(ts=ts, window_start=ws, window_end=we,
                score=s, predicted=int(s >= 0.5), label=y)
    for ts, ws, we, s, y in zip(timestamps, w_starts, w_ends, scores, labels, strict=True)
])

events = events_from_scores(timestamps, scores, threshold=0.5, min_run=2)
write_events(conn, run_id=run_id, events=events)
```

For PostgreSQL / SQL Server, swap `sqlite3.connect(...)` for
`psycopg2.connect(...)` / `pyodbc.connect(...)` and pass `paramstyle="format"`
(psycopg2) — the writer uses `%s` placeholders accordingly.

## Reporting queries

`src/warehouse/queries.py` exposes:

| function                  | what it returns                                         |
| ------------------------- | ------------------------------------------------------- |
| `daily_anomaly_counts`    | per-day rollup: counts, mean / peak score               |
| `top_n_anomaly_windows`   | top-N predicted-anomaly windows by score                |
| `active_events_in_range`  | anomaly events overlapping a given time window          |
| `confusion`               | (TP, FP, FN, TN) over labelled rows for a run           |
| `precision_recall_f1`     | derived from `confusion`                                |
| `rolling_mean_score`      | per-day mean score (drift surface)                      |

## Scheduled report bundle

`scripts/run_report.py` builds `daily.csv`, `top.csv`, and a self-contained
`summary.html` file that the MIS/FIS team can open without any tooling.
Hook it into cron / Task Scheduler / a CI step to deliver reports on a
fixed cadence.

```bash
python scripts/run_report.py \
    --db artifacts/warehouse.sqlite \
    --run-name skab_v1 \
    --out-dir artifacts/reports/skab_v1 \
    --top-n 50
```

## Operational notes

- All write paths are connection-agnostic: pass any PEP-249 connection.
- `init_schema` is idempotent — safe to call on every job start.
- `upsert_run` is idempotent on `run_name` — safe to retry.
- For high-volume ingest, consider wrapping `write_window_scores` calls in a
  single transaction and disabling autocommit on the driver.
