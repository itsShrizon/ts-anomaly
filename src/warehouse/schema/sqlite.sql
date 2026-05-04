-- SQLite DDL — used as a portable test backend for the warehouse module.
-- Mirrors the Postgres / SQL Server star schema with sqlite-compatible types.

CREATE TABLE IF NOT EXISTS dim_run (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name        TEXT NOT NULL UNIQUE,
    model_version   TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    threshold       REAL NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT
);

CREATE TABLE IF NOT EXISTS dim_sensor (
    sensor_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_name     TEXT NOT NULL UNIQUE,
    channel_index   INTEGER NOT NULL,
    unit_of_measure TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_key        INTEGER PRIMARY KEY,
    full_date       TEXT NOT NULL,
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    day             INTEGER NOT NULL,
    weekday         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_window_score (
    window_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES dim_run(run_id),
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    ts              TEXT NOT NULL,
    window_start    TEXT NOT NULL,
    window_end      TEXT NOT NULL,
    score           REAL NOT NULL,
    predicted       INTEGER NOT NULL,
    label           INTEGER
);

CREATE INDEX IF NOT EXISTS ix_fws_run_ts ON fact_window_score(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_fws_date   ON fact_window_score(date_key);

CREATE TABLE IF NOT EXISTS fact_anomaly_event (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES dim_run(run_id),
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    started_at      TEXT NOT NULL,
    ended_at        TEXT NOT NULL,
    duration_sec    INTEGER NOT NULL,
    peak_score      REAL NOT NULL,
    mean_score      REAL NOT NULL,
    n_windows       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_fae_run ON fact_anomaly_event(run_id, started_at);
