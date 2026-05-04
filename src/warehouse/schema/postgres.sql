-- PostgreSQL DDL for the anomaly-detection reporting warehouse.
-- Star schema: dim_run, dim_sensor, dim_date + fact_window_score, fact_anomaly_event.

CREATE TABLE IF NOT EXISTS dim_run (
    run_id          BIGSERIAL PRIMARY KEY,
    run_name        TEXT NOT NULL UNIQUE,
    model_version   TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dim_sensor (
    sensor_id       BIGSERIAL PRIMARY KEY,
    sensor_name     TEXT NOT NULL UNIQUE,
    channel_index   INTEGER NOT NULL,
    unit_of_measure TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_key        INTEGER PRIMARY KEY,           -- yyyymmdd
    full_date       DATE NOT NULL,
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    day             INTEGER NOT NULL,
    weekday         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_window_score (
    window_id       BIGSERIAL PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES dim_run(run_id),
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    ts              TIMESTAMPTZ NOT NULL,
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    score           DOUBLE PRECISION NOT NULL,
    predicted       SMALLINT NOT NULL,
    label           SMALLINT
);

CREATE INDEX IF NOT EXISTS ix_fws_run_ts ON fact_window_score(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_fws_date   ON fact_window_score(date_key);

CREATE TABLE IF NOT EXISTS fact_anomaly_event (
    event_id        BIGSERIAL PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES dim_run(run_id),
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ NOT NULL,
    duration_sec    INTEGER NOT NULL,
    peak_score      DOUBLE PRECISION NOT NULL,
    mean_score      DOUBLE PRECISION NOT NULL,
    n_windows       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_fae_run ON fact_anomaly_event(run_id, started_at);
