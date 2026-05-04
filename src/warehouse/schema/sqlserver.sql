-- SQL Server DDL for the anomaly-detection reporting warehouse.
-- Same star schema as the Postgres DDL; types adapted to T-SQL.

IF OBJECT_ID('dbo.dim_run', 'U') IS NULL
CREATE TABLE dbo.dim_run (
    run_id          BIGINT IDENTITY(1,1) PRIMARY KEY,
    run_name        NVARCHAR(200) NOT NULL UNIQUE,
    model_version   NVARCHAR(64)  NOT NULL,
    dataset         NVARCHAR(64)  NOT NULL,
    threshold       FLOAT NOT NULL,
    started_at      DATETIME2 NOT NULL,
    ended_at        DATETIME2 NULL
);

IF OBJECT_ID('dbo.dim_sensor', 'U') IS NULL
CREATE TABLE dbo.dim_sensor (
    sensor_id       BIGINT IDENTITY(1,1) PRIMARY KEY,
    sensor_name     NVARCHAR(200) NOT NULL UNIQUE,
    channel_index   INT NOT NULL,
    unit_of_measure NVARCHAR(64) NULL
);

IF OBJECT_ID('dbo.dim_date', 'U') IS NULL
CREATE TABLE dbo.dim_date (
    date_key        INT PRIMARY KEY,            -- yyyymmdd
    full_date       DATE NOT NULL,
    [year]          INT NOT NULL,
    [month]         INT NOT NULL,
    [day]           INT NOT NULL,
    weekday         INT NOT NULL
);

IF OBJECT_ID('dbo.fact_window_score', 'U') IS NULL
CREATE TABLE dbo.fact_window_score (
    window_id       BIGINT IDENTITY(1,1) PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES dbo.dim_run(run_id),
    date_key        INT    NOT NULL REFERENCES dbo.dim_date(date_key),
    ts              DATETIME2 NOT NULL,
    window_start    DATETIME2 NOT NULL,
    window_end      DATETIME2 NOT NULL,
    score           FLOAT NOT NULL,
    predicted       SMALLINT NOT NULL,
    label           SMALLINT NULL
);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_fws_run_ts')
    CREATE INDEX ix_fws_run_ts ON dbo.fact_window_score(run_id, ts);
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_fws_date')
    CREATE INDEX ix_fws_date   ON dbo.fact_window_score(date_key);

IF OBJECT_ID('dbo.fact_anomaly_event', 'U') IS NULL
CREATE TABLE dbo.fact_anomaly_event (
    event_id        BIGINT IDENTITY(1,1) PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES dbo.dim_run(run_id),
    date_key        INT    NOT NULL REFERENCES dbo.dim_date(date_key),
    started_at      DATETIME2 NOT NULL,
    ended_at        DATETIME2 NOT NULL,
    duration_sec    INT NOT NULL,
    peak_score      FLOAT NOT NULL,
    mean_score      FLOAT NOT NULL,
    n_windows       INT NOT NULL
);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_fae_run')
    CREATE INDEX ix_fae_run ON dbo.fact_anomaly_event(run_id, started_at);
