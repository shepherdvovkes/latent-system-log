-- Initialize ClickHouse database and table for system logs
-- This script will be executed when ClickHouse starts

-- Create database
CREATE DATABASE IF NOT EXISTS system_logs;

-- Create optimized table for system logs
CREATE TABLE IF NOT EXISTS system_logs.logs (
    timestamp DateTime64(3),
    timestamp_unix Float64,
    source String,
    level Enum8('info' = 1, 'warning' = 2, 'error' = 3, 'debug' = 4),
    message String,
    thread_id String,
    log_type String,
    activity_id String,
    pid UInt32,
    ttl UInt32,
    host String,
    processor String,
    ingestion_time DateTime64(3),
    metadata String
) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (timestamp, source, level)
    TTL timestamp + INTERVAL 1 YEAR
    SETTINGS index_granularity = 8192;

-- Create additional indexes for better performance
CREATE INDEX IF NOT EXISTS idx_source ON system_logs.logs (source) TYPE bloom_filter GRANULARITY 1;
CREATE INDEX IF NOT EXISTS idx_level ON system_logs.logs (level) TYPE bloom_filter GRANULARITY 1;
CREATE INDEX IF NOT EXISTS idx_pid ON system_logs.logs (pid) TYPE bloom_filter GRANULARITY 1;

-- Create materialized view for error logs
CREATE MATERIALIZED VIEW IF NOT EXISTS system_logs.error_logs_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, source)
AS SELECT
    timestamp,
    source,
    message,
    pid,
    thread_id,
    metadata
FROM system_logs.logs
WHERE level = 'error';

-- Create materialized view for warning logs
CREATE MATERIALIZED VIEW IF NOT EXISTS system_logs.warning_logs_mv
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, source)
AS SELECT
    timestamp,
    source,
    message,
    pid,
    thread_id,
    metadata
FROM system_logs.logs
WHERE level = 'warning';

-- Show table structure
DESCRIBE system_logs.logs;
