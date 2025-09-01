#!/usr/bin/env python3
"""
ClickHouse Analytics for System Logs
- Runs baseline analytics
- Exports CSVs to data/exports
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests
import pandas as pd

CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
DATABASE = os.environ.get("CLICKHOUSE_DB", "system_logs")
TABLE = os.environ.get("CLICKHOUSE_TABLE", "raw_logs")
EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "exports"))

os.makedirs(EXPORT_DIR, exist_ok=True)

def execute(query: str, fmt: str = "JSONEachRow") -> List[Dict[str, Any]]:
    if "FORMAT" not in query.upper():
        query = f"{query} FORMAT {fmt}"
    r = requests.post(
        f"{CLICKHOUSE_URL}/",
        params={"query": query},
        headers={"Content-Type": "text/plain"},
        timeout=60,
    )
    r.raise_for_status()
    if fmt == "JSONEachRow":
        return [json.loads(line) for line in r.text.strip().splitlines() if line.strip()]
    return [{"result": r.text.strip()}]

def to_csv(rows: List[Dict[str, Any]], filename: str) -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    path = os.path.join(EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    return path

def main():
    print("🚀 ClickHouse Analytics")
    print("=" * 60)

    # Basic stats
    stats = execute(f"SELECT count() AS total, min(timestamp) AS earliest, max(timestamp) AS latest FROM {DATABASE}.{TABLE}")
    total = int(stats[0]["total"]) if stats else 0
    earliest = stats[0]["earliest"] if stats else None
    latest = stats[0]["latest"] if stats else None
    print(f"Total logs: {total:,}")
    print(f"Range: {earliest} .. {latest}")

    # Top hosts
    hosts = execute(f"SELECT host, count() AS count FROM {DATABASE}.{TABLE} GROUP BY host ORDER BY count DESC LIMIT 10")
    path_hosts = to_csv(hosts, f"top_hosts_{int(time.time())}.csv")
    print(f"Top hosts exported: {path_hosts}")

    # Top files
    files = execute(f"SELECT file, count() AS count FROM {DATABASE}.{TABLE} GROUP BY file ORDER BY count DESC LIMIT 10")
    path_files = to_csv(files, f"top_files_{int(time.time())}.csv")
    print(f"Top files exported: {path_files}")

    # Per-minute counts (string timestamps -> substring)
    per_min = execute(
        f"SELECT substring(timestamp,1,16) AS minute, count() AS count FROM {DATABASE}.{TABLE} GROUP BY minute ORDER BY minute DESC LIMIT 120"
    )
    path_min = to_csv(per_min, f"per_minute_{int(time.time())}.csv")
    print(f"Per-minute counts exported: {path_min}")

    # Message length histogram (bucketed by 100s)
    hist = execute(
        f"SELECT toInt32(floor(length(message)/100))*100 AS bucket, count() AS count FROM {DATABASE}.{TABLE} WHERE message != '' GROUP BY bucket ORDER BY bucket"
    )
    path_hist = to_csv(hist, f"message_length_hist_{int(time.time())}.csv")
    print(f"Message length histogram exported: {path_hist}")

    # Approximate severity by keywords
    severity = execute(
        f"""
        SELECT
          sum(IF(lower(message) ILIKE '%error%' OR lower(message) ILIKE '%failed%' OR lower(message) ILIKE '%violation%', 1, 0)) AS errors,
          sum(IF(lower(message) ILIKE '%warn%' OR lower(message) ILIKE '%deprecated%', 1, 0)) AS warnings,
          sum(IF(lower(message) ILIKE '%debug%' OR lower(message) ILIKE '%trace%', 1, 0)) AS debugs
        FROM {DATABASE}.{TABLE}
        """
    )
    path_sev = to_csv(severity, f"severity_estimate_{int(time.time())}.csv")
    print(f"Severity estimate exported: {path_sev}")

    # Sample export (first 10k)
    sample = execute(
        f"SELECT timestamp, host, file, source_type, message FROM {DATABASE}.{TABLE} WHERE message != '' ORDER BY timestamp LIMIT 10000"
    )
    path_sample = to_csv(sample, f"sample_10k_{int(time.time())}.csv")
    print(f"Sample 10k exported: {path_sample}")

    print("\n✅ Analytics completed.")

if __name__ == "__main__":
    main()
