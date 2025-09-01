#!/usr/bin/env python3
"""
Incremental Latent Space Builder from ClickHouse
- Streams messages from ClickHouse in chunks
- Converts to LogEntry and updates latent space incrementally
- Saves progress periodically
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from loguru import logger

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from app.services.latent_space import LatentSpaceService
from app.models.schemas import LogEntry

CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
DATABASE = os.environ.get("CLICKHOUSE_DB", "system_logs")
TABLE = os.environ.get("CLICKHOUSE_TABLE", "raw_logs")

CHUNK_SIZE = int(os.environ.get("LS_CHUNK_SIZE", "5000"))
MAX_MESSAGES = int(os.environ.get("LS_MAX_MESSAGES", "100000"))


def ch_execute(query: str, fmt: str = "JSONEachRow") -> List[Dict[str, Any]]:
    if "FORMAT" not in query.upper():
        query = f"{query} FORMAT {fmt}"
    r = requests.post(
        f"{CLICKHOUSE_URL}/",
        params={"query": query},
        headers={"Content-Type": "text/plain"},
        timeout=120,
    )
    r.raise_for_status()
    if fmt == "JSONEachRow":
        return [json.loads(line) for line in r.text.strip().splitlines() if line.strip()]
    return [{"result": r.text.strip()}]


def fetch_messages(offset: int, limit: int) -> List[str]:
    # Pull messages in timestamp order for determinism
    q = f"""
    SELECT message, timestamp, host, file
    FROM {DATABASE}.{TABLE}
    WHERE message != ''
    ORDER BY timestamp
    LIMIT {limit} OFFSET {offset}
    """
    rows = ch_execute(q)
    messages: List[str] = []
    for r in rows:
        msg = r.get("message", "")
        if msg and len(msg.strip()) > 10:
            messages.append(msg.strip())
    return messages


def to_log_entries(messages: List[str]) -> List[LogEntry]:
    now = datetime.now()
    entries: List[LogEntry] = []
    for i, m in enumerate(messages):
        entries.append(LogEntry(
            timestamp=now,
            source=f"clickhouse_chunk",
            level="info",
            message=m,
            metadata={"source": "clickhouse"}
        ))
    return entries


async def main():
    logger.info("Starting incremental latent space build")

    # Init latent space service
    ls = LatentSpaceService()
    await ls.initialize()

    processed = 0
    offset = 0
    start = time.time()

    while processed < MAX_MESSAGES:
        remaining = MAX_MESSAGES - processed
        take = min(CHUNK_SIZE, remaining)

        msgs = fetch_messages(offset, take)
        if not msgs:
            logger.info("No more messages from ClickHouse; stopping.")
            break

        entries = to_log_entries(msgs)
        await ls.update_latent_space(entries)

        processed += len(entries)
        offset += len(entries)
        logger.info(f"Processed {processed}/{MAX_MESSAGES} messages so far")

        # brief pause to avoid overwhelming
        time.sleep(0.2)

    elapsed = time.time() - start
    logger.info(f"Incremental build completed: processed={processed}, seconds={elapsed:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
