#!/usr/bin/env python3
"""
ClickHouse Embeddings Manager
- Generate embeddings for messages from raw_logs
- Store as Array(Float32) in system_logs.embeddings
- Provide cosine similarity search
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from datetime import datetime
import requests
import numpy as np
from loguru import logger

# Local app imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from app.services.latent_space import LatentSpaceService

CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
DATABASE = os.environ.get("CLICKHOUSE_DB", "system_logs")
RAW_TABLE = os.environ.get("CLICKHOUSE_TABLE", "raw_logs")
EMB_TABLE = os.environ.get("CLICKHOUSE_EMB_TABLE", "embeddings")

BATCH = int(os.environ.get("EMB_BATCH", "2000"))
LIMIT = int(os.environ.get("EMB_LIMIT", "100000"))


def ch(query: str, fmt: str = None) -> str:
    params = {"query": query}
    r = requests.post(f"{CLICKHOUSE_URL}/", params=params, headers={"Content-Type": "text/plain"}, timeout=120)
    r.raise_for_status()
    return r.text


def ch_rows(query: str) -> List[Dict[str, Any]]:
    if "FORMAT" not in query.upper():
        query += " FORMAT JSONEachRow"
    txt = ch(query)
    return [json.loads(line) for line in txt.strip().splitlines() if line.strip()]


def fetch_messages(offset: int, limit: int) -> List[Dict[str, Any]]:
    q = f"""
    SELECT message, timestamp, host, file
    FROM {DATABASE}.{RAW_TABLE}
    WHERE message != ''
    ORDER BY timestamp
    LIMIT {limit} OFFSET {offset}
    """
    return ch_rows(q)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def insert_embeddings(rows: List[Dict[str, Any]], embeddings: np.ndarray):
    # Prepare VALUES in JSONEachRow for insert
    payload_lines = []
    for row, emb in zip(rows, embeddings):
        payload = {
            "message": row["message"],
            "timestamp": row["timestamp"],
            "host": row["host"],
            "file": row["file"],
            "embedding": [float(x) for x in emb.tolist()],
        }
        payload_lines.append(json.dumps(payload))

    data = "\n".join(payload_lines)
    query = f"INSERT INTO {DATABASE}.{EMB_TABLE} (message, timestamp, host, file, embedding) FORMAT JSONEachRow"
    r = requests.post(f"{CLICKHOUSE_URL}/", params={"query": query}, data=data.encode("utf-8"), headers={"Content-Type": "application/x-ndjson"}, timeout=300)
    r.raise_for_status()


def backfill():
    logger.info("Initializing embedding model via LatentSpaceService…")
    ls = LatentSpaceService()
    import asyncio
    asyncio.run(ls.initialize())

    processed = 0
    offset = 0
    start = time.time()

    while processed < LIMIT:
        take = min(BATCH, LIMIT - processed)
        rows = fetch_messages(offset, take)
        if not rows:
            logger.info("No more rows to process.")
            break

        texts = [r["message"].strip() for r in rows if r.get("message")] 
        if not texts:
            offset += len(rows)
            continue

        logger.info(f"Encoding {len(texts)} messages…")
        embeddings = ls.embedding_model.encode(texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True)
        embeddings = normalize(embeddings.astype("float32"))

        insert_embeddings(rows, embeddings)

        processed += len(texts)
        offset += len(rows)
        logger.info(f"Backfilled {processed}/{LIMIT}")

    elapsed = time.time() - start
    logger.info(f"Backfill complete: {processed} rows in {elapsed:.2f}s")


def search(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    # Encode query
    ls = LatentSpaceService()
    import asyncio
    asyncio.run(ls.initialize())
    qv = ls.embedding_model.encode([query_text], show_progress_bar=False, convert_to_numpy=True).astype("float32")
    qv = normalize(qv)[0]

    # Build a SELECT that computes cosine similarity = dot(embedding, qv)
    # ClickHouse Array(Float32) → use arraySum for dot product
    # dot(a,b) = arraySum(i -> a[i] * b[i])
    coeffs = ",".join(str(float(x)) for x in qv.tolist())
    query = f"""
    SELECT message, timestamp, host, file,
           arraySum(arrayMap((a,b) -> a*b, embedding, [{coeffs}])) AS score
    FROM {DATABASE}.{EMB_TABLE}
    ORDER BY score DESC
    LIMIT {top_k}
    FORMAT JSONEachRow
    """
    return ch_rows(query)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ClickHouse embeddings manager")
    parser.add_argument("action", choices=["backfill", "search"], help="Action to perform")
    parser.add_argument("--query", dest="query", help="Query text for search")
    parser.add_argument("--top", dest="top", type=int, default=10, help="Top K for search")

    args = parser.parse_args()

    if args.action == "backfill":
        backfill()
    elif args.action == "search":
        if not args.query:
            print("--query is required for search")
            sys.exit(1)
        results = search(args.query, args.top)
        for r in results:
            ts = r.get("timestamp")
            score = r.get("score")
            msg = r.get("message", "")[:160].replace("\n", " ")
            print(f"{score:.4f} | {ts} | {msg}")


if __name__ == "__main__":
    main()
