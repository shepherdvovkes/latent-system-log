#!/usr/bin/env python3
"""
Backfill predictions into ClickHouse from precomputed embeddings.
- Fetch first N messages from system_logs.embeddings
- Predict labels/probabilities using EmbeddingClassifierService
- Insert into system_logs.predictions
"""
import os
import json
import time
from typing import List, Dict, Any
import requests
from loguru import logger

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from app.services.embedding_classifier import EmbeddingClassifierService

CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
DATABASE = os.environ.get("CLICKHOUSE_DB", "system_logs")
EMB_TABLE = os.environ.get("CLICKHOUSE_EMB_TABLE", "embeddings")
PRED_TABLE = os.environ.get("CLICKHOUSE_PRED_TABLE", "predictions")
LIMIT = int(os.environ.get("PRED_BACKFILL_LIMIT", "10000"))


def ch(query: str) -> str:
    r = requests.post(
        f"{CLICKHOUSE_URL}/",
        params={"query": query},
        headers={"Content-Type": "text/plain"},
        timeout=120,
    )
    r.raise_for_status()
    return r.text


def ch_rows(query: str) -> List[Dict[str, Any]]:
    if "FORMAT" not in query.upper():
        query += " FORMAT JSONEachRow"
    txt = ch(query)
    return [json.loads(line) for line in txt.strip().splitlines() if line.strip()]


def ensure_predictions_table():
    q = f"""
    CREATE TABLE IF NOT EXISTS {DATABASE}.{PRED_TABLE} (
      message String,
      label String,
      proba_error Float32,
      proba_warning Float32,
      proba_info Float32
    ) ENGINE = MergeTree() ORDER BY (label)
    """
    ch(q)


def insert_predictions(rows: List[Dict[str, Any]]):
    data = "\n".join(json.dumps(r) for r in rows)
    q = f"INSERT INTO {DATABASE}.{PRED_TABLE} (message, label, proba_error, proba_warning, proba_info) FORMAT JSONEachRow"
    r = requests.post(
        f"{CLICKHOUSE_URL}/",
        params={"query": q},
        data=data.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
        timeout=300,
    )
    r.raise_for_status()


def main():
    logger.info(f"Backfilling predictions for first {LIMIT} messages from {DATABASE}.{EMB_TABLE}")

    ensure_predictions_table()

    # Fetch messages
    rows = ch_rows(f"SELECT message FROM {DATABASE}.{EMB_TABLE} LIMIT {LIMIT}")
    texts = [r.get("message", "") for r in rows]
    texts = [t for t in texts if t]
    logger.info(f"Fetched {len(texts)} messages")

    # Predict
    svc = EmbeddingClassifierService()
    preds = svc.predict(texts)
    if not preds:
        logger.error("No predictions produced; ensure model exists in training_output")
        return

    payload = []
    for text, pr in zip(texts, preds):
        payload.append({
            "message": text,
            "label": pr.get("label", "info"),
            "proba_error": float(pr.get("proba_error", 0.0)),
            "proba_warning": float(pr.get("proba_warning", 0.0)),
            "proba_info": float(pr.get("proba_info", 0.0)),
        })

    insert_predictions(payload)
    logger.info(f"Inserted {len(payload)} predictions into {DATABASE}.{PRED_TABLE}")


if __name__ == "__main__":
    main()
