#!/usr/bin/env python3
"""
Train a lightweight classifier using precomputed 384-d embeddings stored in ClickHouse.
- Fetch embeddings + messages
- Heuristic labels: error / warning / info
- Train multinomial logistic regression (scikit-learn)
- Report metrics and save model to training_output/
"""

import os
import json
import time
from typing import List, Dict, Any
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
DATABASE = os.environ.get("CLICKHOUSE_DB", "system_logs")
EMB_TABLE = os.environ.get("CLICKHOUSE_EMB_TABLE", "embeddings")
LIMIT = int(os.environ.get("EMB_TRAIN_LIMIT", "100000"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "training_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ch_rows(query: str) -> List[Dict[str, Any]]:
    if "FORMAT" not in query.upper():
        query += " FORMAT JSONEachRow"
    r = requests.post(
        f"{CLICKHOUSE_URL}/",
        params={"query": query},
        headers={"Content-Type": "text/plain"},
        timeout=300,
    )
    r.raise_for_status()
    return [json.loads(line) for line in r.text.strip().splitlines() if line.strip()]


def build_label(message: str) -> str:
    m = message.lower()
    if any(w in m for w in ["error", "failed", "violation"]):
        return "error"
    if any(w in m for w in ["warning", "warn", "deprecated"]):
        return "warning"
    return "info"


def fetch_embeddings(limit: int):
    q = f"""
    SELECT message, embedding
    FROM {DATABASE}.{EMB_TABLE}
    LIMIT {limit}
    """
    rows = ch_rows(q)
    X = []
    y = []
    for r in rows:
        emb = r.get("embedding")
        msg = r.get("message", "")
        if emb and isinstance(emb, list) and len(emb) == 384:
            X.append(emb)
            y.append(build_label(msg))
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    print("🚀 Training classifier on precomputed embeddings")
    print("=" * 60)
    start = time.time()

    # Fetch data
    X, y = fetch_embeddings(LIMIT)
    print(f"Loaded {len(X):,} embeddings")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: Standardize + LogisticRegression (multinomial)
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # data already near unit norm; keep sparse-friendly
        ("lr", LogisticRegression(max_iter=1000, solver="saga", multi_class="multinomial", n_jobs=-1))
    ])

    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_val)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=3))

    # Save model
    out_path = os.path.join(OUTPUT_DIR, f"emb_classifier_{int(time.time())}.joblib")
    joblib.dump(clf, out_path)
    elapsed = time.time() - start
    print(f"\n✅ Saved model to: {out_path}")
    print(f"⏱️  Total time: {elapsed:.1f} sec")


if __name__ == "__main__":
    main()
