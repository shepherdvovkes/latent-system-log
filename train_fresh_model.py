#!/usr/bin/env python3
"""
Train fresh classifier on M4 MacBook using updated 200K embeddings from ClickHouse.
Optimized for Apple Silicon performance.
"""

import os
import json
import time
from typing import List, Dict, Any
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

CLICKHOUSE_URL = "http://localhost:8123"
DATABASE = "system_logs"
EMB_TABLE = "embeddings"
LIMIT = 200000  # Use all 200K embeddings
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


def enhanced_label_classification(message: str) -> str:
    """Enhanced heuristic labeling with more patterns."""
    m = message.lower()
    
    # Error patterns (more comprehensive)
    error_patterns = [
        "error", "failed", "failure", "panic", "crash", "abort", "fatal", 
        "exception", "assertion failed", "segmentation fault", "timeout",
        "connection refused", "permission denied", "access denied"
    ]
    
    # Security patterns
    security_patterns = [
        "sandbox", "violation", "deny", "denied", "unauthorized", "security",
        "breach", "attack", "malware", "virus", "suspicious", "blocked"
    ]
    
    # Warning patterns  
    warning_patterns = [
        "warning", "warn", "deprecated", "obsolete", "legacy", "fallback",
        "retry", "slow", "degraded", "limited"
    ]
    
    # Hardware patterns
    hardware_patterns = [
        "thermal", "temperature", "battery", "power", "usb", "bluetooth",
        "wifi", "disk", "storage", "memory", "cpu", "gpu", "sensor",
        "camera", "microphone", "speaker", "audio", "video"
    ]
    
    if any(pattern in m for pattern in error_patterns):
        return "error"
    elif any(pattern in m for pattern in security_patterns):
        return "security"  
    elif any(pattern in m for pattern in hardware_patterns):
        return "hardware"
    elif any(pattern in m for pattern in warning_patterns):
        return "warning"
    else:
        return "info"


def fetch_embeddings_and_labels(limit: int):
    """Fetch embeddings and generate enhanced labels."""
    print(f"🔍 Fetching {limit:,} embeddings from ClickHouse...")
    
    q = f"""
    SELECT message, embedding, timestamp
    FROM {DATABASE}.{EMB_TABLE}
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    rows = ch_rows(q)
    
    X = []
    y = []
    timestamps = []
    
    print(f"📊 Processing {len(rows):,} rows...")
    
    for i, r in enumerate(rows):
        if i % 10000 == 0:
            print(f"   Processed {i:,}/{len(rows):,} rows...")
            
        emb = r.get("embedding")
        msg = r.get("message", "")
        ts = r.get("timestamp", "")
        
        if emb and isinstance(emb, list) and len(emb) == 384 and msg:
            X.append(emb)
            y.append(enhanced_label_classification(msg))
            timestamps.append(ts)
    
    print(f"✅ Processed {len(X):,} valid embeddings")
    return np.array(X, dtype=np.float32), np.array(y), timestamps


def main():
    print("🚀 Training Fresh Model on M4 MacBook")
    print("=" * 60)
    start_time = time.time()

    # Fetch fresh data
    X, y, timestamps = fetch_embeddings_and_labels(LIMIT)
    
    # Show label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Label Distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(y) * 100)
        print(f"   {label}: {count:,} ({percentage:.1f}%)")

    # Train/validation split (stratified)
    print(f"\n🔄 Splitting data (80/20 train/val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Validation set: {len(X_val):,} samples")

    # Create optimized pipeline for M4
    print(f"\n🧠 Training model on M4...")
    training_start = time.time()
    
    # Use SAGA solver (optimized for large datasets) with parallel processing
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="saga", 
            multi_class="ovr",  # One-vs-Rest for better performance
            n_jobs=-1,  # Use all M4 cores
            random_state=42,
            C=1.0  # Regularization
        ))
    ])

    # Train the model
    clf.fit(X_train, y_train)
    training_time = time.time() - training_start

    print(f"✅ Training completed in {training_time:.1f} seconds")

    # Evaluate model
    print(f"\n📈 Evaluating model...")
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\n🎯 Model Performance:")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_val, y_pred, digits=4))
    
    print(f"\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save the new model
    timestamp = int(time.time())
    model_path = os.path.join(OUTPUT_DIR, f"emb_classifier_fresh_{timestamp}.joblib")
    joblib.dump(clf, model_path)
    
    # Save metadata
    metadata = {
        "model_path": model_path,
        "training_time": training_time,
        "total_time": time.time() - start_time,
        "accuracy": float(accuracy),
        "samples_trained": len(X_train),
        "samples_validated": len(X_val),
        "label_distribution": {label: int(count) for label, count in zip(unique, counts)},
        "timestamp": timestamp,
        "data_source": f"{DATABASE}.{EMB_TABLE}",
        "embeddings_count": len(X),
        "model_type": "LogisticRegression with StandardScaler",
        "platform": "M4 MacBook"
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f"model_metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    
    print(f"\n🎉 Model Training Complete!")
    print(f"=" * 60)
    print(f"💾 Model saved: {os.path.basename(model_path)}")
    print(f"📄 Metadata saved: {os.path.basename(metadata_path)}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🎯 Final accuracy: {accuracy*100:.2f}%")
    print(f"📊 Trained on: {len(X):,} embeddings")
    print(f"💻 Platform: M4 MacBook Pro")
    
    print(f"\n💡 Next steps:")
    print(f"   - Model is ready for inference")
    print(f"   - Web interface will use this fresh model")
    print(f"   - Better predictions with current system data")


if __name__ == "__main__":
    main()
