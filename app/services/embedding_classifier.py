"""
Embedding-based classifier service.
Loads latest sklearn pipeline trained on 384-d embeddings and provides text inference.
"""
import glob
import os
from typing import List, Dict, Any
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

class EmbeddingClassifierService:
    def __init__(self, model_dir: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_dir = model_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "training_output"))
        self.embedding_model_name = embedding_model
        self.pipeline = None
        self.embedder = None

    def initialize(self) -> bool:
        try:
            # Load latest joblib model
            candidates = sorted(glob.glob(os.path.join(self.model_dir, "emb_classifier_*.joblib")))
            if not candidates:
                logger.error("No trained embedding classifier found.")
                return False
            latest = candidates[-1]
            self.pipeline = joblib.load(latest)
            logger.info(f"Loaded classifier: {os.path.basename(latest)}")

            # Load embedding model
            self.embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedder: {self.embedding_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingClassifierService: {e}")
            return False

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.pipeline is None or self.embedder is None:
            ok = self.initialize()
            if not ok:
                return []
        # Encode texts
        embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        # Predict probabilities
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(embeddings)
            labels = self.pipeline.classes_
            results = []
            for i in range(len(texts)):
                p = probs[i]
                d = {"label": labels[int(np.argmax(p))]}
                # attach per-class probs
                for cls, v in zip(labels, p):
                    d[f"proba_{cls}"] = float(v)
                results.append(d)
            return results
        else:
            preds = self.pipeline.predict(embeddings)
            return [{"label": str(preds[i])} for i in range(len(texts))]
