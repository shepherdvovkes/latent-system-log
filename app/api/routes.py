"""
Main API routes for System Intelligence Dashboard.
"""

from datetime import datetime
from typing import List, Dict, Any
import requests
import json
from fastapi import APIRouter, HTTPException, Body
from sentence_transformers import SentenceTransformer

from app.models.schemas import LogEntry, SystemHealth
from app.core.config import settings
from app.services.embedding_classifier import EmbeddingClassifierService

# Create router
api_router = APIRouter()

# Global service instances (injected from main.py)
log_collector = None
latent_space_service = None

def get_services():
    """Get service instances from main.py"""
    from main import log_collector, latent_space_service
    return log_collector, latent_space_service

# ClickHouse connection
CLICKHOUSE_HTTP = "http://localhost:8123"
_search_embedder = None

def _get_embedder():
    global _search_embedder
    if _search_embedder is None:
        _search_embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _search_embedder


@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        import psutil
        
        # Get system metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        uptime = psutil.boot_time()
        current_time = datetime.now().timestamp()
        uptime_seconds = current_time - uptime
        
        # Check service status
        log_collector, latent_space_service = get_services()
        
        services = {
            "log_collector": "healthy" if log_collector else "unhealthy",
            "latent_space": "healthy" if latent_space_service and latent_space_service.is_initialized else "unhealthy"
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": services,
            "uptime": uptime_seconds,
            "memory_usage": memory.percent,
            "cpu_usage": cpu_percent
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@api_router.get("/insights/summary")
async def insights_summary(hours: int = 24, limit: int = 50):
    """Summarize errors, security, and hardware-related issues from ClickHouse."""
    try:
        base = "system_logs.raw_logs"
        
        # Heuristic filters for different issue types
        error_filter = "lower(message) ILIKE '%error%' OR lower(message) ILIKE '%failed%' OR lower(message) ILIKE '%panic%' OR lower(message) ILIKE '%crash%'"
        security_filter = "lower(message) ILIKE '%sandbox%' OR lower(message) ILIKE '%violation%' OR lower(message) ILIKE '%deny%' OR lower(message) ILIKE '%unauthoriz%'"
        hardware_filter = "lower(message) ILIKE '%bluetooth%' OR lower(message) ILIKE '%wifi%' OR lower(message) ILIKE '%usb%' OR lower(message) ILIKE '%disk%' OR lower(message) ILIKE '%battery%' OR lower(message) ILIKE '%thermal%' OR lower(message) ILIKE '%sensor%' OR lower(message) ILIKE '%camera%' OR lower(message) ILIKE '%audio%' OR lower(message) ILIKE '%microphone%' OR lower(message) ILIKE '%thunderbolt%'"

        def ch(query: str) -> str:
            resp = requests.post(CLICKHOUSE_HTTP + "/", params={"query": query}, headers={"Content-Type": "text/plain"}, timeout=60)
            resp.raise_for_status()
            return resp.text

        def rows(query: str):
            if "FORMAT" not in query.upper():
                query += " FORMAT JSONEachRow"
            txt = ch(query)
            return [json.loads(line) for line in txt.strip().splitlines() if line.strip()]

        summary = {}
        
        # Get counts
        total = rows(f"SELECT count() AS c FROM {base}")
        err = rows(f"SELECT count() AS c FROM {base} WHERE {error_filter}")
        sec = rows(f"SELECT count() AS c FROM {base} WHERE {security_filter}")
        hw = rows(f"SELECT count() AS c FROM {base} WHERE {hardware_filter}")
        
        summary["total_logs"] = int(total[0]["c"]) if total else 0
        summary["errors"] = int(err[0]["c"]) if err else 0
        summary["security_issues"] = int(sec[0]["c"]) if sec else 0
        summary["hardware_issues"] = int(hw[0]["c"]) if hw else 0

        # Get sample messages
        top_err = rows(f"SELECT message FROM {base} WHERE {error_filter} LIMIT {limit}")
        top_sec = rows(f"SELECT message FROM {base} WHERE {security_filter} LIMIT {limit}")
        top_hw = rows(f"SELECT message FROM {base} WHERE {hardware_filter} LIMIT {limit}")
        
        summary["sample_errors"] = [r.get("message", "") for r in top_err]
        summary["sample_security"] = [r.get("message", "") for r in top_sec]
        summary["sample_hardware"] = [r.get("message", "") for r in top_hw]

        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights summary failed: {str(e)}")


@api_router.post("/insights/search")
async def insights_search(payload: dict = Body(...)):
    """Semantic search over ClickHouse embeddings table using cosine similarity."""
    try:
        query = payload.get("query")
        top = int(payload.get("top", 20))
        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="'query' string is required")

        embedder = _get_embedder()
        qv = embedder.encode([query], show_progress_bar=False, convert_to_numpy=True).astype("float32")[0]
        
        # L2 normalize
        import numpy as np
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        coeffs = ",".join(str(float(x)) for x in qv.tolist())

        ch_query = f"""
        SELECT message, timestamp, host,
               arraySum(arrayMap((a,b)->a*b, embedding, [{coeffs}])) AS score
        FROM system_logs.embeddings
        ORDER BY score DESC
        LIMIT {top}
        FORMAT JSONEachRow
        """
        resp = requests.post(CLICKHOUSE_HTTP + "/", params={"query": ch_query}, headers={"Content-Type": "text/plain"}, timeout=120)
        resp.raise_for_status()
        lines = [json.loads(line) for line in resp.text.strip().splitlines() if line.strip()]
        return {"results": lines, "count": len(lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@api_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "System Intelligence",
        "version": "2.0.0",
        "description": "AI-powered system log analysis",
        "status": "operational"
    }