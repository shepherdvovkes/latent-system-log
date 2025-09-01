"""
Main API routes for the System Log Analysis and AI Assistant.
"""

from datetime import datetime
from typing import List
import os
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import JSONResponse, FileResponse
import psutil

from app.models.schemas import (
    LogEntry, LogCollectionResponse, LatentSpaceStatus,
    QuestionRequest, QuestionResponse, ModelStatus,
    SystemHealth, TrainingRequest, TrainingResponse,
    AvailableModel, ModelSelectionRequest, ModelTrainingConfig,
    ModelTrainingStatus, LatentSpaceTrainingData
)
from app.services.hf_model_searcher import HFModelSearcher
from app.core.config import settings
from app.services.embedding_classifier import EmbeddingClassifierService
import requests
import json
from sentence_transformers import SentenceTransformer

# Create router
api_router = APIRouter()

# Global service instances (will be injected from main.py)
log_collector = None
latent_space_service = None
model_trainer = None
model_manager = None
scheduler_service = None
minute_log_capture = None


def get_services():
    """Dependency to get service instances."""
    from main import log_collector, latent_space_service, model_trainer, model_manager, scheduler_service, minute_log_capture, data_exporter
    return log_collector, latent_space_service, model_trainer, model_manager, scheduler_service, minute_log_capture, data_exporter


@api_router.get("/health", response_model=SystemHealth)
async def health_check():
    """Health check endpoint."""
    try:
        # Get system metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get uptime
        uptime = psutil.boot_time()
        current_time = datetime.now().timestamp()
        uptime_seconds = current_time - uptime
        
        # Check service status
        log_collector, latent_space_service, model_trainer, model_manager, scheduler, minute_log_capture, data_exporter = get_services()
        
        services = {
            "log_collector": "healthy" if log_collector else "unhealthy",
            "latent_space": "healthy" if latent_space_service and latent_space_service.is_initialized else "unhealthy", 
            "model_trainer": "healthy" if model_trainer else "unhealthy",
            "scheduler": "healthy" if scheduler else "unhealthy",
            "minute_log_capture": "healthy" if minute_log_capture else "unhealthy"
        }
        
        return SystemHealth(
            status="healthy",
            timestamp=datetime.now(),
            services=services,
            uptime=uptime_seconds,
            memory_usage=memory.percent,
            cpu_usage=cpu_percent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@api_router.get("/logs/recent", response_model=List[LogEntry])
async def get_recent_logs(minutes: int = 10):
    """Get recent system logs."""
    try:
        log_collector, _, _, _, _, _ = get_services()
        logs = await log_collector.get_recent_logs(minutes=minutes)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent logs: {str(e)}")


@api_router.post("/logs/collect", response_model=LogCollectionResponse)
async def collect_logs():
    """Manually trigger log collection."""
    try:
        log_collector, _, _, _, _, _ = get_services()
        logs = await log_collector.collect_system_logs()
        
        return LogCollectionResponse(
            success=True,
            message=f"Successfully collected {len(logs)} log entries",
            logs_collected=len(logs),
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log collection failed: {str(e)}")


# New endpoints for minute log capture service
@api_router.get("/logs/minute")
async def get_last_minute_logs():
    """Get logs from the last minute stored in the database."""
    try:
        _, _, _, _, _, minute_log_capture = get_services()
        logs = await minute_log_capture.get_last_minute_logs()
        return {
            "logs": logs,
            "count": len(logs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get last minute logs: {str(e)}")


@api_router.post("/logs/minute/capture")
async def capture_minute_logs():
    """Manually trigger minute log capture and storage."""
    try:
        _, _, _, _, _, minute_log_capture = get_services()
        result = await minute_log_capture.capture_logs_now()
        
        return {
            "success": True,
            "message": f"Captured {result['logs_collected']} logs, stored {result['logs_stored']}, total in DB: {result['total_logs_in_db']}, last interaction: {result['last_interaction_logs']}",
            "logs_collected": result['logs_collected'],
            "logs_stored": result['logs_stored'],
            "total_logs_in_db": result['total_logs_in_db'],
            "last_interaction_logs": result['last_interaction_logs'],
            "timestamp": result['timestamp']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Minute log capture failed: {str(e)}")


@api_router.get("/logs/minute/statistics")
async def get_minute_log_statistics(minutes: int = 1):
    """Get log statistics for the last N minutes."""
    try:
        _, _, _, _, _, minute_log_capture = get_services()
        stats = await minute_log_capture.get_log_statistics(minutes)
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log statistics: {str(e)}")


@api_router.get("/logs/minute/status")
async def get_minute_log_status():
    """Get current minute log capture status and detailed information."""
    try:
        _, _, _, _, _, minute_log_capture = get_services()
        status = await minute_log_capture.get_capture_status()
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get minute log status: {str(e)}")


@api_router.get("/system-logs/parse")
async def parse_system_logs():
    """Parse the lastmin.log file and store in system log database."""
    try:
        from app.services.system_log_parser import SystemLogParser
        parser = SystemLogParser()
        
        result = await parser.parse_log_file(minutes=1)
        stats = await parser.get_log_statistics(minutes=1)
        
        return {
            "success": True,
            "message": f"Parsed {result.get('parsed_entries', 0)} system log entries",
            "parsed_entries": result.get('parsed_entries', 0),
            "stored_entries": result.get('stored_entries', 0),
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse system logs: {str(e)}")


@api_router.get("/system-logs/statistics")
async def get_system_log_statistics(minutes: int = 1):
    """Get statistics from the system log database."""
    try:
        from app.services.system_log_parser import SystemLogParser
        parser = SystemLogParser()
        
        stats = await parser.get_log_statistics(minutes)
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system log statistics: {str(e)}")


@api_router.get("/system-logs/recent")
async def get_recent_system_logs(limit: int = 100):
    """Get recent system log entries from the database."""
    try:
        from app.core.database import AsyncSessionLocal
        from app.models.system_log_models import SystemLogEntry
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as session:
            stmt = select(SystemLogEntry).order_by(SystemLogEntry.timestamp.desc()).limit(limit)
            result = await session.execute(stmt)
            logs = result.scalars().all()
            
            # Convert to dictionary format
            log_list = []
            for log in logs:
                log_dict = {
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat(),
                    'thread_id': log.thread_id,
                    'log_type': log.log_type,
                    'activity_id': log.activity_id,
                    'process_id': log.process_id,
                    'ttl': log.ttl,
                    'source': log.source,
                    'message': log.message,
                    'level': log.level,
                    'metadata': log.log_metadata,
                    'created_at': log.created_at.isoformat() if log.created_at else None
                }
                log_list.append(log_dict)
            
            return {
                "logs": log_list,
                "count": len(log_list),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent system logs: {str(e)}")


@api_router.post("/logs/cleanup")
async def cleanup_old_logs(days: int = 30):
    """Clean up logs older than specified days."""
    try:
        _, _, _, _, _, minute_log_capture = get_services()
        deleted_count = await minute_log_capture.cleanup_old_logs(days)
        
        return {
            "success": True,
            "message": f"Cleaned up {deleted_count} old log entries",
            "deleted_count": deleted_count,
            "retention_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log cleanup failed: {str(e)}")


@api_router.get("/latent-space/status", response_model=LatentSpaceStatus)
async def get_latent_space_status():
    """Get latent space status and statistics."""
    try:
        _, latent_space_service, _, _, _, _ = get_services()
        stats = await latent_space_service.get_latent_space_stats()
        
        return LatentSpaceStatus(
            is_initialized=stats['is_initialized'],
            total_embeddings=stats['total_embeddings'],
            last_updated=datetime.fromisoformat(stats['last_updated']) if stats['last_updated'] else None,
            dimension=stats['embedding_dimension'],
            model_name=stats['model_name'],
            index_size=stats['index_size'],
            memory_usage_mb=stats['memory_usage_mb'],
            average_embedding_length=stats['average_embedding_length'],
            sources_distribution=stats['sources_distribution'],
            levels_distribution=stats['levels_distribution'],
            index_type=stats['index_type'],
            compression_ratio=stats['compression_ratio']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latent space status: {str(e)}")


@api_router.get("/latent-space/algorithm-info")
async def get_latent_space_algorithm_info():
    """Get detailed information about the latent space algorithms and implementation."""
    try:
        _, latent_space_service, _, _, _, _ = get_services()
        
        # Get algorithm details from the service
        algorithm_info = {
            "embedding_model": {
                "name": settings.EMBEDDING_MODEL,
                "type": "Sentence Transformer",
                "architecture": "MiniLM (Lightweight BERT)",
                "dimension": settings.LATENT_SPACE_DIMENSION,
                "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
                "description": "Uses a lightweight BERT-based model for generating semantic embeddings",
                "advantages": [
                    "Fast inference speed",
                    "Small model size (~80MB)",
                    "Good semantic understanding",
                    "Optimized for sentence-level tasks"
                ],
                "technical_details": {
                    "base_model": "BERT",
                    "distillation": "Knowledge distillation from larger BERT models",
                    "vocabulary_size": "~30,000 tokens",
                    "attention_heads": "12",
                    "layers": "6 (reduced from 12 in standard BERT)"
                }
            },
            "similarity_algorithm": {
                "type": "Cosine Similarity",
                "implementation": "FAISS IndexFlatIP",
                "normalization": "L2 Normalization",
                "threshold": settings.SIMILARITY_THRESHOLD,
                "description": "Uses inner product with L2 normalization for cosine similarity calculation",
                "mathematical_formula": "cos(θ) = (A·B) / (||A|| × ||B||)",
                "advantages": [
                    "Scale and rotation invariant",
                    "Bounded between -1 and 1",
                    "Intuitive interpretation",
                    "Fast computation with FAISS"
                ]
            },
            "indexing_algorithm": {
                "type": "FAISS Flat Index",
                "algorithm": "IndexFlatIP (Inner Product)",
                "batch_size": 100,
                "description": "Simple but effective flat index for exact similarity search",
                "search_complexity": "O(n) where n is number of vectors",
                "memory_usage": "O(n × d) where d is dimension",
                "advantages": [
                    "Exact search results",
                    "No approximation errors",
                    "Simple implementation",
                    "Good for small to medium datasets"
                ],
                "limitations": [
                    "Linear search time",
                    "High memory usage for large datasets",
                    "Not suitable for very large-scale applications"
                ]
            },
            "text_preprocessing": {
                "format": "Structured concatenation",
                "fields": ["Source", "Level", "Message", "Metadata"],
                "separator": " | ",
                "description": "Log entries are converted to structured text format for embedding",
                "example": "Source: system.log | Level: ERROR | Message: Connection timeout | user_id: 12345",
                "advantages": [
                    "Preserves structured information",
                    "Maintains context",
                    "Consistent format",
                    "Easy to parse and understand"
                ]
            },
            "performance_characteristics": {
                "embedding_generation": "~1000 logs/second",
                "similarity_search": "~10,000 queries/second",
                "memory_efficiency": "~1.5MB per 1000 embeddings",
                "storage_format": "Database (SQLite with BLOB storage)",
                "persistence": "Survives application restarts"
            }
        }
        
        return algorithm_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm info: {str(e)}")


@api_router.post("/latent-space/rebuild")
async def rebuild_latent_space():
    """Rebuild the latent space with all available logs."""
    try:
        log_collector, latent_space_service, _, _, _, _ = get_services()
        
        # Get all available logs
        logs = await log_collector.get_recent_logs(minutes=1440)  # Last 24 hours
        
        # If no logs available, collect some fresh logs
        if not logs:
            logs = await log_collector.collect_system_logs()
        
        if not logs:
            raise HTTPException(status_code=400, detail="No logs available for latent space rebuild")
        
        # Rebuild latent space
        await latent_space_service.rebuild_latent_space(logs)
        
        return {"success": True, "message": f"Latent space rebuilt with {len(logs)} log entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild latent space: {str(e)}")


@api_router.post("/qa/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about system state."""
    try:
        _, latent_space_service, model_trainer, _, _, _ = get_services()
        
        # First try to get answer from trained model
        model_answer = await model_trainer.answer_question(request.question, request.context)
        
        # Also search latent space for similar logs
        similar_logs = await latent_space_service.search_similar_logs(request.question, top_k=5)
        
        # Combine sources
        sources = model_answer.get("sources", [])
        for log_text, similarity in similar_logs:
            if similarity > 0.7:  # Only include highly similar logs
                sources.append(f"Similar log (confidence: {similarity:.2f}): {log_text[:100]}...")
        
        return QuestionResponse(
            answer=model_answer.get("answer", "No answer available"),
            confidence=model_answer.get("confidence", 0.0),
            sources=sources,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


@api_router.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get model training status and information."""
    try:
        _, _, model_trainer, _, _, _ = get_services()
        status = await model_trainer.get_model_status()
        
        return ModelStatus(
            is_trained=status['is_trained'],
            last_trained=datetime.fromisoformat(status['last_trained']) if status['last_trained'] else None,
            accuracy=status.get('accuracy'),
            total_samples=len(status.get('training_history', [])),
            model_path=status.get('model_path')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@api_router.post("/model/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Trigger model training."""
    try:
        log_collector, _, model_trainer, _, _, _ = get_services()
        
        # Get logs for training
        logs = await log_collector.get_recent_logs(minutes=1440)  # Last 24 hours
        
        if not logs and not request.force_retrain:
            raise HTTPException(status_code=400, detail="No logs available for training")
        
        # Train model
        result = await model_trainer.train_model(
            logs, 
            epochs=request.epochs, 
            batch_size=request.batch_size
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("message", "Training failed"))
        
        training_info = result.get("training_info", {})
        
        return TrainingResponse(
            success=True,
            message="Model training completed successfully",
            training_id=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            estimated_duration=training_info.get('epochs', 10) * 60  # Rough estimate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@api_router.get("/scheduler/tasks")
async def get_scheduled_tasks():
    """Get information about scheduled tasks."""
    try:
        _, _, _, scheduler_service, _, _ = get_services()
        tasks = scheduler_service.get_scheduled_tasks()
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled tasks: {str(e)}")


@api_router.get("/system/info")
async def get_system_info():
    """Get detailed system information."""
    try:
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network information
        network = psutil.net_io_counters()
        
        return {
            "cpu": {
                "count": cpu_count,
                "frequency": {
                    "current": cpu_freq.current if cpu_freq else None,
                    "min": cpu_freq.min if cpu_freq else None,
                    "max": cpu_freq.max if cpu_freq else None
                },
                "usage_percent": psutil.cpu_percent(interval=1)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@api_router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "System Log Analysis and AI Assistant",
        "version": "1.0.0",
        "description": "A comprehensive system for log analysis and AI-powered system monitoring",
        "endpoints": {
            "health": "/api/v1/health",
            "logs": "/api/v1/logs/recent",
            "minute_logs": "/api/v1/logs/minute",
            "latent_space": "/api/v1/latent-space/status",
            "qa": "/api/v1/qa/ask",
            "model": "/api/v1/model/status",
            "system": "/api/v1/system/info",
            "models": "/api/v1/models/search"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


# Model search and download endpoints
@api_router.post("/models/search")
async def search_models(request: dict):
    """Search for models on Hugging Face."""
    try:
        searcher = HFModelSearcher()
        result = searcher.search_models(
            query=request.get("query", "system log analysis"),
            task=request.get("task", "question-answering"),
            limit=request.get("limit", 20)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model search failed: {str(e)}")


@api_router.get("/models/recommended")
async def get_recommended_models():
    """Get recommended models for system log analysis."""
    try:
        searcher = HFModelSearcher()
        result = searcher.get_recommended_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommended models: {str(e)}")


@api_router.post("/models/download")
async def download_model(request: dict):
    """Download a model from Hugging Face."""
    try:
        model_id = request.get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        # For now, we'll just validate the model exists
        # In a full implementation, you'd download and cache the model
        searcher = HFModelSearcher()
        model_info = searcher.get_model_info(model_id)
        
        if not model_info["success"]:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Update the model trainer to use the new model
        _, _, model_trainer, _, _, _ = get_services()
        if model_trainer:
            # Set the new model ID for future training
            model_trainer.current_model_id = model_id
        
        return {
            "success": True,
            "message": f"Model {model_id} selected successfully",
            "model_info": model_info["model"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model download failed: {str(e)}")


@api_router.get("/models/compare")
async def compare_models(model_ids: str):
    """Compare multiple models."""
    try:
        model_id_list = model_ids.split(",")
        searcher = HFModelSearcher()
        result = searcher.compare_models(model_id_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")


# Data export endpoints for Google Colab
@api_router.post("/export/training-data")
async def export_training_data(hours: int = 24):
    """Export training data for Google Colab."""
    try:
        _, _, _, _, _, _, data_exporter = get_services()
        result = await data_exporter.export_training_data(hours=hours)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "success": True,
            "message": result["message"],
            "export_info": {
                "zip_file": os.path.basename(result["export_files"]["zip_file"]),
                "notebook_file": os.path.basename(result["colab_notebook"]),
                "training_stats": result["training_data_stats"],
                "download_path": result["export_files"]["zip_file"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@api_router.get("/export/history")
async def get_export_history():
    """Get history of data exports."""
    try:
        _, _, _, _, _, _, data_exporter = get_services()
        exports = await data_exporter.get_export_history()
        
        return {
            "exports": exports,
            "total_exports": len(exports)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get export history: {str(e)}")


@api_router.get("/export/download/{filename}")
async def download_export(filename: str):
    """Download an exported file."""
    try:
        _, _, _, _, _, _, data_exporter = get_services()
        file_path = os.path.join(data_exporter.export_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@api_router.post("/inference/embedding-classifier")
async def embedding_classifier_infer(payload: dict = Body(...)):
    """Run inference using the embedding-based classifier on provided texts."""
    try:
        texts = payload.get("texts")
        if not texts or not isinstance(texts, list):
            raise HTTPException(status_code=400, detail="'texts' list is required")
        svc = EmbeddingClassifierService()
        results = svc.predict(texts)
        if not results:
            raise HTTPException(status_code=500, detail="Classifier not available")
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@api_router.post("/inference/backfill-clickhouse")
async def backfill_clickhouse_predictions(limit: int = 10000):
    """Compute predictions for first N embeddings and store them back in ClickHouse."""
    try:
        # Fetch messages from our embeddings table via HTTP ClickHouse
        base_url = "http://localhost:8123"
        query = f"SELECT message FROM system_logs.embeddings LIMIT {limit} FORMAT JSONEachRow"
        r = requests.post(f"{base_url}/", params={"query": query}, headers={"Content-Type": "text/plain"}, timeout=120)
        r.raise_for_status()
        rows = [json.loads(line) for line in r.text.strip().splitlines() if line.strip()]
        texts = [row.get("message", "") for row in rows]
        if not texts:
            return {"updated": 0}

        # Predict
        svc = EmbeddingClassifierService()
        results = svc.predict(texts)

        # Create predictions table if not exists
        create_q = """
        CREATE TABLE IF NOT EXISTS system_logs.predictions (
          message String,
          label String,
          proba_error Float32,
          proba_warning Float32,
          proba_info Float32
        ) ENGINE = MergeTree() ORDER BY (label)
        """
        requests.post(f"{base_url}/", params={"query": create_q}, headers={"Content-Type": "text/plain"}, timeout=60).raise_for_status()

        # Insert
        lines = []
        for text, res in zip(texts, results):
            lines.append(json.dumps({
                "message": text,
                "label": res.get("label", "info"),
                "proba_error": float(res.get("proba_error", 0.0)),
                "proba_warning": float(res.get("proba_warning", 0.0)),
                "proba_info": float(res.get("proba_info", 0.0)),
            }))
        ins_q = "INSERT INTO system_logs.predictions (message, label, proba_error, proba_warning, proba_info) FORMAT JSONEachRow"
        requests.post(f"{base_url}/", params={"query": ins_q}, data="\n".join(lines).encode("utf-8"), headers={"Content-Type": "application/x-ndjson"}, timeout=300).raise_for_status()

        return {"updated": len(lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backfill failed: {str(e)}")

CLICKHOUSE_HTTP = "http://localhost:8123"

_search_embedder = None

def _get_embedder():
    global _search_embedder
    if _search_embedder is None:
        _search_embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _search_embedder


@api_router.get("/insights/summary")
async def insights_summary(hours: int = 24, limit: int = 50):
    """Summarize errors, security, and hardware-related issues from ClickHouse."""
    try:
        # Since timestamps are stored as strings, use substring filters (fallback)
        # Basic counts
        base = "system_logs.raw_logs"
        # Heuristic filters
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
        # Totals
        total = rows(f"SELECT count() AS c FROM {base}")
        summary["total_logs"] = int(total[0]["c"]) if total else 0

        # Error/security/hardware counts
        err = rows(f"SELECT count() AS c FROM {base} WHERE {error_filter}")
        sec = rows(f"SELECT count() AS c FROM {base} WHERE {security_filter}")
        hw = rows(f"SELECT count() AS c FROM {base} WHERE {hardware_filter}")
        summary["errors"] = int(err[0]["c"]) if err else 0
        summary["security_issues"] = int(sec[0]["c"]) if sec else 0
        summary["hardware_issues"] = int(hw[0]["c"]) if hw else 0

        # Top messages per category
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
