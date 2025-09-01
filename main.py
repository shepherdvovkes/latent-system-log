#!/usr/bin/env python3
"""
Main application entry point for the System Log Analysis and AI Assistant.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import schedule
import time
import threading

from app.core.config import settings
from app.api.routes import api_router
from app.services.scheduler import SchedulerService
from app.services.log_collector import LogCollectorService
from app.services.latent_space import LatentSpaceService
from app.services.model_trainer import ModelTrainerService
from app.services.model_manager import ModelManagerService
from app.services.minute_log_capture import MinuteLogCaptureService
from app.services.data_exporter import DataExporterService

# Configure logging
logger.add("logs/app.log", rotation="1 day", retention="30 days", level="INFO")

# Global service instances
scheduler_service = SchedulerService()
log_collector = LogCollectorService()
latent_space_service = LatentSpaceService()
model_trainer = ModelTrainerService()
model_manager = ModelManagerService()
minute_log_capture = MinuteLogCaptureService()
data_exporter = DataExporterService()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting System Log Analysis and AI Assistant...")
    
    # Initialize services
    await log_collector.initialize()
    await latent_space_service.initialize()
    await model_trainer.initialize()
    await model_manager.initialize()
    await minute_log_capture.initialize()
    await data_exporter.initialize()
    
    # Start scheduler and minute log capture
    scheduler_service.start()
    minute_log_capture.start()
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down System Log Analysis and AI Assistant...")
    scheduler_service.stop()
    minute_log_capture.stop()
    await log_collector.cleanup()
    await latent_space_service.cleanup()
    await model_trainer.cleanup()
    await model_manager.cleanup()
    await minute_log_capture.cleanup()
    await data_exporter.cleanup()

app = FastAPI(
    title="System Log Analysis and AI Assistant",
    description="A comprehensive system for log analysis and AI-powered system monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Serve static files (web interface)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    """Serve the web interface."""
    return FileResponse("web_interface.html")

@app.get("/web_interface.html")
async def web_interface():
    """Serve the web interface."""
    return FileResponse("web_interface.html")

def run_scheduler():
    """Run the scheduler in a separate thread."""
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # Start scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
