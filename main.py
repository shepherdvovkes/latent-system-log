#!/usr/bin/env python3
"""
System Log Analysis with AI-powered Insights
Main application entry point.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import settings
from app.api.routes import api_router
from app.services.log_collector import LogCollectorService
from app.services.latent_space import LatentSpaceService
from app.services.embedding_classifier import EmbeddingClassifierService

# Configure logging
logger.add("logs/app.log", rotation="1 day", retention="30 days", level="INFO")

# Global service instances
log_collector = LogCollectorService()
latent_space_service = LatentSpaceService()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting System Log Analysis with AI Assistant...")
    
    # Initialize core services
    await log_collector.initialize()
    await latent_space_service.initialize()
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await log_collector.cleanup()
    await latent_space_service.cleanup()

app = FastAPI(
    title="System Log Analysis with AI",
    description="AI-powered system log monitoring and analysis",
    version="2.0.0",
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

@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse("web_interface.html")

@app.get("/web_interface.html")
async def web_interface():
    """Serve the web interface."""
    return FileResponse("web_interface.html")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )