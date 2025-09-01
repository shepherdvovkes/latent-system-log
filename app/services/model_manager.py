"""
Model manager service for handling model selection, downloading, and training with latent space integration.
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
from pathlib import Path

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.database_models import LatentSpaceDataDB
from app.models.schemas import (
    AvailableModel, 
    ModelTrainingConfig, 
    ModelTrainingStatus,
    LatentSpaceTrainingData
)
from app.services.latent_space import LatentSpaceService
from app.services.model_trainer import ModelTrainerService


class ModelManagerService:
    """Service for managing model selection, downloading, and training."""
    
    def __init__(self):
        self.available_models = self._get_available_models()
        self.current_model_id = None
        self.model_trainer = ModelTrainerService()
        self.latent_space_service = LatentSpaceService()
        self.training_status = {}
        
    def _get_available_models(self) -> List[AvailableModel]:
        """Get list of available models for training."""
        return [
            AvailableModel(
                model_id="distilbert-qa",
                name="DistilBERT Question Answering",
                description="Lightweight BERT model optimized for question answering tasks",
                model_type="question_answering",
                base_model="distilbert-base-cased",
                size_mb=260.0,
                accuracy=0.85,
                download_url="https://huggingface.co/distilbert-base-cased",
                is_downloaded=self._is_model_downloaded("distilbert-base-cased"),
                is_compatible=True
            ),
            AvailableModel(
                model_id="roberta-qa",
                name="RoBERTa Question Answering",
                description="Robustly optimized BERT model for question answering",
                model_type="question_answering",
                base_model="deepset/roberta-base-squad2",
                size_mb=500.0,
                accuracy=0.88,
                download_url="https://huggingface.co/deepset/roberta-base-squad2",
                is_downloaded=self._is_model_downloaded("deepset/roberta-base-squad2"),
                is_compatible=True
            ),
            AvailableModel(
                model_id="bert-tiny-qa",
                name="BERT Tiny Question Answering",
                description="Ultra-lightweight BERT model for fast inference",
                model_type="question_answering",
                base_model="prajjwal1/bert-tiny",
                size_mb=45.0,
                accuracy=0.75,
                download_url="https://huggingface.co/prajjwal1/bert-tiny",
                is_downloaded=self._is_model_downloaded("prajjwal1/bert-tiny"),
                is_compatible=True
            ),
            AvailableModel(
                model_id="custom-qa",
                name="Custom System Log QA",
                description="Custom model trained on system log data",
                model_type="question_answering",
                base_model="custom",
                size_mb=0.0,
                accuracy=None,
                download_url=None,
                is_downloaded=os.path.exists(os.path.join(settings.MODELS_DIR, "system_qa_model")),
                is_compatible=True
            )
        ]
    
    def _is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        model_path = os.path.join(settings.MODELS_DIR, model_name.replace("/", "_"))
        return os.path.exists(model_path)
    
    async def initialize(self):
        """Initialize the model manager service."""
        logger.info("Initializing ModelManagerService...")
        
        try:
            # Initialize sub-services
            await self.model_trainer.initialize()
            await self.latent_space_service.initialize()
            
            # Update model download status
            self._update_model_status()
            
            logger.info("ModelManagerService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ModelManagerService: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up ModelManagerService...")
        await self.model_trainer.cleanup()
        await self.latent_space_service.cleanup()
    
    def _update_model_status(self):
        """Update the download status of available models."""
        for model in self.available_models:
            model.is_downloaded = self._is_model_downloaded(model.base_model)
    
    async def get_available_models(self) -> List[AvailableModel]:
        """Get list of available models."""
        self._update_model_status()
        return self.available_models
    
    async def download_model(self, model_id: str) -> Dict[str, Any]:
        """Download a model from Hugging Face."""
        logger.info(f"Starting download for model: {model_id}")
        
        try:
            # Find the model
            model = next((m for m in self.available_models if m.model_id == model_id), None)
            if not model:
                return {"success": False, "message": f"Model {model_id} not found"}
            
            if model.is_downloaded:
                return {"success": True, "message": f"Model {model.name} is already downloaded"}
            
            # Download the model
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            
            model_path = os.path.join(settings.MODELS_DIR, model.base_model.replace("/", "_"))
            os.makedirs(model_path, exist_ok=True)
            
            logger.info(f"Downloading {model.name} to {model_path}")
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model.base_model)
            model_qa = AutoModelForQuestionAnswering.from_pretrained(model.base_model)
            
            # Save locally
            tokenizer.save_pretrained(model_path)
            model_qa.save_pretrained(model_path)
            
            # Update status
            model.is_downloaded = True
            
            logger.info(f"Successfully downloaded {model.name}")
            return {"success": True, "message": f"Successfully downloaded {model.name}"}
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            return {"success": False, "message": f"Failed to download model: {str(e)}"}
    
    async def get_latent_space_training_data(self) -> LatentSpaceTrainingData:
        """Get latent space data for training."""
        try:
            async with AsyncSessionLocal() as session:
                # Get the latest latent space data
                from sqlalchemy import select
                stmt = select(LatentSpaceDataDB).order_by(LatentSpaceDataDB.id.desc()).limit(1)
                result = await session.execute(stmt)
                latest_data = result.scalar_one_or_none()
                
                if not latest_data:
                    return LatentSpaceTrainingData(
                        total_embeddings=0,
                        total_texts=0,
                        sources_distribution={},
                        levels_distribution={},
                        average_embedding_length=0.0,
                        last_updated=datetime.now(),
                        is_available=False
                    )
                
                # Parse metadata
                metadata = latest_data.algorithm_metadata
                
                return LatentSpaceTrainingData(
                    total_embeddings=metadata.get('total_embeddings', 0),
                    total_texts=metadata.get('total_embeddings', 0),  # Same as embeddings
                    sources_distribution=metadata.get('sources_distribution', {}),
                    levels_distribution=metadata.get('levels_distribution', {}),
                    average_embedding_length=metadata.get('average_embedding_length', 0.0),
                    last_updated=datetime.fromisoformat(metadata.get('last_updated', datetime.now().isoformat())),
                    is_available=True
                )
                
        except Exception as e:
            logger.error(f"Error getting latent space training data: {e}")
            return LatentSpaceTrainingData(
                total_embeddings=0,
                total_texts=0,
                sources_distribution={},
                levels_distribution={},
                average_embedding_length=0.0,
                last_updated=datetime.now(),
                is_available=False
            )
    
    async def start_training_with_latent_space(
        self, 
        model_id: str, 
        config: ModelTrainingConfig
    ) -> Dict[str, Any]:
        """Start training a model using latent space data."""
        logger.info(f"Starting training for model {model_id} with latent space integration")
        
        try:
            # Generate training ID
            training_id = f"train_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize training status
            self.training_status[training_id] = ModelTrainingStatus(
                training_id=training_id,
                status="pending",
                progress=0.0,
                total_epochs=config.epochs,
                message="Initializing training..."
            )
            
            # Start training in background
            asyncio.create_task(self._train_model_async(training_id, model_id, config))
            
            return {
                "success": True,
                "training_id": training_id,
                "message": f"Training started for {model_id}"
            }
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return {"success": False, "message": f"Failed to start training: {str(e)}"}
    
    async def _train_model_async(self, training_id: str, model_id: str, config: ModelTrainingConfig):
        """Train model asynchronously."""
        try:
            # Update status to running
            self.training_status[training_id].status = "running"
            self.training_status[training_id].start_time = datetime.now()
            self.training_status[training_id].message = "Loading latent space data..."
            
            # Get latent space data
            latent_data = await self.get_latent_space_training_data()
            if not latent_data.is_available:
                raise Exception("No latent space data available for training")
            
            # Get logs from database
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            await db_service.initialize()
            
            # Get recent logs (last 24 hours)
            logs = await db_service.get_logs_by_time_range(
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                datetime.now()
            )
            
            if not logs:
                raise Exception("No logs available for training")
            
            # Convert to LogEntry format
            from app.models.schemas import LogEntry
            log_entries = [
                LogEntry(
                    timestamp=log.timestamp,
                    source=log.source,
                    level=log.level,
                    message=log.message,
                    metadata=log.log_metadata
                )
                for log in logs
            ]
            
            # Update status
            self.training_status[training_id].message = f"Training with {len(log_entries)} log entries..."
            
            # Train the model
            result = await self.model_trainer.train_model(
                logs=log_entries,
                epochs=config.epochs,
                batch_size=config.batch_size
            )
            
            # Update final status
            self.training_status[training_id].status = "completed" if result["success"] else "failed"
            self.training_status[training_id].progress = 100.0
            self.training_status[training_id].end_time = datetime.now()
            self.training_status[training_id].message = result.get("message", "Training completed")
            
            logger.info(f"Training {training_id} completed with status: {result['success']}")
            
        except Exception as e:
            logger.error(f"Error in training {training_id}: {e}")
            self.training_status[training_id].status = "failed"
            self.training_status[training_id].message = f"Training failed: {str(e)}"
            self.training_status[training_id].end_time = datetime.now()
    
    async def get_training_status(self, training_id: str) -> Optional[ModelTrainingStatus]:
        """Get training status for a specific training job."""
        return self.training_status.get(training_id)
    
    async def get_all_training_status(self) -> List[ModelTrainingStatus]:
        """Get status of all training jobs."""
        return list(self.training_status.values())
    
    async def select_model(self, model_id: str) -> Dict[str, Any]:
        """Select a model for use."""
        try:
            model = next((m for m in self.available_models if m.model_id == model_id), None)
            if not model:
                return {"success": False, "message": f"Model {model_id} not found"}
            
            if not model.is_downloaded:
                return {"success": False, "message": f"Model {model.name} is not downloaded"}
            
            self.current_model_id = model_id
            logger.info(f"Selected model: {model.name}")
            
            return {"success": True, "message": f"Selected model: {model.name}"}
            
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            return {"success": False, "message": f"Failed to select model: {str(e)}"}
    
    async def get_current_model(self) -> Optional[AvailableModel]:
        """Get the currently selected model."""
        if not self.current_model_id:
            return None
        
        return next((m for m in self.available_models if m.model_id == self.current_model_id), None)
