"""
Model trainer service for building AI models to answer questions about system states.
"""

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.m4_optimization import M4Optimizer, setup_m4_environment
from app.models.schemas import LogEntry
from app.services.minimal_trainer import MinimalModelTrainer


def get_optimal_device():
    """Get the optimal device for training on Apple Silicon."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for GPU acceleration")
        return torch.device("cuda")
    else:
        logger.info("Using CPU for training")
        return torch.device("cpu")


def get_safe_device_for_training():
    """Get a safe device for training, preferring CPU if MPS has memory issues."""
    if torch.backends.mps.is_available():
        # For training, prefer CPU to avoid MPS memory issues
        logger.info("Using CPU for training to avoid MPS memory issues")
        return torch.device("cpu")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for GPU acceleration")
        return torch.device("cuda")
    else:
        logger.info("Using CPU for training")
        return torch.device("cpu")


class SystemQADataset(Dataset):
    """Custom dataset for system question answering."""
    
    def __init__(self, questions, contexts, answers, tokenizer, max_length=512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Find answer span in context
        answer_start, answer_end = self._find_answer_span(context, answer)
        
        # Convert to token positions
        token_start, token_end = self._char_to_token_positions(
            encoding, answer_start, answer_end
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(token_start, dtype=torch.long),
            'end_positions': torch.tensor(token_end, dtype=torch.long)
        }
    
    def _find_answer_span(self, context: str, answer: str) -> Tuple[int, int]:
        """Find the start and end character positions of the answer in context."""
        start = context.find(answer)
        if start == -1:
            return 0, 0
        return start, start + len(answer)
    
    def _char_to_token_positions(self, encoding, char_start: int, char_end: int) -> Tuple[int, int]:
        """Convert character positions to token positions."""
        try:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end - 1)
            
            if token_start is None:
                token_start = 0
            if token_end is None:
                token_end = len(encoding['input_ids'][0]) - 1
                
            return token_start, token_end
        except:
            return 0, 0


class ModelTrainerService:
    """Service for training and managing AI models for system question answering."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = get_optimal_device()
        self.current_model_id = "distilbert-base-cased-distilled-squad"  # Default model
        
        # Setup M4 optimizations
        setup_m4_environment()
        
        logger.info(f"Model trainer initialized with device: {self.device}")
        self.is_trained = False
        self.last_trained = None
        self.training_history = []
        
        # File paths
        self.model_path = os.path.join(settings.MODELS_DIR, "system_qa_model")
        self.tokenizer_path = os.path.join(settings.MODELS_DIR, "system_qa_tokenizer")
        self.training_data_path = os.path.join(settings.DATA_DIR, "training_data.json")
        self.history_path = os.path.join(settings.DATA_DIR, "training_history.json")
        
    async def initialize(self):
        """Initialize the model trainer service."""
        logger.info("Initializing ModelTrainerService...")
        
        try:
            # Load existing model if available
            await self._load_model()
            
            # Load training history
            await self._load_training_history()
            
            logger.info("ModelTrainerService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ModelTrainerService: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up ModelTrainerService...")
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
    
    async def train_model(self, logs: List[LogEntry], epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """Train the question answering model."""
        logger.info("Starting model training...")
        
        try:
            # Try the optimized training first
            result = await self._train_with_optimization(logs, epochs, batch_size)
            if result["success"]:
                return result
            
            # If optimized training fails, fall back to minimal training
            logger.warning("Optimized training failed, falling back to minimal training")
            return await self._train_with_minimal_trainer(logs, epochs, batch_size)
            
        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "message": f"Training failed: {str(e)}"}
    
    async def _train_with_optimization(self, logs: List[LogEntry], epochs: int, batch_size: int) -> Dict[str, Any]:
        """Train with M4 optimization."""
        try:
            # Generate training data from logs
            training_data = await self._generate_training_data(logs)
            
            if not training_data:
                logger.warning("No training data generated")
                return {"success": False, "message": "No training data available"}
            
            # Initialize model and tokenizer
            await self._initialize_model()
            
            # Use safe device for training to avoid MPS memory issues
            training_device = get_safe_device_for_training()
            if self.model:
                self.model = self.model.to(training_device)
            logger.info(f"Training on device: {training_device}")
            
            # Prepare datasets
            train_dataset, eval_dataset = await self._prepare_datasets(training_data)
            
            # Create training arguments with M4 optimization and version compatibility
            # Use much smaller batch size to prevent GPU memory issues
            effective_batch_size = min(batch_size, 4)  # Cap at 4 for M4 memory safety
            
            training_args_dict = {
                "output_dir": "./training_output",
                "num_train_epochs": epochs,
                "per_device_train_batch_size": effective_batch_size,
                "per_device_eval_batch_size": effective_batch_size,
                "logging_dir": "./logs",
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "learning_rate": 2e-5,
                "gradient_accumulation_steps": max(4, batch_size // effective_batch_size),  # Increase gradient accumulation
                "max_grad_norm": 1.0,
                # M4/MPS optimizations
                "dataloader_pin_memory": False,
                "dataloader_num_workers": 0,
                "fp16": False,
            }
            
            # Add version-specific parameters safely
            training_args_dict.update({
                "eval_strategy": "steps",
                "save_strategy": "steps",
            })
            
            # Add optional parameters that might not exist in all versions
            optional_params = {
                "metric_for_best_model": "eval_loss",
            }
            
            # Only add parameters that are supported
            for key, value in optional_params.items():
                try:
                    # Test if parameter is supported by creating a minimal TrainingArguments
                    test_args = TrainingArguments(
                        output_dir="./test",
                        num_train_epochs=1,
                        per_device_train_batch_size=1,
                        **{key: value}
                    )
                    training_args_dict[key] = value
                except:
                    logger.warning(f"Parameter {key} not supported in this transformers version, skipping")
            
            logger.info(f"Training arguments: {training_args_dict}")
            training_args = TrainingArguments(**training_args_dict)
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train the model with fallback to CPU if MPS fails
            logger.info("Training model with optimization...")
            try:
                train_result = trainer.train()
            except RuntimeError as e:
                if "Insufficient Memory" in str(e) or "OutOfMemory" in str(e):
                    logger.warning("MPS memory error detected, retrying with CPU...")
                    # Move model to CPU and retry
                    self.model = self.model.to("cpu")
                    trainer.model = self.model
                    train_result = trainer.train()
                else:
                    raise e
            
            # Evaluate the model
            eval_result = trainer.evaluate()
            
            # Save the model
            await self._save_model()
            
            # Update training history
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'epochs': epochs,
                'batch_size': batch_size,
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result.get('eval_loss', 0.0),
                'training_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
                'method': 'optimized'
            }
            
            self.training_history.append(training_info)
            await self._save_training_history()
            
            self.is_trained = True
            self.last_trained = datetime.now()
            
            logger.info("Optimized model training completed successfully")
            
            return {
                "success": True,
                "message": "Model trained successfully with optimization",
                "training_info": training_info
            }
            
        except Exception as e:
            logger.error(f"Optimized training failed: {e}")
            return {"success": False, "message": f"Optimized training failed: {str(e)}"}
    
    async def _train_with_minimal_trainer(self, logs: List[LogEntry], epochs: int, batch_size: int) -> Dict[str, Any]:
        """Train using minimal trainer as fallback."""
        try:
            minimal_trainer = MinimalModelTrainer()
            result = await minimal_trainer.train_model(logs, epochs, batch_size)
            
            if result["success"]:
                # Update our own state
                self.is_trained = True
                self.last_trained = datetime.now()
                if "training_info" in result:
                    result["training_info"]["method"] = "minimal"
                    self.training_history.append(result["training_info"])
                    await self._save_training_history()
            
            return result
            
        except Exception as e:
            logger.error(f"Minimal training failed: {e}")
            return {"success": False, "message": f"Minimal training failed: {str(e)}"}
    
    async def answer_question(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question about system state."""
        if not self.is_trained or self.model is None:
            return {
                "answer": "Model not trained yet. Please train the model first.",
                "confidence": 0.0,
                "sources": []
            }
        
        try:
            # If no context provided, use recent logs as context
            if not context:
                context = await self._get_recent_logs_context()
            
            # Tokenize input
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get answer span
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)
            
            # Convert to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer_tokens = tokens[start_index:end_index + 1]
            answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
            
            # Calculate confidence
            confidence = float(torch.softmax(start_scores, dim=-1).max() * 
                             torch.softmax(end_scores, dim=-1).max())
            
            return {
                "answer": answer.strip(),
                "confidence": confidence,
                "sources": [context[:200] + "..." if len(context) > 200 else context]
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'model_path': self.model_path if os.path.exists(self.model_path) else None,
            'training_history': self.training_history[-5:] if self.training_history else [],
            'total_training_runs': len(self.training_history),
            'current_model': self.current_model_id
        }
    
    async def _generate_training_data(self, logs: List[LogEntry]) -> List[Dict[str, str]]:
        """Generate training data from logs."""
        training_data = []
        
        # Generate questions and answers based on log patterns
        for log in logs:
            # Security-related questions
            if log.source == 'security' or 'security' in log.message.lower():
                training_data.extend([
                    {
                        'question': 'Are there any security issues?',
                        'context': self._prepare_log_text(log),
                        'answer': log.message
                    },
                    {
                        'question': 'What security events occurred?',
                        'context': self._prepare_log_text(log),
                        'answer': log.message
                    }
                ])
            
            # Performance-related questions
            if 'cpu' in log.message.lower() or 'memory' in log.message.lower():
                training_data.extend([
                    {
                        'question': 'What is the system performance status?',
                        'context': self._prepare_log_text(log),
                        'answer': log.message
                    },
                    {
                        'question': 'Are there any performance issues?',
                        'context': self._prepare_log_text(log),
                        'answer': log.message
                    }
                ])
            
            # Hardware-related questions
            if 'disk' in log.message.lower() or 'hardware' in log.message.lower():
                training_data.extend([
                    {
                        'question': 'Are there any hardware issues?',
                        'context': self._prepare_log_text(log),
                        'answer': log.message
                    },
                    {
                        'question': 'What hardware problems are detected?',
                        'context': self._prepare_log_text(log),
                        'answer': log.message
                    }
                ])
            
            # General system questions
            training_data.extend([
                {
                    'question': 'What system events occurred?',
                    'context': self._prepare_log_text(log),
                    'answer': log.message
                },
                {
                    'question': f'What happened with {log.source}?',
                    'context': self._prepare_log_text(log),
                    'answer': log.message
                }
            ])
        
        return training_data
    
    async def _initialize_model(self):
        """Initialize the model and tokenizer."""
        model_name = self.current_model_id
        
        # Use Hugging Face token if available
        token = settings.HF_TOKEN if settings.HF_TOKEN else None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=token,
                trust_remote_code=True
            )
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                model_name, 
                token=token,
                trust_remote_code=True
            )
            logger.info(f"Model and tokenizer loaded successfully with HF token: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load with HF token, trying without: {e}")
            # Fallback to loading without token
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move model to optimal device
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
    
    async def _prepare_datasets(self, training_data: List[Dict[str, str]]) -> Tuple[SystemQADataset, SystemQADataset]:
        """Prepare training and evaluation datasets."""
        questions = [item['question'] for item in training_data]
        contexts = [item['context'] for item in training_data]
        answers = [item['answer'] for item in training_data]
        
        # Split data
        train_questions, eval_questions, train_contexts, eval_contexts, train_answers, eval_answers = train_test_split(
            questions, contexts, answers, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = SystemQADataset(train_questions, train_contexts, train_answers, self.tokenizer)
        eval_dataset = SystemQADataset(eval_questions, eval_contexts, eval_answers, self.tokenizer)
        
        return train_dataset, eval_dataset
    
    async def _save_model(self):
        """Save the trained model."""
        if self.model and self.tokenizer:
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.tokenizer_path)
            logger.info(f"Model saved to {self.model_path}")
    
    async def _load_model(self):
        """Load existing model if available."""
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
                self.is_trained = True
                logger.info("Loaded existing trained model")
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
    
    async def _save_training_history(self):
        """Save training history to file."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    async def _load_training_history(self):
        """Load training history from file."""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.training_history = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training history: {e}")
            self.training_history = []
    
    async def _get_recent_logs_context(self) -> str:
        """Get recent logs as context for questions."""
        # This would typically load recent logs from the log collector
        return "Recent system logs would be loaded here as context."
    
    def _prepare_log_text(self, log: LogEntry) -> str:
        """Prepare log text for training."""
        text_parts = [
            f"Source: {log.source}",
            f"Level: {log.level}",
            f"Message: {log.message}"
        ]
        
        if log.metadata:
            for key, value in log.metadata.items():
                text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
