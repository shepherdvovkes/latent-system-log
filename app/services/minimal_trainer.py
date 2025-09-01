"""
Minimal model trainer with maximum compatibility.
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
from app.models.schemas import LogEntry


def get_optimal_device():
    """Get the optimal device for training."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for GPU acceleration")
        return torch.device("cuda")
    else:
        logger.info("Using CPU for training")
        return torch.device("cpu")


class MinimalQADataset(Dataset):
    """Minimal dataset for question answering."""
    
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


class MinimalModelTrainer:
    """Minimal model trainer with maximum compatibility."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = get_optimal_device()
        self.is_trained = False
        self.last_trained = None
        self.training_history = []
        self.model_path = "./models/qa_model"
        self.history_path = "./models/training_history.json"
        
        # Create model directory
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./training_output", exist_ok=True)
        
        logger.info(f"Minimal model trainer initialized with device: {self.device}")
    
    async def train_model(self, logs: List[LogEntry], epochs: int = 3, batch_size: int = 8) -> Dict[str, Any]:
        """Train the question answering model with minimal configuration."""
        logger.info("Starting minimal model training...")
        
        try:
            # Generate training data from logs
            training_data = await self._generate_training_data(logs)
            
            if not training_data:
                logger.warning("No training data generated")
                return {"success": False, "message": "No training data available"}
            
            # Initialize model and tokenizer
            await self._initialize_model()
            
            # Prepare datasets
            train_dataset, eval_dataset = await self._prepare_datasets(training_data)
            
            # Create minimal training arguments
            training_args = TrainingArguments(
                output_dir="./training_output",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=100,
                weight_decay=0.01,
                learning_rate=2e-5,
                logging_dir="./logs",
                eval_strategy="steps",
                save_strategy="steps",
            )
            
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
            
            # Train the model
            logger.info("Training model...")
            train_result = trainer.train()
            
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
                'eval_samples': len(eval_dataset)
            }
            
            self.training_history.append(training_info)
            await self._save_training_history()
            
            self.is_trained = True
            self.last_trained = datetime.now()
            
            logger.info("Model training completed successfully")
            
            return {
                "success": True,
                "message": "Model trained successfully",
                "training_info": training_info
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "message": f"Training failed: {str(e)}"}
    
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
    
    def _prepare_log_text(self, log: LogEntry) -> str:
        """Prepare log text for training."""
        return f"Source: {log.source}, Level: {log.level}, Message: {log.message}"
    
    async def _initialize_model(self):
        """Initialize the model and tokenizer."""
        model_name = "distilbert-base-cased-distilled-squad"
        
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
            logger.info(f"Model and tokenizer loaded successfully with HF token")
        except Exception as e:
            logger.warning(f"Failed to load with HF token, trying without: {e}")
            # Fallback to loading without token
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move model to optimal device
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
    
    async def _prepare_datasets(self, training_data: List[Dict[str, str]]) -> Tuple[MinimalQADataset, MinimalQADataset]:
        """Prepare training and evaluation datasets."""
        questions = [item['question'] for item in training_data]
        contexts = [item['context'] for item in training_data]
        answers = [item['answer'] for item in training_data]
        
        # Split data
        train_questions, eval_questions, train_contexts, eval_contexts, train_answers, eval_answers = train_test_split(
            questions, contexts, answers, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = MinimalQADataset(train_questions, train_contexts, train_answers, self.tokenizer)
        eval_dataset = MinimalQADataset(eval_questions, eval_contexts, eval_answers, self.tokenizer)
        
        return train_dataset, eval_dataset
    
    async def _save_model(self):
        """Save the trained model."""
        if self.model and self.tokenizer:
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    async def _save_training_history(self):
        """Save training history."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get training status."""
        return {
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'model_path': self.model_path if os.path.exists(self.model_path) else None,
            'training_history': self.training_history[-5:] if self.training_history else [],
            'total_training_runs': len(self.training_history),
            'device': str(self.device)
        }
