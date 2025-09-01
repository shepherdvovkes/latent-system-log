"""
Data Export Service for Google Colab Training.
"""

import json
import os
import zipfile
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger
import pandas as pd

from app.core.config import settings
from app.models.schemas import LogEntry
from app.services.database_service import DatabaseService


class DataExporterService:
    """Service for exporting training data to Google Colab."""
    
    def __init__(self):
        self.export_dir = os.path.join(settings.DATA_DIR, "exports")
        os.makedirs(self.export_dir, exist_ok=True)
        self.db_service = DatabaseService()
    
    async def initialize(self):
        """Initialize the data exporter service."""
        await self.db_service.initialize()
        logger.info("DataExporterService initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.db_service.cleanup()
    
    async def export_training_data(self, hours: int = 24) -> Dict[str, Any]:
        """Export training data for Google Colab."""
        try:
            logger.info(f"Exporting training data for last {hours} hours...")
            
            # Get logs from database
            from datetime import timedelta
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            logs = await self.db_service.get_logs_by_time_range(start_time, end_time)
            
            if not logs:
                return {
                    "success": False,
                    "message": "No logs found for the specified time range"
                }
            
            # Convert to training format
            training_data = await self._prepare_training_data(logs)
            
            # Create export files
            export_files = await self._create_export_files(training_data)
            
            # Create Colab notebook
            colab_notebook = self._create_colab_notebook(export_files)
            
            return {
                "success": True,
                "message": f"Successfully exported {len(logs)} logs",
                "export_files": export_files,
                "colab_notebook": colab_notebook,
                "training_data_stats": {
                    "total_logs": len(logs),
                    "training_samples": len(training_data.get("questions", [])),
                    "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return {
                "success": False,
                "message": f"Export failed: {str(e)}"
            }
    
    async def _prepare_training_data(self, logs: List) -> Dict[str, Any]:
        """Prepare logs for training format."""
        questions = []
        contexts = []
        answers = []
        
        # Generate QA pairs from logs
        for i, log in enumerate(logs):
            # Create question-answer pairs
            qa_pairs = self._generate_qa_pairs(log)
            
            for question, answer in qa_pairs:
                questions.append(question)
                contexts.append(log.message)
                answers.append(answer)
        
        return {
            "questions": questions,
            "contexts": contexts,
            "answers": answers,
            "metadata": {
                "total_samples": len(questions),
                "export_timestamp": datetime.now().isoformat(),
                "model_config": {
                    "model_name": "distilbert-base-cased-distilled-squad",
                    "max_length": 512,
                    "batch_size": 16,  # Larger batch size for Colab
                    "epochs": 3,
                    "learning_rate": 2e-5
                }
            }
        }
    
    def _generate_qa_pairs(self, log) -> List[tuple]:
        """Generate question-answer pairs from a log entry."""
        qa_pairs = []
        
        # Basic QA pairs based on log content
        if log.level:
            qa_pairs.append((
                f"What is the log level?",
                log.level
            ))
        
        if log.source:
            qa_pairs.append((
                f"What is the source of this log?",
                log.source
            ))
        
        # Extract key information from message
        message = log.message
        if "error" in message.lower():
            qa_pairs.append((
                f"What error occurred?",
                message
            ))
        
        if "timeout" in message.lower():
            qa_pairs.append((
                f"What timeout issue happened?",
                message
            ))
        
        # Generic QA pair
        qa_pairs.append((
            f"What does this log entry indicate?",
            message
        ))
        
        return qa_pairs
    
    async def _create_export_files(self, training_data: Dict[str, Any]) -> Dict[str, str]:
        """Create export files for Colab."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Training data JSON
        training_file = os.path.join(self.export_dir, f"training_data_{timestamp}.json")
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # 2. Training data CSV
        csv_file = os.path.join(self.export_dir, f"training_data_{timestamp}.csv")
        df = pd.DataFrame({
            'question': training_data['questions'],
            'context': training_data['contexts'],
            'answer': training_data['answers']
        })
        df.to_csv(csv_file, index=False)
        
        # 3. Model configuration
        config_file = os.path.join(self.export_dir, f"model_config_{timestamp}.json")
        with open(config_file, 'w') as f:
            json.dump(training_data['metadata']['model_config'], f, indent=2)
        
        # 4. Create zip file
        zip_file = os.path.join(self.export_dir, f"colab_training_data_{timestamp}.zip")
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            zipf.write(training_file, os.path.basename(training_file))
            zipf.write(csv_file, os.path.basename(csv_file))
            zipf.write(config_file, os.path.basename(config_file))
        
        return {
            "training_json": training_file,
            "training_csv": csv_file,
            "config_json": config_file,
            "zip_file": zip_file,
            "timestamp": timestamp
        }
    
    def _create_colab_notebook(self, export_files: Dict[str, str]) -> str:
        """Create a Google Colab notebook for training."""
        notebook_content = f'''{{
  "cells": [
    {{
      "cell_type": "markdown",
      "metadata": {{
        "id": "header"
      }},
      "source": [
        "# System Log Analysis Model Training\\n",
        "\\n",
        "This notebook trains a question-answering model on system log data.\\n",
        "\\n",
        "**Export timestamp**: {export_files['timestamp']}\\n",
        "**Files**: {os.path.basename(export_files['zip_file'])}"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{
        "id": "setup"
      }},
      "outputs": [],
      "source": [
        "# Install dependencies\\n",
        "!pip install transformers datasets torch accelerate\\n",
        "!pip install pandas numpy scikit-learn"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{
        "id": "upload"
      }},
      "outputs": [],
      "source": [
        "# Upload the training data zip file\\n",
        "from google.colab import files\\n",
        "uploaded = files.upload()\\n",
        "\\n",
        "# Extract the zip file\\n",
        "import zipfile\\n",
        "with zipfile.ZipFile('{os.path.basename(export_files['zip_file'])}', 'r') as zip_ref:\\n",
        "    zip_ref.extractall('.')"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{
        "id": "load_data"
      }},
      "outputs": [],
      "source": [
        "# Load training data\\n",
        "import json\\n",
        "import pandas as pd\\n",
        "\\n",
        "with open('{os.path.basename(export_files['training_json'])}', 'r') as f:\\n",
        "    training_data = json.load(f)\\n",
        "\\n",
        "with open('{os.path.basename(export_files['config_json'])}', 'r') as f:\\n",
        "    model_config = json.load(f)\\n",
        "\\n",
        "print(f\"Training samples: {{len(training_data['questions'])}}\")\\n",
        "print(f\"Model config: {{model_config}}\")"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{
        "id": "train"
      }},
      "outputs": [],
      "source": [
        "# Train the model\\n",
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer\\n",
        "from datasets import Dataset\\n",
        "import torch\\n",
        "\\n",
        "# Load model and tokenizer\\n",
        "model_name = model_config['model_name']\\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)\\n",
        "model = AutoModelForQuestionAnswering.from_pretrained(model_name)\\n",
        "\\n",
        "# Prepare dataset\\n",
        "dataset = Dataset.from_dict({{\\n",
        "    'question': training_data['questions'],\\n",
        "    'context': training_data['contexts'],\\n",
        "    'answer': training_data['answers']\\n",
        "}})\\n",
        "\\n",
        "# Training arguments\\n",
        "training_args = TrainingArguments(\\n",
        "    output_dir=\"./results\",\\n",
        "    num_train_epochs=model_config['epochs'],\\n",
        "    per_device_train_batch_size=model_config['batch_size'],\\n",
        "    per_device_eval_batch_size=model_config['batch_size'],\\n",
        "    learning_rate=model_config['learning_rate'],\\n",
        "    logging_steps=10,\\n",
        "    save_steps=100,\\n",
        "    eval_steps=100,\\n",
        "    warmup_steps=100,\\n",
        "    weight_decay=0.01,\\n",
        "    gradient_accumulation_steps=2,\\n",
        "    max_grad_norm=1.0,\\n",
        "    evaluation_strategy=\"steps\",\\n",
        "    save_strategy=\"steps\",\\n",
        "    load_best_model_at_end=True,\\n",
        "    metric_for_best_model=\"eval_loss\"\\n",
        ")\\n",
        "\\n",
        "# Train\\n",
        "trainer = Trainer(\\n",
        "    model=model,\\n",
        "    args=training_args,\\n",
        "    train_dataset=dataset,\\n",
        "    eval_dataset=dataset,\\n",
        "    tokenizer=tokenizer\\n",
        ")\\n",
        "\\n",
        "trainer.train()"
      ]
    }},
    {{
      "cell_type": "code",
      "execution_count": null,
      "metadata": {{
        "id": "download"
      }},
      "outputs": [],
      "source": [
        "# Download the trained model\\n",
        "!zip -r trained_model.zip ./results\\n",
        "files.download('trained_model.zip')"
      ]
    }}
  ],
  "metadata": {{
    "colab": {{
      "name": "System Log Model Training",
      "provenance": []
    }},
    "kernelspec": {{
      "display_name": "Python 3",
      "name": "python3"
    }},
    "language_info": {{
      "name": "python"
    }}
  }},
  "nbformat": 4,
  "nbformat_minor": 0
}}'''
        
        notebook_file = os.path.join(self.export_dir, f"colab_notebook_{export_files['timestamp']}.ipynb")
        with open(notebook_file, 'w') as f:
            f.write(notebook_content)
        
        return notebook_file
    
    async def get_export_history(self) -> List[Dict[str, Any]]:
        """Get history of exports."""
        exports = []
        for file in os.listdir(self.export_dir):
            if file.startswith("colab_training_data_") and file.endswith(".zip"):
                file_path = os.path.join(self.export_dir, file)
                stat = os.stat(file_path)
                exports.append({
                    "file": file,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "path": file_path
                })
        
        return sorted(exports, key=lambda x: x["created"], reverse=True)
