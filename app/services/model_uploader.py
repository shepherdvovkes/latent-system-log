"""
Model uploader service for uploading trained models to Hugging Face Hub.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from app.core.config import settings


class ModelUploader:
    """Service for uploading trained models to Hugging Face Hub."""
    
    def __init__(self):
        self.api = HfApi()
        self.token = settings.HF_TOKEN
        
    def upload_model(
        self, 
        model_path: str, 
        repo_name: str,
        model_description: str = "System log analysis question answering model",
        private: bool = True
    ) -> Dict[str, Any]:
        """
        Upload a trained model to Hugging Face Hub.
        
        Args:
            model_path: Path to the trained model
            repo_name: Name for the repository on HF Hub
            model_description: Description of the model
            private: Whether the repository should be private
            
        Returns:
            Dictionary with upload results
        """
        try:
            if not self.token:
                return {
                    "success": False,
                    "message": "No Hugging Face token configured"
                }
            
            # Check if model exists
            if not os.path.exists(model_path):
                return {
                    "success": False,
                    "message": f"Model path does not exist: {model_path}"
                }
            
            # Create repository
            logger.info(f"Creating repository: {repo_name}")
            repo_url = create_repo(
                repo_id=repo_name,
                token=self.token,
                private=private,
                exist_ok=True
            )
            
            # Upload model files
            logger.info(f"Uploading model from {model_path}")
            upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                token=self.token,
                commit_message=f"Upload system log analysis model - {datetime.now().isoformat()}"
            )
            
            # Create model card
            model_card = self._create_model_card(model_description)
            self.api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_name,
                token=self.token,
                commit_message="Add model card"
            )
            
            logger.info(f"Model uploaded successfully to: {repo_url}")
            
            return {
                "success": True,
                "message": "Model uploaded successfully",
                "repo_url": repo_url,
                "repo_name": repo_name,
                "upload_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return {
                "success": False,
                "message": f"Upload failed: {str(e)}"
            }
    
    def _create_model_card(self, description: str) -> str:
        """Create a model card for the uploaded model."""
        return f"""# System Log Analysis Question Answering Model

{description}

## Model Details

- **Model Type**: DistilBERT for Question Answering
- **Base Model**: distilbert-base-cased-distilled-squad
- **Training Data**: System logs and generated Q&A pairs
- **Use Case**: Answering questions about system state and events

## Usage

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the model
tokenizer = AutoTokenizer.from_pretrained("your-username/{repo_name}")
model = AutoModelForQuestionAnswering.from_pretrained("your-username/{repo_name}")

# Use for question answering
question = "What security events occurred?"
context = "System log context here..."
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
```

## Training Information

- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Framework**: PyTorch with Transformers
- **Optimization**: Apple M4 Neural Chip (MPS)
- **Batch Size**: Optimized for M4 unified memory

## License

This model is trained on system logs and should be used in accordance with your organization's data policies.
"""
    
    def list_uploaded_models(self) -> Dict[str, Any]:
        """List models uploaded by the current user."""
        try:
            if not self.token:
                return {
                    "success": False,
                    "message": "No Hugging Face token configured"
                }
            
            # Get user info
            user_info = self.api.whoami(token=self.token)
            username = user_info.get('name', 'unknown')
            
            # List models
            models = self.api.list_models(author=username, token=self.token)
            
            model_repos = []
            for model in models:
                if model.modelId and 'system-log' in model.modelId.lower():
                    model_repos.append({
                        "name": model.modelId,
                        "url": f"https://huggingface.co/{model.modelId}",
                        "private": model.private,
                        "updated_at": model.lastModified.isoformat() if model.lastModified else None
                    })
            
            return {
                "success": True,
                "models": model_repos,
                "total_count": len(model_repos)
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {
                "success": False,
                "message": f"Failed to list models: {str(e)}"
            }
    
    def delete_model(self, repo_name: str) -> Dict[str, Any]:
        """Delete a model from Hugging Face Hub."""
        try:
            if not self.token:
                return {
                    "success": False,
                    "message": "No Hugging Face token configured"
                }
            
            # Delete repository
            self.api.delete_repo(repo_id=repo_name, token=self.token)
            
            logger.info(f"Model {repo_name} deleted successfully")
            
            return {
                "success": True,
                "message": f"Model {repo_name} deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return {
                "success": False,
                "message": f"Failed to delete model: {str(e)}"
            }
