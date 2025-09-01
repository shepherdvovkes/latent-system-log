"""
Hugging Face Model Searcher for System Log Analysis.
"""

import requests
import json
from typing import List, Dict, Any, Optional
from loguru import logger
from app.core.config import settings


class HFModelSearcher:
    """Service for searching Hugging Face models suitable for system log analysis."""
    
    def __init__(self):
        self.base_url = "https://huggingface.co/api"
        self.token = settings.HF_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.token}" if self.token else None,
            "Content-Type": "application/json"
        }
    
    def search_models(
        self, 
        query: str = "system log analysis",
        task: str = "question-answering",
        limit: int = 20,
        sort: str = "downloads"
    ) -> Dict[str, Any]:
        """
        Search for models on Hugging Face.
        
        Args:
            query: Search query
            task: Model task type
            limit: Number of results to return
            sort: Sort order (downloads, likes, updated)
            
        Returns:
            Dictionary with search results
        """
        try:
            url = f"{self.base_url}/models"
            params = {
                "search": query,
                "filter": f"task:{task}",
                "limit": limit,
                "sort": sort,
                "direction": "-1"  # Descending order
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            models = response.json()
            
            # Filter and enhance results
            filtered_models = []
            for model in models:
                if self._is_suitable_for_system_logs(model):
                    enhanced_model = self._enhance_model_info(model)
                    filtered_models.append(enhanced_model)
            
            return {
                "success": True,
                "models": filtered_models,
                "total_found": len(filtered_models),
                "query": query,
                "task": task
            }
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "models": []
            }
    
    def _is_suitable_for_system_logs(self, model: Dict[str, Any]) -> bool:
        """Check if a model is suitable for system log analysis."""
        # Check model name and tags
        model_id = model.get("modelId", "").lower()
        tags = [tag.lower() for tag in model.get("tags", [])]
        
        # Keywords that indicate suitability for system logs
        log_keywords = [
            "log", "system", "monitoring", "analysis", "qa", "question-answering",
            "distilbert", "bert", "roberta", "albert", "deberta", "t5", "gpt"
        ]
        
        # Check if model name contains relevant keywords
        name_suitable = any(keyword in model_id for keyword in log_keywords)
        
        # Check if tags contain relevant information
        tags_suitable = any(keyword in tags for keyword in log_keywords)
        
        # Check if it's a question-answering model
        is_qa = "question-answering" in tags or "qa" in model_id
        
        # Check if it's not too large (for practical fine-tuning)
        size_suitable = self._check_model_size(model)
        
        return (name_suitable or tags_suitable) and is_qa and size_suitable
    
    def _check_model_size(self, model: Dict[str, Any]) -> bool:
        """Check if model size is suitable for fine-tuning."""
        # Get model size information
        size_info = model.get("siblings", [])
        
        # Calculate total size
        total_size = 0
        for sibling in size_info:
            if sibling.get("rfilename", "").endswith((".bin", ".safetensors")):
                total_size += sibling.get("size", 0)
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        
        # Prefer models under 2GB for practical fine-tuning
        return size_mb < 2048
    
    def _enhance_model_info(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance model information with additional details."""
        return {
            "model_id": model.get("modelId"),
            "name": model.get("modelId", "").split("/")[-1],
            "author": model.get("modelId", "").split("/")[0] if "/" in model.get("modelId", "") else "unknown",
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "tags": model.get("tags", []),
            "last_modified": model.get("lastModified"),
            "size_mb": self._calculate_model_size(model),
            "description": model.get("description", ""),
            "url": f"https://huggingface.co/{model.get('modelId')}",
            "suitability_score": self._calculate_suitability_score(model)
        }
    
    def _calculate_model_size(self, model: Dict[str, Any]) -> float:
        """Calculate model size in MB."""
        size_info = model.get("siblings", [])
        total_size = 0
        
        for sibling in size_info:
            if sibling.get("rfilename", "").endswith((".bin", ".safetensors")):
                total_size += sibling.get("size", 0)
        
        return round(total_size / (1024 * 1024), 2)
    
    def _calculate_suitability_score(self, model: Dict[str, Any]) -> int:
        """Calculate a suitability score for system log analysis."""
        score = 0
        
        # Base score for being a QA model
        if "question-answering" in model.get("tags", []):
            score += 30
        
        # Score for downloads (popularity)
        downloads = model.get("downloads", 0)
        if downloads > 10000:
            score += 25
        elif downloads > 1000:
            score += 15
        elif downloads > 100:
            score += 5
        
        # Score for likes
        likes = model.get("likes", 0)
        if likes > 100:
            score += 20
        elif likes > 10:
            score += 10
        
        # Score for relevant keywords in name
        model_id = model.get("modelId", "").lower()
        if "distilbert" in model_id:
            score += 15  # DistilBERT is good for fine-tuning
        elif "bert" in model_id:
            score += 10
        elif "roberta" in model_id:
            score += 10
        elif "log" in model_id or "system" in model_id:
            score += 20  # Directly relevant
        
        # Score for size (smaller is better for fine-tuning)
        size_mb = self._calculate_model_size(model)
        if size_mb < 500:
            score += 15
        elif size_mb < 1000:
            score += 10
        elif size_mb < 2000:
            score += 5
        
        return score
    
    def get_recommended_models(self) -> Dict[str, Any]:
        """Get a curated list of recommended models for system log analysis."""
        recommendations = {
            "best_overall": [
                "distilbert-base-cased-distilled-squad",  # Current choice
                "deepset/roberta-base-squad2",
                "microsoft/DialoGPT-medium",
                "facebook/bart-base",
                "t5-small"
            ],
            "specialized": [
                "microsoft/DialoGPT-small",  # Good for conversational QA
                "distilbert-base-uncased-distilled-squad",  # Uncased version
                "deepset/bert-base-cased-squad2",  # BERT-based QA
                "microsoft/DialoGPT-large"  # Larger model for better performance
            ],
            "lightweight": [
                "distilbert-base-cased-distilled-squad",  # Already using
                "microsoft/DialoGPT-small",
                "t5-small",
                "facebook/bart-base"
            ]
        }
        
        # Get detailed info for recommended models
        detailed_recommendations = {}
        for category, models in recommendations.items():
            detailed_recommendations[category] = []
            for model_id in models:
                model_info = self.get_model_info(model_id)
                if model_info["success"]:
                    detailed_recommendations[category].append(model_info["model"])
        
        return {
            "success": True,
            "recommendations": detailed_recommendations
        }
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            url = f"{self.base_url}/models/{model_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            model = response.json()
            enhanced_model = self._enhance_model_info(model)
            
            return {
                "success": True,
                "model": enhanced_model
            }
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to get model info: {str(e)}"
            }
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models side by side."""
        comparison = {
            "models": [],
            "comparison_table": {}
        }
        
        for model_id in model_ids:
            model_info = self.get_model_info(model_id)
            if model_info["success"]:
                comparison["models"].append(model_info["model"])
        
        # Create comparison table
        if comparison["models"]:
            comparison["comparison_table"] = {
                "model_id": [m["model_id"] for m in comparison["models"]],
                "size_mb": [m["size_mb"] for m in comparison["models"]],
                "downloads": [m["downloads"] for m in comparison["models"]],
                "likes": [m["likes"] for m in comparison["models"]],
                "suitability_score": [m["suitability_score"] for m in comparison["models"]]
            }
        
        return {
            "success": True,
            "comparison": comparison
        }
