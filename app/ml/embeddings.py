"""
Embeddings module for text processing and similarity calculations.
"""

import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
import torch
from loguru import logger

from app.core.config import settings


class EmbeddingManager:
    """Manager for text embeddings and similarity calculations."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the embedding model."""
        try:
            # Use Hugging Face token if available
            token = settings.HF_TOKEN if settings.HF_TOKEN else None
            
            if token:
                self.model = SentenceTransformer(
                    self.model_name,
                    token=token,
                    trust_remote_code=True
                )
                logger.info(f"Initialized embedding model with HF token: {self.model_name}")
            else:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized embedding model: {self.model_name}")
            
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.is_initialized = False
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts to embeddings."""
        if not self.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        if not self.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Error encoding single text: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            similarity = util.pytorch_cos_sim(
                torch.tensor(embedding1).unsqueeze(0),
                torch.tensor(embedding2).unsqueeze(0)
            )
            return float(similarity[0][0])
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_texts(
        self, 
        query_embedding: np.ndarray, 
        text_embeddings: np.ndarray, 
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find similar texts based on embedding similarity."""
        try:
            # Calculate similarities
            similarities = util.pytorch_cos_sim(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(text_embeddings)
            )[0]
            
            # Get top-k similar texts
            top_indices = torch.topk(similarities, min(top_k, len(texts))).indices
            
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity >= threshold:
                    results.append((texts[idx], similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar texts: {e}")
            return []
    
    def batch_similarity_search(
        self,
        query_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        texts: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Perform batch similarity search."""
        try:
            similarities = util.pytorch_cos_sim(
                torch.tensor(query_embeddings),
                torch.tensor(text_embeddings)
            )
            
            results = []
            for i in range(len(query_embeddings)):
                top_indices = torch.topk(similarities[i], min(top_k, len(texts))).indices
                batch_results = []
                for idx in top_indices:
                    similarity = float(similarities[i][idx])
                    batch_results.append((texts[idx], similarity))
                results.append(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch similarity search: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if not self.is_initialized:
            return 0
        
        # Create a dummy embedding to get dimension
        dummy_embedding = self.encode_single_text("test")
        return len(dummy_embedding)
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        try:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        except Exception as e:
            logger.error(f"Error normalizing embeddings: {e}")
            return embeddings


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?\:\;\(\)\[\]\{\}]', '', text)
        
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can find a good break point
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        import re
        from collections import Counter
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count and return most common
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    @staticmethod
    def create_summary(text: str, max_sentences: int = 3) -> str:
        """Create a summary of the text."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Return first few sentences
        return '. '.join(sentences[:max_sentences]) + '.'
