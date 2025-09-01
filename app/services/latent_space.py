"""
Latent space service for creating embeddings and managing vector representations.
"""

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
import faiss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.schemas import LogEntry
from app.models.database_models import LatentSpaceDataDB


class LatentSpaceService:
    """Service for managing latent space representations of logs."""
    
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.log_embeddings = []
        self.log_texts = []
        self.is_initialized = False
        self.last_updated = None
        
    async def initialize(self):
        """Initialize the latent space service."""
        logger.info("Initializing LatentSpaceService...")
        
        try:
            # Load the embedding model with better error handling
            try:
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.is_initialized = False
                return
            
            # Try to load existing embeddings and index from database
            await self._load_existing_data()
            
            self.is_initialized = True
            logger.info("LatentSpaceService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LatentSpaceService: {e}")
            self.is_initialized = False
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up LatentSpaceService...")
        if self.embedding_model:
            del self.embedding_model
        if self.index:
            del self.index
    
    async def update_latent_space(self, logs: List[LogEntry]):
        """Update latent space with new log entries."""
        if not self.is_initialized:
            logger.warning("LatentSpaceService not initialized")
            return
        
        try:
            # Prepare log texts for embedding
            log_texts = []
            for log in logs:
                text = self._prepare_log_text(log)
                log_texts.append(text)
            
            if not log_texts:
                logger.info("No new logs to process")
                return
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                log_texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            
            # Add to existing data
            self.log_texts.extend(log_texts)
            self.log_embeddings.extend(embeddings.tolist())
            
            # Rebuild FAISS index
            await self._rebuild_index()
            
            # Save updated data
            await self._save_data()
            
            self.last_updated = datetime.now()
            logger.info(f"Updated latent space with {len(logs)} new log entries")
            
        except Exception as e:
            logger.error(f"Error updating latent space: {e}")
    
    async def search_similar_logs(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar logs using the latent space."""
        if not self.is_initialized or self.index is None:
            logger.warning("Latent space not initialized or index not available")
            return []
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query])
            
            # Search the index
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.log_texts))
            )
            
            # Return results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.log_texts):
                    similarity = 1.0 - distance  # Convert distance to similarity
                    results.append((self.log_texts[idx], similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching latent space: {e}")
            return []
    
    async def get_latent_space_stats(self) -> Dict[str, Any]:
        """Get statistics about the latent space."""
        import os
        import psutil
        
        # Calculate memory usage
        memory_usage_mb = None
        if self.log_embeddings:
            # Estimate memory usage of embeddings (float32 = 4 bytes per value)
            embedding_memory = len(self.log_embeddings) * len(self.log_embeddings[0]) * 4 / (1024 * 1024)  # MB
            memory_usage_mb = round(embedding_memory, 2)
        
        # Calculate average embedding length
        average_embedding_length = None
        if self.log_texts:
            total_length = sum(len(text) for text in self.log_texts)
            average_embedding_length = round(total_length / len(self.log_texts), 1)
        
        # Calculate sources and levels distribution
        sources_distribution = {}
        levels_distribution = {}
        if self.log_texts:
            for text in self.log_texts:
                # Parse source and level from text
                if "Source: " in text:
                    source = text.split("Source: ")[1].split(" |")[0]
                    sources_distribution[source] = sources_distribution.get(source, 0) + 1
                
                if "Level: " in text:
                    level = text.split("Level: ")[1].split(" |")[0]
                    levels_distribution[level] = levels_distribution.get(level, 0) + 1
        
        # Get index type
        index_type = None
        if self.index:
            index_type = type(self.index).__name__
        
        # Calculate compression ratio (if applicable)
        compression_ratio = None
        if self.index and hasattr(self.index, 'ntotal') and hasattr(self.index, 'nlist'):
            if self.index.nlist > 0:
                compression_ratio = round(self.index.ntotal / self.index.nlist, 2)
        
        return {
            'is_initialized': self.is_initialized,
            'total_embeddings': len(self.log_embeddings),
            'embedding_dimension': settings.LATENT_SPACE_DIMENSION,
            'model_name': settings.EMBEDDING_MODEL,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'index_size': self.index.ntotal if self.index else 0,
            'memory_usage_mb': memory_usage_mb,
            'average_embedding_length': average_embedding_length,
            'sources_distribution': sources_distribution,
            'levels_distribution': levels_distribution,
            'index_type': index_type,
            'compression_ratio': compression_ratio
        }
    
    async def rebuild_latent_space(self, logs: List[LogEntry]):
        """Completely rebuild the latent space."""
        logger.info("Rebuilding latent space...")
        
        try:
            # Limit the number of logs to prevent memory issues
            max_logs = 1000  # Limit to 1000 logs to prevent memory issues
            if len(logs) > max_logs:
                logger.warning(f"Limiting logs to {max_logs} to prevent memory issues (had {len(logs)})")
                logs = logs[:max_logs]
            
            # Clear existing data
            self.log_embeddings = []
            self.log_texts = []
            
            # Prepare all log texts with error handling
            log_texts = []
            for i, log in enumerate(logs):
                try:
                    text = self._prepare_log_text(log)
                    if text and len(text.strip()) > 0:
                        log_texts.append(text)
                except Exception as e:
                    logger.warning(f"Error processing log {i}: {e}")
                    continue
            
            if not log_texts:
                logger.warning("No valid logs to process for latent space rebuild")
                return
            
            logger.info(f"Processing {len(log_texts)} log texts for embeddings...")
            
            # Generate embeddings with smaller batch size and error handling
            try:
                embeddings = self.embedding_model.encode(
                    log_texts,
                    show_progress_bar=True,
                    batch_size=16,  # Reduced batch size
                    convert_to_numpy=True
                )
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                return
            
            # Store data
            self.log_texts = log_texts
            self.log_embeddings = embeddings.tolist()
            
            # Build new index with error handling
            await self._rebuild_index()
            
            # Save data
            await self._save_data()
            
            self.last_updated = datetime.now()
            logger.info(f"Successfully rebuilt latent space with {len(logs)} log entries")
            
        except Exception as e:
            logger.error(f"Error rebuilding latent space: {e}")
            # Reset to safe state
            self.log_embeddings = []
            self.log_texts = []
            self.index = None
    
    def _prepare_log_text(self, log: LogEntry) -> str:
        """Prepare log text for embedding."""
        # Create a comprehensive text representation
        text_parts = [
            f"Source: {log.source}",
            f"Level: {log.level}",
            f"Message: {log.message}"
        ]
        
        # Add metadata if available
        if log.metadata:
            for key, value in log.metadata.items():
                text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
    
    async def _rebuild_index(self):
        """Rebuild the FAISS index."""
        if not self.log_embeddings:
            logger.warning("No embeddings to build index from")
            return
        
        try:
            # Convert to numpy array with error handling
            embeddings_array = np.array(self.log_embeddings, dtype='float32')
            
            # Validate array
            if embeddings_array.size == 0:
                logger.warning("Empty embeddings array")
                return
            
            if len(embeddings_array.shape) != 2:
                logger.error(f"Invalid embeddings shape: {embeddings_array.shape}")
                return
            
            dimension = embeddings_array.shape[1]
            logger.info(f"Building FAISS index with {embeddings_array.shape[0]} vectors of dimension {dimension}")
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add vectors to index in smaller batches to prevent memory issues
            batch_size = 100
            for i in range(0, len(embeddings_array), batch_size):
                batch = embeddings_array[i:i + batch_size]
                self.index.add(batch)
            
            logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            self.index = None
    
    async def _save_data(self):
        """Save embeddings and metadata to database."""
        try:
            if not self.log_embeddings or not self.log_texts:
                logger.warning("No embeddings or texts to save")
                return
                
            async with AsyncSessionLocal() as session:
                # Prepare data for storage
                embedding_data = pickle.dumps(self.log_embeddings)
                text_data = pickle.dumps(self.log_texts)
                
                # Prepare FAISS index data
                faiss_index_data = None
                if self.index and self.index.ntotal > 0:
                    try:
                        faiss_index_data = faiss.serialize_index(self.index).tobytes()
                    except Exception as e:
                        logger.error(f"Error serializing FAISS index: {e}")
                
                # Prepare metadata with detailed algorithm information
                metadata = {
                    'last_updated': self.last_updated.isoformat() if self.last_updated else None,
                    'total_embeddings': len(self.log_embeddings),
                    'model_name': settings.EMBEDDING_MODEL,
                    'dimension': settings.LATENT_SPACE_DIMENSION,
                    'index_size': self.index.ntotal if self.index else 0,
                    'algorithm_details': {
                        'embedding_model': {
                            'name': settings.EMBEDDING_MODEL,
                            'type': 'Sentence Transformer',
                            'architecture': 'MiniLM (Lightweight BERT)',
                            'dimension': settings.LATENT_SPACE_DIMENSION,
                            'max_sequence_length': settings.MAX_SEQUENCE_LENGTH,
                            'description': 'Uses a lightweight BERT-based model for generating semantic embeddings'
                        },
                        'similarity_algorithm': {
                            'type': 'Cosine Similarity',
                            'implementation': 'FAISS IndexFlatIP',
                            'normalization': 'L2 Normalization',
                            'threshold': settings.SIMILARITY_THRESHOLD,
                            'description': 'Uses inner product with L2 normalization for cosine similarity calculation'
                        },
                        'indexing_algorithm': {
                            'type': 'FAISS Flat Index',
                            'algorithm': 'IndexFlatIP (Inner Product)',
                            'batch_size': 100,
                            'description': 'Simple but effective flat index for exact similarity search'
                        },
                        'text_preprocessing': {
                            'format': 'Structured concatenation',
                            'fields': ['Source', 'Level', 'Message', 'Metadata'],
                            'separator': ' | ',
                            'description': 'Log entries are converted to structured text format for embedding'
                        }
                    }
                }
                
                # Check if we have existing data to update
                stmt = select(LatentSpaceDataDB).order_by(LatentSpaceDataDB.id.desc()).limit(1)
                result = await session.execute(stmt)
                existing_record = result.scalar_one_or_none()
                
                if existing_record:
                    # Update existing record
                    existing_record.embedding_data = embedding_data
                    existing_record.text_data = text_data
                    existing_record.faiss_index = faiss_index_data
                    existing_record.algorithm_metadata = metadata
                    existing_record.updated_at = datetime.now()
                    logger.info(f"Updated existing latent space data with {len(self.log_embeddings)} embeddings")
                else:
                    # Create new record
                    new_record = LatentSpaceDataDB(
                        embedding_data=embedding_data,
                        text_data=text_data,
                        faiss_index=faiss_index_data,
                        algorithm_metadata=metadata
                    )
                    session.add(new_record)
                    logger.info(f"Created new latent space data with {len(self.log_embeddings)} embeddings")
                
                await session.commit()
                logger.info("Successfully saved latent space data to database")
                
        except Exception as e:
            logger.error(f"Error saving latent space data to database: {e}")
            if 'session' in locals():
                await session.rollback()
    
    async def _load_existing_data(self):
        """Load existing embeddings and index from disk."""
        try:
            # Load embeddings
            if os.path.exists(settings.DATA_DIR + "/log_embeddings.pkl"):
                try:
                    with open(settings.DATA_DIR + "/log_embeddings.pkl", 'rb') as f:
                        data = pickle.load(f)
                        self.log_embeddings = data['embeddings']
                        self.log_texts = data['texts']
                    logger.info(f"Loaded {len(self.log_embeddings)} existing embeddings")
                except Exception as e:
                    logger.warning(f"Could not load embeddings file: {e}")
                    self.log_embeddings = []
                    self.log_texts = []
            
            # Load FAISS index with better error handling
            if os.path.exists(settings.DATA_DIR + "/faiss_index.bin"):
                try:
                    self.index = faiss.read_index(settings.DATA_DIR + "/faiss_index.bin")
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(f"Could not load FAISS index (corrupted): {e}")
                    # Remove corrupted file
                    try:
                        os.remove(settings.DATA_DIR + "/faiss_index.bin")
                        logger.info("Removed corrupted FAISS index file")
                    except:
                        pass
                    self.index = None
            
            # Load metadata
            if os.path.exists(settings.DATA_DIR + "/latent_space_metadata.json"):
                try:
                    with open(settings.DATA_DIR + "/latent_space_metadata.json", 'r') as f:
                        metadata = json.load(f)
                        if metadata.get('last_updated'):
                            self.last_updated = datetime.fromisoformat(metadata['last_updated'])
                except Exception as e:
                    logger.warning(f"Could not load metadata file: {e}")
                        
        except Exception as e:
            logger.warning(f"Could not load existing latent space data: {e}")
            # Start fresh
            self.log_embeddings = []
            self.log_texts = []
            self.index = None
