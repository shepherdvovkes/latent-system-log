"""
Configuration settings for the System Log Analysis and AI Assistant.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Log collection settings
    LOG_COLLECTION_INTERVAL: int = Field(default=600, env="LOG_COLLECTION_INTERVAL")  # 10 minutes
    LOG_RETENTION_DAYS: int = Field(default=30, env="LOG_RETENTION_DAYS")
    
    # Database settings
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./data/logs.db", env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")
    
    # ML settings
    MODEL_UPDATE_INTERVAL: int = Field(default=86400, env="MODEL_UPDATE_INTERVAL")  # 24 hours
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    MAX_SEQUENCE_LENGTH: int = Field(default=512, env="MAX_SEQUENCE_LENGTH")
    
    # Data storage
    DATA_DIR: str = Field(default="data", env="DATA_DIR")
    MODELS_DIR: str = Field(default="models", env="MODELS_DIR")
    LOGS_DIR: str = Field(default="logs", env="LOGS_DIR")
    
    # Latent space settings
    LATENT_SPACE_DIMENSION: int = Field(default=384, env="LATENT_SPACE_DIMENSION")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Security settings
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Hugging Face settings
    HF_TOKEN: Optional[str] = Field(default=None, env="HF_TOKEN")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)
