"""
M4 Neural Chip Optimization Configuration
Optimized settings for Apple M4 neural chip training and inference.
"""

import torch
import os
from loguru import logger
from typing import Dict, Any, Optional


class M4Optimizer:
    """Optimization utilities for Apple M4 neural chip."""
    
    @staticmethod
    def get_optimal_batch_size(model_size_mb: float = 100) -> int:
        """
        Calculate optimal batch size for M4 based on model size.
        
        Args:
            model_size_mb: Model size in megabytes
            
        Returns:
            Optimal batch size
        """
        # M4 has unified memory, so we can be more aggressive with batch sizes
        if model_size_mb < 50:
            return 64
        elif model_size_mb < 200:
            return 32
        elif model_size_mb < 500:
            return 16
        else:
            return 8
    
    @staticmethod
    def get_training_config() -> Dict[str, Any]:
        """Get optimal training configuration for M4."""
        return {
            # Memory optimization
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            
            # Learning rate optimization for M4
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            
            # Batch size optimization - reduced for M4 memory safety
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            
            # MPS-specific settings
            "dataloader_pin_memory": False,
            "dataloader_num_workers": 0,
            "fp16": False,  # Disable mixed precision for MPS
            
            # Evaluation (compatible with different transformers versions)
            "eval_strategy": "steps",  # Newer versions use eval_strategy
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 100,
            
            # Logging
            "logging_steps": 10,
            "report_to": None,  # Disable wandb/tensorboard for simplicity
        }
    
    @staticmethod
    def get_inference_config() -> Dict[str, Any]:
        """Get optimal inference configuration for M4."""
        return {
            "batch_size": 32,
            "max_length": 512,
            "use_cache": True,
            "do_sample": False,
        }
    
    @staticmethod
    def optimize_memory_usage():
        """Apply memory optimization techniques for M4."""
        if torch.backends.mps.is_available():
            # Enable memory efficient attention if available
            try:
                torch.backends.mps.enable_memory_efficient_attention()
                logger.info("Enabled memory efficient attention for MPS")
            except:
                logger.info("Memory efficient attention not available")
        
        # Note: Not setting MPS watermark ratios to avoid compatibility issues
        # PyTorch will use default memory management
        logger.info("Using default MPS memory management")
    
    @staticmethod
    def check_m4_compatibility() -> Dict[str, Any]:
        """Check M4 compatibility and capabilities."""
        compatibility = {
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "device_count": 1 if torch.backends.mps.is_available() else 0,
            "current_device": None,
            "memory_info": None,
        }
        
        if compatibility["mps_available"]:
            device = torch.device("mps")
            compatibility["current_device"] = str(device)
            
            # Get memory info if available
            try:
                if hasattr(torch.mps, 'get_device_properties'):
                    props = torch.mps.get_device_properties(device)
                    compatibility["memory_info"] = {
                        "total_memory": getattr(props, 'total_memory', 'Unknown'),
                        "max_threads_per_block": getattr(props, 'max_threads_per_block', 'Unknown'),
                    }
            except:
                compatibility["memory_info"] = "Memory info not available"
        
        return compatibility


def get_m4_optimized_training_args(**kwargs) -> Dict[str, Any]:
    """
    Get training arguments optimized for M4 chip.
    
    Args:
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Optimized training arguments
    """
    base_config = M4Optimizer.get_training_config()
    base_config.update(kwargs)
    return base_config


def setup_m4_environment():
    """Setup optimal environment for M4 training."""
    # Apply memory optimizations
    M4Optimizer.optimize_memory_usage()
    
    # Check compatibility
    compatibility = M4Optimizer.check_m4_compatibility()
    logger.info(f"M4 compatibility check: {compatibility}")
    
    if not compatibility["mps_available"]:
        logger.warning("MPS not available. Training will use CPU.")
    
    return compatibility
