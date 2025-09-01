#!/usr/bin/env python3
"""
Test script to verify M4 neural chip optimization.
"""

import torch
import time
from app.core.m4_optimization import M4Optimizer, setup_m4_environment


def test_m4_compatibility():
    """Test M4 compatibility and capabilities."""
    print("=== M4 Neural Chip Compatibility Test ===")
    
    # Setup M4 environment
    compatibility = setup_m4_environment()
    
    print(f"MPS Available: {compatibility['mps_available']}")
    print(f"MPS Built: {compatibility['mps_built']}")
    print(f"Device Count: {compatibility['device_count']}")
    print(f"Current Device: {compatibility['current_device']}")
    print(f"Memory Info: {compatibility['memory_info']}")
    
    return compatibility


def test_device_performance():
    """Test device performance with a simple tensor operation."""
    print("\n=== Device Performance Test ===")
    
    # Test CPU
    print("Testing CPU performance...")
    start_time = time.time()
    x_cpu = torch.randn(1000, 1000)
    y_cpu = torch.randn(1000, 1000)
    result_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Matrix multiplication time: {cpu_time:.4f} seconds")
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        print("Testing MPS performance...")
        device = torch.device("mps")
        
        start_time = time.time()
        x_mps = torch.randn(1000, 1000, device=device)
        y_mps = torch.randn(1000, 1000, device=device)
        result_mps = torch.mm(x_mps, y_mps)
        # Synchronize to ensure computation is complete
        torch.mps.synchronize()
        mps_time = time.time() - start_time
        print(f"MPS Matrix multiplication time: {mps_time:.4f} seconds")
        
        speedup = cpu_time / mps_time if mps_time > 0 else 0
        print(f"MPS speedup: {speedup:.2f}x")
    else:
        print("MPS not available for testing")


def test_optimal_batch_sizes():
    """Test optimal batch size calculations."""
    print("\n=== Optimal Batch Size Test ===")
    
    model_sizes = [25, 100, 300, 800]
    
    for size in model_sizes:
        optimal_batch = M4Optimizer.get_optimal_batch_size(size)
        print(f"Model size {size}MB -> Optimal batch size: {optimal_batch}")


def test_training_config():
    """Test M4-optimized training configuration."""
    print("\n=== M4 Training Configuration Test ===")
    
    config = M4Optimizer.get_training_config()
    
    print("M4-optimized training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def test_inference_config():
    """Test M4-optimized inference configuration."""
    print("\n=== M4 Inference Configuration Test ===")
    
    config = M4Optimizer.get_inference_config()
    
    print("M4-optimized inference configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def main():
    """Run all M4 optimization tests."""
    print("Starting M4 Neural Chip Optimization Tests...\n")
    
    # Run tests
    test_m4_compatibility()
    test_device_performance()
    test_optimal_batch_sizes()
    test_training_config()
    test_inference_config()
    
    print("\n=== Test Summary ===")
    print("M4 optimization tests completed!")
    print("If MPS is available, your system is ready for M4-optimized training.")
    print("If MPS is not available, training will fall back to CPU.")


if __name__ == "__main__":
    main()
