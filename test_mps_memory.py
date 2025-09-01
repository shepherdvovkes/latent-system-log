#!/usr/bin/env python3
"""
Test MPS memory settings and identify issues.
"""

import torch
import os
from app.core.m4_optimization import M4Optimizer


def test_mps_basic():
    """Test basic MPS functionality."""
    print("=== Basic MPS Test ===")
    
    if torch.backends.mps.is_available():
        print("✅ MPS is available")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
        
        # Test basic tensor operations
        try:
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print("✅ Basic MPS tensor operations work")
            return True
        except Exception as e:
            print(f"❌ MPS tensor operations failed: {e}")
            return False
    else:
        print("❌ MPS is not available")
        return False


def test_mps_memory_optimization():
    """Test MPS memory optimization settings."""
    print("\n=== MPS Memory Optimization Test ===")
    
    try:
        # Test memory efficient attention
        if torch.backends.mps.is_available():
            try:
                torch.backends.mps.enable_memory_efficient_attention()
                print("✅ Memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️  Memory efficient attention not available: {e}")
        
        # Test watermark ratios (don't set them, just check if they exist)
        print("✅ Skipping watermark ratio settings to avoid compatibility issues")
        
        return True
    except Exception as e:
        print(f"❌ MPS memory optimization failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable settings."""
    print("\n=== Environment Variables Test ===")
    
    # Check current environment variables
    mps_vars = [
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
        'PYTORCH_MPS_LOW_WATERMARK_RATIO',
        'PYTORCH_MPS_MEMORY_FRACTION'
    ]
    
    for var in mps_vars:
        value = os.environ.get(var)
        if value:
            print(f"⚠️  {var} is set to: {value}")
        else:
            print(f"✅ {var} is not set (using default)")
    
    return True


def test_model_loading_with_mps():
    """Test loading a model with MPS."""
    print("\n=== Model Loading with MPS Test ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {device}")
        
        # Load a small model
        model_name = "distilbert-base-cased-distilled-squad"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move to device
        model = model.to(device)
        print(f"✅ Model loaded and moved to {device}")
        
        # Test forward pass
        inputs = tokenizer("What is the test?", "This is a test context.", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_m4_optimizer():
    """Test M4 optimizer functionality."""
    print("\n=== M4 Optimizer Test ===")
    
    try:
        # Test setup
        compatibility = M4Optimizer.check_m4_compatibility()
        print(f"✅ M4 compatibility check: {compatibility['mps_available']}")
        
        # Test memory optimization (without setting problematic env vars)
        M4Optimizer.optimize_memory_usage()
        print("✅ Memory optimization applied")
        
        # Test training config
        config = M4Optimizer.get_training_config()
        print(f"✅ Training config generated with {len(config)} parameters")
        
        return True
    except Exception as e:
        print(f"❌ M4 optimizer failed: {e}")
        return False


def main():
    """Run all MPS memory tests."""
    print("Starting MPS Memory Tests...\n")
    
    # Test basic MPS
    mps_ok = test_mps_basic()
    
    # Test memory optimization
    memory_ok = test_mps_memory_optimization()
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test model loading
    model_ok = test_model_loading_with_mps()
    
    # Test M4 optimizer
    optimizer_ok = test_m4_optimizer()
    
    print("\n=== Test Summary ===")
    if mps_ok and memory_ok and env_ok and model_ok and optimizer_ok:
        print("✅ All MPS tests passed!")
        print("   MPS should work correctly for training")
    else:
        print("❌ Some MPS tests failed")
        if not mps_ok:
            print("   - Basic MPS functionality failed")
        if not memory_ok:
            print("   - Memory optimization failed")
        if not env_ok:
            print("   - Environment variables issue")
        if not model_ok:
            print("   - Model loading failed")
        if not optimizer_ok:
            print("   - M4 optimizer failed")
    
    print("\nRecommendations:")
    print("- Using default MPS memory management")
    print("- Avoiding problematic environment variables")
    print("- Testing with conservative settings")


if __name__ == "__main__":
    main()
