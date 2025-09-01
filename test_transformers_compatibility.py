#!/usr/bin/env python3
"""
Test script to check transformers version compatibility.
"""

import transformers
from transformers import TrainingArguments
import torch


def test_transformers_version():
    """Test transformers version and compatibility."""
    print("=== Transformers Version Test ===")
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test basic TrainingArguments
    print("\n=== Basic TrainingArguments Test ===")
    try:
        basic_args = TrainingArguments(
            output_dir="./test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=10,
        )
        print("✅ Basic TrainingArguments created successfully")
    except Exception as e:
        print(f"❌ Basic TrainingArguments failed: {e}")
        return False
    
    # Test evaluation strategy parameters
    print("\n=== Evaluation Strategy Test ===")
    
    # Test eval_strategy (newer versions)
    try:
        eval_args = TrainingArguments(
            output_dir="./test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            eval_strategy="steps",
            eval_steps=100,
        )
        print("✅ eval_strategy parameter works")
        eval_strategy_supported = True
    except Exception as e:
        print(f"❌ eval_strategy parameter failed: {e}")
        eval_strategy_supported = False
    
    # Test evaluation_strategy (older versions)
    try:
        eval_args = TrainingArguments(
            output_dir="./test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            evaluation_strategy="steps",
            eval_steps=100,
        )
        print("✅ evaluation_strategy parameter works")
        evaluation_strategy_supported = True
    except Exception as e:
        print(f"❌ evaluation_strategy parameter failed: {e}")
        evaluation_strategy_supported = False
    
    # Test optional parameters
    print("\n=== Optional Parameters Test ===")
    
    optional_params = {
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "save_strategy": "steps",
        "save_steps": 100,
    }
    
    for param, value in optional_params.items():
        try:
            test_args = TrainingArguments(
                output_dir="./test",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                **{param: value}
            )
            print(f"✅ {param} parameter works")
        except Exception as e:
            print(f"❌ {param} parameter failed: {e}")
    
    # Test MPS/M4 specific parameters
    print("\n=== MPS/M4 Parameters Test ===")
    
    mps_params = {
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "fp16": False,
    }
    
    for param, value in mps_params.items():
        try:
            test_args = TrainingArguments(
                output_dir="./test",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                **{param: value}
            )
            print(f"✅ {param} parameter works")
        except Exception as e:
            print(f"❌ {param} parameter failed: {e}")
    
    return True


def test_mps_availability():
    """Test MPS availability."""
    print("\n=== MPS Availability Test ===")
    
    if torch.backends.mps.is_available():
        print("✅ MPS is available")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
        
        # Test basic MPS operations
        try:
            device = torch.device("mps")
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.mm(x, y)
            print("✅ Basic MPS operations work")
        except Exception as e:
            print(f"❌ MPS operations failed: {e}")
    else:
        print("❌ MPS is not available")
        print("   Training will use CPU")


def main():
    """Run all compatibility tests."""
    print("Starting Transformers Compatibility Tests...\n")
    
    # Test transformers version
    success = test_transformers_version()
    
    # Test MPS availability
    test_mps_availability()
    
    print("\n=== Test Summary ===")
    if success:
        print("✅ Transformers compatibility tests passed")
        print("   Your setup should work with the model trainer")
    else:
        print("❌ Transformers compatibility tests failed")
        print("   Please check your transformers installation")
    
    print("\nRecommendations:")
    print("- If you see any ❌ marks, those parameters will be skipped")
    print("- The model trainer will use only supported parameters")
    print("- Consider updating transformers if many parameters fail")


if __name__ == "__main__":
    main()
