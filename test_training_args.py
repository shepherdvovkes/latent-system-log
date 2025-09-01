#!/usr/bin/env python3
"""
Test training arguments compatibility.
"""

import torch
from transformers import TrainingArguments


def test_training_args():
    """Test that our training arguments work."""
    print("=== Testing Training Arguments ===")
    
    # Test the arguments we use in our model trainer
    try:
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=2e-5,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=False,
            eval_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_loss",
        )
        print("✅ Training arguments created successfully")
        print(f"   Output dir: {training_args.output_dir}")
        print(f"   Epochs: {training_args.num_train_epochs}")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Eval strategy: {training_args.eval_strategy}")
        print(f"   Save strategy: {training_args.save_strategy}")
        return True
    except Exception as e:
        print(f"❌ Training arguments failed: {e}")
        return False


def test_mps_device():
    """Test MPS device availability."""
    print("\n=== Testing MPS Device ===")
    
    if torch.backends.mps.is_available():
        print("✅ MPS is available")
        device = torch.device("mps")
        print(f"   Device: {device}")
        
        # Test basic tensor operations
        try:
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.mm(x, y)
            print("✅ MPS tensor operations work")
            return True
        except Exception as e:
            print(f"❌ MPS tensor operations failed: {e}")
            return False
    else:
        print("❌ MPS is not available")
        return False


def main():
    """Run all tests."""
    print("Starting Training Arguments Tests...\n")
    
    # Test training arguments
    args_ok = test_training_args()
    
    # Test MPS device
    mps_ok = test_mps_device()
    
    print("\n=== Test Summary ===")
    if args_ok and mps_ok:
        print("✅ All tests passed!")
        print("   Your model trainer should work correctly")
    else:
        print("❌ Some tests failed")
        if not args_ok:
            print("   - Training arguments need to be fixed")
        if not mps_ok:
            print("   - MPS device not available, will use CPU")


if __name__ == "__main__":
    main()
