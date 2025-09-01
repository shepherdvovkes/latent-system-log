#!/usr/bin/env python3
"""
Simple test for model trainer without memory optimization issues.
"""

import asyncio
from app.services.model_trainer import ModelTrainerService
from app.models.schemas import LogEntry


async def test_model_trainer_initialization():
    """Test model trainer initialization."""
    print("=== Model Trainer Initialization Test ===")
    
    try:
        trainer = ModelTrainerService()
        print("✅ Model trainer initialized successfully")
        print(f"   Device: {trainer.device}")
        print(f"   Is trained: {trainer.is_trained}")
        return True
    except Exception as e:
        print(f"❌ Model trainer initialization failed: {e}")
        return False


async def test_training_data_generation():
    """Test training data generation."""
    print("\n=== Training Data Generation Test ===")
    
    try:
        trainer = ModelTrainerService()
        
        # Create some sample logs
        sample_logs = [
            LogEntry(
                id=1,
                timestamp="2024-01-01T10:00:00",
                level="INFO",
                source="system",
                message="System started successfully",
                raw_message="System started successfully"
            ),
            LogEntry(
                id=2,
                timestamp="2024-01-01T10:01:00",
                level="WARNING",
                source="security",
                message="Security alert: suspicious activity detected",
                raw_message="Security alert: suspicious activity detected"
            ),
            LogEntry(
                id=3,
                timestamp="2024-01-01T10:02:00",
                level="ERROR",
                source="performance",
                message="High CPU usage detected: 95%",
                raw_message="High CPU usage detected: 95%"
            )
        ]
        
        # Generate training data
        training_data = await trainer._generate_training_data(sample_logs)
        
        print(f"✅ Generated {len(training_data)} training samples")
        
        if training_data:
            print("   Sample training data:")
            for i, sample in enumerate(training_data[:2]):
                print(f"     {i+1}. Q: {sample['question']}")
                print(f"        A: {sample['answer'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Training data generation failed: {e}")
        return False


async def test_model_initialization():
    """Test model initialization."""
    print("\n=== Model Initialization Test ===")
    
    try:
        trainer = ModelTrainerService()
        
        # Initialize model
        await trainer._initialize_model()
        
        print("✅ Model initialized successfully")
        print(f"   Model device: {trainer.model.device}")
        print(f"   Tokenizer vocab size: {trainer.tokenizer.vocab_size}")
        
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False


async def test_minimal_training():
    """Test minimal training setup."""
    print("\n=== Minimal Training Setup Test ===")
    
    try:
        from app.services.minimal_trainer import MinimalModelTrainer
        
        trainer = MinimalModelTrainer()
        print("✅ Minimal trainer initialized")
        
        # Test with sample logs
        sample_logs = [
            LogEntry(
                id=1,
                timestamp="2024-01-01T10:00:00",
                level="INFO",
                source="system",
                message="System started successfully",
                raw_message="System started successfully"
            )
        ]
        
        # Test training data generation
        training_data = await trainer._generate_training_data(sample_logs)
        print(f"✅ Generated {len(training_data)} training samples")
        
        return True
    except Exception as e:
        print(f"❌ Minimal training setup failed: {e}")
        return False


async def main():
    """Run all model trainer tests."""
    print("Starting Model Trainer Tests...\n")
    
    # Test initialization
    init_ok = await test_model_trainer_initialization()
    
    # Test training data generation
    data_ok = await test_training_data_generation()
    
    # Test model initialization
    model_ok = await test_model_initialization()
    
    # Test minimal trainer
    minimal_ok = await test_minimal_training()
    
    print("\n=== Test Summary ===")
    if init_ok and data_ok and model_ok and minimal_ok:
        print("✅ All model trainer tests passed!")
        print("   Model trainer should work correctly")
        print("   Ready for actual training")
    else:
        print("❌ Some model trainer tests failed")
        if not init_ok:
            print("   - Initialization failed")
        if not data_ok:
            print("   - Training data generation failed")
        if not model_ok:
            print("   - Model initialization failed")
        if not minimal_ok:
            print("   - Minimal trainer failed")
    
    print("\nNext Steps:")
    print("- Try training with a small dataset")
    print("- Monitor logs for any errors")
    print("- Use minimal trainer as fallback if needed")


if __name__ == "__main__":
    asyncio.run(main())
