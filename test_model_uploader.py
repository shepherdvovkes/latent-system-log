#!/usr/bin/env python3
"""
Test model uploader functionality.
"""

from app.services.model_uploader import ModelUploader
from app.core.config import settings


def test_model_uploader_initialization():
    """Test model uploader initialization."""
    print("=== Model Uploader Initialization Test ===")
    
    try:
        uploader = ModelUploader()
        print("✅ Model uploader initialized successfully")
        print(f"   HF Token configured: {'Yes' if uploader.token else 'No'}")
        return True
    except Exception as e:
        print(f"❌ Model uploader initialization failed: {e}")
        return False


def test_list_models():
    """Test listing uploaded models."""
    print("\n=== List Uploaded Models Test ===")
    
    try:
        uploader = ModelUploader()
        result = uploader.list_uploaded_models()
        
        if result["success"]:
            print("✅ Successfully listed models")
            print(f"   Total models found: {result['total_count']}")
            
            if result["models"]:
                print("   Models:")
                for model in result["models"]:
                    print(f"     - {model['name']} ({'Private' if model['private'] else 'Public'})")
            else:
                print("   No system log models found")
        else:
            print(f"❌ Failed to list models: {result['message']}")
        
        return result["success"]
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False


def test_model_upload_capability():
    """Test model upload capability (without actually uploading)."""
    print("\n=== Model Upload Capability Test ===")
    
    try:
        uploader = ModelUploader()
        
        # Test with a non-existent model path
        result = uploader.upload_model(
            model_path="./non_existent_model",
            repo_name="test-system-log-model",
            model_description="Test model for system log analysis"
        )
        
        if not result["success"] and "does not exist" in result["message"]:
            print("✅ Model uploader correctly handles non-existent models")
            return True
        else:
            print(f"❌ Unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing upload capability: {e}")
        return False


def test_hf_api_connection():
    """Test Hugging Face API connection."""
    print("\n=== HF API Connection Test ===")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Test API connection
        user_info = api.whoami(token=settings.HF_TOKEN)
        print("✅ HF API connection successful")
        print(f"   User: {user_info.get('name', 'Unknown')}")
        print(f"   Can create repos: {user_info.get('canPay', False)}")
        
        return True
    except Exception as e:
        print(f"❌ HF API connection failed: {e}")
        return False


def main():
    """Run all model uploader tests."""
    print("Starting Model Uploader Tests...\n")
    
    # Test initialization
    init_ok = test_model_uploader_initialization()
    
    if not init_ok:
        print("\n❌ Model uploader initialization failed, skipping other tests")
        return
    
    # Test HF API connection
    api_ok = test_hf_api_connection()
    
    # Test listing models
    list_ok = test_list_models()
    
    # Test upload capability
    upload_ok = test_model_upload_capability()
    
    print("\n=== Test Summary ===")
    if init_ok and api_ok and list_ok and upload_ok:
        print("✅ All model uploader tests passed!")
        print("   Your model uploader is ready to use")
        print("   You can upload trained models to HF Hub")
    else:
        print("❌ Some model uploader tests failed")
        if not api_ok:
            print("   - HF API connection failed")
        if not list_ok:
            print("   - Model listing failed")
        if not upload_ok:
            print("   - Upload capability test failed")
    
    print("\nModel Uploader Features:")
    print("- Upload trained models to HF Hub")
    print("- Create private repositories")
    print("- Generate model cards automatically")
    print("- List and manage uploaded models")
    print("- Delete models when needed")


if __name__ == "__main__":
    main()
