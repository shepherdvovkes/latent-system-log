#!/usr/bin/env python3
"""
Test Hugging Face token functionality.
"""

import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from app.core.config import settings


def test_hf_token_config():
    """Test that the HF token is configured correctly."""
    print("=== Hugging Face Token Configuration Test ===")
    
    token = settings.HF_TOKEN
    if token:
        print(f"✅ HF Token found: {token[:10]}...{token[-10:]}")
        return True
    else:
        print("❌ No HF Token configured")
        return False


def test_model_download_with_token():
    """Test downloading a model with the HF token."""
    print("\n=== Model Download Test ===")
    
    token = settings.HF_TOKEN
    model_name = "distilbert-base-cased-distilled-squad"
    
    try:
        print(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("✅ Tokenizer downloaded successfully")
        
        print(f"Downloading model for {model_name}...")
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("✅ Model downloaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False


def test_embedding_model_download():
    """Test downloading an embedding model with the HF token."""
    print("\n=== Embedding Model Download Test ===")
    
    token = settings.HF_TOKEN
    model_name = "all-MiniLM-L6-v2"
    
    try:
        print(f"Downloading embedding model {model_name}...")
        model = SentenceTransformer(
            model_name,
            token=token,
            trust_remote_code=True
        )
        print("✅ Embedding model downloaded successfully")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✅ Encoding test successful, embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding model download failed: {e}")
        return False


def test_token_authentication():
    """Test HF token authentication."""
    print("\n=== Token Authentication Test ===")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Test if token is valid by trying to get user info
        user_info = api.whoami(token=settings.HF_TOKEN)
        print(f"✅ Token authentication successful")
        print(f"   User: {user_info.get('name', 'Unknown')}")
        print(f"   Email: {user_info.get('email', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Token authentication failed: {e}")
        return False


def main():
    """Run all HF token tests."""
    print("Starting Hugging Face Token Tests...\n")
    
    # Test configuration
    config_ok = test_hf_token_config()
    
    if not config_ok:
        print("\n❌ HF Token not configured, skipping other tests")
        return
    
    # Test authentication
    auth_ok = test_token_authentication()
    
    # Test model downloads
    model_ok = test_model_download_with_token()
    embedding_ok = test_embedding_model_download()
    
    print("\n=== Test Summary ===")
    if config_ok and auth_ok and model_ok and embedding_ok:
        print("✅ All HF token tests passed!")
        print("   Your HF token is working correctly")
        print("   Models will be downloaded with authentication")
    else:
        print("❌ Some HF token tests failed")
        if not auth_ok:
            print("   - Token authentication failed")
        if not model_ok:
            print("   - Model download failed")
        if not embedding_ok:
            print("   - Embedding model download failed")
    
    print("\nBenefits of using HF token:")
    print("- Faster model downloads")
    print("- Access to private models")
    print("- Higher rate limits")
    print("- Better reliability")


if __name__ == "__main__":
    main()
