#!/usr/bin/env python3
"""
Search for appropriate models to fine-tune for system log analysis.
"""

from app.services.hf_model_searcher import HFModelSearcher


def main():
    """Search for models suitable for system log analysis."""
    print("🔍 Searching Hugging Face for models suitable for system log analysis...\n")
    
    searcher = HFModelSearcher()
    
    # 1. Get recommended models
    print("📋 Getting recommended models...")
    recommendations = searcher.get_recommended_models()
    
    if recommendations["success"]:
        for category, models in recommendations["recommendations"].items():
            print(f"\n🎯 {category.replace('_', ' ').title()}:")
            for model in models:
                print(f"   • {model['model_id']}")
                print(f"     Size: {model['size_mb']}MB | Downloads: {model['downloads']:,} | Score: {model['suitability_score']}")
                print(f"     URL: {model['url']}")
                print()
    
    # 2. Search for system log specific models
    print("🔍 Searching for system log specific models...")
    search_results = searcher.search_models(
        query="system log analysis question answering",
        task="question-answering",
        limit=10,
        sort="downloads"
    )
    
    if search_results["success"] and search_results["models"]:
        print(f"\n📊 Found {search_results['total_found']} suitable models:")
        for i, model in enumerate(search_results["models"][:5], 1):
            print(f"\n{i}. {model['model_id']}")
            print(f"   Size: {model['size_mb']}MB")
            print(f"   Downloads: {model['downloads']:,}")
            print(f"   Likes: {model['likes']}")
            print(f"   Suitability Score: {model['suitability_score']}")
            print(f"   Tags: {', '.join(model['tags'][:5])}")
            print(f"   URL: {model['url']}")
    
    # 3. Compare top models
    print("\n⚖️  Comparing top models...")
    top_models = [
        "distilbert-base-cased-distilled-squad",  # Current choice
        "deepset/roberta-base-squad2",
        "microsoft/DialoGPT-medium",
        "facebook/bart-base"
    ]
    
    comparison = searcher.compare_models(top_models)
    
    if comparison["success"] and comparison["comparison"]["models"]:
        print("\n📈 Model Comparison:")
        print("-" * 80)
        print(f"{'Model':<40} {'Size(MB)':<10} {'Downloads':<12} {'Score':<8}")
        print("-" * 80)
        
        for model in comparison["comparison"]["models"]:
            print(f"{model['model_id']:<40} {model['size_mb']:<10} {model['downloads']:<12,} {model['suitability_score']:<8}")
    
    # 4. Recommendations
    print("\n💡 Recommendations for your system log analysis project:")
    print("1. Start with 'distilbert-base-cased-distilled-squad' (current choice) - lightweight and proven")
    print("2. Consider 'deepset/roberta-base-squad2' for better performance if you have more resources")
    print("3. Try 'microsoft/DialoGPT-medium' for more conversational QA capabilities")
    print("4. Use 'facebook/bart-base' if you need text generation capabilities")
    
    print("\n🎯 Best choice for M4 fine-tuning:")
    print("   • distilbert-base-cased-distilled-squad (current)")
    print("   • Size: ~260MB")
    print("   • Fast training on M4")
    print("   • Good balance of performance and speed")


if __name__ == "__main__":
    main()
