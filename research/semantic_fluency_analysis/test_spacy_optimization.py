#!/usr/bin/env python3
"""
Test script for optimized spaCy implementation
"""

import sys
import os
import pandas as pd
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.utils import SpacyOptimizer
from src.analyzer import SemanticFluencyAnalyzer

def test_spacy_optimizer():
    """Test the SpacyOptimizer class"""
    print("🧪 Testing SpacyOptimizer")
    print("=" * 40)
    
    try:
        # Initialize optimizer
        optimizer = SpacyOptimizer(model_name="en_core_web_md", batch_size=100)
        print("✅ SpacyOptimizer initialized successfully")
        
        # Test word coverage
        test_words = ['cat', 'dog', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey']
        coverage = optimizer.validate_word_coverage(test_words)
        print(f"✅ Word coverage test: {coverage['coverage_percentage']:.1f}% coverage")
        
        # Test consecutive similarities
        similarities, valid_words = optimizer.get_consecutive_similarities(test_words)
        print(f"✅ Consecutive similarities: {len(similarities)} similarities for {len(valid_words)} words")
        
        # Test word similarity
        sim = optimizer.get_word_similarity('cat', 'dog')
        print(f"✅ Word similarity (cat-dog): {sim:.3f}")
        
        # Test most similar words
        similar = optimizer.get_most_similar_words('cat', ['dog', 'lion', 'tiger', 'elephant'], top_k=3)
        print(f"✅ Most similar to 'cat': {similar}")
        
        return True
        
    except Exception as e:
        print(f"❌ SpacyOptimizer test failed: {e}")
        return False

def test_analyzer_with_spacy():
    """Test the analyzer with optimized spaCy"""
    print("\n🧪 Testing Analyzer with SpacyOptimizer")
    print("=" * 40)
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        print("✅ Configuration loaded")
        
        # Initialize analyzer
        analyzer = SemanticFluencyAnalyzer(config)
        print("✅ Analyzer initialized")
        
        # Load data
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        print("✅ Data loaded")
        
        # Test with a few participants
        test_participants = analyzer.data['ID'].unique()[:3]
        print(f"Testing with participants: {test_participants}")
        
        results = []
        for participant_id in test_participants:
            participant_data = analyzer.data[analyzer.data['ID'] == participant_id]
            result = analyzer.analyze_participant(participant_data)
            results.append(result)
            
            print(f"   {participant_id}: {result['num_valid_items']} valid items, "
                  f"{result['exploitation_percentage']:.1f}% exploitation, "
                  f"{result['word_coverage']:.1f}% coverage")
        
        # Check results
        avg_coverage = np.mean([r['word_coverage'] for r in results])
        avg_exploitation = np.mean([r['exploitation_percentage'] for r in results])
        
        print(f"\n📊 Summary:")
        print(f"   Average word coverage: {avg_coverage:.1f}%")
        print(f"   Average exploitation: {avg_exploitation:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return False

def test_performance():
    """Test performance improvements"""
    print("\n🧪 Testing Performance")
    print("=" * 40)
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        
        # Initialize analyzer
        analyzer = SemanticFluencyAnalyzer(config)
        analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        
        # Test with first 5 participants
        test_participants = analyzer.data['ID'].unique()[:5]
        
        # Time the analysis
        start_time = time.time()
        
        results = []
        for participant_id in test_participants:
            participant_data = analyzer.data[analyzer.data['ID'] == participant_id]
            result = analyzer.analyze_participant(participant_data)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Processed {len(test_participants)} participants in {total_time:.2f} seconds")
        print(f"   Average time per participant: {total_time/len(test_participants):.2f} seconds")
        
        # Check word coverage
        total_items = sum(r['num_items'] for r in results)
        total_valid = sum(r['num_valid_items'] for r in results)
        coverage = total_valid / total_items * 100 if total_items > 0 else 0
        
        print(f"   Total word coverage: {coverage:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_configuration():
    """Test the updated configuration"""
    print("\n🧪 Testing Configuration")
    print("=" * 40)
    
    try:
        # Load configuration
        config = AnalysisConfig.from_yaml('config/config.yaml')
        
        # Check new parameters
        print(f"✅ spaCy model: {config.spacy_model}")
        print(f"✅ Batch size: {config.batch_size}")
        print(f"✅ Similarity threshold: {config.similarity_threshold}")
        print(f"✅ Min similarity: {config.min_similarity}")
        print(f"✅ Max similarity: {config.max_similarity}")
        print(f"✅ Min phase length: {config.min_phase_length}")
        print(f"✅ Max phase length: {config.max_phase_length}")
        print(f"✅ Enable caching: {config.enable_caching}")
        print(f"✅ Enable batch processing: {config.enable_batch_processing}")
        
        # Validate configuration
        if config.validate():
            print("✅ Configuration validation passed")
            return True
        else:
            print("❌ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Optimized spaCy Implementation")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_spacy_optimizer,
        test_analyzer_with_spacy,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! spaCy optimization successful!")
        print("\nNext steps:")
        print("1. Run the full analysis: python main.py")
        print("2. Check output directory for results")
        print("3. Review analysis.log for detailed information")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
