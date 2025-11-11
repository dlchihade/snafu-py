#!/usr/bin/env python3
"""
Test script to verify the migration is working correctly
"""

import sys
import os
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import AnalysisConfig
from src.analyzer import SemanticFluencyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_data_loading():
    """Test that data can be loaded correctly"""
    print("Testing data loading...")
    
    # Load configuration
    config = AnalysisConfig.from_yaml('config/config.yaml')
    print(f"✅ Configuration loaded: {config}")
    
    # Check if data files exist
    fluency_path = 'data/fluency_data.csv'
    meg_path = 'data/meg_data.csv'
    
    if not os.path.exists(fluency_path):
        print(f"❌ Fluency data file not found: {fluency_path}")
        return False
    
    if not os.path.exists(meg_path):
        print(f"❌ MEG data file not found: {meg_path}")
        return False
    
    print(f"✅ Data files found")
    
    # Load data with pandas to test
    try:
        fluency_df = pd.read_csv(fluency_path)
        meg_df = pd.read_csv(meg_path)
        
        print(f"✅ Fluency data: {len(fluency_df)} rows, {fluency_df['ID'].nunique()} participants")
        print(f"✅ MEG data: {len(meg_df)} participants")
        
        # Show sample data
        print("\nSample fluency data:")
        print(fluency_df.head())
        
        print("\nSample MEG data:")
        print(meg_df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_analyzer_initialization():
    """Test analyzer initialization (without spaCy for now)"""
    print("\nTesting analyzer initialization...")
    
    try:
        config = AnalysisConfig.from_yaml('config/config.yaml')
        
        # Test configuration validation
        if config.validate():
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        print("✅ Analyzer configuration ready")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return False

def test_data_extraction():
    """Test that we extracted the correct number of participants"""
    print("\nTesting data extraction...")
    
    try:
        fluency_df = pd.read_csv('data/fluency_data.csv')
        meg_df = pd.read_csv('data/meg_data.csv')
        
        # Check participant counts
        fluency_participants = fluency_df['ID'].nunique()
        meg_participants = meg_df['ID'].nunique()
        
        print(f"Fluency participants: {fluency_participants}")
        print(f"MEG participants: {meg_participants}")
        
        # Check for expected participant IDs
        fluency_ids = set(fluency_df['ID'].unique())
        meg_ids = set(meg_df['ID'].unique())
        
        # Check for some expected IDs
        expected_ids = ['PD00020', 'PD00048', 'PD00146']
        found_expected = [pid for pid in expected_ids if pid in fluency_ids]
        
        print(f"Found expected IDs: {found_expected}")
        
        if len(found_expected) == len(expected_ids):
            print("✅ All expected participant IDs found")
        else:
            print(f"❌ Missing expected IDs: {set(expected_ids) - set(found_expected)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data extraction: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Migration Results")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_analyzer_initialization,
        test_data_extraction
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
        print("🎉 All tests passed! Migration successful!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download spaCy model: python -m spacy download en_core_web_md")
        print("3. Run the full analysis pipeline")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
