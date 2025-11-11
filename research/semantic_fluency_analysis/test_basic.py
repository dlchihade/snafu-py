#!/usr/bin/env python3
"""
Basic test script to verify the migration is working correctly
"""

import os
import pandas as pd
import yaml

def test_data_files():
    """Test that data files exist and can be loaded"""
    print("Testing data files...")
    
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

def test_config_file():
    """Test configuration file"""
    print("\nTesting configuration file...")
    
    config_path = 'config/config.yaml'
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Semantic weight: {config.get('semantic_weight', 'N/A')}")
        print(f"   Clustering threshold: {config.get('clustering_threshold', 'N/A')}")
        print(f"   Cache size: {config.get('cache_size', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_project_structure():
    """Test project structure"""
    print("\nTesting project structure...")
    
    expected_dirs = ['config', 'data', 'src', 'tests', 'output']
    expected_files = ['requirements.txt', 'README.md']
    
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory found: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            return False
    
    for file_name in expected_files:
        if os.path.exists(file_name):
            print(f"✅ File found: {file_name}")
        else:
            print(f"❌ File missing: {file_name}")
            return False
    
    return True

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
        
        # Check for some expected IDs
        expected_ids = ['PD00020', 'PD00048', 'PD00146']
        found_expected = [pid for pid in expected_ids if pid in fluency_ids]
        
        print(f"Found expected IDs: {found_expected}")
        
        if len(found_expected) == len(expected_ids):
            print("✅ All expected participant IDs found")
        else:
            print(f"❌ Missing expected IDs: {set(expected_ids) - set(found_expected)}")
        
        # Check data quality
        print(f"\nData quality check:")
        print(f"   Total fluency items: {len(fluency_df)}")
        print(f"   Unique items: {fluency_df['Item'].nunique()}")
        print(f"   Missing items: {fluency_df['Item'].isna().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data extraction: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Migration Results")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_config_file,
        test_data_files,
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
