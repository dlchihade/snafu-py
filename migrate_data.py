#!/usr/bin/env python3
"""
Migration script to extract data from the original mediation script
"""

import pandas as pd
import re
import io
import yaml
from pathlib import Path

def extract_fluency_data(script_path):
    """Extract fluency data from original script"""
    print("Extracting fluency data...")
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the data string pattern
    pattern = r'data_str = \'\'\'(.*?)\'\'\''
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        data_str = match.group(1)
        
        # Parse the data
        lines = data_str.strip().split('\n')
        data_rows = []
        
        for line in lines:
            if line.strip() and 'ID,Item' not in line:
                # Parse each participant's data
                if 'PD' in line and ',"' in line:
                    # Extract ID and items
                    id_part = line.split(',"')[0].strip()
                    items_part = line.split(',"')[1].strip().rstrip('"')
                    
                    # Split items and create rows
                    items = [item.strip() for item in items_part.split(',')]
                    
                    for item in items:
                        if item:  # Skip empty items
                            data_rows.append({
                                'ID': id_part,
                                'Item': item
                            })
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Save to CSV
        output_path = Path('semantic_fluency_analysis/data/fluency_data.csv')
        df.to_csv(output_path, index=False)
        
        print(f"✅ Extracted fluency data: {len(df)} rows, {df['ID'].nunique()} participants")
        print(f"   Saved to: {output_path}")
        
        return df
    else:
        print("❌ Could not find fluency data in script")
        return None

def is_valid_float(value):
    """Check if a string can be converted to float"""
    if value == 'nan' or not value:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

def extract_meg_data(script_path):
    """Extract MEG data from original script"""
    print("Extracting MEG data...")
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the MEG data string pattern
    pattern = r'meg_lc_data = pd\.read_csv\(io\.StringIO\(\'\'\'(.*?)\'\'\'\)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        meg_data_str = match.group(1)
        
        # Parse the MEG data
        lines = meg_data_str.strip().split('\n')
        data_rows = []
        
        for line in lines:
            if line.strip() and 'PD' in line:
                # Split by comma and clean up
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 9:  # Ensure we have enough columns
                    try:
                        data_rows.append({
                            'ID': parts[0],
                            'alpha_NET_mean': float(parts[1]) if is_valid_float(parts[1]) else None,
                            'status': parts[2],
                            'visit': parts[3],
                            'norm_SN_l': float(parts[4]) if is_valid_float(parts[4]) else None,
                            'norm_SN_r': float(parts[5]) if is_valid_float(parts[5]) else None,
                            'norm_LC_l': float(parts[6]) if is_valid_float(parts[6]) else None,
                            'norm_LC_r': float(parts[7]) if is_valid_float(parts[7]) else None,
                            'norm_SN_avg': float(parts[8]) if is_valid_float(parts[8]) else None,
                            'norm_LC_avg': float(parts[9]) if len(parts) > 9 and is_valid_float(parts[9]) else None
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed line: {line.strip()} - Error: {e}")
                        continue
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Save to CSV
        output_path = Path('semantic_fluency_analysis/data/meg_data.csv')
        df.to_csv(output_path, index=False)
        
        print(f"✅ Extracted MEG data: {len(df)} participants")
        print(f"   Saved to: {output_path}")
        
        return df
    else:
        print("❌ Could not find MEG data in script")
        return None

def create_config_file():
    """Create configuration file"""
    print("Creating configuration file...")
    
    config_data = {
        'semantic_weight': 0.7,
        'clustering_threshold': 0.5,
        'similarity_threshold': None,  # Will be calculated from data
        'cache_size': 1000,
        'output_dir': 'output',
        'save_plots': True,
        'plot_format': 'svg',
        'data_paths': {
            'fluency_data': 'data/fluency_data.csv',
            'meg_data': 'data/meg_data.csv'
        }
    }
    
    output_path = Path('semantic_fluency_analysis/config/config.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created configuration file: {output_path}")

def create_requirements_file():
    """Create requirements.txt file"""
    print("Creating requirements file...")
    
    requirements = [
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'spacy>=3.0.0',
        'wordfreq>=2.4.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'statsmodels>=0.13.0',
        'pyyaml>=6.0'
    ]
    
    output_path = Path('semantic_fluency_analysis/requirements.txt')
    with open(output_path, 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"✅ Created requirements file: {output_path}")

def create_readme():
    """Create README file"""
    print("Creating README file...")
    
    readme_content = """# Semantic Fluency Analysis System

## Overview
This is a refactored, production-ready system for analyzing semantic verbal fluency data with exploitation/exploration patterns.

## Features
- Semantic similarity analysis using spaCy word vectors
- Phase identification (exploitation vs exploration)
- Word frequency analysis
- Comprehensive visualization suite
- Statistical analysis and correlation testing
- MEG/LC data integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_md
```

## Usage

```python
from src.pipeline import SemanticFluencyPipeline

# Initialize pipeline
pipeline = SemanticFluencyPipeline('config/config.yaml')

# Run analysis
results_df = pipeline.run_analysis(
    fluency_path='data/fluency_data.csv',
    meg_path='data/meg_data.csv'
)

# Generate visualizations
pipeline.generate_visualizations(results_df)

# Save results
pipeline.save_results(results_df)
```

## Project Structure
```
semantic_fluency_analysis/
├── config/          # Configuration files
├── data/           # Data files
├── src/            # Source code
├── tests/          # Unit tests
├── output/         # Results and plots
└── requirements.txt
```

## Configuration
Edit `config/config.yaml` to modify analysis parameters.

## Testing
Run tests with:
```bash
python -m pytest tests/
```
"""
    
    output_path = Path('semantic_fluency_analysis/README.md')
    with open(output_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README file: {output_path}")

def main():
    """Main migration function"""
    script_path = "/Users/diettachihade/Downloads/mediation_with_new_scores_022_02_mvt (3).py"
    
    print("🚀 Starting migration process...")
    print("=" * 50)
    
    # Check if script exists
    if not Path(script_path).exists():
        print(f"❌ Script not found: {script_path}")
        print("Please update the script_path variable in this file")
        return
    
    # Extract data
    fluency_df = extract_fluency_data(script_path)
    meg_df = extract_meg_data(script_path)
    
    # Create project files
    create_config_file()
    create_requirements_file()
    create_readme()
    
    print("=" * 50)
    print("✅ Migration completed!")
    print("\nNext steps:")
    print("1. cd semantic_fluency_analysis")
    print("2. pip install -r requirements.txt")
    print("3. python -m spacy download en_core_web_md")
    print("4. Start implementing the refactored classes")

if __name__ == "__main__":
    main()
