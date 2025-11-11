#!/usr/bin/env python3
"""Extract disease severity variables and cross-match with mediation data"""

import pandas as pd
from pathlib import Path

def extract_disease_severity_variables():
    """Extract the specific disease severity variables from neuropsychology data"""
    print("Extracting disease severity variables...")
    
    # Load the neuropsychology data
    neuro_file = Path.home() / 'Downloads' / 'Dietta Chihade_BD_RPQ_UPDATE_Neuropsy.xlsx - Parkinson patients.csv'
    
    if not neuro_file.exists():
        print("Neuropsychology file not found")
        return None
    
    # Read the neuropsychology data
    df = pd.read_csv(neuro_file)
    print(f"Neuropsychology data shape: {df.shape}")
    
    # Find the data rows (skip header rows)
    pd_rows = []
    for i, row in df.iterrows():
        row_str = ' '.join(str(val) for val in row.values if pd.notna(val))
        if 'PD' in row_str:
            pd_rows.append(i)
    
    if not pd_rows:
        print("No PD IDs found in neuropsychology data")
        return None
    
    # Extract data starting from the first PD row
    data_df = df.iloc[pd_rows[0]:].copy()
    
    # Based on the image, we need to extract these specific variables:
    # 1. Disease duration since first symptom
    # 2. Disease duration since diagnosis  
    # 3. Hoehn and Yahr score
    # 4. Patient cognitive complaint
    
    # Map the columns based on the structure we found earlier
    # Column 2: Disease duration since first symptom
    # Column 3: Disease duration since diagnosis
    # Column 4: Hoehn and Yahr score
    # Column 5: Patient cognitive complaint (or similar)
    
    # Create a clean dataframe with these variables
    severity_df = pd.DataFrame()
    
    # ID column (column 10 based on our previous analysis)
    severity_df['ID'] = data_df.iloc[:, 10]  # Assuming ID is in column 10
    
    # Disease duration since first symptom (column 2)
    severity_df['disease_duration_symptoms'] = pd.to_numeric(data_df.iloc[:, 2], errors='coerce')
    
    # Disease duration since diagnosis (column 3)
    severity_df['disease_duration_diagnosis'] = pd.to_numeric(data_df.iloc[:, 3], errors='coerce')
    
    # Hoehn and Yahr score (column 4)
    severity_df['hoehn_yahr_score'] = pd.to_numeric(data_df.iloc[:, 4], errors='coerce')
    
    # Patient cognitive complaint (column 5) - keep as string for now
    severity_df['patient_cognitive_complaint'] = data_df.iloc[:, 5]
    
    # Clean the data
    severity_df = severity_df[severity_df['ID'] != 'Patient #'].copy()
    severity_df['ID'] = severity_df['ID'].astype(str).str.strip()
    
    print(f"Extracted disease severity data shape: {severity_df.shape}")
    print("Columns:", severity_df.columns.tolist())
    
    # Show sample data
    print("\nSample of extracted disease severity data:")
    print(severity_df.head(10))
    
    # Check data quality
    print("\nData quality check:")
    for col in severity_df.columns:
        if col != 'ID':
            missing = severity_df[col].isna().sum()
            total = len(severity_df)
            print(f"  {col}: {total - missing}/{total} participants have data ({missing} missing)")
    
    return severity_df

def cross_match_with_mediation_data(severity_df):
    """Cross-match disease severity data with mediation data"""
    print("\nCross-matching with mediation data...")
    
    # Load the mediation data
    mediation_df = pd.read_csv('final_clean_mediation_data.csv')
    print(f"Mediation data shape: {mediation_df.shape}")
    
    # Get IDs from both datasets
    severity_ids = set(severity_df['ID'].tolist())
    mediation_ids = set(mediation_df['ID'].tolist())
    
    print(f"Severity data IDs: {len(severity_ids)}")
    print(f"Mediation data IDs: {len(mediation_ids)}")
    
    # Find overlapping IDs
    common_ids = severity_ids & mediation_ids
    print(f"Common IDs: {len(common_ids)}")
    
    # Create cross-matched dataset
    matched_severity = severity_df[severity_df['ID'].isin(common_ids)].copy()
    matched_mediation = mediation_df[mediation_df['ID'].isin(common_ids)].copy()
    
    # Merge the datasets
    cross_matched_df = matched_mediation.merge(matched_severity, on='ID', how='left', suffixes=('_current', '_extracted'))
    
    print(f"Cross-matched data shape: {cross_matched_df.shape}")
    
    # Compare the disease severity variables
    print("\nComparing disease severity variables:")
    severity_vars = ['disease_duration_symptoms', 'disease_duration_diagnosis', 'hoehn_yahr_score']
    
    for var in severity_vars:
        current_col = var + '_current'
        extracted_col = var + '_extracted'
        
        if current_col in cross_matched_df.columns and extracted_col in cross_matched_df.columns:
            # Check for matches
            matches = (cross_matched_df[current_col] == cross_matched_df[extracted_col]).sum()
            total = len(cross_matched_df)
            print(f"  {var}: {matches}/{total} participants have matching data")
            
            # Show differences
            differences = cross_matched_df[current_col != cross_matched_df[extracted_col]]
            if len(differences) > 0:
                print(f"    Differences found for {len(differences)} participants")
    
    # Save cross-matched data
    cross_matched_df.to_csv('cross_matched_disease_severity.csv', index=False)
    print(f"\nSaved cross-matched data: cross_matched_disease_severity.csv")
    
    # Show sample of cross-matched data
    print("\nSample of cross-matched data:")
    sample_cols = ['ID', 'Age', 'disease_duration_symptoms_current', 'disease_duration_symptoms_extracted', 
                   'disease_duration_diagnosis_current', 'disease_duration_diagnosis_extracted',
                   'hoehn_yahr_score_current', 'hoehn_yahr_score_extracted', 'patient_cognitive_complaint']
    available_cols = [col for col in sample_cols if col in cross_matched_df.columns]
    print(cross_matched_df[available_cols].head(10))
    
    return cross_matched_df

def main():
    """Main function to extract and cross-match disease severity data"""
    # Extract disease severity variables
    severity_df = extract_disease_severity_variables()
    
    if severity_df is not None:
        # Cross-match with mediation data
        cross_matched_df = cross_match_with_mediation_data(severity_df)
        
        # Summary
        print(f"\nFinal Summary:")
        print(f"- Extracted disease severity data: {len(severity_df)} participants")
        print(f"- Cross-matched with mediation data: {len(cross_matched_df)} participants")
        print(f"- Variables extracted:")
        print(f"  1. Disease duration since first symptom")
        print(f"  2. Disease duration since diagnosis")
        print(f"  3. Hoehn and Yahr score")
        print(f"  4. Patient cognitive complaint")
        
        return cross_matched_df
    else:
        print("Failed to extract disease severity data")
        return None

if __name__ == '__main__':
    cross_matched_df = main()

