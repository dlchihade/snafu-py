#!/usr/bin/env python3
"""Clean the final comprehensive dataframe by removing duplicates"""

import pandas as pd

def clean_final_dataframe():
    """Clean the final comprehensive dataframe"""
    print("Cleaning final comprehensive dataframe...")
    
    # Load the final comprehensive data
    df = pd.read_csv('final_complete_mediation_data.csv')
    print(f"Original shape: {df.shape}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['ID'], keep=False)
    print(f"Duplicate rows: {duplicates.sum()}")
    
    if duplicates.sum() > 0:
        print("Duplicate IDs found:")
        duplicate_ids = df[duplicates]['ID'].value_counts()
        print(duplicate_ids)
        
        # Remove duplicates, keeping the first occurrence
        df_clean = df.drop_duplicates(subset=['ID'], keep='first')
        print(f"After removing duplicates: {df_clean.shape}")
    else:
        df_clean = df.copy()
    
    # Check for missing values in key variables
    mediation_vars = ['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', 'SVF_count', 'Age']
    print(f"\nMissing values in mediation variables:")
    for var in mediation_vars:
        missing = df_clean[var].isna().sum()
        print(f"  {var}: {missing} missing")
    
    # Check disease severity information
    severity_vars = ['hoehn_yahr_score', 'disease_duration_symptoms', 'disease_duration_diagnosis']
    print(f"\nDisease severity information:")
    for var in severity_vars:
        if var in df_clean.columns:
            missing = df_clean[var].isna().sum()
            total = len(df_clean)
            available = total - missing
            print(f"  {var}: {available}/{total} participants have data ({missing} missing)")
            
            if available > 0:
                non_null_data = df_clean[var].dropna()
                print(f"    Range: {non_null_data.min()} - {non_null_data.max()}")
                print(f"    Mean: {non_null_data.mean():.2f}")
    
    # Show sample of disease severity data
    print(f"\nSample of disease severity data:")
    severity_sample = df_clean[['ID', 'Age', 'hoehn_yahr_score', 'disease_duration_symptoms', 'disease_duration_diagnosis']].head(10)
    print(severity_sample)
    
    # Save cleaned dataframe
    df_clean.to_csv('final_clean_mediation_data.csv', index=False)
    print(f"\nSaved cleaned dataframe: final_clean_mediation_data.csv")
    
    # Create summary of available covariates
    print(f"\nAvailable covariates for mediation analysis:")
    all_vars = df_clean.columns.tolist()
    mediation_vars = ['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', 'SVF_count', 'Age']
    other_vars = [var for var in all_vars if var not in mediation_vars and var != 'ID']
    
    print(f"Mediation variables ({len(mediation_vars)}):")
    for var in mediation_vars:
        print(f"  - {var}")
    
    print(f"\nAdditional covariates ({len(other_vars)}):")
    for var in other_vars:
        missing = df_clean[var].isna().sum()
        total = len(df_clean)
        available = total - missing
        print(f"  - {var}: {available}/{total} participants have data")
    
    return df_clean

if __name__ == '__main__':
    clean_df = clean_final_dataframe()

