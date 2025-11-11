#!/usr/bin/env python3
"""Create final clean dataset with extracted disease severity variables"""

import pandas as pd

def create_final_disease_severity_dataset():
    """Create the final clean dataset with disease severity variables"""
    print("Creating final disease severity dataset...")
    
    # Load the cross-matched data
    cross_matched_df = pd.read_csv('cross_matched_disease_severity.csv')
    print(f"Cross-matched data shape: {cross_matched_df.shape}")
    
    # Remove duplicates
    df_clean = cross_matched_df.drop_duplicates(subset=['ID'], keep='first')
    print(f"After removing duplicates: {df_clean.shape}")
    
    # Create final dataset with the extracted disease severity variables
    # Use the extracted values as the primary source, fall back to current if missing
    
    final_df = df_clean.copy()
    
    # Replace current disease severity variables with extracted ones where available
    severity_mappings = {
        'disease_duration_symptoms_current': 'disease_duration_symptoms_extracted',
        'disease_duration_diagnosis_current': 'disease_duration_diagnosis_extracted', 
        'hoehn_yahr_score_current': 'hoehn_yahr_score_extracted'
    }
    
    for current_col, extracted_col in severity_mappings.items():
        if current_col in final_df.columns and extracted_col in final_df.columns:
            # Use extracted values where available, otherwise keep current
            final_df[current_col] = final_df[extracted_col].fillna(final_df[current_col])
            
            # Rename to clean column names
            clean_name = current_col.replace('_current', '')
            final_df[clean_name] = final_df[current_col]
            final_df = final_df.drop(columns=[current_col, extracted_col])
    
    # Add patient cognitive complaint if available
    if 'patient_cognitive_complaint' in final_df.columns:
        final_df['cognitive_complaint'] = final_df['patient_cognitive_complaint']
        final_df = final_df.drop(columns=['patient_cognitive_complaint'])
    
    # Identify mediation variables
    mediation_vars = ['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', 'SVF_count', 'Age']
    
    # Check complete cases for mediation
    complete_cases = final_df.dropna(subset=mediation_vars)
    print(f"\nComplete cases for mediation analysis: {len(complete_cases)}")
    
    # Show disease severity data quality
    print(f"\nDisease severity data quality:")
    severity_vars = ['disease_duration_symptoms', 'disease_duration_diagnosis', 'hoehn_yahr_score', 'cognitive_complaint']
    
    for var in severity_vars:
        if var in final_df.columns:
            missing = final_df[var].isna().sum()
            total = len(final_df)
            available = total - missing
            print(f"  {var}: {available}/{total} participants have data ({missing} missing)")
            
            if available > 0 and var != 'cognitive_complaint':
                non_null_data = final_df[var].dropna()
                print(f"    Range: {non_null_data.min()} - {non_null_data.max()}")
                print(f"    Mean: {non_null_data.mean():.2f}")
    
    # Show sample of final data
    print(f"\nSample of final disease severity data:")
    sample_cols = ['ID', 'Age', 'disease_duration_symptoms', 'disease_duration_diagnosis', 
                   'hoehn_yahr_score', 'cognitive_complaint']
    available_cols = [col for col in sample_cols if col in final_df.columns]
    print(final_df[available_cols].head(10))
    
    # Save final dataset
    final_df.to_csv('final_disease_severity_mediation_data.csv', index=False)
    print(f"\nSaved final dataset: final_disease_severity_mediation_data.csv")
    
    # Save complete cases only
    complete_cases.to_csv('final_complete_disease_severity_mediation_data.csv', index=False)
    print(f"Saved complete cases: final_complete_disease_severity_mediation_data.csv")
    
    # Create variable descriptions
    var_descriptions = {
        'ID': 'Participant ID',
        'norm_LC_avg': 'LC integrity (normalized)',
        'alpha_NET_mean': 'α-power (8-12 Hz)',
        'exploitation_coherence_ratio': 'EE metric (Exploitation Coherence Ratio)',
        'SVF_count': 'SVF Count (number of words)',
        'Age': 'Age in years',
        'disease_duration_symptoms': 'Disease duration since first symptoms (years)',
        'disease_duration_diagnosis': 'Disease duration since diagnosis (years)',
        'hoehn_yahr_score': 'Hoehn and Yahr score (disease severity: 1-5)',
        'cognitive_complaint': 'Patient cognitive complaint (Yes/No/Uncertain)',
        'num_switches': 'Number of switches between exploitation/exploration',
        'novelty_score': 'Novelty score',
        'exploitation_intra_mean': 'Mean exploitation coherence',
        'exploration_intra_mean': 'Mean exploration coherence',
        'inter_phase_mean': 'Mean inter-phase coherence',
        'exploration_coherence_ratio': 'Exploration coherence ratio',
        'phase_separation_index': 'Phase separation index',
        'mean_zipf': 'Mean Zipf frequency',
        'median_zipf': 'Median Zipf frequency',
        'std_zipf': 'Standard deviation of Zipf frequency',
        'mean_rank': 'Mean word rank',
        'median_rank': 'Median word rank',
        'std_rank': 'Standard deviation of word rank'
    }
    
    # Save variable descriptions
    descriptions_df = pd.DataFrame(list(var_descriptions.items()), columns=['Variable', 'Description'])
    descriptions_df.to_csv('disease_severity_variable_descriptions.csv', index=False)
    print(f"Saved variable descriptions: disease_severity_variable_descriptions.csv")
    
    # Summary
    print(f"\nFinal Summary:")
    print(f"- Total participants: {len(final_df)}")
    print(f"- Complete mediation cases: {len(complete_cases)}")
    print(f"- Mediation variables: {mediation_vars}")
    print(f"- Disease severity variables:")
    print(f"  1. Disease duration since first symptom")
    print(f"  2. Disease duration since diagnosis")
    print(f"  3. Hoehn and Yahr score")
    print(f"  4. Patient cognitive complaint")
    print(f"- Total variables: {len(final_df.columns)}")
    
    return final_df, complete_cases

if __name__ == '__main__':
    final_df, complete_cases = create_final_disease_severity_dataset()

