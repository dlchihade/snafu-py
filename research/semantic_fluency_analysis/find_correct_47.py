#!/usr/bin/env python3
"""Find the correct participants for mediation analysis"""

import pandas as pd

# Load data
metrics = pd.read_csv('output/NATURE_REAL_metrics.csv')
ages_df = pd.read_csv('participant_ages.csv')

# Find participants with complete MEG/LC + EE data
complete_meg_ee = metrics.dropna(subset=['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio'])
print(f'Participants with complete MEG/LC + EE data: {len(complete_meg_ee)}')

# Check which ones have age data
complete_meg_ee_ids = set(complete_meg_ee['ID'].tolist())
age_ids = set(ages_df['ID'].tolist())
common = complete_meg_ee_ids & age_ids
missing_age = complete_meg_ee_ids - age_ids

print(f'Have age data: {len(common)}')
print(f'Missing age data: {len(missing_age)}')

if missing_age:
    print(f'Missing age data for: {sorted(list(missing_age))}')

print('\nCorrect participants (with age data):')
correct_participants = sorted(list(common))
print(correct_participants)
print(f'Total: {len(correct_participants)} participants')

# Create correct age file with ONLY these participants
correct_ages = ages_df[ages_df['ID'].isin(correct_participants)]
correct_ages.to_csv('participant_ages_correct.csv', index=False)
print(f'\nSaved correct age file with {len(correct_ages)} participants')

# Verify this gives us the correct number
print('\nVerification:')
test_data = metrics.merge(correct_ages, on='ID', how='inner')
test_complete = test_data.dropna(subset=['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', 'Age'])
print(f'Final complete cases: {len(test_complete)}')

print('\nSummary:')
print(f'- Original 47 participants: Complete MEG/LC + EE data')
print(f'- Age-adjusted participants: {len(correct_participants)} (missing PD00959 age data)')
print(f'- This is the correct sample size for age-adjusted mediation analysis')
