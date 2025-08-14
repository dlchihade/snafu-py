import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('fluency_data/SVF Data1.csv', encoding='latin-1')

print(f"Total rows in dataset: {len(df)}")
print(f"Raw unique IDs: {df['ID'].nunique()}")

# Clean the IDs
df['ID_clean'] = df['ID'].astype(str).str.strip("'")
print(f"Unique IDs after cleaning: {df['ID_clean'].nunique()}")

# Check for missing values
print(f"Missing IDs: {df['ID'].isna().sum()}")
print(f"Empty string IDs: {(df['ID_clean'] == '').sum()}")

# Get all unique IDs
unique_ids = sorted(df['ID_clean'].unique())
print(f"\nAll unique participant IDs ({len(unique_ids)}):")
for i, pid in enumerate(unique_ids):
    print(f"{i+1:2d}. {pid}")

# Check for any issues with specific IDs
print(f"\nChecking for problematic IDs:")
for pid in unique_ids:
    if pd.isna(pid) or pid == '' or pid == 'nan':
        print(f"Problematic ID: '{pid}'")
        count = len(df[df['ID_clean'] == pid])
        print(f"  Count: {count}")

# Group by participant and show word counts
print(f"\nWord counts per participant:")
participant_counts = df.groupby('ID_clean').size().sort_values(ascending=False)
print(participant_counts.head(10))
print(f"\nTotal participants with data: {len(participant_counts)}")
