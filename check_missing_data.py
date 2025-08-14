import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('fluency_data/SVF Data1.csv', encoding='latin-1')

# Find rows with missing IDs
missing_mask = df['ID'].isna()
missing_data = df[missing_mask]

print(f"Rows with missing IDs: {len(missing_data)}")
print(f"Missing data items:")
for i, item in enumerate(missing_data['Item']):
    print(f"{i+1:2d}. {item}")

# Check if there are patterns that might indicate separate participants
print(f"\nLooking for patterns in missing data...")

# Check if there are empty rows or separators
print(f"\nChecking for empty rows or separators:")
for i, row in missing_data.iterrows():
    print(f"Row {i}: ID='{row['ID']}', Item='{row['Item']}'")

# Let's also check the raw data around the missing values
print(f"\nChecking context around missing values:")
for i in range(len(df)):
    if pd.isna(df.iloc[i]['ID']):
        print(f"\nMissing ID at row {i}:")
        print(f"  Previous 3 rows:")
        for j in range(max(0, i-3), i):
            print(f"    {j}: {df.iloc[j]['ID']} - {df.iloc[j]['Item']}")
        print(f"  Current row: {df.iloc[i]['ID']} - {df.iloc[i]['Item']}")
        print(f"  Next 3 rows:")
        for j in range(i+1, min(len(df), i+4)):
            print(f"    {j}: {df.iloc[j]['ID']} - {df.iloc[j]['Item']}")
        break  # Just show the first occurrence
