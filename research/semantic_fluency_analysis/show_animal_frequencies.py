#!/usr/bin/env python3
"""Show frequency ranks for animals in SVF data"""

import pandas as pd
from wordfreq import top_n_list

# Load the frequency list
print("Loading top 50,000 English words...")
top_50k = top_n_list('en', 50000)
rank_dict = {word: rank+1 for rank, word in enumerate(top_50k)}

def get_word_rank(word):
    """Get frequency rank (1=most common, 50000=least common, 0=not in top 50k)"""
    return rank_dict.get(word.lower(), 0)

def get_word_frequency_score(word):
    """Convert rank to frequency score (higher score = more common)"""
    rank = get_word_rank(word)
    if rank == 0:
        return 0.0
    else:
        return 1.0 - (rank / 50000)

# Load your actual data
print("\nLoading SVF data...")
data = pd.read_csv('data/fluency_data.csv')

# Get unique animals from your data
unique_animals = sorted(data['Item'].str.lower().unique())

print(f"\nFound {len(unique_animals)} unique animals in your data")

# Get ranks for all animals
animal_data = []
for animal in unique_animals:
    rank = get_word_rank(animal)
    score = get_word_frequency_score(animal)
    
    if rank > 0:
        if rank <= 1000:
            category = "Very Common (Top 1k)"
        elif rank <= 5000:
            category = "Common (1k-5k)"
        elif rank <= 10000:
            category = "Fairly Common (5k-10k)"
        elif rank <= 25000:
            category = "Less Common (10k-25k)"
        else:
            category = "Uncommon (25k-50k)"
    else:
        category = "Not in Top 50k"
    
    animal_data.append({
        'Animal': animal,
        'Rank': rank if rank > 0 else None,
        'Frequency_Score': score,
        'Category': category
    })

# Convert to DataFrame and sort
df = pd.DataFrame(animal_data)
df_sorted = df.sort_values('Rank', na_position='last')

print("\n" + "="*80)
print("FREQUENCY RANKS FOR ALL ANIMALS IN YOUR SVF DATA")
print("="*80)
print(f"\n{'Animal':<25} {'Rank':<12} {'Frequency Score':<18} {'Category'}")
print("-"*80)

for _, row in df_sorted.iterrows():
    if pd.notna(row['Rank']):
        print(f"{row['Animal']:<25} {int(row['Rank']):<12} {row['Frequency_Score']:<18.4f} {row['Category']}")
    else:
        print(f"{row['Animal']:<25} {'N/A':<12} {'0.0000':<18} {row['Category']}")

# Summary by category
print("\n" + "="*80)
print("SUMMARY BY FREQUENCY CATEGORY:")
print("="*80)

for category in df_sorted['Category'].unique():
    animals = df_sorted[df_sorted['Category'] == category]['Animal'].tolist()
    print(f"\n{category}: {len(animals)} animals")
    if len(animals) <= 20:
        print(f"  {', '.join(animals)}")
    else:
        print(f"  {', '.join(animals[:20])}... (+{len(animals)-20} more)")

# Show specific example: PD00020
print("\n" + "="*80)
print("EXAMPLE: PD00020's animals with frequency ranks:")
print("="*80)

pd00020_animals = data[data['ID'] == 'PD00020']['Item'].str.lower().tolist()
print(f"\nPD00020 said: {', '.join(pd00020_animals)}")
print(f"\n{'Animal':<20} {'Rank':<12} {'Score':<12} {'Category'}")
print("-"*70)

for animal in pd00020_animals:
    rank = get_word_rank(animal)
    score = get_word_frequency_score(animal)
    
    if rank > 0:
        if rank <= 1000:
            cat = "Very Common"
        elif rank <= 5000:
            cat = "Common"
        elif rank <= 10000:
            cat = "Fairly Common"
        elif rank <= 25000:
            cat = "Less Common"
        else:
            cat = "Uncommon"
        print(f"{animal:<20} {rank:<12} {score:<12.4f} {cat}")
    else:
        print(f"{animal:<20} {'N/A':<12} {'0.0000':<12} Not in Top 50k")

# Show frequency transitions for PD00020
print("\n" + "="*80)
print("FREQUENCY TRANSITIONS for PD00020:")
print("="*80)
print(f"\n{'Word Pair':<30} {'Transition Score':<20} {'Interpretation'}")
print("-"*70)

for i in range(len(pd00020_animals) - 1):
    word1 = pd00020_animals[i]
    word2 = pd00020_animals[i+1]
    
    score1 = get_word_frequency_score(word1)
    score2 = get_word_frequency_score(word2)
    transition = score2 - score1
    
    if transition < -0.1:
        interpretation = "Strong Exploration (common→rare)"
    elif transition < -0.05:
        interpretation = "Moderate Exploration"
    elif transition < 0.05:
        interpretation = "Similar frequency"
    elif transition < 0.1:
        interpretation = "Moderate Exploitation"
    else:
        interpretation = "Strong Exploitation (rare→common)"
    
    print(f"{word1} → {word2:<20} {transition:<20.4f} {interpretation}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

