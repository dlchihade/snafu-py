#!/usr/bin/env python3
"""Compile comprehensive dataframe with all mediation variables and covariates"""

import pandas as pd
import numpy as np
from pathlib import Path
import io

def load_original_fluency_data():
    """Load the original fluency data to calculate SVF counts"""
    data_str = '''ID,Item
PD00020,"Lion,Tiger,Sheep,Dog,Cat,Camel,Monkey,Chimpanzee,Buffalo,Hyena,Dog,Cat,Elephant,Hyena,Dog,Cat,Mouse,Bird,Camel,Dragon"
PD00048,"Lion,Hare,Elephant,Rhinoceros,Monkey,Giraffe,Cow,Elk,Fish,Horse,Tiger,Leopard,Jaguar"
PD00119,"Lion,Tiger,Duck,Goose,Deer,Horse,Zebra,Elephant,Bird,Giraffe,Hippo,Crocodile,Elephant,Sheep,Goat,Ewe,Duck"
PD00146,"Dog,Pig,Chicken,Partridge,Swallow,Squirrel,Rabbit,Horse,Hare,Calf,Bull,Cow,Lion,Tiger,Monkey,Giraffe,Elephant,Snake,Frog,Shark,Whale,Dolphin"
PD00215,"Donkey,Horse,Cow,Ox,Elephant,Llama,Cat,Dog,Mouse,Tiger,Lion,Leopard,Cheetah,Hyena,Bear,Goat,Partridge,Hare,Manatee,Turtle,Iguana,Frog,Toad"
PD00219,"Monkey,Lion,Tiger,Horse,Cat,Dog,Snake,Wolf,Coyote,Horse,Cow,Camel,Scorpion"
PD00267,"Cat,Dog,Lion,Beaver,Jaguar,Elephant,Gazelle,Bear,Fox,Horse,Ox,Calf,Giraffe"
PD00457,"rhinoceros,fox,tiger,hippopotamus,tiger,lion,leopard,gazelle,dog,cat,horse,cow,sheep,horse,lion,gazelle,turkey"
PD00458,"Dog,Cat,Hamster,Tiger,Fox,Lion,Rhinoceros,Wolf,Partridge,Coyote,Eagle,Pigeon,Parrot,Crocodile,Whale,Dolphin,Marmot"
PD00471,"Cat,Dog,Horse,Cow,Calf,Lion,Pig,Monkey,Rhinoceros,Zebra,Elephant,Antelope,Fish,Panther,Camel,Lion,Hare,Crocodile,Rabbit,Mouse,Butterfly,Monkey"
PD00472,"Cat,dog,duck,Fish,Lark,Horse,pig,snake,crocodile,lion,Gazelle,Zebra,Elephant,Wolf,Fox,Panther,jaguar,Chimpanzee,Panda,Hen,Rooster"
PD00576,"Wolf,Horse,Camel,Cat,Dog,Tiger,Leopard,Lion,Elephant,Antelope,Caterpillar,Ostrich,Wasp,Ant,Fish,Shark,Crow,Trout,Salmon,Frog,Swallow,Spider,Panther"
PD00583,"Dog,Cat,Goat,Wolf,Fox,Hippopotamus,Elephant,Rooster,Hen,Turkey,Rabbit,Parrot,Crocodile,Butterfly,Grasshopper"
PD00647,"Elephant,Tiger,Alligator,Monkey,Butterfly,Whale,Dog,Cat,Rat,Chimpanzee,Cow,Cat,Flea,bird,Insect,Fish"
PD00653,"Cow,Cat,Bear,Horse,Pig,Dog,Wolf,Bird,Elk,Crocodile,Hippopotamus,Lion,Tiger,Bird,Snake,Pig,Cow,Chicken,Lamb"
PD00660,"Dog,Cat,Rat,Crocodile,Lion,Tiger,Horse,Cow,Fish,Bird,Hen,Giraffe,Hippopotamus,Chimpanzee,Goat,Fox"
PD00666,"Cat,Goat,Mouse,Weasel,Cow,Bizon,Horse,Lion,Panther,Elephant,Gazelle,Orangutan,Cat,Eagle,Marmot,Moray eel,Snake,Whale,Shark"
PD00757,"Dog,Cat,Lion,Tiger,Fox,Snake,Spider,Elephant,Sheep"
PD00786,"Cat,Dog,Monkey,Hen,Rooster,Pig,Horse,Cow,Calf,Lamb,Sheep,Ox,Chicken,Duck"
PD00849,"Lion,Hare,Wolf,Fox,Cat,Dog,Tiger,Elephant,Rhinoceros,Ostrich,Zebra,Pheasant,Giraffe,Leopard,Tiger,Cougar"
PD00869,"Cow,Calf,Horse,Lion,Hippopotamus,Rhinoceros,Monkey,Hen,Rabbit,Pheasant,Parrot,Gazelle,Giraffe,Lion,Marmot,dog,Donkey,Cheetah,Swan,Duck,Mule"
PD00955,"Dog,Cat,Rat,Mole,Lion,Tiger,Giraffe,Hyena,Leopard,Lynx,Fox,Ram,Horse"
PD00959,"Tiger,Lion,Hen,Rhino,Hippo,Llama,Giraffe,Goat,Sheep"
PD00999,"Lion,Tiger,Elephant,Rhinoceros,Crocodile,Coyote,Duck,Camel,Wolf,Hare,Lynx,Turtle,Horse,Cat,Dog,Snake,Panther,Whale,Dolphin"
PD01003,"Ostrich,Bison,Goat,Zebra,Dog,Cat,Cow,Pig,Hare,Lion,Tiger,Donkey,Monkey,Dog,Vulture,Turtle,Rhinoceros,Whale"
PD01126,"Dog,Cat,Tiger,Elephant,Cow,Calf,Horse,Panda"
PD01133,"Dog,Cat,Monkey,Mule,Eel,Tiger,Lion,Moose,Elk,Wolf,Fox,Marmot,Beaver,Gopher,Crow,Giraffe"
PD01145,"Lion,Tiger,Bear,Wolf,Cat,Goat,Duck,Goose,Bull,Antelope"
PD01146,"Lion,Cat,Dog,Cow,Sheep,Duck,Giraffe,Pig,Hare,Antelope,Zebra,Camel,Elephant,Rhinoceros,Dog,Donkey,Horse,Goat,Mouse"
PD01156,"Reptile,Cow,Deer,Salamander,Bear,Fish,Cat,Mouse"
PD01160,"Lion,Bear,Tiger,Fox,Rat,Cat,Dog,Horse,Monkey,Hippo,Wolf,Deer,Roe deer,Crocodile,Salamander,Reptile,Panda"
PD01161,"Horse,Cow,Sheep,Ewe,Camel,Hippo,Elephant,Turkey,Cat,Otter,Rabbit,Hare,Tiger,Lion,Snake,Bee,Wasp,Mosquito,Wolf,Fox,Zebra,Coyote,Hen,Bird,Crow"
PD01199,"Lion,Tiger,Duck,Goose,Roe Deer,Horse,Zebra,Elephant,Bird,Giraffe,Hippopotamus,Crocodile,Elephant,Sheep,Goat,Ewe,Duck"
PD01201,"Cat,Dog,Elephant,Lion,Hippopotamus,Rhinoceros,Eagle,Zebra,Panther,Pigeon,Cow"
PD01223,"Elephant,Lion,Crocodile,Caiman,Snake,Dog,Cat,Fish,Turtle,Skunk"
PD01225,"Lion,Dog,Cat,Horse,Bird,Eagle,Donkey,Rabbit,Deer,Hare,Bird,Mouse,Rat,Snake,Bird,Camel,Dromedary,Alligator,Crocodile"
PD01237,"Cat,Dog,Horse,Pig,Pig,Goose,Goat,Dromedary,Gazelle,Zebra,Bull,Swallow,Pigeon,Llama,Fox,Elephant,Marmot,Wildcat,Spider,Seagull,Mongoose,Penguin"
PD01247,"Albatross,Whale,Bulldog,Poodle,Toad,Donkey,Monkey,Hippopotamus,Ram,Bull,Cat,Dog,Porcupine"
PD01270,"Lion,Giraffe,Tiger,Panther,Pigeon,Monkey,Wolf,Dog,Cat,Mouse,Horse,Sheep,Camel,Leopard,Emu"
PD01282,"Cow,Tiger,Lion,Horse,Dog,Cat,Mouse,Snake,Squirrel,Slug,Snail,Monkey,Giraffe,Elephant,Wolf,Fish,Bird"
PD01284,"Lion,Tiger,Caribou,Owl,Shark,Fish,Monkey,Dog,Cat,Alligator,Crocodile,Pelican,Jaguar,Penguin,Cow,Pig,Rooster,Raccoon,Beaver"
PD01290,"Dog,Cat,Rabbit,Lion,Lamb,Unicorn,Tiger,Elephant,Camel,Mole"
PD01306,"Dog,Cat,Raccoon,Lion,Elephant,Giraffe,Leopard,Tiger,Eagle,Turtle,Horse,Wolf,Hare,Horse,Cow,Mule,Sheep,Llama,Donkey,Parrot,Crow,Finch"
PD01312,"Dog,Cat,Horse,Goat,Sheep,Hen,Weasel,Hedgehog,Rabbit,Lion,Tiger,Monkey,Meerkat,Hippopotamus,Rhino,Zebra,Elephant,Panther,Jaguar,Chimpanzee,Gibbon,Fox,Raccoon,Rat,Elephant,Otter,Camel"
PD01319,"Dog,Cat,Mouse,Rat,Lion,Elephant,Tiger,Giraffe,Zebra,Eagle,Deer,Wolf,Fox,Bear,Cow,Ox"
PD01369,"Cat,Dog,Mouse,Rat,Snake,Caiman,Tiger,Giraffe,Fox,Crocodile,Lion,Elephant,Tiger,Deer,Crocodile,Dolphin,Otter,Whale,Dog,Swallow,Jaguar,Zebra,Mouse"
PD01377,"Cat,Dog,Mosquito,Vulture,Cow,Bull,Dromedary,Goat,Parrot,Monkey,Leopard,Lion,Tiger,Zebra,Giraffe,Squirrel,Bird,Crocodile,Hen,Wolf,Rabbit"
PD01435,"Rhinoceros,Dog,Mouse,Zebra,Raccoon,Wildcat,Kangaroo,Bear,Weasel,Skunk,Elephant,Monkey,Bull,Cow,Wolf,Wild boar,Fox"
PD01440,"Horse,Fox,Dog,Deer,Eagle,Goat,Cat,Raccoon,Dromedary,Camel,Rhinoceros,Lion,Tiger,Whale,Dolphin,Elephant"
PD01457,"Horse,Dog,Cat,Lion,Bird,Sheep,Giraffe,Whale,Hummingbird,Hippopotamus,Leopard,Deer,Panther"
PD01485,"Horse,Cow,Hen,Pig,Parrot,Parakeet,Snake,Tiger,Lion,Crow,Sheep,Goat,Mare,Moose,Zebra,Pig,Squirrel"
PD01559,"Dog,Cat,Perch,Golden eagle,Blackbird,Panther,Cow,Pig,Fox,Partridge,Gazelle,Hare,Jellyfish,Lizard,Tench,Fish,Shark,Cod"
PD01623,"Dog,Cat,Hippopotamus,Giraffe,Lion,Zebra,Mammoth,Blue Whale,Dolphin,Humpback Whale,Whale,Eel,Tiger,Chimpanzee"
PD01660,"Lion,Bear,Tiger,Fox,Dog,Cat,Rat,Mouse,Horse,Monkey,Hippopotamus,Ox,Deer,Roe Deer,Crocodile,Salamander,Turtle"
PD01667,"Dog,Cat,Horse,Cow,Hen,Chick,Swan,Bird,Giraffe,Lion,Zebra,Gazelle,Monkey,Chimpanzee,Squirrel,Frog,Fish,Ox,Chicken"
PD01715,"Donkey,Boa,Goat,Frog,Zebra,Hippopotamus,Elephant,Bird,Crocodile,Caiman,Fox,Wolf,Dog,Horse,Turkey,Chicken"
'''
    compressed_data = pd.read_csv(io.StringIO(data_str))
    # Calculate SVF counts
    svf_counts = compressed_data.set_index('ID')['Item'].str.split(',').apply(len).reset_index(name='SVF_count')
    return svf_counts

def examine_neuropsychology_files():
    """Examine available neuropsychology files for additional covariates"""
    downloads_path = Path.home() / 'Downloads'
    neuro_files = []
    
    # Check for relevant files
    potential_files = [
        'clinical_data.xlsx',
        'Dietta Chihade_BD_RPQ_UPDATE_Neuropsy.xlsx - Parkinson patients.csv',
        'Filtered_Cross_Match_Data.xlsx',
        'Subject_Similarity_Data_Corrected.xlsx'
    ]
    
    for file_name in potential_files:
        file_path = downloads_path / file_name
        if file_path.exists():
            try:
                if file_name.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                neuro_files.append({
                    'file': file_name,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'data': df
                })
                print(f"✓ Found: {file_name} - Shape: {df.shape}")
            except Exception as e:
                print(f"✗ Error reading {file_name}: {e}")
    
    return neuro_files

def compile_comprehensive_dataframe():
    """Compile comprehensive dataframe with all variables"""
    print("Loading data...")
    
    # Load core data
    metrics = pd.read_csv('output/NATURE_REAL_metrics.csv')
    ages_df = pd.read_csv('participant_ages_correct_46.csv')
    svf_counts = load_original_fluency_data()
    
    print(f"Metrics: {len(metrics)} participants")
    print(f"Age data: {len(ages_df)} participants")
    print(f"SVF counts: {len(svf_counts)} participants")
    
    # Start with metrics
    comprehensive_df = metrics.copy()
    
    # Add SVF counts
    comprehensive_df = comprehensive_df.merge(svf_counts, on='ID', how='left')
    
    # Add age data
    comprehensive_df = comprehensive_df.merge(ages_df, on='ID', how='left')
    
    # Examine neuropsychology files for additional covariates
    print("\nExamining neuropsychology files...")
    neuro_files = examine_neuropsychology_files()
    
    # Try to add additional covariates from neuropsychology files
    for neuro_file in neuro_files:
        df = neuro_file['data']
        print(f"\nFile: {neuro_file['file']}")
        print(f"Columns: {neuro_file['columns']}")
        
        # Look for ID column
        id_cols = [col for col in df.columns if 'id' in col.lower() or 'ID' in col]
        if id_cols:
            print(f"Potential ID columns: {id_cols}")
            # Try to merge if we find matching IDs
            for id_col in id_cols:
                try:
                    # Check for overlapping IDs
                    neuro_ids = set(df[id_col].astype(str).str.strip())
                    our_ids = set(comprehensive_df['ID'].astype(str).str.strip())
                    overlap = neuro_ids & our_ids
                    if len(overlap) > 0:
                        print(f"Found {len(overlap)} overlapping IDs with column {id_col}")
                        # Add relevant columns (exclude ID column)
                        relevant_cols = [col for col in df.columns if col != id_col]
                        if relevant_cols:
                            print(f"Adding columns: {relevant_cols[:5]}...")  # Show first 5
                            temp_df = df[[id_col] + relevant_cols].copy()
                            temp_df = temp_df.rename(columns={id_col: 'ID'})
                            comprehensive_df = comprehensive_df.merge(temp_df, on='ID', how='left')
                            break
                except Exception as e:
                    print(f"Error merging with {id_col}: {e}")
    
    # Identify mediation variables
    mediation_vars = ['norm_LC_avg', 'alpha_NET_mean', 'exploitation_coherence_ratio', 'SVF_count', 'Age']
    
    print(f"\nComprehensive dataframe shape: {comprehensive_df.shape}")
    print(f"Columns: {comprehensive_df.columns.tolist()}")
    
    # Check complete cases for mediation
    complete_cases = comprehensive_df.dropna(subset=mediation_vars)
    print(f"\nComplete cases for mediation analysis: {len(complete_cases)}")
    
    # Save comprehensive dataframe
    comprehensive_df.to_csv('comprehensive_mediation_data.csv', index=False)
    print(f"\nSaved comprehensive dataframe: comprehensive_mediation_data.csv")
    
    # Save complete cases only
    complete_cases.to_csv('complete_mediation_data.csv', index=False)
    print(f"Saved complete cases: complete_mediation_data.csv")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"- Total participants: {len(comprehensive_df)}")
    print(f"- Complete mediation cases: {len(complete_cases)}")
    print(f"- Mediation variables: {mediation_vars}")
    print(f"- Additional covariates: {len(comprehensive_df.columns) - len(mediation_vars) - 1}")  # -1 for ID
    
    return comprehensive_df, complete_cases

if __name__ == '__main__':
    comprehensive_df, complete_cases = compile_comprehensive_dataframe()

