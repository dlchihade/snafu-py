#!/usr/bin/env python3
"""
Improved spaCy semantic similarity correlation matrix with better colors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from create_nature_quality_figures_real import setup_nature_style
import matplotlib.colors as mcolors

def load_spacy_model():
    """Load spaCy model"""
    try:
        nlp = spacy.load('en_core_web_md')
        return nlp
    except OSError:
        print("Please install spaCy model: python -m spacy download en_core_web_md")
        return None

def create_improved_correlation_matrix():
    """Create correlation matrix figure with improved, readable colors"""
    
    # Real animal words from the fluency task
    animal_words = [
        'lion', 'tiger', 'elephant', 'cat', 'dog', 'horse', 'cow', 'sheep',
        'monkey', 'giraffe', 'rhinoceros', 'crocodile', 'zebra', 'bear',
        'wolf', 'fox', 'deer', 'rabbit', 'mouse', 'bird', 'snake', 'fish',
        'shark', 'whale', 'dolphin', 'penguin', 'eagle', 'owl', 'duck', 'goose'
    ]
    
    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        return
    
    # Get spaCy vectors
    spacy_vectors = {}
    for word in animal_words:
        doc = nlp(word.lower())
        spacy_vectors[word] = doc.vector
    
    # Create similarity matrix
    word_vectors = [spacy_vectors[word] for word in animal_words]
    similarity_matrix = cosine_similarity(word_vectors)
    
    # Apply nature style
    colors = setup_nature_style()
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create custom colormap for better readability
    # Option 1: Viridis-based (excellent for colorblind accessibility)
    # cmap = 'viridis'
    
    # Option 2: Custom blue-white-red colormap
    colors_custom = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_diverging', colors_custom)
    
    # Option 3: Simple and clear sequential colormap
    # cmap = 'Blues'  # or 'Greens', 'Oranges', 'Purples'
    
    # Create correlation matrix with improved styling
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Set title
    ax.set_title('SpaCy Semantic Similarity Matrix\n(Animal Words from Semantic Verbal Fluency Task)', 
                 fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(animal_words)))
    ax.set_yticks(range(len(animal_words)))
    ax.set_xticklabels(animal_words, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(animal_words, fontsize=9)
    
    # Add subtle grid for readability
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Improved colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)
    cbar.set_label('Cosine Similarity', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Add threshold line on colorbar to show 0.6 threshold
    cbar.ax.axhline(y=0.6, color='black', linestyle='--', linewidth=2, alpha=0.8)
    cbar.ax.text(0.5, 0.6, ' 0.6', transform=cbar.ax.transAxes, 
                fontsize=8, verticalalignment='center', fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'spacy_correlation_matrix_improved.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'spacy_correlation_matrix_improved.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    
    print("✅ Created improved spaCy correlation matrix figure with better colors")
    print(f"   Saved to: output/spacy_correlation_matrix_improved.(png|pdf)")
    
    # Print key statistics for reference
    print(f"\n📊 Matrix Statistics (for reference):")
    print(f"   Average similarity: {np.mean(similarity_matrix):.3f}")
    print(f"   Max similarity: {np.max(similarity_matrix):.3f}")
    print(f"   Min similarity: {np.min(similarity_matrix):.3f}")
    print(f"   Standard deviation: {np.std(similarity_matrix):.3f}")
    print(f"   Colors: Blue (low) → White → Red (high similarity)")
    
    return fig

def create_multiple_color_options():
    """Create multiple versions with different color schemes"""
    
    # Real animal words from the fluency task
    animal_words = [
        'lion', 'tiger', 'elephant', 'cat', 'dog', 'horse', 'cow', 'sheep',
        'monkey', 'giraffe', 'rhinoceros', 'crocodile', 'zebra', 'bear',
        'wolf', 'fox', 'deer', 'rabbit', 'mouse', 'bird', 'snake', 'fish',
        'shark', 'whale', 'dolphin', 'penguin', 'eagle', 'owl', 'duck', 'goose'
    ]
    
    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        return
    
    # Get spaCy vectors
    spacy_vectors = {}
    for word in animal_words:
        doc = nlp(word.lower())
        spacy_vectors[word] = doc.vector
    
    # Create similarity matrix
    word_vectors = [spacy_vectors[word] for word in animal_words]
    similarity_matrix = cosine_similarity(word_vectors)
    
    # Apply nature style
    colors = setup_nature_style()
    
    # Different color schemes to try
    color_schemes = [
        ('viridis', 'Viridis (Colorblind-friendly)'),
        ('plasma', 'Plasma (High contrast)'),
        ('Blues', 'Blues (Sequential)'),
        ('Greens', 'Greens (Sequential)'),
        ('Oranges', 'Oranges (Sequential)'),
        ('Purples', 'Purples (Sequential)'),
        ('Reds', 'Reds (Sequential)'),
        ('YlOrRd', 'Yellow-Orange-Red (Sequential)'),
        ('BuPu', 'Blue-Purple (Sequential)'),
        ('GnBu', 'Green-Blue (Sequential)')
    ]
    
    for cmap_name, cmap_title in color_schemes:
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create correlation matrix
        im = ax.imshow(similarity_matrix, cmap=cmap_name, vmin=0, vmax=1)
        
        # Set title
        ax.set_title(f'SpaCy Semantic Similarity Matrix\n{cmap_title}', 
                     fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
        
        # Set ticks and labels
        ax.set_xticks(range(len(animal_words)))
        ax.set_yticks(range(len(animal_words)))
        ax.set_xticklabels(animal_words, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(animal_words, fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)
        cbar.set_label('Cosine Similarity', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # Add threshold line
        cbar.ax.axhline(y=0.6, color='black', linestyle='--', linewidth=2, alpha=0.8)
        cbar.ax.text(0.5, 0.6, ' 0.6', transform=cbar.ax.transAxes, 
                    fontsize=8, verticalalignment='center', fontweight='bold')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        safe_name = cmap_name.replace('/', '_').replace(' ', '_')
        fig.savefig(output_dir / f'spacy_correlation_matrix_{safe_name}.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close(fig)
        
        print(f"✅ Created {cmap_title} version: spacy_correlation_matrix_{safe_name}.png")
    
    return True

if __name__ == "__main__":
    # Create the improved version
    create_improved_correlation_matrix()
    
    # Uncomment the line below to create multiple color options
    # create_multiple_color_options()

