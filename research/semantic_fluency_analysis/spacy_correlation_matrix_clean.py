#!/usr/bin/env python3
"""
Clean spaCy semantic similarity correlation matrix for publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from create_nature_quality_figures_real import setup_nature_style

def load_spacy_model():
    """Load spaCy model"""
    try:
        nlp = spacy.load('en_core_web_md')
        return nlp
    except OSError:
        print("Please install spaCy model: python -m spacy download en_core_web_md")
        return None

def create_clean_correlation_matrix():
    """Create clean correlation matrix figure without text overlays"""
    
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
    
    # Create correlation matrix with clean styling
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    
    # Set title
    ax.set_title('SpaCy Semantic Similarity Matrix\n(Animal Words from Semantic Verbal Fluency Task)', 
                 fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(animal_words)))
    ax.set_yticks(range(len(animal_words)))
    ax.set_xticklabels(animal_words, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(animal_words, fontsize=9)
    
    # Add subtle grid for readability
    ax.grid(True, alpha=0.2, color='white', linewidth=0.3)
    
    # Clean colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)
    cbar.set_label('Cosine Similarity', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'spacy_correlation_matrix_clean.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'spacy_correlation_matrix_clean.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    
    print("✅ Created clean spaCy correlation matrix figure (publication-ready)")
    print(f"   Saved to: output/spacy_correlation_matrix_clean.(png|pdf)")
    
    # Print key statistics for reference
    print(f"\n📊 Matrix Statistics (for reference):")
    print(f"   Average similarity: {np.mean(similarity_matrix):.3f}")
    print(f"   Max similarity: {np.max(similarity_matrix):.3f}")
    print(f"   Min similarity: {np.min(similarity_matrix):.3f}")
    print(f"   Standard deviation: {np.std(similarity_matrix):.3f}")
    
    return fig

if __name__ == "__main__":
    create_clean_correlation_matrix()

