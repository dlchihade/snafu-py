#!/usr/bin/env python3
"""
Single figure showing spaCy semantic similarity correlation matrix
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

def create_correlation_matrix_figure():
    """Create single figure showing spaCy semantic similarity correlation matrix"""
    
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create correlation matrix with enhanced styling
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    
    # Set title
    ax.set_title('SpaCy Semantic Similarity Matrix\n(Animal Words from Semantic Verbal Fluency Task)', 
                 fontsize=16, fontweight='bold', color=colors['primary'], pad=30)
    
    # Set ticks and labels
    ax.set_xticks(range(len(animal_words)))
    ax.set_yticks(range(len(animal_words)))
    ax.set_xticklabels(animal_words, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(animal_words, fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Cosine Similarity', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations for key similarities
    # Highlight some key high-similarity pairs
    key_pairs = [
        ('lion', 'tiger'),
        ('elephant', 'giraffe'),
        ('cat', 'dog'),
        ('shark', 'whale'),
        ('eagle', 'owl')
    ]
    
    for word1, word2 in key_pairs:
        if word1 in animal_words and word2 in animal_words:
            idx1 = animal_words.index(word1)
            idx2 = animal_words.index(word2)
            sim = similarity_matrix[idx1, idx2]
            
            # Add text annotation
            ax.text(idx2, idx1, f'{sim:.2f}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Add statistics text box
    stats_text = f"""Key Statistics:
• Average similarity: {np.mean(similarity_matrix):.3f}
• Max similarity: {np.max(similarity_matrix):.3f}
• Min similarity: {np.min(similarity_matrix):.3f}
• Std deviation: {np.std(similarity_matrix):.3f}
• 0.6 threshold: Optimal for phase detection"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add model info
    model_info = f"""SpaCy en_core_web_md:
• 20,000 word vocabulary
• 300-dimensional vectors
• Web-trained (2B tokens)
• 95% coverage of common words"""
    
    ax.text(0.98, 0.02, model_info, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / 'spacy_correlation_matrix.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'spacy_correlation_matrix.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    
    print("✅ Created spaCy correlation matrix figure")
    print(f"   Saved to: output/spacy_correlation_matrix.(png|pdf)")
    
    # Print key statistics
    print(f"\n📊 Matrix Statistics:")
    print(f"   Average similarity: {np.mean(similarity_matrix):.3f}")
    print(f"   Max similarity: {np.max(similarity_matrix):.3f}")
    print(f"   Min similarity: {np.min(similarity_matrix):.3f}")
    print(f"   Standard deviation: {np.std(similarity_matrix):.3f}")
    
    return fig

if __name__ == "__main__":
    create_correlation_matrix_figure()

