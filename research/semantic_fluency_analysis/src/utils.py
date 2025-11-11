"""
Utility functions for semantic fluency analysis with optimized spaCy operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import logging
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import warnings
try:
    from wordfreq import top_n_list, zipf_frequency
except Exception:  # wordfreq optional, handled gracefully
    top_n_list = None
    def zipf_frequency(word: str, lang: str = 'en') -> float:
        return 0.0
from pathlib import Path

logger = logging.getLogger(__name__)

class SemanticUtils:
    """Utility class for semantic operations with caching"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.word_ranks = {}
    
    @lru_cache(maxsize=1000)
    def cosine_similarity(self, vec1: Tuple[float, ...], vec2: Tuple[float, ...]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get_word_frequency(self, word: str) -> int:
        """Get word frequency rank (lower = more frequent)"""
        return self.word_ranks.get(word.lower(), 10000)  # Default high rank for unknown words
    
    def load_word_ranks(self, filepath: Optional[str] = None) -> None:
        """Load word frequency ranks using wordfreq top-50k or a provided file.

        If filepath is provided, expects a newline-separated list of words from most
        frequent to least frequent. Otherwise uses wordfreq.top_n_list("en", 50000).
        """
        try:
            ranks: Dict[str, int] = {}
            words: List[str] = []
            if filepath and Path(filepath).exists():
                with open(filepath, 'r') as f:
                    words = [w.strip().lower() for w in f if w.strip()]
            elif top_n_list is not None:
                words = [w.lower() for w in top_n_list('en', 50000)]
            for idx, w in enumerate(words):
                ranks[w] = idx + 1  # 1-based rank
            self.word_ranks = ranks
            logger.info(f"Word ranks loaded (size={len(self.word_ranks)})")
        except Exception as e:
            logger.warning(f"Failed to load word ranks: {e}")
            self.word_ranks = {}
    
    def get_word_rank(self, word: str) -> int:
        """Get word rank with caching"""
        return self.get_word_frequency(word)

    def get_zipf(self, word: str) -> float:
        """Return Zipf frequency (log10 per million) using wordfreq. 0.0 if unknown."""
        try:
            return float(zipf_frequency(word, 'en'))
        except Exception:
            return 0.0
    
    def calculate_similarities_vectorized(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Calculate similarities between consecutive vectors efficiently"""
        if len(vectors) < 2:
            return np.array([])
        
        # Convert to numpy array for vectorized operations
        vectors_array = np.array(vectors)
        
        # Calculate similarities between consecutive vectors
        similarities = []
        for i in range(len(vectors_array) - 1):
            sim = cosine_similarity([vectors_array[i]], [vectors_array[i + 1]])[0][0]
            similarities.append(sim)
        
        return np.array(similarities)
    
    def get_frequency_stats(self, items: List[str]) -> Dict:
        """Calculate frequency statistics (rank and Zipf) for a list of items."""
        ranks = [self.get_word_rank(item) for item in items]
        zipfs = [self.get_zipf(item) for item in items]
        return {
            'mean_rank': float(np.mean(ranks)) if ranks else 0.0,
            'median_rank': float(np.median(ranks)) if ranks else 0.0,
            'std_rank': float(np.std(ranks)) if ranks else 0.0,
            'min_rank': float(np.min(ranks)) if ranks else 0.0,
            'max_rank': float(np.max(ranks)) if ranks else 0.0,
            'mean_zipf': float(np.mean(zipfs)) if zipfs else 0.0,
            'median_zipf': float(np.median(zipfs)) if zipfs else 0.0,
            'std_zipf': float(np.std(zipfs)) if zipfs else 0.0,
            'min_zipf': float(np.min(zipfs)) if zipfs else 0.0,
            'max_zipf': float(np.max(zipfs)) if zipfs else 0.0,
        }
    
    def get_rank_stats(self, items: List[str]) -> Dict:
        """Calculate rank statistics (alias for frequency stats)"""
        return self.get_frequency_stats(items)


class SpacyOptimizer:
    """Optimized spaCy operations for semantic fluency analysis"""
    
    def __init__(self, model_name: str = "en_core_web_md", batch_size: int = 1000):
        self.model_name = model_name
        self.batch_size = batch_size
        self.nlp = None
        self._load_model()
        
    def _load_model(self):
        """Load spaCy model with error handling"""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.error(f"spaCy model '{self.model_name}' not found. Please install with:")
            logger.error(f"python -m spacy download {self.model_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            raise
    
    def get_vectors_batch(self, words: List[str], 
                         min_similarity: float = 0.1,
                         max_similarity: float = 0.95) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """
        Get vectors for a batch of words efficiently
        
        Returns:
            Tuple of (vectors, valid_words, valid_indices)
        """
        if not words:
            return [], [], []
        
        # Clean and normalize words
        clean_words = [word.strip().lower() for word in words if word and word.strip()]
        
        if not clean_words:
            return [], [], []
        
        # Process in batches for efficiency
        vectors = []
        valid_words = []
        valid_indices = []
        
        for i in range(0, len(clean_words), self.batch_size):
            batch_words = clean_words[i:i + self.batch_size]
            
            try:
                # Process batch with spaCy
                docs = list(self.nlp.pipe(batch_words))
                
                for j, doc in enumerate(docs):
                    word_idx = i + j
                    word = batch_words[j]
                    
                    if doc.has_vector:
                        vector = doc.vector
                        
                        # Check if vector is valid (not all zeros)
                        if np.any(vector).item() and np.linalg.norm(vector) > 0:
                            vectors.append(vector)
                            valid_words.append(word)
                            valid_indices.append(word_idx)
                        else:
                            logger.debug(f"Zero vector for word: {word}")
                    else:
                        logger.debug(f"No vector for word: {word}")
                        
            except Exception as e:
                logger.warning(f"Error processing batch starting at index {i}: {e}")
                continue
        
        logger.info(f"Processed {len(clean_words)} words, got {len(vectors)} valid vectors")
        return vectors, valid_words, valid_indices
    
    def get_similarity_matrix(self, words: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get similarity matrix for a list of words"""
        vectors, valid_words, _ = self.get_vectors_batch(words)
        
        if len(vectors) < 2:
            return np.array([]), valid_words
        
        # Calculate similarity matrix
        vectors_array = np.array(vectors)
        similarity_matrix = cosine_similarity(vectors_array)
        
        return similarity_matrix, valid_words
    
    def get_consecutive_similarities(self, words: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get similarities between consecutive words"""
        vectors, valid_words, _ = self.get_vectors_batch(words)
        
        if len(vectors) < 2:
            return np.array([]), valid_words
        
        # Calculate similarities between consecutive vectors
        similarities = []
        for i in range(len(vectors) - 1):
            sim = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
            similarities.append(sim)
        
        return np.array(similarities), valid_words
    
    def get_word_similarity(self, word1: str, word2: str) -> float:
        """Get similarity between two specific words"""
        vectors, _, _ = self.get_vectors_batch([word1, word2])
        
        if len(vectors) < 2:
            return 0.0
        
        return cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    
    def get_most_similar_words(self, target_word: str, candidate_words: List[str], 
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most similar words to a target word"""
        all_words = [target_word] + candidate_words
        vectors, valid_words, _ = self.get_vectors_batch(all_words)
        
        if len(vectors) < 2:
            return []
        
        # Find target word index
        try:
            target_idx = valid_words.index(target_word.lower().strip())
        except ValueError:
            return []
        
        # Calculate similarities to target
        target_vector = vectors[target_idx]
        similarities = []
        
        for i, vector in enumerate(vectors):
            if i != target_idx:
                sim = cosine_similarity([target_vector], [vector])[0][0]
                similarities.append((valid_words[i], sim))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def validate_word_coverage(self, words: List[str]) -> Dict:
        """Validate word coverage and provide statistics"""
        vectors, valid_words, _ = self.get_vectors_batch(words)
        
        coverage = len(valid_words) / len(words) if words else 0
        missing_words = [w for w in words if w.lower().strip() not in valid_words]
        
        return {
            'total_words': len(words),
            'covered_words': len(valid_words),
            'coverage_percentage': coverage * 100,
            'missing_words': missing_words,
            'missing_count': len(missing_words)
        }
