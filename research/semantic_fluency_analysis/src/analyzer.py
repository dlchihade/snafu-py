"""
Main analyzer class for semantic fluency analysis
"""

import pandas as pd
import numpy as np
import spacy
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.cluster import AgglomerativeClustering

from .config import AnalysisConfig
from .utils import SemanticUtils, SpacyOptimizer

logger = logging.getLogger(__name__)

class SemanticFluencyAnalyzer:
    """
    Main class for semantic fluency analysis with improved architecture
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.spacy_optimizer = None
        self.utils = SemanticUtils(config.cache_size)
        self.data = None
        self.meg_data = None
        self.results = []
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid configuration")
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Initialize spaCy and word frequency data"""
        try:
            # Initialize optimized spaCy operations
            self.spacy_optimizer = SpacyOptimizer(
                model_name=self.config.spacy_model,
                batch_size=self.config.batch_size
            )
            logger.info("Initialized spaCy optimizer successfully")
            
            # Load word frequency data
            self.utils.load_word_ranks()
            
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            raise
    
    def load_data(self, fluency_path: str, meg_path: str) -> None:
        """Load and validate data from external files"""
        try:
            # Load fluency data
            self.data = pd.read_csv(fluency_path)
            logger.info(f"Loaded fluency data: {len(self.data)} rows, {self.data['ID'].nunique()} participants")
            
            # Load MEG data
            self.meg_data = pd.read_csv(meg_path)
            logger.info(f"Loaded MEG data: {len(self.meg_data)} participants")
            
            # Validate data
            self._validate_data()
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self):
        """Validate data integrity"""
        if self.data is None or self.meg_data is None:
            raise ValueError("Data not loaded")
        
        # Check for required columns
        required_fluency_cols = ['ID', 'Item']
        required_meg_cols = ['ID', 'alpha_NET_mean']
        
        missing_fluency = [col for col in required_fluency_cols if col not in self.data.columns]
        missing_meg = [col for col in required_meg_cols if col not in self.meg_data.columns]
        
        if missing_fluency:
            raise ValueError(f"Missing fluency columns: {missing_fluency}")
        if missing_meg:
            raise ValueError(f"Missing MEG columns: {missing_meg}")
        
        # Check for empty datasets
        if self.data.empty:
            raise ValueError("Fluency dataset is empty")
        if self.meg_data.empty:
            raise ValueError("MEG dataset is empty")
        
        # Check for missing values
        missing_fluency = self.data['Item'].isna().sum()
        if missing_fluency > 0:
            logger.warning(f"Found {missing_fluency} missing items in fluency data")
        
        logger.info("Data validation passed")
    
    def analyze_participant(self, participant_data: pd.DataFrame) -> Dict:
        """Analyze a single participant's fluency data"""
        if participant_data.empty:
            return self._create_empty_result("unknown")
        
        participant_id = participant_data['ID'].iloc[0]
        items = participant_data['Item'].tolist()
        
        try:
            # Get word vectors using optimized spaCy operations
            vectors, valid_words, valid_indices = self.spacy_optimizer.get_vectors_batch(
                items,
                min_similarity=self.config.min_similarity,
                max_similarity=self.config.max_similarity
            )
            
            if len(vectors) < 2:
                logger.warning(f"Insufficient valid vectors for participant {participant_id}")
                return self._create_empty_result(participant_id)
            
            # Calculate similarities between consecutive words
            similarities = self.utils.calculate_similarities_vectorized(vectors)
            
            # Identify phases using configurable threshold
            phases = self._identify_phases(
                similarities, valid_words, vectors, self.config.similarity_threshold
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(phases, similarities, valid_words)
            
            # Calculate clustering metrics
            clustering_metrics = self._calculate_clustering_metrics(valid_words, vectors)
            
            # Calculate frequency/rank statistics
            frequency_stats = self.utils.get_frequency_stats(valid_words)
            
            # Create comprehensive result
            result = {
                'ID': participant_id,
                'num_items': len(items),
                'num_valid_items': len(valid_words),
                'word_coverage': len(valid_words) / len(items) * 100,
                'items': items,
                'valid_items': valid_words,
                'vectors': vectors,
                'similarities': similarities.tolist(),
                'phases': phases,
                'exploitation_percentage': metrics['exploitation_percentage'],
                'exploration_percentage': metrics['exploration_percentage'],
                'num_switches': metrics['num_switches'],
                'avg_exploitation_size': metrics['avg_exploitation_size'],
                'avg_exploration_size': metrics['avg_exploration_size'],
                'novelty_score': metrics['novelty_score'],
                'clustering_coefficient': clustering_metrics['clustering_coefficient'],
                'num_clusters': clustering_metrics['num_clusters'],
                'mean_rank': frequency_stats['mean_rank'],
                'median_rank': frequency_stats['median_rank'],
                'std_rank': frequency_stats['std_rank'],
                'min_rank': frequency_stats['min_rank'],
                'max_rank': frequency_stats['max_rank']
            }
            
            logger.info(f"Analyzed participant {participant_id}: {len(valid_words)} valid items, "
                       f"{metrics['exploitation_percentage']:.1f}% exploitation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing participant {participant_id}: {e}")
            return self._create_empty_result(participant_id)
    
    def _identify_phases(self, similarities: np.ndarray, items: List[str], 
                        vectors: List[np.ndarray], threshold: float) -> List[Dict]:
        """Identify exploitation and exploration phases with improved logic"""
        if len(similarities) < 1:
            return []
        
        phases = []
        current_phase = "Exploitation" if similarities[0] > threshold else "Exploration"
        phase_start = 0
        
        for i in range(1, len(similarities)):
            # Check for phase transition with minimum phase length
            if (self._should_transition(current_phase, similarities[i], threshold) and 
                i - phase_start >= self.config.min_phase_length):
                
                # Create phase if it meets minimum length requirement
                if i - phase_start >= self.config.min_phase_length:
                    phase = self._create_phase_dict(
                        current_phase, phase_start, i, items, vectors, similarities
                    )
                    phases.append(phase)
                
                # Switch phase
                current_phase = "Exploration" if current_phase == "Exploitation" else "Exploitation"
                phase_start = i
        
        # Add final phase if it meets minimum length
        if len(similarities) - phase_start >= self.config.min_phase_length:
            final_phase = self._create_phase_dict(
                current_phase, phase_start, len(similarities), items, vectors, similarities
            )
            phases.append(final_phase)
        
        return phases
    
    def _should_transition(self, current_phase: str, similarity: float, threshold: float) -> bool:
        """Determine if phase should transition with improved logic"""
        if current_phase == "Exploitation" and similarity <= threshold:
            return True
        elif current_phase == "Exploration" and similarity > threshold:
            return True
        return False
    
    def _create_phase_dict(self, phase_type: str, start: int, end: int, 
                          items: List[str], vectors: List[np.ndarray], 
                          similarities: np.ndarray) -> Dict:
        """Create standardized phase dictionary"""
        phase_items = items[start:end+1]
        phase_vectors = vectors[start:end+1]
        phase_similarities = similarities[start:end] if start < len(similarities) else []
        
        return {
            'type': phase_type,
            'start': start,
            'end': end,
            'items': phase_items,
            'vectors': phase_vectors,
            'similarities': phase_similarities,
            'size': end - start + 1,
            'avg_similarity': np.mean(phase_similarities) if len(phase_similarities) > 0 else 0
        }
    
    def _calculate_metrics(self, phases: List[Dict], similarities: np.ndarray, 
                          items: List[str]) -> Dict:
        """Calculate comprehensive metrics"""
        exploitation_phases = [p for p in phases if p['type'] == 'Exploitation']
        exploration_phases = [p for p in phases if p['type'] == 'Exploration']
        
        # Basic metrics
        num_switches = len(phases) - 1
        exploitation_time = sum(p['size'] for p in exploitation_phases)
        exploration_time = sum(p['size'] for p in exploration_phases)
        total_time = exploitation_time + exploration_time
        
        # Percentages
        exploitation_percentage = (exploitation_time / total_time * 100) if total_time > 0 else 0
        exploration_percentage = (exploration_time / total_time * 100) if total_time > 0 else 0
        
        # Phase sizes
        exploitation_sizes = [p['size'] for p in exploitation_phases]
        exploration_sizes = [p['size'] for p in exploration_phases]
        
        # Novelty calculation
        unique_items = set()
        novelty_scores = []
        for item in items:
            if item not in unique_items:
                novelty_scores.append(1)
                unique_items.add(item)
            else:
                novelty_scores.append(0)
        
        novelty_score = np.mean(novelty_scores) if novelty_scores else 0
        
        return {
            'exploitation_percentage': exploitation_percentage,
            'exploration_percentage': exploration_percentage,
            'num_switches': num_switches,
            'avg_exploitation_size': np.mean(exploitation_sizes) if exploitation_sizes else 0,
            'avg_exploration_size': np.mean(exploration_sizes) if exploration_sizes else 0,
            'novelty_score': novelty_score,
            'total_similarity': np.mean(similarities) if len(similarities) > 0 else 0
        }
    
    def _calculate_clustering_metrics(self, items: List[str], vectors: List[np.ndarray]) -> Dict:
        """Calculate clustering metrics using hierarchical clustering"""
        if len(vectors) < 2:
            return {'clustering_coefficient': 0, 'num_clusters': 0}
        
        try:
            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.config.clustering_threshold,
                linkage='ward'
            )
            
            vectors_array = np.array(vectors)
            cluster_labels = clustering.fit_predict(vectors_array)
            
            # Calculate clustering coefficient (simplified)
            num_clusters = len(set(cluster_labels))
            clustering_coefficient = 1 - (num_clusters / len(vectors))
            
            return {
                'clustering_coefficient': clustering_coefficient,
                'num_clusters': num_clusters,
                'cluster_labels': cluster_labels.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Error in clustering: {e}")
            return {'clustering_coefficient': 0, 'num_clusters': 0}
    
    def _create_empty_result(self, participant_id: str) -> Dict:
        """Create empty result for failed analysis"""
        return {
            'ID': participant_id,
            'num_items': 0,
            'num_valid_items': 0,
            'word_coverage': 0,
            'items': [],
            'valid_items': [],
            'vectors': [],
            'similarities': [],
            'phases': [],
            'exploitation_percentage': 0,
            'exploration_percentage': 0,
            'num_switches': 0,
            'avg_exploitation_size': 0,
            'avg_exploration_size': 0,
            'novelty_score': 0,
            'clustering_coefficient': 0,
            'num_clusters': 0,
            'mean_rank': 0,
            'median_rank': 0,
            'std_rank': 0,
            'min_rank': 0,
            'max_rank': 0
        }
    
    def analyze_all_participants(self) -> pd.DataFrame:
        """Analyze all participants and return results as DataFrame"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting analysis of all participants...")
        
        results = []
        participant_ids = self.data['ID'].unique()
        
        for participant_id in participant_ids:
            participant_data = self.data[self.data['ID'] == participant_id]
            result = self.analyze_participant(participant_data)
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results
        
        logger.info(f"Completed analysis of {len(results_df)} participants")
        return results_df
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all participants"""
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        summary = {
            'total_participants': len(results_df),
            'avg_items_per_participant': results_df['num_items'].mean(),
            'avg_valid_items_per_participant': results_df['num_valid_items'].mean(),
            'avg_word_coverage': results_df['word_coverage'].mean(),
            'avg_exploitation_percentage': results_df['exploitation_percentage'].mean(),
            'avg_exploration_percentage': results_df['exploration_percentage'].mean(),
            'avg_num_switches': results_df['num_switches'].mean(),
            'avg_novelty_score': results_df['novelty_score'].mean(),
            'avg_clustering_coefficient': results_df['clustering_coefficient'].mean()
        }
        
        return summary
