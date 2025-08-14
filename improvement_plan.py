#!/usr/bin/env python3
"""
IMPROVEMENT PLAN: Refactored Semantic Fluency Analysis System
============================================================

This file demonstrates how to improve the original mediation script by:
1. Creating a class-based architecture
2. Extracting configuration
3. Improving performance
4. Adding proper error handling
5. Implementing testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import yaml
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import unittest
from wordfreq import zipf_frequency, top_n_list
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration class for analysis parameters"""
    semantic_weight: float = 0.7
    clustering_threshold: float = 0.5
    similarity_threshold: Optional[float] = None  # Will be calculated from data
    cache_size: int = 1000
    output_dir: str = "output"
    save_plots: bool = True
    plot_format: str = "svg"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AnalysisConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return cls()

# =============================================================================
# 2. CORE ANALYSIS CLASS
# =============================================================================

class SemanticFluencyAnalyzer:
    """
    Main class for semantic fluency analysis with improved architecture
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.nlp = None
        self.word_ranks = None
        self.data = None
        self.meg_data = None
        self.results = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Initialize spaCy and word frequency data"""
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.error("spaCy model not found. Please run: python -m spacy download en_core_web_md")
            raise
        
        # Load word frequency data once
        try:
            top_50k = top_n_list('en', 50000)
            self.word_ranks = {word: rank+1 for rank, word in enumerate(top_50k)}
            logger.info("Loaded word frequency data")
        except Exception as e:
            logger.warning(f"Could not load word frequency data: {e}")
            self.word_ranks = {}
    
    def load_data(self, fluency_path: str, meg_path: str) -> None:
        """Load and validate data from external files"""
        try:
            # Load fluency data
            self.data = pd.read_csv(fluency_path)
            logger.info(f"Loaded fluency data: {len(self.data)} participants")
            
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
        
        logger.info("Data validation passed")

# =============================================================================
# 3. OPTIMIZED UTILITY FUNCTIONS
# =============================================================================

class SemanticUtils:
    """Optimized utility functions with caching"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cosine_similarity(vec1: Tuple[float, ...], vec2: Tuple[float, ...]) -> float:
        """Calculate cosine similarity with caching"""
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        return np.dot(vec1_array, vec2_array) / (np.linalg.norm(vec1_array) * np.linalg.norm(vec2_array))
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_word_frequency(word: str) -> float:
        """Get word frequency with caching"""
        return zipf_frequency(word.lower(), 'en')
    
    @staticmethod
    def get_word_rank(word: str, word_ranks: Dict[str, int]) -> int:
        """Get word rank from pre-loaded dictionary"""
        return word_ranks.get(word.lower(), 0)
    
    @staticmethod
    def calculate_similarities_vectorized(vectors: List[np.ndarray]) -> np.ndarray:
        """Vectorized similarity calculation"""
        if len(vectors) < 2:
            return np.array([])
        
        vectors_array = np.array(vectors)
        similarities = []
        
        for i in range(len(vectors_array) - 1):
            sim = SemanticUtils.cosine_similarity(
                tuple(vectors_array[i]), 
                tuple(vectors_array[i+1])
            )
            similarities.append(sim)
        
        return np.array(similarities)

# =============================================================================
# 4. PHASE ANALYSIS WITH IMPROVED ALGORITHM
# =============================================================================

class PhaseAnalyzer:
    """Improved phase identification and analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def identify_phases(self, similarities: np.ndarray, items: List[str], 
                       vectors: List[np.ndarray], threshold: float) -> List[Dict]:
        """
        Identify exploitation and exploration phases with improved algorithm
        """
        if len(similarities) < 1:
            return []
        
        phases = []
        current_phase = "Exploitation" if similarities[0] > threshold else "Exploration"
        phase_start = 0
        
        for i in range(1, len(similarities)):
            # Check for phase transition
            if self._should_transition(current_phase, similarities[i], threshold):
                phase = self._create_phase_dict(
                    current_phase, phase_start, i, items, vectors, similarities
                )
                phases.append(phase)
                
                # Switch phase
                current_phase = "Exploration" if current_phase == "Exploitation" else "Exploitation"
                phase_start = i
        
        # Add final phase
        final_phase = self._create_phase_dict(
            current_phase, phase_start, len(similarities), items, vectors, similarities
        )
        phases.append(final_phase)
        
        return phases
    
    def _should_transition(self, current_phase: str, similarity: float, threshold: float) -> bool:
        """Determine if phase should transition"""
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
            'avg_similarity': np.mean(phase_similarities) if phase_similarities else 0
        }

# =============================================================================
# 5. PARTICIPANT ANALYSIS WITH ERROR HANDLING
# =============================================================================

class ParticipantAnalyzer:
    """Analyze individual participants with comprehensive error handling"""
    
    def __init__(self, config: AnalysisConfig, nlp, word_ranks: Dict[str, int]):
        self.config = config
        self.nlp = nlp
        self.word_ranks = word_ranks
        self.phase_analyzer = PhaseAnalyzer(config)
    
    def analyze_participant(self, participant_data: pd.DataFrame) -> Dict:
        """Analyze a single participant with error handling"""
        try:
            participant_id = participant_data['ID'].iloc[0]
            items = participant_data['Item'].tolist()
            
            logger.info(f"Analyzing participant {participant_id}")
            
            # Get word vectors
            vectors = self._get_word_vectors(items)
            if not vectors:
                logger.warning(f"No valid vectors for participant {participant_id}")
                return self._create_empty_result(participant_id)
            
            # Calculate similarities
            similarities = SemanticUtils.calculate_similarities_vectorized(vectors)
            
            # Determine threshold if not set
            threshold = self.config.similarity_threshold
            if threshold is None:
                threshold = np.mean(similarities) if len(similarities) > 0 else 0.5
            
            # Identify phases
            phases = self.phase_analyzer.identify_phases(similarities, items, vectors, threshold)
            
            # Calculate metrics
            metrics = self._calculate_metrics(phases, similarities, items)
            
            return {
                'participant_id': participant_id,
                'items': items,
                'vectors': vectors,
                'similarities': similarities,
                'phases': phases,
                'threshold': threshold,
                **metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing participant {participant_id}: {e}")
            return self._create_empty_result(participant_id)
    
    def _get_word_vectors(self, items: List[str]) -> List[np.ndarray]:
        """Get word vectors with error handling"""
        vectors = []
        for item in items:
            try:
                doc = self.nlp(item.strip().lower())
                if doc.has_vector:
                    vectors.append(doc.vector)
                else:
                    logger.warning(f"No vector for word: {item}")
            except Exception as e:
                logger.warning(f"Error processing word {item}: {e}")
        
        return vectors
    
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
        
        return {
            'num_items': len(items),
            'avg_similarity': np.mean(similarities) if len(similarities) > 0 else 0,
            'num_phases': len(phases),
            'num_switches': num_switches,
            'exploitation_time': exploitation_time,
            'exploration_time': exploration_time,
            'exploitation_percentage': exploitation_percentage,
            'exploration_percentage': exploration_percentage,
            'exploitation_mean_phase_size': np.mean(exploitation_sizes) if exploitation_sizes else 0,
            'exploration_mean_phase_size': np.mean(exploration_sizes) if exploration_sizes else 0,
            'novelty_ratio': sum(novelty_scores) / len(items) if items else 0,
            'ee_tradeoff': np.mean(exploitation_sizes + exploration_sizes) / num_switches if num_switches > 0 else 0
        }
    
    def _create_empty_result(self, participant_id: str) -> Dict:
        """Create empty result for failed analysis"""
        return {
            'participant_id': participant_id,
            'items': [],
            'vectors': [],
            'similarities': [],
            'phases': [],
            'threshold': 0,
            'num_items': 0,
            'avg_similarity': 0,
            'num_phases': 0,
            'num_switches': 0,
            'exploitation_time': 0,
            'exploration_time': 0,
            'exploitation_percentage': 0,
            'exploration_percentage': 0,
            'exploitation_mean_phase_size': 0,
            'exploration_mean_phase_size': 0,
            'novelty_ratio': 0,
            'ee_tradeoff': 0
        }

# =============================================================================
# 6. VISUALIZATION MODULE
# =============================================================================

class VisualizationManager:
    """Centralized visualization management"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_participant_analysis(self, participant_result: Dict, save: bool = True) -> None:
        """Create comprehensive participant visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Semantic Navigation Analysis - {participant_result["participant_id"]}', fontsize=16)
        
        # Plot 1: Similarity scores
        self._plot_similarities(axes[0, 0], participant_result)
        
        # Plot 2: Phase transitions
        self._plot_phase_transitions(axes[0, 1], participant_result)
        
        # Plot 3: Exploitation vs Exploration
        self._plot_ee_percentages(axes[1, 0], participant_result)
        
        # Plot 4: Metrics summary
        self._plot_metrics_summary(axes[1, 1], participant_result)
        
        plt.tight_layout()
        
        if save and self.config.save_plots:
            filename = f"participant_{participant_result['participant_id']}.{self.config.plot_format}"
            plt.savefig(self.output_dir / filename, format=self.config.plot_format, 
                       bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def _plot_similarities(self, ax, result: Dict):
        """Plot similarity scores"""
        similarities = result['similarities']
        if len(similarities) > 0:
            x = range(len(similarities))
            ax.plot(x, similarities, 'b-', marker='o', markersize=4)
            ax.axhline(y=result['threshold'], color='r', linestyle='--', label='Threshold')
            ax.set_title('Semantic Similarity Scores')
            ax.set_xlabel('Transition Position')
            ax.set_ylabel('Similarity Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_phase_transitions(self, ax, result: Dict):
        """Plot phase transitions"""
        phases = result['phases']
        if phases:
            phase_numbers = []
            phase_types = []
            for phase in phases:
                phase_numbers.extend([phase['start'], phase['end']])
                phase_types.extend([phase['type'], phase['type']])
            
            # Create step plot
            phase_values = [1 if pt == 'Exploitation' else 0 for pt in phase_types]
            ax.step(phase_numbers, phase_values, where='post', linewidth=2, color='purple')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Exploration', 'Exploitation'])
            ax.set_title('Phase Transitions')
            ax.set_xlabel('Word Position')
            ax.grid(True, alpha=0.3)
    
    def _plot_ee_percentages(self, ax, result: Dict):
        """Plot exploitation vs exploration percentages"""
        labels = ['Exploitation', 'Exploration']
        sizes = [result['exploitation_percentage'], result['exploration_percentage']]
        colors = ['#2ca02c', '#d62728']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.set_title('Exploitation vs Exploration')
    
    def _plot_metrics_summary(self, ax, result: Dict):
        """Plot metrics summary"""
        metrics = ['exploitation_percentage', 'exploration_percentage', 
                  'novelty_ratio', 'ee_tradeoff']
        values = [result[metric] for metric in metrics]
        labels = ['Exploit %', 'Explore %', 'Novelty', 'EE Tradeoff']
        
        bars = ax.bar(labels, values, color=['green', 'red', 'blue', 'orange'])
        ax.set_title('Key Metrics')
        ax.set_ylabel('Score')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')

# =============================================================================
# 7. MAIN ANALYSIS PIPELINE
# =============================================================================

class SemanticFluencyPipeline:
    """Main pipeline for semantic fluency analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = AnalysisConfig.from_yaml(config_path)
        self.analyzer = SemanticFluencyAnalyzer(self.config)
        self.participant_analyzer = None
        self.viz_manager = VisualizationManager(self.config)
        self.results = []
    
    def run_analysis(self, fluency_path: str, meg_path: str) -> pd.DataFrame:
        """Run complete analysis pipeline"""
        try:
            # Load data
            self.analyzer.load_data(fluency_path, meg_path)
            
            # Initialize participant analyzer
            self.participant_analyzer = ParticipantAnalyzer(
                self.config, self.analyzer.nlp, self.analyzer.word_ranks
            )
            
            # Analyze each participant
            self.results = []
            for participant_id, group in self.analyzer.data.groupby('ID'):
                result = self.participant_analyzer.analyze_participant(group)
                self.results.append(result)
            
            # Create results dataframe
            results_df = pd.DataFrame(self.results)
            
            # Merge with MEG data
            if self.analyzer.meg_data is not None:
                results_df = pd.merge(
                    results_df, 
                    self.analyzer.meg_data, 
                    left_on='participant_id', 
                    right_on='ID', 
                    how='left'
                )
            
            logger.info(f"Analysis complete. Processed {len(self.results)} participants")
            return results_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def generate_visualizations(self, results_df: pd.DataFrame, 
                              participant_ids: Optional[List[str]] = None) -> None:
        """Generate visualizations for participants"""
        if participant_ids is None:
            participant_ids = results_df['participant_id'].tolist()
        
        for participant_id in participant_ids:
            participant_result = results_df[results_df['participant_id'] == participant_id].iloc[0].to_dict()
            self.viz_manager.plot_participant_analysis(participant_result)
    
    def save_results(self, results_df: pd.DataFrame, filename: str = "analysis_results.csv") -> None:
        """Save results to file"""
        output_path = self.config.output_dir / filename
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

# =============================================================================
# 8. TESTING FRAMEWORK
# =============================================================================

class TestSemanticAnalysis(unittest.TestCase):
    """Unit tests for semantic analysis functions"""
    
    def setUp(self):
        self.config = AnalysisConfig()
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = (1.0, 0.0, 0.0)
        vec2 = (0.0, 1.0, 0.0)
        result = SemanticUtils.cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)
        
        # Test identical vectors
        result = SemanticUtils.cosine_similarity(vec1, vec1)
        self.assertEqual(result, 1.0)
    
    def test_phase_identification(self):
        """Test phase identification"""
        similarities = [0.8, 0.7, 0.3, 0.2, 0.8, 0.9]
        items = ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger']
        vectors = [np.random.rand(300) for _ in range(6)]
        threshold = 0.5
        
        phase_analyzer = PhaseAnalyzer(self.config)
        phases = phase_analyzer.identify_phases(similarities, items, vectors, threshold)
        
        self.assertGreater(len(phases), 0)
        self.assertIn('type', phases[0])
        self.assertIn('start', phases[0])
        self.assertIn('end', phases[0])

# =============================================================================
# 9. USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of the improved system"""
    
    # Create configuration file
    config_data = {
        'semantic_weight': 0.7,
        'clustering_threshold': 0.5,
        'similarity_threshold': None,  # Will be calculated from data
        'cache_size': 1000,
        'output_dir': 'output',
        'save_plots': True,
        'plot_format': 'svg'
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config_data, f)
    
    # Run analysis
    pipeline = SemanticFluencyPipeline('config.yaml')
    
    try:
        # Run analysis (replace with actual file paths)
        results_df = pipeline.run_analysis(
            fluency_path='fluency_data.csv',
            meg_path='meg_data.csv'
        )
        
        # Generate visualizations for first 3 participants
        participant_ids = results_df['participant_id'].head(3).tolist()
        pipeline.generate_visualizations(results_df, participant_ids)
        
        # Save results
        pipeline.save_results(results_df)
        
        print("Analysis completed successfully!")
        print(f"Processed {len(results_df)} participants")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
