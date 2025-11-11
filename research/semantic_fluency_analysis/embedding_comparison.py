#!/usr/bin/env python3
"""
Comparative Analysis of Word Embedding Models for Semantic Fluency Analysis
"""

import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModelComparison:
    """Compare different word embedding models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.test_words = []
        self.fluency_data = None
        
    def load_fluency_data(self):
        """Load fluency data to get real test words"""
        try:
            self.fluency_data = pd.read_csv('data/fluency_data.csv')
            # Get unique words from fluency data
            self.test_words = self.fluency_data['Item'].unique().tolist()
            logger.info(f"Loaded {len(self.test_words)} unique words from fluency data")
        except Exception as e:
            logger.warning(f"Could not load fluency data: {e}")
            # Fallback to common animal words
            self.test_words = [
                'cat', 'dog', 'lion', 'tiger', 'elephant', 'giraffe', 'monkey', 
                'zebra', 'bear', 'wolf', 'fox', 'deer', 'rabbit', 'squirrel',
                'mouse', 'rat', 'hamster', 'guinea', 'pig', 'horse', 'cow',
                'sheep', 'goat', 'pig', 'chicken', 'duck', 'turkey', 'penguin',
                'eagle', 'hawk', 'owl', 'crow', 'sparrow', 'robin', 'bluejay'
            ]
    
    def load_spacy_model(self):
        """Load spaCy model"""
        try:
            import spacy
            logger.info("Loading spaCy model...")
            start_time = time.time()
            nlp = spacy.load("en_core_web_md")
            load_time = time.time() - start_time
            
            self.models['spacy'] = {
                'model': nlp,
                'name': 'spaCy (en_core_web_md)',
                'load_time': load_time,
                'vocab_size': len(nlp.vocab),
                'vector_dim': nlp.vocab.vectors.shape[1] if nlp.vocab.vectors.shape[0] > 0 else 0
            }
            logger.info(f"spaCy loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load spaCy: {e}")
    
    def load_gensim_model(self):
        """Load Gensim Word2Vec model"""
        try:
            from gensim.models import KeyedVectors
            logger.info("Loading Gensim Word2Vec model...")
            start_time = time.time()
            
            # Try to load pre-trained model, fallback to training on fluency data
            try:
                # This would require downloading the model first
                # word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
                # For now, we'll create a simple model from our data
                from gensim.models import Word2Vec
                sentences = []
                for _, group in self.fluency_data.groupby('ID'):
                    words = group['Item'].tolist()
                    sentences.append(words)
                
                model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
                word_vectors = model.wv
                
            except Exception as e:
                logger.warning(f"Could not load pre-trained Word2Vec: {e}")
                return
            
            load_time = time.time() - start_time
            
            self.models['gensim'] = {
                'model': word_vectors,
                'name': 'Gensim Word2Vec',
                'load_time': load_time,
                'vocab_size': len(word_vectors.key_to_index),
                'vector_dim': word_vectors.vector_size
            }
            logger.info(f"Gensim Word2Vec loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Gensim: {e}")
    
    def load_fasttext_model(self):
        """Load FastText model"""
        try:
            import fasttext
            logger.info("Loading FastText model...")
            start_time = time.time()
            
            # Try to load pre-trained model
            try:
                # This would require downloading the model first
                # model = fasttext.load_model('cc.en.300.bin')
                # For now, we'll skip FastText as it requires large model download
                logger.warning("FastText model not available (requires large download)")
                return
            except Exception as e:
                logger.warning(f"Could not load FastText: {e}")
                return
            
        except Exception as e:
            logger.error(f"Failed to load FastText: {e}")
    
    def load_transformers_model(self):
        """Load Transformers model (BERT/RoBERTa)"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            logger.info("Loading Transformers model...")
            start_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            model = AutoModel.from_pretrained('roberta-base')
            
            load_time = time.time() - start_time
            
            self.models['transformers'] = {
                'model': model,
                'tokenizer': tokenizer,
                'name': 'RoBERTa (Transformers)',
                'load_time': load_time,
                'vocab_size': tokenizer.vocab_size,
                'vector_dim': model.config.hidden_size
            }
            logger.info(f"Transformers model loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Transformers: {e}")
    
    def get_word_vector(self, model_name: str, word: str) -> Optional[np.ndarray]:
        """Get word vector from specified model"""
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name]
        
        try:
            if model_name == 'spacy':
                doc = model_info['model'](word.lower())
                if doc.has_vector:
                    return doc.vector
                return None
            
            elif model_name == 'gensim':
                if word.lower() in model_info['model'].key_to_index:
                    return model_info['model'][word.lower()]
                return None
            
            elif model_name == 'transformers':
                # For transformers, we need to get contextual embeddings
                # This is simplified - in practice you'd want to average across contexts
                tokenizer = model_info['tokenizer']
                model = model_info['model']
                
                inputs = tokenizer(word.lower(), return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use the [CLS] token embedding as word representation
                    return outputs.last_hidden_state[0, 0, :].numpy()
            
        except Exception as e:
            logger.warning(f"Error getting vector for '{word}' from {model_name}: {e}")
            return None
        
        return None
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0
    
    def test_word_coverage(self) -> Dict:
        """Test how many words each model can handle"""
        logger.info("Testing word coverage...")
        
        coverage_results = {}
        
        for model_name in self.models.keys():
            covered_words = 0
            total_words = len(self.test_words)
            
            for word in self.test_words:
                vector = self.get_word_vector(model_name, word)
                if vector is not None:
                    covered_words += 1
            
            coverage = covered_words / total_words * 100
            coverage_results[model_name] = {
                'covered': covered_words,
                'total': total_words,
                'coverage_percent': coverage
            }
            
            logger.info(f"{model_name}: {covered_words}/{total_words} words ({coverage:.1f}%)")
        
        return coverage_results
    
    def test_similarity_consistency(self) -> Dict:
        """Test similarity consistency between models"""
        logger.info("Testing similarity consistency...")
        
        # Create word pairs for testing
        word_pairs = [
            ('cat', 'dog'), ('lion', 'tiger'), ('elephant', 'giraffe'),
            ('monkey', 'gorilla'), ('wolf', 'fox'), ('bear', 'panda'),
            ('eagle', 'hawk'), ('owl', 'crow'), ('penguin', 'duck'),
            ('cat', 'elephant'), ('dog', 'penguin'), ('lion', 'mouse')
        ]
        
        similarity_results = {}
        
        for model_name in self.models.keys():
            similarities = []
            valid_pairs = 0
            
            for word1, word2 in word_pairs:
                vec1 = self.get_word_vector(model_name, word1)
                vec2 = self.get_word_vector(model_name, word2)
                
                if vec1 is not None and vec2 is not None:
                    sim = self.calculate_similarity(vec1, vec2)
                    similarities.append(sim)
                    valid_pairs += 1
            
            if similarities:
                similarity_results[model_name] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'valid_pairs': valid_pairs,
                    'similarities': similarities
                }
        
        return similarity_results
    
    def test_fluency_sequence_similarity(self) -> Dict:
        """Test similarity calculations on actual fluency sequences"""
        logger.info("Testing fluency sequence similarity...")
        
        if self.fluency_data is None:
            logger.warning("No fluency data available for sequence testing")
            return {}
        
        sequence_results = {}
        
        # Get a few participant sequences
        participant_sequences = []
        for participant_id, group in self.fluency_data.groupby('ID'):
            if len(group) >= 5:  # Only use sequences with at least 5 words
                words = group['Item'].tolist()
                participant_sequences.append((participant_id, words))
                if len(participant_sequences) >= 3:  # Test with 3 participants
                    break
        
        for model_name in self.models.keys():
            model_similarities = []
            processing_times = []
            
            for participant_id, words in participant_sequences:
                start_time = time.time()
                
                # Calculate similarities between consecutive words
                sequence_similarities = []
                for i in range(len(words) - 1):
                    vec1 = self.get_word_vector(model_name, words[i])
                    vec2 = self.get_word_vector(model_name, words[i + 1])
                    
                    if vec1 is not None and vec2 is not None:
                        sim = self.calculate_similarity(vec1, vec2)
                        sequence_similarities.append(sim)
                
                processing_time = time.time() - start_time
                
                if sequence_similarities:
                    model_similarities.extend(sequence_similarities)
                    processing_times.append(processing_time)
            
            if model_similarities:
                sequence_results[model_name] = {
                    'mean_similarity': np.mean(model_similarities),
                    'std_similarity': np.std(model_similarities),
                    'avg_processing_time': np.mean(processing_times),
                    'total_similarities': len(model_similarities)
                }
        
        return sequence_results
    
    def test_performance(self) -> Dict:
        """Test performance metrics"""
        logger.info("Testing performance...")
        
        performance_results = {}
        
        for model_name in self.models.keys():
            model_info = self.models[model_name]
            
            # Test vector retrieval speed
            start_time = time.time()
            vectors_retrieved = 0
            
            for word in self.test_words[:100]:  # Test with first 100 words
                vector = self.get_word_vector(model_name, word)
                if vector is not None: # Corrected from not not None to None
                    vectors_retrieved += 1
            
            retrieval_time = time.time() - start_time
            
            performance_results[model_name] = {
                'load_time': model_info['load_time'],
                'retrieval_time_per_word': retrieval_time / 100,
                'vectors_retrieved': vectors_retrieved,
                'vocab_size': model_info['vocab_size'],
                'vector_dim': model_info['vector_dim']
            }
        
        return performance_results
    
    def run_comparison(self) -> Dict:
        """Run complete comparison"""
        logger.info("Starting embedding model comparison...")
        
        # Load data
        self.load_fluency_data()
        
        # Load models
        self.load_spacy_model()
        self.load_gensim_model()
        self.load_fasttext_model()
        self.load_transformers_model()
        
        if not self.models:
            logger.error("No models loaded successfully")
            return {}
        
        # Run tests
        results = {
            'coverage': self.test_word_coverage(),
            'similarity_consistency': self.test_similarity_consistency(),
            'fluency_sequence': self.test_fluency_sequence_similarity(),
            'performance': self.test_performance()
        }
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate comparison report"""
        if not self.results:
            logger.error("No results available. Run comparison first.")
            return ""
        
        report_path = Path('output/embedding_comparison_report.txt')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("WORD EMBEDDING MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Model overview
            f.write("MODEL OVERVIEW\n")
            f.write("-" * 15 + "\n")
            for model_name, model_info in self.models.items():
                f.write(f"{model_info['name']}:\n")
                f.write(f"  Load time: {model_info['load_time']:.2f}s\n")
                f.write(f"  Vocabulary size: {model_info['vocab_size']:,}\n")
                f.write(f"  Vector dimension: {model_info['vector_dim']}\n\n")
            
            # Coverage results
            f.write("WORD COVERAGE\n")
            f.write("-" * 15 + "\n")
            for model_name, coverage in self.results['coverage'].items():
                model_name_display = self.models[model_name]['name']
                f.write(f"{model_name_display}:\n")
                f.write(f"  Covered: {coverage['covered']}/{coverage['total']} words\n")
                f.write(f"  Coverage: {coverage['coverage_percent']:.1f}%\n\n")
            
            # Performance results
            f.write("PERFORMANCE\n")
            f.write("-" * 12 + "\n")
            for model_name, perf in self.results['performance'].items():
                model_name_display = self.models[model_name]['name']
                f.write(f"{model_name_display}:\n")
                f.write(f"  Load time: {perf['load_time']:.2f}s\n")
                f.write(f"  Retrieval time per word: {perf['retrieval_time_per_word']*1000:.2f}ms\n")
                f.write(f"  Vectors retrieved: {perf['vectors_retrieved']}/100\n\n")
            
            # Similarity results
            f.write("SIMILARITY CONSISTENCY\n")
            f.write("-" * 25 + "\n")
            for model_name, sim in self.results['similarity_consistency'].items():
                model_name_display = self.models[model_name]['name']
                f.write(f"{model_name_display}:\n")
                f.write(f"  Mean similarity: {sim['mean_similarity']:.3f}\n")
                f.write(f"  Std similarity: {sim['std_similarity']:.3f}\n")
                f.write(f"  Valid pairs: {sim['valid_pairs']}\n\n")
            
            # Fluency sequence results
            f.write("FLUENCY SEQUENCE ANALYSIS\n")
            f.write("-" * 28 + "\n")
            for model_name, seq in self.results['fluency_sequence'].items():
                model_name_display = self.models[model_name]['name']
                f.write(f"{model_name_display}:\n")
                f.write(f"  Mean sequence similarity: {seq['mean_similarity']:.3f}\n")
                f.write(f"  Std sequence similarity: {seq['std_similarity']:.3f}\n")
                f.write(f"  Avg processing time: {seq['avg_processing_time']*1000:.2f}ms\n")
                f.write(f"  Total similarities: {seq['total_similarities']}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Find best coverage
            best_coverage = max(self.results['coverage'].items(), 
                              key=lambda x: x[1]['coverage_percent'])
            f.write(f"Best word coverage: {self.models[best_coverage[0]]['name']} ({best_coverage[1]['coverage_percent']:.1f}%)\n")
            
            # Find fastest
            fastest = min(self.results['performance'].items(), 
                         key=lambda x: x[1]['retrieval_time_per_word'])
            f.write(f"Fastest retrieval: {self.models[fastest[0]]['name']} ({fastest[1]['retrieval_time_per_word']*1000:.2f}ms per word)\n")
            
            # Find most consistent
            if self.results['similarity_consistency']:
                most_consistent = max(self.results['similarity_consistency'].items(), 
                                    key=lambda x: x[1]['mean_similarity'])
                f.write(f"Most consistent similarities: {self.models[most_consistent[0]]['name']} (mean: {most_consistent[1]['mean_similarity']:.3f})\n")
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def plot_comparison(self):
        """Create comparison plots"""
        if not self.results:
            logger.error("No results available. Run comparison first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Word Embedding Model Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Coverage comparison
        model_names = [self.models[name]['name'] for name in self.results['coverage'].keys()]
        coverage_values = [self.results['coverage'][name]['coverage_percent'] for name in self.results['coverage'].keys()]
        
        axes[0, 0].bar(model_names, coverage_values, color=['blue', 'green', 'red', 'orange'])
        axes[0, 0].set_title('Word Coverage (%)', fontweight='bold')
        axes[0, 0].set_ylabel('Coverage (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance comparison
        model_names = [self.models[name]['name'] for name in self.results['performance'].keys()]
        retrieval_times = [self.results['performance'][name]['retrieval_time_per_word']*1000 for name in self.results['performance'].keys()]
        
        axes[0, 1].bar(model_names, retrieval_times, color=['blue', 'green', 'red', 'orange'])
        axes[0, 1].set_title('Retrieval Time (ms per word)', fontweight='bold')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Similarity consistency
        if self.results['similarity_consistency']:
            model_names = [self.models[name]['name'] for name in self.results['similarity_consistency'].keys()]
            mean_similarities = [self.results['similarity_consistency'][name]['mean_similarity'] for name in self.results['similarity_consistency'].keys()]
            
            axes[1, 0].bar(model_names, mean_similarities, color=['blue', 'green', 'red', 'orange'])
            axes[1, 0].set_title('Mean Similarity Score', fontweight='bold')
            axes[1, 0].set_ylabel('Similarity')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Fluency sequence similarity
        if self.results['fluency_sequence']:
            model_names = [self.models[name]['name'] for name in self.results['fluency_sequence'].keys()]
            seq_similarities = [self.results['fluency_sequence'][name]['mean_similarity'] for name in self.results['fluency_sequence'].keys()]
            
            axes[1, 1].bar(model_names, seq_similarities, color=['blue', 'green', 'red', 'orange'])
            axes[1, 1].set_title('Fluency Sequence Similarity', fontweight='bold')
            axes[1, 1].set_ylabel('Similarity')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('output/embedding_comparison.png')
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Comparison plot saved: {plot_path}")

def main():
    """Run the comparison"""
    print("🔍 Word Embedding Model Comparison")
    print("=" * 40)
    
    comparison = EmbeddingModelComparison()
    results = comparison.run_comparison()
    
    if results:
        print("\n✅ Comparison completed!")
        
        # Generate report
        report_path = comparison.generate_report()
        print(f"📋 Report generated: {report_path}")
        
        # Create plots
        comparison.plot_comparison()
        print("📊 Comparison plots created")
        
        # Print summary
        print("\n📊 SUMMARY:")
        for model_name, coverage in results['coverage'].items():
            model_display = comparison.models[model_name]['name']
            print(f"   {model_display}: {coverage['coverage_percent']:.1f}% coverage")
        
        # Find best model
        best_coverage = max(results['coverage'].items(), key=lambda x: x[1]['coverage_percent'])
        best_model = comparison.models[best_coverage[0]]['name']
        print(f"\n🏆 RECOMMENDATION: {best_model}")
        
    else:
        print("❌ Comparison failed")

if __name__ == "__main__":
    main()
