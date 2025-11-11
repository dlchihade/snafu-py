"""
Visualization module for semantic fluency analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .config import AnalysisConfig

logger = logging.getLogger(__name__)

class VisualizationManager:
    """Centralized visualization management"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        sns.set_style("whitegrid")
        
        # Set figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def plot_participant_analysis(self, participant_result: Dict, save: bool = True) -> None:
        """Create comprehensive participant visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Semantic Navigation Analysis - {participant_result["ID"]}', 
                    fontsize=16, fontweight='bold')
        
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
            logger.info(f"Saved participant plot: {filename}")
        
        plt.show()
    
    def _plot_similarities(self, ax, result: Dict):
        """Plot similarity scores over time"""
        similarities = result['similarities']
        if len(similarities) > 0:
            x = range(len(similarities))
            ax.plot(x, similarities, 'b-', marker='o', markersize=4, linewidth=2, alpha=0.7)
            ax.axhline(y=0.6, color='r', linestyle='--', 
                                              label=f'Threshold (0.600)', linewidth=2)
            ax.set_title('Semantic Similarity Scores', fontweight='bold')
            ax.set_xlabel('Transition Position')
            ax.set_ylabel('Similarity Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_sim = np.mean(similarities)
            ax.text(0.02, 0.98, f'Mean: {mean_sim:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
            ax.step(phase_numbers, phase_values, where='post', linewidth=3, color='purple', alpha=0.8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Exploration', 'Exploitation'])
            ax.set_title('Phase Transitions', fontweight='bold')
            ax.set_xlabel('Word Position')
            ax.grid(True, alpha=0.3)
            
            # Add phase count
            ax.text(0.02, 0.98, f'Phases: {len(phases)}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _plot_ee_percentages(self, ax, result: Dict):
        """Plot exploitation vs exploration percentages"""
        labels = ['Exploitation', 'Exploration']
        sizes = [result['exploitation_percentage'], result['exploration_percentage']]
        colors = ['#2ca02c', '#d62728']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=140, explode=(0.05, 0.05))
        ax.set_title('Exploitation vs Exploration', fontweight='bold')
        
        # Make autotexts more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_metrics_summary(self, ax, result: Dict):
        """Plot metrics summary"""
        metrics = ['exploitation_percentage', 'exploration_percentage', 
                  'novelty_ratio', 'ee_tradeoff']
        values = [result[metric] for metric in metrics]
        labels = ['Exploit %', 'Explore %', 'Novelty', 'EE Tradeoff']
        colors = ['green', 'red', 'blue', 'orange']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_title('Key Metrics', fontweight='bold')
        ax.set_ylabel('Score')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_summary_statistics(self, results_df: pd.DataFrame, save: bool = True) -> None:
        """Create summary statistics plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Semantic Fluency Analysis - Summary Statistics', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of exploitation percentages
        self._plot_exploitation_distribution(axes[0, 0], results_df)
        
        # Plot 2: Distribution of similarity scores
        self._plot_similarity_distribution(axes[0, 1], results_df)
        
        # Plot 3: Number of phases distribution
        self._plot_phases_distribution(axes[0, 2], results_df)
        
        # Plot 4: Novelty ratio vs exploitation percentage
        self._plot_novelty_vs_exploitation(axes[1, 0], results_df)
        
        # Plot 5: Number of items vs exploitation percentage
        self._plot_items_vs_exploitation(axes[1, 1], results_df)
        
        # Plot 6: Correlation matrix
        self._plot_correlation_matrix(axes[1, 2], results_df)
        
        plt.tight_layout()
        
        if save and self.config.save_plots:
            filename = f"summary_statistics.{self.config.plot_format}"
            plt.savefig(self.output_dir / filename, format=self.config.plot_format, 
                       bbox_inches='tight', dpi=300)
            logger.info(f"Saved summary statistics plot: {filename}")
        
        plt.show()
    
    def _plot_exploitation_distribution(self, ax, results_df: pd.DataFrame):
        """Plot distribution of exploitation percentages"""
        ax.hist(results_df['exploitation_percentage'], bins=15, alpha=0.7, 
               color='green', edgecolor='black')
        ax.set_title('Exploitation Percentage Distribution', fontweight='bold')
        ax.set_xlabel('Exploitation Percentage')
        ax.set_ylabel('Frequency')
        ax.axvline(results_df['exploitation_percentage'].mean(), color='red', 
                  linestyle='--', label=f'Mean: {results_df["exploitation_percentage"].mean():.1f}%')
        ax.legend()
    
    def _plot_similarity_distribution(self, ax, results_df: pd.DataFrame):
        """Plot distribution of similarity scores"""
        # Calculate average similarity from similarities column
        avg_similarities = []
        for _, row in results_df.iterrows():
            if row['similarities'] and len(row['similarities']) > 0:
                avg_similarities.append(np.mean(row['similarities']))
            else:
                avg_similarities.append(0)
        
        ax.hist(avg_similarities, bins=15, alpha=0.7, 
               color='blue', edgecolor='black')
        ax.set_title('Average Similarity Distribution', fontweight='bold')
        ax.set_xlabel('Average Similarity')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(avg_similarities), color='red', 
                  linestyle='--', label=f'Mean: {np.mean(avg_similarities):.3f}')
        ax.legend()
    
    def _plot_phases_distribution(self, ax, results_df: pd.DataFrame):
        """Plot distribution of number of phases"""
        # Calculate number of phases from phases column
        num_phases = []
        for _, row in results_df.iterrows():
            num_phases.append(len(row['phases']))
        
        if num_phases:
            ax.hist(num_phases, bins=range(1, max(num_phases)+2), 
                   alpha=0.7, color='purple', edgecolor='black')
            ax.set_title('Number of Phases Distribution', fontweight='bold')
            ax.set_xlabel('Number of Phases')
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(num_phases), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(num_phases):.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No phase data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Number of Phases Distribution', fontweight='bold')
    
    def _plot_novelty_vs_exploitation(self, ax, results_df: pd.DataFrame):
        """Plot novelty score vs exploitation percentage"""
        ax.scatter(results_df['exploitation_percentage'], results_df['novelty_score'], 
                  alpha=0.6, color='orange')
        ax.set_title('Novelty vs Exploitation', fontweight='bold')
        ax.set_xlabel('Exploitation Percentage')
        ax.set_ylabel('Novelty Score')
        
        # Add trend line
        z = np.polyfit(results_df['exploitation_percentage'], results_df['novelty_score'], 1)
        p = np.poly1d(z)
        ax.plot(results_df['exploitation_percentage'], p(results_df['exploitation_percentage']), 
               "r--", alpha=0.8)
    
    def _plot_items_vs_exploitation(self, ax, results_df: pd.DataFrame):
        """Plot number of items vs exploitation percentage"""
        ax.scatter(results_df['exploitation_percentage'], results_df['num_items'], 
                  alpha=0.6, color='green')
        ax.set_title('Number of Items vs Exploitation', fontweight='bold')
        ax.set_xlabel('Exploitation Percentage')
        ax.set_ylabel('Number of Items')
        
        # Add trend line
        z = np.polyfit(results_df['exploitation_percentage'], results_df['num_items'], 1)
        p = np.poly1d(z)
        ax.plot(results_df['exploitation_percentage'], p(results_df['exploitation_percentage']), 
               "r--", alpha=0.8)
    
    def _plot_correlation_matrix(self, ax, results_df: pd.DataFrame):
        """Plot correlation matrix of key metrics"""
        # Select numeric columns for correlation
        numeric_cols = ['exploitation_percentage', 'exploration_percentage', 
                       'num_switches', 'novelty_score', 'clustering_coefficient', 
                       'num_clusters', 'num_items', 'word_coverage']
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in numeric_cols if col in results_df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = results_df[available_cols].corr()
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(available_cols)))
            ax.set_yticks(range(len(available_cols)))
            ax.set_xticklabels(available_cols, rotation=45, ha='right')
            ax.set_yticklabels(available_cols)
            ax.set_title('Correlation Matrix', fontweight='bold')
            
            # Add correlation values
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for correlation matrix', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Correlation Matrix', fontweight='bold')
    
    def plot_meg_correlations(self, results_df: pd.DataFrame, save: bool = True) -> None:
        """Plot correlations with MEG/LC data"""
        # Check if MEG data is available
        meg_cols = [col for col in results_df.columns if 'norm_' in col or 'alpha_' in col]
        
        if not meg_cols:
            logger.warning("No MEG/LC data found for correlation plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MEG/LC Data Correlations', fontsize=16, fontweight='bold')
        
        # Plot correlations with key metrics
        metrics = ['exploitation_percentage', 'avg_similarity', 'novelty_ratio', 'num_phases']
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                ax = axes[i//2, i%2]
                self._plot_meg_correlation(ax, results_df, metric, meg_cols[0] if meg_cols else None)
        
        plt.tight_layout()
        
        if save and self.config.save_plots:
            filename = f"meg_correlations.{self.config.plot_format}"
            plt.savefig(self.output_dir / filename, format=self.config.plot_format, 
                       bbox_inches='tight', dpi=300)
            logger.info(f"Saved MEG correlations plot: {filename}")
        
        plt.show()
    
    def _plot_meg_correlation(self, ax, results_df: pd.DataFrame, metric: str, meg_col: str):
        """Plot correlation between a metric and MEG data"""
        if meg_col and metric in results_df.columns and meg_col in results_df.columns:
            # Remove rows with missing data
            valid_data = results_df[[metric, meg_col]].dropna()
            
            if len(valid_data) > 1:
                ax.scatter(valid_data[metric], valid_data[meg_col], alpha=0.6)
                ax.set_title(f'{metric} vs {meg_col}', fontweight='bold')
                ax.set_xlabel(metric)
                ax.set_ylabel(meg_col)
                
                # Add trend line
                z = np.polyfit(valid_data[metric], valid_data[meg_col], 1)
                p = np.poly1d(z)
                ax.plot(valid_data[metric], p(valid_data[metric]), "r--", alpha=0.8)
                
                # Add correlation coefficient
                corr = valid_data[metric].corr(valid_data[meg_col])
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric} vs {meg_col}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric} vs MEG Data', fontweight='bold')
