"""
Main pipeline for semantic fluency analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .config import AnalysisConfig
from .analyzer import SemanticFluencyAnalyzer
from .visualization import VisualizationManager

logger = logging.getLogger(__name__)

class SemanticFluencyPipeline:
    """Main pipeline for semantic fluency analysis"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = AnalysisConfig.from_yaml(config_path)
        self.analyzer = SemanticFluencyAnalyzer(self.config)
        self.viz_manager = VisualizationManager(self.config)
        self.results = None
        self.summary_stats = None
        
        logger.info("Semantic Fluency Pipeline initialized")
    
    def run_analysis(self, fluency_path: str = None, meg_path: str = None) -> pd.DataFrame:
        """Run complete analysis pipeline"""
        try:
            # Use config paths if not provided
            if fluency_path is None:
                fluency_path = self.config.data_paths['fluency_data']
            if meg_path is None:
                meg_path = self.config.data_paths['meg_data']
            
            logger.info("Starting semantic fluency analysis pipeline...")
            
            # Load data
            self.analyzer.load_data(fluency_path, meg_path)
            
            # Run analysis
            self.results = self.analyzer.analyze_all_participants()
            
            # Get summary statistics
            self.summary_stats = self.analyzer.get_summary_statistics()
            
            logger.info(f"Analysis complete. Processed {len(self.results)} participants")
            logger.info(f"Summary: {self.summary_stats}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise
    
    def generate_visualizations(self, results_df: pd.DataFrame = None, 
                              participant_ids: Optional[List[str]] = None,
                              save_plots: bool = True) -> None:
        """Generate comprehensive visualizations"""
        if results_df is None:
            if self.results is None:
                raise ValueError("No results available. Run run_analysis() first.")
            results_df = self.results
        
        logger.info("Generating visualizations...")
        
        # Generate summary statistics plots
        self.viz_manager.plot_summary_statistics(results_df, save=save_plots)
        
        # Generate MEG correlations if available
        self.viz_manager.plot_meg_correlations(results_df, save=save_plots)
        
        # Generate individual participant plots
        if participant_ids is None:
            # Plot first 3 participants as examples
            participant_ids = results_df['ID'].head(3).tolist()
        
        for participant_id in participant_ids:
            if participant_id in results_df['ID'].values:
                participant_result = results_df[results_df['ID'] == participant_id].iloc[0].to_dict()
                self.viz_manager.plot_participant_analysis(participant_result, save=save_plots)
            else:
                logger.warning(f"Participant {participant_id} not found in results")
        
        logger.info("Visualization generation complete")
    
    def save_results(self, results_df: pd.DataFrame = None, 
                    filename: str = None) -> str:
        """Save results to file"""
        if results_df is None:
            if self.results is None:
                raise ValueError("No results available. Run run_analysis() first.")
            results_df = self.results
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.csv"
        
        output_path = self.config.output_dir / filename
        
        # Save main results
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Save summary statistics
        if self.summary_stats:
            summary_path = self.config.output_dir / f"summary_stats_{timestamp}.csv"
            summary_df = pd.DataFrame([self.summary_stats])
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary statistics saved to {summary_path}")
        
        return str(output_path)
    
    def generate_report(self, results_df: pd.DataFrame = None) -> str:
        """Generate a comprehensive analysis report"""
        if results_df is None:
            if self.results is None:
                raise ValueError("No results available. Run run_analysis() first.")
            results_df = self.results
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.config.output_dir / f"analysis_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("SEMANTIC FLUENCY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            if self.summary_stats:
                for key, value in self.summary_stats.items():
                    f.write(f"{key}: {value:.3f}\n")
            f.write("\n")
            
            # Participant-level statistics
            f.write("PARTICIPANT-LEVEL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total participants: {len(results_df)}\n")
            f.write(f"Participants with data: {len(results_df[results_df['num_items'] > 0])}\n")
            f.write(f"Average items per participant: {results_df['num_items'].mean():.2f}\n")
            f.write(f"Standard deviation items: {results_df['num_items'].std():.2f}\n\n")
            
            # Exploitation/Exploration analysis
            f.write("EXPLOITATION/EXPLORATION ANALYSIS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Average exploitation percentage: {results_df['exploitation_percentage'].mean():.2f}%\n")
            f.write(f"Average exploration percentage: {results_df['exploration_percentage'].mean():.2f}%\n")
            f.write(f"Average number of switches: {results_df['num_switches'].mean():.2f}\n\n")
            
            # Clustering analysis
            f.write("CLUSTERING ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average clustering coefficient: {results_df['clustering_coefficient'].mean():.3f}\n")
            f.write(f"Average number of clusters: {results_df['num_clusters'].mean():.2f}\n\n")
            
            # Novelty analysis
            f.write("NOVELTY ANALYSIS\n")
            f.write("-" * 18 + "\n")
            f.write(f"Average novelty score: {results_df['novelty_score'].mean():.3f}\n")
            f.write(f"Novelty standard deviation: {results_df['novelty_score'].std():.3f}\n\n")
            
            # Top performers
            f.write("TOP PERFORMERS\n")
            f.write("-" * 15 + "\n")
            
            # Most items
            top_items = results_df.nlargest(5, 'num_items')[['participant_id', 'num_items']]
            f.write("Most items generated:\n")
            for _, row in top_items.iterrows():
                f.write(f"  {row['participant_id']}: {row['num_items']} items\n")
            f.write("\n")
            
            # Highest exploitation
            top_exploit = results_df.nlargest(5, 'exploitation_percentage')[['participant_id', 'exploitation_percentage']]
            f.write("Highest exploitation percentage:\n")
            for _, row in top_exploit.iterrows():
                f.write(f"  {row['participant_id']}: {row['exploitation_percentage']:.1f}%\n")
            f.write("\n")
            
            # Highest novelty
            top_novelty = results_df.nlargest(5, 'novelty_ratio')[['participant_id', 'novelty_ratio']]
            f.write("Highest novelty ratio:\n")
            for _, row in top_novelty.iterrows():
                f.write(f"  {row['participant_id']}: {row['novelty_ratio']:.3f}\n")
            f.write("\n")
            
            # MEG/LC correlations if available
            meg_cols = [col for col in results_df.columns if 'norm_' in col or 'alpha_' in col]
            if meg_cols:
                f.write("MEG/LC CORRELATIONS\n")
                f.write("-" * 20 + "\n")
                for meg_col in meg_cols:
                    for metric in ['exploitation_percentage', 'avg_similarity', 'novelty_ratio']:
                        if metric in results_df.columns:
                            corr = results_df[metric].corr(results_df[meg_col])
                            f.write(f"{metric} vs {meg_col}: r = {corr:.3f}\n")
                f.write("\n")
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def get_participant_details(self, participant_id: str) -> Dict:
        """Get detailed analysis for a specific participant"""
        if self.results is None:
            raise ValueError("No results available. Run run_analysis() first.")
        
        participant_data = self.results[self.results['participant_id'] == participant_id]
        
        if participant_data.empty:
            raise ValueError(f"Participant {participant_id} not found")
        
        return participant_data.iloc[0].to_dict()
    
    def compare_participants(self, participant_ids: List[str]) -> pd.DataFrame:
        """Compare multiple participants"""
        if self.results is None:
            raise ValueError("No results available. Run run_analysis() first.")
        
        comparison_data = self.results[self.results['participant_id'].isin(participant_ids)]
        
        if comparison_data.empty:
            raise ValueError("None of the specified participants found")
        
        return comparison_data
    
    def get_statistical_tests(self, results_df: pd.DataFrame = None) -> Dict:
        """Perform statistical tests on the results"""
        if results_df is None:
            if self.results is None:
                raise ValueError("No results available. Run run_analysis() first.")
            results_df = self.results
        
        from scipy import stats
        
        tests = {}
        
        # Test for normality of key metrics
        for metric in ['exploitation_percentage', 'avg_similarity', 'novelty_ratio']:
            if metric in results_df.columns:
                data = results_df[metric].dropna()
                if len(data) > 3:
                    statistic, p_value = stats.shapiro(data)
                    tests[f'{metric}_normality'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
        
        # Correlation tests
        if 'exploitation_percentage' in results_df.columns and 'novelty_ratio' in results_df.columns:
            corr, p_value = stats.pearsonr(
                results_df['exploitation_percentage'].dropna(),
                results_df['novelty_ratio'].dropna()
            )
            tests['exploitation_novelty_correlation'] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # MEG/LC correlations
        meg_cols = [col for col in results_df.columns if 'norm_' in col or 'alpha_' in col]
        for meg_col in meg_cols:
            for metric in ['exploitation_percentage', 'avg_similarity']:
                if metric in results_df.columns:
                    valid_data = results_df[[metric, meg_col]].dropna()
                    if len(valid_data) > 3:
                        corr, p_value = stats.pearsonr(valid_data[metric], valid_data[meg_col])
                        tests[f'{metric}_{meg_col}_correlation'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        
        return tests
    
    def export_for_statistical_analysis(self, results_df: pd.DataFrame = None, 
                                      filename: str = None) -> str:
        """Export results for external statistical analysis (SPSS, R, etc.)"""
        if results_df is None:
            if self.results is None:
                raise ValueError("No results available. Run run_analysis() first.")
            results_df = self.results
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistical_export_{timestamp}.csv"
        
        output_path = self.config.output_dir / filename
        
        # Select key columns for statistical analysis
        key_columns = [
            'participant_id', 'num_items', 'avg_similarity', 'num_phases', 
            'num_switches', 'exploitation_percentage', 'exploration_percentage',
            'exploitation_time', 'exploration_time', 'novelty_ratio', 'ee_tradeoff',
            'mean_frequency', 'std_frequency', 'mean_rank', 'pct_in_top50k'
        ]
        
        # Add MEG/LC columns if available
        meg_cols = [col for col in results_df.columns if 'norm_' in col or 'alpha_' in col]
        key_columns.extend(meg_cols)
        
        # Filter to available columns
        available_columns = [col for col in key_columns if col in results_df.columns]
        
        export_df = results_df[available_columns].copy()
        export_df.to_csv(output_path, index=False)
        
        logger.info(f"Statistical export saved to {output_path}")
        return str(output_path)
