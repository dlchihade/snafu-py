#!/usr/bin/env python3
"""
Main script for semantic fluency analysis
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import SemanticFluencyPipeline
from src.config import AnalysisConfig

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main analysis function"""
    print("🚀 Semantic Fluency Analysis Pipeline")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        logger.info("Initializing analysis pipeline...")
        pipeline = SemanticFluencyPipeline('config/config.yaml')
        
        # Run analysis
        logger.info("Starting analysis...")
        results_df = pipeline.run_analysis()
        
        print(f"\n✅ Analysis complete!")
        print(f"   Processed {len(results_df)} participants")
        print(f"   Results shape: {results_df.shape}")
        
        # Display summary statistics
        if pipeline.summary_stats:
            print(f"\n📊 Summary Statistics:")
            for key, value in pipeline.summary_stats.items():
                print(f"   {key}: {value:.3f}")
        
        # Generate visualizations
        print(f"\n📈 Generating visualizations...")
        pipeline.generate_visualizations(save_plots=True)
        
        # Save results
        print(f"\n💾 Saving results...")
        results_path = pipeline.save_results()
        print(f"   Results saved to: {results_path}")
        
        # Generate report
        print(f"\n📋 Generating report...")
        report_path = pipeline.generate_report()
        print(f"   Report saved to: {report_path}")
        
        # Export for statistical analysis
        print(f"\n📤 Exporting for statistical analysis...")
        export_path = pipeline.export_for_statistical_analysis()
        print(f"   Export saved to: {export_path}")
        
        # Perform statistical tests
        print(f"\n🔬 Performing statistical tests...")
        tests = pipeline.get_statistical_tests()
        print(f"   Performed {len(tests)} statistical tests")
        
        # Display key correlations
        print(f"\n📊 Key Correlations:")
        for test_name, test_result in tests.items():
            if 'correlation' in test_name and test_result.get('significant', False):
                print(f"   {test_name}: r = {test_result['correlation']:.3f} (p < 0.05)")
        
        print(f"\n🎉 Analysis pipeline completed successfully!")
        print(f"   Check the 'output' directory for all results and visualizations")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        raise

def run_quick_test():
    """Run a quick test with a subset of data"""
    print("🧪 Running Quick Test")
    print("=" * 30)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        pipeline = SemanticFluencyPipeline('config/config.yaml')
        
        # Load data
        pipeline.analyzer.load_data('data/fluency_data.csv', 'data/meg_data.csv')
        
        # Test with first 3 participants
        test_participants = pipeline.analyzer.data['ID'].unique()[:3]
        print(f"Testing with participants: {test_participants}")
        
        results = []
        for participant_id in test_participants:
            participant_data = pipeline.analyzer.data[pipeline.analyzer.data['ID'] == participant_id]
            result = pipeline.analyzer.analyze_participant(participant_data)
            results.append(result)
            print(f"   {participant_id}: {result['num_items']} items, "
                  f"{result['exploitation_percentage']:.1f}% exploitation")
        
        print(f"\n✅ Quick test completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        print(f"\n❌ Quick test failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic Fluency Analysis Pipeline')
    parser.add_argument('--test', action='store_true', 
                       help='Run a quick test with subset of data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.test:
        run_quick_test()
    else:
        main()
