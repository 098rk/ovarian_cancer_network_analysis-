#!/usr/bin/env python3
"""
Main pipeline script for ovarian cancer network analysis.
Reproducible implementation of the computational pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_acquisition import (
    PathwayCommonsLoader,
    CellTalkLoader,
    AnimalTFLoader,
    TCGALoader
)
from src.network_analysis import (
    BooleanNetworkAnalyzer,
    PageRankAnalyzer,
    RandomWalkAnalyzer
)
from src.database import DatabaseManager
from src.visualization import NetworkVisualizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OvarianCancerPipeline:
    """
    Integrated computational pipeline for ovarian cancer network analysis.
    """
    
    def __init__(self, config_path="config/settings.py"):
        self.config = self.load_config(config_path)
        self.db_manager = DatabaseManager(self.config['database'])
        
    def load_config(self, config_path):
        """Load configuration settings with error handling"""
        try:
            # Implementation for config loading
            config = {
                'database': {
                    'host': 'localhost',
                    'user': 'root',
                    'password': 'root1',
                    'database': 'genomic_data_ovarian_cancer'
                },
                'data_sources': {
                    'pathway_commons_url': 'http://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.All.BINARY_SIF.gz',
                    'celltalk_url': 'https://example.com/CellTalkData.gz'
                }
            }
            return config
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise
    
    def run_data_acquisition(self):
        """Execute data acquisition from all sources with error handling"""
        logger.info("Starting data acquisition phase...")
        
        try:
            # Pathway Commons data
            pc_loader = PathwayCommonsLoader()
            pc_data = pc_loader.download_and_process()
            logger.info("Pathway Commons data processed successfully")
            
            # CellTalk data
            ct_loader = CellTalkLoader()
            ct_data = ct_loader.download_and_process()
            logger.info("CellTalk data processed successfully")
            
            # AnimalTF data
            atf_loader = AnimalTFLoader()
            atf_data = atf_loader.download_and_process()
            logger.info("AnimalTF data processed successfully")
            
            # TCGA-OV data with chunked processing
            tcga_loader = TCGALoader()
            tcga_data = tcga_loader.download_with_chunking(chunk_size=1000)
            logger.info("TCGA-OV data processed successfully")
            
            return {
                'pathway_commons': pc_data,
                'celltalk': ct_data,
                'animaltf': atf_data,
                'tcga_ov': tcga_data
            }
            
        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            raise
    
    def run_network_construction(self, data_dict):
        """Construct and analyze biological networks"""
        logger.info("Starting network construction phase...")
        
        try:
            # Boolean network analysis
            boolean_analyzer = BooleanNetworkAnalyzer()
            boolean_results = boolean_analyzer.analyze_pathways(data_dict)
            
            # PageRank analysis
            pagerank_analyzer = PageRankAnalyzer()
            pagerank_results = pagerank_analyzer.calculate_centrality(data_dict)
            
            # Random walk analysis
            rw_analyzer = RandomWalkAnalyzer()
            rw_results = rw_analyzer.perform_random_walks(data_dict)
            
            # Validate convergence
            convergence_rate = self.calculate_convergence_rate(
                boolean_results, pagerank_results, rw_results
            )
            logger.info(f"Algorithm convergence rate: {convergence_rate:.2%}")
            
            return {
                'boolean_networks': boolean_results,
                'pagerank': pagerank_results,
                'random_walk': rw_results,
                'convergence_rate': convergence_rate
            }
            
        except Exception as e:
            logger.error(f"Network construction failed: {e}")
            raise
    
    def calculate_convergence_rate(self, boolean_results, pagerank_results, rw_results):
        """
        Calculate convergence rate across methods (>80% as per H1)
        """
        # Implementation to compare key nodes identified by different methods
        boolean_nodes = set(boolean_results['key_nodes'])
        pagerank_nodes = set(pagerank_results['top_nodes'])
        rw_nodes = set(rw_results['significant_nodes'])
        
        # Calculate intersection
        common_nodes = boolean_nodes & pagerank_nodes & rw_nodes
        total_unique = boolean_nodes | pagerank_nodes | rw_nodes
        
        if len(total_unique) == 0:
            return 0.0
            
        convergence_rate = len(common_nodes) / len(total_unique)
        return convergence_rate
    
    def run_analysis(self):
        """Execute the complete pipeline"""
        logger.info("Starting ovarian cancer network analysis pipeline...")
        
        try:
            # Phase 1: Data Acquisition
            data_dict = self.run_data_acquisition()
            
            # Phase 2: Network Construction and Analysis
            results = self.run_network_construction(data_dict)
            
            # Phase 3: Validation
            self.validate_hypotheses(results)
            
            # Phase 4: Output Generation
            self.generate_outputs(results)
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def validate_hypotheses(self, results):
        """Validate research hypotheses explicitly"""
        # H1 Validation
        if results['convergence_rate'] > 0.8:
            logger.info("H1 SUPPORTED: Convergence rate >80% achieved")
        else:
            logger.warning(f"H1 NOT SUPPORTED: Convergence rate {results['convergence_rate']:.2%}")
        
        # H2 and H3 validation would be implemented here
        logger.info("Hypothesis validation completed")
    
    def generate_outputs(self, results):
        """Generate all output files and visualizations"""
        visualizer = NetworkVisualizer()
        
        # Generate static visualizations
        visualizer.create_static_plots(results)
        
        # Generate interactive visualizations
        visualizer.create_interactive_plots(results)
        
        # Save results to files
        self.save_results(results)
        
        logger.info("Output generation completed")

def main():
    """Main execution function"""
    try:
        pipeline = OvarianCancerPipeline()
        results = pipeline.run_analysis()
        
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Convergence Rate: {results['convergence_rate']:.2%}")
        print("Key nodes identified across methods:")
        # Print key findings
        print("="*50)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
