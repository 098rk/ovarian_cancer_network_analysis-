#!/usr/bin/env python3
"""
Main pipeline script for ovarian cancer network analysis.
Reproducible implementation of the computational pipeline.

Master script for the complete ovarian cancer network analysis framework.
Coordinates data acquisition, network construction, analysis, and visualization.
"""

import sys
import os
import json
import yaml
import argparse
from pathlib import Path
import logging
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    # Import all required modules with error handling
    from src.data_acquisition import (
        PathwayCommonsLoader,
        CellTalkLoader,
        AnimalTFLoader,
        TCGALoader
    )
    from src.network_analysis import (
        BooleanNetworkAnalyzer,
        PageRankAnalyzer,
        RandomWalkAnalyzer,
        RCNNAnalyzer,
        MultiLayerNetworkAnalyzer
    )
    from src.data_processing import DataFilter
    from src.database import DatabaseManager
    from src.visualization import NetworkVisualizer
    from src.validation import HypothesisValidator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and the src directory structure is correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OvarianCancerPipeline:
    """
    Integrated computational pipeline for ovarian cancer network analysis.

    This is the master orchestrator that coordinates all phases of the analysis:
    1. Data acquisition from multiple sources
    2. Data filtration and preprocessing
    3. Network construction and analysis
    4. Hypothesis validation
    5. Results visualization and reporting
    """

    def __init__(self, config_path=None, output_dir=None):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path (str): Path to configuration file
            output_dir (str): Directory for output files
        """
        self.config_path = config_path or "config/settings.yaml"
        self.config = self.load_config()

        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.get('output_dir', 'results'))

        # Create output directories
        self.setup_output_directories()

        # Initialize components
        self.db_manager = None
        if self.config['database'].get('enabled', True):
            self.db_manager = DatabaseManager(self.config['database'])

        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")
        logger.info(f"Configuration loaded from: {self.config_path}")

    def load_config(self):
        """Load configuration settings from file."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            return self.get_default_config()

        try:
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix == '.py':
                # For Python config files
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                config = {k: v for k, v in vars(config_module).items()
                          if not k.startswith('_') and not callable(v)}
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            logger.info(f"Configuration loaded successfully from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Falling back to default configuration")
            return self.get_default_config()

    def get_default_config(self):
        """Return default configuration."""
        return {
            'database': {
                'host': 'localhost',
                'user': 'root',
                'password': 'root1',
                'database': 'genomic_data_ovarian_cancer',
                'port': 3306,
                'enabled': True
            },
            'data_sources': {
                'pathway_commons': {
                    'url': 'http://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.All.BINARY_SIF.gz',
                    'cache_dir': 'data/external/pathway_commons',
                    'enabled': True
                },
                'celltalk': {
                    'url': 'http://tcm.zju.edu.cn/celltalkdb/download/Human_lr_pair.txt.gz',
                    'cache_dir': 'data/external/celltalk',
                    'enabled': True
                },
                'animal_tf': {
                    'url': 'http://bioinfo.life.hust.edu.cn/AnimalTFDB4.0/download/Homo_sapiens_TF',
                    'cache_dir': 'data/external/animal_tf',
                    'enabled': True
                },
                'tcga_ov': {
                    'output_dir': 'data/external/tcga_ov',
                    'max_files_per_category': 20,
                    'enabled': True
                }
            },
            'analysis': {
                'boolean_network': {'enabled': True, 'threshold': 0.5},
                'pagerank': {'enabled': True, 'damping_factor': 0.85},
                'random_walk': {'enabled': True, 'restart_prob': 0.15},
                'rcnn': {'enabled': True, 'epochs': 50},
                'multilayer': {'enabled': True}
            },
            'output_dir': 'results',
            'logging': {
                'level': 'INFO',
                'file': 'pipeline_execution.log'
            }
        }

    def setup_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_dir,
            self.output_dir / 'networks',
            self.output_dir / 'rankings',
            self.output_dir / 'visualizations',
            self.output_dir / 'logs',
            self.output_dir / 'summaries',
            self.output_dir / 'intermediate'
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def run_data_acquisition(self):
        """Execute data acquisition from all sources."""
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA ACQUISITION")
        logger.info("=" * 60)

        data_dict = {}

        try:
            # Pathway Commons data
            if self.config['data_sources']['pathway_commons']['enabled']:
                logger.info("Downloading Pathway Commons data...")
                pc_config = self.config['data_sources']['pathway_commons']
                pc_loader = PathwayCommonsLoader(
                    url=pc_config['url'],
                    cache_dir=pc_config['cache_dir']
                )
                pc_data = pc_loader.download_and_process()
                data_dict['pathway_commons'] = pc_data
                logger.info(f"Pathway Commons data: {len(pc_data)} interactions")

            # CellTalk data
            if self.config['data_sources']['celltalk']['enabled']:
                logger.info("Downloading CellTalk data...")
                ct_config = self.config['data_sources']['celltalk']
                ct_loader = CellTalkLoader(
                    url=ct_config['url'],
                    cache_dir=ct_config['cache_dir']
                )
                ct_data = ct_loader.download_and_process()
                data_dict['celltalk'] = ct_data
                logger.info(f"CellTalk data: {len(ct_data)} ligand-receptor pairs")

            # AnimalTF data
            if self.config['data_sources']['animal_tf']['enabled']:
                logger.info("Downloading AnimalTF data...")
                atf_config = self.config['data_sources']['animal_tf']
                atf_loader = AnimalTFLoader(
                    url=atf_config['url'],
                    cache_dir=atf_config['cache_dir']
                )
                atf_data = atf_loader.download_and_process()
                data_dict['animal_tf'] = atf_data
                logger.info(f"AnimalTF data: {len(atf_data)} transcription factors")

            # TCGA-OV data
            if self.config['data_sources']['tcga_ov']['enabled']:
                logger.info("Downloading TCGA-OV data...")
                tcga_config = self.config['data_sources']['tcga_ov']
                tcga_loader = TCGALoader(output_dir=tcga_config['output_dir'])
                tcga_data = tcga_loader.download_with_chunking(
                    chunk_size=tcga_config.get('chunk_size', 1000),
                    max_files=tcga_config.get('max_files_per_category', 20)
                )
                data_dict['tcga_ov'] = tcga_data
                logger.info(f"TCGA-OV data acquired: {len(tcga_data)} samples")

            logger.info("Data acquisition phase completed successfully")
            return data_dict

        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            raise

    def run_data_filtration(self, data_dict):
        """Apply data filtration and preprocessing."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DATA FILTRATION")
        logger.info("=" * 60)

        try:
            data_filter = DataFilter()
            filtered_data = {}

            for data_type, data in data_dict.items():
                logger.info(f"Filtering {data_type} data...")
                filtered = data_filter.filter_data(data, data_type)
                filtered_data[data_type] = filtered
                logger.info(f"  Before: {len(data)}, After: {len(filtered)}")

            # Save filtered data
            filtered_path = self.output_dir / 'intermediate' / 'filtered_data.pkl'
            import pickle
            with open(filtered_path, 'wb') as f:
                pickle.dump(filtered_data, f)
            logger.info(f"Filtered data saved to: {filtered_path}")

            return filtered_data

        except Exception as e:
            logger.error(f"Data filtration failed: {e}")
            raise

    def run_network_analysis(self, filtered_data):
        """Construct and analyze biological networks."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: NETWORK ANALYSIS")
        logger.info("=" * 60)

        results = {}

        try:
            # Boolean network analysis
            if self.config['analysis']['boolean_network']['enabled']:
                logger.info("Running Boolean network analysis...")
                boolean_analyzer = BooleanNetworkAnalyzer()
                boolean_results = boolean_analyzer.analyze_pathways(filtered_data)
                results['boolean_networks'] = boolean_results
                logger.info(f"Boolean analysis completed: {len(boolean_results.get('key_nodes', []))} key nodes")

            # PageRank analysis
            if self.config['analysis']['pagerank']['enabled']:
                logger.info("Running PageRank analysis...")
                pagerank_analyzer = PageRankAnalyzer(
                    damping_factor=self.config['analysis']['pagerank']['damping_factor']
                )
                pagerank_results = pagerank_analyzer.calculate_centrality(filtered_data)
                results['pagerank'] = pagerank_results
                logger.info(f"PageRank analysis completed: {len(pagerank_results.get('top_nodes', []))} top nodes")

            # Random walk analysis
            if self.config['analysis']['random_walk']['enabled']:
                logger.info("Running Random Walk analysis...")
                rw_analyzer = RandomWalkAnalyzer(
                    restart_prob=self.config['analysis']['random_walk']['restart_prob']
                )
                rw_results = rw_analyzer.perform_random_walks(filtered_data)
                results['random_walk'] = rw_results
                logger.info(
                    f"Random Walk analysis completed: {len(rw_results.get('significant_nodes', []))} significant nodes")

            # RCNN analysis
            if self.config['analysis']['rcnn']['enabled']:
                logger.info("Running RCNN analysis...")
                rcnn_analyzer = RCNNAnalyzer(
                    epochs=self.config['analysis']['rcnn']['epochs']
                )
                rcnn_results = rcnn_analyzer.train_and_evaluate(filtered_data)
                results['rcnn'] = rcnn_results
                logger.info(f"RCNN analysis completed: Accuracy = {rcnn_results.get('accuracy', 0):.3f}")

            # Multi-layer network analysis
            if self.config['analysis']['multilayer']['enabled']:
                logger.info("Running Multi-layer network analysis...")
                ml_analyzer = MultiLayerNetworkAnalyzer()
                ml_results = ml_analyzer.analyze_multilayer_network(filtered_data)
                results['multilayer'] = ml_results
                logger.info(
                    f"Multi-layer analysis completed: {len(ml_results.get('inter_layer_edges', []))} inter-layer edges")

            # Calculate convergence rate
            convergence_rate = self.calculate_convergence_rate(results)
            results['convergence_rate'] = convergence_rate
            logger.info(f"Algorithm convergence rate: {convergence_rate:.2%}")

            logger.info("Network analysis phase completed successfully")
            return results

        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            raise

    def calculate_convergence_rate(self, results):
        """
        Calculate convergence rate across methods (>80% as per H1).

        Args:
            results: Dictionary containing results from all analysis methods

        Returns:
            float: Convergence rate between 0 and 1
        """
        all_nodes = []

        # Collect key nodes from each method
        if 'boolean_networks' in results:
            all_nodes.append(set(results['boolean_networks'].get('key_nodes', [])))

        if 'pagerank' in results:
            all_nodes.append(set(results['pagerank'].get('top_nodes', [])))

        if 'random_walk' in results:
            all_nodes.append(set(results['random_walk'].get('significant_nodes', [])))

        if 'rcnn' in results:
            all_nodes.append(set(results['rcnn'].get('important_features', [])))

        if 'multilayer' in results:
            all_nodes.append(set(results['multilayer'].get('key_nodes', [])))

        if not all_nodes:
            return 0.0

        # Calculate intersection of all sets
        common_nodes = set.intersection(*all_nodes)

        # Calculate union of all sets
        total_unique = set.union(*all_nodes)

        if len(total_unique) == 0:
            return 0.0

        convergence_rate = len(common_nodes) / len(total_unique)
        return convergence_rate

    def run_validation(self, results):
        """Validate research hypotheses."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: HYPOTHESIS VALIDATION")
        logger.info("=" * 60)

        try:
            validator = HypothesisValidator()
            validation_results = validator.validate_all_hypotheses(results)

            # H1 Validation
            convergence_rate = results.get('convergence_rate', 0)
            if convergence_rate > 0.8:
                logger.info("✓ H1 SUPPORTED: Convergence rate >80% achieved")
                logger.info(f"  Actual convergence rate: {convergence_rate:.2%}")
            else:
                logger.warning("✗ H1 NOT SUPPORTED: Convergence rate <80%")
                logger.warning(f"  Actual convergence rate: {convergence_rate:.2%}")

            # Additional validation results
            if validation_results:
                for hypothesis, result in validation_results.items():
                    status = "SUPPORTED" if result['supported'] else "NOT SUPPORTED"
                    logger.info(f"  {hypothesis}: {status}")
                    if 'p_value' in result:
                        logger.info(f"    p-value: {result['p_value']:.4f}")

            logger.info("Hypothesis validation completed")
            return validation_results

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    def generate_outputs(self, results, validation_results):
        """Generate all output files and visualizations."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: OUTPUT GENERATION")
        logger.info("=" * 60)

        try:
            visualizer = NetworkVisualizer(output_dir=self.output_dir / 'visualizations')

            # Generate static visualizations
            logger.info("Generating static visualizations...")
            static_files = visualizer.create_static_plots(results)
            for file_path in static_files:
                logger.info(f"  Created: {file_path}")

            # Generate interactive visualizations
            logger.info("Generating interactive visualizations...")
            interactive_files = visualizer.create_interactive_plots(results)
            for file_path in interactive_files:
                logger.info(f"  Created: {file_path}")

            # Save results to files
            self.save_results(results, validation_results)

            # Generate comprehensive report
            self.generate_report(results, validation_results)

            logger.info("Output generation completed successfully")

        except Exception as e:
            logger.error(f"Output generation failed: {e}")
            raise

    def save_results(self, results, validation_results):
        """Save analysis results to files."""
        import pickle
        import json

        # Save raw results
        results_path = self.output_dir / 'results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to: {results_path}")

        # Save validation results as JSON
        validation_path = self.output_dir / 'validation_results.json'
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to: {validation_path}")

        # Save key nodes
        key_nodes = self.extract_key_nodes(results)
        nodes_path = self.output_dir / 'rankings' / 'key_nodes.csv'
        key_nodes.to_csv(nodes_path, index=False)
        logger.info(f"Key nodes saved to: {nodes_path}")

    def extract_key_nodes(self, results):
        """Extract and combine key nodes from all methods."""
        import pandas as pd

        nodes_data = []

        # Extract from each method
        methods = ['boolean_networks', 'pagerank', 'random_walk', 'rcnn', 'multilayer']
        for method in methods:
            if method in results:
                nodes = results[method].get('key_nodes', []) or \
                        results[method].get('top_nodes', []) or \
                        results[method].get('significant_nodes', []) or \
                        results[method].get('important_features', [])

                for node in nodes[:50]:  # Top 50 from each method
                    nodes_data.append({
                        'node': node,
                        'method': method,
                        'rank': nodes.index(node) + 1 if node in nodes else 99
                    })

        return pd.DataFrame(nodes_data)

    def generate_report(self, results, validation_results):
        """Generate comprehensive analysis report."""
        report_path = self.output_dir / 'summaries' / 'analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OVARIAN CANCER NETWORK ANALYSIS - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pipeline Version: 1.0\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")

            # Convergence analysis
            convergence_rate = results.get('convergence_rate', 0)
            f.write("CONVERGENCE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Convergence Rate: {convergence_rate:.2%}\n")
            f.write(f"Hypothesis H1 Support: {'YES' if convergence_rate > 0.8 else 'NO'}\n\n")

            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n")

            key_nodes = self.extract_key_nodes(results)
            if not key_nodes.empty:
                top_nodes = key_nodes['node'].value_counts().head(10)
                f.write("Top 10 Most Frequently Identified Nodes:\n")
                for i, (node, count) in enumerate(top_nodes.items(), 1):
                    f.write(f"  {i:2d}. {node:20s} (identified by {count} methods)\n")
            f.write("\n")

            # Method-specific results
            f.write("METHOD-SPECIFIC RESULTS\n")
            f.write("-" * 80 + "\n")
            for method, result in results.items():
                if method != 'convergence_rate':
                    f.write(f"\n{method.upper()}:\n")
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key}: {value}\n")
                        elif isinstance(value, list) and len(value) <= 10:
                            f.write(f"  {key}: {', '.join(map(str, value[:10]))}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Analysis report generated: {report_path}")

    def run_analysis(self):
        """Execute the complete pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING OVARIAN CANCER NETWORK ANALYSIS PIPELINE")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Phase 1: Data Acquisition
            data_dict = self.run_data_acquisition()

            # Phase 2: Data Filtration
            filtered_data = self.run_data_filtration(data_dict)

            # Phase 3: Network Analysis
            results = self.run_network_analysis(filtered_data)

            # Phase 4: Validation
            validation_results = self.run_validation(results)

            # Phase 5: Output Generation
            self.generate_outputs(results, validation_results)

            elapsed_time = time.time() - start_time

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
            logger.info("=" * 80)

            return results

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"\nPipeline failed after {elapsed_time:.2f} seconds: {e}")
            logger.error("Check pipeline_execution.log for details")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ovarian Cancer Network Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ovarian_cancer_pipeline.py --config config/settings.yaml
  python ovarian_cancer_pipeline.py --output results/run_2024
  python ovarian_cancer_pipeline.py --skip-db --data-only
        """
    )

    parser.add_argument('--config', type=str, default='config/settings.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--skip-db', action='store_true',
                        help='Skip database operations')
    parser.add_argument('--data-only', action='store_true',
                        help='Run only data acquisition phase')
    parser.add_argument('--skip-download', action='store_true',
                        help='Use existing data files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        print("\n" + "=" * 80)
        print("OVARIAN CANCER NETWORK ANALYSIS PIPELINE")
        print("=" * 80)

        # Initialize pipeline
        pipeline = OvarianCancerPipeline(
            config_path=args.config,
            output_dir=args.output
        )

        # Modify config based on arguments
        if args.skip_db:
            pipeline.config['database']['enabled'] = False
            logger.info("Database operations disabled")

        if args.skip_download:
            for source in pipeline.config['data_sources'].values():
                source['enabled'] = False
            logger.info("Data download disabled - using existing files")

        # Run analysis
        results = pipeline.run_analysis()

        # Print summary
        print("\n" + "=" * 50)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 50)

        convergence_rate = results.get('convergence_rate', 0)
        print(f"\nConvergence Rate: {convergence_rate:.2%}")

        if convergence_rate > 0.8:
            print("✓ H1 SUPPORTED: Methods show strong convergence (>80%)")
        else:
            print("✗ H1 NOT SUPPORTED: Convergence rate below threshold")

        # Extract and display top nodes
        key_nodes_df = pipeline.extract_key_nodes(results)
        if not key_nodes_df.empty:
            top_nodes = key_nodes_df['node'].value_counts().head(5)
            print(f"\nTop 5 Key Regulatory Nodes:")
            for i, (node, count) in enumerate(top_nodes.items(), 1):
                print(f"  {i}. {node} (identified by {count} methods)")

        print(f"\nResults saved to: {pipeline.output_dir}")
        print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
