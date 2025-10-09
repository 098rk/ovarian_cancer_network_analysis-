"""
Configuration settings for the ovarian cancer network analysis pipeline.
"""

# Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root1',
    'database': 'genomic_data_ovarian_cancer',
    'port': 3306
}

# Data Source URLs
DATA_SOURCES = {
    'pathway_commons': {
        'url': 'http://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.All.BINARY_SIF.gz',
        'format': 'SIF'
    },
    'celltalk_db': {
        'url': 'https://example.com/CellTalkData.gz',  # Replace with actual URL
        'format': 'TSV'
    }
}

# Analysis Parameters
ANALYSIS_PARAMS = {
    'pagerank': {
        'alpha': 0.85,
        'max_iter': 100,
        'tol': 1e-6
    },
    'random_walk': {
        'num_walks': 5000,
        'walk_length': 15,
        'restart_prob': 0.15
    },
    'boolean_network': {
        'time_steps': 50,
        'convergence_threshold': 1e-4
    }
}

# Validation Thresholds (for hypothesis testing)
VALIDATION_THRESHOLDS = {
    'convergence_rate': 0.8,  # H1: >80% convergence
    'robustness_threshold': 0.85,  # H3: ≥85% connectivity retention
    'centrality_threshold': 0.7  # For identifying high-centrality nodes
}
