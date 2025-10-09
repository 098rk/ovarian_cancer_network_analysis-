# Ovarian Cancer Network Analysis Pipeline

A reproducible computational pipeline for identifying key regulatory nodes in ovarian cancer signaling networks using integrated multi-omics data and network centrality analyses.

## Research Hypotheses

- **H1**: Integrated pipeline identifies key nodes with >80% convergence across methods
- **H2**: Multi-omics integration reveals novel therapeutic targets  
- **H3**: Methodology demonstrates ≥85% robustness under perturbation

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/ovarian-cancer-networks.git
cd ovarian-cancer-networks

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/run_pipeline.py

# Run specific analysis
python scripts/reproduce_analysis.py --analysis pagerank
