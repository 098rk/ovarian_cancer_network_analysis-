# Computational Framework for Biological Network Analysis

## Thesis Project
**Development of Methods for Identifying Key Variables in Complex Mathematical Models of Biological Systems**

PhD Dissertation by Ruby Khan  
Department of Systems Biology and Engineering  
Silesian University of Technology

## Overview

This repository contains the complete computational framework for identifying critical regulatory nodes in complex biological networks using multi-omics data integration and advanced computational techniques.

## Repository Structure

### Core Analysis Scripts
- run_pipeline.py - Main orchestration script
- boolean_networks.py - Boolean network modeling
- random_walk.py - Stochastic network exploration
- centrality_measures.py - PageRank and network centrality
- model_training_and_evaluation.py - RCNN implementation
- layer_network.py - Multi-layer network analysis

### Data Processing
- animal_tf.py - Transcription factor data extraction
- pathway_commons.py - Pathway interactions
- celltalk_loader.py - Ligand-receptor interactions
- database_setup.py - MySQL database initialization
- data_filtration.py - Data cleaning and filtration pipeline

### Pathway-Specific Analysis
- MAPK.py - MAPK signaling pathway analysis
- Cell Cycle and MAPK signaling/ - Additional pathway analyses
- p53_network_visualization.py - p53 signaling network visualization

### Data Files
- TCGA.OV.sampleMap_HiSeq.gz - TCGA ovarian cancer genomic data
- ovarian_cancer_diffexp.csv - Differential expression results
- ov_py/ - Processed ovarian cancer datasets

### Database
- Detailed SQL DDL Script for Database Implementation/ - Complete database schema

## Quick Start

### Prerequisites
- Python 3.8+
- MySQL 8.0+
- 20GB disk space

### Installation
```bash
# Clone repository
git clone https://github.com/098rk/ovarian_cancer_network_analysis.git
cd network_analysis

# Install dependencies
pip install pandas numpy networkx matplotlib scikit-learn tensorflow mysql-connector-python

# Set up database
python database_setup.py

# Run data filtration
python data_filtration.py
