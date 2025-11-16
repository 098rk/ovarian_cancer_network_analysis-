# Ovarian Cancer Network Analysis Pipeline

A comprehensive computational framework for identifying key regulatory nodes in biological signaling networks through integrated multi-omics data analysis and network modeling. This reproducible pipeline implements the methodology developed in the PhD thesis *"Development of Methods for Identifying Key Variables in Complex Mathematical Models of Biological Systems."*

**Reproducibility Note:**  
Once the filtered network is generated, reproducing the complete set of analyses — including network construction, filtering, topological analysis, Boolean simulations, and random walk experiments — typically requires **approximately 6 to 8 hours** on a standard desktop computer.

## Research Hypotheses

- **H1:** Integrated computational pipeline combining Boolean modeling, PageRank, and random walk algorithms consistently identifies key regulatory nodes (NF-κB, p53, ATM) with >80% convergence across methods  
- **H2:** Multi-omics data integration uncovers regulatory elements (IKKα, Wip1) with high centrality scores representing novel therapeutic targets  
- **H3:** Methodology demonstrates ≥85% robustness and generalizability across Cell Cycle and MAPK signaling pathways

## Quick Start


### Installation

```bash
# Clone repository
git clone https://github.com/098rk/ovarian_cancer_network_analysis-.git
cd ovarian_cancer_network_analysis-

# Create and activate conda environment
conda env create -f environment.yml
conda activate ovarian-cancer-networks
Database Setup:
# Configure MySQL connection parameters in database_setup.py
# Run database initialization
python src/network_analysis/database_setup.py
Pipeline Execution:
# Run complete end-to-end analysis
python src/network_analysis/run_pipeline.py

# Execute specific analytical modules
python src/network_analysis/boolean_networks.py        # Boolean modeling
python src/network_analysis/random_walk.py             # Random walk analysis  
python src/network_analysis/centrality_measures.py     # PageRank and centrality
python src/network_analysis/MAPK.py                    # MAPK pathway validation


Computational Pipeline
Phase 1: Data Extraction & Database Design

Automated Components:

    API-based data extraction from Pathway Commons, AnimalTFDB, CellTalkDB, GDC

    Automated data transformation and quality control

    MySQL database population with relational schema enforcement

Scripts:

    pathway_commons.py - Pathway interaction data

    animal_tf.py - Transcription factor data

    celltalk_loader.py - Ligand-receptor interactions

    database_setup.py - MySQL database management

Phase 2: Data Filtration & Network Construction

Automated Components:

    PubMed ID and KEGG pathway-based filtering

    Directed network graph construction using NetworkX

    Mutation-based edge weight assignment from TCGA-OV data

    Integration of TF, ligand-receptor, and protein-protein interactions

Key Output: Initial network with 1,492 molecular entities and 3,527 curated interactions
Phase 3: Multi-Algorithmic Analysis

Automated Components:

    Boolean Modeling (boolean_networks.py) - State transition dynamics

    PageRank Analysis (centrality_measures.py) - Node centrality quantification

    Random Walk Simulations (random_walk.py) - Stochastic network exploration

    RCNN Implementation (model_training_and_evaluation.py) - Temporal pattern recognition

Key Output: Identification of master regulators (NF-κB, p53, ATM) and novel targets
Phase 4: Validation & Generalization

Automated Components:

    Batch processing of Cell Cycle and MAPK signaling pathways

    Comparative analysis across network configurations

    Robustness testing under node removal conditions

    Results aggregation and visualization generation

Scripts: MAPK.py, Cell Cycle analysis scripts

File Descriptions
Data Extraction

    pathway_commons.py - Extracts pathway interactions from Pathway Commons database

    animal_tf.py - Retrieves transcription factor data from AnimalTFDB 3.0

    celltalk_loader.py - Processes ligand-receptor pairs from CellTalkDB

    reproduction_pipeline.py - Complete data reproduction workflow

Database & Storage

    database_setup.py - MySQL database creation and table initialization

    data filtration process for all databases - Filtering scripts for quality control

Network Analysis

    boolean_networks.py - Boolean modeling of network dynamics

    random_walk.py - Random walk simulations and node significance

    centrality_measures.py - PageRank, degree, betweenness centrality

    model_training_and_evaluation.py - RCNN model for temporal patterns

    layer_network.py - Multi-layer network construction

Validation & Applications

    MAPK.py - MAPK signaling pathway analysis

    Cell Cycle and MAPK signaling/ - Additional network validation tests

    ovarian_cancer_diffexp.csv - Differential expression data

    TCGA.OV.sampleMap_HiSeq.gz - TCGA ovarian cancer genomic data

Orchestration

    run_pipeline.py - Master script for end-to-end execution

    setup.py - Package configuration and dependencies

Automated vs. Manual Components
Fully Automated

    Data extraction and transformation from biological databases

    Network construction and topological computation

    Multi-algorithmic execution (Boolean, PageRank, Random Walks, RCNN)

    Batch processing and results aggregation

    Database management and schema enforcement

Researcher-Guided

    Biological context specification and pathway selection

    Parameter tuning for analytical methods

    Interpretation and validation of computational findings

    Therapeutic target prioritization based on domain knowledge

Key Outputs

    Integrated Biological Networks: Multi-omics signaling networks with mutation-weighted edges

    Key Regulatory Nodes: Consistently identified master regulators (NF-κB, p53, ATM) and novel targets (IKKα, Wip1)

    Therapeutic Target Prioritization: Ranked list of potential drug targets

    Cross-Pathway Validation: Results from Cell Cycle and MAPK signaling pathway analyses

    Network Topology Metrics: Scale-free properties and connectivity patterns
Dependencies

    Python 3.8+

    MySQL Server

    NetworkX, pandas, numpy, scikit-learn

    TensorFlow/Keras (for RCNN)

    Cytoscape (for visualization)

    Additional packages listed in environment.yml

Citation

If you use this computational framework in your research, please cite:
bibtex

@phdthesis{khan2025development,
  title={Development of Methods for Identifying Key Variables in Complex Mathematical Models of Biological Systems},
  author={Khan, Ruby},
  year={2025},
  school={Silesian University of Technology}
}

Contact

Ruby Khan - ruby.khan@polsl.pl

Project Link: https://github.com/098rk/ovarian_cancer_network_analysis-

This repository contains the complete computational pipeline for the PhD thesis "Development of Methods for Identifying Key Variables in Complex Mathematical Models of Biological Systems" submitted to Silesian University of Technology, Poland.



# Install Python dependencies
pip install -r requirements.txt
