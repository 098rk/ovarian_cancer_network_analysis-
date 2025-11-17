Quick Reproduction (4-8 hours on standard PC)

For quick reproduction of the results, follow these steps:
# 1. Upload repository using command:
git clone https://github.com/098rk/ovarian_cancer_network_analysis-.git
cd ovarian_cancer_network_analysis-

# 2. Run the command to setup SQL:
python src/network_analysis/database_setup.py

# 3. Import the filtered database which includes curated data from the thesis
# (This is automatically handled by the setup script)

# 4. Go to the network analysis directory:
cd src/network_analysis

# 5. Run the reproduction pipeline:
python reproduction_pipeline.py

# 6. The results are stored in src/network_analysis/results/ directory

Full Implementation

For the full pipeline run which may take up to 24 hours on standard PC and requires manual adjustments or filtration follow these steps:
bash

# 1. Upload repository and setup environment:
git clone https://github.com/098rk/ovarian_cancer_network_analysis-.git
cd ovarian_cancer_network_analysis-
conda env create -f environment.yml
conda activate ovarian-cancer-networks

# 2. Manual configuration required:
# - Adjust the database connection parameters in database_setup.py
# - Filter the data removing low-quality samples in data filtration process
# - Set pathway-specific thresholds in configuration files

# 3. Run complete analysis:
python src/network_analysis/run_pipeline.py

# OR run individual analyses:
python src/network_analysis/boolean_networks.py        # Boolean modeling
python src/network_analysis/random_walk.py             # Random walk analysis
python src/network_analysis/centrality_measures.py     # PageRank and centrality
python src/network_analysis/MAPK.py                    # MAPK pathway analysis
python src/network_analysis/model_training_and_evaluation.py  # RCNN analysis

Detailed Description of the Repository
Repository Structure
text

ovarian_cancer_network_analysis/
├── src/network_analysis/
│   ├── 📊 Data Files
│   │   ├── AnimalTFData.csv              # Transcription factor database
│   │   ├── CellTalk.csv                  # Cell-cell communication data
│   │   ├── ClinicalData.csv              # Patient clinical information
│   │   ├── ovarian_cancer_diffexp.csv    # Differential expression data
│   │   ├── TCGA.OV.sampleMap_HiSeq.gz    # TCGA ovarian cancer genomic data
│   │   └── meta_*.csv files              # TCGA metadata (clinical, mutations, expression)
│   │
│   ├── 🔬 Data Extraction & Network Construction
│   │   ├── pathway_commons.py            # Extract pathway interactions
│   │   ├── animal_tf.py                  # Get transcription factor data
│   │   ├── celltalk_loader.py            # Process ligand-receptor pairs
│   │   ├── data filtration process/      # Quality control and filtering
│   │   └── database_setup.py             # MySQL database management
│   │
│   ├── 🧮 Multi-Algorithmic Analysis
│   │   ├── boolean_networks.py           # Boolean modeling of network dynamics
│   │   ├── random_walk.py                # Random walk simulations
│   │   ├── centrality_measures.py        # PageRank and centrality analysis
│   │   ├── model_training_and_evaluation.py  # RCNN for temporal patterns
│   │   ├── layer_network.py              # Multi-layer network construction
│   │   └── RCNN_MPAK_CELL_CYCLE/         # RCNN implementations
│   │
│   ├── 🧪 Pathway-Specific Validation
│   │   ├── MAPK.py                       # MAPK signaling pathway analysis
│   │   ├── Cell Cycle and MAPK signaling/ # Network validation tests
│   │   └── ov_py/                        # Ovarian cancer specific scripts
│   │
│   ├── 📈 Results Directory
│   │   ├── python_analysis_results.md    # p53 and NF-κB pathway results
│   │   ├── mapk_cell_cycle_results.md    # MAPK and Cell Cycle analysis
│   │   └── results/                      # Generated analysis results
│   │       ├── Algorithm Performance Results Computational Efficiency
│   │       ├── Betweenness Centrality Results
│   │       ├── Boolean Network Simulation Results
│   │       ├── Comprehensive Network Metrics
│   │       ├── Correlation Analysis
│   │       ├── Degree Centrality Rankings
│   │       ├── Key Biological Insights from Python Analysis Network Architecture Findings
│   │       ├── Limitations and Technical Notes Analysis Constraints
│   │       ├── Model Validation Results
│   │       ├── NF-κB Pathway Node Connectivity
│   │       ├── Network Topology Statistics
│   │       ├── Ovarian Cancer Relevance
│   │       ├── Ovarian Cancer Specific Results Therapeutic Target Prioritization
│   │       ├── PageRank Algorithm Results
│   │       ├── Pathway Cross-Talk Analysis
│   │       ├── Pathway-Specific Analysis Results p53 Pathway Node Connectivity
│   │       ├── Perturbation Analysis Results
│   │       ├── Random Walk Simulation Results Signal Propagation Analysis
│   │       ├── Signal Flow Efficiency
│   │       ├── Therapeutic Implications
│   │       └── Therapeutic Target Prioritization
│   │
│   └── ⚙️  Orchestration
│       ├── run_pipeline.py               # Master script for end-to-end execution
│       ├── reproduction_pipeline.py      # Quick reproduction pipeline
│       ├── setup.py                      # Package configuration
│       └── Detailed SQL DDL Script for Database Implementation
│
└── 📚 Documentation
    ├── environment.yml                   # Conda environment specification
    ├── Citation                         # Thesis citation
    └── Repository Structure             # Detailed file structure

File Descriptions
Data Extraction & Network Construction

    pathway_commons.py - Extracts biological pathway interactions from Pathway Commons database via API

    animal_tf.py - Retrieves comprehensive transcription factor data from AnimalTFDB 3.0

    celltalk_loader.py - Processes ligand-receptor pairs from CellTalkDB for cell communication analysis

    database_setup.py - MySQL database initialization with relational schema enforcement

    data filtration process - Quality control scripts for filtering low-quality samples

Multi-Algorithmic Analysis

    boolean_networks.py - Boolean modeling for dynamic state analysis of signaling networks

    random_walk.py - Random walk simulations for stochastic network exploration

    centrality_measures.py - Computes PageRank, degree, and betweenness centrality metrics

    model_training_and_evaluation.py - RCNN implementation for temporal pattern recognition

    layer_network.py - Multi-layer network construction and analysis

Pathway-Specific Validation

    MAPK.py - Comprehensive MAPK signaling pathway analysis and cross-talk validation

    Cell Cycle and MAPK signaling - Cell cycle network construction and comparative analysis

    ov_py - Ovarian cancer-specific network analysis focusing on TCGA-OV data

Orchestration

    run_pipeline.py - Master script coordinating data extraction, network construction, and multi-algorithmic analysis

    reproduction_pipeline.py - Optimized pipeline for quick reproduction of core results

Data Files Description
Core Biological Data

    AnimalTFData.csv - Comprehensive transcription factor database from AnimalTFDB

    CellTalk.csv - Cell-cell communication interactions and pathways from CellTalkDB

    ClinicalData.csv - Patient clinical information and treatment outcomes from TCGA

    ovarian_cancer_diffexp.csv - Differential expression analysis results for ovarian cancer

TCGA Meta Data

    meta_clinical_patient.csv - Patient clinical metadata including staging and survival

    meta_mutations.csv - Somatic mutation data for ovarian cancer samples

    meta_mrna_seq_*.csv - mRNA sequencing data in multiple formats (FPKM, TPM, read counts)

    meta_study.csv - Study metadata and sample descriptions

Analysis Pipeline
Phase 1: Data Extraction

    Automated API calls to Pathway Commons, AnimalTFDB, CellTalkDB

    Data transformation and quality control

    MySQL database population with structured schema

Phase 2: Network Construction

    PubMed ID and KEGG pathway-based filtering

    Directed network graph construction using NetworkX

    Mutation-based edge weight assignment from TCGA-OV data

    Integration of TF, ligand-receptor, and protein-protein interactions

Phase 3: Multi-Algorithmic Analysis

    Boolean modeling for dynamic state analysis

    PageRank and centrality measures for topological importance

    Random walk simulations for stochastic network exploration

    RCNN for temporal pattern recognition

Phase 4: Validation & Generalization

    Batch processing across Cell Cycle and MAPK signaling pathways

    Comparative analysis and robustness testing

    Results aggregation and therapeutic target prioritization

Key Outputs
Network Analysis Results

    Integrated multi-omics networks with mutation-weighted edges

    Key regulatory nodes: NF-κB, p53, ATM as master regulators

    Novel therapeutic targets: IKKα, Wip1 identified through network analysis

    Cross-pathway validation results from Cell Cycle and MAPK analyses

Computational Findings

    p53-NF-κB Network: 43 nodes, scale-free topology confirmed

    Cell Cycle Network: 79 nodes, hierarchical organization

    MAPK Network: 71 nodes, distributed signaling architecture

    Network robustness quantified through perturbation analysis

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

Dependencies
bash

# Core requirements
Python 3.8+
MySQL Server
NetworkX, pandas, numpy, scikit-learn
TensorFlow/Keras (for RCNN)
Matplotlib for visualizations

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

Ruby Khan - ruby.khan@polsl.pl\rubykhanutk@gmail.com

Project Link: https://github.com/098rk/ovarian_cancer_network_analysis-

This repository contains the complete computational pipeline for the PhD thesis "Development of Methods for Identifying Key Variables in Complex Mathematical Models of Biological Systems" submitted to Silesian University of Technology, Poland.
