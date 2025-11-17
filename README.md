Ovarian Cancer Network Analysis Pipeline – Overview for Reproducibility

This repository provides a fully reproducible computational pipeline for identifying key regulatory nodes in ovarian cancer signaling networks using integrated multi-omics data.

It implements the methodology developed in the PhD thesis:

“Development of Methods for Identifying Key Variables in Complex Mathematical Models of Biological Systems” (Khan, 2025).

The pipeline integrates Boolean modeling, PageRank centrality, Random Walk analysis, and RCNNs to systematically uncover master regulators and potential therapeutic targets.


Here’s a polished version of your **Quick Reproduction** section formatted correctly for a GitHub README:

---

## Quick Reproduction

To quickly reproduce the results (≈4–8 hours on a standard PC), follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/098rk/ovarian_cancer_network_analysis-.git
   cd ovarian_cancer_network_analysis-
   ```

2. **Data Extraction**
   Use the provided scripts to extract the required datasets:

   ```bash
   python src/network_analysis/pathway_commons.py
   python src/network_analysis/animal_tf.py
   python src/network_analysis/celltalk_loader.py
   python src/network_analysis/tcga.py
   ```

3. **Set up the MySQL database**

   ```bash
   python src/network_analysis/database_setup.py
   ```

   * Add the downloaded data files to the database as needed.

4. **Import prefiltered data**

   * Use the included filtering script to clean and filter the database for relevant interactions.

5. **Navigate to the analysis directory**:

   ```bash
   cd src/network_analysis
   ```

6. **Run the quick reproduction script**:

   ```bash
   python reproduction_pipeline.py
   ```

7. **Results location**

   * All outputs are stored in the `results/` directory, including:

     * Boolean network simulations
     * Centrality measures (PageRank, degree, betweenness)
     * Random Walk simulation results
     * RCNN


Overview of Pipeline

The pipeline consists of four main phases:

Data Extraction & Database Setup

Automated retrieval from Pathway Commons, AnimalTFDB, CellTalkDB, TCGA, and other repositories.

Data are cleaned, transformed, and stored in a MySQL relational database.

Automated Data Filtering & Network Construction

Filtration of irrelevant interactions and low-confidence edges using Python/Pandas scripts.

Construction of a directed, weighted network integrating:

Transcription factors

Protein-protein interactions

Ligand-receptor pairs

Mutation data

Multi-Algorithmic Analysis

Boolean modeling (boolean_networks.py) – Analyzes network dynamics.

PageRank & centrality (centrality_measures.py) – Quantifies node importance.

Random Walk analysis (random_walk.py) – Stochastic network exploration.

RCNN modeling (model_training_and_evaluation.py) – Temporal pattern recognition.

Validation & Generalization

Batch processing for Cell Cycle and MAPK pathways.

Robustness testing under node removal conditions.

Output aggregation and visualizations.


Full Pipeline Execution (End-to-End)

For a full run, which may take longer (~xx hours) and includes automated filtration and parameter adjustments:

Clone the repository (as above).

Set up the database (as above).

Adjust parameters in scripts (optional):

layer_network.py, boolean_networks.py, model_training_and_evaluation.py → update analysis parameters or thresholds.

Automated Data Filtering & Processing

Python/Pandas scripts remove irrelevant interactions and low-confidence edges from:

AnimalTFData.csv

CellTalk.csv

meta_clinical_patient.csv

meta_mrna_seq_*.csv

meta_mutations.csv

Ensures only biologically relevant nodes and interactions are retained.

Run the complete pipeline

Run the complete pipeline:
