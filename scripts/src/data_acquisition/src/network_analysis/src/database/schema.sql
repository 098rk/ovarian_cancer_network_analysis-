-- Enhanced database schema with proper indexing and constraints
CREATE DATABASE IF NOT EXISTS genomic_data_ovarian_cancer;
USE genomic_data_ovarian_cancer;

-- Genomic Data Table
CREATE TABLE IF NOT EXISTS Genomic_Data (
    gene_id VARCHAR(20) PRIMARY KEY,
    gene_symbol VARCHAR(50) NOT NULL,
    gene_name VARCHAR(100),
    annotation TEXT,
    entrez_id VARCHAR(20),
    ensembl_id VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_gene_symbol (gene_symbol),
    INDEX idx_entrez_id (entrez_id)
) ENGINE=InnoDB;

-- Pathway Commons Data
CREATE TABLE IF NOT EXISTS PathwayCommons (
    interaction_id INT AUTO_INCREMENT PRIMARY KEY,
    participant_a VARCHAR(100) NOT NULL,
    participant_b VARCHAR(100) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    data_source VARCHAR(50),
    pubmed_id VARCHAR(20),
    pathway_names TEXT,
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (participant_a) REFERENCES Genomic_Data(gene_symbol),
    FOREIGN KEY (participant_b) REFERENCES Genomic_Data(gene_symbol),
    INDEX idx_participants (participant_a, participant_b),
    INDEX idx_interaction_type (interaction_type)
) ENGINE=InnoDB;

-- CellTalk Data
CREATE TABLE IF NOT EXISTS CellTalk (
    interaction_id INT AUTO_INCREMENT PRIMARY KEY,
    ligand_gene_symbol VARCHAR(50) NOT NULL,
    receptor_gene_symbol VARCHAR(50) NOT NULL,
    ligand_gene_id VARCHAR(20),
    receptor_gene_id VARCHAR(20),
    evidence TEXT,
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ligand_gene_symbol) REFERENCES Genomic_Data(gene_symbol),
    FOREIGN KEY (receptor_gene_symbol) REFERENCES Genomic_Data(gene_symbol),
    INDEX idx_ligand_receptor (ligand_gene_symbol, receptor_gene_symbol)
) ENGINE=InnoDB;

-- Analysis Results Table
CREATE TABLE IF NOT EXISTS AnalysisResults (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_type VARCHAR(50) NOT NULL,
    node_id VARCHAR(100) NOT NULL,
    centrality_score FLOAT NOT NULL,
    algorithm_parameters JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_node_id (node_id),
    INDEX idx_centrality_score (centrality_score)
) ENGINE=InnoDB;
