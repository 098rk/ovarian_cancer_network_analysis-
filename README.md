# Network_Construction_network_analysis-
FULL-ANALYSIS
thesis_repository/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   ├── pathway_commons.py
│   │   ├── celltalk_db.py
│   │   ├── animal_tf.py
│   │   └── tcga_ov.py
│   ├── network_analysis/
│   │   ├── __init__.py
│   │   ├── boolean_networks.py
│   │   ├── pagerank_analysis.py
│   │   ├── random_walk.py
│   │   └── centrality_measures.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql
│   │   └── database_manager.py
│   └── visualization/
│       ├── __init__.py
│       └── network_plots.py
├── config/
│   └── settings.py
├── tests/
│   ├── __init__.py
│   ├── test_data_acquisition.py
│   └── test_network_analysis.py
├── scripts/
│   ├── run_pipeline.py
│   └── reproduce_analysis.py
└── data/
    ├── raw/
    ├── processed/
    └── results/
