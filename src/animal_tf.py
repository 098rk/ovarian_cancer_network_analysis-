#!/usr/bin/env python3
"""
AnimalTFDB Transcription Factor Data Processor
Processes locally downloaded or sample TF data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import urllib.request
import zipfile
import tempfile
from pathlib import Path

class AnimalTFDBProcessor:
    """
    Processes AnimalTFDB data from local files or creates sample datasets
    """
    
    def __init__(self, data_dir="tf_data"):
        """
        Initialize the TF data processor
        
        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Define expected file names for AnimalTFDB 4.0
        self.expected_files = {
            'human_tf': 'Homo_sapiens_TF.txt',
            'human_tf_basic': 'Homo_sapiens_TF_basic.txt',
            'human_tf_dbd': 'Homo_sapiens_TF_DBD.txt',
            'human_cofactor': 'Homo_sapiens_cofactor.txt',
            'human_target': 'Homo_sapiens_target.txt',
            'tf_families': 'TF_family_classification.txt'
        }
        
        # Alternative data sources (more accessible)
        self.alternative_sources = {
            'jaspar_tfs': {
                'name': 'JASPAR TF Profiles',
                'url': 'https://jaspar.genereg.net/download/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.zip',
                'description': 'TF binding profiles from JASPAR database'
            },
            'uniprot_tfs': {
                'name': 'UniProt TF Data',
                'url': 'https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=(reviewed:true)%20AND%20(annotation:(type:%22transcription%22))%20AND%20(organism_id:9606)',
                'description': 'Human transcription factors from UniProt'
            }
        }

    def check_local_files(self):
        """
        Check for locally downloaded AnimalTFDB files
        
        Returns:
            Dictionary of found files and their paths
        """
        print("Checking for local AnimalTFDB files...")
        found_files = {}
        
        for file_type, filename in self.expected_files.items():
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                found_files[file_type] = file_path
                print(f"✓ Found: {filename}")
            else:
                # Also check with .txt.gz extension
                gz_path = os.path.join(self.data_dir, f"{filename}.gz")
                if os.path.exists(gz_path):
                    found_files[file_type] = gz_path
                    print(f"✓ Found: {filename}.gz")
        
        return found_files

    def process_local_file(self, filepath):
        """
        Process a local AnimalTFDB file
        
        Args:
            filepath: Path to the local file
            
        Returns:
            Processed DataFrame
        """
        print(f"Processing: {os.path.basename(filepath)}")
        
        try:
            # Handle gzipped files
            if filepath.endswith('.gz'):
                import gzip
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Try different delimiters
            lines = content.strip().split('\n')
            
            if len(lines) == 0:
                print("  File is empty")
                return None
            
            # Try to detect delimiter from first line
            first_line = lines[0]
            
            if '\t' in first_line:
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
            elif ',' in first_line:
                df = pd.read_csv(filepath, sep=',', low_memory=False)
            else:
                # Try space as delimiter
                df = pd.read_csv(filepath, sep='\\s+', low_memory=False, engine='python')
            
            print(f"  Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"  Error processing file: {e}")
            return None

    def download_from_alternative_source(self, source_name='uniprot_tfs'):
        """
        Download TF data from alternative accessible sources
        
        Args:
            source_name: Name of the alternative source
            
        Returns:
            DataFrame with downloaded data
        """
        if source_name not in self.alternative_sources:
            print(f"Source '{source_name}' not found. Available sources:")
            for key, source in self.alternative_sources.items():
                print(f"  - {key}: {source['name']}")
            return None
        
        source = self.alternative_sources[source_name]
        print(f"Downloading from {source['name']}...")
        print(f"URL: {source['url']}")
        
        try:
            # Download the data
            response = urllib.request.urlopen(source['url'])
            
            if source_name == 'jaspar_tfs':
                # JASPAR provides a ZIP file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    tmp_file.write(response.read())
                    tmp_path = tmp_file.name
                
                # Extract the ZIP
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    # Extract to data directory
                    extract_dir = os.path.join(self.data_dir, 'jaspar_data')
                    zip_ref.extractall(extract_dir)
                
                # Find extracted files
                extracted_files = list(Path(extract_dir).rglob('*.jaspar'))
                if extracted_files:
                    # Parse JASPAR format
                    data = self.parse_jaspar_file(str(extracted_files[0]))
                    os.remove(tmp_path)
                    return data
            
            elif source_name == 'uniprot_tfs':
                # UniProt provides TSV
                content = response.read().decode('utf-8')
                df = pd.read_csv(pd.io.common.StringIO(content), sep='\t')
                
                # Save locally
                output_path = os.path.join(self.data_dir, 'uniprot_tfs.csv')
                df.to_csv(output_path, index=False)
                print(f"Saved to: {output_path}")
                return df
                
        except Exception as e:
            print(f"Error downloading from {source_name}: {e}")
            return None

    def parse_jaspar_file(self, filepath):
        """
        Parse JASPAR format file
        
        Args:
            filepath: Path to JASPAR file
            
        Returns:
            DataFrame with TF information
        """
        print(f"Parsing JASPAR file: {filepath}")
        
        tfs = []
        current_tf = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # New TF entry
                    if current_tf:
                        tfs.append(current_tf)
                    
                    # Parse header
                    # Format: >MA0001.1	AG
                    parts = line[1:].split('\t')
                    if len(parts) >= 2:
                        current_tf = {
                            'JASPAR_ID': parts[0],
                            'TF_Name': parts[1],
                            'Species': 'Homo_sapiens' if len(parts) > 2 else 'Unknown'
                        }
                elif line and 'AC' in current_tf:
                    # This line contains the matrix (simplified parsing)
                    pass
        
        if current_tf:
            tfs.append(current_tf)
        
        df = pd.DataFrame(tfs)
        print(f"Extracted {len(df)} TF profiles")
        return df

    def create_comprehensive_sample_data(self):
        """
        Create comprehensive sample TF dataset for testing/development
        """
        print("\nCreating comprehensive sample TF dataset...")
        
        # Generate realistic TF data
        np.random.seed(42)
        
        # TF families and their characteristics
        tf_families_info = {
            'Zinc finger': {'size': 120, 'dbd': 'Zinc finger', 'class': 'Both'},
            'Homeobox': {'size': 90, 'dbd': 'Helix-turn-helix', 'class': 'Activator'},
            'bHLH': {'size': 85, 'dbd': 'Basic helix-loop-helix', 'class': 'Both'},
            'bZIP': {'size': 60, 'dbd': 'Basic leucine zipper', 'class': 'Activator'},
            'Nuclear receptor': {'size': 48, 'dbd': 'Zinc finger', 'class': 'Both'},
            'Forkhead': {'size': 42, 'dbd': 'Winged helix', 'class': 'Activator'},
            'ETS': {'size': 28, 'dbd': 'Winged helix-turn-helix', 'class': 'Activator'},
            'STAT': {'size': 7, 'dbd': 'SH2 domain', 'class': 'Activator'},
            'p53': {'size': 1, 'dbd': 'p53-type', 'class': 'Both'},
            'SOX': {'size': 20, 'dbd': 'HMG box', 'class': 'Activator'},
            'GATA': {'size': 6, 'dbd': 'Zinc finger', 'class': 'Activator'},
            'MADS': {'size': 5, 'dbd': 'MADS-box', 'class': 'Activator'},
            'NF-κB': {'size': 5, 'dbd': 'Rel homology domain', 'class': 'Activator'},
            'AP-2': {'size': 5, 'dbd': 'Helix-turn-helix', 'class': 'Activator'},
            'SMAD': {'size': 8, 'dbd': 'MAD homology domain', 'class': 'Both'},
            'MYB': {'size': 4, 'dbd': 'MYB domain', 'class': 'Activator'}
        }
        
        # Well-known human TFs
        well_known_tfs = [
            # Tumor suppressors and oncogenes
            ('TP53', 'p53', 7157, 'P04637', 'Tumor protein p53'),
            ('MYC', 'bHLH', 4609, 'P01106, P01108', 'MYC proto-oncogene'),
            ('RB1', 'Pocket protein', 5925, 'P06400', 'Retinoblastoma protein'),
            ('BRCA1', 'BRCT', 672, 'P38398', 'Breast cancer type 1 susceptibility protein'),
            
            # Signaling TFs
            ('STAT3', 'STAT', 6774, 'P40763', 'Signal transducer and activator of transcription 3'),
            ('NFKB1', 'NF-κB', 4790, 'P19838', 'Nuclear factor NF-kappa-B p105 subunit'),
            ('JUN', 'bZIP', 3725, 'P05412', 'Transcription factor AP-1'),
            ('FOS', 'bZIP', 2353, 'P01100', 'Proto-oncogene c-Fos'),
            
            # Development and differentiation
            ('SOX2', 'SOX', 6657, 'P48431', 'Transcription factor SOX-2'),
            ('OCT4', 'POU', 5460, 'Q01860', 'POU domain, class 5, transcription factor 1'),
            ('NANOG', 'Homeobox', 79923, 'Q9H9S0', 'Homeobox protein NANOG'),
            ('GATA4', 'GATA', 2626, 'P43694', 'Transcription factor GATA-4'),
            
            # Nuclear receptors
            ('ESR1', 'Nuclear receptor', 2099, 'P03372', 'Estrogen receptor'),
            ('AR', 'Nuclear receptor', 367, 'P10275', 'Androgen receptor'),
            ('PPARG', 'Nuclear receptor', 5468, 'P37231', 'Peroxisome proliferator-activated receptor gamma'),
            
            # Cell cycle and apoptosis
            ('E2F1', 'E2F', 1869, 'Q01094', 'Transcription factor E2F1'),
            ('p63', 'p53', 8626, 'Q9H3D4', 'Tumor protein p63'),
            ('p73', 'p53', 7161, 'O15350', 'Tumor protein p73'),
            
            # Metabolism
            ('SREBF1', 'bHLH', 6720, 'P36956', 'Sterol regulatory element-binding protein 1'),
            ('HNF4A', 'Nuclear receptor', 3172, 'P41235', 'Hepatocyte nuclear factor 4-alpha'),
            
            # Immune system
            ('IRF3', 'IRF', 3661, 'Q14653', 'Interferon regulatory factor 3'),
            ('TBX21', 'T-box', 30009, 'Q9UL17', 'T-box transcription factor TBX21'),
            
            # Stress response
            ('HSF1', 'HSF', 3297, 'Q00613', 'Heat shock factor protein 1'),
            ('ATF4', 'bZIP', 468, 'P18848', 'Cyclic AMP-dependent transcription factor ATF-4'),
            
            # Epigenetic regulators
            ('DNMT1', 'DNA methyltransferase', 1786, 'P26358', 'DNA (cytosine-5)-methyltransferase 1'),
            ('EZH2', 'SET domain', 2146, 'Q15910', 'Histone-lysine N-methyltransferase EZH2'),
            
            # Chromatin remodelers
            ('SMARCA4', 'SWI/SNF', 6597, 'P51532', 'Transcription activator BRG1'),
            ('CHD1', 'Chromodomain', 1105, 'O14646', 'Chromodomain-helicase-DNA-binding protein 1'),
            
            # RNA polymerase subunits
            ('POLR2A', 'RNA polymerase II', 5430, 'P24928', 'DNA-directed RNA polymerase II subunit RPB1'),
            
            # General transcription factors
            ('TBP', 'TATA-binding', 6908, 'P20226', 'TATA-box-binding protein'),
            ('TFIIB', 'General TF', 2950, 'Q00403', 'Transcription initiation factor IIB'),
            
            # Tissue-specific TFs
            ('MYOD1', 'bHLH', 4654, 'P15172', 'Myoblast determination protein 1'),
            ('PAX6', 'Paired box', 5080, 'P26367', 'Paired box protein Pax-6'),
            ('MITF', 'bHLH', 4286, 'O75030', 'Microphthalmia-associated transcription factor'),
            
            # Hormone receptors
            ('GR', 'Nuclear receptor', 2908, 'P04150', 'Glucocorticoid receptor'),
            ('TR', 'Nuclear receptor', 7068, 'P10827', 'Thyroid hormone receptor alpha'),
            
            # Circadian clock
            ('CLOCK', 'bHLH', 9575, 'O15516', 'Circadian locomoter output cycles protein kaput'),
            ('BMAL1', 'bHLH', 406, 'O00327', 'Aryl hydrocarbon receptor nuclear translocator-like protein 1')
        ]
        
        # Generate comprehensive dataset
        all_tfs = []
        
        # Add well-known TFs
        for symbol, family, entrez, uniprot, description in well_known_tfs:
            ensembl = f"ENSG{np.random.randint(10000000, 99999999):08d}"
            
            tf_info = {
                'Ensembl_ID': ensembl,
                'Gene_Symbol': symbol,
                'Family': family,
                'Ensembl_Protein_ID': f"ENSP{np.random.randint(10000000, 99999999):08d}",
                'Entrez_ID': entrez,
                'Uniprot_ID': uniprot,
                'TF_Classification': tf_families_info.get(family, {}).get('class', 'Activator'),
                'DNA_Binding_Domain': tf_families_info.get(family, {}).get('dbd', 'Unknown'),
                'Species': 'Homo_sapiens',
                'Chromosome': f"chr{np.random.choice(list(range(1, 23)) + ['X', 'Y'])}",
                'Start_Position': np.random.randint(1_000_000, 250_000_000),
                'End_Position': None,  # Will set below
                'Description': description,
                'Function': np.random.choice([
                    'Transcriptional regulation',
                    'Cell cycle control',
                    'Development',
                    'Differentiation',
                    'Metabolism',
                    'Immune response',
                    'Apoptosis',
                    'DNA repair',
                    'Chromatin remodeling'
                ]),
                'PubMed_References': np.random.randint(1, 500),
                'Conservation': np.random.choice(['High', 'Medium', 'Low']),
                'Protein_Length': np.random.randint(100, 2000),
                'Isoforms': np.random.randint(1, 10)
            }
            
            tf_info['End_Position'] = tf_info['Start_Position'] + np.random.randint(1000, 50000)
            all_tfs.append(tf_info)
        
        # Add additional TFs to reach ~100
        for i in range(len(well_known_tfs), 100):
            family = np.random.choice(list(tf_families_info.keys()))
            
            tf_info = {
                'Ensembl_ID': f"ENSG{np.random.randint(10000000, 99999999):08d}",
                'Gene_Symbol': f"TF{chr(65 + i % 26)}{i//26 + 1}",
                'Family': family,
                'Ensembl_Protein_ID': f"ENSP{np.random.randint(10000000, 99999999):08d}",
                'Entrez_ID': np.random.randint(1000, 99999),
                'Uniprot_ID': f"P{np.random.randint(10000, 99999):05d}",
                'TF_Classification': tf_families_info[family]['class'],
                'DNA_Binding_Domain': tf_families_info[family]['dbd'],
                'Species': 'Homo_sapiens',
                'Chromosome': f"chr{np.random.choice(list(range(1, 23)) + ['X', 'Y'])}",
                'Start_Position': np.random.randint(1_000_000, 250_000_000),
                'End_Position': None,
                'Description': f'{family} family transcription factor',
                'Function': np.random.choice([
                    'Transcriptional regulation',
                    'Cell cycle control',
                    'Development',
                    'Differentiation',
                    'Metabolism',
                    'Immune response'
                ]),
                'PubMed_References': np.random.randint(0, 100),
                'Conservation': np.random.choice(['High', 'Medium', 'Low']),
                'Protein_Length': np.random.randint(100, 2000),
                'Isoforms': np.random.randint(1, 5)
            }
            
            tf_info['End_Position'] = tf_info['Start_Position'] + np.random.randint(1000, 50000)
            all_tfs.append(tf_info)
        
        # Create DataFrame
        df = pd.DataFrame(all_tfs)
        
        # Add metadata
        df['Dataset'] = 'Human_TF_Sample'
        df['Download_Date'] = datetime.now().strftime('%Y-%m-%d')
        df['Source'] = 'Generated_Sample_Data'
        df['Version'] = '1.0'
        
        # Save the data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.data_dir, f'human_tfs_comprehensive_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        
        # Create additional datasets
        self.create_related_datasets(df)
        
        print(f"\nCreated comprehensive sample dataset with {len(df)} human TFs")
        print(f"Saved to: {output_file}")
        
        return df
    
    def create_related_datasets(self, tf_df):
        """
        Create related datasets (targets, cofactors, etc.)
        
        Args:
            tf_df: Main TF DataFrame
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. TF-Target Interactions
        print("\nCreating TF-Target interactions dataset...")
        
        targets = []
        for _, tf in tf_df.iterrows():
            n_targets = np.random.randint(5, 50)
            for _ in range(n_targets):
                target = {
                    'TF_Symbol': tf['Gene_Symbol'],
                    'TF_Ensembl': tf['Ensembl_ID'],
                    'Target_Symbol': f"TARGET{np.random.randint(1, 1000)}",
                    'Target_Ensembl': f"ENSG{np.random.randint(10000000, 99999999):08d}",
                    'Interaction_Type': np.random.choice(['Direct binding', 'Regulation', 'Co-expression']),
                    'Evidence': np.random.choice(['ChIP-seq', 'Luciferase assay', 'EMSA', 'Yeast two-hybrid']),
                    'Confidence_Score': np.round(np.random.uniform(0.5, 1.0), 3),
                    'Tissue': np.random.choice(['Liver', 'Brain', 'Heart', 'Kidney', 'Lung', 'Blood']),
                    'PubMed_ID': np.random.randint(1000000, 9999999) if np.random.random() > 0.3 else None
                }
                targets.append(target)
        
        targets_df = pd.DataFrame(targets)
        targets_file = os.path.join(self.data_dir, f'tf_targets_{timestamp}.csv')
        targets_df.to_csv(targets_file, index=False)
        print(f"  Created {len(targets_df)} TF-target interactions")
        print(f"  Saved to: {targets_file}")
        
        # 2. TF Cofactors
        print("\nCreating TF cofactors dataset...")
        
        cofactors = []
        for _, tf in tf_df.iterrows():
            n_cofactors = np.random.randint(0, 10)
            for _ in range(n_cofactors):
                cofactor = {
                    'TF_Symbol': tf['Gene_Symbol'],
                    'TF_Ensembl': tf['Ensembl_ID'],
                    'Cofactor_Symbol': f"COF{np.random.randint(1, 100)}",
                    'Cofactor_Type': np.random.choice(['Co-activator', 'Co-repressor', 'Chromatin remodeler']),
                    'Interaction_Evidence': np.random.choice(['Co-IP', 'FRET', 'Mass spectrometry']),
                    'Function': np.random.choice([
                        'Histone modification',
                        'Chromatin binding',
                        'Transcriptional activation',
                        'DNA binding assistance'
                    ])
                }
                cofactors.append(cofactor)
        
        cofactors_df = pd.DataFrame(cofactors)
        cofactors_file = os.path.join(self.data_dir, f'tf_cofactors_{timestamp}.csv')
        cofactors_df.to_csv(cofactors_file, index=False)
        print(f"  Created {len(cofactors_df)} TF-cofactor interactions")
        print(f"  Saved to: {targets_file}")
        
        # 3. TF Family Classification
        print("\nCreating TF family classification dataset...")
        
        families = tf_df['Family'].value_counts().reset_index()
        families.columns = ['Family', 'Gene_Count']
        
        # Add family descriptions
        family_descriptions = {
            'Zinc finger': 'Largest TF family, involved in diverse cellular processes',
            'Homeobox': 'Development and body plan specification',
            'bHLH': 'Cell differentiation and development',
            'bZIP': 'Dimerization via leucine zipper, stress response',
            'Nuclear receptor': 'Ligand-activated, hormone response',
            'Forkhead': 'Metabolism, immunity, lifespan regulation',
            'ETS': 'Cell proliferation, differentiation, apoptosis',
            'STAT': 'Cytokine signaling, immune response',
            'p53': 'Tumor suppression, cell cycle arrest',
            'SOX': 'Sex determination, neural development',
            'GATA': 'Hematopoiesis, heart development',
            'MADS': 'Flower development in plants, some in animals',
            'NF-κB': 'Inflammation, immunity, stress response',
            'AP-2': 'Development, carcinogenesis',
            'SMAD': 'TGF-β signaling, cell growth',
            'MYB': 'Cell cycle, differentiation, apoptosis'
        }
        
        families['Description'] = families['Family'].map(family_descriptions)
        families['DBD_Type'] = families['Family'].map({
            'Zinc finger': 'Zinc finger',
            'Homeobox': 'Helix-turn-helix',
            'bHLH': 'Basic helix-loop-helix',
            'bZIP': 'Basic leucine zipper',
            'Nuclear receptor': 'Zinc finger',
            'Forkhead': 'Winged helix',
            'ETS': 'Winged helix-turn-helix',
            'STAT': 'SH2 domain',
            'p53': 'p53-type',
            'SOX': 'HMG box',
            'GATA': 'Zinc finger',
            'MADS': 'MADS-box',
            'NF-κB': 'Rel homology domain',
            'AP-2': 'Helix-turn-helix',
            'SMAD': 'MAD homology domain',
            'MYB': 'MYB domain'
        })
        
        families['Conservation'] = np.random.choice(['High', 'Medium', 'Low'], len(families))
        
        families_file = os.path.join(self.data_dir, f'tf_families_{timestamp}.csv')
        families.to_csv(families_file, index=False)
        print(f"  Created classification for {len(families)} TF families")
        print(f"  Saved to: {families_file}")

    def run(self):
        """
        Main execution pipeline
        """
        print("\n" + "=" * 70)
        print("ANIMALTFDB DATA PROCESSOR")
        print("=" * 70)
        
        # Step 1: Check for local files
        local_files = self.check_local_files()
        
        if local_files:
            print(f"\nFound {len(local_files)} local file(s). Processing...")
            
            # Process main TF file if available
            if 'human_tf' in local_files:
                df = self.process_local_file(local_files['human_tf'])
                if df is not None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_file = os.path.join(self.data_dir, f'human_tfs_processed_{timestamp}.csv')
                    df.to_csv(output_file, index=False)
                    print(f"\nProcessed data saved to: {output_file}")
                    
                    # Show statistics
                    print(f"\nDataset Statistics:")
                    print(f"  Total TFs: {len(df)}")
                    if 'Family' in df.columns:
                        print(f"  TF Families: {df['Family'].nunique()}")
                    if 'TF_Classification' in df.columns:
                        print(f"  TF Classes: {df['TF_Classification'].nunique()}")
                    
                    return df
            
            print("\nNo main TF file found or could not process.")
        
        # Step 2: Try alternative sources
        print("\n" + "=" * 60)
        print("TRYING ALTERNATIVE DATA SOURCES")
        print("=" * 60)
        
        print("\nAvailable alternative sources:")
        for key, source in self.alternative_sources.items():
            print(f"  {key}: {source['name']}")
            print(f"     {source['description']}")
        
        choice = input("\nEnter source name to try (or press Enter to skip): ").strip()
        
        if choice and choice in self.alternative_sources:
            df = self.download_from_alternative_source(choice)
            if df is not None:
                print(f"\nSuccessfully downloaded {len(df)} records from {choice}")
                return df
        
        # Step 3: Create sample data
        print("\n" + "=" * 60)
        print("CREATING COMPREHENSIVE SAMPLE DATA")
        print("=" * 60)
        
        print("\nSince AnimalTFDB is not accessible and no local files were found,")
        print("creating a comprehensive sample dataset for analysis.")
        
        df = self.create_comprehensive_sample_data()
        
        print("\n" + "=" * 70)
        print("NEXT STEPS FOR REAL ANIMALTFDB DATA:")
        print("=" * 70)
        print("\nTo obtain real AnimalTFDB data:")
        print("1. Visit: http://bioinfo.life.hust.edu.cn/AnimalTFDB/")
        print("2. Download required files")
        print("3. Place them in the 'tf_data' folder")
        print("4. Run this script again")
        print("\nIf the website is blocked, try:")
        print("• Using a VPN")
        print("• Contacting: animaltfdb@163.com")
        print("• Using institutional access if available")
        
        return df


def main():
    """
    Main function to run the processor
    """
    processor = AnimalTFDBProcessor(data_dir="animaltfdb_data")
    
    # Run the processor
    data = processor.run()
    
    if data is not None:
        print("\n" + "=" * 70)
        print("DATA PROCESSING COMPLETE")
        print("=" * 70)
        print("\nData is now available in the 'animaltfdb_data' folder")
        print("\nYou can now:")
        print("1. Analyze the TF data")
        print("2. Use it for network analysis")
        print("3. Integrate with other datasets")
        print("4. Develop TF-related applications")
    else:
        print("\nFailed to obtain TF data.")


if __name__ == "__main__":
    main()
