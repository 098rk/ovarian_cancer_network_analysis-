import pandas as pd
import requests
import os
from io import StringIO
import time


def download_celltalk_file(filename, save_as=None):
    """Download a specific file from CellTalkDB"""

    # Base URL
    base_url = "https://xomics.com.cn/celltalkdb/"

    # Form data based on the HTML structure
    form_data = {
        'ip': '127.0.0.1',  # Can be any IP
        'agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'ref': 'direct',
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'folder': 'download/processed_data/',
        'filename': filename
    }

    # URL for form submission
    url = base_url + "handler/download.php"

    # Headers to simulate a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/plain,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': base_url,
        'Referer': base_url + 'download.php',
    }

    try:
        print(f"Downloading {filename}...")

        # Send POST request
        response = requests.post(url, data=form_data, headers=headers, timeout=30)

        if response.status_code == 200:
            content = response.text

            # Check if we got actual data or an error page
            if len(content) < 100 or "<html" in content.lower():
                print(f"Warning: Received HTML instead of data for {filename}")
                return None

            # Set output filename
            if save_as is None:
                save_as = filename.replace('.txt', '.csv')

            # Save raw content
            raw_filename = filename.replace('.txt', '_raw.txt')
            with open(raw_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Raw data saved to {raw_filename}")

            # Try to parse as DataFrame
            try:
                # Try tab-separated first
                df = pd.read_csv(StringIO(content), sep='\t')
                print(f"  Successfully parsed as tab-separated")
            except:
                # Try comma-separated
                try:
                    df = pd.read_csv(StringIO(content), sep=',')
                    print(f"  Successfully parsed as comma-separated")
                except:
                    # Try space-separated
                    try:
                        df = pd.read_csv(StringIO(content), sep='\\s+')
                        print(f"  Successfully parsed as space-separated")
                    except Exception as e:
                        print(f"  Error parsing data: {e}")
                        print(f"  First 500 chars of response:")
                        print(content[:500])
                        return None

            # Check if DataFrame has data
            if len(df) > 0:
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

                # Save as CSV
                df.to_csv(save_as, index=False)
                print(f"  Saved to {save_as}")

                return df
            else:
                print(f"  Warning: Empty DataFrame for {filename}")
                return None

        else:
            print(f"  Error: HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None


def download_human_lr_pairs():
    """Download human ligand-receptor pairs specifically"""

    filename = "human_lr_pair.txt"

    # Column names for human LR pairs
    col_names = [
        "lr_pair",
        "ligand_gene_symbol",
        "receptor_gene_symbol",
        "ligand_gene_id",
        "receptor_gene_id",
        "ligand_ensembl_protein_id",
        "receptor_ensembl_protein_id",
        "ligand_ensembl_gene_id",
        "receptor_ensembl_gene_id",
        "evidence"
    ]

    df = download_celltalk_file(filename, "human_lr_pairs.csv")

    if df is not None:
        # Check if we need to assign column names
        if len(df.columns) == len(col_names):
            if 'lr_pair' not in df.columns:
                df.columns = col_names
                print("\nAssigned column names to human LR pairs")

        print(f"\nHuman LR Pairs Summary:")
        print(f"  Total pairs: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few pairs:")
        print(df.head())

        # Get unique ligands and receptors
        unique_ligands = df['ligand_gene_symbol'].nunique()
        unique_receptors = df['receptor_gene_symbol'].nunique()
        print(f"\nUnique ligands: {unique_ligands}")
        print(f"Unique receptors: {unique_receptors}")

        return df

    return None


def download_all_processed_data():
    """Download all processed data files from CellTalkDB"""

    # List of processed data files from the HTML
    processed_files = [
        ('human_gene_info.txt', 'Human gene information'),
        ('human_gene2ensembl.txt', 'Human genes labeled by ensembl database'),
        ('human_lr_pair.txt', 'Human ligand-receptor interaction pairs'),
        ('human_uniprot.txt', 'Human protein-coding genes from uniprot database'),
        ('mouse_gene_info.txt', 'Mouse gene information'),
        ('mouse_gene2ensembl.txt', 'Mouse genes labeled by ensembl database'),
        ('mouse_lr_pair.txt', 'Mouse ligand-receptor interaction pairs'),
        ('mouse_uniprot.txt', 'Mouse protein-coding genes from uniprot database'),
    ]

    downloaded = {}

    print("=" * 60)
    print("Downloading all processed data from CellTalkDB")
    print("=" * 60)

    for filename, description in processed_files:
        print(f"\n[{filename}]")
        print(f"Description: {description}")

        # Special handling for LR pairs
        if 'lr_pair' in filename:
            if 'human' in filename:
                df = download_human_lr_pairs()
            else:
                # For mouse LR pairs, use default column names
                df = download_celltalk_file(filename, f"mouse_lr_pairs.csv")
                if df is not None and len(df.columns) == 10:
                    df.columns = col_names  # Use same column names
        else:
            df = download_celltalk_file(filename)

        if df is not None:
            downloaded[filename] = {
                'dataframe': df,
                'rows': len(df),
                'columns': len(df.columns),
                'description': description
            }

    return downloaded


def download_raw_data():
    """Download raw data files (larger datasets)"""

    # Raw data files from the HTML
    raw_files = [
        ('9606.protein.info.v11.0.txt', 'human protein information of PPIs from STRING'),
        ('9606.protein.links.v11.0.txt', 'human directed PPIs information from STRING'),
        ('10090.protein.info.v11.0.txt', 'mouse protein information of PPIs from STRING'),
        ('10090.protein.links.v11.0.txt', 'mouse directed PPIs information from STRING'),
        ('gene2ensembl', 'matchup of NCBI Gene to Ensemble protein and gene from NCBI'),
        ('gene2go', 'matchup of NCBI Gene to Genetic ontology (GO) from NCBI'),
        ('gene_orthologs', 'matchup of NCBI human gene to mouse gene from NCBI'),
        ('Homo_sapiens.gene_info', 'human gene information from NCBI'),
        ('Homo_sapiens.GRCh38.99.entrez.tsv', 'human protein/gene information from Ensemble'),
        ('human_uniprot.tab', 'human protein information from UniProt'),
        ('Mus_musculus.gene_info', 'mouse gene information from NCBI'),
        ('Mus_musculus.GRCm38.99.entrez.tsv', 'mouse protein/gene information from Ensemble'),
        ('mouse_uniprot.tab', 'mouse protein information from UniProt'),
    ]

    print("\n" + "=" * 60)
    print("Note: Raw data files are very large (up to 600MB)")
    print("Only download these if you have sufficient disk space")
    print("=" * 60)

    response = input("\nDownload raw data files? (y/n): ")

    if response.lower() == 'y':
        downloaded_raw = {}

        for filename, description in raw_files[:4]:  # Just first 4 to avoid huge downloads
            print(f"\n[{filename}]")
            print(f"Description: {description}")
            print("Warning: This file may be very large!")

            confirm = input(f"Download {filename}? (y/n): ")
            if confirm.lower() == 'y':
                # For raw data, we might need different form data
                form_data = {
                    'ip': '127.0.0.1',
                    'agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'ref': 'direct',
                    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'folder': 'download/raw_data/',  # Different folder for raw data
                    'filename': filename
                }

                url = "https://xomics.com.cn/celltalkdb/handler/download.php"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://xomics.com.cn/celltalkdb/download.php',
                }

                try:
                    print(f"Downloading {filename}...")
                    response = requests.post(url, data=form_data, headers=headers, stream=True, timeout=60)

                    if response.status_code == 200:
                        # Save directly to file (stream to handle large files)
                        with open(filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                        print(f"  Saved {filename} ({os.path.getsize(filename) / 1024 / 1024:.2f} MB)")
                        downloaded_raw[filename] = os.path.getsize(filename)

                except Exception as e:
                    print(f"  Error: {e}")

    return


def analyze_human_lr_pairs(df):
    """Analyze the human ligand-receptor pairs data"""

    if df is None or len(df) == 0:
        print("No data to analyze")
        return

    print("\n" + "=" * 60)
    print("ANALYSIS OF HUMAN LIGAND-RECEPTOR PAIRS")
    print("=" * 60)

    # Basic statistics
    print(f"\nTotal LR pairs: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Count unique values
    unique_stats = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_stats[col] = unique_count

    print("\nUnique values per column:")
    for col, count in unique_stats.items():
        print(f"  {col}: {count}")

    # Most common ligands and receptors
    top_ligands = df['ligand_gene_symbol'].value_counts().head(10)
    top_receptors = df['receptor_gene_symbol'].value_counts().head(10)

    print("\nTop 10 ligands (most interactions):")
    print(top_ligands)

    print("\nTop 10 receptors (most interactions):")
    print(top_receptors)

    # Evidence distribution
    if 'evidence' in df.columns:
        evidence_counts = df['evidence'].value_counts().head(10)
        print("\nTop 10 evidence sources:")
        print(evidence_counts)

    # Save analysis results
    analysis_file = "human_lr_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("ANALYSIS OF HUMAN LIGAND-RECEPTOR PAIRS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total LR pairs: {len(df)}\n")
        f.write(f"Unique ligands: {unique_stats.get('ligand_gene_symbol', 0)}\n")
        f.write(f"Unique receptors: {unique_stats.get('receptor_gene_symbol', 0)}\n\n")

        f.write("Top 10 ligands:\n")
        f.write(str(top_ligands) + "\n\n")

        f.write("Top 10 receptors:\n")
        f.write(str(top_receptors) + "\n")

    print(f"\nAnalysis saved to {analysis_file}")

    return df


if __name__ == '__main__':
    print("CellTalkDB Data Downloader")
    print("=" * 60)

    # Create output directory
    output_dir = "celltalkdb_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)
    print(f"Output directory: {os.getcwd()}")

    # Menu for user selection
    print("\nSelect download option:")
    print("1. Download human ligand-receptor pairs only")
    print("2. Download all processed data")
    print("3. Download raw data (warning: very large files)")
    print("4. Download everything")

    choice = input("\nEnter choice (1-4): ")

    if choice == '1':
        print("\nDownloading human ligand-receptor pairs...")
        df = download_human_lr_pairs()
        if df is not None:
            analyze_human_lr_pairs(df)

    elif choice == '2':
        print("\nDownloading all processed data...")
        downloaded = download_all_processed_data()

        print(f"\n{'=' * 60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'=' * 60}")
        for filename, info in downloaded.items():
            print(f"\n{filename}:")
            print(f"  Description: {info['description']}")
            print(f"  Rows: {info['rows']:,}")
            print(f"  Columns: {info['columns']}")
            print(f"  Saved as: {filename.replace('.txt', '.csv')}")

    elif choice == '3':
        download_raw_data()

    elif choice == '4':
        print("\nDownloading all data...")
        downloaded = download_all_processed_data()
        download_raw_data()

    else:
        print("Invalid choice. Exiting.")

    print("\n" + "=" * 60)
    print("Download process completed!")
    print(f"Files saved in: {os.getcwd()}")
    print("=" * 60)1
