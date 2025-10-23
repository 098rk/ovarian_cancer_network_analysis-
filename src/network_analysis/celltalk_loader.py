
import pandas as pd
import urllib.request
import gzip
import os

class CellTalkDataLoader:
    """
    A class to download and process CellTalk data.
    """
    def __init__(self, filename="CellTalk.csv", 
                 url="https://example.com/CellTalkData.gz"):
        self.filename = filename
        self.url = url
        self.output_file = "CellTalkData.gz"

    def download_data(self):
        """Downloads the CellTalk data file."""
        print("Downloading data...")
        urllib.request.urlretrieve(self.url, self.output_file)
        print("Download complete.")

    def process_data(self):
        """Extracts and processes the downloaded data into a structured format."""
        print("Processing data...")
        
        with gzip.open(self.output_file, 'rt') as f:
            data = [line.strip().split("\t") for line in f.readlines()]
        
        df = pd.DataFrame(data, columns=[
            "lr_pair", "ligand_gene_symbol", "receptor_gene_symbol",
            "ligand_gene_id", "receptor_gene_id", "ligand_ensembl_protein_id",
            "receptor_ensembl_protein_id", "ligand_ensembl_gene_id",
            "receptor_ensembl_gene_id", "evidence"
        ])

        # Save to CSV
        df.to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")

    def run(self):
        """Executes the full pipeline: downloading and processing."""
        self.download_data()
        self.process_data()
        print("Process completed successfully.")

if __name__ == "__main__":
    loader = CellTalkDataLoader(filename="D://ProjectFiles//CellTalk.csv")
    loader.run()
