import pandas as pd
import urllib.request
import gzip
import logging
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

class PathwayCommonsLoader:
    """
    Data loader for Pathway Commons with robust error handling and chunked processing.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.base_url = "http://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.All.BINARY_SIF.gz"
    
    def download_with_retry(self, max_retries: int = 3) -> str:
        """
        Download file with retry mechanism and progress tracking.
        """
        local_filename = os.path.join(self.output_dir, "PathwayCommons12.All.BINARY_SIF.gz")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1} for Pathway Commons data")
                
                def report_progress(block_num, block_size, total_size):
                    if total_size > 0:
                        percent = min(100, (block_num * block_size * 100) // total_size)
                        if block_num % 100 == 0:  # Log every 100 blocks
                            logger.info(f"Download progress: {percent}%")
                
                urllib.request.urlretrieve(
                    self.base_url, 
                    local_filename,
                    report_progress
                )
                
                logger.info("Download completed successfully")
                return local_filename
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All download attempts failed")
                    raise
        
        return local_filename
    
    def process_data_chunked(self, input_file: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Process large files in chunks to manage memory usage.
        """
        logger.info(f"Processing Pathway Commons data in chunks of {chunk_size} rows")
        
        interaction_types = {
            "controls-state-change-of", 
            "controls-phosphorylation-of", 
            "controls-expression-of"
        }
        
        processed_chunks = []
        
        try:
            with gzip.open(input_file, 'rt') as f:
                chunk = []
                for i, line in enumerate(f):
                    chunk.append(line.strip().split("\t"))
                    
                    if len(chunk) >= chunk_size:
                        df_chunk = self._process_chunk(chunk, interaction_types)
                        processed_chunks.append(df_chunk)
                        chunk = []
                        logger.info(f"Processed {i + 1} lines...")
                
                # Process remaining lines
                if chunk:
                    df_chunk = self._process_chunk(chunk, interaction_types)
                    processed_chunks.append(df_chunk)
            
            # Combine all chunks
            if processed_chunks:
                final_df = pd.concat(processed_chunks, ignore_index=True)
                logger.info(f"Processing complete. Final dataset: {len(final_df)} interactions")
                return final_df
            else:
                logger.warning("No data processed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def _process_chunk(self, chunk: list, interaction_types: set) -> pd.DataFrame:
        """Process a single chunk of data"""
        df_chunk = pd.DataFrame(chunk, columns=[
            "Participant A", "Interaction Type", "Participant B", 
            "Source", "PubMed ID", "Pathway Names"
        ])
        
        # Filter by interaction type
        df_filtered = df_chunk[df_chunk["Interaction Type"].isin(interaction_types)]
        
        # Select relevant columns
        df_filtered = df_filtered[["Participant A", "Participant B", "Interaction Type"]]
        
        return df_filtered
    
    def download_and_process(self, filename: str = "PathwayCommons_processed.csv") -> pd.DataFrame:
        """
        Main method to download and process Pathway Commons data.
        """
        try:
            # Download data
            downloaded_file = self.download_with_retry()
            
            # Process data in chunks
            processed_data = self.process_data_chunked(downloaded_file)
            
            # Save processed data
            output_path = os.path.join(self.output_dir, filename)
            processed_data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Pathway Commons processing failed: {e}")
            raise

# Maintain backward compatibility with original function
def DownloadAndProcessPathwayCommonsData(filename: str = "PathwayCommons.csv") -> str:
    """
    Legacy function for backward compatibility.
    """
    loader = PathwayCommonsLoader()
    loader.download_and_process(filename)
    return "Data downloaded and processed successfully."
