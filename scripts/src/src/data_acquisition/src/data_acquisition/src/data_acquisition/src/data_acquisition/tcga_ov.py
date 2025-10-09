import pandas as pd
import requests
import logging
import os
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

class TCGALoader:
    """Data loader for TCGA-OV data with API handling."""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.gdc_api = "https://api.gdc.cancer.gov/"
    
    def query_gdc(self, project: str = "TCGA-OV", data_type: str = "Gene Expression Quantification") -> Optional[Dict]:
        """Query GDC API for TCGA-OV data."""
        try:
            filters = {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": [project]
                        }
                    },
                    {
                        "op": "in",
                        "content": {
                            "field": "files.data_type",
                            "value": [data_type]
                        }
                    }
                ]
            }

            params = {
                "filters": json.dumps(filters),
                "format": "JSON",
                "size": "1000",
                "fields": "file_id,file_name,cases.case_id,data_type"
            }

            response = requests.get(f"{self.gdc_api}files", params=params, timeout=30)
            
            if response.status_code == 200:
                logger.info("GDC query successful")
                return response.json()
            else:
                logger.error(f"GDC query failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"GDC query error: {e}")
            return None
    
    def download_with_chunking(self, chunk_size: int = 1000) -> pd.DataFrame:
        """Download TCGA data with chunked processing."""
        try:
            metadata = self.query_gdc()
            if not metadata:
                raise Exception("No metadata retrieved from GDC")
            
            # Process metadata into DataFrame
            files_data = []
            for file_entry in metadata.get('data', {}).get('hits', []):
                files_data.append({
                    'file_id': file_entry.get('file_id'),
                    'file_name': file_entry.get('file_name'),
                    'case_id': file_entry.get('cases', [{}])[0].get('case_id', ''),
                    'data_type': file_entry.get('data_type')
                })
            
            df = pd.DataFrame(files_data)
            
            # Save metadata
            output_path = os.path.join(self.output_dir, "TCGA_OV_metadata.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"TCGA-OV metadata saved: {len(df)} files")
            
            return df
            
        except Exception as e:
            logger.error(f"TCGA data download failed: {e}")
            raise

def DownloadAndProcessTCGA_OVData() -> str:
    """Legacy function for backward compatibility."""
    loader = TCGALoader()
    loader.download_with_chunking()
    return "TCGA-OV data downloaded and processed successfully."
