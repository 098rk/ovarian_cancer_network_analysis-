"""
Data acquisition module for ovarian cancer network analysis.

This module handles downloading and preprocessing data from:
- Pathway Commons
- CellTalkDB
- AnimalTFDB  
- TCGA-OV (GDC)
"""

from .pathway_commons import PathwayCommonsLoader
from .celltalk_db import CellTalkLoader
from .animal_tf import AnimalTFLoader
from .tcga_ov import TCGALoader

__all__ = [
    'PathwayCommonsLoader',
    'CellTalkLoader', 
    'AnimalTFLoader',
    'TCGALoader'
]
