import networkx as nx
import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PageRankAnalyzer:
    """PageRank analysis for identifying key nodes in ovarian cancer networks."""
    
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def calculate_pagerank(self, G: nx.DiGraph, personalization: Dict = None) -> Dict[str, float]:
        """Calculate PageRank scores with optional personalization."""
        try:
            pagerank_scores = nx.pagerank(
                G, 
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                personalization=personalization,
                weight='weight'
            )
            
            logger.info(f"PageRank calculated for {len(pagerank_scores)} nodes")
            return pagerank_scores
            
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
            raise
    
    def identify_top_nodes(self, pagerank_scores: Dict[str, float], top_n: int = 10) -> List[str]:
        """Identify top nodes based on PageRank scores."""
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, score in sorted_nodes[:top_n]]
        
        logger.info(f"Identified top {len(top_nodes)} nodes by PageRank")
        return top_nodes
    
    def calculate_convergence(self, G: nx.DiGraph) -> int:
        """Calculate number of iterations to convergence."""
        # Custom implementation to track convergence
        n = G.number_of_nodes()
        M = nx.to_numpy_array(G, weight='weight')
        
        # Normalize columns
        M = M / np.where(M.sum(axis=0) > 0, M.sum(axis=0), 1)
        
        # Personalization vector
        v = np.ones(n) / n
        
        # Dangling nodes
        dangling_nodes = np.where(M.sum(axis=0) == 0)[0]
        dangling_weights = np.zeros(n)
        if len(dangling_nodes) > 0:
            dangling_weights[dangling_nodes] = 1.0 / len(dangling_nodes)
        
        # Power iteration
        x = np.ones(n) / n
        for i in range(self.max_iter):
            x_last = x.copy()
            
            # PageRank update
            x = (self.alpha * (x_last @ M + 
                 dangling_weights * x_last.sum()) + 
                 (1 - self.alpha) * v)
            
            # Check convergence
            if np.linalg.norm(x - x_last, 1) < self.tol:
                logger.info(f"PageRank converged in {i+1} iterations")
                return i + 1
        
        logger.warning(f"PageRank did not converge in {self.max_iter} iterations")
        return self.max_iter
    
    def analyze_centrality(self, data_dict: Dict) -> Dict[str, Any]:
        """Main analysis method."""
        try:
            # Create or use existing network
            if 'network' in data_dict:
                G = data_dict['network']
            else:
                # Create sample network
                G = self._create_sample_network()
            
            # Calculate PageRank
            pagerank_scores = self.calculate_pagerank(G)
            
            # Identify top nodes
            top_nodes = self.identify_top_nodes(pagerank_scores)
            
            # Calculate convergence
            convergence_iter = self.calculate_convergence(G)
            
            return {
                'pagerank_scores': pagerank_scores,
                'top_nodes': top_nodes,
                'convergence_iterations': convergence_iter,
                'network': G
            }
            
        except Exception as e:
            logger.error(f"PageRank analysis failed: {e}")
            raise
    
    def _create_sample_network(self) -> nx.DiGraph:
        """Create sample ovarian cancer network for testing."""
        G = nx.DiGraph()
        
        # Add nodes and edges
        nodes = ['p53', 'NFkB', 'ATM', 'AKT', 'EGFR', 'BRCA1', 'apoptosis']
        edges = [
            ('p53', 'apoptosis'), ('NFkB', 'apoptosis'), 
            ('ATM', 'p53'), ('AKT', 'p53'), ('EGFR', 'AKT')
        ]
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
        
        return G
