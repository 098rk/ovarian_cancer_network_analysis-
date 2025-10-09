import networkx as nx
import pandas as pd
import logging
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class CentralityAnalyzer:
    """Comprehensive centrality analysis for ovarian cancer networks."""
    
    def __init__(self):
        self.centrality_measures = [
            'degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank'
        ]
    
    def calculate_all_centralities(self, G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Calculate multiple centrality measures."""
        centralities = {}
        
        try:
            # Degree centrality
            centralities['degree'] = nx.degree_centrality(G)
            
            # Betweenness centrality (can be slow for large networks)
            if G.number_of_nodes() < 1000:
                centralities['betweenness'] = nx.betweenness_centrality(G, weight='weight')
            else:
                # Sample for large networks
                centralities['betweenness'] = nx.betweenness_centrality(
                    G, weight='weight', k=min(100, G.number_of_nodes())
                )
            
            # Closeness centrality
            centralities['closeness'] = nx.closeness_centrality(G)
            
            # Eigenvector centrality
            try:
                centralities['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            except:
                centralities['eigenvector'] = {node: 0 for node in G.nodes()}
            
            # PageRank
            centralities['pagerank'] = nx.pagerank(G, weight='weight')
            
            logger.info(f"Calculated {len(centralities)} centrality measures")
            return centralities
            
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            raise
    
    def identify_high_centrality_nodes(self, centralities: Dict, top_n: int = 10) -> Dict[str, List[str]]:
        """Identify top nodes for each centrality measure."""
        top_nodes = {}
        
        for measure, scores in centralities.items():
            sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_nodes[measure] = [node for node, score in sorted_nodes[:top_n]]
        
        logger.info(f"Identified top nodes for {len(top_nodes)} centrality measures")
        return top_nodes
    
    def calculate_centrality_correlation(self, centralities: Dict) -> pd.DataFrame:
        """Calculate correlation between different centrality measures."""
        # Create DataFrame
        df = pd.DataFrame(centralities)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        logger.info("Calculated centrality correlations")
        return correlation_matrix
    
    def find_consensus_nodes(self, top_nodes: Dict[str, List[str]], min_measures: int = 3) -> List[str]:
        """Find nodes that are important across multiple centrality measures."""
        node_counts = {}
        
        for measure, nodes in top_nodes.items():
            for node in nodes:
                node_counts[node] = node_counts.get(node, 0) + 1
        
        # Filter nodes that appear in multiple measures
        consensus_nodes = [node for node, count in node_counts.items() if count >= min_measures]
        consensus_nodes.sort(key=lambda x: node_counts[x], reverse=True)
        
        logger.info(f"Found {len(consensus_nodes)} consensus nodes across measures")
        return consensus_nodes
    
    def analyze_centrality(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Comprehensive centrality analysis."""
        try:
            # Calculate all centralities
            centralities = self.calculate_all_centralities(G)
            
            # Identify top nodes
            top_nodes = self.identify_high_centrality_nodes(centralities)
            
            # Find consensus nodes
            consensus_nodes = self.find_consensus_nodes(top_nodes)
            
            # Calculate correlations
            correlation_matrix = self.calculate_centrality_correlation(centralities)
            
            return {
                'centralities': centralities,
                'top_nodes': top_nodes,
                'consensus_nodes': consensus_nodes,
                'correlation_matrix': correlation_matrix
            }
            
        except Exception as e:
            logger.error(f"Centrality analysis failed: {e}")
            raise
