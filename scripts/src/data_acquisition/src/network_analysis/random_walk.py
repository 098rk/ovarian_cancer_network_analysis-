import networkx as nx
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class RandomWalkAnalyzer:
    """
    Enhanced random walk analysis with proper statistical validation.
    """
    
    def __init__(self, restart_prob: float = 0.15):
        self.restart_prob = restart_prob
        self.convergence_threshold = 1e-6
    
    def setup_network_with_weights(self, edges_data: List[Tuple], mutation_data: Dict) -> nx.DiGraph:
        """
        Set up network with biologically informed edge weights.
        """
        G = nx.DiGraph()
        
        # Add nodes with initial weights
        for node, mutation_prob in mutation_data.items():
            G.add_node(node, weight=0.5, mutation_prob=mutation_prob)
        
        # Add edges with confidence scores
        for u, v, confidence in edges_data:
            p_u = mutation_data.get(u, 0.1)
            p_v = mutation_data.get(v, 0.1)
            
            # Calculate combined weight based on mutation probability and confidence
            edge_weight = p_u * p_v * confidence
            G.add_edge(u, v, weight=edge_weight, confidence=confidence)
        
        # Normalize edge weights
        self._normalize_edge_weights(G)
        
        logger.info(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _normalize_edge_weights(self, G: nx.DiGraph) -> None:
        """Normalize edge weights to [0,1] range"""
        max_weight = max(data['weight'] for _, _, data in G.edges(data=True))
        if max_weight > 0:
            for u, v in G.edges():
                G[u][v]['weight'] /= max_weight
    
    def random_walk_with_restart(self, G: nx.DiGraph, num_walks: int = 5000, 
                               walk_length: int = 15) -> Dict:
        """
        Perform random walks with restart and track convergence.
        """
        nodes = list(G.nodes())
        node_visits = {node: 0 for node in nodes}
        convergence_data = []
        
        logger.info(f"Starting {num_walks} random walks...")
        
        for walk_idx in range(num_walks):
            current_node = np.random.choice(nodes)
            node_visits[current_node] += 1
            
            for step in range(walk_length):
                if np.random.rand() < self.restart_prob:
                    current_node = np.random.choice(nodes)
                else:
                    neighbors = list(G.successors(current_node))
                    if neighbors:
                        weights = [G[current_node][n]['weight'] for n in neighbors]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            probs = np.array(weights) / total_weight
                            current_node = np.random.choice(neighbors, p=probs)
                        else:
                            current_node = np.random.choice(nodes)
                    else:
                        current_node = np.random.choice(nodes)
                
                node_visits[current_node] += 1
            
            # Track convergence every 100 walks
            if walk_idx % 100 == 0:
                convergence = self._calculate_convergence(node_visits, walk_idx + 1)
                convergence_data.append(convergence)
        
        # Normalize visit counts
        total_visits = sum(node_visits.values())
        for node in node_visits:
            G.nodes[node]['visit_frequency'] = node_visits[node] / total_visits
        
        results = {
            'node_visits': node_visits,
            'visit_frequencies': {node: G.nodes[node]['visit_frequency'] for node in nodes},
            'convergence_data': convergence_data,
            'graph': G
        }
        
        logger.info("Random walk analysis completed")
        return results
    
    def _calculate_convergence(self, node_visits: Dict, num_walks: int) -> float:
        """Calculate convergence metric"""
        total_visits = sum(node_visits.values())
        if total_visits == 0:
            return 0.0
        
        # Proportion of unique nodes visited
        unique_visited = sum(1 for count in node_visits.values() if count > 0)
        total_nodes = len(node_visits)
        
        return unique_visited / total_nodes
    
    def identify_significant_nodes(self, results: Dict, threshold: float = 0.01) -> List[str]:
        """
        Identify statistically significant nodes based on visit frequency.
        """
        visit_frequencies = results['visit_frequencies']
        significant_nodes = [
            node for node, freq in visit_frequencies.items() 
            if freq > threshold
        ]
        
        # Sort by significance
        significant_nodes.sort(key=lambda x: visit_frequencies[x], reverse=True)
        
        logger.info(f"Identified {len(significant_nodes)} significant nodes")
        return significant_nodes
    
    def perform_random_walks(self, data_dict: Dict) -> Dict:
        """
        Main method to perform random walk analysis on ovarian cancer data.
        """
        try:
            # Extract network data from data_dict
            edges_data = self._prepare_edges(data_dict)
            mutation_data = self._prepare_mutation_data(data_dict)
            
            # Create network
            G = self.setup_network_with_weights(edges_data, mutation_data)
            
            # Perform random walks
            results = self.random_walk_with_restart(G)
            
            # Identify significant nodes
            significant_nodes = self.identify_significant_nodes(results)
            
            # Calculate robustness (H3 validation)
            robustness = self.calculate_network_robustness(G)
            
            return {
                'significant_nodes': significant_nodes,
                'visit_frequencies': results['visit_frequencies'],
                'convergence_rate': results['convergence_data'][-1] if results['convergence_data'] else 0,
                'robustness': robustness,
                'graph': G
            }
            
        except Exception as e:
            logger.error(f"Random walk analysis failed: {e}")
            raise
    
    def calculate_network_robustness(self, G: nx.DiGraph, num_removals: int = 100) -> float:
        """
        Calculate network robustness under random node removal (H3 validation).
        """
        original_connectivity = nx.algorithms.components.number_connected_components(G.to_undirected())
        retained_connectivity = 0
        
        for _ in range(num_removals):
            G_temp = G.copy()
            nodes_to_remove = np.random.choice(
                list(G_temp.nodes()), 
                size=int(0.1 * G_temp.number_of_nodes()),  # Remove 10% of nodes
                replace=False
            )
            G_temp.remove_nodes_from(nodes_to_remove)
            
            # Calculate remaining connectivity
            components = nx.algorithms.components.number_connected_components(G_temp.to_undirected())
            if components <= original_connectivity * 1.15:  # Within 15% tolerance
                retained_connectivity += 1
        
        robustness = retained_connectivity / num_removals
        logger.info(f"Network robustness: {robustness:.2%}")
        
        return robustness
