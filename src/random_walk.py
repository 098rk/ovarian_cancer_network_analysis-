import networkx as nx
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RandomWalkAnalyzer:
    """Random walk analysis for ovarian cancer networks."""
    
    def __init__(self, restart_prob: float = 0.15, convergence_threshold: float = 1e-6):
        self.restart_prob = restart_prob
        self.convergence_threshold = convergence_threshold
    
    def setup_network_with_weights(self, edges_data: List[Tuple], mutation_data: Dict) -> nx.DiGraph:
        """Set up network with biologically informed edge weights."""
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
        """Normalize edge weights to [0,1] range."""
        max_weight = max(data['weight'] for _, _, data in G.edges(data=True))
        if max_weight > 0:
            for u, v in G.edges():
                G[u][v]['weight'] /= max_weight
    
    def random_walk_with_restart(self, G: nx.DiGraph, num_walks: int = 5000, 
                               walk_length: int = 15) -> Dict[str, Any]:
        """Perform random walks with restart and track convergence."""
        nodes = list(G.nodes())
        node_visits = {node: 0 for node in nodes}
        convergence_data = []
        
        logger.info(f"Starting {num_walks} random walks...")
        
        for walk_idx in tqdm(range(num_walks), desc="Random Walks"):
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
        """Calculate convergence metric."""
        total_visits = sum(node_visits.values())
        if total_visits == 0:
            return 0.0
        
        # Proportion of unique nodes visited
        unique_visited = sum(1 for count in node_visits.values() if count > 0)
        total_nodes = len(node_visits)
        
        return unique_visited / total_nodes
    
    def identify_significant_nodes(self, results: Dict, threshold: float = 0.01) -> List[str]:
        """Identify statistically significant nodes based on visit frequency."""
        visit_frequencies = results['visit_frequencies']
        significant_nodes = [
            node for node, freq in visit_frequencies.items() 
            if freq > threshold
        ]
        
        # Sort by significance
        significant_nodes.sort(key=lambda x: visit_frequencies[x], reverse=True)
        
        logger.info(f"Identified {len(significant_nodes)} significant nodes")
        return significant_nodes
    
    def calculate_network_robustness(self, G: nx.DiGraph, num_removals: int = 100) -> float:
        """Calculate network robustness under random node removal."""
        if G.number_of_nodes() == 0:
            return 0.0
            
        original_components = nx.number_weakly_connected_components(G)
        retained_connectivity = 0
        
        for _ in range(num_removals):
            G_temp = G.copy()
            nodes_to_remove = np.random.choice(
                list(G_temp.nodes()), 
                size=max(1, int(0.1 * G_temp.number_of_nodes())),  # Remove 10% of nodes
                replace=False
            )
            G_temp.remove_nodes_from(nodes_to_remove)
            
            # Calculate remaining connectivity
            current_components = nx.number_weakly_connected_components(G_temp)
            if current_components <= original_components * 1.15:  # Within 15% tolerance
                retained_connectivity += 1
        
        robustness = retained_connectivity / num_removals
        logger.info(f"Network robustness: {robustness:.2%}")
        
        return robustness
    
    def perform_random_walks(self, data_dict: Dict) -> Dict[str, Any]:
        """Main method to perform random walk analysis."""
        try:
            # Prepare sample data
            edges_data = self._prepare_sample_edges()
            mutation_data = self._prepare_sample_mutations()
            
            # Create network
            G = self.setup_network_with_weights(edges_data, mutation_data)
            
            # Perform random walks
            results = self.random_walk_with_restart(G)
            
            # Identify significant nodes
            significant_nodes = self.identify_significant_nodes(results)
            
            # Calculate robustness
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
    
    def _prepare_sample_edges(self) -> List[Tuple]:
        """Prepare sample edge data for testing."""
        return [
            ('p53', 'apoptosis', 0.9),
            ('NFkB', 'apoptosis', 0.7),
            ('ATM', 'p53', 0.85),
            ('AKT', 'p53', 0.75),
            ('EGFR', 'AKT', 0.8)
        ]
    
    def _prepare_sample_mutations(self) -> Dict[str, float]:
        """Prepare sample mutation data for testing."""
        return {
            'p53': 0.85, 'NFkB': 0.45, 'ATM': 0.25,
            'AKT': 0.35, 'EGFR': 0.30, 'apoptosis': 0.10
        }
