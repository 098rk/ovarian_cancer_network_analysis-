import networkx as nx
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BooleanNetworkAnalyzer:
    """
    Boolean network analysis for ovarian cancer signaling pathways.
    """
    
    def __init__(self, time_steps: int = 50, convergence_threshold: float = 1e-4):
        self.time_steps = time_steps
        self.convergence_threshold = convergence_threshold
    
    def create_ovarian_cancer_network(self) -> nx.DiGraph:
        """Create comprehensive ovarian cancer signaling network."""
        G = nx.DiGraph()
        
        # Define nodes with properties
        nodes_data = [
            ('p53', {'type': 'transcription_factor', 'function': 'tumor_suppressor'}),
            ('NFkB', {'type': 'transcription_factor', 'function': 'inflammatory_response'}),
            ('ATM', {'type': 'kinase', 'function': 'dna_damage_response'}),
            ('AKT', {'type': 'kinase', 'function': 'cell_survival'}),
            ('EGFR', {'type': 'receptor', 'function': 'growth_signaling'}),
            ('BRCA1', {'type': 'dna_repair', 'function': 'tumor_suppressor'}),
            ('apoptosis', {'type': 'phenotype', 'function': 'cell_death'}),
            ('cell_cycle_arrest', {'type': 'phenotype', 'function': 'growth_control'})
        ]
        
        G.add_nodes_from(nodes_data)
        
        # Define interactions
        edges_data = [
            ('p53', 'apoptosis', {'type': 'activation', 'weight': 0.9}),
            ('p53', 'cell_cycle_arrest', {'type': 'activation', 'weight': 0.8}),
            ('NFkB', 'apoptosis', {'type': 'inhibition', 'weight': 0.7}),
            ('ATM', 'p53', {'type': 'activation', 'weight': 0.85}),
            ('AKT', 'p53', {'type': 'inhibition', 'weight': 0.75}),
            ('EGFR', 'AKT', {'type': 'activation', 'weight': 0.8}),
            ('BRCA1', 'ATM', {'type': 'activation', 'weight': 0.9})
        ]
        
        G.add_edges_from(edges_data)
        logger.info(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def simulate_boolean_dynamics(self, G: nx.DiGraph, initial_states: Dict = None) -> pd.DataFrame:
        """Simulate Boolean network dynamics."""
        if initial_states is None:
            # Initialize random states
            initial_states = {node: np.random.choice([0, 1]) for node in G.nodes()}
        
        states_history = {0: initial_states.copy()}
        
        for t in range(1, self.time_steps):
            current_states = states_history[t-1].copy()
            new_states = {}
            
            for node in G.nodes():
                # Get input nodes
                inputs = list(G.predecessors(node))
                
                if not inputs:  # No inputs, maintain state
                    new_states[node] = current_states[node]
                else:
                    # Sum weighted inputs
                    total_input = 0
                    for input_node in inputs:
                        weight = G[input_node][node].get('weight', 1.0)
                        total_input += current_states[input_node] * weight
                    
                    # Update rule: activate if total input > threshold
                    threshold = len(inputs) * 0.5  # Average threshold
                    new_states[node] = 1 if total_input > threshold else 0
            
            states_history[t] = new_states
            
            # Check for convergence
            if t > 1:
                changes = sum(states_history[t][node] != states_history[t-1][node] 
                            for node in G.nodes())
                if changes == 0:
                    logger.info(f"Network converged at time step {t}")
                    break
        
        # Convert to DataFrame
        states_df = pd.DataFrame.from_dict(states_history, orient='index')
        return states_df
    
    def identify_pivotal_nodes(self, G: nx.DiGraph, states_df: pd.DataFrame) -> List[str]:
        """Identify nodes that significantly influence network dynamics."""
        # Calculate node influence based on state changes
        node_influence = {}
        
        for node in G.nodes():
            state_changes = np.diff(states_df[node].values).sum()
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            # Combined influence score
            influence_score = (state_changes + out_degree) / (in_degree + 1)
            node_influence[node] = influence_score
        
        # Sort by influence
        pivotal_nodes = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Identified {len(pivotal_nodes)} pivotal nodes")
        
        return [node for node, score in pivotal_nodes[:5]]  # Top 5 nodes
    
    def analyze_pathways(self, data_dict: Dict) -> Dict[str, Any]:
        """Main analysis method integrating multiple data sources."""
        try:
            # Create network
            G = self.create_ovarian_cancer_network()
            
            # Simulate dynamics
            states_df = self.simulate_boolean_dynamics(G)
            
            # Identify key nodes
            pivotal_nodes = self.identify_pivotal_nodes(G, states_df)
            
            return {
                'network': G,
                'states_history': states_df,
                'pivotal_nodes': pivotal_nodes,
                'convergence_step': len(states_df) - 1
            }
            
        except Exception as e:
            logger.error(f"Boolean network analysis failed: {e}")
            raise
