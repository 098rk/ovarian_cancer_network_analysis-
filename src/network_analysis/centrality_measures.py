import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OvarianCancerPageRank:
    """PageRank analysis for ovarian cancer signaling network"""
    
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol
        self.G = nx.DiGraph()
        
    def build_network_from_data(self, interactions_data: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """Build directed graph from interaction data"""
        logger.info("Building ovarian cancer network from interaction data...")
        
        for source, target, interaction_type in interactions_data:
            # Add nodes with properties
            self.G.add_node(source, node_type=self._classify_node_type(source))
            self.G.add_node(target, node_type=self._classify_node_type(target))
            
            # Add edge with weight based on interaction type
            weight = self._calculate_edge_weight(interaction_type)
            self.G.add_edge(source, target, weight=weight, interaction_type=interaction_type)
        
        logger.info(f"Network built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G
    
    def _classify_node_type(self, node_name: str) -> str:
        """Classify nodes into biological categories"""
        node_lower = node_name.lower()
        
        if any(x in node_lower for x in ['mrna', 'mirna', 'pre-mirna']):
            return 'RNA'
        elif any(x in node_lower for x in ['p53', 'nfkb', 'creb', 'bax', 'akt', 'mdm2', 'ikk', 'kbp', 'chk2', 'atm', 'wip1']):
            return 'Protein'
        elif any(x in node_lower for x in ['dsb', 'dna', 'nuc', 'cyt']):
            return 'Cellular_Component'
        elif any(x in node_lower for x in ['apoptosis', 'arrest']):
            return 'Phenotype'
        elif any(x in node_lower for x in ['ir', 'tnf']):
            return 'Stimulus'
        else:
            return 'Other'
    
    def _calculate_edge_weight(self, interaction_type: str) -> float:
        """Assign weights based on interaction type confidence"""
        weight_mapping = {
            'interacts_with': 1.0,
            'controls-expression': 0.9,
            'controls-phosphorylation': 0.8,
            'in-complex-with': 0.7,
            'controls-state-change': 0.85,
            'controls-transport': 0.75
        }
        return weight_mapping.get(interaction_type, 0.5)
    
    def custom_pagerank(self, personalization: Dict[str, float] = None) -> Dict[str, float]:
        """
        Implement custom PageRank algorithm with biological considerations
        """
        logger.info("Running custom PageRank algorithm...")
        
        nodes = list(self.G.nodes())
        n = len(nodes)
        if n == 0:
            return {}
        
        # Initialize PageRank scores
        pagerank = {node: 1.0 / n for node in nodes}
        
        # Create personalization vector based on biological importance
        if personalization is None:
            personalization = self._create_biological_personalization()
        
        # Normalize personalization vector
        personalization_sum = sum(personalization.values())
        if personalization_sum > 0:
            personalization = {k: v / personalization_sum for k, v in personalization.items()}
        else:
            personalization = {node: 1.0 / n for node in nodes}
        
        convergence_history = []
        
        for iteration in range(self.max_iter):
            new_pagerank = {}
            total_change = 0
            
            for node in nodes:
                # Calculate contribution from incoming links
                incoming_score = 0.0
                for predecessor in self.G.predecessors(node):
                    # Get edge weight
                    edge_weight = self.G[predecessor][node].get('weight', 1.0)
                    
                    # Calculate out-degree with weights
                    out_degree = sum(self.G[predecessor][succ].get('weight', 1.0) 
                                   for succ in self.G.successors(predecessor))
                    
                    if out_degree > 0:
                        incoming_score += pagerank[predecessor] * (edge_weight / out_degree)
                
                # Apply damping factor and personalization
                new_score = (1 - self.damping_factor) * personalization.get(node, 0) + \
                           self.damping_factor * incoming_score
                
                new_pagerank[node] = new_score
                total_change += abs(new_score - pagerank[node])
            
            convergence_history.append(total_change)
            
            # Check convergence
            if total_change < self.tol:
                logger.info(f"PageRank converged after {iteration + 1} iterations")
                break
            
            pagerank = new_pagerank
        
        # Normalize final scores
        total_score = sum(pagerank.values())
        if total_score > 0:
            pagerank = {k: v / total_score for k, v in pagerank.items()}
        
        self.convergence_history = convergence_history
        return pagerank
    
    def _create_biological_personalization(self) -> Dict[str, float]:
        """Create personalization vector based on biological importance"""
        personalization = {}
        
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('node_type', 'Other')
            
            # Assign higher personalization to key biological entities
            if node_type == 'Phenotype':
                personalization[node] = 2.0  # High importance for phenotypes like apoptosis
            elif node_type == 'Protein':
                # Extra importance for key cancer proteins
                if any(x in node.lower() for x in ['p53', 'nfkb', 'akt', 'bax']):
                    personalization[node] = 2.0
                else:
                    personalization[node] = 1.5
            elif node_type == 'RNA':
                personalization[node] = 1.2  # Medium for RNA molecules
            elif node_type == 'Stimulus':
                personalization[node] = 1.0  # Medium for stimuli
            else:
                personalization[node] = 0.8  # Lower for other components
        
        return personalization
    
    def networkx_pagerank(self, weight: str = 'weight') -> Dict[str, float]:
        """Run NetworkX's built-in PageRank for comparison"""
        logger.info("Running NetworkX PageRank...")
        return nx.pagerank(self.G, alpha=self.damping_factor, weight=weight)
    
    def identify_key_players(self, pagerank_scores: Dict[str, float], top_k: int = 20) -> List[Tuple[str, float]]:
        """Identify top key players based on PageRank scores"""
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def analyze_node_roles(self, pagerank_scores: Dict[str, float]) -> pd.DataFrame:
        """Analyze node roles and their PageRank scores"""
        analysis_data = []
        
        for node, score in pagerank_scores.items():
            node_type = self.G.nodes[node].get('node_type', 'Unknown')
            in_degree = self.G.in_degree(node)
            out_degree = self.G.out_degree(node)
            
            analysis_data.append({
                'Node': node,
                'PageRank': score,
                'Node_Type': node_type,
                'In_Degree': in_degree,
                'Out_Degree': out_degree,
                'Total_Degree': in_degree + out_degree
            })
        
        return pd.DataFrame(analysis_data)
    
    def visualize_network(self, pagerank_scores: Dict[str, float], top_nodes: int = 25):
        """Visualize network with PageRank-based node sizing"""
        plt.figure(figsize=(20, 15))
        
        # Use spring layout
        pos = nx.spring_layout(self.G, k=3, iterations=100, seed=42)
        
        # Prepare node sizes based on PageRank
        node_sizes = [pagerank_scores[node] * 50000 + 200 for node in self.G.nodes()]
        
        # Get top nodes for highlighting
        top_node_list = [node for node, _ in self.identify_key_players(pagerank_scores, top_nodes)]
        
        # Color nodes based on type and importance
        node_colors = []
        for node in self.G.nodes():
            if node in top_node_list:
                node_colors.append('red')  # Highlight top nodes in red
            else:
                node_type = self.G.nodes[node].get('node_type', 'Other')
                color_map = {
                    'Protein': 'lightblue',
                    'RNA': 'lightgreen',
                    'Cellular_Component': 'yellow',
                    'Phenotype': 'orange',
                    'Stimulus': 'purple',
                    'Other': 'gray'
                }
                node_colors.append(color_map.get(node_type, 'gray'))
        
        # Draw the network
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.8, edgecolors='black')
        nx.draw_networkx_edges(self.G, pos, alpha=0.4, arrowstyle='->', 
                              arrowsize=15, edge_color='gray', width=1.5)
        
        # Only label top nodes to avoid clutter
        labels = {node: node for node in top_node_list}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=9, 
                               font_weight='bold')
        
        plt.title(f"Ovarian Cancer Signaling Network (100 Interactions)\nPageRank Analysis - Top {top_nodes} nodes highlighted in red", 
                 fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_pagerank_distribution(self, pagerank_scores: Dict[str, float]):
        """Plot distribution of PageRank scores"""
        plt.figure(figsize=(12, 6))
        
        scores = list(pagerank_scores.values())
        plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('PageRank Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of PageRank Scores in Ovarian Cancer Network (100 Interactions)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_convergence(self):
        """Plot convergence history of PageRank algorithm"""
        if hasattr(self, 'convergence_history'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.convergence_history, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Total Change')
            plt.title('PageRank Algorithm Convergence')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.show()
    
    def plot_top_nodes_comparison(self, custom_scores: Dict[str, float], nx_scores: Dict[str, float], top_n: int = 15):
        """Compare top nodes between custom and NetworkX PageRank"""
        custom_top = self.identify_key_players(custom_scores, top_n)
        nx_top = self.identify_key_players(nx_scores, top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Custom PageRank
        nodes_custom = [node for node, _ in custom_top]
        scores_custom = [score for _, score in custom_top]
        ax1.barh(nodes_custom, scores_custom, color='lightblue')
        ax1.set_xlabel('PageRank Score')
        ax1.set_title('Custom PageRank - Top Nodes')
        ax1.grid(True, alpha=0.3)
        
        # NetworkX PageRank
        nodes_nx = [node for node, _ in nx_top]
        scores_nx = [score for _, score in nx_top]
        ax2.barh(nodes_nx, scores_nx, color='lightgreen')
        ax2.set_xlabel('PageRank Score')
        ax2.set_title('NetworkX PageRank - Top Nodes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self, pagerank_scores: Dict[str, float]) -> Dict:
        """Generate comprehensive analysis report"""
        key_players = self.identify_key_players(pagerank_scores)
        analysis_df = self.analyze_node_roles(pagerank_scores)
        
        # Calculate node type statistics - FIXED VERSION
        node_type_stats = {}
        for node_type in analysis_df['Node_Type'].unique():
            type_data = analysis_df[analysis_df['Node_Type'] == node_type]
            node_type_stats[node_type] = {
                'count': len(type_data),
                'mean_pageRank': type_data['PageRank'].mean(),
                'std_pageRank': type_data['PageRank'].std(),
                'max_pageRank': type_data['PageRank'].max()
            }
        
        # Calculate network statistics
        report = {
            'network_stats': {
                'total_nodes': self.G.number_of_nodes(),
                'total_edges': self.G.number_of_edges(),
                'network_density': nx.density(self.G),
                'average_degree': np.mean([d for n, d in self.G.degree()]),
                'is_strongly_connected': nx.is_strongly_connected(self.G),
                'is_weakly_connected': nx.is_weakly_connected(self.G),
                'number_strongly_connected_components': nx.number_strongly_connected_components(self.G),
                'number_weakly_connected_components': nx.number_weakly_connected_components(self.G)
            },
            'top_key_players': key_players,
            'node_type_analysis': node_type_stats,
            'correlation_analysis': {
                'pagerank_in_degree_corr': analysis_df['PageRank'].corr(analysis_df['In_Degree']),
                'pagerank_out_degree_corr': analysis_df['PageRank'].corr(analysis_df['Out_Degree']),
                'pagerank_total_degree_corr': analysis_df['PageRank'].corr(analysis_df['Total_Degree'])
            }
        }
        
        return report

def main():
    """Main function to run PageRank analysis on ovarian cancer network"""
    
    # Define 100 interactions from the ovarian cancer network
    interactions = [
        # DNA Damage Response Pathway (1-10)
        ('IR', 'DSB', 'interacts_with'),
        ('DSB', 'DNA', 'interacts_with'),
        ('DNA', 'DSB', 'interacts_with'),
        ('DNA', 'p53mRNA', 'interacts_with'),
        ('p53mRNA', 'p53', 'interacts_with'),
        ('p53', 'p53-p', 'interacts_with'),
        ('p53-p', 'p53', 'interacts_with'),
        ('DNA', 'PTENmRNA nuc', 'interacts_with'),
        ('PTENmRNA nuc', 'PTEN cyt', 'interacts_with'),
        ('PTEN cyt', 'PIP2', 'interacts_with'),

        # PI3K/AKT/mTOR Pathway (11-20)
        ('PIP3', 'PIP2', 'interacts_with'),
        ('PIP2', 'PIP3', 'interacts_with'),
        ('DNA', 'Mdm2mRNA nuc', 'interacts_with'),
        ('Mdm2mRNA nuc', 'Mdm2 cyt', 'interacts_with'),
        ('Mdm2-p cyt', 'Mdm2 cyt', 'interacts_with'),
        ('Mdm2 cyt', 'Mdm2-p cyt', 'interacts_with'),
        ('AKT-p cyt', 'AKT cyt', 'interacts_with'),
        ('PIP3', 'AKT-p cyt', 'interacts_with'),
        ('AKT-p cyt', 'Mdm2-p cyt', 'interacts_with'),
        ('AKT cyt', 'AKT-p cyt', 'interacts_with'),

        # MDM2-p53 Feedback Loop (21-30)
        ('Mdm2 cyt', 'Mdm2-p nuc', 'interacts_with'),
        ('Mdm2-p cyt', 'Mdm2-p nuc', 'interacts_with'),
        ('p53-p', 'Bax mRNA', 'interacts_with'),
        ('p53-p', 'p21 mRNA', 'interacts_with'),
        ('p53-p', 'PTENmRNA nuc', 'interacts_with'),
        ('p53-p', 'IkBα', 'interacts_with'),
        ('p53-p', 'A20 mRNA', 'interacts_with'),
        ('p53-p', 'Mdm2mRNA nuc', 'interacts_with'),
        ('Mdm2-p nuc', 'p53', 'interacts_with'),
        ('Mdm2-p-p nuc', 'Mdm2-p nuc', 'interacts_with'),

        # Apoptosis Regulation (31-40)
        ('Mdm2-p nuc', 'Mdm2-p-p nuc', 'interacts_with'),
        ('Mdm2-p nuc', 'p53', 'interacts_with'),
        ('p53', 'DSB', 'interacts_with'),
        ('p53', 'DNA', 'interacts_with'),
        ('DNA', 'Bax mRNA nuc', 'interacts_with'),
        ('DNA', 'p21 mRNA nuc', 'interacts_with'),
        ('p21 mRNA nuc', 'Bax mRNA nuc', 'interacts_with'),
        ('p21', 'p21 mRNA nuc', 'interacts_with'),
        ('p21 mRNA nuc', 'cell cycle arrest', 'interacts_with'),
        ('Bax', 'p21', 'interacts_with'),

        # Cell Death Pathways (41-50)
        ('Bax mRNA nuc', 'Bax', 'interacts_with'),
        ('Bax', 'apoptosis', 'interacts_with'),
        ('DNA', 'A20 mRNA nuc', 'interacts_with'),
        ('DNA', 'IkBα mRNA nuc', 'interacts_with'),
        ('IkBα mRNA nuc', 'IkBα nuc', 'interacts_with'),
        ('A20 mRNA nuc', 'A20 cyt', 'interacts_with'),
        ('A20 cyt', 'IKKKa cyt', 'interacts_with'),
        ('A20 cyt', 'IKKi cyt', 'interacts_with'),
        ('A20 cyt', 'IKKa cyt', 'interacts_with'),
        ('IKKi cyt', 'IKKii cyt', 'interacts_with'),

        # NF-κB Signaling (51-60)
        ('A20 cyt', 'IKKKa cyt', 'interacts_with'),
        ('IKKKa cyt', 'IKKKn cyt', 'interacts_with'),
        ('IKKKn cyt', 'IKKKa cyt', 'interacts_with'),
        ('IKKKa cyt', 'IKKa cyt', 'interacts_with'),
        ('IKKa cyt', 'IKKi cyt', 'interacts_with'),
        ('IKKii cyt', 'IKKKn cyt', 'interacts_with'),
        ('IKKn cyt', 'IKKa cyt', 'interacts_with'),
        ('IKKa cyt', 'IkBα:NFkB cyt', 'interacts_with'),
        ('IkBα:NFkB cyt', 'NFkB cyt', 'interacts_with'),
        ('IkBα:NFkB cyt', 'IkBα * cyt', 'interacts_with'),

        # NF-κB Nuclear Transport (61-70)
        ('NFkB cyt', 'IkBα:NFkB cyt', 'interacts_with'),
        ('IkBα cyt', 'IkBα:NFkB cyt', 'interacts_with'),
        ('IkBα cyt', 'IkBα nuc', 'interacts_with'),
        ('NFkB cyt', 'NFkB nuc', 'interacts_with'),
        ('NFkB nuc', 'IkBα mRNA nuc', 'interacts_with'),
        ('NFkB nuc', 'IkBα:NFkB nuc', 'interacts_with'),
        ('IkBα nuc', 'IkBα:NFkB nuc', 'interacts_with'),
        ('IkBα:NFkB nuc', 'IkBα:NFkB cyt', 'interacts_with'),
        ('IkBα mRNA', 'IkBα', 'interacts_with'),
        ('NFkB nuc', 'A20 mRNA nuc', 'interacts_with'),

        # miRNA and Post-transcriptional Regulation (71-80)
        ('NFkB nuc', 'p53mRNA nuc', 'interacts_with'),
        ('DNA', 'pre-mRNA-16 nuc', 'interacts_with'),
        ('DNA', 'Wip1 mRNA nuc', 'interacts_with'),
        ('pre-mRNA-16 nuc', 'miR-16 nuc', 'interacts_with'),
        ('miR-16 nuc', 'Wip1 mRNA nuc', 'interacts_with'),
        ('Wip1 mRNA nuc', 'Wip1 nuc', 'interacts_with'),
        ('Wip1 nuc', 'p53', 'interacts_with'),
        ('Wip1 nuc', 'Wip1 mRNA', 'interacts_with'),
        ('Wip1 nuc', 'A20 mRNA', 'interacts_with'),
        ('Wip1 nuc', 'IkBα mRNA', 'interacts_with'),

        # DNA Damage Checkpoints (81-90)
        ('Wip1 nuc', 'Mdm2-p nuc', 'interacts_with'),
        ('Wip1 nuc', 'ChK2 nuc', 'interacts_with'),
        ('Wip1 mRNA nuc', 'ChK2 mRNA', 'interacts_with'),
        ('Wip1 nuc', 'ATM nuc', 'interacts_with'),
        ('Wip1 nuc', 'ATM-p nuc', 'interacts_with'),
        ('KSRP cyt', 'KSRP-p cyt', 'interacts_with'),
        ('KSRP-p cyt', 'KSRP cyt', 'interacts_with'),
        ('KSRP-p cyt', 'KSRP-p nuc', 'interacts_with'),
        ('KSRP-p nuc', 'pre-mRNA-16 nuc', 'interacts_with'),
        ('DNA', 'ChK2 mRNA nuc', 'interacts_with'),

        # ATM/ATR Signaling (91-100)
        ('ChK2 mRNA nuc', 'ChK2 nuc', 'interacts_with'),
        ('ChK2 nuc', 'ChK2-p nuc', 'interacts_with'),
        ('ChK2-p nuc', 'Chk2 nuc', 'interacts_with'),
        ('ChK2-p nuc', 'p53-p', 'interacts_with'),
        ('ChK2-p nuc', 'Mdm2-p-p nuc', 'interacts_with'),
        ('ChK2-p nuc', 'Mdm2-p nuc', 'interacts_with'),
        ('ChK2-p nuc', 'Mdm2 cyt', 'interacts_with'),
        ('ChK2-p nuc', 'Mdm2-p cyt', 'interacts_with'),
        ('DNA', 'ATM mRNA nuc', 'interacts_with'),
        ('ATM mRNA nuc', 'ATM nuc', 'interacts_with')
    ]
    
    # Verify we have exactly 100 interactions
    logger.info(f"Total interactions: {len(interactions)}")
    
    # Initialize PageRank analyzer
    pr_analyzer = OvarianCancerPageRank(damping_factor=0.85, max_iter=100)
    
    # Build network
    network = pr_analyzer.build_network_from_data(interactions)
    
    # Run custom PageRank
    custom_pagerank_scores = pr_analyzer.custom_pagerank()
    
    # Run NetworkX PageRank for comparison
    nx_pagerank_scores = pr_analyzer.networkx_pagerank()
    
    # Identify key players
    key_players_custom = pr_analyzer.identify_key_players(custom_pagerank_scores, top_k=20)
    key_players_nx = pr_analyzer.identify_key_players(nx_pagerank_scores, top_k=20)
    
    # Generate reports
    custom_report = pr_analyzer.generate_comprehensive_report(custom_pagerank_scores)
    nx_report = pr_analyzer.generate_comprehensive_report(nx_pagerank_scores)
    
    # Display results
    print("=" * 100)
    print("OVARIAN CANCER NETWORK PAGERANK ANALYSIS - 100 INTERACTIONS")
    print("=" * 100)
    
    print(f"\nNETWORK STATISTICS:")
    stats = custom_report['network_stats']
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print(f"Network Density: {stats['network_density']:.4f}")
    print(f"Average Degree: {stats['average_degree']:.2f}")
    print(f"Strongly Connected: {stats['is_strongly_connected']}")
    print(f"Weakly Connected: {stats['is_weakly_connected']}")
    print(f"Strongly Connected Components: {stats['number_strongly_connected_components']}")
    print(f"Weakly Connected Components: {stats['number_weakly_connected_components']}")
    
    print(f"\nTOP 20 KEY PLAYERS (CUSTOM PAGERANK):")
    for i, (node, score) in enumerate(key_players_custom, 1):
        node_type = network.nodes[node].get('node_type', 'Unknown')
        in_deg = network.in_degree(node)
        out_deg = network.out_degree(node)
        print(f"{i:2d}. {node:25s} (Score: {score:.6f}, Type: {node_type:20s}, In: {in_deg:2d}, Out: {out_deg:2d})")
    
    print(f"\nTOP 20 KEY PLAYERS (NETWORKX PAGERANK):")
    for i, (node, score) in enumerate(key_players_nx, 1):
        node_type = network.nodes[node].get('node_type', 'Unknown')
        in_deg = network.in_degree(node)
        out_deg = network.out_degree(node)
        print(f"{i:2d}. {node:25s} (Score: {score:.6f}, Type: {node_type:20s}, In: {in_deg:2d}, Out: {out_deg:2d})")
    
    print(f"\nCORRELATION ANALYSIS:")
    corr = custom_report['correlation_analysis']
    print(f"PageRank vs In-Degree Correlation: {corr['pagerank_in_degree_corr']:.4f}")
    print(f"PageRank vs Out-Degree Correlation: {corr['pagerank_out_degree_corr']:.4f}")
    print(f"PageRank vs Total Degree Correlation: {corr['pagerank_total_degree_corr']:.4f}")
    
    print(f"\nNODE TYPE ANALYSIS:")
    for node_type, metrics in custom_report['node_type_analysis'].items():
        count = metrics['count']
        mean_score = metrics['mean_pageRank']
        print(f"{node_type:25s}: {count:2d} nodes, Average PageRank: {mean_score:.6f}")
    
    # Visualizations
    pr_analyzer.visualize_network(custom_pagerank_scores)
    pr_analyzer.plot_pagerank_distribution(custom_pagerank_scores)
    pr_analyzer.plot_convergence()
    pr_analyzer.plot_top_nodes_comparison(custom_pagerank_scores, nx_pagerank_scores)
    
    return pr_analyzer, custom_pagerank_scores, custom_report

if __name__ == "__main__":
    analyzer, pagerank_scores, report = main()
