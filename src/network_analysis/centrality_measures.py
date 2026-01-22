import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
import warnings
import os
import json
import codecs
warnings.filterwarnings('ignore')

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ovarian_cancer_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OvarianCancerNetworkAnalyzer:
    """Comprehensive network analysis for ovarian cancer signaling network"""
    
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol
        self.G = nx.DiGraph()
        self.centrality_measures = {}
        self.analysis_results = {}
        
    def build_network_from_data(self, interactions_data: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """Build directed graph from interaction data with enhanced node properties"""
        logger.info("Building ovarian cancer network from interaction data...")
        
        # Clear existing graph
        self.G = nx.DiGraph()
        
        # First pass: add all nodes with properties
        unique_nodes = set()
        for source, target, _ in interactions_data:
            unique_nodes.update([source, target])
        
        for node in unique_nodes:
            node_type = self._classify_node_type(node)
            self.G.add_node(node, 
                          node_type=node_type,
                          biological_role=self._determine_biological_role(node),
                          is_hub=False)
        
        # Second pass: add edges with enhanced properties
        edge_types_count = defaultdict(int)
        for source, target, interaction_type in interactions_data:
            weight = self._calculate_edge_weight(interaction_type)
            edge_type = self._classify_edge_type(interaction_type)
            
            self.G.add_edge(source, target, 
                          weight=weight,
                          interaction_type=interaction_type,
                          edge_type=edge_type)
            edge_types_count[edge_type] += 1
        
        logger.info(f"Network built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        # Count node types
        node_types = [self.G.nodes[node]['node_type'] for node in self.G.nodes()]
        logger.info(f"Node types: {pd.Series(node_types).value_counts().to_dict()}")
        
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
        elif any(x in node_lower for x in ['pip2', 'pip3']):
            return 'Metabolite'
        else:
            return 'Other'
    
    def _determine_biological_role(self, node_name: str) -> str:
        """Determine the biological role of a node"""
        node_lower = node_name.lower()
        
        if any(x in node_lower for x in ['p53', 'tp53']):
            return 'Tumor_Suppressor'
        elif any(x in node_lower for x in ['mdm2', 'akt', 'nfkb']):
            return 'Oncogene'
        elif any(x in node_lower for x in ['atm', 'chk2', 'atr']):
            return 'DNA_Repair'
        elif any(x in node_lower for x in ['bax', 'apoptosis']):
            return 'Apoptosis_Regulator'
        elif any(x in node_lower for x in ['p21', 'arrest']):
            return 'Cell_Cycle_Regulator'
        elif any(x in node_lower for x in ['mirna', 'microrna']):
            return 'Post_Transcriptional_Regulator'
        
        return 'Regulator'
    
    def _classify_edge_type(self, interaction_type: str) -> str:
        """Classify edges into broader categories"""
        edge_type_mapping = {
            'interacts_with': 'Interaction',
            'controls-expression': 'Transcriptional_Regulation',
            'controls-phosphorylation': 'Post_Translational_Modification',
            'in-complex-with': 'Complex_Formation',
            'controls-state-change': 'State_Change',
            'controls-transport': 'Transport'
        }
        return edge_type_mapping.get(interaction_type, 'Other')
    
    def _calculate_edge_weight(self, interaction_type: str) -> float:
        """Assign weights based on interaction type confidence"""
        weight_mapping = {
            'controls-expression': 0.95,
            'controls-phosphorylation': 0.90,
            'controls-state-change': 0.85,
            'interacts_with': 0.80,
            'in-complex-with': 0.75,
            'controls-transport': 0.70
        }
        return weight_mapping.get(interaction_type, 0.60)
    
    def compute_all_centrality_measures(self):
        """Compute multiple centrality measures for comprehensive analysis"""
        logger.info("Computing centrality measures...")
        
        # Compute standard centrality measures
        self.centrality_measures = {
            'degree_centrality': nx.degree_centrality(self.G),
            'in_degree_centrality': nx.in_degree_centrality(self.G),
            'out_degree_centrality': nx.out_degree_centrality(self.G),
            'betweenness_centrality': nx.betweenness_centrality(self.G, normalized=True),
            'closeness_centrality': nx.closeness_centrality(self.G),
            'pagerank': nx.pagerank(self.G, alpha=self.damping_factor, weight='weight')
        }
        
        # Compute eigenvector centrality with fallback
        try:
            self.centrality_measures['eigenvector_centrality'] = nx.eigenvector_centrality(
                self.G, max_iter=1000, tol=1e-6)
        except:
            logger.warning("Eigenvector centrality computation failed, using PageRank as fallback")
            self.centrality_measures['eigenvector_centrality'] = self.centrality_measures['pagerank']
        
        # Compute custom PageRank
        self.centrality_measures['custom_pagerank'] = self.custom_pagerank()
        
        logger.info("Centrality measures computed successfully")
        return self.centrality_measures
    
    def custom_pagerank(self, personalization: Dict[str, float] = None) -> Dict[str, float]:
        """Implement custom PageRank algorithm with biological considerations"""
        logger.info("Running custom PageRank algorithm...")
        
        nodes = list(self.G.nodes())
        n = len(nodes)
        if n == 0:
            return {}
        
        # Initialize PageRank scores
        pagerank = {node: 1.0 / n for node in nodes}
        
        # Create biological personalization
        if personalization is None:
            personalization = self._create_biological_personalization()
        
        # Normalize personalization vector
        personalization_sum = sum(personalization.values())
        if personalization_sum > 0:
            personalization = {k: v / personalization_sum for k, v in personalization.items()}
        else:
            personalization = {node: 1.0 / n for node in nodes}
        
        convergence_history = []
        
        # Precompute weighted out-degrees for efficiency
        weighted_out_degrees = {}
        for node in nodes:
            total = sum(self.G[node][succ].get('weight', 1.0) 
                       for succ in self.G.successors(node))
            weighted_out_degrees[node] = total if total > 0 else 1.0
        
        for iteration in range(self.max_iter):
            new_pagerank = {}
            total_change = 0
            
            for node in nodes:
                incoming_score = 0.0
                for predecessor in self.G.predecessors(node):
                    edge_weight = self.G[predecessor][node].get('weight', 1.0)
                    out_degree = weighted_out_degrees[predecessor]
                    incoming_score += pagerank[predecessor] * (edge_weight / out_degree)
                
                # Apply damping factor and personalization
                new_score = (1 - self.damping_factor) * personalization.get(node, 0) + \
                           self.damping_factor * incoming_score
                
                new_pagerank[node] = new_score
                total_change += abs(new_score - pagerank[node])
            
            convergence_history.append(total_change)
            
            # Check convergence
            if total_change < self.tol:
                logger.info(f"Custom PageRank converged after {iteration + 1} iterations")
                break
            
            pagerank = new_pagerank
        
        # Store convergence history
        self.convergence_history = convergence_history
        
        # Normalize final scores
        total_score = sum(pagerank.values())
        if total_score > 0:
            pagerank = {k: v / total_score for k, v in pagerank.items()}
        
        return pagerank
    
    def _create_biological_personalization(self) -> Dict[str, float]:
        """Create personalization vector based on biological importance"""
        personalization = {}
        
        importance_scores = {
            'Tumor_Suppressor': 3.0,
            'Oncogene': 2.5,
            'DNA_Repair': 2.0,
            'Apoptosis_Regulator': 2.0,
            'Cell_Cycle_Regulator': 1.8,
            'Post_Transcriptional_Regulator': 1.5,
            'Regulator': 1.2
        }
        
        for node in self.G.nodes():
            biological_role = self.G.nodes[node].get('biological_role', 'Regulator')
            node_type = self.G.nodes[node].get('node_type', 'Other')
            
            base_score = importance_scores.get(biological_role, 1.0)
            
            # Adjust based on node type
            type_multiplier = {
                'Phenotype': 1.5,
                'Protein': 1.3,
                'RNA': 1.2,
                'Cellular_Component': 1.1,
                'Stimulus': 1.0,
                'Metabolite': 0.9,
                'Other': 0.8
            }
            
            score = base_score * type_multiplier.get(node_type, 1.0)
            
            # Extra importance for known key players
            if any(x in node.lower() for x in ['p53', 'nfkb', 'akt', 'bax', 'atm']):
                score *= 1.5
            
            personalization[node] = score
        
        return personalization
    
    def identify_hub_nodes(self, threshold_percentile: float = 80.0) -> List[Tuple[str, float]]:
        """Identify hub nodes based on degree centrality"""
        degree_centrality = self.centrality_measures.get('degree_centrality', nx.degree_centrality(self.G))
        
        # Calculate threshold
        values = list(degree_centrality.values())
        if len(values) == 0:
            return []
        
        threshold = np.percentile(values, threshold_percentile)
        
        hub_nodes = [(node, score) for node, score in degree_centrality.items() if score >= threshold]
        hub_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Update node attributes
        for node, _ in hub_nodes:
            self.G.nodes[node]['is_hub'] = True
            self.G.nodes[node]['degree_centrality'] = degree_centrality[node]
        
        logger.info(f"Identified {len(hub_nodes)} hub nodes (top {threshold_percentile}% by degree centrality)")
        return hub_nodes
    
    def analyze_network_topology(self) -> Dict[str, Any]:
        """Perform comprehensive network topology analysis"""
        logger.info("Analyzing network topology...")
        
        # Basic network statistics
        topology_stats = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'network_density': nx.density(self.G),
            'average_degree': np.mean([d for _, d in self.G.degree()]),
            'degree_assortativity': nx.degree_assortativity_coefficient(self.G),
            'transitivity': nx.transitivity(self.G),
            'average_clustering': nx.average_clustering(self.G.to_undirected()),
        }
        
        # Connectivity analysis
        topology_stats['is_strongly_connected'] = nx.is_strongly_connected(self.G)
        topology_stats['is_weakly_connected'] = nx.is_weakly_connected(self.G)
        topology_stats['strongly_connected_components'] = nx.number_strongly_connected_components(self.G)
        topology_stats['weakly_connected_components'] = nx.number_weakly_connected_components(self.G)
        
        # Path analysis
        if nx.is_weakly_connected(self.G):
            undirected_G = self.G.to_undirected()
            topology_stats['average_shortest_path_length'] = nx.average_shortest_path_length(undirected_G)
            topology_stats['diameter'] = nx.diameter(undirected_G)
        else:
            topology_stats['average_shortest_path_length'] = None
            topology_stats['diameter'] = None
        
        # Degree distribution analysis
        degrees = [d for _, d in self.G.degree()]
        if len(degrees) > 0:
            topology_stats['degree_distribution'] = {
                'min': float(np.min(degrees)),
                'max': float(np.max(degrees)),
                'mean': float(np.mean(degrees)),
                'median': float(np.median(degrees)),
                'std': float(np.std(degrees)),
                'skewness': float(scipy_stats.skew(degrees)) if len(degrees) > 2 else 0.0,
                'kurtosis': float(scipy_stats.kurtosis(degrees)) if len(degrees) > 3 else 0.0
            }
        
        # Check for scale-free properties
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        topology_stats['scale_free_r_squared'] = self._fit_power_law(degree_counts)
        
        self.analysis_results['topology'] = topology_stats
        return topology_stats
    
    def _fit_power_law(self, degree_counts):
        """Fit power law to degree distribution"""
        try:
            x = np.log(degree_counts.index.values.astype(float))
            y = np.log(degree_counts.values.astype(float))
            
            mask = (x > -np.inf) & (y > -np.inf) & np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 2:
                return None
            
            slope, intercept, r_value, _, _ = scipy_stats.linregress(x[mask], y[mask])
            return float(r_value ** 2)
        except Exception as e:
            logger.warning(f"Power law fitting failed: {e}")
            return None
    
    def perform_statistical_analysis(self) -> pd.DataFrame:
        """Perform statistical analysis on centrality measures"""
        logger.info("Performing statistical analysis...")
        
        # Create comprehensive DataFrame
        analysis_data = []
        
        for node in self.G.nodes():
            row = {
                'Node': node,
                'Node_Type': self.G.nodes[node].get('node_type', 'Unknown'),
                'Biological_Role': self.G.nodes[node].get('biological_role', 'Unknown'),
                'In_Degree': self.G.in_degree(node),
                'Out_Degree': self.G.out_degree(node),
                'Total_Degree': self.G.degree(node),
                'Is_Hub': self.G.nodes[node].get('is_hub', False)
            }
            
            # Add all centrality measures
            for measure_name, measure_dict in self.centrality_measures.items():
                row[f'{measure_name}_score'] = measure_dict.get(node, 0)
            
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        
        # Calculate correlations
        centrality_cols = [col for col in df.columns if '_score' in col]
        degree_cols = ['In_Degree', 'Out_Degree', 'Total_Degree']
        
        if len(df) > 1 and len(centrality_cols) > 0:
            correlation_matrix = df[centrality_cols + degree_cols].corr()
        else:
            all_cols = centrality_cols + degree_cols
            correlation_matrix = pd.DataFrame(1, index=all_cols, columns=all_cols)
        
        self.analysis_results['statistical_analysis'] = {
            'dataframe': df,
            'correlation_matrix': correlation_matrix,
            'summary_stats': df.describe()
        }
        
        return df
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report with all results"""
        logger.info("Generating comprehensive report...")
        
        if not self.centrality_measures:
            self.compute_all_centrality_measures()
        
        if 'topology' not in self.analysis_results:
            self.analyze_network_topology()
        
        if 'statistical_analysis' not in self.analysis_results:
            self.perform_statistical_analysis()
        
        df = self.analysis_results['statistical_analysis']['dataframe']
        
        # Identify top key players across different measures
        top_players = {}
        for measure in ['custom_pagerank', 'betweenness_centrality', 'eigenvector_centrality', 'degree_centrality']:
            col_name = f'{measure}_score'
            if col_name in df.columns and len(df) > 0:
                top_df = df.nlargest(10, col_name)[['Node', col_name, 'Node_Type']]
                top_players[measure] = top_df.to_dict('records')
        
        # Node type analysis
        node_type_stats = {}
        for node_type in df['Node_Type'].unique():
            type_data = df[df['Node_Type'] == node_type]
            node_type_stats[node_type] = {
                'count': len(type_data),
                'avg_degree': float(type_data['Total_Degree'].mean()),
                'avg_pagerank': float(type_data['custom_pagerank_score'].mean()),
                'hub_count': int(type_data['Is_Hub'].sum())
            }
        
        # Hub analysis
        hub_nodes = df[df['Is_Hub'] == True]
        hub_analysis = {
            'total_hubs': len(hub_nodes),
            'hub_by_type': hub_nodes['Node_Type'].value_counts().to_dict(),
            'top_hubs': hub_nodes.nlargest(10, 'Total_Degree')[['Node', 'Total_Degree', 'Node_Type']].to_dict('records')
        }
        
        report = {
            'network_topology': self.analysis_results['topology'],
            'top_key_players': top_players,
            'hub_analysis': hub_analysis,
            'node_type_analysis': node_type_stats,
            'correlation_matrix': self.analysis_results['statistical_analysis']['correlation_matrix'].to_dict(),
            'degree_distribution': self.analysis_results['topology']['degree_distribution']
        }
        
        # Add implications based on analysis
        report['therapeutic_implications'] = self._generate_therapeutic_implications(df)
        
        self.analysis_results['comprehensive_report'] = report
        return report
    
    def _generate_therapeutic_implications(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate therapeutic implications based on network analysis"""
        implications = {
            'critical_targets': [],
            'network_vulnerabilities': [],
            'combination_therapy_candidates': []
        }
        
        if len(df) == 0:
            return implications
        
        # Identify critical targets (high centrality across multiple measures)
        centrality_cols = [col for col in df.columns if '_score' in col]
        if len(centrality_cols) > 0:
            df['centrality_composite'] = df[centrality_cols].mean(axis=1)
            critical_targets = df.nlargest(10, 'centrality_composite')
            for _, row in critical_targets.iterrows():
                implications['critical_targets'].append({
                    'node': row['Node'],
                    'node_type': row['Node_Type'],
                    'biological_role': row['Biological_Role'],
                    'composite_score': float(row['centrality_composite']),
                    'rationale': f"High centrality across {len(centrality_cols)} measures indicates key regulatory role"
                })
        
        # Identify network vulnerabilities (high betweenness nodes)
        if 'betweenness_centrality_score' in df.columns:
            vulnerable_nodes = df.nlargest(5, 'betweenness_centrality_score')
            for _, row in vulnerable_nodes.iterrows():
                implications['network_vulnerabilities'].append({
                    'node': row['Node'],
                    'betweenness': float(row['betweenness_centrality_score']),
                    'rationale': "High betweenness indicates critical bridge in network - disruption could fragment signaling"
                })
        
        # Identify combination therapy candidates
        hub_nodes = df[df['Is_Hub'] == True]
        protein_hubs = hub_nodes[hub_nodes['Node_Type'] == 'Protein']
        rna_hubs = hub_nodes[hub_nodes['Node_Type'] == 'RNA']
        
        if len(protein_hubs) >= 2 and len(rna_hubs) >= 1:
            implications['combination_therapy_candidates'] = [{
                'protein_target': protein_hubs.iloc[0]['Node'],
                'rna_target': rna_hubs.iloc[0]['Node'],
                'rationale': "Targeting both protein hub and RNA regulator could provide synergistic effect"
            }]
        
        return implications
    
    def visualize_comprehensive_analysis(self, output_dir: str = "./visualizations"):
        """Generate all visualizations for comprehensive analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Generating visualizations in {output_dir}...")
        
        # 1. Degree Distribution
        self.plot_degree_distribution(save_path=f"{output_dir}/degree_distribution.png")
        
        # 2. Network Visualization
        self.visualize_network_with_centrality(save_path=f"{output_dir}/network_visualization.png")
        
        # 3. Centrality Comparison
        self.plot_centrality_comparison(save_path=f"{output_dir}/centrality_comparison.png")
        
        # 4. PageRank Distribution
        self.plot_pagerank_distribution(save_path=f"{output_dir}/pagerank_distribution.png")
        
        # 5. Correlation Heatmap
        self.plot_correlation_heatmap(save_path=f"{output_dir}/correlation_heatmap.png")
        
        # 6. Node Type Analysis
        self.plot_node_type_analysis(save_path=f"{output_dir}/node_type_analysis.png")
        
        # 7. Convergence Plot
        if hasattr(self, 'convergence_history'):
            self.plot_convergence(save_path=f"{output_dir}/convergence_plot.png")
        
        logger.info("All visualizations generated successfully")
    
    def plot_degree_distribution(self, save_path: Optional[str] = None):
        """Plot degree distribution as requested"""
        plt.figure(figsize=(15, 8))
        
        degrees = [self.G.degree(node) for node in self.G.nodes()]
        nodes = list(self.G.nodes())
        
        # Sort by degree
        sorted_indices = np.argsort(degrees)[::-1]
        sorted_nodes = [nodes[i] for i in sorted_indices]
        sorted_degrees = [degrees[i] for i in sorted_indices]
        
        # Create bar plot
        bars = plt.bar(range(len(sorted_nodes)), sorted_degrees, color='steelblue', alpha=0.8)
        
        # Highlight top 20% as hubs
        hub_threshold = np.percentile(degrees, 80)
        hub_indices = []
        for i, (node, degree) in enumerate(zip(sorted_nodes, sorted_degrees)):
            if degree >= hub_threshold:
                bars[i].set_color('firebrick')
                bars[i].set_alpha(1.0)
                hub_indices.append(i)
        
        plt.xlabel('Node Index (Sorted by Degree)', fontsize=12)
        plt.ylabel('Degree (Number of Connections)', fontsize=12)
        plt.title('Node Degree Distribution of Ovarian Cancer Signaling Network\n'
                 f'Network Density: {nx.density(self.G):.4f}, Average Degree: {np.mean(degrees):.2f}',
                 fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add legend
        hub_patch = Patch(color='firebrick', alpha=1.0, label=f'Hub Nodes (Top 20%, Degree ≥ {int(hub_threshold)})')
        normal_patch = Patch(color='steelblue', alpha=0.8, label='Non-Hub Nodes')
        plt.legend(handles=[hub_patch, normal_patch], loc='upper right')
        
        # Add statistics box
        stats_text = f'Total Nodes: {len(nodes)}\nTotal Edges: {self.G.number_of_edges()}\n'
        stats_text += f'Min Degree: {np.min(degrees)}\nMax Degree: {np.max(degrees)}\n'
        stats_text += f'Mean Degree: {np.mean(degrees):.2f}\nMedian Degree: {np.median(degrees)}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_network_with_centrality(self, save_path: Optional[str] = None):
        """Enhanced network visualization with centrality-based sizing"""
        plt.figure(figsize=(20, 16))
        
        # Use spring layout
        try:
            pos = nx.spring_layout(self.G, k=2, iterations=100, seed=42)
        except:
            pos = nx.circular_layout(self.G)
        
        # Get PageRank scores for sizing
        if 'custom_pagerank' in self.centrality_measures:
            pagerank_scores = self.centrality_measures['custom_pagerank']
        else:
            pagerank_scores = nx.pagerank(self.G)
        
        # Prepare node sizes based on PageRank (log scale for better visualization)
        node_sizes = []
        for node in self.G.nodes():
            score = pagerank_scores.get(node, 0.0001)
            size = np.log(score * 100000 + 100) * 200
            node_sizes.append(max(size, 50))  # Minimum size
        
        # Get top nodes for highlighting (top 10 by PageRank)
        top_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        top_node_names = [node for node, _ in top_nodes]
        
        # Color nodes based on type and hub status
        node_colors = []
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('node_type', 'Other')
            
            # Base colors by node type
            color_map = {
                'Protein': 'lightcoral',
                'RNA': 'lightgreen',
                'Cellular_Component': 'gold',
                'Phenotype': 'darkorange',
                'Stimulus': 'violet',
                'Metabolite': 'lightblue',
                'Other': 'lightgray'
            }
            
            base_color = color_map.get(node_type, 'lightgray')
            
            # If node is a hub, darken the color
            if self.G.nodes[node].get('is_hub', False):
                # Convert to RGB and darken
                import matplotlib.colors as mcolors
                rgb = mcolors.to_rgb(base_color)
                darkened = tuple([max(0, c * 0.7) for c in rgb])
                node_colors.append(darkened)
            else:
                node_colors.append(base_color)
        
        # Draw the network
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes,
                              node_color=node_colors, alpha=0.9,
                              edgecolors='black', linewidths=1)
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, arrowstyle='->',
                              arrowsize=15, edge_color='gray', width=1)
        
        # Label top nodes only
        labels = {node: node for node in top_node_names}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=9,
                               font_weight='bold')
        
        # Create custom legend
        legend_elements = []
        for node_type, color in {
            'Protein': 'lightcoral',
            'RNA': 'lightgreen',
            'Cellular_Component': 'gold',
            'Phenotype': 'darkorange',
            'Hub Node (darker)': 'darkred'
        }.items():
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=node_type))
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        # Add title with key statistics
        hub_count = sum(1 for node in self.G.nodes() if self.G.nodes[node].get('is_hub', False))
        plt.title('Ovarian Cancer Signaling Network with PageRank Centrality\n'
                 f'Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()}, Hubs: {hub_count}\n'
                 'Node Size ∝ PageRank Score | Top 10 Nodes Labeled',
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_centrality_comparison(self, save_path: Optional[str] = None):
        """Compare different centrality measures"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        centrality_types = [
            ('custom_pagerank', 'Custom PageRank', 'lightblue'),
            ('betweenness_centrality', 'Betweenness Centrality', 'lightgreen'),
            ('degree_centrality', 'Degree Centrality', 'lightcoral'),
            ('eigenvector_centrality', 'Eigenvector Centrality', 'gold')
        ]
        
        for idx, (measure, title, color) in enumerate(centrality_types):
            if measure in self.centrality_measures:
                scores = self.centrality_measures[measure]
                top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                
                nodes = [node for node, _ in top_nodes]
                values = [score for _, score in top_nodes]
                
                bars = axes[idx].barh(nodes, values, color=color, alpha=0.8)
                axes[idx].set_xlabel('Score', fontsize=11)
                axes[idx].set_title(title, fontsize=13, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='x')
                axes[idx].invert_yaxis()  # Highest at top
                
                # Add value labels
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    axes[idx].text(width * 1.01, bar.get_y() + bar.get_height()/2,
                                 f'{value:.4f}', va='center', fontsize=9)
        
        plt.suptitle('Comparison of Centrality Measures - Top 10 Nodes for Each Measure', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pagerank_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of PageRank scores"""
        plt.figure(figsize=(14, 8))
        
        if 'custom_pagerank' in self.centrality_measures:
            scores = list(self.centrality_measures['custom_pagerank'].values())
        else:
            scores = list(nx.pagerank(self.G).values())
        
        # Create histogram with KDE
        n, bins, patches = plt.hist(scores, bins=30, alpha=0.7, color='skyblue', 
                                   edgecolor='black', density=True)
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(scores)
        x_range = np.linspace(min(scores), max(scores), 1000)
        plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Kernel Density Estimate')
        
        # Add vertical lines for mean and median
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        plt.axvline(mean_score, color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_score:.6f}')
        plt.axvline(median_score, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_score:.6f}')
        
        # Add statistics box
        stats_text = f'Min: {np.min(scores):.6f}\nMax: {np.max(scores):.6f}\n'
        stats_text += f'Std: {np.std(scores):.6f}\nSkew: {scipy_stats.skew(scores):.3f}\n'
        stats_text += f'Kurtosis: {scipy_stats.kurtosis(scores):.3f}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.xlabel('PageRank Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of PageRank Scores in Ovarian Cancer Network', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """Plot correlation heatmap of centrality measures"""
        if 'statistical_analysis' not in self.analysis_results:
            self.perform_statistical_analysis()
        
        corr_matrix = self.analysis_results['statistical_analysis']['correlation_matrix']
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   annot_kws={"size": 9})
        
        plt.title('Correlation Matrix of Centrality Measures and Node Degrees\n'
                 'Values near +1 indicate strong positive correlation, -1 indicates strong negative correlation',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_node_type_analysis(self, save_path: Optional[str] = None):
        """Plot analysis of node types"""
        if 'statistical_analysis' not in self.analysis_results:
            self.perform_statistical_analysis()
        
        df = self.analysis_results['statistical_analysis']['dataframe']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Node type distribution
        type_counts = df['Node_Type'].value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                      colors=sns.color_palette('Set3', len(type_counts)),
                      startangle=90)
        axes[0, 0].set_title('Distribution of Node Types', fontweight='bold')
        
        # 2. Average PageRank by node type
        if 'custom_pagerank_score' in df.columns:
            type_pagerank = df.groupby('Node_Type')['custom_pagerank_score'].mean().sort_values(ascending=False)
            bars = axes[0, 1].bar(range(len(type_pagerank)), type_pagerank.values, 
                                color=sns.color_palette('viridis', len(type_pagerank)))
            axes[0, 1].set_xticks(range(len(type_pagerank)))
            axes[0, 1].set_xticklabels(type_pagerank.index, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Average PageRank Score')
            axes[0, 1].set_title('Average PageRank by Node Type', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, type_pagerank.values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, height * 1.01,
                              f'{value:.6f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Degree distribution by node type
        sns.boxplot(data=df, x='Node_Type', y='Total_Degree', ax=axes[1, 0],
                   palette='Set2')
        axes[1, 0].set_title('Degree Distribution by Node Type', fontweight='bold')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Degree')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Hub distribution by node type
        if 'Is_Hub' in df.columns:
            hub_by_type = df[df['Is_Hub']].groupby('Node_Type').size()
            if len(hub_by_type) > 0:
                bars = axes[1, 1].bar(range(len(hub_by_type)), hub_by_type.values,
                                    color=sns.color_palette('rocket', len(hub_by_type)))
                axes[1, 1].set_xticks(range(len(hub_by_type)))
                axes[1, 1].set_xticklabels(hub_by_type.index, rotation=45, ha='right')
                axes[1, 1].set_ylabel('Number of Hub Nodes')
                axes[1, 1].set_title('Hub Nodes by Node Type', fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, hub_by_type.values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, height * 1.01,
                                  str(int(value)), ha='center', va='bottom', fontsize=9)
            else:
                axes[1, 1].text(0.5, 0.5, 'No hub nodes identified', 
                               ha='center', va='center', fontsize=12)
                axes[1, 1].set_title('Hub Nodes by Node Type', fontweight='bold')
        
        plt.suptitle('Comprehensive Node Type Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot convergence history of PageRank algorithm"""
        if hasattr(self, 'convergence_history'):
            plt.figure(figsize=(12, 6))
            
            iterations = range(1, len(self.convergence_history) + 1)
            
            plt.semilogy(iterations, self.convergence_history, 'b-', linewidth=2, marker='o', 
                        markersize=6, markevery=max(1, len(iterations)//10))
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Total Change (Log Scale)', fontsize=12)
            plt.title(f'Custom PageRank Algorithm Convergence\n'
                     f'Converged after {len(self.convergence_history)} iterations '
                     f'(Threshold: {self.tol})', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Add convergence threshold line
            plt.axhline(y=self.tol, color='r', linestyle='--', linewidth=1.5, 
                       label=f'Convergence Threshold ({self.tol})')
            
            # Add final iteration annotation
            final_change = self.convergence_history[-1]
            plt.annotate(f'Final Change: {final_change:.2e}', 
                        xy=(len(self.convergence_history), final_change),
                        xytext=(len(self.convergence_history)*0.7, final_change*10),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)
            
            plt.legend()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def export_results(self, output_dir: str = "./results"):
        """Export all analysis results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Exporting results to {output_dir}...")
        
        # Generate report once
        report = self.generate_comprehensive_report()
        
        # 1. Export comprehensive report
        with codecs.open(f"{output_dir}/comprehensive_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, default=str)
        
        # 2. Export network data
        nx.write_graphml(self.G, f"{output_dir}/ovarian_cancer_network.graphml")
        
        # 3. Export centrality measures
        centrality_df = pd.DataFrame(self.centrality_measures)
        centrality_df.to_csv(f"{output_dir}/centrality_measures.csv", index_label='Node', encoding='utf-8')
        
        # 4. Export statistical analysis
        df = self.analysis_results['statistical_analysis']['dataframe']
        df.to_csv(f"{output_dir}/node_statistics.csv", index=False, encoding='utf-8')
        
        # 5. Export correlation matrix
        corr_matrix = self.analysis_results['statistical_analysis']['correlation_matrix']
        corr_matrix.to_csv(f"{output_dir}/correlation_matrix.csv", encoding='utf-8')
        
        # 6. Export summary report as text
        self._export_text_report(f"{output_dir}/analysis_summary.txt", report)
        
        logger.info("Results export completed")
    
    def _export_text_report(self, filepath: str, report: Dict[str, Any]):
        """Export a human-readable text summary"""
        try:
            with codecs.open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("OVARIAN CANCER SIGNALING NETWORK - COMPREHENSIVE ANALYSIS REPORT\n")
                f.write("=" * 100 + "\n\n")
                
                # Network overview
                f.write("1. NETWORK OVERVIEW\n")
                f.write("-" * 50 + "\n")
                stats = report['network_topology']
                f.write(f"Total Nodes: {stats['total_nodes']}\n")
                f.write(f"Total Edges: {stats['total_edges']}\n")
                f.write(f"Network Density: {stats['network_density']:.4f}\n")
                f.write(f"Average Degree: {stats['average_degree']:.2f}\n")
                f.write(f"Transitivity: {stats['transitivity']:.4f}\n")
                f.write(f"Average Clustering: {stats['average_clustering']:.4f}\n")
                f.write(f"Is Strongly Connected: {stats['is_strongly_connected']}\n")
                f.write(f"Is Weakly Connected: {stats['is_weakly_connected']}\n\n")
                
                # Degree distribution
                f.write("2. DEGREE DISTRIBUTION\n")
                f.write("-" * 50 + "\n")
                degree_dist = stats['degree_distribution']
                f.write(f"Min Degree: {degree_dist['min']}\n")
                f.write(f"Max Degree: {degree_dist['max']}\n")
                f.write(f"Mean Degree: {degree_dist['mean']:.2f}\n")
                f.write(f"Median Degree: {degree_dist['median']}\n")
                f.write(f"Degree Skewness: {degree_dist['skewness']:.3f}\n")
                f.write(f"Scale-free R²: {stats.get('scale_free_r_squared', 'N/A')}\n\n")
                
                # Top key players
                f.write("3. TOP 10 KEY PLAYERS (CUSTOM PAGERANK)\n")
                f.write("-" * 50 + "\n")
                top_players = report['top_key_players'].get('custom_pagerank', [])
                for i, player in enumerate(top_players[:10], 1):
                    f.write(f"{i:2d}. {player['Node']:25s} Score: {player['custom_pagerank_score']:.6f} "
                           f"Type: {player['Node_Type']}\n")
                f.write("\n")
                
                # Hub analysis
                f.write("4. HUB NODES ANALYSIS\n")
                f.write("-" * 50 + "\n")
                hub_stats = report['hub_analysis']
                f.write(f"Total Hub Nodes: {hub_stats['total_hubs']}\n")
                f.write("Hub Distribution by Type:\n")
                for node_type, count in hub_stats['hub_by_type'].items():
                    f.write(f"  {node_type}: {count}\n")
                f.write("\nTop 10 Hub Nodes by Degree:\n")
                for i, hub in enumerate(hub_stats['top_hubs'][:10], 1):
                    f.write(f"{i:2d}. {hub['Node']:25s} Degree: {hub['Total_Degree']} "
                           f"Type: {hub['Node_Type']}\n")
                f.write("\n")
                
                # Node type analysis
                f.write("5. NODE TYPE ANALYSIS\n")
                f.write("-" * 50 + "\n")
                for node_type, stats in report['node_type_analysis'].items():
                    f.write(f"{node_type:25s}: {stats['count']:2d} nodes | "
                           f"Avg Degree: {stats['avg_degree']:.2f} | "
                           f"Avg PageRank: {stats['avg_pagerank']:.6f} | "
                           f"Hubs: {stats['hub_count']}\n")
                f.write("\n")
                
                # Therapeutic implications
                f.write("6. THERAPEUTIC IMPLICATIONS\n")
                f.write("-" * 50 + "\n")
                implications = report['therapeutic_implications']
                f.write(f"Critical Targets Identified: {len(implications['critical_targets'])}\n")
                if implications['critical_targets']:
                    f.write("Top Critical Targets:\n")
                    for target in implications['critical_targets'][:5]:
                        f.write(f"  • {target['node']} ({target['node_type']}, {target['biological_role']})\n")
                        f.write(f"    Composite Score: {target['composite_score']:.4f}\n")
                        f.write(f"    Rationale: {target['rationale']}\n")
                
                f.write(f"\nNetwork Vulnerabilities: {len(implications['network_vulnerabilities'])}\n")
                if implications['network_vulnerabilities']:
                    f.write("Key Network Vulnerabilities:\n")
                    for vuln in implications['network_vulnerabilities']:
                        f.write(f"  • {vuln['node']} (Betweenness: {vuln['betweenness']:.4f})\n")
                        f.write(f"    Rationale: {vuln['rationale']}\n")
                
                f.write(f"\nCombination Therapy Candidates: {len(implications['combination_therapy_candidates'])}\n")
                if implications['combination_therapy_candidates']:
                    for combo in implications['combination_therapy_candidates']:
                        f.write(f"  • {combo['protein_target']} + {combo['rna_target']}\n")
                        f.write(f"    Rationale: {combo['rationale']}\n")
                
        except Exception as e:
            logger.error(f"Error writing text report: {e}")


def main():
    """Main function to run comprehensive ovarian cancer network analysis"""
    
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
    
    logger.info(f"Total interactions: {len(interactions)}")
    
    # Initialize comprehensive analyzer
    analyzer = OvarianCancerNetworkAnalyzer(damping_factor=0.85, max_iter=100)
    
    # Build network
    network = analyzer.build_network_from_data(interactions)
    
    # Compute all centrality measures
    centrality_measures = analyzer.compute_all_centrality_measures()
    
    # Identify hub nodes
    hub_nodes = analyzer.identify_hub_nodes()
    
    # Perform comprehensive analyses
    topology_stats = analyzer.analyze_network_topology()
    statistical_results = analyzer.perform_statistical_analysis()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Display key results in console
    print("=" * 120)
    print("OVARIAN CANCER SIGNALING NETWORK - COMPREHENSIVE ANALYSIS")
    print("=" * 120)
    
    print("\n1. NETWORK OVERVIEW:")
    print("-" * 60)
    print(f"Total Nodes: {topology_stats['total_nodes']}")
    print(f"Total Edges: {topology_stats['total_edges']}")
    print(f"Network Density: {topology_stats['network_density']:.4f}")
    print(f"Average Degree: {topology_stats['average_degree']:.2f}")
    print(f"Transitivity: {topology_stats['transitivity']:.4f}")
    print(f"Is Strongly Connected: {topology_stats['is_strongly_connected']}")
    print(f"Is Weakly Connected: {topology_stats['is_weakly_connected']}")
    
    print("\n2. DEGREE DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    degree_dist = topology_stats['degree_distribution']
    print(f"Min Degree: {degree_dist['min']}")
    print(f"Max Degree: {degree_dist['max']}")
    print(f"Mean Degree: {degree_dist['mean']:.2f}")
    print(f"Median Degree: {degree_dist['median']}")
    print(f"Degree Skewness: {degree_dist['skewness']:.2f}")
    print(f"Scale-free R²: {topology_stats.get('scale_free_r_squared', 'N/A')}")
    
    print("\n3. TOP 10 KEY PLAYERS (CUSTOM PAGERANK):")
    print("-" * 60)
    custom_pr = centrality_measures['custom_pagerank']
    top_players = sorted(custom_pr.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (node, score) in enumerate(top_players, 1):
        node_type = network.nodes[node].get('node_type', 'Unknown')
        in_deg = network.in_degree(node)
        out_deg = network.out_degree(node)
        print(f"{i:2d}. {node:25s} (Score: {score:.6f}, Type: {node_type:20s}, In: {in_deg:2d}, Out: {out_deg:2d})")
    
    print("\n4. HUB NODES IDENTIFICATION:")
    print("-" * 60)
    print(f"Number of Hub Nodes (top 20% by degree): {len(hub_nodes)}")
    print("\nTop 10 Hub Nodes:")
    for i, (node, score) in enumerate(hub_nodes[:10], 1):
        degree = network.degree(node)
        node_type = network.nodes[node].get('node_type', 'Unknown')
        print(f"{i:2d}. {node:25s} (Degree Centrality: {score:.4f}, Total Degree: {degree}, Type: {node_type})")
    
    print("\n5. CENTRALITY CORRELATIONS:")
    print("-" * 60)
    if 'statistical_analysis' in analyzer.analysis_results:
        corr_matrix = analyzer.analysis_results['statistical_analysis']['correlation_matrix']
        # Get correlations with custom PageRank
        if 'custom_pagerank_score' in corr_matrix.index:
            print(f"PageRank vs In-Degree Correlation: {corr_matrix.loc['custom_pagerank_score', 'In_Degree']:.4f}")
            print(f"PageRank vs Out-Degree Correlation: {corr_matrix.loc['custom_pagerank_score', 'Out_Degree']:.4f}")
            print(f"PageRank vs Total Degree Correlation: {corr_matrix.loc['custom_pagerank_score', 'Total_Degree']:.4f}")
            print(f"Betweenness vs Degree Correlation: {corr_matrix.loc['betweenness_centrality_score', 'degree_centrality_score']:.4f}")
    
    print("\n6. NODE TYPE ANALYSIS:")
    print("-" * 60)
    df = analyzer.analysis_results['statistical_analysis']['dataframe']
    for node_type in sorted(df['Node_Type'].unique()):
        type_data = df[df['Node_Type'] == node_type]
        count = len(type_data)
        avg_degree = type_data['Total_Degree'].mean()
        avg_pagerank = type_data['custom_pagerank_score'].mean()
        hub_count = type_data['Is_Hub'].sum()
        print(f"{node_type:25s}: {count:2d} nodes | Avg Degree: {avg_degree:.2f} | Avg PageRank: {avg_pagerank:.6f} | Hubs: {hub_count}")
    
    print("\n7. THERAPEUTIC IMPLICATIONS:")
    print("-" * 60)
    if 'therapeutic_implications' in report:
        implications = report['therapeutic_implications']
        print(f"Critical Targets Identified: {len(implications.get('critical_targets', []))}")
        if implications.get('critical_targets'):
            print("  Top Critical Targets:")
            for target in implications['critical_targets'][:3]:
                print(f"    • {target['node']} ({target['biological_role']}): {target['rationale']}")
        
        print(f"\nNetwork Vulnerabilities: {len(implications.get('network_vulnerabilities', []))}")
        if implications.get('network_vulnerabilities'):
            print("  Key Vulnerabilities:")
            for vuln in implications['network_vulnerabilities'][:2]:
                print(f"    • {vuln['node']} (Betweenness: {vuln['betweenness']:.4f}): {vuln['rationale']}")
        
        print(f"\nCombination Therapy Candidates: {len(implications.get('combination_therapy_candidates', []))}")
        if implications.get('combination_therapy_candidates'):
            for combo in implications['combination_therapy_candidates']:
                print(f"  • {combo['protein_target']} + {combo['rna_target']}: {combo['rationale']}")
    
    print("\n" + "=" * 120)
    print("Analysis Complete! Generating visualizations and exporting results...")
    print("=" * 120)
    
    # Generate visualizations
    analyzer.visualize_comprehensive_analysis()
    
    # Export all results
    analyzer.export_results()
    
    logger.info("All analyses completed successfully!")
    
    return analyzer, centrality_measures, report


if __name__ == "__main__":
    analyzer, centrality_measures, report = main()
