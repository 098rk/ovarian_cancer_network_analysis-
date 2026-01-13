import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, List, Tuple, Any
import warnings
import random
from collections import defaultdict

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


class OvarianCancerNetworkProcessor:
    """Process ovarian cancer network data for RCNN with ALL interactions"""

    def __init__(self):
        self.G = nx.DiGraph()
        self.node_features = {}
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = []
        self.node_to_index = {}

    def build_complete_network(self) -> nx.DiGraph:
        """Build the complete ovarian cancer network with ALL interactions"""
        logger.info("Building complete ovarian cancer network with ALL interactions...")

        # Define ALL interactions from the provided tables - COMPLETE SET
        all_interactions = [
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
            ('ChK2-p nuc', 'ChK2 nuc', 'interacts_with'),
            ('ChK2-p nuc', 'p53-p', 'interacts_with'),
            ('ChK2-p nuc', 'Mdm2-p-p nuc', 'interacts_with'),
            ('ChK2-p nuc', 'Mdm2-p nuc', 'interacts_with'),
            ('ChK2-p nuc', 'Mdm2 cyt', 'interacts_with'),
            ('ChK2-p nuc', 'Mdm2-p cyt', 'interacts_with'),
            ('DNA', 'ATM mRNA nuc', 'interacts_with'),
            ('ATM mRNA nuc', 'ATM nuc', 'interacts_with'),

            # ATM/ATR Signaling Continued (101-110)
            ('ATM-p nuc', 'ATM nuc', 'interacts_with'),
            ('ATM nuc', 'ATM-p nuc', 'interacts_with'),
            ('ATMa-p nuc', 'ATM-p nuc', 'interacts_with'),
            ('ATM-p nuc', 'ATMa-p nuc', 'interacts_with'),
            ('ATMa-p nuc', 'p53-p', 'interacts_with'),
            ('ATMa-p nuc', 'IKKa cyt', 'interacts_with'),
            ('ATMa-p nuc', 'Mdm2-p-p nuc', 'interacts_with'),
            ('ATMa-p nuc', 'KSRP-p cyt', 'interacts_with'),
            ('ATMa-p nuc', 'AKT-p cyt', 'interacts_with'),
            ('ATMa-p nuc', 'CREB nuc', 'interacts_with'),

            # CREB Signaling (111-115)
            ('CREB nuc', 'CREB-p nuc', 'interacts_with'),
            ('CREB-p nuc', 'CREB nuc', 'interacts_with'),
            ('CREB nuc', 'Wip1 mRNA nuc', 'interacts_with'),
            ('CREB nuc', 'ATM mRNA nuc', 'interacts_with'),

            # MRN Complex (116-120)
            ('ATM-p nuc', 'MRN-p nuc', 'interacts_with'),
            ('DSB nuc', 'MRN-p nuc', 'interacts_with'),
            ('MRN-p nuc', 'MRN nuc', 'interacts_with'),
            ('MRN nuc', 'MRN-p nuc', 'interacts_with'),
            ('MRN-p nuc', 'ATMa-p nuc', 'interacts_with'),

            # TNFα Signaling (121-123)
            ('TNFα', 'TNFR1 cyt', 'interacts_with'),
            ('TNFR1 cyt', 'IKKKa cyt', 'interacts_with'),
        ]

        # Add all nodes and edges to the graph
        for source, target, interaction_type in all_interactions:
            # Add nodes with properties
            if source not in self.G:
                self.G.add_node(source, node_type=self._classify_node_type(source))
            if target not in self.G:
                self.G.add_node(target, node_type=self._classify_node_type(target))

            # Add edge with weight
            weight = self._calculate_edge_weight(interaction_type)
            self.G.add_edge(source, target, weight=weight, interaction_type=interaction_type)

        logger.info(
            f"Complete network built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")

        # Print node and edge statistics
        self._print_network_statistics()

        return self.G

    def _print_network_statistics(self):
        """Print detailed network statistics"""
        print("\n" + "=" * 60)
        print("NETWORK STATISTICS")
        print("=" * 60)
        print(f"Total Nodes: {self.G.number_of_nodes()}")
        print(f"Total Edges: {self.G.number_of_edges()}")
        print(f"Network Density: {nx.density(self.G):.4f}")

        # Node type distribution
        node_types = {}
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('node_type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"\nNode Type Distribution:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count} nodes")

        # Degree statistics
        degrees = [d for n, d in self.G.degree()]
        print(f"\nDegree Statistics:")
        print(f"  Average Degree: {np.mean(degrees):.2f}")
        print(f"  Maximum Degree: {max(degrees)}")
        print(f"  Minimum Degree: {min(degrees)}")

        # Central nodes
        try:
            pagerank = nx.pagerank(self.G)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 Central Nodes (PageRank):")
            for node, score in top_nodes:
                node_type = self.G.nodes[node].get('node_type', 'Unknown')
                print(f"  {node}: {score:.4f} ({node_type})")
        except Exception as e:
            print(f"\nError computing PageRank: {e}")

    def _classify_node_type(self, node_name: str) -> str:
        """Classify nodes into biological categories"""
        node_lower = node_name.lower()

        if any(x in node_lower for x in ['mrna', 'mirna', 'pre-mirna', 'mir']):
            return 'RNA'
        elif any(x in node_lower for x in ['mrn']):
            return 'Cellular_Component'
        elif any(x in node_lower for x in ['creb', 'bax', 'akt', 'mdm2', 'ikk', 'chk2', 'atm', 'wip1', 'ksrp', 'tnfr']):
            return 'Protein'
        elif any(x in node_lower for x in ['p53', 'nfkb']):
            return 'Protein'
        elif any(x in node_lower for x in ['dsb', 'dna']):
            return 'Cellular_Component'
        elif 'nuc' in node_lower or 'cyt' in node_lower:
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

    def compute_pagerank(self, damping_factor: float = 0.85) -> Dict[str, float]:
        """Compute PageRank scores for all nodes"""
        logger.info("Computing PageRank scores...")
        try:
            pagerank_scores = nx.pagerank(self.G, alpha=damping_factor, weight='weight')
            return pagerank_scores
        except Exception as e:
            logger.error(f"Error computing PageRank: {e}")
            # Return uniform scores if PageRank fails
            return {node: 1.0 / len(self.G.nodes()) for node in self.G.nodes()}

    def extract_comprehensive_features(self, pagerank_scores: Dict[str, float]):
        """Extract comprehensive node features WITHOUT PageRank to avoid target leakage"""
        logger.info("Extracting comprehensive node features...")

        # Define comprehensive feature names (EXCLUDING PageRank)
        self.feature_names = [
                                 'in_degree', 'out_degree', 'total_degree', 'degree_centrality',
                                 'clustering_coeff', 'betweenness', 'eigenvector', 'closeness',
                                 'avg_neighbor_degree', 'is_critical_pathway', 'node_importance',
                                 'pathway_hub_score', 'regulatory_potential'
                             ] + ['type_' + t for t in
                                  ['Protein', 'RNA', 'Cellular_Component', 'Phenotype', 'Stimulus', 'Other']]

        # Compute comprehensive network centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.G, weight='weight')
        except:
            betweenness = {node: 0 for node in self.G.nodes()}

        try:
            eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000, weight='weight', tol=1e-3)
        except:
            eigenvector = {node: 0 for node in self.G.nodes()}

        try:
            closeness = nx.closeness_centrality(self.G)
        except:
            closeness = {node: 0 for node in self.G.nodes()}

        degree_centrality = nx.degree_centrality(self.G)

        feature_data = []
        node_list = list(self.G.nodes())

        # Create node to index mapping
        self.node_to_index = {node: i for i, node in enumerate(node_list)}

        for node in node_list:
            # Basic network features
            in_degree = self.G.in_degree(node)
            out_degree = self.G.out_degree(node)
            total_degree = in_degree + out_degree

            # Centrality features (EXCLUDING PageRank)
            betweenness_cent = betweenness.get(node, 0)
            eigenvector_cent = eigenvector.get(node, 0)
            closeness_cent = closeness.get(node, 0)
            degree_cent = degree_centrality.get(node, 0)

            # Clustering coefficient
            try:
                clustering = nx.clustering(nx.Graph(self.G), node)  # Convert to undirected for clustering
            except:
                clustering = 0

            # Neighborhood features
            neighbors = list(self.G.neighbors(node))
            avg_neighbor_degree = np.mean([self.G.degree(n) for n in neighbors]) if neighbors else 0

            # Biological importance features
            is_critical_pathway = 1.0 if any(
                x in node.lower() for x in ['p53', 'nfkb', 'akt', 'bax', 'apoptosis', 'atm', 'mdm2']) else 0.0
            node_importance = self._calculate_node_importance(node)

            # Advanced biological features
            pathway_hub_score = self._calculate_pathway_hub_score(node)
            regulatory_potential = self._calculate_regulatory_potential(node)

            # Node type encoding
            node_type = self.G.nodes[node].get('node_type', 'Other')
            type_encoding = self._one_hot_encode_node_type(node_type)

            # Combine all features (EXCLUDING PageRank)
            features = [
                in_degree,
                out_degree,
                total_degree,
                degree_cent,
                clustering,
                betweenness_cent,
                eigenvector_cent,
                closeness_cent,
                avg_neighbor_degree,
                is_critical_pathway,
                node_importance,
                pathway_hub_score,
                regulatory_potential,
                *type_encoding
            ]

            feature_data.append(features)
            self.node_features[node] = np.array(features, dtype=np.float32)

        # Scale features
        feature_data = np.array(feature_data)
        if len(feature_data) > 0:
            self.feature_scaler.fit(feature_data)

            # Apply scaling to all features
            for i, node in enumerate(node_list):
                self.node_features[node] = self.feature_scaler.transform([feature_data[i]])[0]

        logger.info(f"Extracted {len(self.feature_names)} comprehensive features for {len(self.node_features)} nodes")

    def _one_hot_encode_node_type(self, node_type: str) -> List[float]:
        """One-hot encode node types"""
        types = ['Protein', 'RNA', 'Cellular_Component', 'Phenotype', 'Stimulus', 'Other']
        encoding = [0] * len(types)
        if node_type in types:
            encoding[types.index(node_type)] = 1
        return encoding

    def _calculate_node_importance(self, node: str) -> float:
        """Calculate biological importance score"""
        importance_factors = {
            'p53': 1.0, 'nfkb': 0.95, 'akt': 0.9, 'bax': 0.85, 'apoptosis': 0.9,
            'mdm2': 0.85, 'atm': 0.88, 'chk2': 0.8, 'wip1': 0.8, 'creb': 0.75,
            'ikk': 0.8, 'tnf': 0.7, 'pi3k': 0.8, 'pten': 0.8, 'pip': 0.6
        }

        node_lower = node.lower()
        for key, value in importance_factors.items():
            if key in node_lower:
                return value
        return 0.4  # Default importance

    def _calculate_pathway_hub_score(self, node: str) -> float:
        """Calculate pathway hub score based on network position"""
        try:
            # Calculate how many shortest paths go through this node
            betweenness = nx.betweenness_centrality(self.G).get(node, 0)
            return betweenness
        except:
            return 0

    def _calculate_regulatory_potential(self, node: str) -> float:
        """Calculate regulatory potential based on out-degree and biological role"""
        out_degree = self.G.out_degree(node)
        out_degrees = [d for n, d in self.G.out_degree()]
        max_out_degree = max(out_degrees) if out_degrees else 1

        # Proteins and transcription factors have higher regulatory potential
        node_type = self.G.nodes[node].get('node_type', 'Other')
        type_multiplier = 1.0
        if node_type == 'Protein':
            type_multiplier = 1.5
        elif node_type == 'RNA':
            type_multiplier = 0.8

        return (out_degree / max_out_degree) * type_multiplier if max_out_degree > 0 else 0

    def generate_comprehensive_sequences(self, sequence_length: int = 10, walks_per_node: int = 5) -> Tuple[
        np.ndarray, np.ndarray]:
        """Generate comprehensive sequences using biased random walks"""
        logger.info(
            f"Generating comprehensive sequences (length: {sequence_length}, walks per node: {walks_per_node})...")

        sequences = []
        targets = []
        node_list = list(self.G.nodes())

        if len(node_list) == 0:
            logger.error("No nodes in graph!")
            return np.array([]), np.array([])

        # Get PageRank scores for targets
        pagerank_scores = self.compute_pagerank()
        if len(pagerank_scores) == 0:
            logger.error("No PageRank scores computed!")
            return np.array([]), np.array([])

        target_values = np.array([pagerank_scores.get(node, 0) for node in node_list]).reshape(-1, 1)

        # Check if target values are valid
        if len(target_values) > 0 and not np.isnan(target_values).any():
            target_values = self.target_scaler.fit_transform(target_values).flatten()
        else:
            logger.error("Invalid target values!")
            target_values = np.zeros(len(node_list))

        target_dict = {node: target_values[i] for i, node in enumerate(node_list)}

        for node in node_list:
            for _ in range(walks_per_node):
                # Use biased random walk preferring high-weight edges
                walk = self._biased_random_walk(node, sequence_length)

                # Convert to feature sequence
                try:
                    feature_sequence = np.array([self.node_features[n] for n in walk])
                    sequences.append(feature_sequence)

                    # Target: Scaled PageRank score
                    if node in target_dict:
                        target = target_dict[node]
                        targets.append(target)
                    else:
                        targets.append(0)
                except KeyError as e:
                    logger.warning(f"Missing node features for {e}, skipping sequence")
                    continue
                except Exception as e:
                    logger.warning(f"Error generating sequence: {e}")
                    continue

        if len(sequences) == 0:
            logger.error("No sequences generated! Check node features.")
            return np.array([]), np.array([])

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        logger.info(f"Generated {len(sequences)} comprehensive sequences")
        return sequences, targets

    def _biased_random_walk(self, start_node: str, length: int) -> List[str]:
        """Perform biased random walk preferring biologically important paths"""
        walk = [start_node]
        current_node = start_node

        for step in range(length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                # If no neighbors, restart or stay
                if np.random.random() < 0.3:  # 30% chance to restart
                    current_node = start_node
                walk.append(current_node)
                continue

            # Bias towards neighbors with higher edge weights and importance
            weights = []
            for neighbor in neighbors:
                try:
                    edge_weight = self.G[current_node][neighbor].get('weight', 1.0)
                except:
                    edge_weight = 1.0
                neighbor_importance = self._calculate_node_importance(neighbor)
                total_weight = edge_weight * (1 + neighbor_importance)
                weights.append(total_weight)

            weights = np.array(weights, dtype=np.float64)
            if weights.sum() > 0:
                # Normalize to avoid numerical issues
                weights = weights / weights.sum()
                # Ensure no NaN values
                weights = np.nan_to_num(weights)
                next_node = np.random.choice(neighbors, p=weights)
            else:
                next_node = np.random.choice(neighbors)

            walk.append(next_node)
            current_node = next_node

        # Pad sequence if necessary
        while len(walk) < length:
            walk.append(walk[-1] if walk else start_node)

        return walk[:length]


class OvarianCancerDataset(Dataset):
    """PyTorch Dataset for ovarian cancer network sequences"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        if len(sequences) > 0 and len(targets) > 0:
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
        else:
            self.sequences = torch.FloatTensor()
            self.targets = torch.FloatTensor()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class ComprehensiveRCNNModel(nn.Module):
    """Comprehensive RCNN model for ovarian cancer network analysis"""

    def __init__(self, input_size: int, sequence_length: int = 10,
                 hidden_size: int = 64, num_layers: int = 2,
                 num_classes: int = 1, dropout: float = 0.3):
        super(ComprehensiveRCNNModel, self).__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length

        # Enhanced convolutional layers
        self.conv1d_1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)

        # Enhanced LSTM with bidirectional
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)

        # Enhanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, dropout=dropout,
                                               batch_first=True)

        # Enhanced fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

        # Enhanced activation and regularization
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, input_size = x.size()

        # Reshape for convolutional layers
        x = x.transpose(1, 2)

        # Enhanced convolutional feature extraction
        x = self.leaky_relu(self.batch_norm1(self.conv1d_1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm2(self.conv1d_2(x)))
        x = self.dropout(x)

        # Reshape back for LSTM
        x = x.transpose(1, 2)

        # Enhanced LSTM for sequential patterns
        lstm_out, (hidden, cell) = self.lstm(x)

        # Enhanced attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last hidden state with attention
        x = attn_out[:, -1, :]
        x = self.layer_norm(x)

        # Enhanced fully connected layers
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x.squeeze()


class ComprehensiveRCNNTrainer:
    """Comprehensive trainer for RCNN model"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        # FIXED: Removed verbose parameter from ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        logger.info(f"Using device: {device}")

    def train(self, train_loader, val_loader, epochs=100):
        if len(train_loader) == 0:
            logger.error("No training data available!")
            return [], []

        logger.info("Starting comprehensive RCNN training...")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    val_loss += self.criterion(outputs, targets).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_comprehensive_rcnn_model.pth')
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f'Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')

            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

        # Load best model
        try:
            self.model.load_state_dict(torch.load('best_comprehensive_rcnn_model.pth', map_location=self.device))
            logger.info("Loaded best model weights")
        except Exception as e:
            logger.warning(f"Could not load best model: {e}")

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")

        return train_losses, val_losses

    def evaluate(self, test_loader, processor):
        """Evaluate model with inverse transformed predictions"""
        if len(test_loader) == 0:
            logger.error("No test data available!")
            return {
                'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'),
                'predictions': np.array([]), 'actuals': np.array([])
            }

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        if len(predictions) == 0 or len(actuals) == 0:
            return {
                'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'),
                'predictions': np.array([]), 'actuals': np.array([])
            }

        # Inverse transform predictions and actuals
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)

        try:
            predictions_original = processor.target_scaler.inverse_transform(predictions).flatten()
            actuals_original = processor.target_scaler.inverse_transform(actuals).flatten()
        except Exception as e:
            logger.warning(f"Could not inverse transform: {e}")
            predictions_original = predictions.flatten()
            actuals_original = actuals.flatten()

        mse = mean_squared_error(actuals_original, predictions_original)
        mae = mean_absolute_error(actuals_original, predictions_original)
        try:
            r2 = r2_score(actuals_original, predictions_original)
        except:
            r2 = -1.0

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions_original,
            'actuals': actuals_original
        }


def plot_comprehensive_results(train_losses, val_losses, evaluation, processor):
    """Plot comprehensive results with biological insights"""

    if len(train_losses) == 0:
        logger.error("No training history to plot!")
        return pd.DataFrame()

    plt.figure(figsize=(15, 10))

    # 1. Training curves
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2, alpha=0.8, color='blue')
    plt.plot(val_losses, label='Validation Loss', linewidth=2, alpha=0.8, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('RCNN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if max(train_losses) > 0:
        plt.yscale('log')

    # 2. Predictions vs actuals
    plt.subplot(2, 2, 2)
    if len(evaluation['actuals']) > 0 and len(evaluation['predictions']) > 0:
        plt.scatter(evaluation['actuals'], evaluation['predictions'], alpha=0.6, color='green', s=60)
        min_val = min(evaluation['actuals'].min(), evaluation['predictions'].min())
        max_val = max(evaluation['actuals'].max(), evaluation['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual PageRank Scores')
        plt.ylabel('Predicted PageRank Scores')
        plt.title(f'RCNN Predictions vs Actual\nR² = {evaluation["r2"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Prediction Data Available')

    # 3. Feature importance based on variability
    plt.subplot(2, 2, 3)
    feature_ranges = []
    for feature in processor.feature_names:
        feature_values = []
        for node in processor.G.nodes():
            try:
                if node in processor.node_features:
                    idx = processor.feature_names.index(feature)
                    feature_values.append(processor.node_features[node][idx])
            except (KeyError, IndexError, ValueError):
                continue
        if feature_values:
            feature_ranges.append(np.ptp(feature_values))
        else:
            feature_ranges.append(0)

    if feature_ranges:
        importance_df = pd.DataFrame({
            'Feature': processor.feature_names,
            'Variability': feature_ranges
        }).sort_values('Variability', ascending=True)

        # Take top 10 features
        top_features = importance_df.tail(10)
        plt.barh(top_features['Feature'], top_features['Variability'], color='purple', alpha=0.7)
        plt.xlabel('Feature Value Range')
        plt.title('Top Feature Variability')
        plt.grid(True, alpha=0.3, axis='x')
    else:
        importance_df = pd.DataFrame()
        plt.text(0.5, 0.5, 'No feature data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Feature Data')

    # 4. Network degree distribution
    plt.subplot(2, 2, 4)
    degrees = [d for n, d in processor.G.degree()]
    if degrees:
        plt.hist(degrees, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Node Degree')
        plt.ylabel('Frequency')
        plt.title('Network Degree Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No degree data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Network Data')

    plt.tight_layout()
    plt.show()

    return importance_df


def run_comprehensive_analysis():
    """Run comprehensive RCNN analysis with ALL interactions"""

    # Initialize comprehensive processor
    processor = OvarianCancerNetworkProcessor()

    # Build complete network with ALL interactions
    network = processor.build_complete_network()

    if network.number_of_nodes() == 0:
        logger.error("Network has no nodes! Cannot proceed.")
        return None

    # Compute PageRank (for targets only)
    pagerank_scores = processor.compute_pagerank()

    # Extract comprehensive node features (EXCLUDING PageRank)
    processor.extract_comprehensive_features(pagerank_scores)

    if len(processor.node_features) == 0:
        logger.error("No node features extracted! Cannot proceed.")
        return None

    # Generate comprehensive sequences
    sequences, targets = processor.generate_comprehensive_sequences(sequence_length=10, walks_per_node=5)

    if len(sequences) == 0 or len(targets) == 0:
        logger.error("No sequences generated! Cannot proceed with training.")
        return None

    # Create dataset
    dataset = OvarianCancerDataset(sequences, targets)

    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return None

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    try:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    logger.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Initialize comprehensive RCNN model
    input_size = sequences.shape[2]
    model = ComprehensiveRCNNModel(
        input_size=input_size,
        sequence_length=10,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )

    logger.info(f"Comprehensive RCNN Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    trainer = ComprehensiveRCNNTrainer(model)
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=100)

    # Evaluate model
    evaluation = trainer.evaluate(test_loader, processor)

    # Generate comprehensive analysis
    importance_df = plot_comprehensive_results(train_losses, val_losses, evaluation, processor)

    # Display comprehensive results
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RCNN ANALYSIS - OVARIAN CANCER SIGNALING NETWORK")
    print("=" * 100)

    print(f"\nMODEL PERFORMANCE:")
    print(f"Mean Squared Error (MSE): {evaluation['mse']:.6f}")
    print(f"Mean Absolute Error (MAE): {evaluation['mae']:.6f}")
    print(f"R² Score: {evaluation['r2']:.4f}")

    print(f"\nMODEL ARCHITECTURE:")
    print(f"Input Features: {input_size} (excluding PageRank)")
    print(f"Sequence Length: 10")
    print(f"LSTM Hidden Size: 64 (Bidirectional)")
    print(f"LSTM Layers: 2")
    print(f"Convolutional Filters: 32 -> 64")
    print(f"Attention Heads: 4")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTRAINING SUMMARY:")
    if train_losses:
        print(f"Final Training Loss: {train_losses[-1]:.6f}")
    else:
        print(f"Final Training Loss: N/A")
    if val_losses:
        print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    else:
        print(f"Final Validation Loss: N/A")
    print(f"Training Sequences: {len(train_dataset)}")
    print(f"Validation Sequences: {len(val_dataset)}")
    print(f"Test Sequences: {len(test_dataset)}")

    print(f"\nTOP 10 MOST VARIABLE FEATURES:")
    if importance_df is not None and len(importance_df) > 0:
        top_features = importance_df.tail(10)
        for i, row in top_features.iterrows():
            print(f"  {row['Feature']}: {row['Variability']:.4f}")

    # Biological insights
    print(f"\nBIOLOGICAL INSIGHTS:")
    top_pagerank_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    print("Top 8 Most Important Nodes by PageRank:")
    for node, score in top_pagerank_nodes:
        node_type = network.nodes[node].get('node_type', 'Unknown')
        degree = network.degree(node)
        print(f"  {node:25s}: {score:.4f} (Type: {node_type:20s}, Degree: {degree:2d})")

    # Save comprehensive results
    results = {
        'model': model,
        'trainer': trainer,
        'processor': processor,
        'evaluation': evaluation,
        'feature_importance': importance_df,
        'pagerank_scores': pagerank_scores,
        'network': network,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    }

    logger.info("Comprehensive RCNN analysis with ALL interactions completed successfully!")

    return results


if __name__ == "__main__":
    results = run_comprehensive_analysis()
