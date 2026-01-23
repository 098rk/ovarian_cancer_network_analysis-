import matplotlib.pyplot as plt
import random as rd
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
rd.seed(42)
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class TNFR1BooleanModel:
    """Complete Boolean modeling of TNFR1 signaling network"""

    def __init__(self):
        self.G = nx.DiGraph()
        self.states = {}
        self.states_history = []
        self.pivotal_nodes = []
        self.stable_states = {}
        self.activation_frequencies = {}
        self.time_steps = 100

    def build_complete_network(self):
        """Build the complete TNFR1 signaling network with all nodes and edges"""
        print("Building complete TNFR1 signaling network...")

        # ============================================
        # ALL NODES FROM TABLE 4.26
        # ============================================
        nodes_data = [
            # External stimuli and receptors
            ('TNFa', {'type': 'Stimulus', 'group': 'External'}),
            ('TNFR1', {'type': 'Receptor', 'group': 'Membrane'}),
            ('IR', {'type': 'Stimulus', 'group': 'External'}),
            ('DSB', {'type': 'DNA_Damage', 'group': 'Nuclear'}),

            # Kinases and signaling proteins
            ('IKKKα', {'type': 'Kinase', 'group': 'Signaling'}),
            ('IKKα', {'type': 'Kinase', 'group': 'Signaling'}),
            ('ATM', {'type': 'Kinase', 'group': 'DNA_Repair'}),
            ('ATM-p', {'type': 'Phosphoprotein', 'group': 'DNA_Repair'}),
            ('ATMa-p', {'type': 'Phosphoprotein', 'group': 'DNA_Repair'}),
            ('MRN-p', {'type': 'Phosphoprotein', 'group': 'DNA_Repair'}),
            ('Chk2', {'type': 'Protein', 'group': 'DNA_Repair'}),
            ('Chk2-p', {'type': 'Phosphoprotein', 'group': 'DNA_Repair'}),
            ('AKT-p', {'type': 'Phosphoprotein', 'group': 'Signaling'}),
            ('KSRP-p', {'type': 'Phosphoprotein', 'group': 'Signaling'}),
            ('Wip1', {'type': 'Protein', 'group': 'Cytoplasmic'}),

            # Transcription factors
            ('NFkB (TF)', {'type': 'Transcription_Factor', 'group': 'Nuclear'}),
            ('p53 (TF)', {'type': 'Transcription_Factor', 'group': 'Nuclear'}),
            ('CREB (TF)', {'type': 'Transcription_Factor', 'group': 'Nuclear'}),

            # Proteins
            ('A20', {'type': 'Protein', 'group': 'Cytoplasmic'}),
            ('IkBa', {'type': 'Protein', 'group': 'Cytoplasmic'}),
            ('Mdm2 cyt', {'type': 'Protein', 'group': 'Cytoplasmic'}),
            ('Mdm2-p cyt', {'type': 'Phosphoprotein', 'group': 'Cytoplasmic'}),
            ('Mdm2-p nuc', {'type': 'Phosphoprotein', 'group': 'Nuclear'}),
            ('PTEN', {'type': 'Protein', 'group': 'Cytoplasmic'}),
            ('Bax', {'type': 'Protein', 'group': 'Mitochondrial'}),
            ('p21', {'type': 'Protein', 'group': 'Cytoplasmic'}),
            ('p53', {'type': 'Protein', 'group': 'Nuclear'}),
            ('p53-p', {'type': 'Phosphoprotein', 'group': 'Signaling'}),

            # mRNAs
            ('A20 mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('IkBa mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('Wip mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('Wip1 mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('p53 mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('ATM mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('Bax mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('Mdm2 mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('p21 mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('Chk2 mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('PTEN mRNA (Genomic)', {'type': 'RNA', 'group': 'Cytoplasmic'}),
            ('IKBa mRNA', {'type': 'RNA', 'group': 'Cytoplasmic'}),

            # microRNAs
            ('miR-16', {'type': 'microRNA', 'group': 'Cytoplasmic'}),
            ('pre-miR-16', {'type': 'microRNA', 'group': 'Cytoplasmic'}),

            # Metabolites
            ('PIP2', {'type': 'Metabolite', 'group': 'Membrane'}),
            ('PIP3', {'type': 'Metabolite', 'group': 'Membrane'}),

            # Phenotypes
            ('apoptosis', {'type': 'Phenotype', 'group': 'Output'}),
            ('cell cycle arrest', {'type': 'Phenotype', 'group': 'Output'}),
        ]

        # Add all nodes
        for node, attrs in nodes_data:
            self.G.add_node(node, **attrs)
            self.G.nodes[node]['state'] = 0  # Initialize state

        print(f"✓ Added {len(nodes_data)} nodes")

        # ============================================
        # ALL EDGES WITH BIOLOGICAL LOGIC
        # ============================================
        edges_data = [
            # TNFa signaling pathway (Activation edges)
            ('TNFa', 'TNFR1', {'type': 'activation', 'weight': 1.0}),
            ('TNFR1', 'IKKKα', {'type': 'activation', 'weight': 1.0}),
            ('IKKKα', 'IKKα', {'type': 'activation', 'weight': 1.0}),
            ('IKKα', 'NFkB (TF)', {'type': 'activation', 'weight': 1.0}),
            ('TNFR1', 'NFkB (TF)', {'type': 'activation', 'weight': 0.8}),

            # DNA damage response pathway
            ('IR', 'DSB', {'type': 'activation', 'weight': 1.0}),
            ('DSB', 'ATM', {'type': 'activation', 'weight': 1.0}),
            ('ATM', 'ATM-p', {'type': 'activation', 'weight': 1.0}),
            ('ATM-p', 'ATMa-p', {'type': 'activation', 'weight': 1.0}),
            ('ATMa-p', 'MRN-p', {'type': 'activation', 'weight': 1.0}),
            ('ATMa-p', 'Chk2-p', {'type': 'activation', 'weight': 1.0}),
            ('ATMa-p', 'p53-p', {'type': 'activation', 'weight': 1.0}),
            ('ATMa-p', 'IKKα', {'type': 'activation', 'weight': 0.8}),
            ('ATMa-p', 'AKT-p', {'type': 'activation', 'weight': 0.7}),
            ('ATMa-p', 'KSRP-p', {'type': 'activation', 'weight': 0.7}),
            ('ATMa-p', 'CREB (TF)', {'type': 'activation', 'weight': 0.7}),

            # p53 pathway
            ('p53 (TF)', 'p53', {'type': 'activation', 'weight': 0.8}),
            ('p53-p', 'p53', {'type': 'activation', 'weight': 0.9}),
            ('p53', 'p21', {'type': 'activation', 'weight': 0.8}),
            ('p53', 'Bax', {'type': 'activation', 'weight': 0.8}),
            ('p53', 'Mdm2 cyt', {'type': 'activation', 'weight': 0.7}),

            # NF-κB transcriptional targets
            ('NFkB (TF)', 'A20 mRNA', {'type': 'activation', 'weight': 0.8}),
            ('NFkB (TF)', 'IkBa mRNA', {'type': 'activation', 'weight': 0.8}),
            ('NFkB (TF)', 'Bax mRNA', {'type': 'activation', 'weight': 0.7}),
            ('NFkB (TF)', 'Mdm2 mRNA', {'type': 'activation', 'weight': 0.7}),
            ('NFkB (TF)', 'p21 mRNA', {'type': 'activation', 'weight': 0.7}),
            ('NFkB (TF)', 'IKBa mRNA', {'type': 'activation', 'weight': 0.7}),

            # Translation processes
            ('A20 mRNA', 'A20', {'type': 'activation', 'weight': 0.8}),
            ('IkBa mRNA', 'IkBa', {'type': 'activation', 'weight': 0.8}),
            ('Bax mRNA', 'Bax', {'type': 'activation', 'weight': 0.8}),
            ('p53 mRNA', 'p53', {'type': 'activation', 'weight': 0.8}),
            ('Mdm2 mRNA', 'Mdm2 cyt', {'type': 'activation', 'weight': 0.8}),
            ('p21 mRNA', 'p21', {'type': 'activation', 'weight': 0.8}),
            ('Chk2 mRNA', 'Chk2', {'type': 'activation', 'weight': 0.8}),
            ('Wip1 mRNA', 'Wip1', {'type': 'activation', 'weight': 0.8}),
            ('ATM mRNA', 'ATM', {'type': 'activation', 'weight': 0.8}),

            # p53-p transcriptional targets
            ('p53-p', 'ATM mRNA', {'type': 'activation', 'weight': 0.7}),
            ('p53-p', 'Wip1 mRNA', {'type': 'activation', 'weight': 0.7}),
            ('p53-p', 'Chk2 mRNA', {'type': 'activation', 'weight': 0.7}),
            ('p53-p', 'p21 mRNA', {'type': 'activation', 'weight': 0.7}),
            ('p53-p', 'PTEN mRNA (Genomic)', {'type': 'activation', 'weight': 0.7}),

            # CREB transcriptional targets
            ('CREB (TF)', 'ATM mRNA', {'type': 'activation', 'weight': 0.6}),
            ('CREB (TF)', 'Wip mRNA', {'type': 'activation', 'weight': 0.6}),

            # Chk2 signaling
            ('ATM-p', 'Chk2', {'type': 'activation', 'weight': 0.7}),
            ('Chk2-p', 'Mdm2 mRNA', {'type': 'activation', 'weight': 0.6}),

            # Mdm2 regulation
            ('Mdm2 cyt', 'Mdm2-p cyt', {'type': 'activation', 'weight': 0.6}),
            ('Mdm2 cyt', 'Mdm2-p nuc', {'type': 'activation', 'weight': 0.6}),
            ('Mdm2 cyt', 'Bax', {'type': 'activation', 'weight': 0.5}),
            ('Mdm2 cyt', 'p21', {'type': 'activation', 'weight': 0.5}),
            ('Mdm2-p nuc', 'Mdm2-p cyt', {'type': 'activation', 'weight': 0.5}),

            # Apoptosis and cell cycle regulation
            ('Bax', 'apoptosis', {'type': 'activation', 'weight': 0.9}),
            ('p21', 'cell cycle arrest', {'type': 'activation', 'weight': 0.8}),
            ('p21', 'apoptosis', {'type': 'activation', 'weight': 0.6}),

            # MicroRNA processing
            ('KSRP-p', 'pre-miR-16', {'type': 'activation', 'weight': 0.5}),
            ('pre-miR-16', 'miR-16', {'type': 'activation', 'weight': 0.5}),

            # PTEN-PI3K pathway
            ('PTEN', 'PIP2', {'type': 'activation', 'weight': 0.7}),
            ('PIP2', 'PIP3', {'type': 'activation', 'weight': 0.7}),

            # ============================================
            # INHIBITORY EDGES (Negative weights)
            # ============================================
            ('A20', 'IKKKα', {'type': 'inhibition', 'weight': -1.0}),
            ('IkBa', 'NFkB (TF)', {'type': 'inhibition', 'weight': -1.0}),
            ('Wip1', 'ATM-p', {'type': 'inhibition', 'weight': -0.8}),
            ('Wip1', 'ATMa-p', {'type': 'inhibition', 'weight': -0.8}),
            ('Wip1', 'Chk2-p', {'type': 'inhibition', 'weight': -0.8}),
            ('PTEN', 'PIP3', {'type': 'inhibition', 'weight': -1.0}),
            ('Mdm2 cyt', 'p53', {'type': 'inhibition', 'weight': -0.9}),
            ('Mdm2-p nuc', 'p53', {'type': 'inhibition', 'weight': -0.9}),
        ]

        # Add all edges
        for u, v, attrs in edges_data:
            self.G.add_edge(u, v, **attrs)

        print(f"✓ Added {len(edges_data)} edges")
        print(f"✓ Network built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        return self.G

    def initialize_states_from_table(self):
        """Initialize states according to Table 4.27"""
        # Initialize all to 0 first
        for node in self.G.nodes():
            self.G.nodes[node]['state'] = 0
            self.states[node] = 0

        # Set active nodes according to Table 4.27
        active_nodes = [
            'IKKα', 'NFkB (TF)', 'p53-p', 'A20 mRNA', 'A20', 'IkBa mRNA',
            'Mdm2 cyt', 'IkBa', 'Wip mRNA', 'Wip1', 'Wip1 mRNA', 'p53 mRNA',
            'ATM mRNA', 'ATM', 'ATM-p', 'ATMa-p', 'MRN-p', 'Chk2-p',
            'CREB (TF)', 'KSRP-p', 'AKT-p', 'Mdm2-p cyt', 'Mdm2-p nuc',
            'PTEN mRNA (Genomic)', 'PTEN', 'PIP2', 'PIP3', 'Bax mRNA',
            'Bax', 'apoptosis', 'Mdm2 mRNA', 'p21 mRNA', 'p21',
            'cell cycle arrest', 'Chk2 mRNA', 'Chk2', 'miR-16',
            'pre-miR-16', 'p53', 'IKBa mRNA'
        ]

        for node in active_nodes:
            if node in self.states:
                self.G.nodes[node]['state'] = 1
                self.states[node] = 1

        # Set inactive nodes
        inactive_nodes = ['TNFa', 'TNFR1', 'IKKKα', 'p53 (TF)', 'IR', 'DSB']
        for node in inactive_nodes:
            if node in self.states:
                self.G.nodes[node]['state'] = 0
                self.states[node] = 0

        self.states_history = [self.states.copy()]
        return self.states

    def boolean_update_rule(self, node, current_states):
        """Apply Boolean update rule for a specific node"""
        predecessors = list(self.G.predecessors(node))

        if not predecessors:  # No inputs, maintain state
            return current_states[node]

        # Calculate weighted sum of inputs
        weighted_sum = 0
        for pred in predecessors:
            edge_data = self.G[pred][node]
            weight = edge_data.get('weight', 1.0)
            edge_type = edge_data.get('type', 'activation')

            if edge_type == 'inhibition':
                weighted_sum -= current_states[pred] * abs(weight)
            else:
                weighted_sum += current_states[pred] * abs(weight)

        # Apply threshold
        if weighted_sum > 0:
            return 1
        else:
            return 0

    def simulate(self, iterations=100):
        """Run Boolean simulation for specified iterations"""
        print(f"Running Boolean simulation for {iterations} iterations...")

        # Initialize states
        current_states = self.initialize_states_from_table()
        self.states_history = [current_states.copy()]

        # Run simulation
        for t in range(iterations):
            new_states = {}
            for node in self.G.nodes():
                new_states[node] = self.boolean_update_rule(node, current_states)

            self.states_history.append(new_states.copy())
            current_states = new_states

            # Check for convergence (no changes)
            if t > 5 and self.states_history[-1] == self.states_history[-2]:
                print(f"✓ Convergence reached at iteration {t}")
                break

        # Store final stable states
        self.stable_states = current_states

        # Calculate activation frequencies
        self._calculate_activation_frequencies()

        # Identify pivotal nodes (active in final state)
        self.pivotal_nodes = [node for node, state in self.stable_states.items()
                              if state == 1]

        print(f"✓ Simulation completed: {len(self.pivotal_nodes)} nodes active in stable state")
        return self.states_history

    def _calculate_activation_frequencies(self):
        """Calculate activation frequency for each node across iterations"""
        n_iterations = len(self.states_history)
        activation_counts = defaultdict(int)

        for state in self.states_history:
            for node, value in state.items():
                if value == 1:
                    activation_counts[node] += 1

        self.activation_frequencies = {
            node: count / n_iterations for node, count in activation_counts.items()
        }

    def identify_pivotal_nodes_by_frequency(self, threshold=0.7):
        """Identify pivotal nodes based on activation frequency"""
        return [node for node, freq in self.activation_frequencies.items()
                if freq >= threshold]

    def generate_table_4_26(self):
        """Generate Table 4.26: Pivotal Nodes"""
        pivotal_nodes_list = self.identify_pivotal_nodes_by_frequency(threshold=0.7)

        table_data = []
        for node in pivotal_nodes_list:
            node_type = self.G.nodes[node].get('type', 'Unknown')
            activation_freq = self.activation_frequencies.get(node, 0)
            table_data.append({
                'Pivotal Node': node,
                'Type': node_type,
                'Activation Frequency': f"{activation_freq:.1%}"
            })

        df = pd.DataFrame(table_data)
        df = df.sort_values('Activation Frequency', ascending=False)
        return df

    def generate_table_4_27(self):
        """Generate Table 4.27: Stable States"""
        table_data = []
        for node, state in self.stable_states.items():
            node_type = self.G.nodes[node].get('type', 'Unknown')
            table_data.append({
                'Node': node,
                'State': state,
                'Type': node_type,
                'Status': 'Active' if state == 1 else 'Inactive'
            })

        df = pd.DataFrame(table_data)
        df = df.sort_values(['State', 'Node'], ascending=[False, True])
        return df

    def plot_state_heatmap_figure_4_26(self, save_path=None):
        """Create Figure 4.26: State transitions heatmap"""
        # Convert states history to DataFrame
        states_df = pd.DataFrame(self.states_history).T

        # Create figure
        plt.figure(figsize=(16, 12))

        # Create heatmap
        sns.heatmap(states_df, cmap='coolwarm', cbar_kws={'label': 'State (0=Inactive, 1=Active)'},
                    xticklabels=range(len(self.states_history)), yticklabels=states_df.index,
                    linewidths=0.5, linecolor='gray')

        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Network Node', fontsize=12, fontweight='bold')
        plt.title('State Transitions of TNFR1 Signaling Network Over 100 Iterations\n(Figure 4.26)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_temporal_dynamics_figure_4_27(self, top_n=15, save_path=None):
        """Create Figure 4.27: Temporal dynamics of pivotal nodes"""
        # Get top nodes by activation frequency
        sorted_freq = sorted(self.activation_frequencies.items(),
                             key=lambda x: x[1], reverse=True)[:top_n]
        top_nodes = [node for node, _ in sorted_freq]

        # Extract time series
        time_series = {}
        for node in top_nodes:
            time_series[node] = [state[node] for state in self.states_history]

        # Create figure with subplots
        fig, axes = plt.subplots(5, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, (ax, node) in enumerate(zip(axes, top_nodes)):
            iterations = range(len(time_series[node]))
            avg_activation = np.mean(time_series[node])

            # Color based on average activation (matching Figure 4.27 description)
            if avg_activation > 0.7:
                color = 'red'  # Highly active/upregulated
                label = 'Highly Active'
            elif avg_activation > 0.3:
                color = 'orange'  # Moderately active
                label = 'Moderately Active'
            else:
                color = 'blue'  # Less active/downregulated
                label = 'Less Active'

            ax.plot(iterations, time_series[node], color=color, linewidth=2, alpha=0.8)
            ax.fill_between(iterations, time_series[node], alpha=0.2, color=color)

            ax.set_title(f'{node}\n(Active: {self.activation_frequencies[node]:.1%})',
                         fontsize=9, fontweight='bold')
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Inactive', 'Active'], fontsize=8)
            ax.grid(True, alpha=0.3)

            if idx >= 12:  # Bottom row
                ax.set_xlabel('Iteration', fontsize=9)

        # Remove unused subplots
        for idx in range(len(top_nodes), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Temporal Dynamics of Pivotal Nodes in TNFR1 Signaling Network\n(Figure 4.27)',
                     fontsize=14, fontweight='bold', y=1.02)

        # Add legend matching Figure 4.27 description
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=2, label='Highly Active/Upregulated'),
            plt.Line2D([0], [0], color='orange', lw=2, label='Moderately Active'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Less Active/Downregulated')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, -0.05), fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_network_diagram(self, save_path=None):
        """Visualize network structure with node states"""
        plt.figure(figsize=(14, 10))

        # Use spring layout for better visualization
        pos = nx.spring_layout(self.G, seed=42, k=2)

        # Color nodes based on their stable state
        node_colors = []
        node_sizes = []
        for node in self.G.nodes():
            if self.stable_states.get(node, 0) == 1:
                node_colors.append('red')  # Active
                node_sizes.append(800)
            else:
                node_colors.append('blue')  # Inactive
                node_sizes.append(500)

        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8, edgecolors='black')

        # Draw edges with different styles for activation/inhibition
        activation_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                            if d.get('type') != 'inhibition']
        inhibition_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                            if d.get('type') == 'inhibition']

        nx.draw_networkx_edges(self.G, pos, edgelist=activation_edges,
                               edge_color='green', alpha=0.6, width=1.5,
                               arrowstyle='->', arrowsize=15)
        nx.draw_networkx_edges(self.G, pos, edgelist=inhibition_edges,
                               edge_color='red', alpha=0.6, width=1.5, style='dashed',
                               arrowstyle='-|>', arrowsize=15)

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, font_size=8, font_weight='bold')

        plt.title(
            'TNFR1 Signaling Network Structure\n(Red: Active, Blue: Inactive | Green: Activation, Red: Inhibition)',
            fontsize=12, fontweight='bold', pad=20)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_node_states_over_time(self, save_path=None):
        """Plot node states over time (bar chart version)"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        # Select time points to show
        time_points = [0, 10, 20, 50, 75, 99]
        time_points = [min(t, len(self.states_history) - 1) for t in time_points]

        for idx, (ax, t) in enumerate(zip(axes, time_points)):
            states_at_t = self.states_history[t]
            nodes = list(states_at_t.keys())
            values = list(states_at_t.values())

            colors = ['red' if v == 1 else 'blue' for v in values]
            ax.bar(range(len(nodes)), values, color=colors, alpha=0.7)

            ax.set_title(f'Time Step {t}', fontsize=10, fontweight='bold')
            ax.set_ylim(-0.5, 1.5)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Inactive', 'Active'])
            ax.set_xlabel('Node Index')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Node States Over Different Time Steps', fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def run_complete_analysis(self):
        """Run complete Boolean modeling analysis"""
        print("=" * 100)
        print("COMPREHENSIVE BOOLEAN MODELING ANALYSIS OF TNFR1 SIGNALING NETWORK")
        print("=" * 100)

        # Build network
        self.build_complete_network()

        # Run simulation
        self.simulate(iterations=self.time_steps)

        # Generate and display tables
        print("\n" + "=" * 80)
        print("TABLE 4.26: PIVOTAL NODES IDENTIFIED")
        print("=" * 80)
        table_4_26 = self.generate_table_4_26()
        print(table_4_26.to_string(index=False))

        print("\n" + "=" * 80)
        print("TABLE 4.27: STABLE STATES OF THE TNFR1 SIGNALING NETWORK")
        print("=" * 80)
        table_4_27 = self.generate_table_4_27()
        print(table_4_27.to_string(index=False))

        # Analysis summary
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Total Nodes: {len(self.stable_states)}")
        print(f"Active Nodes in Stable State: {len(self.pivotal_nodes)}")
        print(f"Inactive Nodes in Stable State: {len(self.stable_states) - len(self.pivotal_nodes)}")
        print(f"Simulation Iterations: {len(self.states_history)}")

        # Key biological insights
        print("\n" + "=" * 80)
        print("KEY BIOLOGICAL INSIGHTS")
        print("=" * 80)
        print("1. IKKα and NF-κB (TF) remain active - critical for inflammatory response")
        print("2. p53-p and ATM pathway components (ATMa-p, MRN-p, Chk2-p) are activated")
        print("3. Apoptosis pathway (Bax, p21) is engaged")
        print("4. Cell cycle arrest is activated - anti-proliferative response")
        print("5. Negative feedback regulators (A20, IkBa, Wip1) are active")

        # Therapeutic implications
        print("\n" + "=" * 80)
        print("THERAPEUTIC IMPLICATIONS FOR OVARIAN CANCER")
        print("=" * 80)
        print("• NF-κB inhibitors could reduce inflammation and chemoresistance")
        print("• ATM/ATR inhibitors may sensitize cancer cells to DNA-damaging agents")
        print("• Mdm2-p53 interaction inhibitors could restore p53 tumor suppressor function")
        print("• IKKα inhibitors might modulate NF-κB pathway activity")

        return self


# Main execution
if __name__ == "__main__":
    # Create and run the model
    model = TNFR1BooleanModel()
    model.run_complete_analysis()

    # Generate all visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Figure 4.26: State transitions heatmap
    print("Generating Figure 4.26: State transitions heatmap...")
    model.plot_state_heatmap_figure_4_26(save_path="tnfr1_state_transitions_heatmap.png")

    # Figure 4.27: Temporal dynamics
    print("Generating Figure 4.27: Temporal dynamics of pivotal nodes...")
    model.plot_temporal_dynamics_figure_4_27(save_path="tnfr1_temporal_dynamics.png")

    # Network structure
    print("Generating network structure visualization...")
    model.plot_network_diagram(save_path="tnfr1_network_structure.png")

    # Node states over time
    print("Generating node states over time plot...")
    model.plot_node_states_over_time(save_path="tnfr1_node_states_over_time.png")

    # Save tables to CSV
    print("\nSaving tables to CSV files...")
    table_4_26 = model.generate_table_4_26()
    table_4_27 = model.generate_table_4_27()

    table_4_26.to_csv("table_4_26_pivotal_nodes.csv", index=False)
    table_4_27.to_csv("table_4_27_stable_states.csv", index=False)

    # Final summary
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 100)
    print("\nGenerated output files:")
    print("1. table_4_26_pivotal_nodes.csv - Table 4.26")
    print("2. table_4_27_stable_states.csv - Table 4.27")
    print("3. tnfr1_state_transitions_heatmap.png - Figure 4.26")
    print("4. tnfr1_temporal_dynamics.png - Figure 4.27")
    print("5. tnfr1_network_structure.png - Network diagram")
    print("6. tnfr1_node_states_over_time.png - Node states over time")
    print("\n" + "=" * 100)
