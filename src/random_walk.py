import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend for interactive plotting
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from networkx.algorithms import community

# Create a directed graph
G = nx.DiGraph()

# Define edges from the interaction data provided
edges = [
    ('p53', 'p53-p'), ('p53 mRNA', 'p53'), ('p53-p', 'Mdm2 mRNA'),
    ('Mdm2 cyt', 'Mdm2-p cyt'), ('Mdm2 mRNA', 'Mdm2 cyt'),
    ('Mdm2-p cyt', 'Mdm2-p nuc'), ('DSB', 'ATM-p'),
    ('ATM mRNA', 'ATM'), ('p53-p', 'ATM mRNA'),
    ('ATMa-p', 'p53-p'), ('ATMa-p', 'AKT-p'),
    ('ATMa-p', 'KSRP-p'), ('ATMa-p', 'CREB'),
    ('ATMa-p', 'Chk2-p'), ('ATM-p', 'MRN-p'),
    ('DSB', 'MRN-p'), ('CREB', 'ATM mRNA'),
    ('MRN-p', 'ATMa-p'), ('CREB', 'Wip1 mRNA'),
    ('p53-p', 'Chk2 mRNA'), ('p53-p', 'Bax mRNA'),
    ('p53-p', 'p21 mRNA'), ('p53-p', 'PTEN mRNA'),
    ('p53-p', 'Wip1 mRNA'), ('Wip1 mRNA', 'Wip1'),
    ('pre-miR-16', 'miR-16'), ('KSRP-p', 'pre-miR-16'),
    ('Chk2 mRNA', 'Chk2'), ('Chk2-p', 'p53-p'),
    ('Bax mRNA', 'Bax'), ('Bax', 'apoptosis'),
    ('p21 mRNA', 'p21'), ('p21', 'cell cycle arrest'),
    ('IR', 'DSB'), ('p53-p', 'PTEN mRNA'),
    ('PTEN mRNA', 'PTEN'), ('PTEN', 'PIP2'),
    ('PIP2', 'PIP3'), ('PIP3', 'AKT-p'),
    ('AKT-p', 'Mdm2-p cyt'), ('TNFa', 'TNFR1'),
    ('TNFR1', 'IKKKa'), ('IKKKa', 'IKKa'),
    ('A20 mRNA', 'A20 cyt'), ('IKKa', 'NFkB'),
    ('NFkB', 'IkBa mRNA'), ('NFkB', 'A20 mRNA'),
    ('NFkB', 'p53 mRNA'), ('IkBa mRNA', 'IkBa'),
    ('NFkB', 'Wip1 mRNA')
]

G.add_edges_from(edges)

# Assign random weights to edges
for u, v in G.edges():
    G[u][v]['weight'] = np.random.rand()

# Random Walk Function
def random_walk(G, start_node, num_steps):
    current_node = start_node
    visiting_counts = {node: 0 for node in G.nodes()}

    for _ in range(num_steps):
        visiting_counts[current_node] += 1
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        weights = [G[current_node][neighbor]['weight'] for neighbor in neighbors]
        current_node = np.random.choice(neighbors, p=np.array(weights) / sum(weights))

    return visiting_counts

# Parameters for the random walk
num_walks = 1000
num_steps = 50
start_node = 'p53'

# Perform random walks
total_visiting_counts = {node: 0 for node in G.nodes()}
convergence_data = []

for i in range(num_walks):
    counts = random_walk(G, start_node, num_steps)
    for node in total_visiting_counts:
        total_visiting_counts[node] += counts[node]

    # Record convergence data
    unique_nodes_visited = sum(1 for count in total_visiting_counts.values() if count > 0)
    convergence_data.append(unique_nodes_visited / len(G.nodes()))

# Normalize visit counts
for node in total_visiting_counts:
    total_visiting_counts[node] /= num_walks

# Plot Convergence Rate
def plot_convergence_rate(convergence_data):
    plt.figure(figsize=(10, 5))
    plt.plot(convergence_data, marker='o')
    plt.title('Convergence Rate of Random Walks', fontsize=14)
    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('Proportion of Unique Nodes Visited', fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig('convergence_rate.png')
    plt.show()

# Function to plot the biological interaction network
def plot_biological_network(G, visiting_counts):
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Node sizes based on visiting counts
    node_sizes = [500 * visiting_counts[node] + 50 for node in G.nodes()]

    # Create a ScalarMappable for the color normalization
    norm = plt.Normalize(vmin=0, vmax=max(total_visiting_counts.values()))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Draw nodes with improved aesthetics
    node_colors = [sm.to_rgba(visiting_counts[node]) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='black', ax=ax)

    # Draw edges with colors based on weights
    edge_colors = [plt.cm.Blues(G[u][v]['weight']) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.8, style='solid', width=2, ax=ax)

    # Add labels for nodes
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black', ax=ax)

    # Add labels for edges with weights
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax, label='Visit Count (normalized)', shrink=0.8, pad=0.02)

    plt.title('Enhanced Visualization of the Biological Interaction Network', fontsize=16)
    plt.axis('off')
    plt.savefig('biological_network.png')
    plt.show()

# Function to plot normalized visit counts
def plot_visit_counts(visiting_counts):
    plt.figure(figsize=(12, 6))
    nodes = list(visiting_counts.keys())
    counts = list(visiting_counts.values())
    
    plt.bar(nodes, counts, color='royalblue')
    plt.xlabel('Nodes', fontsize=12)
    plt.ylabel('Normalized Visit Count', fontsize=12)
    plt.title('Normalized Visit Counts per Node', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('normalized_visit_counts.png')
    plt.show()

# Function to plot edge weights distribution
def plot_edge_weights_distribution(G):
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    plt.figure(figsize=(12, 6))
    plt.hist(edge_weights, bins=20, color='lightcoral', edgecolor='black')
    plt.xlabel('Edge Weight', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Edge Weights', fontsize=14)
    plt.tight_layout()
    plt.savefig('edge_weights_distribution.png')
    plt.show()

# Function to plot community sizes
def plot_community_sizes(G):
    communities = community.greedy_modularity_communities(G)
    community_sizes = [len(comm) for comm in communities]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(community_sizes)), community_sizes, color='mediumseagreen')
    plt.xlabel('Community Index', fontsize=12)
    plt.ylabel('Community Size', fontsize=12)
    plt.title('Sizes of Detected Communities', fontsize=14)
    plt.tight_layout()
    plt.savefig('community_sizes.png')
    plt.show()

# Run all plots
plot_convergence_rate(convergence_data)
plot_biological_network(G, total_visiting_counts)
plot_visit_counts(total_visiting_counts)
plot_edge_weights_distribution(G)
plot_community_sizes(G)
