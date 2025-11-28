import matplotlib.pyplot as plt
import random as rd
import networkx as nx
import numpy as np

# Define Graph
G = nx.DiGraph()

# Add Nodes (Part 1)
nodes_data = [
    ('TNFa', {'property': "trans", 'weight': rd.randint(0, 1)}),
    ('TNFR1', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('IKKKa', {'property': "pro", 'weight': rd.randint(0, 1)}),
    ('IKKa', {'property': "pro", 'weight': rd.randint(0, 1)}),
    ('NFkB (TF)', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('p53 (TF)', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('p53-p', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('A20 mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('A20', {'property': "pro", 'weight': rd.randint(0, 1)}),
    ('IkBa mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('Mdm2 cyt', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('IkBa', {'property': "pro", 'weight': rd.randint(0, 1)}),
    ('Wip mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('Wip1', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('Wip1 mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('p53 mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('ATM mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('ATM', {'property': 'gen', 'weight': rd.randint(0, 1)}),
    ('ATM-p', {'property': 'gen', 'weight': rd.randint(0, 1)}),
    ('ATMa-p', {'property': 'gen', 'weight': rd.randint(0, 1)}),
    ('MRN-p', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('Chk2-p', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('CREB (TF)', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('KSRP-p', {'property': 'prot', 'weight': rd.randint(0, 1)}),
    ('AKT-p', {'property': 'prot', 'weight': rd.randint(0, 1)}),
    ('Mdm2-p cyt', {'property': 'prot', 'weight': rd.randint(0, 1)}),
    ('Mdm2-p nuc', {'property': 'prot', 'weight': rd.randint(0, 1)}),
    ('PTEN mRNA (Genomic)', {'property': 'gen', 'weight': rd.randint(0, 1)}),
    ('PTEN', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('PIP2', {'property': "lipo", 'weight': rd.randint(0, 1)}),
    ('PIP3', {'property': "lipo", 'weight': rd.randint(0, 1)}),
    ('Bax mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('Bax', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('apoptosis', {'property': "gen", 'weight': rd.randint(0, 1)}),
    ('Mdm2 mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('p21 mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('p21', {'property': 'prot', 'weight': rd.randint(0, 1)}),
    ('cell cycle arrest', {'property': "trans", 'weight': rd.randint(0, 1)}),
    ('Chk2 mRNA', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('Chk2', {'property': 'pro', 'weight': rd.randint(0, 1)}),
    ('IR', {'property': 'gen', 'weight': rd.randint(0, 1)}),
    ('DSB', {'property': "gen", 'weight': rd.randint(0, 1)}),
    ('miR-16', {'property': 'trans', 'weight': rd.randint(0, 1)}),
    ('pre-miR-16', {'property': 'trans', 'weight': rd.randint(0, 1)}),
]

G.add_nodes_from(nodes_data)

# Initialize 'weight' attribute for all nodes
for node, data in nodes_data:
    G.nodes[node]['weight'] = data['weight']

# Add Edges (Part 1 and 2)
edges_data = [
    ('TNFa', 'TNFR1', {'weight': rd.randint(0, 1)}),
    ('TNFR1', 'IKKKa', {'weight': rd.randint(0, 1)}),
    ('IKKKa', 'IKKa', {'weight': rd.randint(0, 1)}),
    ('IKKa', 'NFkB (TF)', {'weight': rd.randint(0, 1)}),
    ('TNFR1', 'NFkB (TF)', {'weight': rd.randint(0, 1)}),
    ('ATMa-p', 'IKKa', {'weight': rd.randint(0, 1)}),
    ('p53 (TF)', 'p53-p', {'weight': rd.randint(0, 1)}),
    ('PIP2', 'PIP3', {'weight': rd.randint(0, 1)}),
    ('p53 mRNA', 'p53', {'weight': rd.randint(0, 1)}),
    ('DSB', 'ATM-p', {'weight': rd.randint(0, 1)}),
    ('p53-p', 'ATM mRNA', {'weight': rd.randint(0, 1)}),
    ('ATMa-p', 'p53-p', {'weight': rd.randint(0, 1)}),
    ('ATMa-p', 'AKT-p', {'weight': rd.randint(0, 1)}),
    ('ATMa-p', 'KSRP-p', {'weight': rd.randint(0, 1)}),
    ('ATMa-p', 'CREB (TF)', {'weight': rd.randint(0, 1)}),
    ('ATMa-p', 'Chk2-p', {'weight': rd.randint(0, 1)}),
    ('ATM-p', 'MRN-p', {'weight': rd.randint(0, 1)}),
    ('DSB', 'MRN-p', {'weight': rd.randint(0, 1)}),
    ('CREB (TF)', 'ATM mRNA', {'weight': rd.randint(0, 1)}),
    ('MRN-p', 'ATMa-p', {'weight': rd.randint(0, 1)}),
    ('CREB (TF)', 'Wip mRNA', {'weight': rd.randint(0, 1)}),
    ('p53-p', 'Chk2 mRNA', {'weight': rd.randint(0, 1)}),
    ('p53-p', 'p21 mRNA', {'weight': rd.randint(0, 1)}),
    ('p53-p', 'PTEN mRNA (Genomic)', {'weight': rd.randint(0, 1)}),
    ('p53-p', 'Wip1 mRNA', {'weight': rd.randint(0, 1)}),
    ('Wip1 mRNA', 'Wip1', {'weight': rd.randint(0, 1)}),
    ('KSRP-p', 'pre-miR-16', {'weight': rd.randint(0, 1)}),
    ('ATM-p', 'Chk2', {'weight': rd.randint(0, 1)}),
    ('Chk2-p', 'Mdm2 mRNA', {'weight': rd.randint(0, 1)}),
    ('Mdm2 mRNA', 'Mdm2 cyt', {'weight': rd.randint(0, 1)}),
    ('Mdm2 cyt', 'Mdm2-p nuc', {'weight': rd.randint(0, 1)}),
    ('Mdm2 cyt', 'Bax', {'weight': rd.randint(0, 1)}),
    ('Mdm2 cyt', 'Mdm2-p cyt', {'weight': rd.randint(0, 1)}),
    ('Mdm2 cyt', 'p21', {'weight': rd.randint(0, 1)}),
    ('p21', 'cell cycle arrest', {'weight': rd.randint(0, 1)}),
    ('p21', 'apoptosis', {'weight': rd.randint(0, 1)}),
    ('Bax', 'apoptosis', {'weight': rd.randint(0, 1)}),
    ('Mdm2-p nuc', 'Mdm2-p cyt', {'weight': rd.randint(0, 1)}),
    ('PTEN', 'PIP2', {'weight': rd.randint(0, 1)}),
]

G.add_edges_from(edges_data)

# Initialize edge weights
for u, v, data in G.edges(data=True):
    data['weight'] = rd.randint(0, 1)

# Run simulation
time_steps = 10
states = {}
node_states = {node: rd.choice([0, 1]) for node in G.nodes}

for t in range(time_steps):
    states[t] = node_states.copy()

    for node in G.nodes:
        inputs = list(G.predecessors(node))
        total_input = sum(node_states[input_node] * G[input_node][node]['weight'] for input_node in inputs)
        if total_input > 0:
            node_states[node] = 1
        else:
            node_states[node] = 0

# Visualization
for t in range(time_steps):
    plt.bar(states[t].keys(), states[t].values(), label=f'Time {t}')
    plt.ylim(-0.5, 1.5)
    plt.title('Node States Over Time')
    plt.xlabel('Nodes')
    plt.ylabel('State')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Identify pivotal nodes - CORRECTED FUNCTION
def identify_pivotal_nodes(G):
    pivotal_nodes = []
    for node in G.nodes:
        # Calculate total input weight by summing weights of all incoming edges
        total_input = sum(G[u][node]['weight'] for u in G.predecessors(node))
        if total_input > 0:
            pivotal_nodes.append(node)
    return pivotal_nodes


pivotal_nodes = identify_pivotal_nodes(G)
print("Pivotal nodes:", pivotal_nodes)

# Plot network dynamics
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
plt.title("Network Dynamics")
plt.show()
