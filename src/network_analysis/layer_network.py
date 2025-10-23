import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Create the network
G = nx.DiGraph()

# Add edges from your data
edges = [
    ("p53", "p53-p", "phosphorylation"),
    ("p53 mRNA", "p53", "translation"),
    ("p53-p", "Mdm2 mRNA", "transcription"),
    ("Mdm2 cyt", "Mdm2-p cyt", "phosphorylation"),
    ("Mdm2 mRNA", "Mdm2 cyt", "translation"),
    ("Mdm2-p cyt", "Mdm2-p nuc", "translocation"),
    ("DSB", "ATM-p", "activation"),
    ("ATM mRNA", "ATM", "translation"),
    ("p53-p", "ATM mRNA", "transcription"),
    ("ATMa-p", "p53-p", "phosphorylation"),
    ("ATMa-p", "AKT-p", "activation"),
    ("ATMa-p", "KSRP-p", "phosphorylation"),
    ("ATMa-p", "CREB", "phosphorylation"),
    ("ATMa-p", "Chk2-p", "phosphorylation"),
    ("ATM-p", "MRN-p", "recruitment"),
    ("DSB", "MRN-p", "recruitment"),
    ("CREB", "ATM mRNA", "transcription"),
    ("MRN-p", "ATMa-p", "activation"),
    ("CREB", "Wip1 mRNA", "transcription"),
    ("p53-p", "Chk2 mRNA", "transcription"),
    ("p53-p", "Bax mRNA", "transcription"),
    ("p53-p", "p21 mRNA", "transcription"),
    ("p53-p", "PTEN mRNA", "transcription"),
    ("p53-p", "Wip1 mRNA", "transcription"),
    ("Wip1 mRNA", "Wip1", "translation"),
    ("pre-miR-16", "miR-16", "processing"),
    ("KSRP-p", "pre-miR-16", "binding"),
    ("Chk2 mRNA", "Chk2", "translation"),
    ("Chk2-p", "p53-p", "phosphorylation"),
    ("Bax mRNA", "Bax", "translation"),
    ("Bax", "apoptosis", "activation"),
    ("p21 mRNA", "p21", "translation"),
    ("p21", "cell cycle arrest", "activation"),
    ("IR", "DSB", "induction"),
    ("p53-p", "PTEN mRNA", "transcription"),
    ("PTEN mRNA", "PTEN", "translation"),
    ("PTEN", "PIP2", "regulation"),
    ("PIP2", "PIP3", "conversion"),
    ("PIP3", "AKT-p", "activation"),
    ("AKT-p", "Mdm2-p cyt", "phosphorylation"),
    ("TNFa", "TNFR1", "binding"),
    ("TNFR1", "IKKKa", "activation"),
    ("IKKKa", "IKKa", "phosphorylation"),
    ("A20 mRNA", "A20 cyt", "translation"),
    ("IKKa", "NFkB", "activation"),
    ("NFkB", "IkBa mRNA", "transcription"),
    ("NFkB", "A20 mRNA", "transcription"),
    ("NFkB", "p53 mRNA", "transcription"),
    ("IkBa mRNA", "IkBa", "translation"),
    ("NFkB", "Wip1 mRNA", "transcription")
]

for source, target, interaction in edges:
    G.add_edge(source, target, interaction=interaction)

# Set up the figure with larger size
plt.figure(figsize=(24, 18))

# Define node positions using a layout algorithm with more space
pos = nx.spring_layout(G, k=1.5, seed=42, iterations=200, scale=2)

# Manually adjust overlapping nodes if needed
def adjust_overlaps(pos, min_distance=0.2):
    nodes = list(pos.keys())
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            dist = np.linalg.norm(np.array(pos[node1]) - np.array(pos[node2]))
            if dist < min_distance:
                # Move node2 away from node1
                direction = np.array(pos[node2]) - np.array(pos[node1])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                pos[node2] += direction * (min_distance - dist)
    return pos

pos = adjust_overlaps(pos)

# Define node colors and shapes based on function
node_colors = []
node_shapes = []
for node in G.nodes():
    if 'mRNA' in node:
        node_colors.append('#FFA07A')  # Light salmon
        node_shapes.append('s')  # Square
    elif node in ['apoptosis', 'cell cycle arrest']:
        node_colors.append('#9370DB')  # Medium purple
        node_shapes.append('*')  # Star
    elif 'p53' in node or 'ATM' in node or 'Chk2' in node:
        node_colors.append('#4682B4')  # Steel blue
        node_shapes.append('o')  # Circle
    elif 'Mdm2' in node or 'Wip1' in node:
        node_colors.append('#32CD32')  # Lime green
        node_shapes.append('d')  # Diamond
    elif 'NFkB' in node or 'IKK' in node:
        node_colors.append('#FF6347')  # Tomato red
        node_shapes.append('h')  # Hexagon
    elif 'AKT' in node or 'PTEN' in node or 'PIP' in node:
        node_colors.append('#FFD700')  # Gold
        node_shapes.append('p')  # Pentagon
    else:
        node_colors.append('#D3D3D3')  # Light gray
        node_shapes.append('o')  # Circle

# Define edge colors and widths based on interaction type
edge_colors = []
edge_widths = []
for u, v, data in G.edges(data=True):
    if data['interaction'] == 'phosphorylation':
        edge_colors.append('#FF0000')  # Red
        edge_widths.append(3.0)
    elif data['interaction'] == 'transcription':
        edge_colors.append('#0000FF')  # Blue
        edge_widths.append(2.5)
    elif data['interaction'] == 'translation':
        edge_colors.append('#008000')  # Green
        edge_widths.append(2.5)
    elif data['interaction'] == 'activation':
        edge_colors.append('#FFA500')  # Orange
        edge_widths.append(3.0)
    else:
        edge_colors.append('#A9A9A9')  # Dark gray
        edge_widths.append(2.0)

# Draw nodes with different shapes
for shape in set(node_shapes):
    nodes = [node for node, s in zip(G.nodes(), node_shapes) if s == shape]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                          node_shape=shape,
                          node_color=[node_colors[list(G.nodes()).index(node)] for node in nodes],
                          node_size=2000, alpha=0.9)

# Draw labels with white background for better readability
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round'))

# Draw edges with different styles
for i, (u, v, data) in enumerate(G.edges(data=True)):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                          width=edge_widths[i],
                          edge_color=edge_colors[i],
                          arrowsize=25,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.2')

# Create legends
node_legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='DNA Damage Response',
           markerfacecolor='#4682B4', markersize=15),
    Line2D([0], [0], marker='s', color='w', label='mRNAs',
           markerfacecolor='#FFA07A', markersize=15),
    Line2D([0], [0], marker='*', color='w', label='Phenotypes',
           markerfacecolor='#9370DB', markersize=15),
    Line2D([0], [0], marker='d', color='w', label='Regulators',
           markerfacecolor='#32CD32', markersize=15),
    Line2D([0], [0], marker='h', color='w', label='NFkB Pathway',
           markerfacecolor='#FF6347', markersize=15),
    Line2D([0], [0], marker='p', color='w', label='PI3K/AKT Pathway',
           markerfacecolor='#FFD700', markersize=15)
]

edge_legend_elements = [
    Line2D([0], [0], color='#FF0000', lw=3, label='Phosphorylation'),
    Line2D([0], [0], color='#0000FF', lw=2.5, label='Transcription'),
    Line2D([0], [0], color='#008000', lw=2.5, label='Translation'),
    Line2D([0], [0], color='#FFA500', lw=3, label='Activation'),
    Line2D([0], [0], color='#A9A9A9', lw=2, label='Other')
]

# Add legends with larger font
plt.legend(handles=node_legend_elements, loc='upper left',
           bbox_to_anchor=(1, 1), fontsize=12, title='Node Types', title_fontsize=13)
plt.legend(handles=edge_legend_elements, loc='lower left',
           bbox_to_anchor=(1, 0), fontsize=12, title='Interaction Types', title_fontsize=13)

plt.title("Comprehensive p53 Signaling Network", fontsize=20, pad=20)
plt.tight_layout()
plt.axis('off')

# Save high-quality image
plt.savefig('p53_signaling_network_improved.png', dpi=600, bbox_inches='tight', transparent=True)
plt.show()

plt.figure(figsize=(20, 20))
pos = nx.circular_layout(G, scale=2)

# Highlight p53 as the central node
node_sizes = [3000 if node == 'p53' else 1500 for node in G.nodes()]
node_colors = ['gold' if node == 'p53' else '#4682B4' for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

# Draw edges with varying curvature
for i, (u, v, data) in enumerate(G.edges(data=True)):
    rad = 0.2 if u == 'p53' or v == 'p53' else 0.1
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                         width=2.5, edge_color='#555555',
                         arrowsize=20, arrowstyle='->',
                         connectionstyle=f'arc3,rad={rad}')

plt.title("Circular p53 Network Hierarchy", fontsize=18)
plt.tight_layout()
plt.axis('off')
plt.savefig('p53_circular_hierarchy.png', dpi=300, transparent=True)
plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create the graph
G = nx.DiGraph()

# Add all nodes and edges from your data
edges = [
    ('p53', 'p53-p', 'interacts_with'),
    ('p53 mRNA', 'p53', 'interacts_with'),
    ('p53-p', 'Mdm2 mRNA', 'interacts_with'),
    ('Mdm2 cyt', 'Mdm2-p cyt', 'interacts_with'),
    ('Mdm2 mRNA', 'Mdm2 cyt', 'interacts_with'),
    ('Mdm2-p cyt', 'Mdm2-p nuc', 'interacts_with'),
    ('DSB', 'ATM-p', 'interacts_with'),
    ('ATM mRNA', 'ATM', 'interacts_with'),
    ('p53-p', 'ATM mRNA', 'interacts_with'),
    ('ATMa-p', 'p53-p', 'interacts_with'),
    ('ATMa-p', 'AKT-p', 'interacts_with'),
    ('ATMa-p', 'KSRP-p', 'interacts_with'),
    ('ATMa-p', 'CREB', 'interacts_with'),
    ('ATMa-p', 'Chk2-p', 'interacts_with'),
    ('ATM-p', 'MRN-p', 'interacts_with'),
    ('DSB', 'MRN-p', 'interacts_with'),
    ('CREB', 'ATM mRNA', 'interacts_with'),
    ('MRN-p', 'ATMa-p', 'interacts_with'),
    ('CREB', 'Wip1 mRNA', 'interacts_with'),
    ('p53-p', 'Chk2 mRNA', 'interacts_with'),
    ('p53-p', 'Bax mRNA', 'interacts_with'),
    ('p53-p', 'p21 mRNA', 'interacts_with'),
    ('p53-p', 'PTEN mRNA', 'interacts_with'),
    ('p53-p', 'Wip1 mRNA', 'interacts_with'),
    ('Wip1 mRNA', 'Wip1', 'interacts_with'),
    ('pre-miR-16', 'miR-16', 'interacts_with'),
    ('KSRP-p', 'pre-miR-16', 'interacts_with'),
    ('Chk2 mRNA', 'Chk2', 'interacts_with'),
    ('Chk2-p', 'p53-p', 'interacts_with'),
    ('Bax mRNA', 'Bax', 'interacts_with'),
    ('Bax', 'apoptosis', 'interacts_with'),
    ('p21 mRNA', 'p21', 'interacts_with'),
    ('p21', 'cell cycle arrest', 'interacts_with'),
    ('IR', 'DSB', 'interacts_with'),
    ('p53-p', 'PTEN mRNA', 'interacts_with'),
    ('PTEN mRNA', 'PTEN', 'interacts_with'),
    ('PTEN', 'PIP2', 'interacts_with'),
    ('PIP2', 'PIP3', 'interacts_with'),
    ('PIP3', 'AKT-p', 'interacts_with'),
    ('AKT-p', 'Mdm2-p cyt', 'interacts_with'),
    ('TNFa', 'TNFR1', 'interacts_with'),
    ('TNFR1', 'IKKKa', 'interacts_with'),
    ('IKKKa', 'IKKa', 'interacts_with'),
    ('A20 mRNA', 'A20 cyt', 'interacts_with'),
    ('IKKa', 'NFkB', 'interacts_with'),
    ('NFkB', 'IkBa mRNA', 'interacts_with'),
    ('NFkB', 'A20 mRNA', 'interacts_with'),
    ('NFkB', 'p53 mRNA', 'interacts_with'),
    ('IkBa mRNA', 'IkBa', 'interacts_with'),
    ('NFkB', 'Wip1 mRNA', 'interacts_with')
]

for source, target, interaction in edges:
    G.add_edge(source, target, interaction=interaction)

# Add any isolated nodes if needed
all_nodes = {
    'p53', 'p53-p', 'p53 mRNA', 'Mdm2 mRNA', 'Mdm2 cyt', 'Mdm2-p cyt',
    'Mdm2-p nuc', 'DSB', 'ATM-p', 'ATM mRNA', 'ATM', 'ATMa-p', 'AKT-p',
    'KSRP-p', 'CREB', 'Chk2-p', 'MRN-p', 'Chk2 mRNA', 'Chk2', 'Bax mRNA',
    'Bax', 'apoptosis', 'p21 mRNA', 'p21', 'cell cycle arrest', 'IR',
    'PTEN mRNA', 'PTEN', 'PIP2', 'PIP3', 'TNFa', 'TNFR1', 'IKKKa', 'IKKa',
    'A20 mRNA', 'A20 cyt', 'NFkB', 'IkBa mRNA', 'IkBa', 'Wip1 mRNA', 'Wip1',
    'pre-miR-16', 'miR-16'
}

for node in all_nodes:
    if node not in G:
        G.add_node(node)

# Visualization 1: Biochemical Pathway Flowchart
plt.figure(figsize=(30, 18))
pos1 = {
    # Input signals
    'IR': (0, 10), 'TNFa': (0, 8),
    # DNA damage response
    'DSB': (3, 11), 'ATM-p': (6, 11), 'MRN-p': (4.5, 12), 'ATMa-p': (9, 11),
    # p53 core
    'p53': (12, 10), 'p53-p': (15, 10),
    # Downstream effectors
    'Mdm2 mRNA': (15, 8), 'Bax mRNA': (15, 12), 'p21 mRNA': (18, 10),
    # Terminal nodes
    'apoptosis': (21, 12), 'cell cycle arrest': (21, 8),
    # Additional important nodes
    'NFkB': (9, 7), 'IKKa': (6, 7), 'TNFR1': (3, 7),
    'AKT-p': (12, 6), 'PIP3': (9, 5), 'PTEN': (6, 4),
    'CREB': (15, 13), 'Chk2-p': (12, 13), 'Wip1 mRNA': (18, 13)
}

# Position remaining nodes programmatically
y_pos = 3
x_pos = 15
for node in G.nodes():
    if node not in pos1:
        pos1[node] = (x_pos, y_pos)
        y_pos -= 0.8
        if y_pos < -5:
            y_pos = 3
            x_pos += 3

nx.draw_networkx_nodes(G, pos1, node_size=2500, node_color='#2e8b57', alpha=0.9)
nx.draw_networkx_labels(G, pos1, font_size=8, font_weight='bold')

for u, v, data in G.edges(data=True):
    style = '-' if data['interaction'] != 'inhibition' else '--'
    nx.draw_networkx_edges(G, pos1, edgelist=[(u, v)],
                           width=2.5, edge_color='#333333',
                           arrowsize=25, arrowstyle='->',
                           style=style)

plt.title("Biochemical Flowchart of p53 Signaling", fontsize=20)
plt.tight_layout()
plt.axis('off')
plt.savefig('p53_biochemical_flow.png', dpi=300, transparent=True)
plt.show()

# Visualization 2: Molecular Interaction Constellation
plt.figure(figsize=(22, 22))
pos2 = {}
center = (0, 0)
angle = 0
angle_step = 2 * np.pi / len(G.nodes())

# Position p53 at center
pos2['p53'] = center

# Arrange main functional groups
functional_groups = {
    'DNA Damage': ['DSB', 'ATM-p', 'MRN-p', 'ATMa-p', 'ATM', 'ATM mRNA', 'Chk2-p', 'Chk2'],
    'Transcriptional': ['p53-p', 'Mdm2 mRNA', 'Bax mRNA', 'p21 mRNA', 'PTEN mRNA', 'Wip1 mRNA'],
    'NFkB Pathway': ['TNFa', 'TNFR1', 'IKKKa', 'IKKa', 'NFkB', 'A20 mRNA', 'IkBa mRNA'],
    'AKT Pathway': ['AKT-p', 'PIP3', 'PIP2', 'PTEN'],
    'Outcomes': ['apoptosis', 'cell cycle arrest', 'p21', 'Bax']
}

# Position functional groups
group_angle = 0
group_step = 2 * np.pi / len(functional_groups)
for group, nodes in functional_groups.items():
    r = 8
    pos2[group] = (r * np.cos(group_angle), r * np.sin(group_angle))
    for i, node in enumerate(nodes):
        sub_angle = group_angle + (i - len(nodes) / 2) * 0.3
        pos2[node] = ((r - 3) * np.cos(sub_angle), (r - 3) * np.sin(sub_angle))
    group_angle += group_step

# Position remaining nodes
remaining = [n for n in G.nodes() if n not in pos2]
for i, node in enumerate(remaining):
    r = 5 + (i % 2) * 3
    pos2[node] = (r * np.cos(i * angle_step), r * np.sin(i * angle_step))

# Draw with glow effect
nx.draw_networkx_nodes(G, pos2, node_size=1800,
                       node_color=['gold' if n == 'p53' else '#4169e1' for n in G.nodes()],
                       alpha=0.8, edgecolors='white', linewidths=2)

nx.draw_networkx_labels(G, pos2, font_size=8, font_weight='bold')

# Draw curved edges with varying opacities
for u, v, data in G.edges(data=True):
    alpha = 0.7 if 'mRNA' in u or 'mRNA' in v else 0.4
    nx.draw_networkx_edges(G, pos2, edgelist=[(u, v)],
                           width=2, edge_color='#ff1493',
                           arrowsize=20, arrowstyle='->',
                           alpha=alpha,
                           connectionstyle='arc3,rad=0.3')

plt.title("Molecular Constellation of p53 Network", fontsize=18)
plt.tight_layout()
plt.axis('off')
plt.savefig('p53_constellation.png', dpi=300, transparent=True)
plt.show()

# Visualization 3: Temporal Cascade Timeline
plt.figure(figsize=(28, 12))

# Define temporal layers
time_steps = {
    'Initiation': ['IR', 'TNFa', 'DSB'],
    'Early Signaling': ['ATM-p', 'MRN-p', 'TNFR1', 'IKKKa', 'PIP2'],
    'Core Response': ['ATMa-p', 'p53-p', 'IKKa', 'NFkB', 'AKT-p'],
    'Transcriptional': ['Mdm2 mRNA', 'Bax mRNA', 'p21 mRNA', 'PTEN mRNA', 'Wip1 mRNA', 'A20 mRNA'],
    'Protein Activity': ['Mdm2-p cyt', 'Bax', 'p21', 'PTEN', 'Wip1'],
    'Outcomes': ['apoptosis', 'cell cycle arrest']
}

# Position nodes in temporal lanes
pos3 = {}
y_levels = {'Initiation': 0, 'Early Signaling': 2, 'Core Response': 4,
            'Transcriptional': 6, 'Protein Activity': 8, 'Outcomes': 10}

for step, nodes in time_steps.items():
    x_positions = np.linspace(0, 25, len(nodes) + 2)[1:-1]
    for i, node in enumerate(nodes):
        pos3[node] = (x_positions[i], y_levels[step])

# Position remaining nodes
other_nodes = [n for n in G.nodes() if n not in pos3]
for i, node in enumerate(other_nodes):
    pos3[node] = (12 + i % 7, 1 + (i % 5))

# Draw with timeline
nx.draw_networkx_nodes(G, pos3, node_size=1800, node_color='#8a2be2', alpha=0.8)
nx.draw_networkx_labels(G, pos3, font_size=8, font_weight='bold')

# Draw horizontal edges for timeline flow
for u, v, data in G.edges(data=True):
    y_diff = abs(pos3[u][1] - pos3[v][1])
    nx.draw_networkx_edges(G, pos3, edgelist=[(u, v)],
                           width=2, edge_color='#20b2aa',
                           arrowsize=20, arrowstyle='->',
                           connectionstyle=f'arc3,rad={0.1 * y_diff}')

# Add timeline labels
for step, y in y_levels.items():
    plt.axhline(y=y - 0.5, color='gray', linestyle='--', alpha=0.3)
    plt.text(-2, y, step, ha='right', va='center', fontsize=12)

plt.title("Temporal Cascade of p53 Signaling", fontsize=20)
plt.tight_layout()
plt.axis('off')
plt.savefig('p53_temporal_cascade.png', dpi=300, transparent=True)
plt.show()

# Visualization 4: Interactive-Inspired Radial Network
plt.figure(figsize=(22, 22))

# Create radial layout with p53 at center
pos4 = {'p53': (0, 0)}
main_branches = ['DNA Damage', 'NFkB Pathway', 'AKT Pathway', 'Apoptosis', 'Cell Cycle']
angles = np.linspace(0, 2 * np.pi, len(main_branches), endpoint=False)

# Define nodes for each branch
branch_nodes = {
    'DNA Damage': ['DSB', 'ATM-p', 'MRN-p', 'ATMa-p', 'ATM', 'Chk2-p'],
    'NFkB Pathway': ['TNFa', 'TNFR1', 'IKKKa', 'IKKa', 'NFkB', 'A20 mRNA'],
    'AKT Pathway': ['AKT-p', 'PIP3', 'PIP2', 'PTEN', 'PTEN mRNA'],
    'Apoptosis': ['Bax mRNA', 'Bax', 'apoptosis'],
    'Cell Cycle': ['p21 mRNA', 'p21', 'cell cycle arrest', 'Wip1 mRNA']
}

# Position main pathway nodes
for i, branch in enumerate(main_branches):
    r = 7
    pos4[branch] = (r * np.cos(angles[i]), r * np.sin(angles[i]))

    # Position related nodes around each branch
    for j, node in enumerate(branch_nodes.get(branch, [])):
        sub_angle = angles[i] + np.pi / 8 * (j - len(branch_nodes[branch]) / 2)
        pos4[node] = ((r + 4) * np.cos(sub_angle), (r + 4) * np.sin(sub_angle))

# Position remaining nodes in outer ring
remaining = [n for n in G.nodes() if n not in pos4]
angle_step = 2 * np.pi / len(remaining)
for i, node in enumerate(remaining):
    pos4[node] = (10 * np.cos(i * angle_step), 10 * np.sin(i * angle_step))

# Draw with radial aesthetics
nx.draw_networkx_nodes(G, pos4, node_size=[2500 if n == 'p53' else 1200 for n in G.nodes()],
                       node_color=['red' if n == 'p53' else 'orange' for n in G.nodes()])

nx.draw_networkx_labels(G, pos4, font_size=8, font_weight='bold')

# Draw radial edges
for u, v, data in G.edges(data=True):
    edge_rad = 0.3 if 'p53' in u or 'p53' in v else 0.1
    nx.draw_networkx_edges(G, pos4, edgelist=[(u, v)],
                           width=2, edge_color='#4682b4',
                           arrowsize=20, arrowstyle='->',
                           connectionstyle=f'arc3,rad={edge_rad}')

# Highlight main pathways
for branch in main_branches:
    if branch in pos4:
        plt.scatter(pos4[branch][0], pos4[branch][1], s=3000,
                    facecolors='none', edgecolors='gold', linewidths=2)

plt.title("Radial View of p53 Signaling Network", fontsize=20)
plt.tight_layout()
plt.axis('off')
plt.savefig('p53_radial_view.png', dpi=300, transparent=True)
plt.show()
