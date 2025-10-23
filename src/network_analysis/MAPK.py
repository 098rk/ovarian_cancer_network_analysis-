import networkx as nx
import matplotlib.pyplot as plt
import random

# Create expanded Cell Cycle Network
G_cell_cycle = nx.DiGraph()

# Core Cell Cycle Nodes (Expanded)
cell_cycle_nodes = [
    # Cyclins & CDKs
    'Cyclin D1', 'Cyclin D2', 'Cyclin D3', 'Cyclin E1', 'Cyclin E2', 'Cyclin A1',
    'Cyclin A2', 'Cyclin B1', 'Cyclin B2', 'Cyclin B3', 'CDK1', 'CDK2', 'CDK4', 'CDK6',
    # CDK Inhibitors
    'p21', 'p27', 'p57', 'p16', 'p15', 'p18', 'p19',
    # Rb-E2F Pathway
    'Rb', 'Rb-p', 'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5', 'DP1', 'DP2',
    # DNA Damage Response
    'ATM', 'ATR', 'Chk1', 'Chk2', 'p53', 'Mdm2', 'Wip1', '14-3-3', 'CDC25A', 'CDC25B', 'CDC25C',
    # Checkpoints
    'BubR1', 'Bub1', 'Bub3', 'Mad1', 'Mad2', 'Aurora A', 'Aurora B', 'PLK1', 'WEE1', 'MYT1',
    # Ubiquitin Ligases
    'APC/C', 'CDC20', 'CDH1', 'SCF', 'SKP2', 'CUL1', 'FBXW7',
    # Transcription Factors
    'MYC', 'FOXM1', 'HIF1A', 'NF-kB', 'STAT3', 'YAP1', 'NOTCH1',
    # Phenotypes
    'G1_phase', 'S_phase', 'G2_phase', 'M_phase', 'cell_cycle_arrest', 'senescence', 'apoptosis'
]

# Add nodes
G_cell_cycle.add_nodes_from(cell_cycle_nodes)

# Expanded Interactions (More Complex)
cell_cycle_edges = [
    # Cyclin-CDK Complexes
    ('Cyclin D1', 'CDK4', 'binding'), ('Cyclin D1', 'CDK6', 'binding'),
    ('Cyclin E1', 'CDK2', 'binding'), ('Cyclin A1', 'CDK2', 'binding'),
    ('Cyclin B1', 'CDK1', 'binding'),

    # Rb-E2F Regulation
    ('Cyclin D1/CDK4', 'Rb', 'phosphorylation'), ('Cyclin E1/CDK2', 'Rb', 'phosphorylation'),
    ('Rb', 'E2F1', 'inhibition'), ('Rb-p', 'E2F1', 'release'),
    ('E2F1', 'Cyclin E1', 'transcription'), ('E2F1', 'Cyclin A1', 'transcription'),

    # CDK Inhibitors
    ('p21', 'Cyclin E1/CDK2', 'inhibition'), ('p21', 'Cyclin A1/CDK2', 'inhibition'),
    ('p27', 'Cyclin D1/CDK4', 'inhibition'), ('p16', 'CDK4', 'inhibition'),
    ('p53', 'p21', 'transcription'),

    # DNA Damage Response
    ('ATM', 'Chk2', 'phosphorylation'), ('ATR', 'Chk1', 'phosphorylation'),
    ('Chk1', 'CDC25A', 'inhibition'), ('Chk2', 'CDC25C', 'inhibition'),
    ('Chk1', 'p53', 'phosphorylation'), ('p53', 'Mdm2', 'transcription'),
    ('Mdm2', 'p53', 'ubiquitination'), ('Wip1', 'ATM', 'dephosphorylation'),

    # Checkpoints
    ('BubR1', 'APC/C', 'inhibition'), ('Mad2', 'CDC20', 'inhibition'),
    ('Aurora B', 'BubR1', 'phosphorylation'), ('PLK1', 'CDC25C', 'activation'),

    # Ubiquitin-Mediated Degradation
    ('APC/C-CDC20', 'Cyclin B1', 'degradation'), ('APC/C-CDH1', 'Cyclin A1', 'degradation'),
    ('SCF-SKP2', 'p27', 'degradation'), ('FBXW7', 'Cyclin E1', 'degradation'),

    # Cross-Talk with Other Pathways
    ('MYC', 'Cyclin D1', 'transcription'), ('NF-kB', 'Cyclin D1', 'transcription'),
    ('STAT3', 'Cyclin D1', 'transcription'), ('YAP1', 'Cyclin E1', 'transcription'),
    ('NOTCH1', 'Cyclin D1', 'transcription'), ('HIF1A', 'Cyclin D1', 'transcription'),

    # Phenotypic Outcomes
    ('Cyclin B1/CDK1', 'M_phase', 'initiation'), ('Cyclin D1/CDK4', 'G1_phase', 'progression'),
    ('p21', 'cell_cycle_arrest', 'induction'), ('p53', 'apoptosis', 'induction'),
]

# Add edges with random weights
for source, target, interaction in cell_cycle_edges:
    G_cell_cycle.add_edge(source, target, weight=random.uniform(0.5, 2.0), interaction=interaction)

# Add random edges to increase complexity
for _ in range(150):
    src = random.choice(cell_cycle_nodes)
    tgt = random.choice(cell_cycle_nodes)
    if src != tgt and not G_cell_cycle.has_edge(src, tgt):
        G_cell_cycle.add_edge(src, tgt, weight=random.uniform(0.1, 0.5), interaction='random_crosstalk')

# Visualization
plt.figure(figsize=(25, 25))
pos = nx.spring_layout(G_cell_cycle, k=0.4, seed=42)

# Color nodes by function
node_colors = []
for node in G_cell_cycle.nodes():
    if 'Cyclin' in node:
        node_colors.append('red')
    elif 'CDK' in node:
        node_colors.append('orange')
    elif node in ['p21', 'p27', 'p16']:
        node_colors.append('lime')
    elif node in ['ATM', 'ATR', 'Chk1', 'Chk2', 'p53']:
        node_colors.append('skyblue')
    elif node in ['G1_phase', 'S_phase', 'G2_phase', 'M_phase']:
        node_colors.append('purple')
    else:
        node_colors.append('lightgray')

nx.draw(G_cell_cycle, pos, with_labels=True, node_size=800, node_color=node_colors,
        font_size=8, edge_color='gray', width=0.7, arrowsize=15)
plt.title("Expanded Cell Cycle Network with Cross-Talk", size=20)
plt.tight_layout()
plt.savefig('expanded_cell_cycle_network.png', dpi=300)
plt.show()

# Network Analysis
print("\nCell Cycle Network Stats:")
print(f"Nodes: {G_cell_cycle.number_of_nodes()}, Edges: {G_cell_cycle.number_of_edges()}")
print(f"Avg Degree: {sum(dict(G_cell_cycle.degree()).values()) / len(G_cell_cycle):.2f}")

# Create expanded MAPK Network
G_mapk = nx.DiGraph()

# Core MAPK Nodes (Expanded)
mapk_nodes = [
    # Growth Factor Receptors
    'EGFR', 'HER2', 'HER3', 'HER4', 'IGF1R', 'INSR', 'PDGFR', 'FGFR', 'MET', 'NTRK',
    # RAS-RAF-MEK-ERK Pathway
    'GRB2', 'SOS1', 'RAS', 'HRAS', 'KRAS', 'NRAS', 'RAF1', 'BRAF', 'MEK1', 'MEK2', 'ERK1', 'ERK2',
    # PI3K-AKT-mTOR Pathway
    'PI3K', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PTEN', 'PIP2', 'PIP3', 'AKT1', 'AKT2', 'AKT3', 'PDK1', 'mTORC1', 'mTORC2',
    # Downstream Effectors
    'RSK1', 'RSK2', 'RSK3', 'MSK1', 'MSK2', 'MNK1', 'MNK2', 'ELK1', 'CREB', 'c-FOS', 'c-JUN', 'MYC',
    # Negative Regulators
    'DUSP1', 'DUSP6', 'SPRY2', 'SPRY4', 'PP2A', 'PTPN11', 'NF1',
    # Cross-Talk Nodes
    'GSK3B', 'FOXO1', 'FOXO3', 'BAD', 'BCL2', 'BCLXL', 'p70S6K', '4E-BP1', 'eIF4E', 'HIF1A',
    # Phenotypes
    'cell_proliferation', 'cell_survival', 'migration', 'invasion', 'angiogenesis'
]

# Add nodes
G_mapk.add_nodes_from(mapk_nodes)

# Expanded Interactions (More Complex)
mapk_edges = [
    # EGFR-RAS-ERK Pathway
    ('EGFR', 'GRB2', 'binding'), ('GRB2', 'SOS1', 'recruitment'),
    ('SOS1', 'RAS', 'activation'), ('RAS', 'RAF1', 'activation'),
    ('RAF1', 'MEK1', 'phosphorylation'), ('MEK1', 'ERK1', 'phosphorylation'),
    ('ERK1', 'RSK1', 'activation'), ('ERK1', 'c-FOS', 'phosphorylation'),
    ('ERK1', 'MYC', 'stabilization'),

    # PI3K-AKT Pathway
    ('EGFR', 'PI3K', 'activation'), ('PI3K', 'PIP3', 'production'),
    ('PIP3', 'AKT1', 'recruitment'), ('PDK1', 'AKT1', 'phosphorylation'),
    ('AKT1', 'mTORC1', 'activation'), ('mTORC1', 'p70S6K', 'activation'),
    ('mTORC1', '4E-BP1', 'inhibition'), ('AKT1', 'GSK3B', 'inhibition'),
    ('AKT1', 'FOXO1', 'inhibition'),

    # Negative Feedback
    ('DUSP1', 'ERK1', 'dephosphorylation'), ('SPRY2', 'GRB2', 'inhibition'),
    ('PP2A', 'AKT1', 'dephosphorylation'), ('PTEN', 'PIP3', 'dephosphorylation'),

    # Cross-Talk with Other Pathways
    ('ERK1', 'AR', 'phosphorylation'), ('AKT1', 'AR', 'phosphorylation'),
    ('ERK1', 'ESR1', 'phosphorylation'), ('AKT1', 'ESR1', 'phosphorylation'),
    ('ERK1', 'HIF1A', 'stabilization'), ('mTORC1', 'HIF1A', 'translation'),

    # Phenotypic Outcomes
    ('MYC', 'cell_proliferation', 'promotion'), ('AKT1', 'cell_survival', 'promotion'),
    ('ERK1', 'migration', 'promotion'), ('HIF1A', 'angiogenesis', 'promotion'),
]

# Add edges with random weights
for source, target, interaction in mapk_edges:
    G_mapk.add_edge(source, target, weight=random.uniform(0.5, 2.0), interaction=interaction)

# Add random edges to increase complexity
for _ in range(150):
    src = random.choice(mapk_nodes)
    tgt = random.choice(mapk_nodes)
    if src != tgt and not G_mapk.has_edge(src, tgt):
        G_mapk.add_edge(src, tgt, weight=random.uniform(0.1, 0.5), interaction='random_crosstalk')

# Visualization
plt.figure(figsize=(25, 25))
pos = nx.spring_layout(G_mapk, k=0.4, seed=42)

# Color nodes by function
node_colors = []
for node in G_mapk.nodes():
    if node in ['EGFR', 'HER2', 'IGF1R']:
        node_colors.append('red')
    elif node in ['RAS', 'RAF1', 'MEK1', 'ERK1']:
        node_colors.append('orange')
    elif node in ['PI3K', 'AKT1', 'mTORC1']:
        node_colors.append('lime')
    elif node in ['DUSP1', 'SPRY2', 'PTEN']:
        node_colors.append('skyblue')
    elif node in ['cell_proliferation', 'cell_survival', 'angiogenesis']:
        node_colors.append('purple')
    else:
        node_colors.append('lightgray')

nx.draw(G_mapk, pos, with_labels=True, node_size=800, node_color=node_colors,
        font_size=8, edge_color='gray', width=0.7, arrowsize=15)
plt.title("Expanded MAPK Signaling Network with Cross-Talk", size=20)
plt.tight_layout()
plt.savefig('expanded_mapk_network.png', dpi=300)
plt.show()

# Network Analysis
print("\nMAPK Network Stats:")
print(f"Nodes: {G_mapk.number_of_nodes()}, Edges: {G_mapk.number_of_edges()}")
print(f"Avg Degree: {sum(dict(G_mapk.degree()).values()) / len(G_mapk):.2f}")
