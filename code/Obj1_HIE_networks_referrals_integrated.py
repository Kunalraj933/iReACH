import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

#############SYNTHETIC DATA GENERATION###############

# Step 3: Generate synthetic referral data (outpatient dataset)
# This represents Layer 2: Referral Network
def generate_referral_data(hospitals, n_referrals=8000, date_range_days=1095):
    
    start_date = datetime(2021, 4, 1)
    referrals = []
    
    # Create hospital weights based on realistic referral patterns
    # Referral patterns may differ from transfer patterns
    hospital_sizes = np.random.lognormal(mean=2, sigma=0.8, size=len(hospitals))
    
    # Referring weights: smaller hospitals refer more outpatients out
    # Inverse relationship - smaller hospitals refer proportionally more
    referring_weights = (1 / hospital_sizes) / (1 / hospital_sizes).sum()
    
    # Receiving weights: larger hospitals receive more referrals (specialty centers)
    # Even stronger hub effect than transfers since referrals seek specialized care
    receiving_weights = (hospital_sizes ** 2) / (hospital_sizes ** 2).sum()
    
    for i in range(n_referrals):
        # Referring hospital (weighted toward smaller hospitals)
        referring = np.random.choice(hospitals, p=referring_weights)
        
        # Receiving hospital (weighted toward larger hospitals, excluding referring hospital)
        available_hospitals = [h for h in hospitals if h != referring]
        recv_weights = np.array([receiving_weights[hospitals.index(h)] for h in available_hospitals])
        recv_weights = recv_weights / recv_weights.sum()  # Re-normalize after removing referring hospital
        
        receiving = np.random.choice(available_hospitals, p=recv_weights)
        
        # Referral date
        referral_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        
        # Patient ID (synthetic)
        patient_id = f"REF{i:06d}"  # Different prefix to distinguish from transfers
        
        referrals.append({
            'patient_id': patient_id,
            'referring_hospital': referring,
            'receiving_hospital': receiving,
            'referral_date': referral_date,
        })
    
    return pd.DataFrame(referrals)

referral_data = generate_referral_data(hospitals)
print(f"\nGenerated {len(referral_data)} patient referrals")
print(referral_data.head())

# Optional: View referral distribution
print("\nReferral distribution by hospital:")
print("Referring (who sends):")
print(referral_data['referring_hospital'].value_counts().head(10))
print("\nReceiving (who gets referrals):")
print(referral_data['receiving_hospital'].value_counts().head(10))

############NETWORK LAYERS DEGREE CENTRALITY################

# Step 2: Referral Network
def build_referral_network(referral_data):
    """
    Build directed, weighted network from referral data
    Nodes: Hospitals
    Edges: Weighted by number of referrals (referring -> receiving)
    """
    G_referral = nx.DiGraph()
    
    # Add all hospitals as nodes
    G_referral.add_nodes_from(referral_data['referring_hospital'].unique())
    G_referral.add_nodes_from(referral_data['receiving_hospital'].unique())
    
    # Count referrals between hospital pairs (excluding self-referrals for inter-trust focus)
    referral_counts = referral_data[
        referral_data['referring_hospital'] != referral_data['receiving_hospital']
    ].groupby(
        ['referring_hospital', 'receiving_hospital']
    ).size().reset_index(name='weight')
    
    # Add weighted edges
    for _, row in referral_counts.iterrows():
        G_referral.add_edge(
            row['referring_hospital'],
            row['receiving_hospital'],
            weight=row['weight'],
            edge_type='referral'
        )
    
    return G_referral

G_referral = build_referral_network(referral_data)
print("\nLAYER 2: Referral Network")
print(f"Nodes: {G_referral.number_of_nodes()}")
print(f"Edges: {G_referral.number_of_edges()}")
print(f"Average degree: {sum(dict(G_referral.degree()).values()) / G_referral.number_of_nodes():.2f}")



# Step 3: Multi-layered Integrated Network

def build_integrated_network(G_transfer, G_referral):
    """
    Combine all three layers into comprehensive HIE network
    Edge weights are summed where multiple connection types exist
    """
    G_integrated = nx.DiGraph()
    
    # Add all nodes from all networks
    for G in [G_transfer, G_referral]:
        for node, data in G.nodes(data=True):
            if node not in G_integrated:
                G_integrated.add_node(node, **data)
    
    # Add edges from all networks
    # Transfer edges
    for u, v, data in G_transfer.edges(data=True):
        if G_integrated.has_edge(u, v):
            G_integrated[u][v]['weight'] += data['weight']
            G_integrated[u][v]['edge_types'] = G_integrated[u][v].get('edge_types', []) + ['transfer']
        else:
            G_integrated.add_edge(u, v, weight=data['weight'], edge_types=['transfer'])
    
    # Referral edges
    for u, v, data in G_referral.edges(data=True):
        if G_integrated.has_edge(u, v):
            G_integrated[u][v]['weight'] += data['weight']
            G_integrated[u][v]['edge_types'] = G_integrated[u][v].get('edge_types', []) + ['referral']
        else:
            G_integrated.add_edge(u, v, weight=data['weight'], edge_types=['referral'])
    
    return G_integrated

G_integrated = build_integrated_network(G_transfer, G_referral)
print("\nMULTI-LAYERED INTEGRATED NETWORK")
print(f"Nodes: {G_integrated.number_of_nodes()}")
print(f"Edges: {G_integrated.number_of_edges()}")
print(f"Average degree: {sum(dict(G_integrated.degree()).values()) / G_integrated.number_of_nodes():.2f}")


############ VISUALISATION OF DEGREE CENTRALITY ################

def visualize_referral_network(G, title="Layer 2: Referral Network"):
    """Visualize the referral network"""
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on in-degree (receiving referrals)
    in_degrees = dict(G.in_degree())
    max_in_degree = max(in_degrees.values()) if in_degrees else 1
    node_sizes = [3000 * in_degrees[node] / max_in_degree for node in G.nodes()]
    
    # Edge widths
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [3 * w / max_weight for w in weights]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                          node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                          arrows=True, arrowsize=15, edge_color='green',
                          connectionstyle='arc3,rad=0.1')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    return plt

def visualize_integrated_network(G, title="Multi-Layered Integrated HIE Network"):
    """Visualize the integrated network with all layers"""
    plt.figure(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Node sizes based on total degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # Ensure hospital_nodes is defined from the graph's nodes
    hospital_nodes = list(G.nodes())

    hospital_sizes = [2000 * degrees[node] / max_degree for node in hospital_nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=hospital_nodes, 
                          node_color='lightcoral', node_size=hospital_sizes, 
                          alpha=0.8, label='Hospitals')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [2 * w / max_weight for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, 
                          arrows=True, arrowsize=12, edge_color='darkgray',
                          connectionstyle='arc3,rad=0.05')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(scatterpoints=1, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    return plt

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

# Layer 2: Referral Network
visualize_referral_network(G_referral)
plt.savefig('layer2_referral_network.png', dpi=300, bbox_inches='tight')
plt.show()

# Integrated Network
visualize_integrated_network(G_integrated)
plt.savefig('integrated_hie_network.png', dpi=300, bbox_inches='tight')
plt.show()