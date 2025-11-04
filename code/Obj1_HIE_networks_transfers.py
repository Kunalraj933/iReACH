import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

np.random.seed(42)

#############SYNTHETIC DATA GENERATION###############

# Step 1: Create synthetic hospital data
# Generate list of NHS Type 1 NWL Major A&E Departments
hospitals = [
    "Northwick Park Hospital", 
    "Hillingdon Hospital",    
    "Ealing Hospital",      
    "Chelsea and Westminster Hospital", 
    "Charing Cross Hospital",   
    "West Middlesex University Hospital",  
    "St Mary's Hospital"
]

# Step 2: Generate synthetic patient transfer data (Emergency dataset + Admitted patient dataset)
# Layer 1: Transfer Network
def generate_transfer_data(hospitals, n_transfers=5000, date_range_days=1095):
    
    start_date = datetime(2021, 4, 1)
    transfers = []
    
    # Create hospital weights based on realistic transfer patterns
    # Larger hospitals tend to receive more transfers
    hospital_sizes = np.random.lognormal(mean=2, sigma=0.8, size=len(hospitals))
    
    # Normalize to create probability distributions
    origin_weights = hospital_sizes / hospital_sizes.sum()
    destination_weights = (hospital_sizes ** 1.5) / (hospital_sizes ** 1.5).sum()  # Larger hospitals receive proportionally more
    
    for i in range(n_transfers):
        # Origin hospital (weighted by hospital size)
        origin = np.random.choice(hospitals, p=origin_weights)
        
        # Destination hospital (cannot be same as origin, weighted toward larger hospitals)
        available_hospitals = [h for h in hospitals if h != origin]
        dest_weights = np.array([destination_weights[hospitals.index(h)] for h in available_hospitals])
        dest_weights = dest_weights / dest_weights.sum()  # Re-normalize after removing origin
        
        destination = np.random.choice(available_hospitals, p=dest_weights)
        
        # Transfer date
        transfer_date = start_date + timedelta(days=np.random.randint(0, date_range_days))
        
        # Patient ID (synthetic)
        patient_id = f"PT{i:06d}"
        
        transfers.append({
            'patient_id': patient_id,
            'origin_hospital': origin,
            'destination_hospital': destination,
            'transfer_date': transfer_date,
            'transfer_type': np.random.choice(['emergency', 'planned'], p=[0.7, 0.3])
        })
    
    return pd.DataFrame(transfers)

transfer_data = generate_transfer_data(hospitals)
print(f"\nGenerated {len(transfer_data)} patient transfers")
print(transfer_data.head())

# Optional: View transfer distribution
print("\nTransfer distribution by hospital:")
print("Origins:")
print(transfer_data['origin_hospital'].value_counts().head(10))
print("\nDestinations:")
print(transfer_data['destination_hospital'].value_counts().head(10))

############NETWORK LAYERS DEGREE CENTRALITY################

# Step 1: Transfer Network
def build_transfer_network(transfer_data):
    """
    Build directed, weighted network from patient transfer data
    Nodes: Hospitals (sized by in-degree centrality)
    Edges: Weighted by number of transfers (origin -> destination)
    """
    G_transfer = nx.DiGraph()
    
    # Add all hospitals as nodes
    G_transfer.add_nodes_from(transfer_data['origin_hospital'].unique())
    G_transfer.add_nodes_from(transfer_data['destination_hospital'].unique())
    
    # Count transfers between hospital pairs
    transfer_counts = transfer_data.groupby(
        ['origin_hospital', 'destination_hospital']
    ).size().reset_index(name='weight')
    
    # Add weighted edges
    for _, row in transfer_counts.iterrows():
        G_transfer.add_edge(
            row['origin_hospital'],
            row['destination_hospital'],
            weight=row['weight'],
            edge_type='transfer'
        )
    
    # Calculate in-degree centrality (transfers received)
    in_degree_centrality = nx.in_degree_centrality(G_transfer)
    
    # Add node attributes for size based on in-degree centrality
    for node in G_transfer.nodes():
        G_transfer.nodes[node]['in_degree_centrality'] = in_degree_centrality[node]
        # Scale the size for visualization (multiply by a factor for visibility)
        G_transfer.nodes[node]['node_size'] = in_degree_centrality[node] * 500 + 10  # Min size of 100
    
    return G_transfer

G_transfer = build_transfer_network(transfer_data)
print("LAYER 1: Transfer Network")
print(f"Nodes: {G_transfer.number_of_nodes()}")
print(f"Edges: {G_transfer.number_of_edges()}")
print(f"Average degree: {sum(dict(G_transfer.degree()).values()) / G_transfer.number_of_nodes():.2f}")

# Display top hospitals by transfers received
in_degrees = dict(G_transfer.in_degree())
top_receivers = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 hospitals by transfers received:")
for hospital, count in top_receivers:
    print(f"  {hospital}: {count} transfers (centrality: {G_transfer.nodes[hospital]['in_degree_centrality']:.4f})")


############ VISUALISATION OF DEGREE CENTRALITY ################

def visualize_transfer_network(G, title="Layer 1: Transfer Network"):
    """Visualize the patient transfer network"""
    plt.figure(figsize=(14, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on in-degree centrality (already stored in node attributes)
    if 'node_size' in G.nodes[list(G.nodes())[0]]:
        # Use pre-calculated node sizes from the graph
        node_sizes = [G.nodes[node]['node_size'] for node in G.nodes()]
    else:
        # Fallback to degree-based sizing if node_size not available
        node_sizes = [3000 * G.degree(node) / max(dict(G.degree()).values()) 
                      for node in G.nodes()]
    
    # Color nodes based on in-degree centrality for better visualization
    if 'in_degree_centrality' in G.nodes[list(G.nodes())[0]]:
        node_colors = [G.nodes[node]['in_degree_centrality'] for node in G.nodes()]
        cmap = plt.cm.YlOrRd  # Yellow to Red colormap
    else:
        node_colors = 'lightblue'
        cmap = None
    
    # Edge widths based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [3 * w / max_weight for w in weights]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9,
                          cmap=cmap, vmin=0, vmax=max(node_colors) if isinstance(node_colors, list) else 1)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                          arrows=True, arrowsize=15, edge_color='gray',
                          connectionstyle='arc3,rad=0.1')
    
    # Add colorbar if using centrality-based coloring
    if isinstance(node_colors, list):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
        cbar.set_label('In-Degree Centrality', rotation=270, labelpad=20)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    return plt



# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

# Assuming networks from Part 2 are available: G_transfer, G_referral, G_integrated

print("Creating visualizations...")

# Layer 1: Transfer Network
visualize_transfer_network(G_transfer)
plt.savefig('layer1_transfer_network.png', dpi=300, bbox_inches='tight')
plt.show()


# ==============================================================================
# WEIGHTED CENTRALITY ANALYSIS
# ==============================================================================

def calculate_weighted_centralities(G):
    """
    Calculate various weighted centrality metrics for the network
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Directed graph with weighted edges
    
    Returns:
    --------
    dict : Dictionary containing various centrality metrics
    """
    
    # 1. Weighted In-Degree (total transfers received)
    weighted_in_degree = dict(G.in_degree(weight='weight'))
    max_weighted_in_degree = max(weighted_in_degree.values()) if weighted_in_degree else 1
    
    # 2. Weighted Out-Degree (total transfers sent)
    weighted_out_degree = dict(G.out_degree(weight='weight'))
    max_weighted_out_degree = max(weighted_out_degree.values()) if weighted_out_degree else 1
    
    # 3. Normalized metrics (0-1 scale)
    normalized_in_degree = {
        node: weighted_in_degree.get(node, 0) / max_weighted_in_degree 
        for node in G.nodes()
    }
    
    normalized_out_degree = {
        node: weighted_out_degree.get(node, 0) / max_weighted_out_degree 
        for node in G.nodes()
    }
    
    # 4. Net flow (difference between incoming and outgoing)
    net_flow = {
        node: weighted_in_degree.get(node, 0) - weighted_out_degree.get(node, 0)
        for node in G.nodes()
    }
    
    # Store all metrics in node attributes
    for node in G.nodes():
        G.nodes[node]['weighted_in_degree'] = weighted_in_degree.get(node, 0)
        G.nodes[node]['weighted_out_degree'] = weighted_out_degree.get(node, 0)
        G.nodes[node]['normalized_in_degree'] = normalized_in_degree[node]
        G.nodes[node]['normalized_out_degree'] = normalized_out_degree[node]
        G.nodes[node]['net_flow'] = net_flow[node]
        
        # Node size for visualization (based on incoming transfers)
        G.nodes[node]['node_size'] = normalized_in_degree[node] * 5000 + 100
    
    # Return summary statistics
    centrality_metrics = {
        'weighted_in_degree': weighted_in_degree,
        'weighted_out_degree': weighted_out_degree,
        'normalized_in_degree': normalized_in_degree,
        'normalized_out_degree': normalized_out_degree,
        'net_flow': net_flow,
        'max_in_degree': max_weighted_in_degree,
        'max_out_degree': max_weighted_out_degree
    }
    
    return centrality_metrics


def print_centrality_report(G, centrality_metrics, top_n=10):
    """
    Print a detailed report of centrality metrics
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph
    centrality_metrics : dict
        Dictionary of centrality metrics from calculate_weighted_centralities
    top_n : int
        Number of top hospitals to display
    """
    
    print("\n" + "="*80)
    print("WEIGHTED CENTRALITY ANALYSIS")
    print("="*80)
    
    # Top hospitals by incoming transfers (referral hubs)
    print(f"\nTop {top_n} Hospitals by INCOMING Transfers (Referral Hubs):")
    print("-" * 80)
    top_in = sorted(
        centrality_metrics['weighted_in_degree'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    for i, (hospital, count) in enumerate(top_in, 1):
        norm_val = centrality_metrics['normalized_in_degree'][hospital]
        print(f"{i:2d}. {hospital:30s} | Transfers In: {count:6.0f} | Normalized: {norm_val:.4f}")
    
    # Top hospitals by outgoing transfers (referring hospitals)
    print(f"\nTop {top_n} Hospitals by OUTGOING Transfers (Referring Hospitals):")
    print("-" * 80)
    top_out = sorted(
        centrality_metrics['weighted_out_degree'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    for i, (hospital, count) in enumerate(top_out, 1):
        norm_val = centrality_metrics['normalized_out_degree'][hospital]
        print(f"{i:2d}. {hospital:30s} | Transfers Out: {count:6.0f} | Normalized: {norm_val:.4f}")
    
    # Net flow analysis (hubs vs spokes)
    print(f"\nTop {top_n} Net RECEIVERS (Major Hubs - more in than out):")
    print("-" * 80)
    top_receivers = sorted(
        centrality_metrics['net_flow'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    for i, (hospital, flow) in enumerate(top_receivers, 1):
        in_count = centrality_metrics['weighted_in_degree'][hospital]
        out_count = centrality_metrics['weighted_out_degree'][hospital]
        print(f"{i:2d}. {hospital:30s} | Net Flow: {flow:+7.0f} | In: {in_count:5.0f} | Out: {out_count:5.0f}")
    
    print(f"\nTop {top_n} Net SENDERS (Spoke Hospitals - more out than in):")
    print("-" * 80)
    top_senders = sorted(
        centrality_metrics['net_flow'].items(),
        key=lambda x: x[1]
    )[:top_n]
    
    for i, (hospital, flow) in enumerate(top_senders, 1):
        in_count = centrality_metrics['weighted_in_degree'][hospital]
        out_count = centrality_metrics['weighted_out_degree'][hospital]
        print(f"{i:2d}. {hospital:30s} | Net Flow: {flow:+7.0f} | In: {in_count:5.0f} | Out: {out_count:5.0f}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Maximum weighted in-degree:  {centrality_metrics['max_in_degree']:.0f} transfers")
    print(f"Maximum weighted out-degree: {centrality_metrics['max_out_degree']:.0f} transfers")
    print(f"Average weighted in-degree:  {sum(centrality_metrics['weighted_in_degree'].values()) / len(G.nodes()):.2f} transfers")
    print(f"Average weighted out-degree: {sum(centrality_metrics['weighted_out_degree'].values()) / len(G.nodes()):.2f} transfers")
    print("="*80 + "\n")


# ==============================================================================
# APPLY WEIGHTED CENTRALITY ANALYSIS TO TRANSFER NETWORK
# ==============================================================================

# Calculate weighted centralities for the transfer network
transfer_centrality = calculate_weighted_centralities(G_transfer)

# Print detailed report
print_centrality_report(G_transfer, transfer_centrality, top_n=10)

# You can also access individual metrics
print("\nExample: Accessing centrality data for a specific hospital")
sample_hospital = list(G_transfer.nodes())[0]
print(f"Hospital: {sample_hospital}")
print(f"  Weighted In-Degree:  {G_transfer.nodes[sample_hospital]['weighted_in_degree']}")
print(f"  Weighted Out-Degree: {G_transfer.nodes[sample_hospital]['weighted_out_degree']}")
print(f"  Net Flow:            {G_transfer.nodes[sample_hospital]['net_flow']}")
print(f"  Node Size:           {G_transfer.nodes[sample_hospital]['node_size']:.2f}")


# ==============================================================================
# WEIGHTED NETWORK GRAPH VISUALIZATION
# ==============================================================================

def visualize_weighted_network(G, centrality_metrics, network_name="Transfer Network", 
                               layout='spring', min_edge_weight=None):
    """
    Visualize the weighted network with nodes sized and colored by centrality
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph with centrality metrics
    centrality_metrics : dict
        Dictionary of centrality metrics
    network_name : str
        Name of the network for title
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai'
    min_edge_weight : int
        Minimum edge weight to display (filters weak connections)
    """
    
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Filter edges by weight if specified
    if min_edge_weight is not None:
        G_filtered = G.copy()
        edges_to_remove = [(u, v) for u, v, d in G_filtered.edges(data=True) 
                          if d['weight'] < min_edge_weight]
        G_filtered.remove_edges_from(edges_to_remove)
        print(f"Filtered to {G_filtered.number_of_edges()} edges (min weight: {min_edge_weight})")
    else:
        G_filtered = G
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G_filtered, k=3, iterations=50, seed=42, weight='weight')
    elif layout == 'circular':
        pos = nx.circular_layout(G_filtered)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_filtered, weight='weight')
    else:
        pos = nx.spring_layout(G_filtered, k=3, iterations=50, seed=42)
    
    # Node sizes based on weighted in-degree (transfers received)
    node_sizes = [G.nodes[node].get('node_size', 100) * 0.2 for node in G_filtered.nodes()]
    
    # Node colors based on normalized in-degree
    node_colors = [G.nodes[node].get('normalized_in_degree', 0) for node in G_filtered.nodes()]
    
    # Edge properties
    edges = G_filtered.edges()
    edge_weights = [G_filtered[u][v]['weight'] for u, v in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 1
    
    # Scale edge widths (0.5 to 2 range)
    edge_widths = [0.5 + 2 * (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight 
                   else 0.4 for w in edge_weights]
    
    # Edge colors based on weight (lighter = fewer transfers, darker = more transfers)
    edge_colors = [plt.cm.Greys(0.3 + 0.6 * (w - min_weight) / (max_weight - min_weight)) 
                   if max_weight > min_weight else plt.cm.Greys(0.5) 
                   for w in edge_weights]
    
    # Draw edges
    nx.draw_networkx_edges(
        G_filtered, pos, 
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        node_size=node_sizes,
        ax=ax
    )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G_filtered, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.YlOrRd,
        vmin=0,
        vmax=1,
        alpha=0.9,
        edgecolors=None,
        linewidths=2,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G_filtered, pos,
        font_size=8,
        font_weight='bold',
        font_color='black',
        ax=ax
    )
    
    # Add colorbar for node colors
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                     norm=plt.Normalize(vmin=0, vmax=1))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar_nodes.set_label('Normalized In-Degree\n(Transfers Received)', 
                        rotation=270, labelpad=25, fontsize=11, fontweight='bold')
    
    # Title and legend
    title = f'{network_name}\nWeighted Network Visualization'
    if min_edge_weight:
        title += f'\n(Showing edges ≥ {min_edge_weight} transfers)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    
    legend_elements = [
        Line2D([0], [0], color='none', marker='o', markerfacecolor='#fee5d9', 
               markeredgecolor='black', markersize=8, label='Low centrality (small hospitals)', 
               markeredgewidth=1.5),
        Line2D([0], [0], color='none', marker='o', markerfacecolor='#fc9272', 
               markeredgecolor='black', markersize=12, label='Medium centrality', 
               markeredgewidth=1.5),
        Line2D([0], [0], color='none', marker='o', markerfacecolor='#de2d26', 
               markeredgecolor='black', markersize=16, label='High centrality (major hubs)', 
               markeredgewidth=1.5),
        Line2D([0], [0], color='gray', linewidth=1, label='Few transfers'),
        Line2D([0], [0], color='black', linewidth=4, label='Many transfers'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.9, title='Node Size & Edge Width', title_fontsize=11)
    
    # Add statistics box
    stats_text = f"Network Statistics:\n"
    stats_text += f"Nodes: {G_filtered.number_of_nodes()}\n"
    stats_text += f"Edges: {G_filtered.number_of_edges()}\n"
    stats_text += f"Max In-Degree: {centrality_metrics['max_in_degree']:.0f}\n"
    stats_text += f"Max Out-Degree: {centrality_metrics['max_out_degree']:.0f}"
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax


# ==============================================================================
# CREATE NETWORK VISUALIZATIONS
# ==============================================================================

# 1. Full network (may be cluttered if many hospitals)
print("Creating full network visualization...")
fig1, ax1 = visualize_weighted_network(
    G_transfer, 
    transfer_centrality,
    network_name="Patient Transfer Network",
    layout='spring'
)
plt.show()

# 2. Filtered network (only show significant connections)
#print("\nCreating filtered network visualization (edges with ≥5 transfers)...")
#fig2, ax2 = visualize_weighted_network(
    #G_transfer, 
    #transfer_centrality,
    #network_name="Patient Transfer Network",
    #layout='spring',
    #min_edge_weight=5  # Only show edges with 5+ transfers)
#plt.show()

# 3. Alternative layout (Kamada-Kawai - better for weighted networks)
print("\nCreating Kamada-Kawai layout visualization...")
fig3, ax3 = visualize_weighted_network(
    G_transfer, 
    transfer_centrality,
    network_name="Patient Transfer Network",
    layout='kamada_kawai',
    min_edge_weight=3)
plt.show()



# ============================================================================
# NETWORK ANALYSIS FUNCTIONS
# ============================================================================

def analyze_centrality_metrics(G, network_name):
    """
    Calculate key centrality metrics to identify critical nodes
    in the HIE network (vulnerable to cyberattack propagation)
    """
    print(f"\n{'='*60}")
    print(f"CENTRALITY ANALYSIS: {network_name}")
    print(f"{'='*60}")
    
    # Degree Centrality (normalized)
    degree_cent = nx.degree_centrality(G)
    
    # Betweenness Centrality (identifies bridge nodes)
    betweenness_cent = nx.betweenness_centrality(G, weight='weight')
    
    # Closeness Centrality (speed of information spread)
    closeness_cent = nx.closeness_centrality(G)
    
    # PageRank (importance based on connections)
    pagerank = nx.pagerank(G, weight='weight')
    
    # Create summary dataframe
    nodes = list(G.nodes())
    metrics_df = pd.DataFrame({
        'Node': nodes,
        'Degree_Centrality': [degree_cent[n] for n in nodes],
        'Betweenness_Centrality': [betweenness_cent[n] for n in nodes],
        'Closeness_Centrality': [closeness_cent[n] for n in nodes],
        'PageRank': [pagerank[n] for n in nodes],
        'In_Degree': [G.in_degree(n) for n in nodes],
        'Out_Degree': [G.out_degree(n) for n in nodes]
    })
    
    # Sort by PageRank (overall importance)
    metrics_df = metrics_df.sort_values('PageRank', ascending=False)
    
    print("\nTop 10 Most Central Nodes (by PageRank):")
    print(metrics_df.head(10).to_string(index=False))
    
    return metrics_df

def identify_vulnerability_hotspots(G, metrics_df, top_n=5):
    """
    Identify nodes that represent critical vulnerabilities
    High betweenness = critical bridges for attack propagation
    """
    print(f"\n{'='*60}")
    print(f"VULNERABILITY HOTSPOT ANALYSIS")
    print(f"{'='*60}")
    
    # Nodes with highest betweenness are critical bridges
    high_betweenness = metrics_df.nlargest(top_n, 'Betweenness_Centrality')
    
    print(f"\nTop {top_n} Critical Bridge Nodes (Cyberattack Propagation Risk):")
    print(high_betweenness[['Node', 'Betweenness_Centrality', 
                            'Degree_Centrality', 'PageRank']].to_string(index=False))
    
    # Nodes with highest degree = most connections
    high_degree = metrics_df.nlargest(top_n, 'Degree_Centrality')
    
    print(f"\nTop {top_n} Most Connected Nodes (Exposure Risk):")
    print(high_degree[['Node', 'Degree_Centrality', 
                       'In_Degree', 'Out_Degree']].to_string(index=False))
    
    return high_betweenness, high_degree

def analyze_network_density(G, network_name):
    """
    Calculate network density and clustering coefficient
    Higher density = more interconnected = faster attack propagation
    """
    print(f"\n{'='*60}")
    print(f"NETWORK DENSITY ANALYSIS: {network_name}")
    print(f"{'='*60}")
    
    # Density (proportion of possible edges that exist)
    density = nx.density(G)
    print(f"Network Density: {density:.4f}")
    
    # Average clustering coefficient (local interconnection)
    # For directed graphs, use weakly connected components
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
        clustering = nx.average_clustering(G_undirected)
    else:
        clustering = nx.average_clustering(G)
    
    print(f"Average Clustering Coefficient: {clustering:.4f}")
    
    # Strongly connected components (for directed graphs)
    if nx.is_directed(G):
        scc = list(nx.strongly_connected_components(G))
        print(f"Number of Strongly Connected Components: {len(scc)}")
        print(f"Largest SCC size: {len(max(scc, key=len))} nodes")
    
    # Weakly connected components
    wcc = list(nx.weakly_connected_components(G) if nx.is_directed(G) 
               else nx.connected_components(G))