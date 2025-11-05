import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.geometry as geom
import matplotlib.patches as mpatches

np.random.seed(42)

############### Read London LSOA shapefile
lsoa = gpd.read_file("/Users/swethamavuru/Desktop/Kunal work/OPR-CAN/Coding/ESRI/LSOA_2011_London_gen_MHW.shp")

############## Filter for NWL LSOAs
nwl_boroughs = ['Brent','Ealing', 'Hammersmith and Fulham', 'Harrow', 'Hillingdon', 'Hounslow', 'Kensington and Chelsea','Westminster']
nwl_lsoa = lsoa[lsoa['LAD11NM'].isin(nwl_boroughs)]
print(nwl_lsoa.head())


############## NWL Type 1 A&E Hospitals and their coordinates
type_1_ED = ["Northwick Park Hospital", "Hillingdon Hospital", "Ealing Hospital", "Chelsea and Westminster Hospital", "Charing Cross Hospital", "West Middlesex University Hospital", "St Mary's Hospital"]
hospital_coords = pd.DataFrame({'hospital_name': type_1_ED})

hospitals = {
    "Northwick Park Hospital": (51.57466, -0.32004), 
    "Hillingdon Hospital": (51.52568, -0.46098),    
    "Ealing Hospital": (51.50784, -0.34592),      
    "Chelsea and Westminster Hospital": (51.48476, -0.18282), 
    "Charing Cross Hospital": (51.48796, -0.22145),   
    "West Middlesex University Hospital": (51.47401, -0.32428),  
    "St Mary's Hospital": (51.51766, -0.17437)   
}

hospital_coords = pd.DataFrame.from_dict(hospitals, orient='index', columns=['latitude', 'longitude'])
hospital_coords['hospital_name'] = hospital_coords.index

geo_hospital_coords = gpd.GeoDataFrame(hospital_coords, 
                       geometry=gpd.points_from_xy(hospital_coords['longitude'], hospital_coords['latitude']), 
                       crs='EPSG:4326').to_crs('EPSG:27700')

print(geo_hospital_coords)

############## Plot NWL LSOAs and hospitals
ax = nwl_lsoa.plot(figsize=(12, 8), color='lightgrey',
                    edgecolor='darkgrey', alpha=0.5)

hospital_colors = {
    'Northwick Park Hospital': 'red',
    'Hillingdon Hospital': 'blue', 
    'Ealing Hospital': 'green',
    'Chelsea and Westminster Hospital': 'orange',
    'Charing Cross Hospital': 'purple',
    'West Middlesex University Hospital': 'brown',
    "St Mary's Hospital": 'pink'
}

for idx, row in geo_hospital_coords.iterrows():
    color = hospital_colors[row['hospital_name']]
    ax.scatter(row.geometry.x, row.geometry.y, 
              color=color, s=100, 
              edgecolor='black', linewidth=1, alpha=0.8)
    
legend_patches = [mpatches.Patch(color=color, label=hospital) 
                 for hospital, color in hospital_colors.items()]
ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title("North West London LSOAs and Type 1 A&E Hospitals", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
plt.savefig('nwl_hospitals_map.jpg', dpi=300, bbox_inches='tight')
plt.show()



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

# Layer 1: Transfer Network
visualize_transfer_network(G_transfer)
plt.savefig('layer1_transfer_network.png', dpi=300, bbox_inches='tight')
plt.show()

############### Geographic Transfer Network on Map (NetworkX Version) ###############

def plot_transfer_network_on_map(nwl_lsoa, geo_hospital_coords, G_transfer, hospital_colors):
    """Simple overlay of transfer network on LSOA map using NetworkX"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. Plot LSOAs as base layer
    nwl_lsoa.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', 
                  alpha=0.3, linewidth=0.5)
    
    # 2. Create position dictionary (hospital coordinates)
    pos = {}
    for idx, row in geo_hospital_coords.iterrows():
        hospital_name = row['hospital_name']
        pos[hospital_name] = (row.geometry.x, row.geometry.y)
    
    # 3. Calculate edge widths based on transfer volume
    edges = G_transfer.edges()
    weights = [G_transfer[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [3 * w / max_weight for w in weights]
    
    # 4. Draw transfer network edges using NetworkX
    nx.draw_networkx_edges(G_transfer, pos, 
                          width=edge_widths, 
                          alpha=0.6, 
                          arrows=True, 
                          arrowsize=20, 
                          edge_color='gray',
                          connectionstyle='arc3,rad=0.15',
                          ax=ax)
    
    # 5. Calculate node sizes based on weighted in-degree (transfers received)
    weighted_in_degree = dict(G_transfer.in_degree(weight='weight'))
    max_transfers = max(weighted_in_degree.values()) if weighted_in_degree else 1
    
    node_sizes = []
    node_colors_list = []
    for node in G_transfer.nodes():
        # Size based on transfer volume
        transfers = weighted_in_degree.get(node, 0)
        size = 50 + 200 * (transfers / max_transfers) if max_transfers > 0 else 100
        node_sizes.append(size)
        
        # Color from hospital_colors dictionary
        node_colors_list.append(hospital_colors.get(node, 'gray'))
    
    # 6. Draw hospital nodes using NetworkX
    nx.draw_networkx_nodes(G_transfer, pos,
                          node_color=node_colors_list,
                          node_size=node_sizes,
                          edgecolors= None,
                          linewidths=2,
                          alpha=0.9,
                          ax=ax)
    
    # 8. Create legend
    legend_patches = [mpatches.Patch(color=color, label=hospital) 
                     for hospital, color in hospital_colors.items()]
    ax.legend(handles=legend_patches, loc='center left', 
             bbox_to_anchor=(1, 0.5), fontsize=10,
             title='Hospitals', title_fontsize=11)
    
    # 9. Add statistics box
    total_transfers = sum(weights)
    stats_text = f"Network Statistics:\n"
    stats_text += f"Hospitals: {G_transfer.number_of_nodes()}\n"
    stats_text += f"Connections: {G_transfer.number_of_edges()}\n"
    stats_text += f"Total Transfers: {int(total_transfers)}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 10. Formatting
    ax.set_title("North West London Hospital Transfer Network", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('on')
    
    plt.tight_layout()
    return fig, ax


# Use the function
print("\nCreating geographic transfer network with NetworkX...")
fig, ax = plot_transfer_network_on_map(nwl_lsoa, geo_hospital_coords, G_transfer, hospital_colors)
plt.savefig('nwl_map_with_network.png', dpi=300, bbox_inches='tight')
plt.show()



def visualize_referral_network(G, title="Layer 2: Referral Network"):
    """Visualize the referral network"""
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on in-degree (receiving referrals)
    in_degrees = dict(G.in_degree())
    max_in_degree = max(in_degrees.values()) if in_degrees else 1
    node_sizes = [500 * in_degrees[node] / max_in_degree for node in G.nodes()]
    
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
    plt.figure(figsize=(12, 8))
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Node sizes based on total degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # Ensure hospital_nodes is defined from the graph's nodes
    hospital_nodes = list(G.nodes())

    hospital_sizes = [500 * degrees[node] / max_degree for node in hospital_nodes]
    
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

# Layer 2: Referral Network
visualize_referral_network(G_referral)
plt.savefig('layer2_referral_network.png', dpi=300, bbox_inches='tight')
plt.show()

# Integrated Network
visualize_integrated_network(G_integrated)
plt.savefig('integrated_hie_network.png', dpi=300, bbox_inches='tight')
plt.show()
