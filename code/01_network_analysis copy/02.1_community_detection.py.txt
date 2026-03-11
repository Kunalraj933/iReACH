import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import matplotlib.cm as cm

def detect_communities(G):
    """
    Detect communities using Louvain algorithm
    Communities represent natural clusters of hospitals
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities
    partition = community_louvain.best_partition(
        G_undirected, 
        weight='weight'
    )
    
    # Calculate modularity
    modularity = community_louvain.modularity(
        partition, 
        G_undirected
    )
    
    # Add community as node attribute
    nx.set_node_attributes(G, partition, 'community')
    
    return partition, modularity

def analyze_communities(G, partition):
    """Analyze characteristics of detected communities"""
    
    communities_df = pd.DataFrame([
        {
            'hospital_id': node,
            'community': partition[node],
            'region': G.nodes[node].get('region'),
            'capacity': G.nodes[node].get('capacity'),
            'type': G.nodes[node].get('type')
        }
        for node in G.nodes()
    ])
    
    # Community statistics
    community_stats = communities_df.groupby('community').agg({
        'hospital_id': 'count',
        'capacity': 'sum',
        'region': lambda x: x.mode()[0] if len(x) > 0 else None
    }).rename(columns={'hospital_id': 'num_hospitals'})
    
    # Inter vs intra-community edges
    intra_edges = sum(
        1 for u, v in G.edges() 
        if partition[u] == partition[v]
    )
    inter_edges = sum(
        1 for u, v in G.edges() 
        if partition[u] != partition[v]
    )
    
    community_stats['intra_edges_pct'] = (
        intra_edges / G.number_of_edges() * 100
    )
    
    return communities_df, community_stats


