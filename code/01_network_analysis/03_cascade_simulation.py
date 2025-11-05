def simulate_cascade(G, initial_infected, propagation_prob=0.3, max_steps=10):
    """
    Simulate cyberattack cascade through network
    
    Parameters:
    - initial_infected: list of initially compromised hospitals
    - propagation_prob: probability of spreading along edge
    - max_steps: maximum cascade steps
    
    Returns:
    - Timeline of infection spread
    - Final infected set
    """
    infected = set(initial_infected)
    cascade_timeline = [infected.copy()]
    
    for step in range(max_steps):
        newly_infected = set()
        
        # Check all edges from currently infected nodes
        for infected_node in infected:
            for neighbor in G.successors(infected_node):
                if neighbor not in infected:
                    # Probability weighted by connection strength
                    edge_weight = G[infected_node][neighbor]['weight']
                    max_weight = max(
                        data['weight'] 
                        for _, _, data in G.edges(data=True)
                    )
                    normalized_weight = edge_weight / max_weight
                    
                    # Probability increases with connection strength
                    infection_prob = propagation_prob * normalized_weight
                    
                    if np.random.random() < infection_prob:
                        newly_infected.add(neighbor)
        
        if not newly_infected:
            break  # Cascade stopped
        
        infected.update(newly_infected)
        cascade_timeline.append(infected.copy())
    
    return cascade_timeline, infected

def analyze_cascade_scenarios(G, critical_hospitals, n_simulations=100):
    """
    Run multiple cascade simulations from different starting points
    """
    results = []
    
    for initial_hospital in critical_hospitals:
        cascade_sizes = []
        
        for _ in range(n_simulations):
            _, final_infected = simulate_cascade(
                G, 
                [initial_hospital],
                propagation_prob=0.3
            )
            cascade_sizes.append(len(final_infected))
        
        results.append({
            'initial_hospital': initial_hospital,
            'mean_cascade_size': np.mean(cascade_sizes),
            'max_cascade_size': np.max(cascade_sizes),
            'min_cascade_size': np.min(cascade_sizes),
            'std_cascade_size': np.std(cascade_sizes)
        })
    
    return pd.DataFrame(results).sort_values(
        'mean_cascade_size', 
        ascending=False
    )