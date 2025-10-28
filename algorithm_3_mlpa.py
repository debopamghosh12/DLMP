# --- FILE 1: Algorithm 3 (MLPA) ---
# This file defines the core Multi-Label Propagation Algorithm.
# It works on a single, static graph snapshot.
# [Based on Algorithm 3 in the PDF]

import networkx as nx

def apply_inflation(distribution, in_operator):
    """
    Applies the inflation operator (e.g., squaring) and re-normalizes.
    [Part of Algorithm 3, Step 3]
    """
    if not distribution:
        return {}
        
    inflated = {label: prob ** in_operator for label, prob in distribution.items()}
    total = sum(inflated.values())
    if total > 0:
        return {label: prob / total for label, prob in inflated.items()}
    return distribution

def check_stop_criterion(current_dist, previous_dist):
    """
    Checks if the label distributions have stabilized.
    [Part of Algorithm 3, Step 3]
    """
    if previous_dist is None:
        return False
    
    # Check if distributions are equal within a small tolerance
    if current_dist.keys() != previous_dist.keys():
        return False
    
    for node in current_dist:
        if node not in previous_dist or current_dist[node].keys() != previous_dist[node].keys():
            return False
        for label in current_dist[node]:
            if abs(current_dist[node][label] - previous_dist[node].get(label, 0)) > 1e-6:
                return False
    return True # Distributions are effectively the same

def run_propagation_loop(G, label_distribution, ordered_nodes, T, in_op):
    """
    The main label propagation loop (the "Speaker/Listener" part).
    [Algorithm 3, Step 3]
    """
    print("  Running propagation loop...")
    previous_distribution = None
    for i in range(T):
        visited = set()
        
        for node in ordered_nodes:
            if node not in visited:
                received_labels = {}
                for neighbor in G.neighbors(node):
                    if neighbor_distribution := label_distribution.get(neighbor):
                        for label, prob in neighbor_distribution.items():
                            received_labels[label] = received_labels.get(label, 0) + prob
                
                # Update rule (always update if labels received)
                if received_labels:
                    total_prob = sum(received_labels.values())
                    new_distribution = {label: prob / total_prob for label, prob in received_labels.items()}
                    inflated_distribution = apply_inflation(new_distribution, in_op)
                    label_distribution[node] = inflated_distribution # Update the state
                    visited.add(node)
        
        if check_stop_criterion(label_distribution, previous_distribution):
            print(f"    Propagation stabilized after {i+1} iterations.")
            break
        previous_distribution = {node: dist.copy() for node, dist in label_distribution.items()}
    return label_distribution # Return the stabilized distribution

def post_process_communities(label_distribution, r):
    """
    Forms the final communities based on the threshold 'r'.
    [Algorithm 3, Step 4]
    """
    communities = {}
    for node, distribution in label_distribution.items():
        final_labels = set()
        if distribution:
            threshold = r 
            for label, prob in distribution.items():
                if prob >= threshold:
                    final_labels.add(label)
        if not final_labels:
             # Fallback: assign to most probable label
             top_label = max(distribution, key=distribution.get, default=node)
             final_labels.add(top_label)
        communities[node] = list(final_labels)
    return communities

def mlpa_algorithm(G_initial, T, r, q, in_op):
    """
    Main function for Algorithm 3: MLPA.
    Runs the full static algorithm on a given graph.
    """
    print("Running Algorithm 3 (MLPA) on initial graph...")
    
    # --- Step 1: Initialization ---
    G_network = G_initial.copy()
    label_distribution = {}

    # --- Step 2: Setup ---
    for node in G_network.nodes:
        G_network.add_edge(node, node) # Add self-loop
    
    ordered_nodes = sorted(G_network.nodes, key=lambda n: G_network.degree(n), reverse=True)
    
    for node in G_network.nodes:
        neighbors = list(G_network.neighbors(node))
        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            label_distribution[node] = {n: 1.0 / num_neighbors for n in neighbors}
        else:
            label_distribution[node] = {node: 1.0}
    
    print("  Initialization complete.")
    
    # --- Step 3: Label Propagation ---
    label_distribution = run_propagation_loop(G_network, label_distribution, ordered_nodes, T, in_op)
    
    # --- Step 4: Post-processing ---
    communities = post_process_communities(label_distribution, r)
    
    # Return everything needed by Algorithm 1
    return communities, G_network, label_distribution, ordered_nodes
