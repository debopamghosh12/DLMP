# --- FILE 2: Algorithm 2 (Update Network) ---
# This file defines the logic for updating the network state
# based on a list of edge changes.
# [Based on Algorithm 2 in the PDF]

import networkx as nx
# Import the helper functions from our first file
try:
    from algorithm_3_mlpa import run_propagation_loop, post_process_communities
except ImportError:
    print("ERROR: algorithm_3_mlpa.py not found.")
    # Define placeholder functions if import fails
    def run_propagation_loop(G, ld, on, T, io):
        print("Error: run_propagation_loop not found")
        return ld
    def post_process_communities(ld, r):
        print("Error: post_process_communities not found")
        return {}


def update_network(G_network, label_distribution, ordered_nodes, edge_changes, T, r, in_op):
    """
    Updates the network state based on edge changes.
    [Implements Algorithm 2]
    """
    print(f"Running Algorithm 2 (Update Network) for {len(edge_changes)} changes...")
    
    nodes_affected = set() # Track nodes that need label propagation

    for edge, label in edge_changes:
        u, v = edge
        nodes_affected.add(u)
        nodes_affected.add(v)

        if label == "add":
            # --- [Algorithm 2, Steps 81-86, 90-99] ---
            print(f"  Adding edge: {(u, v)}")
            
            # Add nodes if they don't exist
            for node in [u, v]:
                if node not in G_network:
                    G_network.add_node(node)
                    G_network.add_edge(node, node) # Add self-loop
                    label_distribution[node] = {node: 1.0} # Initial label
            
            G_network.add_edge(u, v)

        elif label == "delete":
            # --- [Algorithm 2, Steps 101-104] ---
            print(f"  Deleting edge: {(u, v)}")
            if G_network.has_edge(u, v):
                G_network.remove_edge(u, v)

    # Re-calculate initial probabilities *only* for affected nodes
    # This is the "P_e <- 1/K" part
    for node in nodes_affected:
        if node in G_network: # Check if node still exists
            neighbors = list(G_network.neighbors(node))
            num_neighbors = len(neighbors)
            if num_neighbors > 0:
                label_distribution[node] = {n: 1.0 / num_neighbors for n in neighbors}
            else:
                # Node is now isolated (or was deleted and re-added)
                label_distribution[node] = {node: 1.0}

    # Re-sort the entire node list based on new degrees
    ordered_nodes[:] = sorted(G_network.nodes, key=lambda n: G_network.degree(n), reverse=True)
    
    # --- [Algorithm 2, "Speaker(x)"] ---
    # Call the propagation loop to re-stabilize the labels
    # We run it for a few iterations (T_update)
    T_update = max(5, T // 4) # Run for fewer iterations than a full MLPA
    print(f"  Calling 'Speaker' (propagation loop for {T_update} iterations)...")
    label_distribution = run_propagation_loop(G_network, label_distribution, ordered_nodes, T_update, in_op)
    
    # --- Post-process to get new communities ---
    new_communities = post_process_communities(label_distribution, r)
    
    # Return the new state
    return new_communities, G_network, label_distribution, ordered_nodes
