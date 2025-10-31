#!/usr/bin/env python
# coding: utf-8

# # DOMLPA: Dynamic Overlapping Multi-Label Propagation Algorithm
# 
# This file contains a Python implementation of the DOMLPA algorithm,
# based on the 2015 paper "Overlapping Community Detection in Temporal Networks"
# by Anupama Angadi and P. Suresh Varma.
# 
# This script is designed to be reusable. The algorithm logic is contained
# in functions, and an example of how to run it is provided at the bottom.

# ---
# ### 1. Setup (Imports and Libraries)
# ---
# We use `networkx` for graph objects, `matplotlib` for plotting,
# and `copy` for safely duplicating data.

# In[ ]:


# This cell installs the required libraries in Google Colab
# try:
#     import networkx
#     import matplotlib
# except ImportError:
#     print("Installing networkx and matplotlib...")
#     get_ipython().system('pip install networkx matplotlib')

import networkx as nx
import matplotlib.pyplot as plt
import copy

# ---
# ### 2. Algorithm 3 (MLPA) - The Foundation
# ---
# This section contains the core logic for finding communities in a
# *single, static* graph.

# In[ ]:


def apply_inflation(distribution, in_op):
    """
    Applies the inflation operator (Algorithm 3, Step 3).
    This strengthens strong labels and weakens weak ones.
    """
    if not distribution:
        return {}
    
    inflated_dist = {}
    total = 0
    
    # Raise each probability to the power of 'in_op'
    for label, prob in distribution.items():
        inflated_prob = prob ** in_op
        inflated_dist[label] = inflated_prob
        total += inflated_prob

    if total == 0:
        # Avoid division by zero if all probabilities become 0
        return distribution

    # Normalize the probabilities so they sum to 1 again
    for label in inflated_dist:
        inflated_dist[label] /= total
        
    return inflated_dist

def check_stop_criterion(dist_a, dist_b):
    """
    Checks if the label distributions have stabilized (Algorithm 3, Step 3).
    Returns True if propagation should stop, False otherwise.
    """
    if dist_a is None or dist_b is None:
        return False
    
    # Check if the set of nodes is the same
    if dist_a.keys() != dist_b.keys():
        return False

    # Check if the labels and probabilities for each node are "close enough"
    for node, labels in dist_a.items():
        if node not in dist_b:
            return False
        
        # This is a simple check. A more robust check would use a
        # tolerance (e.g., abs(prob_a - prob_b) < 0.001)
        if labels != dist_b[node]:
            return False
            
    return True

def post_process_communities(label_distribution, r):
    """
    Forms the final communities based on the threshold 'r' (Algorithm 3, Step 4).
    This is where overlaps are officially identified.
    """
    communities = {}
    for node, distribution in label_distribution.items():
        final_labels = []
        if not distribution:
            # If node has no distribution, it's its own community
            communities[node] = [node]
            continue

        # Find all labels with a probability >= threshold 'r'
        for label, prob in distribution.items():
            if prob >= r:
                final_labels.append(label)
        
        if not final_labels:
            # If no label meets the threshold (e.g., r=0.4, dist={'a':0.3, 'b':0.3})
            # just pick the single strongest label as a fallback.
            strongest_label = max(distribution, key=distribution.get)
            final_labels.append(strongest_label)
            
        communities[node] = sorted(list(set(final_labels))) # Store unique, sorted labels
    return communities

def run_propagation_loop(G, T_max, in_op, ordered_nodes, label_distribution, q_thresh):
    """
    The main label propagation loop (Algorithm 3, Step 3).
    This is the "speaking" and "listening" part of the algorithm.
    """
    previous_distribution = None
    
    for i in range(T_max):
        current_distribution = copy.deepcopy(label_distribution)
        visited = set()
        
        for node in ordered_nodes:
            if node in visited:
                continue

            # "Listening" rule: Node collects all labels from its neighbors
            received_labels = {}
            for neighbor in G.neighbors(node):
                # Get the neighbor's current label distribution
                neighbor_dist = label_distribution.get(neighbor, {})
                if not neighbor_dist:
                    # If neighbor has no labels, it sends its own
                    neighbor_dist = {neighbor: 1.0}
                    
                # "Speaking" rule: Neighbor sends its labels
                for label, prob in neighbor_dist.items():
                    received_labels[label] = received_labels.get(label, 0) + prob
            
            if not received_labels:
                # Node has no neighbors (other than self-loop)
                continue

            # Normalize the received probabilities to sum to 1
            total_prob_sum = sum(received_labels.values())
            if total_prob_sum == 0:
                continue
            
            new_distribution = {label: prob / total_prob_sum 
                                for label, prob in received_labels.items()}
            
            # --- Update Condition 'q' (from paper) ---
            # This is a simple interpretation of 'q':
            # We only update if the new distribution is different.
            # A more complex 'q' might check *how* different.
            if new_distribution != current_distribution[node]:
                
                # Apply inflation to strengthen the new choice
                inflated_distribution = apply_inflation(new_distribution, in_op)
                
                # Update the node's distribution for the *next* iteration
                current_distribution[node] = inflated_distribution
                visited.add(node)
        
        # Update the main distribution for all nodes at the end of the iteration
        label_distribution = current_distribution

        # Check if the algorithm has stabilized
        if check_stop_criterion(label_distribution, previous_distribution):
            print(f"    Propagation stabilized after {i+1} iterations.")
            break
        
        previous_distribution = copy.deepcopy(label_distribution)
        
    return label_distribution

def mlpa_algorithm(G_initial, T, r, q, in_op):
    """
    Main function for Algorithm 3 (MLPA).
    Runs the full static community detection on a single graph.
    """
    print("Running Algorithm 3 (MLPA) on initial graph...")
    
    # Create a working copy
    G = G_initial.copy()
    
    # --- Initialization (Algorithm 3, Step 1 & 2) ---
    label_distribution = {}
    
    # Add self-loops to all nodes
    for node in G.nodes():
        G.add_edge(node, node)
    
    # Sort nodes by degree (highest first)
    ordered_nodes = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)
    
    # Initialize probabilities: each node starts with its own label
    for node in G.nodes():
        label_distribution[node] = {node: 1.0}
        
    print("  Initialization complete.")
    
    # --- Propagation (Algorithm 3, Step 3) ---
    print("  Running propagation loop...")
    label_distribution = run_propagation_loop(G, T, in_op, ordered_nodes, label_distribution, q)
    
    # --- Post-processing (Algorithm 3, Step 4) ---
    communities = post_process_communities(label_distribution, r)
    
    print("  MLPA run complete.")
    return communities, G, label_distribution, ordered_nodes


# ---
# ### 3. Algorithm 2 (Update Network) - The Updater
# ---
# This section handles *changes* to the network.

# In[ ]:


def update_network(G, label_distribution, ordered_nodes, changes, T_iter, r, q, in_op):
    """
    Main function for Algorithm 2 (Update Network).
    Applies edge changes and re-runs propagation to stabilize.
    """
    print(f"Running Algorithm 2 (Update Network) for {len(changes)} changes...")
    
    nodes_to_reinit = set()

    for edge, label in changes:
        u, v = edge
        
        if label == "add":
            print(f"  Adding edge: {(u, v)}")
            # --- Steps 81-86 and 90-99 ---
            
            # Add nodes if they don't exist
            for node in [u, v]:
                if node not in G:
                    G.add_node(node)
                    G.add_edge(node, node) # Add self-loop
                    label_distribution[node] = {node: 1.0} # Give it its own label
                    nodes_to_reinit.add(node)
            
            # Add the new edge
            G.add_edge(u, v)
            # Mark nodes for re-initialization
            nodes_to_reinit.add(u)
            nodes_to_reinit.add(v)

        elif label == "delete":
            print(f"  Deleting edge: {(u, v)}")
            # --- Steps 101-104 ---
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                nodes_to_reinit.add(u)
                nodes_to_reinit.add(v)
            else:
                print(f"    Warning: Tried to delete non-existent edge {(u, v)}")

    # --- Re-initialize affected nodes ---
    # The paper suggests a complex update (P_e = 1/K).
    # A simpler, effective way is to reset the affected nodes
    # to their own unique labels, forcing them to "listen" again.
    print(f"  Re-initializing {len(nodes_to_reinit)} affected nodes...")
    for node in nodes_to_reinit:
        if node in G: # Ensure node still exists
             label_distribution[node] = {node: 1.0}

    # Re-sort the node list based on new degrees
    ordered_nodes = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)
    
    # --- Call Speaker Function (Algorithm 2) ---
    # This is done by re-running the propagation loop
    # for a *few* iterations to let the labels re-stabilize.
    print(f"  Calling 'Speaker' (propagation loop for {T_iter} iterations)...")
    label_distribution = run_propagation_loop(G, T_iter, in_op, ordered_nodes, label_distribution, q)
    
    # Get the new communities
    communities = post_process_communities(label_distribution, r)
    
    return communities, G, label_distribution, ordered_nodes


# ---
# ### 4. Algorithm 1 (DOMLPA) - The Manager
# ---
# This is the main "manager" function that runs the full simulation.

# In[ ]:


def draw_graph(G, title, communities):
    """
    Helper function to draw the graph using matplotlib.
    """
    plt.figure(figsize=(8, 6))
    
    # Create a color map from the communities
    # This is a simple way to color nodes. It doesn't show overlaps.
    color_map = {}
    color_index = 0
    community_colors = plt.cm.get_cmap('viridis', len(set(tuple(c) for c in communities.values())))
    
    # Generate a unique color for each unique set of labels
    community_to_color = {}
    for node, labels in communities.items():
        label_tuple = tuple(sorted(labels))
        if label_tuple not in community_to_color:
            community_to_color[label_tuple] = community_colors(color_index)
            color_index += 1
        color_map[node] = community_to_color[label_tuple]
        
    node_colors = [color_map.get(node, 'gray') for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42) # 'seed' makes layout reproducible
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=800, font_size=12, font_weight='bold')
    plt.title(title)
    plt.show()

def run_domlpa_simulation(initial_graph_edges, snapshot_changes_list, T_iter, r_thresh, q_thresh, in_operator, show_plots=True):
    """
    Main function to run the entire DOMLPA simulation.
    This function is NOT hardcoded. You pass it the data and parameters.
    
    Args:
        initial_graph_edges (list): List of edge tuples, e.g., [(1,2), (2,3)]
        snapshot_changes_list (list): A list of lists. Each inner list
            contains the changes for one timestamp.
            e.g., [ [((4,5),"add"), ...], [((1,2),"delete"), ...] ]
        T_iter (int): Max iterations for the propagation loop.
        r_thresh (float): Threshold for post-processing communities.
        q_thresh (float): Threshold for conditional update (currently unused).
        in_operator (int/float): Inflation operator power.
        show_plots (bool): Whether to draw the graphs.
    """
    
    print("=== Starting DOMLPA (Dynamic Overlapping Multi-Label Propagation) ===")
    
    # --- T=0: Initial Graph ---
    # Create the initial graph G0
    G0 = nx.Graph()
    G0.add_edges_from(initial_graph_edges)
    
    # --- Run Algorithm 3 on G0 ---
    communities, G, label_distribution, ordered_nodes = mlpa_algorithm(
        G0, T=T_iter, r=r_thresh, q=q_thresh, in_op=in_operator
    )
    
    print("\n--- Initial Communities (T=0) ---")
    for node, labels in sorted(communities.items()):
        print(f"Node {node}: Belongs to {labels}")
    
    if show_plots:
        draw_graph(G, "Initial Graph (T=0)", communities)
        
    # --- Loop T=1 to T=N ---
    for t, changes in enumerate(snapshot_changes_list, 1):
        print(f"\n--- Processing Timestamp T={t} ---")
        
        # --- Run Algorithm 2 on the changes ---
        communities, G, label_distribution, ordered_nodes = update_network(
            G, label_distribution, ordered_nodes, changes,
            T_iter=5, # Run a few stabilization iterations
            r=r_thresh, q=q_thresh, in_op=in_operator
        )
        
        print(f"\n--- Communities (T={t}) ---")
        for node, labels in sorted(communities.items()):
            print(f"Node {node}: Belongs to {labels}")
        
        if show_plots:
            draw_graph(G, f"Updated Graph (T={t})", communities)
            
    print("\n=== DOMLPA Simulation Complete ===")


# ---
# ### 5. Example Usage
# ---
# This section shows how to run the simulation.
# You can replace this data with your own.

# In[ ]:


if __name__ == "__main__":
    
    # --- 1. Define Your Data ---
    
    # T=0: Two separate communities
    initial_edges = [
        (1, 2), (1, 3), (2, 3),  # Community A
        (5, 6), (5, 7), (6, 7)   # Community B
    ]
    
    # T=1: Add a "bridge" node (4) to connect them
    changes_t1 = [
        ((4, 3), "add"), # Connects node 4 to Community A
        ((4, 5), "add")  # Connects node 4 to Community B
    ]
    
    # T=2: Add a new node (8) to Comm B, delete node 1 from Comm A
    changes_t2 = [
        ((8, 6), "add"),
        ((8, 7), "add"),
        ((1, 2), "delete") # Node 1 starts detaching
    ]
    
    # T=3: Node 1 is now fully isolated
    changes_t3 = [
        ((1, 3), "delete")
    ]
    
    # Pack the changes into a list for the simulation
    all_snapshot_changes = [
        changes_t1,
        changes_t2,
        changes_t3
    ]
    
    # --- 2. Define Your Parameters ---
    
    # Max iterations for the *initial* full run (Algorithm 3)
    param_T_initial = 50 
    
    # Overlap threshold (e.g., 0.3 = 30% "belief" to be in a community)
    param_r_threshold = 0.3 
    
    # Update threshold (not deeply implemented, from paper)
    param_q_threshold = 0.01
    
    # Inflation operator (e.g., 2 = square probabilities)
    param_inflation = 2
    
    # --- 3. Run the Simulation ---
    run_domlpa_simulation(
        initial_graph_edges = initial_edges,
        snapshot_changes_list = all_snapshot_changes,
        T_iter = param_T_initial,
        r_thresh = param_r_threshold,
        q_thresh = param_q_threshold,
        in_operator = param_inflation,
        show_plots = True
    )
