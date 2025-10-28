# --- FILE 3: Algorithm 1 (DOMLPA) ---
# This is the main runnable script that manages the
# dynamic network over time.
# [Based on Algorithm 1 in the PDF]

import networkx as nx
import matplotlib.pyplot as plt # <-- ADD THIS IMPORT

# -------------------------------------------------------------------
# --- 1. IMPORTS FROM OTHER FILES ---
# -------------------------------------------------------------------
# We try to import the functions from our other two .py files.
# If this fails, it's because the files are not in the same folder.
try:
    # Import the (static) MLPA function from Algorithm 3
    from algorithm_3_mlpa import mlpa_algorithm
    # Import the (updater) function from Algorithm 2
    from algorithm_2_update import update_network
except ImportError:
    # ###############################################################
    # --- ERROR FALLBACK (PLACEHOLDER FUNCTIONS) ---
    # If the import fails, print a fatal error.
    # We define "dummy" functions here so the script
    # doesn't crash, but it will print errors to show
    # that the *real* functions were not called.
    # ###############################################################
    print("FATAL ERROR: Could not import from 'algorithm_3_mlpa.py' or 'algorithm_2_update.py'")
    print("Please make sure all three files are in the same directory.")
    
    def mlpa_algorithm(G, T, r, q, in_op): 
        print("Error: mlpa_algorithm (placeholder) was called")
        return {}, G, {}, [] # Return empty/dummy values
    def update_network(G, ld, on, ec, T, r, in_op):
        print("Error: update_network (placeholder) was called")
        return {}, G, ld, on # Return empty/dummy values

# -------------------------------------------------------------------
# --- 2. ALGORITHM 1 (DOMLPA) - THE "MANAGER" FUNCTION ---
# -------------------------------------------------------------------
def domlpa(graph_snapshots, T, r, q, in_op):
    """
    Main function for Algorithm 1: DOMLPA.
    Manages the network evolution over time.
    
    Args:
        graph_snapshots (list): 
            - Item 0 is the initial nx.Graph (G0)
            - Items 1..N are lists of edge changes for each timestamp
        T (int): Max propagation iterations (for MLPA & Update)
        r (float): Post-processing threshold
        q (float): Conditional update threshold
        in_op (float): Inflation operator
    """
    
    # --- STEP 1: PROCESS INITIAL GRAPH (T=0) ---
    # [This corresponds to Algorithm 1, Step 58: "run MLPA for G0"]
    print("=== Processing T=0 (Initial Graph) ===")
    G0 = graph_snapshots[0]
    
    # ###############################################################
    # --- Call Algorithm 3 ---
    # We call the imported 'mlpa_algorithm' to process the
    # first snapshot. This gives us our starting communities
    # and the initial state of the network.
    # ###############################################################
    communities, G_network, label_distribution, ordered_nodes = mlpa_algorithm(
        G0, T=T, r=r, q=q, in_op=in_op
    )
    
    # --- VISUALIZE T=0 ---
    print("  Displaying initial graph (T=0)...")
    plt.figure(figsize=(8, 5))
    plt.title("Graph at T=0 (Initial State)")
    # We draw 'G_network' (not G0) because mlpa_algorithm
    # adds self-loops, which we want to see.
    nx.draw(G_network, with_labels=True, font_weight='bold', node_color='skyblue', edge_color='gray')
    plt.show() # This will pause the script until you close the window
    
    # --- PRINT COMMUNITIES T=0 ---
    print("\n--- Initial Communities (T=0) ---")
    for node, labels in communities.items():
        print(f"Node {node}: Belongs to {labels}")

    # ---------------------------------------------------------------
    # --- STEP 2: LOOP THROUGH TIME (T=1, T=2, ...) ---
    # ---------------------------------------------------------------
    # [This corresponds to Algorithm 1, Step 59: "for t in 1 thru T"]
    for t in range(1, len(graph_snapshots)):
        print(f"\n=== Processing Timestamp T={t} ===")
        
        # Get the list of edge changes for this timestamp
        edge_changes_t = graph_snapshots[t]
        
        # ###############################################################
        # --- Call Algorithm 2 ---
        # [This corresponds to Algorithm 1, Step 74: "updateNetwork(G, E)"]
        # We call the imported 'update_network' function.
        # We pass it the *current state* of the network
        # (G_network, label_distribution, etc.) and the list
        # of changes. It returns the *new state*.
        # ###############################################################
        communities, G_network, label_distribution, ordered_nodes = update_network(
            G_network, label_distribution, ordered_nodes, edge_changes_t,
            T=T, r=r, in_op=in_op
        )
        
        # --- VISUALIZE T>0 ---
        print(f"  Displaying updated graph (T={t})...")
        plt.figure(figsize=(8, 5))
        plt.title(f"Graph at T={t} (After Updates)")
        nx.draw(G_network, with_labels=True, font_weight='bold', node_color='skyblue', edge_color='gray')
        plt.show() # This will pause the script until you close the window
        
        # --- PRINT COMMUNITIES T>0 ---
        print(f"\n--- Communities (T={t}) ---")
        for node, labels in communities.items():
            print(f"Node {node}: Belongs to {labels}")

# -------------------------------------------------------------------
# --- 3. EXAMPLE USAGE (THE MAIN SCRIPT) ---
# -------------------------------------------------------------------
# This block only runs if you execute this file directly
# (e.g., "python algorithm_1_domlpa_main.py")
if __name__ == "__main__":
    
    print("=== Starting DOMLPA (Dynamic Overlapping Multi-Label Propagation) ===")
    
    # --- DEFINE PARAMETERS ---
    # These are the global parameters for the whole simulation
    T_iter = 20 # Max iterations for propagation
    r_thresh = 0.3 # Post-processing threshold (keep low for overlaps)
    q_thresh = 0.01 # Conditional update threshold
    in_operator = 2 # Inflation operator
    
    # --- DEFINE THE SNAPSHOTS ---
    # We create a "story" for our graph to evolve.
    
    # ###############################################################
    # --- Snapshot 0 (T=0): The initial graph ---
    # ###############################################################
    # Two communities that are not connected.
    G0 = nx.Graph()
    G0.add_edges_from([
        (1, 2), (1, 3), (2, 3) # Community 1
    ])
    G0.add_edges_from([
        (5, 6), (5, 7), (6, 7) # Community 2
    ])
    
    # ###############################################################
    # --- Snapshot 1 (T=1): A list of edge changes ---
    # ###############################################################
    # We add a new "bridge" node (4) that connects to BOTH communities.
    # This should create an overlap.
    changes_t1 = [
        ((4, 3), "add"), # Connect node 4 to Community 1
        ((4, 5), "add")  # Connect node 4 to Community 2
    ]
    
    # ###############################################################
    # --- Snapshot 2 (T=2): More changes ---
    # ###############################################################
    # We add node 8 to Community 2
    # And we delete node 1 from Community 1
    changes_t2 = [
        ((8, 6), "add"),
        ((8, 7), "add"),
        ((1, 2), "delete"), # Node 1 starts detaching
        ((1, 3), "delete")
    ]
    
    # --- COMBINE SNAPSHOTS ---
    # This list is the main input for our 'domlpa' function
    graph_snapshots = [
        G0,          # T=0
        changes_t1,  # T=1
        changes_t2   # T=2
    ]
    
    # ###############################################################
    # --- RUN THE ALGORITHM ---
    # This is the final step that starts the whole process.
    # ###############################################################
    domlpa(graph_snapshots, T_iter, r_thresh, q_thresh, in_operator)
    print("\n=== DOMLPA Finished ===")