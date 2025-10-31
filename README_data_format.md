Data File Format

This simulation loads all data from external .txt files.

1. Initial Graph File (initial_edges_t0.txt)

This file defines the graph at T=0.

Each line should contain two node IDs separated by a comma.

Node IDs should be integers.

Lines starting with # are ignored as comments.

Example:

# This is a comment
1,2
1,3
2,3


2. Changes Files (changes_t1.txt, etc.)

These files define the list of changes for a single timestamp.

Each line should contain three values separated by commas: node1,node2,action.

action must be either add or delete.

Example:

# Add node 4 as a bridge
4,3,add
4,5,add

# Delete an old edge
1,2,delete
