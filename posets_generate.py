import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def generate_sparse_square_matrix(size=4, random_size=False):
    """
    Generates a sparse square matrix that represents a poset (positions or coordinates of the poset).
    Each row must have at least one '1'.
    5% chance of having exactly one '1' per row.
    80% chance of having single nodes at top and bottom levels with multiple nodes in middle rows.
    
    Args:
        size (int): Size of the matrix (by default 4)
        random_size (bool): If True, randomly chooses size between 4-6 with bias towards 4

    Returns:
        matrix (np.ndarray): A sparse binary matrix (2D numpy array) where '1' represents the position of a node.
    """
    if random_size:
        # 60% chance for 4, 20% for 5, 20% for 6
        size = random.choices([4, 5, 6], weights=[60, 20, 20])[0]
    
    matrix = np.zeros((size, size), dtype=int)
    
    # 5% chance for exactly one '1' per row
    if random.random() < 0.05:
        for row in range(size):
            col = random.randint(0, row)  # Ensure we stay in lower triangle
            matrix[row][col] = 1
    
    # 80% chance for single nodes at top and bottom with multiple middle nodes
    elif random.random() < 0.80:
        # Place one '1' in the first row (highest level)
        top_col = random.randint(0, size-1)
        matrix[0][top_col] = 1
        
        # Place one '1' in the last row (lowest level)
        bottom_col = random.randint(0, size-1)
        matrix[size-1][bottom_col] = 1
        
        # Add multiple '1's in the middle rows (at least one per row)
        for row in range(1, size-1):
            # Ensure at least one '1' per row
            col = random.randint(0, row)
            matrix[row][col] = 1
            
            # Possibly add more '1's
            additional_positions = [(row, j) for j in range(row) if j != col]
            if additional_positions:
                num_additional = random.randint(0, min(2, len(additional_positions)))
                chosen = random.sample(additional_positions, num_additional)
                for i, j in chosen:
                    matrix[i][j] = 1
    
    # Remaining 15% - more random distribution but still at least one per row
    else:
        for row in range(size):
            # Ensure at least one '1' per row
            col = random.randint(0, row)
            matrix[row][col] = 1
            
            # Add additional '1's with higher probability
            positions = [(row, j) for j in range(row) if j != col]
            if positions:
                num_additional = random.randint(1, min(3, len(positions)))
                chosen = random.sample(positions, num_additional)
                for i, j in chosen:
                    matrix[i][j] = 1
    
    return matrix


def find_positions(matrix):
    """
    Organize the positions of '1's in a sparse matrix into hierarchical 'levels' based on their row and column indices.

    This function is used to assign specific coordinates to nodes in a poset.
    The levels are determined by flipping the row indexing so that the bottom row is the lowest level.
    Levels are then sorted from left to right (columns) and bottom to top (rows).

    Args:
        matrix (np.ndarray): A sparse binary matrix (2D numpy array) where '1' represents the position of a node.

    Returns:
        dict: A dictionary where keys are level names (e.g., "bottom_pos", "middle_1"),
              and values are lists of tuples representing the coordinates (y, x) of '1's in each level.
              Here, the first number (y) indicates the vertical level, and the second number (x) indicates the horizontal position.
    """

    rows, cols = matrix.shape
    levels = [f"middle_{i}" for i in range(1, rows - 1)]  # create level names
    levels = ["bottom_pos"] + levels + ["top"]  # add "bottom_pos" and "top"

    positions = {level: [] for level in levels}

    # populate positions based on matrix values
    for row in range(rows):
        for col in range(cols):
            if matrix[row, col] == 1: # ensure that only positions in the matrix containing a 1 are considered (valid nodes)
                translated_row = rows - row  # reverse row indexing (bottom row becomes "1")
                translated_col = col + 1   # shift the column indexing to start from 1 instead of 0.

                # determine the level for the current position
                if translated_row == 1:
                    positions["bottom_pos"].append((translated_row, translated_col))
                elif translated_row == rows:
                    positions["top"].append((translated_row, translated_col))
                else:
                    level_name = f"middle_{translated_row - 1}"
                    positions[level_name].append((translated_row, translated_col))

    # remove empty levels
    positions = {key: value for key, value in positions.items() if value}

    return positions


def create_hasse_diagram(positions, max_edges_per_node=None):
    """
    Create a Hasse diagram based on given node positions with control over the number of connections.

    Args:
        positions (dict): A dictionary where keys are levels (e.g., "bottom_pos", "middle_1") and values
                          are lists of tuples representing the (level, horizontal position) of nodes (y,x).
        max_edges_per_node (int, optional): Maximum number of outgoing edges allowed per node.
                                             If None, the number of edges is not explicitly limited.

    Returns:
        graph (nx.DiGraph): A directed graph representing the Hasse diagram with controlled connectivity.
    """
    graph = nx.DiGraph()  # directed graph for Hasse diagram

    for level, nodes in positions.items():
        for node in nodes:
            y, x = node  # extract vertical (level) and horizontal position
            graph.add_node(node, level=y, x=x)  # add nodes to the graph with their attributes

    # Create edges between nodes in consecutive levels with some control over connectivity
    levels = list(positions.keys())
    for i in range(len(levels) - 1):
        current_level = positions[levels[i]]
        next_level = positions[levels[i + 1]]

        for source_node in current_level:
            # Randomly shuffle the next level nodes to create random edges
            random.shuffle(next_level)
            connections_made = 0

            # Loop over the next level and add edges based on the maximum edges per node
            for target_node in next_level:
                if source_node[0] < target_node[0]:  # ensure vertical hierarchy (Y should increase)
                    graph.add_edge(source_node, target_node)
                    connections_made += 1

                    # Control the maximum number of outgoing connections for this source node
                    if max_edges_per_node and connections_made >= max_edges_per_node:
                        break  # Stop adding edges once the limit is reached

    # Ensure no isolated nodes (no islands), ensuring connectivity without making everything fully connected
    for node in graph.nodes:
        if graph.degree(node) == 0:
            # Find a node to connect it to (preferably from previous level to maintain hierarchy)
            possible_sources = [n for n, d in graph.degree() if d > 0]
            if possible_sources:
                source = random.choice(possible_sources)
                graph.add_edge(source, node)

    return graph


def create_adjacency_matrix(graph, positions):
    """
    Creates an adjacency matrix for a Hasse diagram represented by a directed graph (DiGraph) `graph`.
    The adjacency matrix is a square matrix where the element at row 'i' and column 'j' represents
    the shortest path distance between nodes 'i' and 'j'. As nodes are not connected to themselves,
    the diagonal elements are set to 0. Evaluate the whole matrix is redundunt as it is symmetric.
    Fill only position in the upper triangle to optimize performance.

    Args:
        graph (nx.DiGraph): A graph representing the Hasse diagram of a randomly generated poset.

        positions (dict): A dictionary where each key represents a level (e.g., "bottom_pos", "middle_1")
                          and its value is a list of nodes (tuples representing (y,x) coordinates).

    Returns:
        adj_matrix (np.ndarray): An N x N adjacency matrix where 'N' is the number of nodes in the graph.
                                 The matrix contains the shortest path distances between nodes.
                                 If there is no path between two nodes, the matrix entry is set to 0.

    """

    sorted_nodes = sorted(graph.nodes, key=lambda x: (x[0], x[1]))  # Sort by Y, then X

    # generate an empty matrix (initially all zeros)
    size = len(sorted_nodes)
    adj_matrix = np.zeros((size, size), dtype=int)

    # create the adjacency matrix by calculating the shortest path distance between nodes
    for i in range(size):
        for j in range(i + 1, size):  # only compute for the upper triangle, matrix is symmetric
            node_i = sorted_nodes[i]
            node_j = sorted_nodes[j]

            # calculate the shortest path distance between nodes
            if nx.has_path(graph, node_i, node_j):
                path_length = nx.shortest_path_length(graph, source=node_i, target=node_j)
                adj_matrix[i][j] = path_length  # set the upper triangle entry
                adj_matrix[j][i] = path_length  # set the symmetric entry (for the upper triangle)

    # set all lower triangle elements to zero
    for i in range(size):
        for j in range(i):
            adj_matrix[i][j] = 0  # Set the lower triangle to 0
    return adj_matrix


# extract all nodes from the position relations
def extract_nodes(positions):
    nodes = []
    for level_nodes in positions.values():
        nodes.extend(level_nodes)

# sort nodes first by Y (row), then by X (column)
    sorted_nodes = sorted(nodes, key=lambda node: (node[0], node[1]))
    return sorted_nodes

def select_random_nodes(nodes):
    """"
    Selects two distinct random nodes from a list of nodes.
    
    Args:
        nodes (list): A list of nodes to select from.
    
    Returns:
        tuple: A tuple containing two distinct random nodes.
    """
    while True:
        selected_nodes = random.sample(nodes, 2)
        node_1, node_2 = selected_nodes

        if node_1 != node_2:
            return node_1, node_2


def find_bounds(adj_matrix, nodes, node_1=None, node_2=None):
    """
    Finds both lower and upper bounds of two nodes.
    If nodes are not specified, selects random nodes.
    """
    node_indices = {node: index for index, node in enumerate(nodes)}
    coords_to_letter, letter_to_coords = create_node_labels(nodes)

    # Use provided nodes or select random ones
    if node_1 is None or node_2 is None:
        node_1, node_2 = select_random_nodes(nodes)
    
    index_1 = node_indices[node_1]
    index_2 = node_indices[node_2]

    # Find all reachable nodes using path existence
    n = len(nodes)
    reachable_matrix = np.zeros((n, n), dtype=bool)
    adj_bool = adj_matrix > 0

    # Compute transitive closure to find all reachable nodes
    reachable_matrix = adj_bool.copy()
    for k in range(n):
        reachable_matrix = reachable_matrix | (reachable_matrix @ adj_bool)

    # Find lower bounds (nodes that can reach both selected nodes)
    lower_bound_indices = set()
    for i in range(n):
        if (reachable_matrix[i, index_1] or i == index_1) and \
           (reachable_matrix[i, index_2] or i == index_2):
            lower_bound_indices.add(i)

    # Find upper bounds (nodes that can be reached from both selected nodes)
    upper_bound_indices = set()
    for i in range(n):
        if (reachable_matrix[index_1, i] or i == index_1) and \
           (reachable_matrix[index_2, i] or i == index_2):
            upper_bound_indices.add(i)

    # Convert indices to nodes and filter by Y-coordinate
    lowest_y = min(node_1[0], node_2[0])
    highest_y = max(node_1[0], node_2[0])

    lower_bounds = [nodes[i] for i in lower_bound_indices 
                   if nodes[i][0] <= lowest_y]
    upper_bounds = [nodes[i] for i in upper_bound_indices 
                   if nodes[i][0] >= highest_y]

    return node_1, node_2, lower_bounds, upper_bounds


def find_extreme_bounds(lower_bounds, upper_bounds):
    """
    Finds the greatest lower bound (GLB) and least upper bound (LUB).
    In our coordinate system:
    - GLB is the node with the HIGHEST Y-coordinate among lower bounds
    - LUB is the node with the LOWEST Y-coordinate among upper bounds
    - If there are multiple nodes with the same Y-coordinate, the (upper/lower) bound does not exist.
    """
    # Find GLB (highest Y-coordinate among lower bounds)
    glb = None
    if lower_bounds:
        max_y = max(node[0] for node in lower_bounds)
        highest_nodes = [node for node in lower_bounds if node[0] == max_y]
        if len(highest_nodes) == 1:
            glb = highest_nodes[0]

    # Find LUB (lowest Y-coordinate among upper bounds)
    lub = None
    if upper_bounds:
        min_y = min(node[0] for node in upper_bounds)
        lowest_nodes = [node for node in upper_bounds if node[0] == min_y]
        if len(lowest_nodes) == 1:
            lub = lowest_nodes[0]

    return glb, lub


def create_node_labels(nodes):
    """
    Creates a mapping between coordinate tuples and letter labels.
    Labels are assigned from bottom to top, left to right, starting with 'a'.
    
    Args:
        nodes (list): List of coordinate tuples (Y, X)
        In the coordinate system (Y,X):
        - Lower Y values are at the bottom (start with 'a')
        - Higher Y values are at the top
        - Lower X values are on the left
        - Higher X values are on the right
    
    Returns:
        dict: Two dictionaries - coords_to_letter and letter_to_coords
    """
    import string
    letters = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
    
    # Sort nodes:
    # 1. First by Y ascending (bottom to top)
    # 2. Then by X ascending (left to right)
    sorted_nodes = sorted(nodes, key=lambda x: (x[0], x[1]))  # Removed negative sign
    
    # Create bidirectional mappings
    coords_to_letter = {coord: letters[i] for i, coord in enumerate(sorted_nodes)}
    letter_to_coords = {letters[i]: coord for i, coord in enumerate(sorted_nodes)}
    
    return coords_to_letter, letter_to_coords


def check_lattice_properties(adj_matrix, nodes, coords_to_letter):
    """
    Checks if the poset is a join-semilattice, meet-semilattice, and lattice.
    
    Args:
        adj_matrix (np.ndarray): Adjacency matrix of the poset
        nodes (list): List of nodes
        coords_to_letter (dict): Mapping from coordinates to letters
    
    Returns:
        tuple: (is_join_semilattice, is_meet_semilattice, is_lattice)
    """
    n = len(nodes)
    is_join = True
    is_meet = True
    
    # Check all possible pairs
    for i in range(n):
        for j in range(i + 1, n):
            node_1, node_2 = nodes[i], nodes[j]
            
            # Find bounds for this specific pair
            _, _, lower_bounds, upper_bounds = find_bounds(adj_matrix, nodes, node_1, node_2)
            glb, lub = find_extreme_bounds(lower_bounds, upper_bounds)
            
            # Check LUB and GLB existence
            if not lub:
                is_join = False
            if not glb:
                is_meet = False
            
            # If both properties are already false, we can stop checking
            if not is_join and not is_meet:
                break
                
        if not is_join and not is_meet:
            break
    
    # A lattice exists if both join and meet semilattices exist
    is_lattice = is_join and is_meet
    
    return is_join, is_meet, is_lattice


if __name__ == "__main__":
    sparse_matrix = generate_sparse_square_matrix()
    positions = find_positions(sparse_matrix)
    graph = create_hasse_diagram(positions)
    
    nodes = extract_nodes(positions)
    coords_to_letter, _ = create_node_labels(nodes)
    
    adj_matrix = create_adjacency_matrix(graph, positions)
    
    node_1, node_2, lower_bounds, upper_bounds = find_bounds(adj_matrix, nodes)
    glb, lub = find_extreme_bounds(lower_bounds, upper_bounds)
    
    # Check lattice properties
    is_join, is_meet, is_lattice = check_lattice_properties(adj_matrix, nodes, coords_to_letter)