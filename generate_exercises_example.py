from posets import *
import random

def generate_latex_document(filename="posets_exercises.tex"):
    """
    Generates a LaTeX document with 100 different poset exercises and their solutions.
    """
    latex_code = [
        "\\documentclass{article}",
        "\\usepackage{tikz}",
        "\\usepackage{enumitem}",
        "\\usepackage{geometry}",
        "\\usepackage{fancyhdr}",
        "\\pagestyle{fancy}",
        "\\geometry{margin=1in}",
        "\\begin{document}",
        "\\title{Posets \&\ Lattices. Exercises.}",
        "\\author{Generated Exercises}",
        "\\maketitle",
        "\\newpage"
    ]

    # Keep track of generated matrices to ensure uniqueness
    seen_matrices = set()
    posets = []
    attempts = 0
    max_attempts = 10000  # prevent infinite loop

    # Generate 100 unique posets
    while len(posets) < 100 and attempts < max_attempts:
        attempts += 1
        
        # Generate new matrix
        sparse_matrix = generate_sparse_square_matrix(random_size=True)
        
        # Convert matrix to tuple of tuples for hashing
        matrix_tuple = tuple(tuple(row) for row in sparse_matrix)
        
        # Skip if we've seen this matrix before
        if matrix_tuple in seen_matrices:
            continue
            
        # Add to seen matrices
        seen_matrices.add(matrix_tuple)
        
        # Process the new unique matrix
        positions = find_positions(sparse_matrix)
        graph = create_hasse_diagram(positions)
        nodes = extract_nodes(positions)
        coords_to_letter, _ = create_node_labels(nodes)
        adj_matrix = create_adjacency_matrix(graph, positions)
        
        node_1, node_2, lower_bounds, upper_bounds = find_bounds(adj_matrix, nodes)
        glb, lub = find_extreme_bounds(lower_bounds, upper_bounds)
        is_join, is_meet, is_lattice = check_lattice_properties(adj_matrix, nodes, coords_to_letter)
        
        # Store all the data for this poset
        posets.append({
            'graph': graph,
            'coords_to_letter': coords_to_letter,
            'node_1': node_1,
            'node_2': node_2,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'glb': glb,
            'lub': lub,
            'is_join': is_join,
            'is_meet': is_meet,
            'is_lattice': is_lattice
        })


    # Generate LaTeX for each unique poset
    for i, poset in enumerate(posets, 1):
        latex_code.extend([
            f"\\section*{{Exercise {i}}}",
            "Consider the following poset:",
            "\\begin{center}",
            "\\begin{tikzpicture}[",
            "    vertex/.style={circle, draw=black, thick, fill=white,",
            "        minimum size=24pt, inner sep=2pt, font=\\bfseries},",
            "    edge/.style={thick, -}",
            "]"
        ])

        # Add nodes and edges
        for node in poset['graph'].nodes():
            x, y = node[1]*2, node[0]*2
            label = poset['coords_to_letter'][node]
            latex_code.append(f"    \\node[vertex] ({label}) at ({x},{y}) {{{label}}};")
        
        for edge in poset['graph'].edges():
            source = poset['coords_to_letter'][edge[0]]
            target = poset['coords_to_letter'][edge[1]]
            latex_code.append(f"    \\draw[edge] ({source}) -- ({target});")

        # Add questions and solutions
        latex_code.extend([
            "\\end{tikzpicture}",
            "\\end{center}",
            "",
            f"    \\textbf{{For elements ${poset['coords_to_letter'][poset['node_1']]}$ and ${poset['coords_to_letter'][poset['node_2']]}$, answer the following questions:}}",
            "\\begin{enumerate}",
            "    \\item What are the lower bounds?",
            "    \\item What are the upper bounds?",
            "    \\item What is the greatest lower bound (GLB)?",
            "    \\item What is the least upper bound (LUB)?",
            "\\end{enumerate}",
            "    \\hspace*{3ex} \\textbf{Additional questions:}",
            "\\begin{enumerate}",
            "    \\setcounter{enumi}{4}",
            "    \\item Is this poset a meet-semilattice?",
            "    \\item Is this poset a join-semilattice?",
            "    \\item Is this poset a lattice?",
            "\\end{enumerate}",
            "",
            "\\textbf{Solutions:}",
            "\\begin{enumerate}",
            f"    \\item Lower bounds: {'{' + ', '.join(poset['coords_to_letter'][lb] for lb in poset['lower_bounds']) + '}' if poset['lower_bounds'] else 'Do not exist'}",
            f"    \\item Upper bounds: {'{' + ', '.join(poset['coords_to_letter'][ub] for ub in poset['upper_bounds']) + '}' if poset['upper_bounds'] else 'Do not exist'}",
            f"    \\item LUB: {poset['coords_to_letter'][poset['lub']] if poset['lub'] else 'Does not exist'}",
            f"    \\item GLB: {poset['coords_to_letter'][poset['glb']] if poset['glb'] else 'Does not exist'}",
            f"    \\item Meet-semilattice: {'Yes' if poset['is_meet'] else 'No'}",
            f"    \\item Join-semilattice: {'Yes' if poset['is_join'] else 'No'}",
            f"    \\item Lattice: {'Yes' if poset['is_lattice'] else 'No'}",
            "\\end{enumerate}",
            "\\newpage"
        ])

    # End document
    latex_code.append("\\end{document}")

    # Write to file
    with open(filename, 'w') as f:
        f.write('\n'.join(latex_code))

if __name__ == "__main__":
    generate_latex_document() 