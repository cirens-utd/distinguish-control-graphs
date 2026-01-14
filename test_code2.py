import networkx as nx
import numpy as np

def remove_effectively_zero_elements(matrix, rtol=1e-10):
    """Sets elements with absolute value below tol to zero."""
    matrix_cleaned = matrix.copy()
    matrix_cleaned[np.abs(matrix) < rtol*np.max(np.abs(matrix))] = 0.0
    return matrix_cleaned

def create_star_graph():
    """Creates a K1,4 star graph with 5 vertices."""
    G = nx.Graph()
    # Node 0 is the center, nodes 1-4 are leaves
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    return G

def create_c4_union_k1_graph():
    """Creates a graph with a C4 cycle and an isolated vertex."""
    H = nx.Graph()
    # Cycle C4 on nodes 0, 1, 2, 3
    H.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    # Isolated node 4 is automatically added to the node set
    H.add_node(4)
    return H

# --- Graph Creation ---
G1 = create_star_graph()
G2 = create_c4_union_k1_graph()

# --- Verify they are non-isomorphic ---
# A star graph is connected, the other is not. 
is_connected_G1 = nx.is_connected(G1)
is_connected_G2 = nx.is_connected(G2)
print(f"G1 is connected: {is_connected_G1}")
print(f"G2 is connected: {is_connected_G2}")

# Since one is connected and the other is not, they cannot be isomorphic.
is_isomorphic = nx.is_isomorphic(G1, G2)
print(f"G1 and G2 are isomorphic: {is_isomorphic}\n")

# --- Verify they are cospectral ---
# Get adjacency matrices
A1 = nx.adjacency_matrix(G1)
A2 = nx.adjacency_matrix(G2)

# Calculate eigenvalues using numpy
# np.linalg.eigvals returns unsorted eigenvalues, so we sort for comparison
L1, Q1 = np.linalg.eigh(A1.toarray())
L2, Q2 = np.linalg.eigh(A2.toarray())

L1 = np.sort(L1)
L2 = np.sort(L2)

print(f"Eigenvalues of G1: {np.round(L1, 5)}")
print(f"Eigenvalues of G2: {np.round(L2, 5)}")

# Check if eigenvalues are numerically close
are_cospectral = np.allclose(L1, L2, atol=np.max(np.abs(L1)) * 1e-10)
print(f"G1 and G2 are cospectral: {are_cospectral}")

T = Q1 @ Q2.T
A2_test = remove_effectively_zero_elements(T.T @ A1.toarray() @ T)
if not np.allclose(A2_test, A2.toarray(), atol=1e-10):
    raise ValueError("The transformation matrix T does not map A1 to A2.")

exit()