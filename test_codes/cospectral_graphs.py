import networkx as nx
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


t = sp.symbols("t")

X_edges = [
    (0,2), (1,2), (1,3), (1,4), (2,5), (3,5), (4,5),
    (6,7), (6,8), (6,9), (6,10)
]

Y_edges = [
    (0,2), (0,5), (1,3), (1,4), (2,3), (3,4), (3,5),
    (6,7), (6,8), (6,9), (6,10)
]

Z_edges = [
    (0,2), (1,2), (1,3), (1,4), (2,5), (3,5), (4,5),
    (6,7), (7,8), (8,9), (9,6)
]

def make_graph(edges):
    G = nx.Graph()
    G.add_nodes_from(range(11))
    G.add_edges_from(edges)
    return G

def spectra_and_polys(G):
    A = nx.to_numpy_array(G, nodelist=range(11), dtype=int)
    L = nx.laplacian_matrix(G, nodelist=range(11)).toarray()

    A_poly = sp.factor(sp.Matrix(A).charpoly(t).as_expr())
    L_poly = sp.factor(sp.Matrix(L).charpoly(t).as_expr())

    A_spec = np.round(np.linalg.eigvalsh(A), 6)
    L_spec = np.round(np.linalg.eigvalsh(L), 6)

    return A_poly, L_poly, A_spec, L_spec

graphs = {
    "X": make_graph(X_edges),
    "Y": make_graph(Y_edges),
    "Z": make_graph(Z_edges),
}

for name, G in graphs.items():
    A_poly, L_poly, A_spec, L_spec = spectra_and_polys(G)

    print(f"\n{name}")
    print("Adjacency characteristic polynomial:")
    print(A_poly)
    print("Adjacency spectrum:")
    print(A_spec)

    print("Laplacian characteristic polynomial:")
    print(L_poly)
    print("Laplacian spectrum:")
    print(L_spec)

print("\nChecks:")
print("X and Y Laplacian cospectral:",
      spectra_and_polys(graphs["X"])[1] == spectra_and_polys(graphs["Y"])[1])

print("X and Y adjacency cospectral:",
      spectra_and_polys(graphs["X"])[0] == spectra_and_polys(graphs["Y"])[0])

print("X and Z adjacency cospectral:",
      spectra_and_polys(graphs["X"])[0] == spectra_and_polys(graphs["Z"])[0])

print("X and Z Laplacian cospectral:",
      spectra_and_polys(graphs["X"])[1] == spectra_and_polys(graphs["Z"])[1])

X = make_graph(X_edges)
Y = make_graph(Y_edges)
Z = make_graph(Z_edges)

graphs = {
    "X: G + K_{1,4}": X,
    "Y: H + K_{1,4}": Y,
    "Z: G + C_4 + K_1": Z,
}

# Fixed layout:
# vertices 0--5 are the first component,
# vertices 6--10 are the second component.
pos = {
    # First component: vertices 0--5
    0: (-2.5,  1.0),
    1: (-1.0,  1.0),
    2: (-2.0,  0.0),
    3: (-1.0,  0.0),
    4: ( 0.0,  0.0),
    5: (-1.0, -1.0),

    # Second component: vertices 6--10
    6: (1.5,  0.0),
    7: (2.5,  1.0),
    8: (3.5,  0.5),
    9: (3.5, -0.5),
    10: (2.5, -1.0),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (title, G) in zip(axes, graphs.items()):
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=650,
        node_color="#dbeafe",
        edgecolors="#1e3a8a",
        linewidths=1.5,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=2,
        edge_color="#334155",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
    )

    ax.set_title(title, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.show()