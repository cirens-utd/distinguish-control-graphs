import numpy as np

def adjacency_matrix(n, edges):
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A

# Number of vertices
n = 8

# Edges of G
edges_G = [
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (6, 1), (6, 4),
    (7, 1), (7, 4)
]

# Edges of G~
edges_G_tilde = edges_G + [
    (0, 3),
    (6, 7)
]

# Adjacency matrices
A_G = adjacency_matrix(n, edges_G)
A_G_tilde = adjacency_matrix(n, edges_G_tilde)

# Spectra
spec_G = np.linalg.eigvals(A_G)
spec_G_tilde = np.linalg.eigvals(A_G_tilde)

# Sort and clean numerical noise
spec_G = np.sort(np.real_if_close(spec_G))
spec_G_tilde = np.sort(np.real_if_close(spec_G_tilde))

# ℓ1-norm of spectral difference
l1_difference = np.linalg.norm(spec_G - spec_G_tilde, ord=1)

# Output
print("Spectrum of G:\n", spec_G)
print("\nSpectrum of G~:\n", spec_G_tilde)
print("\nℓ1-norm of eigenvalue difference:")
print(l1_difference)