import sys
import os
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(dirname(abspath(__file__))), 'modules'))

from graphs import get_graph, compute_matrix_of_pairwise_spectral_distances
import numpy as np
import networkx as nx

matrix_types = ['adjacency', 'laplacian', 'signless_laplacian', 'normalized_laplacian', 'distance_normalized_laplacian']
n = 100
n_graphs_per_param_val = 10

m_values = np.linspace(3, 8, 6)
graphs_BA = [get_graph({'type': 'BA', 'n': n, 'm': m, 'init': {'type': 'complete', 'n': 8}}) for m in m_values for _ in range(n_graphs_per_param_val)]
_ = compute_matrix_of_pairwise_spectral_distances(graphs_BA, matrix_type='adjacency')
_ = compute_matrix_of_pairwise_spectral_distances(graphs_BA, matrix_type='neg_laplacian')