import sys
import os
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(abspath(os.getcwd()), 'modules'))

from graphs import get_graph, get_input, compute_matrix_of_pairwise_spectral_distances, plot_average_spectral_distance_for_same_param_val, real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix, compute_zfs_transf_Gramian_spectra_of_given_two_graphs, spectral_distance
from my_control_lib import finite_time_gramian
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

matrix_types = ['adjacency', 'laplacian', 'signless_laplacian', 'normalized_laplacian', 'distance_normalized_laplacian']
n = 100


n_graphs_per_param_val_for_paper_2 = 5

fig, ax = plt.subplots(figsize=(4,3))
ax2 = ax.twinx()

p_values_for_paper_2 = np.arange(0, 1 + 1e-10, 0.025)
p_values_for_adj_for_paper_2 = np.concatenate([[0.005], p_values_for_paper_2[1:-1], [0.995]])

graphs_ER_for_adj_for_paper_2 = [get_graph({'type': 'ER', 'n': n, 'p': p}) for p in p_values_for_adj_for_paper_2 for _ in range(n_graphs_per_param_val_for_paper_2)]
_ = plot_average_spectral_distance_for_same_param_val(graphs_ER_for_adj_for_paper_2, matrix_type='adjacency', param_name='p',
                                                        param_values=p_values_for_adj_for_paper_2,
                                                        n_graphs_per_param_val=n_graphs_per_param_val_for_paper_2,
                                                        figfile='figures/adj_avg_spec_dist.pdf', axis=ax)#, plot_violin_and_medians=True)

graphs_ER_for_paper_2 = [get_graph({'type': 'ER', 'n': n, 'p': p}) for p in p_values_for_paper_2 for _ in range(n_graphs_per_param_val_for_paper_2)]
_ = plot_average_spectral_distance_for_same_param_val(graphs_ER_for_paper_2, matrix_type='laplacian', param_name='p', param_values=p_values_for_paper_2,
                                                        n_graphs_per_param_val=n_graphs_per_param_val_for_paper_2,
                                                        figfile='figures/lapl_avg_spec_dist.pdf', axis=ax2, right_axis=True),#, plot_violin_and_medians=True)

plt.show()