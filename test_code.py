from modules.single_graph_edge_expt import graph_edge_toggling_expt_using_given_graphs_and_scoring_choice
from modules.graphs import get_graph
import networkx as nx
import numpy as np
# %config InlineBackend.figure_format = 'svg'

matrix_choices = ['adjacency', 'neg_laplacian', 'normalized_laplacian', 'neg_normalized_laplacian',
                  'signless_laplacian', 'neg_signless_laplacian', 'distance_normalized_laplacian', 'neg_distance_normalized_laplacian']
# input_choices = ['all_ones', 'identity', 'identity_transf', 'zfs', 'zfs_transf', 'zfs_new']
input_choices = ['all_ones', 'identity', 'zfs', 'zfs_transf', 'zfs_new']

graph_choices = [{'type': 'connected_ER', 'n': 20, 'p': 0.5},
                 {'type': 'connected_RG', 'n': 20, 'r': 0.25},
                 {'type':           'BA', 'n': 20, 'm': 5, 'init': {'type': 'complete', 'n': 5}}]
graphs = [get_graph(graph_choice=choice) for choice in graph_choices]

graph_edge_toggling_expt_using_given_graphs_and_scoring_choice(graph_choices, graphs, matrix_choices=['neg_laplacian'], input_choices=['identity'],
        edge_score_choices=['ETEC_sys_mat'], t_horizon=40, multiple_toggles=True, plot_all_ignoring_low_corr=True)







# from modules.single_graph_edge_expt import graph_edge_toggling_expt_using_given_graphs_and_scoring_choice
# from modules.graphs import get_graph
# import numpy as np
# # %config InlineBackend.figure_format = 'svg'

# graph_choices = [{'type': 'connected_ER', 'n': 20, 'p': 0.5},
#                  {'type': 'connected_RG', 'n': 20, 'r': 0.25},
#                  {'type':           'BA', 'n': 20, 'm': 5, 'init': {'type': 'complete', 'n': 5}}]
# graphs = [get_graph(graph_choice=choice) for choice in graph_choices]
# matrix_choices = ['adjacency', 'neg_laplacian', 'normalized_laplacian', 'neg_normalized_laplacian',
#                   'signless_laplacian', 'neg_signless_laplacian', 'distance_normalized_laplacian', 'neg_distance_normalized_laplacian']
# # input_choices = ['all_ones', 'identity', 'identity_transf', 'zfs', 'zfs_transf', 'zfs_new']
# input_choices = ['all_ones', 'identity', 'zfs', 'zfs_transf', 'zfs_new']

# G = graphs[0]
# rand_edge_order = {(u, v): np.random.rand() for u in G.nodes() for v in G.nodes() if u != v}

# graph_edge_toggling_expt_using_given_graphs_and_scoring_choice(graph_choices, graphs, matrix_choices, input_choices,
#         edge_score_choices=['Wc_spec_dist', 'random'], plot_and_sort_by_this_quantity_instead_of_random_edge_score='Wc_spec_dist',
#         multiple_toggles=True, rand_edge_order=rand_edge_order)#, plot_single_edge_flip_scores=True)#, plot_all_ignoring_low_corr=True)


