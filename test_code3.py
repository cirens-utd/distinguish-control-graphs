from modules.single_graph_edge_expt import graph_edge_toggling_expt_using_given_graphs_and_scoring_choice
from modules.graphs import get_graph
import numpy as np
# %config InlineBackend.figure_format = 'svg'

graph_choices = [{'type': 'connected_ER', 'n': 20, 'p': 0.5},
                 {'type': 'connected_RG', 'n': 20, 'r': 0.25},
                 {'type':           'BA', 'n': 20, 'm': 5, 'init': {'type': 'complete', 'n': 5}}]
n_graphs = 5
graphs = [[get_graph(graph_choice=choice) for _ in range(n_graphs)] for choice in graph_choices]
matrix_choices = ['neg_laplacian']
# matrix_choices = ['adjacency', 'neg_laplacian', 'normalized_laplacian', 'neg_normalized_laplacian',
#                   'signless_laplacian', 'neg_signless_laplacian', 'distance_normalized_laplacian', 'neg_distance_normalized_laplacian']
input_choices = ['zfs_transf']
# input_choices = ['all_ones', 'identity', 'zfs', 'zfs_transf', 'zfs_new']


graph_edge_toggling_expt_using_given_graphs_and_scoring_choice(graph_choices, graphs, matrix_choices, input_choices,
        edge_score_choices=['Wc_spec_dist'], multiple_toggles=True, results_file='results.txt')