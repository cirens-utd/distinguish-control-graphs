import sys
import os
from os.path import abspath
sys.path.insert(0, os.path.join(abspath(os.getcwd()), 'modules'))

from single_graph_edge_expt import graph_edge_toggling_expt_using_given_graphs_and_scoring_choice
from graphs import get_graph
import numpy as np

matrix_choices = ['adjacency', 'neg_laplacian']
input_choices = ['all_ones', 'identity', 'zfs', 'zfs_transf', 'zfs_new']

n_graphs = 100

graph_choice_sets = [[{'type': 'connected_ER', 'n': 20, 'p': 0.3},
                      {'type': 'connected_RG', 'n': 20, 'r': 0.25},
                      {'type':           'BA', 'n': 20, 'm': 3, 'init': {'type': 'complete', 'n': 5}}],
                     [{'type': 'connected_ER', 'n': 20, 'p': 0.5},
                      {'type': 'connected_RG', 'n': 20, 'r': 0.5},
                      {'type':           'BA', 'n': 20, 'm': 5, 'init': {'type': 'complete', 'n': 5}}],
                     [{'type': 'connected_ER', 'n': 20, 'p': 0.7},
                      {'type': 'connected_RG', 'n': 20, 'r': 0.75},
                      {'type':           'BA', 'n': 20, 'm': 8, 'init': {'type': 'complete', 'n': 8}}]]
list_of_graph_lists = [[[get_graph(graph_choice=choice) for _ in range(n_graphs)] for choice in graph_choice_set] for graph_choice_set in graph_choice_sets]


# controlled-density perturbations
fractions_of_removals_in_randomly_flipped_edges = np.linspace(0, 1, 21)
for fraction in fractions_of_removals_in_randomly_flipped_edges:
    for graph_choices_set, graphs in zip(graph_choice_sets, list_of_graph_lists):
        graph_edge_toggling_expt_using_given_graphs_and_scoring_choice(graph_choices_set, graphs,
            matrix_choices=matrix_choices, input_choices=input_choices,
            edge_score_choices=['random'], plot_this='density',
            sample_multiple_edges_uniformly_num_trials=100, sort_by='graph_edit_distance',
            fraction_of_removals_in_randomly_flipped_edges=fraction,
            results_file='controlled_density_edge_flip_results_no_averaging.csv',
            other_pairs_of_quantities_to_plot=[
                {'y1': 'sys_mat_spec_dist', 'y2': 'Wc_spec_dist'},
                {'y1': 'density', 'y2': 'Wc_spec_dist'}],
            t_horizon=1, average_over_all_graphs=False)