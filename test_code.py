from modules.single_graph_edge_expt import graph_edge_toggling_expt
# %config InlineBackend.figure_format = 'svg'

graph_setting_ER = {'type': 'connected_ER', 'n': 20, 'p': 0.5}
graph_setting_RG = {'type': 'connected_RG', 'n': 20, 'r': 0.25}

graph_choices = [graph_setting_ER, graph_setting_RG]
# graph_choices = [graph_setting_ER]
matrix_choices = ['adjacency', 'laplacian', 'normalized_laplacian', 'signless_laplacian', 'distance_normalized_laplacian']
input_choices = ['all_ones', 'identity', 'zfs', 'zfs_new', 'identity_transf', 'zfs_transf']

options =  {'graph_choice': graph_choices,
            'edge_score_choice': 'Wc_spec_dist', 
            't_horizon': 1, 
            'plots': [{'y1': 'sys_mat_spec_dist', 'y2': 'Wc_spec_dist'}]}

for matrix_choice in matrix_choices:
    for input_choice in input_choices:
        if matrix_choice == 'laplacian' and input_choice == 'all_ones':
            # Laplacian with all-ones input results in a trivial all-ones gramian.
            continue
        options['graph_matrix_choice'] = matrix_choice
        options['input'] = input_choice
        graph_edge_toggling_expt(options, debug_dont_plot=True)#, plot_all_ignoring_low_corr=True)




# from modules.single_graph_edge_expt import graph_edge_toggling_expt

# graph_setting_ER = {'type': 'connected_ER', 'n': 20, 'p': 0.5}
# graph_setting_RG = {'type': 'connected_RG', 'n': 20, 'r': 0.25}

# graph_choices = [graph_setting_ER, graph_setting_RG]
# matrix_choices = ['adjacency', 'laplacian', 'normalized_laplacian', 'signless_laplacian', 'distance_normalized_laplacian']
# input_choices = ['all_ones', 'identity', 'zfs', 'zfs_new']

# options =  {'graph_choice': graph_choices,
#             't_horizon': 1}

# edge_score_choices = ['Wc_trace_diff', 'Wc_lambda_min_diff', 'Wc_pinv_trace_diff', 'Wc_logdet_diff']

# for matrix_choice in matrix_choices:
#     for input_choice in input_choices:
#         for edge_score_choice in edge_score_choices:
#                 options['edge_score_choice'] = edge_score_choice
#                 options['plots'] = [{'y1': 'sys_mat_spec_dist', 'y2': edge_score_choice}]
#                 options['graph_matrix_choice'] = matrix_choice
#                 options['input'] = input_choice
#                 graph_edge_toggling_expt(options, cumulative=True, debug_dont_plot=True)




