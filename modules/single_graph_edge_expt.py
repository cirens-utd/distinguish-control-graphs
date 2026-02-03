import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from tqdm import trange, tqdm
from math import isfinite
from modules.graphs import rank_edges_based_on_toggling_single_edge
from copy import deepcopy

# Default plotting style (optional)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.35,
})

# # Default RNG
# GLOBAL_SEED = 7
# rng = np.random.default_rng(GLOBAL_SEED)


def plot_results(results_per_edge, other_results, options, xlabel=None, ax1=None,
                 ranking_of_edges_by_single_edge_flip=None):
    sorted_data_asc = sorted(results_per_edge.items(), key=lambda item: item[1][options['edge_score_choice']])
    n = len(sorted_data_asc)
    for plot in options['plots']:
        x = list(range(1, n + 1))
        y1 = [item[1][plot['y1']] for item in sorted_data_asc]
        y2 = [item[1][plot['y2']] for item in sorted_data_asc]
        y3 = []
        corr_y1_with_this = y2
        if 'plot_these_graph_matrices_spec_dist' in options and len(options['plot_these_graph_matrices_spec_dist']) > 0:
            for matrix_choice in options['plot_these_graph_matrices_spec_dist']:
                y3.append([item[1][f'{matrix_choice}_spec_dist'] for item in sorted_data_asc])
                if 'use_this_sys_matrix_spec_dist_for_corr' in options and len(options['use_this_sys_matrix_spec_dist_for_corr']) > 0:
                    if matrix_choice == options['use_this_sys_matrix_spec_dist_for_corr'] and matrix_choice != options['graph_matrix_choice']:
                        # print(f"Using {matrix_choice} for correlation.")
                        corr_y1_with_this = y3[-1]

        # pearson_coef = np.corrcoef(y1, y2)[0, 1]
        spearman_coef = scipy.stats.spearmanr(y1, corr_y1_with_this).statistic
        kendalltau_coef = scipy.stats.kendalltau(y1, corr_y1_with_this).statistic
        corr_coef_scores = [spearman_coef, kendalltau_coef]
        
        # plt.figure(figsize=(7.2, 4.6))
        # plot_scatter(x, y2)

        if ax1 is None:
            created_figure = True
            fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
        else:
            created_figure = False

        y1_label = plot['y1']
        ax1.plot(x, y1, marker="o", label=y1_label)
        if xlabel is not None:
            ax1.set_xlabel(xlabel)
        ax1.set_ylabel(y1_label, color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.grid(alpha=0.3)

        if len(y3) > 0:
            ymin_ax, ymax_ax = ax1.get_ylim()
            for i, matrix_choice in enumerate(options['plot_these_graph_matrices_spec_dist']):
                # get current axis limits and rescale y3[i] to fit
                # new_data = np.interp(y3[i], (ymin_ax, ymax_ax), (0, 1))
                # Avoid division by zero
                if np.max(y3[i]) != np.min(y3[i]):
                    y_scaled = ymin_ax + (y3[i] - np.min(y3[i])) * (ymax_ax - ymin_ax) / (np.max(y3[i]) - np.min(y3[i]))
                else:
                    y_scaled = np.full_like(y3[i], (ymin_ax + ymax_ax) / 2)
                ax1.plot(x, y_scaled, marker="o", linestyle="--", color=f"C{i+3}", label=f'{matrix_choice}_spec_dist')
            ax1.legend(loc="upper left")

        y2_label = plot['y2']
        ax2 = ax1.twinx()
        ax2.plot(x, y2, marker="s", linestyle="--", color="C1", label=y2_label)
        ax2.set_ylabel(y2_label, color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        
        if 'plot_single_edge_flip_scores' in options and options['plot_single_edge_flip_scores']:
            y4 = [ranking_of_edges_by_single_edge_flip[item[0]][options['edge_score_choice']] for item in sorted_data_asc]
            ymin_ax2, ymax_ax2 = ax2.get_ylim()
            if np.max(y4) != np.min(y4):
                y4_scaled = ymin_ax2 + (y4 - np.min(y4)) * (ymax_ax2 - ymin_ax2) / (np.max(y4) - np.min(y4))
            else:
                y4_scaled = np.full_like(y4, (ymin_ax2 + ymax_ax2) / 2)
            ax2.plot(x, y4_scaled, marker="o", linestyle="--", color="C2", label=f'{options['edge_score_choice']}_single_edge_flip')
            ax2.legend(loc="upper right")
        
        plt.title(f'Graph: {options['graph_choice']}\n' + 
                  f'System matrix: {options['graph_matrix_choice']}\n' + 
                    # f'($\\lambda_1$ = {other_results['A_lambda_min']:.2g}' +
                    # f', $\\lambda_n$ = {other_results['A_lambda_max']:.2g})\n' + 
                  f'Input: {options['input']}\n' +
                #   f'Correlation coefficient: {pearson_coef:.2g}') +
                  f'Spearman coefficient: {spearman_coef:.2g}\n' +
                  f'Kendall\'s Tau coefficient: {kendalltau_coef:.2g}\n' +
                  f'Gramian: {options['gramian_choice']}')

        if created_figure:
            plt.tight_layout()
            plt.show()
    return corr_coef_scores


# generate a single ER graph
# get its adjacency matrix (include options for other matrices)
# rank its pairs of nodes (whether edge is present or not) by spectral distance due to flipping the edge
    # (include options for other rankings)
# compute spectral distance of Gramian due to this with B = 1 (include options for other metrics and inputs)
# plot given variables (include options for various variables to plot)

def graph_edge_toggling_expt(options, debug_dont_plot=False, multiple_toggles=False, plot_all_ignoring_low_corr=False):
    graphs = options['graphs']
    graph_choices = options['graph_choices']
    n_plots = len(graphs)
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    low_corr_coef_score = None
    for idx, graph in enumerate(graphs):
        ax1 = axes[0, idx]
        temp_options = deepcopy(options)
        temp_options['graph'] = graph
        temp_options['graph_choice'] = graph_choices[idx]
        if multiple_toggles:
            ranking_of_edges_by_single_edge_flip, other_results = rank_edges_based_on_toggling_single_edge(graphs[idx], temp_options)
            results_per_edge_toggled, other_results = rank_edges_based_on_toggling_single_edge(graphs[idx], temp_options, ranking_of_edges=ranking_of_edges_by_single_edge_flip)
        else:
            ranking_of_edges_by_single_edge_flip = None
            results_per_edge_toggled, other_results = rank_edges_based_on_toggling_single_edge(graphs[idx], temp_options)
        corr_coef_score_this_iter = plot_results(results_per_edge_toggled, other_results, temp_options, ax1=ax1,
                                                 ranking_of_edges_by_single_edge_flip=ranking_of_edges_by_single_edge_flip)
        if np.sum(np.array(corr_coef_score_this_iter)) < len(corr_coef_score_this_iter)/2:
            low_corr_coef_score = corr_coef_score_this_iter
    if (low_corr_coef_score is not None) and (not plot_all_ignoring_low_corr):
        plt.close(fig)
        print(f"Skipping plot since correlation coefficients are {[f'{c:.2g}, ' for c in low_corr_coef_score]}.")
    else:
        if multiple_toggles:
            fig.supxlabel(f"Multiple edges flipped. X-axis is the number of the experiment.")
        else:
            fig.supxlabel(f"Edge flips (one at a time) sorted by change in {options['edge_score_choice']}")
        plt.tight_layout()
        if not debug_dont_plot:
            # fig.legend(loc='outside lower center', ncol=2)
            plt.show()
        else:
            plt.close(fig)
    return


def graph_edge_toggling_expt_using_given_graphs_and_scoring_choice(graph_choices, graphs, matrix_choices, input_choices,
        edge_score_choices, gramian_choices=['finite_continuous'],
        t_horizon=1, plot_all_ignoring_low_corr=False, multiple_toggles=False, debug_dont_plot=False,
        plot_these_graph_matrices_spec_dist=[], skip_toggling_of_edges_that_disconnect_graph=False,
        use_this_sys_matrix_spec_dist_for_corr=[], also_plot_random_order_in_cumulative=False, plot_single_edge_flip_scores=False):
    
    options = {'graph_choices': graph_choices,
               'graphs': graphs,
               't_horizon': t_horizon,
               'plot_these_graph_matrices_spec_dist': plot_these_graph_matrices_spec_dist,
               'skip_toggling_of_edges_that_disconnect_graph': skip_toggling_of_edges_that_disconnect_graph,
               'use_this_sys_matrix_spec_dist_for_corr': use_this_sys_matrix_spec_dist_for_corr,
               'also_plot_random_order_in_cumulative': also_plot_random_order_in_cumulative,
               'plot_single_edge_flip_scores': plot_single_edge_flip_scores}
    
    if len(use_this_sys_matrix_spec_dist_for_corr) > 0:
        options['use_this_sys_matrix_spec_dist_for_corr'] = use_this_sys_matrix_spec_dist_for_corr[0]

    for matrix_choice in matrix_choices:
        for input_choice in input_choices:
            for edge_score_choice in edge_score_choices:
                for gramian_choice in gramian_choices:
                    options['gramian_choice'] = gramian_choice
                    if (matrix_choice == 'neg_laplacian' or matrix_choice == 'laplacian') and input_choice == 'all_ones':
                        # neg_laplacian with all-ones input results in a trivial all-ones gramian.
                        continue
                    if gramian_choice == 'pseudo_infinite_continuous' and \
                        matrix_choice in ['adjacency', 'normalized_laplacian', 'distance_normalized_laplacian',
                                'signless_laplacian']:
                        # Pseudo-gramian is only defined for semistable systems
                        continue
                    options['edge_score_choice'] = edge_score_choice
                    options['plots'] = [{'y1': 'sys_mat_spec_dist', 'y2': edge_score_choice}]
                    options['graph_matrix_choice'] = matrix_choice
                    options['input'] = input_choice
                    graph_edge_toggling_expt(options, plot_all_ignoring_low_corr=plot_all_ignoring_low_corr,
                                            multiple_toggles=multiple_toggles, debug_dont_plot=debug_dont_plot)


