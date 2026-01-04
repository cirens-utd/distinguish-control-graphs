import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from math import isfinite
from modules.graphs import rank_edges_based_on_toggling_single_edge, results_for_cumulative_edge_toggles
from copy import deepcopy

# Default plotting style (optional)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.35,
})

# Default RNG
GLOBAL_SEED = 7
rng = np.random.default_rng(GLOBAL_SEED)


def plot_results(results_per_edge, other_results, options, xlabel=None, ax1=None):
    sorted_data_asc = sorted(results_per_edge.items(), key=lambda item: item[1][options['edge_score_choice']])
    n = len(sorted_data_asc)
    for plot in options['plots']:
        x = list(range(1, n + 1))
        y1 = [item[1][plot['y1']] for item in sorted_data_asc]
        y2 = [item[1][plot['y2']] for item in sorted_data_asc]
        
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

        y2_label = plot['y2']
        ax2 = ax1.twinx()
        ax2.plot(x, y2, marker="s", linestyle="--", color="C1", label=y2_label)
        ax2.set_ylabel(y2_label, color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

        plt.title(f'Graph: {options['graph_choice']}\n' + 
                  f'System matrix: {options['graph_matrix_choice']}\n' + 
                    # f'($\\lambda_1$ = {other_results['A_lambda_min']:.2g}' +
                    # f', $\\lambda_n$ = {other_results['A_lambda_max']:.2g})\n' + 
                  f'Input: {options['input']}')

        if created_figure:
            plt.tight_layout()
            plt.show()
    return


# generate a single ER graph
# get its adjacency matrix (include options for other matrices)
# rank its pairs of nodes (whether edge is present or not) by spectral distance due to flipping the edge
    # (include options for other rankings)
# compute spectral distance of Gramian due to this with B = 1 (include options for other metrics and inputs)
# plot given variables (include options for various variables to plot)

def single_graph_edge_expt(options):
    graph_choices = options['graph_choice']
    n_plots = len(graph_choices)
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    for idx, graph_choice in enumerate(graph_choices):
        ax1 = axes[0, idx]
        temp_options = deepcopy(options)
        temp_options['graph_choice'] = graph_choice
        results_per_edge, other_results = rank_edges_based_on_toggling_single_edge(temp_options, rng)
        plot_results(results_per_edge, other_results, temp_options, ax1=ax1)
    fig.supxlabel(f"Edge flips (one at a time) sorted by change in {options['edge_score_choice']}")
    plt.tight_layout()
    plt.show()
    return


def cumulative_graph_edges_expt(options, debug_dont_plot=False):
    graph_choices = options['graph_choice']
    n_plots = len(graph_choices)
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    for idx, graph_choice in enumerate(graph_choices):
        ax1 = axes[0, idx]
        temp_options = deepcopy(options)
        temp_options['graph_choice'] = graph_choice
        results_per_edge, other_results = results_for_cumulative_edge_toggles(temp_options, rng)
        plot_results(results_per_edge, other_results, temp_options, ax1=ax1)
    fig.supxlabel(f"Edge flips, cumulative, in the order sorted by change in {temp_options['edge_score_choice']}")
    plt.tight_layout()
    if not debug_dont_plot:
        plt.show()
    return


