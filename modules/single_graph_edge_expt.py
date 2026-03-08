import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from tqdm import trange, tqdm
from math import isfinite
from graphs import rank_edges_based_on_toggling_single_edge, get_graph_param_from_graph_choice
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


def adjust_ylabel_for_paper(ylabel, options):
    new_label = ylabel
    if options['label_figure_for_paper']:
        match ylabel:
            case 'sys_mat_spec_dist':
                matrix_name = options["graph_matrix_choice"]
                if matrix_name == 'adjacency':
                    matrix_name = 'Adjacency'
                elif matrix_name == 'neg_laplacian':
                    matrix_name = 'Laplacian'
                new_label = f'{matrix_name} Spectral Distance'
            case _:
                pass
    return new_label


def plot_results(results_per_edge, other_results, options, xlabel=None, ax1=None,
                 ranking_of_edges_by_single_edge_flip=None, sort_by=None, debug_dont_plot=False):
    sorted_data_asc = sorted(results_per_edge.items(), key=lambda item: item[1][sort_by])
    n = len(sorted_data_asc)
    
    plot = options['plots'][0]
    if (not debug_dont_plot) and len(options['plots']) > 1:
        raise ValueError(f"Multiple plots specified, only one expected: {options['plots']}")
    
    all_corr_coef_scores = {}
    
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

        for axis in plot:
            if axis == 'y1':
                continue
            quantity = [item[1][plot[axis]] for item in sorted_data_asc]
            all_corr_coef_scores[(plot['y1'], plot[axis])] = {'spearmanr': scipy.stats.spearmanr(y1, quantity).statistic,
                                                            'kendalltau': scipy.stats.kendalltau(y1, quantity).statistic}
        
        # plt.figure(figsize=(7.2, 4.6))
        # plot_scatter(x, y2)

        if ax1 is None:
            created_figure = True
            fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
        else:
            created_figure = False
        
        y1_label = adjust_ylabel_for_paper(plot['y1'], options)
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

        y2_label = adjust_ylabel_for_paper(plot['y2'], options)
        ax2 = ax1.twinx()
        ax2.plot(x, y2, marker="s", linestyle="--", color="C1", label=y2_label)
        ax2.set_ylabel(y2_label, color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        
        if 'third_plot' in options and options['third_plot'] is not None:
            if options['third_plot'] == 'single_edge_flip_scores':
                y4 = [ranking_of_edges_by_single_edge_flip[item[0]][options['edge_score_choice']] for item in sorted_data_asc]
                label = f'{options['edge_score_choice']}_single_edge_flip'
            else:
                y4 = [item[1][options['third_plot']] for item in sorted_data_asc]
                label = options['third_plot']
            ymin_ax2, ymax_ax2 = ax2.get_ylim()
            if np.max(y4) != np.min(y4):
                y4_scaled = ymin_ax2 + (y4 - np.min(y4)) * (ymax_ax2 - ymin_ax2) / (np.max(y4) - np.min(y4))
            else:
                y4_scaled = np.full_like(y4, (ymin_ax2 + ymax_ax2) / 2)
            ax2.plot(x, y4_scaled, marker="o", linestyle="--", color="C2", label=label)
            ax2.legend(loc="upper right")
        
        coefficients_in_title = f'Spearman rank coefficient: {spearman_coef:.2g}\n' + \
                                f'Kendall rank coefficient: {kendalltau_coef:.2g}'
        if options['label_figure_for_paper']:
            title = coefficients_in_title
        else:
            title = f'Graph: {options['graph_choice']}\n' + \
                f'System matrix: {options['graph_matrix_choice']}\n' + \
                f'Input: {options['input']}\n' + \
                f'{coefficients_in_title}\n' + \
                f'Gramian: {options['gramian_choice']}'
                # f'($\\lambda_1$ = {other_results['A_lambda_min']:.2g}' + \
                # f', $\\lambda_n$ = {other_results['A_lambda_max']:.2g})\n' + \
        if options['edge_score_choice'] == 'random':
            title += f'\nRandom order of edges'

        plt.title(title)

        if created_figure:
            plt.tight_layout()
            plt.show()
    
    return corr_coef_scores, all_corr_coef_scores


def compute_average_corr_coef_scores_across_all_graphs_and_write_to_file(all_corr_coef_score_results_for_all_graphs, options):
    results_file = options['results_file']
    corr_coef_scores_avg = deepcopy(all_corr_coef_score_results_for_all_graphs[0])
    corr_coef_scores_std = deepcopy(all_corr_coef_score_results_for_all_graphs[0])
    densities = options['graph_metrics']['density']
    density_avg = np.mean(densities)
    density_std = np.std(densities)

    if results_file is not None:
        f = open(results_file, 'a')
        # format: n_graphs, sys_matrix, graph_type, n_nodes, graph_param, density_avg, density_std, input, edge_score, quantity_pair_in_quotations, metric_1, corr_coef_avg_1, corr_coef_std_1, metric_2, corr_coef_avg_2, corr_coef_std_2
        f.write(f"{len(options['graphs'])}, ")
        f.write(f"{options['graph_matrix_choice']}, ")
        graph_type = options['graph_choices'][0]['type']
        f.write(f"{graph_type}, ")
        f.write(f"{options['graph_choices'][0]['n']}, ")
        param = get_graph_param_from_graph_choice(options['graph_choices'][0])
        f.write(f"{param}, ")
        f.write(f"{density_avg:.3g}, ")
        f.write(f"{density_std:.3g}, ")
        f.write(f"{options['input']}, ")
        f.write(f"{options['edge_score_choice']}, ")
    
    for quantity_pair in corr_coef_scores_avg.keys():
        results_for_this_pair = [all_corr_coef_score_results_for_all_graphs[j][quantity_pair] for j in range(len(all_corr_coef_score_results_for_all_graphs))]
        corr_coef_scores_avg[quantity_pair] = {}
        corr_coef_scores_std[quantity_pair] = {}
        for metric in results_for_this_pair[0].keys():
            values = [result[metric] for result in results_for_this_pair]
            corr_coef_scores_avg[quantity_pair][metric] = np.mean(values)
            corr_coef_scores_std[quantity_pair][metric] = np.std(values)
        
        if results_file is not None:
            f.write(f"\" {quantity_pair}\", ")
            for metric in corr_coef_scores_avg[quantity_pair].keys():
                f.write(f"{metric}, {corr_coef_scores_avg[quantity_pair][metric]:.3g}, {corr_coef_scores_std[quantity_pair][metric]:.3g}, ")
    
    if results_file is not None:
        f.write("\n")
        f.close()
    
    return corr_coef_scores_avg, corr_coef_scores_std


# generate a single ER graph
# get its adjacency matrix (include options for other matrices)
# rank its pairs of nodes (whether edge is present or not) by spectral distance due to flipping the edge
    # (include options for other rankings)
# compute spectral distance of Gramian due to this with B = 1 (include options for other metrics and inputs)
# plot given variables (include options for various variables to plot)

def graph_edge_toggling_expt(options, debug_dont_plot=False, multiple_toggles=False, plot_all_ignoring_low_corr=False,
                             sort_by=None, results_file=None):
    graphs = options['graphs']
    graph_choices = options['graph_choices']
    n_plots = len(graphs)
    fig_height = 4
    if options['label_figure_for_paper']:
        fig_height = 3.3
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5 * n_plots, fig_height), squeeze=False)
    low_corr_coef_score = None
    high_corr_coef_score = None
    all_corr_coef_score_results_for_all_graphs = []
    options['graph_metrics'] = {}
    options['graph_metrics']['density'] = []
    for idx, graph in enumerate(tqdm(graphs, desc="Processing graphs")):
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
        corr_coef_score_this_iter, all_corr_coef_scores_this_iter = plot_results(results_per_edge_toggled, other_results, temp_options,
                ax1=ax1, ranking_of_edges_by_single_edge_flip=ranking_of_edges_by_single_edge_flip, sort_by=sort_by, debug_dont_plot=debug_dont_plot)
        all_corr_coef_score_results_for_all_graphs.append(all_corr_coef_scores_this_iter)
        sum_of_corr_coef_scores_this_iter = np.sum(np.array(corr_coef_score_this_iter))
        if sum_of_corr_coef_scores_this_iter < 0.5*len(corr_coef_score_this_iter):
            low_corr_coef_score = corr_coef_score_this_iter
        elif sum_of_corr_coef_scores_this_iter > 0.7*len(corr_coef_score_this_iter):
            high_corr_coef_score = corr_coef_score_this_iter
        options['graph_metrics']['density'].append(nx.density(graph))
    
    compute_average_corr_coef_scores_across_all_graphs_and_write_to_file(all_corr_coef_score_results_for_all_graphs, options)

    if (not debug_dont_plot) and ((low_corr_coef_score is not None) and (high_corr_coef_score is None)) and (not plot_all_ignoring_low_corr):
        plt.close(fig)
        print(f"Skipping plot since correlation coefficients are {[f'{c:.2g}, ' for c in low_corr_coef_score]}.")
    else:
        if multiple_toggles:
            if options.get('first_quantity_plotted_is_edge_score', False):
                fig.supxlabel(f"Number of edges flipped")
            else:
                fig.supxlabel(f"Multiple edges flipped. X-axis is the number of the experiment.")
        else:
            fig.supxlabel(f"Edge flips (one at a time) sorted by change in {options['edge_score_choice']}")
        plt.tight_layout()
        if not debug_dont_plot:
            # fig.legend(loc='outside lower center', ncol=2)
            if options.get('fig_output_file_name', None) is not None:
                plt.savefig(options['fig_output_file_name'])
            plt.show()
        else:
            plt.close(fig)
    return


def graph_edge_toggling_expt_using_given_graphs_and_scoring_choice(graph_choices, graphs, matrix_choices, input_choices,
        edge_score_choices, gramian_choices=['finite_continuous'], t_horizon=1, plot_all_ignoring_low_corr=False,
        multiple_toggles=False, debug_dont_plot=False, plot_these_graph_matrices_spec_dist=[],
        skip_toggling_of_edges_that_disconnect_graph=False, use_this_sys_matrix_spec_dist_for_corr=[],
        sort_by=None, plot_this=None, third_plot=None, label_figure_for_paper=False,
        plot_single_edge_flip_scores=False, rand_edge_order={}, fig_output_file_name=None,
        results_file=None, other_pairs_of_quantities_to_plot=[], t_horizon_setting_for_ETEC='2n'):
    
    options = {'graph_choices': graph_choices,
               'graphs': graphs,
               't_horizon': t_horizon,
               'plot_these_graph_matrices_spec_dist': plot_these_graph_matrices_spec_dist,
               'skip_toggling_of_edges_that_disconnect_graph': skip_toggling_of_edges_that_disconnect_graph,
               'use_this_sys_matrix_spec_dist_for_corr': use_this_sys_matrix_spec_dist_for_corr,
               'plot_single_edge_flip_scores': plot_single_edge_flip_scores,
               'results_file': results_file,
               'rand_edge_order': rand_edge_order,
               't_horizon_setting_for_ETEC': t_horizon_setting_for_ETEC,
               'third_plot': third_plot,
               'label_figure_for_paper': label_figure_for_paper,
               'fig_output_file_name': fig_output_file_name}
    
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

                    # if edge_score_choice == 'random' and plot_this_quantity_instead_of_rand_edge_score is not None:
                    #     quantity_to_plot_if_rand_edge_order = plot_this_quantity_instead_of_rand_edge_score
                    # else:
                    #     quantity_to_plot_if_rand_edge_order = edge_score_choice
                    # options['plots'] = [{'y1': 'sys_mat_spec_dist', 'y2': quantity_to_plot_if_rand_edge_order}]
                    if plot_this is None:
                        plot_this = edge_score_choice
                    options['plots'] = [{'y1': 'sys_mat_spec_dist', 'y2': plot_this}]
                    if plot_this == edge_score_choice:
                        options['first_quantity_plotted_is_edge_score'] = True
                    else:
                        options['first_quantity_plotted_is_edge_score'] = False
                    options['plots'].extend(other_pairs_of_quantities_to_plot)
                
                    options['graph_matrix_choice'] = matrix_choice
                    options['input'] = input_choice

                    if sort_by is None:
                        sort_by = edge_score_choice
                    
                    if isinstance(graphs[0], list):
                        n_expt = len(graphs)
                        n_graphs = len(graphs[0])
                        results = []
                        for expt_no in range(n_expt):
                            options['graphs'] = [graphs[expt_no][k] for k in range(n_graphs)]
                            options['graph_choices'] = [graph_choices[expt_no] for _ in range(n_graphs)]
                            results.append(graph_edge_toggling_expt(options, multiple_toggles=multiple_toggles, debug_dont_plot=True,
                                                                    sort_by=sort_by, results_file=results_file))
                    else:
                        graph_edge_toggling_expt(options, plot_all_ignoring_low_corr=plot_all_ignoring_low_corr,
                                                multiple_toggles=multiple_toggles, debug_dont_plot=debug_dont_plot,
                                                sort_by=sort_by)


