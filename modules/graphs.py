import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from numpy.linalg import eigvalsh
from my_control_lib import (
    finite_time_gramian, 
    pseudo_gramian_for_semistable_A_inf_horizon,
    finite_time_discrete_gramian, 
    compute_controllability_rank, 
    finite_horizon_gramian_through_integration, 
    compute_energy_transfer_edge_centrality
)
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy


# Default RNG
GLOBAL_SEED = 7
rng = np.random.default_rng(GLOBAL_SEED)


# ---------- ER: connected sampler ----------
def connected_erdos_renyi(n, p, rng, max_attempts=200, need_connected=True):
    """Sample a connected G(n,p)."""
    for _ in range(max_attempts):
        G = nx.fast_gnp_random_graph(n, p, seed=int(rng.integers(0, 2**32 - 1)))
        if (not need_connected) or nx.is_connected(G):
            return G
    raise RuntimeError(f"Could not sample a connected G(n,{p:.3f}) after {max_attempts} tries.")


def connected_random_geometric(n, r, rng, max_attempts=200, need_connected=True):
    """Sample a connected random geometric graph."""
    for _ in range(max_attempts):
        G = nx.random_geometric_graph(n, r, seed=int(rng.integers(0, 2**32 - 1)))
        if (not need_connected) or nx.is_connected(G):
            return G
    raise RuntimeError(f"Could not sample a connected RG(n,{r:.3f}) after {max_attempts} tries.")


def get_graph(graph_choice):
    match graph_choice['type']:
        case 'ER':
            G = connected_erdos_renyi(n=graph_choice['n'], p=graph_choice['p'], rng=rng, need_connected=False)
        case 'RG':
            G = connected_random_geometric(n=graph_choice['n'], r=graph_choice['r'], rng=rng, need_connected=False)
        case 'connected_ER':
            G = connected_erdos_renyi(n=graph_choice['n'], p=graph_choice['p'], rng=rng)
        case 'connected_RG':
            G = connected_random_geometric(n=graph_choice['n'], r=graph_choice['r'], rng=rng)
        case 'BA':
            initial_graph = get_graph(graph_choice['init']) if 'init' in graph_choice else nx.complete_graph(graph_choice['m'])
            G = nx.barabasi_albert_graph(n=graph_choice['n'], m=int(round(graph_choice['m'])), seed=int(rng.integers(0, 2**32 - 1)),
                                         initial_graph=initial_graph, create_using=nx.Graph)
        case 'complete':
            G = nx.complete_graph(n=graph_choice['n'])
        case _:
            raise ValueError(f"Graph choice {graph_choice} unsupported.")
    return G


def get_graph_param_from_graph_choice(graph_choice):
    match graph_choice['type']:
        case 'connected_ER':
            return graph_choice['p']
        case 'connected_RG':
            return graph_choice['r']
        case 'BA':
            return graph_choice['m']
        case _:
            raise ValueError(f"Unsupported graph type: {graph_choice['type']}")


# ---------- get system matrix related to graph ----------
def get_system_matrix_from_graph(G, matrix_choice="adjacency"):

    if matrix_choice[0:4] == 'neg_':
        return -get_system_matrix_from_graph(G, matrix_choice=matrix_choice[4:])
    if matrix_choice[0] == '-':
        return -get_system_matrix_from_graph(G, matrix_choice=matrix_choice[1:])
    
    A = nx.to_numpy_array(G, dtype=np.float64)
    D = np.diag(A.sum(axis=1))
    I = np.eye(A.shape[0])

    match matrix_choice:
        case "adjacency": # | "A":
            S = A
        case "laplacian": # | "L":
            S = D - A
        case "signless_laplacian": # | "Q":
            S = D + A
        case "normalized_laplacian": # | "L_normal":
            deg = A.sum(axis=1)
            D_recip = np.diag(1.0 / np.sqrt(deg))
            S = I - D_recip @ A @ D_recip
        case "distance_normalized_laplacian": # | "L_dist_normal":
            dist = shortest_path(A, directed=nx.is_directed(G))
            t = dist.sum(axis=1)
            T_inv_sqrt = np.diag(1.0 / np.sqrt(t))
            S = I - T_inv_sqrt @ dist @ T_inv_sqrt
        case _:
            raise ValueError(f"Matrix choice {matrix_choice} unsupported.")
    
    # if np.isnan(S).any():
    #     raise RuntimeError(f"System matrix has NaN elements for matrix choice {matrix_choice}.")
    
    return S


def get_input(G, options, B_old=None, T=None, modified_matrix=False):
    n = len(G)
    match options['input']:
        case 'all_ones':
            B = np.ones((n, 1))
        case 'identity':
            B = np.eye(n)
        case 'identity_transf':
            if modified_matrix:
                B = T.T# @ np.eye(n)
            else:
                B = np.eye(n)
        case 'zfs':
            if modified_matrix and (B_old is not None):
                B = B_old
            else:
                B = zero_forcing_set_greedy(G)
        case 'zfs_transf':
            if modified_matrix and (B_old is not None):
                B = T.T @ B_old
            else:
                B = zero_forcing_set_greedy(G)
        case 'zfs_new':
            B = zero_forcing_set_greedy(G)
        case 'none':
            B = None
        case _:
            raise ValueError(f'Unsupported input {options['input']}.')
    
    return B


def real_eigval_for_potentially_nonsymmetric_matrix(M):
    tol = 1e-4*np.max(np.abs(M))
    if not scipy.linalg.issymmetric(M, atol=tol, rtol=0):
        raise RuntimeError(f"Matrix is not symmetric. Norm of differnce: {np.linalg.norm(M - M.T)}. Max val: {np.max(np.abs(M))}, tol: {tol}")
    else:
        eigvals_M = eigvalsh(M)
    # eigvals_M = eigvals(M)
    # if np.max(np.abs(eigvals_M.imag)) > 1e-10*np.max(np.abs(eigvals_M)):
    #     raise RuntimeError("Matrix has significant imaginary eigenvalues.")
    # return eigvals_M.real
    return eigvals_M


def real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(M):
    if not scipy.linalg.issymmetric(M, atol=1e-10*np.max(np.abs(M))):
        raise RuntimeError(f"Matrix is not symmetric. Norm of differnce: {np.linalg.norm(M - M.T)}. Max val: {np.max(np.abs(M))}")
    else:
        eigvals_M, eigvecs_M = np.linalg.eigh(M)
    return eigvals_M, eigvecs_M


def rank_edges_based_on_toggling_single_edge(G, options, ranking_of_edges=None):
    # If a ranking of edges is provided, edges are toggled cumulatively instead of one-by-one.
    # To toggle edges one-by-one, do not provide ranking_of_edges
    if 'skip_toggling_of_edges_that_disconnect_graph' not in options:
        options['skip_toggling_of_edges_that_disconnect_graph'] = False
    skip_toggling_of_edges_that_disconnect_graph = options['skip_toggling_of_edges_that_disconnect_graph']

    if 'plot_these_graph_matrices_spec_dist' not in options or len(options['plot_these_graph_matrices_spec_dist']) == 0:
        options['plot_these_graph_matrices_spec_dist'] = []
        compute_spec_dist_of_other_sys_matrices = False
    else:
        compute_spec_dist_of_other_sys_matrices = True

    if 'use_pseudo_gramian' in options and options['use_pseudo_gramian']:
        if 'laplacian' not in options['graph_matrix_choice']:
            raise ValueError("Pseudo-gramian is only defined for semistable system matrices.")
        else:
            use_pseudo_gramian = True
    else:
        use_pseudo_gramian = False
    
    if (not skip_toggling_of_edges_that_disconnect_graph) and compute_spec_dist_of_other_sys_matrices:
        raise ValueError("Skipping toggling of edges that disconnect graph is not supported for computing spectral distance of all system matrices.")
    
    t = options['t_horizon']
    edge_score_choice = options['edge_score_choice']
    
    A = get_system_matrix_from_graph(G, options['graph_matrix_choice'])
    B = get_input(G, options)
    
    match options['t_horizon_setting_for_ETEC']:
        case '2n':
            t_for_ETEC = 2 * G.number_of_nodes()
        case 'n':
            t_for_ETEC = G.number_of_nodes()
        case '2n-1':
            t_for_ETEC = 2 * G.number_of_nodes() - 1
        case 'n-1':
            t_for_ETEC = G.number_of_nodes() - 1
        case _:
            t_for_ETEC = options['t_horizon_setting_for_ETEC']

    if edge_score_choice == 'ETEC':
        ETEC = compute_energy_transfer_edge_centrality(get_system_matrix_from_graph(G, 'adjacency'), t_for_ETEC)
    elif edge_score_choice == 'ETEC_sys_mat':
        ETEC = compute_energy_transfer_edge_centrality(A, t_for_ETEC)
    
    match options['gramian_choice']:
        case 'finite_continuous':
            gramian_func = finite_time_gramian
        case 'finite_discrete':
            gramian_func = finite_time_discrete_gramian
        case 'pseudo_infinite_continuous':
            gramian_func = pseudo_gramian_for_semistable_A_inf_horizon
        case _:
            raise ValueError(f"Gramian choice {options['gramian_choice']} unsupported.")
    
    W = gramian_func(A, B, t=t)
    # # calculate_W_stuff = True
    # # if np.isinf(W).any() and edge_score_choice == 'ETEC':
    # #     calculate_W_stuff = False
    # # elif np.isinf(W).any():
    # #     raise ValueError("Gramian is not well-defined.")
    # calculate_W_stuff = 'ETEC' not in edge_score_choice
    
    A_eigvals, Q1 = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(A)
    # if calculate_W_stuff:
    W_eigvals = real_eigval_for_potentially_nonsymmetric_matrix(W)
    W_trace = float(np.trace(W))
    W_pinv = np.linalg.pinv(W)
    W_pinv_trace = float(np.trace(W_pinv))
    W_logdet = logdet_psd(W, W_eigval=W_eigvals)[0]
    W_rank = np.linalg.matrix_rank(W)

    if compute_spec_dist_of_other_sys_matrices:
        other_system_matrix_choices = ['adjacency', 'neg_laplacian', 'normalized_laplacian', 'signless_laplacian', 'distance_normalized_laplacian']
        other_system_matrices = [get_system_matrix_from_graph(G, matrix_choice) for matrix_choice in other_system_matrix_choices]
        other_system_matrices_eigvals = [real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(S)[0] for S in other_system_matrices]

    if options['input'] == 'zfs':
        laplacian = get_system_matrix_from_graph(G, matrix_choice="neg_laplacian")
        W_test = finite_time_gramian(laplacian, B, t=t)
        if np.linalg.matrix_rank(W_test) < A.shape[0]:
            raise RuntimeError("System is not controllable with zero forcing set as input.")

    other_results = {}
    other_results['A_lambda_min'] = float(np.min(A_eigvals))
    other_results['A_lambda_max'] = float(np.max(A_eigvals))
    # if calculate_W_stuff:
    W_lambda_min = float(np.min(W_eigvals))

    nodes = list(G.nodes())
    n = len(nodes)
    results_per_edge = {}

    G_mod = deepcopy(G)

    for i in range(n):
        u = nodes[i]
        if G.has_edge(u, u):
            raise ValueError(f"Non-simple graphs not supported.")

    if ranking_of_edges is None:
        if nx.is_directed(G):
            edges = [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            edges = [(i, j) for i in range(n) for j in range(n) if j > i + 1]
    else:
        sorted_data_asc = sorted(ranking_of_edges.items(), key=lambda item: item[1][options['edge_score_choice']])
        edges = [item[0] for item in sorted_data_asc]

    for i, j in tqdm(edges, leave=False, disable=True):
        u, v = nodes[i], nodes[j]

        # Modify the graph: toggle edge presence
        if ranking_of_edges is None:
            G_mod = deepcopy(G)
        G_mod_before_edge_toggle = deepcopy(G_mod)
        if G_mod.has_edge(u, v):
            G_mod.remove_edge(u, v)
        else:
            G_mod.add_edge(u, v)

        if skip_toggling_of_edges_that_disconnect_graph and (not nx.is_connected(G_mod)):
            # print(f"Skipping disconnected graph obtained by removing edge ({u}, {v}).")
            G_mod = G_mod_before_edge_toggle
            continue

        A_mod = get_system_matrix_from_graph(G_mod, options['graph_matrix_choice'])
        if np.isnan(A_mod).any():
            if not nx.is_connected(G_mod):
                # print(f"Skipping disconnected graph obtained by removing edge ({u}, {v}) because system matrix choice is {options['graph_matrix_choice']}.")
                G_mod = G_mod_before_edge_toggle
                continue
            else:
                raise RuntimeError("NaN encountered in modified system matrix.")
        
        A_mod_eigvals, Q2 = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(A_mod)
        T = Q1 @ Q2.T
        A_spec_dist = spectral_distance(A, A_mod, M1_eigvals=A_eigvals, M2_eigvals=A_mod_eigvals)
        B_mod = get_input(G, options, B_old=B, T=T, modified_matrix=True)
        if use_pseudo_gramian:
            W_mod = pseudo_gramian_for_semistable_A_inf_horizon(A_mod, B_mod)
        else:
            W_mod = gramian_func(A_mod, B_mod, t=t)
        
        # if calculate_W_stuff:
        W_mod_eigvals = real_eigval_for_potentially_nonsymmetric_matrix(W_mod)
        Wc_spec_dist = spectral_distance(W, W_mod, M1_eigvals=W_eigvals, M2_eigvals=W_mod_eigvals)
        W_mod_lambda_min = float(np.min(W_mod_eigvals))
        W_mod_pinv = np.linalg.pinv(W_mod)
        W_mod_pinv_trace = float(np.trace(W_mod_pinv))
        W_mod_logdet = logdet_psd(W_mod)[0]
        W_mod_rank = np.linalg.matrix_rank(W_mod)

        W_trace_diff = np.abs(W_trace - float(np.trace(W_mod)))
        W_lambda_min_diff = np.abs(W_lambda_min - W_mod_lambda_min)
        W_pinv_trace_diff = np.abs(W_pinv_trace - W_mod_pinv_trace)
        W_logdet_diff = np.abs(W_logdet - W_mod_logdet)
        W_rank_diff = np.abs(W_rank - W_mod_rank)

        if 'ETEC' in edge_score_choice:
            ETEC_score = ETEC[i, j]

        match edge_score_choice:
            case 'sys_mat_spec_dist':
                score = A_spec_dist
            case 'Wc_spec_dist':
                score = A_spec_dist
            case 'Wc_trace_diff':
                score = W_trace_diff
            case 'Wc_lambda_min_diff':
                score = W_lambda_min_diff
            case 'Wc_pinv_trace_diff':
                score = W_pinv_trace_diff
            case 'Wc_logdet_diff':
                score = W_logdet_diff
            case 'Wc_rank_diff':
                score = W_rank_diff
            case 'ETEC' | 'ETEC_sys_mat':
                score = ETEC_score
            case 'random':
                if len(options['rand_edge_order']) > 0:
                    score = options['rand_edge_order'][(u, v)]
                else:
                    score = np.random.rand()
            case _:
                raise ValueError(f'Edge score choice {edge_score_choice} not supported.')
        
        if compute_spec_dist_of_other_sys_matrices:
            other_system_matrices_mod = [get_system_matrix_from_graph(G_mod, matrix_choice) for matrix_choice in other_system_matrix_choices]
            other_system_matrices_mod_eigvals = [real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(S)[0] for S in other_system_matrices_mod]
            other_system_matrices_spec_dist = [spectral_distance(S, S_mod, M1_eigvals=S_eigvals, M2_eigvals=S_mod_eigvals) for S, S_mod, S_eigvals, S_mod_eigvals in zip(other_system_matrices, other_system_matrices_mod, other_system_matrices_eigvals, other_system_matrices_mod_eigvals)]
        
        results_per_edge[(u, v)] = {}
        results_per_edge[(u, v)][edge_score_choice] = score
        results_per_edge[(u, v)]['sys_mat_spec_dist'] = A_spec_dist
        if edge_score_choice == 'ETEC':
            results_per_edge[(u, v)]['ETEC'] = ETEC_score
        # if calculate_W_stuff:
        results_per_edge[(u, v)]['Wc_trace_diff'] = W_trace_diff
        results_per_edge[(u, v)]['Wc_lambda_min_diff'] = W_lambda_min_diff
        results_per_edge[(u, v)]['Wc_pinv_trace_diff'] = W_pinv_trace_diff
        results_per_edge[(u, v)]['Wc_logdet_diff'] = W_logdet_diff
        results_per_edge[(u, v)]['Wc_rank_diff'] = W_rank_diff
        results_per_edge[(u, v)]['Wc_spec_dist'] = Wc_spec_dist
        results_per_edge[(u, v)]['density'] = nx.density(G_mod)
        
        if compute_spec_dist_of_other_sys_matrices:
            for i, matrix_choice in enumerate(other_system_matrix_choices):
                results_per_edge[(u, v)][f'{matrix_choice}_spec_dist'] = other_system_matrices_spec_dist[i]

    return results_per_edge, other_results


def spectral_distance(M1, M2, M1_eigvals=None, M2_eigvals=None, ord_norm=1):
    """
    || lambda(M1)) - lambda(M2)) ||_{ord_norm},
    where M1 and M2 are symmetric and eigenvalues are sorted.
    """
    if M1_eigvals is None:
        M1_eigvals = real_eigval_for_potentially_nonsymmetric_matrix(M1)
    if M2_eigvals is None:
        M2_eigvals = real_eigval_for_potentially_nonsymmetric_matrix(M2)
    cs = np.linalg.norm(np.sort(M1_eigvals) - np.sort(M2_eigvals), ord=ord_norm)
    return cs


def safe_det_psd(W, W_eigval=None, tol=1e-12):
    if W_eigval is not None:
        ev = W_eigval
    else:
        ev = real_eigval_for_potentially_nonsymmetric_matrix(W)
    full = np.min(ev) > tol
    return (float(np.prod(ev)) if full else 0.0), full

def logdet_psd(W, W_eigval=None, tol=1e-12):
    if W_eigval is not None:
        ev = W_eigval
    else:
        ev = real_eigval_for_potentially_nonsymmetric_matrix(W)
    kept = ev[ev > tol]
    if kept.size == 0:
        return -np.inf, 0, False
    return float(np.sum(np.log(kept))), kept.size, (kept.size == ev.size)


def plot_scatter(x, y, *, title=None, xlabel=None, ylabel=None, figfile=None):
    plt.scatter(x, y, s=24, alpha=0.7)
    plt.xlabel(xlabel or "x")
    plt.ylabel(ylabel or "y")
    plt.title(title or "")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if figfile is not None:
        plt.savefig(figfile, dpi=300, bbox_inches='tight')


def zero_forcing_set_greedy(G):
    """
    Given an adjacency matrix (NumPy 2D array), constructs a NetworkX graph,
    computes a zero forcing set, and returns a binary vector B.
    """
    n = len(G)

    # Define zero forcing process
    def apply_zero_forcing(initial_set):
        """Apply zero forcing rule until no more vertices can be forced."""
        blue = set(initial_set)
        changed = True
        while changed:
            changed = False
            for u in list(blue):
                # find white neighbors
                white_neighbors = [v for v in G.neighbors(u) if v not in blue]
                if len(white_neighbors) == 1:
                    w = white_neighbors[0]
                    if w not in blue:
                        blue.add(w)
                        changed = True
        return blue

    # Fallback: greedy approach if brute force fails
    # (not guaranteed minimal, but gives a valid ZFS)
    S = set()
    blue = set()
    while len(blue) < n:
        best_node, best_gain = None, -1
        for v in range(n):
            if v in S: continue
            cand = S | {v}
            final = apply_zero_forcing(cand)
            gain = len(final - blue)
            if gain > best_gain:
                best_gain, best_node = gain, v
        S.add(best_node)
        blue = apply_zero_forcing(S)

    B_vec = np.zeros((n, 1), dtype=int)
    for idx in S:
        B_vec[idx] = 1
    
    # Build B as a matrix with one column per selected ZFS node
    k = len(S)
    B = np.zeros((n, k), dtype=int)
    for col_idx, node in enumerate(sorted(S)):
        B[node, col_idx] = 1

    # final_blue = apply_zero_forcing(S)
    # if len(final_blue) != n:
    #     raise RuntimeError("Greedy zero forcing set construction failed to color all nodes.")
    # else:
    #     L = get_system_matrix_from_graph(G, matrix_choice="neg_laplacian")
    #     rank, n, is_controllable = compute_controllability_rank(L, B)
    #     if not is_controllable:
    #         raise RuntimeError("Computed zero forcing set does not yield controllable system with neg_laplacian as system matrix.")

    return B


def compute_matrix_of_pairwise_spectral_distances(graphs, matrix_type='adjacency', plot_type='matrix', figfile=None):
    """
    Given a list of graphs, compute a matrix of pairwise spectral distances.
    Returns a symmetric distance matrix.
    """
    N = len(graphs)
    dist_matrix = np.zeros((N, N))

    # Precompute eigenvalues for all graphs
    eigvals_list = []
    graph_matrices = []
    for G in graphs:
        A = get_system_matrix_from_graph(G, matrix_choice=matrix_type)
        graph_matrices.append(A)
        eigvals, _ = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(A)
        eigvals_list.append(eigvals)

    density_diff_list = []
    spec_dist_list = []
    # Compute pairwise spectral distances
    for i in range(N):
        for j in range(i, N):
            if i == j:
                dist = 0.0
            else:
                dist = spectral_distance(graph_matrices[i], graph_matrices[j], M1_eigvals=eigvals_list[i], M2_eigvals=eigvals_list[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric
            density_diff_list.append(abs(nx.density(graphs[i]) - nx.density(graphs[j])))
            spec_dist_list.append(dist)
    
    if plot_type != 'none':
        plt.figure(figsize = (5, 4))
        if plot_type == 'matrix':
            plt.imshow(dist_matrix)
            plt.colorbar(label=f'Spectral Distance ({matrix_type})')
            plt.title(f'Spectral Distance Matrix ({matrix_type})')
            plt.xlabel('Graph Index')
            plt.ylabel('Graph Index')
            plt.tight_layout()
            if figfile is not None:
                plt.savefig(figfile, dpi=300, bbox_inches='tight')
            plt.show()
        elif plot_type == 'scatter':
            plot_scatter(density_diff_list, spec_dist_list,
                        #  title=f'Pairwise Spectral Distance vs Density Difference ({matrix_type})', 
                         xlabel='Absolute Density Difference', 
                         ylabel=f'{matrix_type.capitalize()} Spectral Distance',
                         figfile=figfile)
            plt.show()

    return dist_matrix, density_diff_list, spec_dist_list


def plot_average_spectral_distance_for_same_param_val(graphs, matrix_type, param_name, param_values, n_graphs_per_param_val,
        plot_violin_and_medians=False, figfile=None, xlabel_step=4):
    
    param_wise_graph_lists = []
    graphs_idx = 0
    for param_val in param_values:
        graph_list = []
        while len(graph_list) < n_graphs_per_param_val:
            graph_list.append(graphs[graphs_idx])
            graphs_idx += 1
        param_wise_graph_lists.append(graph_list)
    
    spec_dist_means = []
    spec_dist_distributions = []
    for graph_list in param_wise_graph_lists:
        dist_matrix, _, _ = compute_matrix_of_pairwise_spectral_distances(graph_list, matrix_type=matrix_type, plot_type='none')
        spec_dist_means.append(np.mean(dist_matrix))
        spec_dist_distributions.append(dist_matrix.flatten())
    
    # plt.figure(figsize=(6, 3))
    # plt.plot(param_values, spec_dist_means, 'o-')
    # plt.xlabel(param_name)
    # plt.ylabel(f'Average {matrix_type.capitalize()} Spectral Distance')
    # # plt.title(f'Average Spectral Distance vs {param_name}')
    # plt.grid(True)
    # plt.show()
    
    positions = np.arange(len(param_values))


    if plot_violin_and_medians:
        plt.figure(figsize=(6,5))

        vp = plt.violinplot(
            spec_dist_distributions,
            positions=positions,
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=True
        )

        # Make violins clearly visible
        for body in vp['bodies']:
            body.set_facecolor("skyblue")
            body.set_edgecolor("black")
            body.set_alpha(0.7)
            body.set_linewidth(1)

        # summary statistics
        medians = [np.median(d) for d in spec_dist_distributions]

        # median markers
        plt.scatter(positions, medians, color="black", s=25, label="Median")

        # mean markers
        plt.scatter(
            positions,
            spec_dist_means,
            color="white",
            edgecolor="black",
            s=70,
            zorder=3,
            label="Mean"
        )
        plt.ylabel(f'{matrix_type.capitalize()} Spectral Distance')

        # mean trend line
        plt.plot(positions, spec_dist_means, color="black", alpha=0.6)
        
        plt.legend()
    else:
        plt.figure(figsize=(5,3.5))

        # just plot the means
        plt.plot(positions, spec_dist_means, 'o-')
        plt.ylabel(f'{matrix_type.capitalize()} Average Spectral Distance')
    
    plt.xticks(positions[::xlabel_step], np.round(param_values[::xlabel_step], 3))
    
    plt.xlabel(param_name)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if figfile is not None:
        plt.savefig(figfile, dpi=300, bbox_inches='tight')
    plt.show()

    return spec_dist_means, spec_dist_distributions


def compute_zfs_transf_Gramian_spectra_of_given_two_graphs(adj1, adj2, t=1, matrix='adjacency'):
    G1 = nx.from_numpy_array(adj1)
    G2 = nx.from_numpy_array(adj2)

    L1 = nx.laplacian_matrix(G1).toarray()
    L2 = nx.laplacian_matrix(G2).toarray()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    fig.suptitle("Pair of Cospectral Graphs", fontsize=14)
    nx.draw(G1, ax=ax1, with_labels=True, node_color='skyblue', node_size=800)
    nx.draw(G2, ax=ax2, with_labels=True, node_color='lightgreen', node_size=800)
    plt.tight_layout()
    plt.show()

    # 1. Verify they are NOT isomorphic
    isomorphic = nx.is_isomorphic(G1, G2)

    # 2. Compare their spectrum (eigenvalues)
    if matrix == 'adjacency':
        M1 = adj1
        M2 = adj2
    elif matrix == 'laplacian':
        M1 = -L1
        M2 = -L2
    else:
        raise ValueError("matrix must be 'adjacency' or 'laplacian'")
    
    # print(f"M1: {M1}")
    # print(f"M2: {M2}")
    
    M1_eigvals, Q1 = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(M1)
    M2_eigvals, Q2 = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(M2)

    print(f"Spec1: {np.round(M1_eigvals, 4)}")
    print(f"Spec2: {np.round(M2_eigvals, 4)}")

    print(f"Are they isomorphic? {isomorphic}")
    print(f"Are they cospectral? {np.allclose(M1_eigvals, M2_eigvals)}\n")

    T = Q1 @ Q2.T

    B1 = get_input(G1, {'input': 'zfs_transf'})
    B2 = get_input(G2, {'input': 'zfs_transf'}, B_old=B1, T=T, modified_matrix=True)

    # B1 = get_input(G1, {'input': 'all_ones'})
    # B2 = get_input(G2, {'input': 'all_ones'})

    # print(f"B1: {B1}")
    # print(f"B2: {B2}")
    # print(f"B1 transformed: {T.T @ B1}")

    W1 = finite_time_gramian(M1, B1, t=t)
    W2 = finite_time_gramian(M2, B2, t=t)

    W1_eigvals, _ = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(W1)
    W2_eigvals, _ = real_eigval_and_eigvec_for_potentially_nonsymmetric_matrix(W2)

    # print("W1 eigenvalues:", W1_eigvals)
    # print("W2 eigenvalues:", W2_eigvals)

    print(f"W1 eigenvalues: {np.round(W1_eigvals, 4)}")
    print(f"W2 eigenvalues: {np.round(W2_eigvals, 4)}")

    print(f"Are the Gramians cospectral? {np.allclose(W1_eigvals, W2_eigvals)}")

    return W1_eigvals, W2_eigvals, M1_eigvals, M2_eigvals, G1, G2