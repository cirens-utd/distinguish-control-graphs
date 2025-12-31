import numpy as np
import networkx as nx
import itertools
from scipy.sparse.csgraph import shortest_path
from numpy.linalg import eigvalsh
from modules.control import finite_time_gramian, compute_controllability_rank
from tqdm import trange, tqdm
import matplotlib.pyplot as plt



# ---------- ER: connected sampler ----------
def connected_erdos_renyi(n, p, rng, max_attempts=200):
    """Sample a connected G(n,p)."""
    for _ in range(max_attempts):
        G = nx.fast_gnp_random_graph(n, p, seed=int(rng.integers(0, 2**32 - 1)))
        if nx.is_connected(G):
            return G
    raise RuntimeError(f"Could not sample a connected G(n,{p:.3f}) after {max_attempts} tries.")


def get_graph(graph_choice, rng=np.nan):
    match graph_choice['type']:
        case 'connected_ER':
            G = connected_erdos_renyi(n=graph_choice['n'], p=graph_choice['p'], rng=rng)
        case _:
            raise ValueError(f"Graph choice {graph_choice} unsupported.")
    return G


# ---------- get system matrix related to graph ----------
def get_system_matrix_from_graph(G, matrix_choice="adjacency"):
    A = nx.to_numpy_array(G, dtype=np.float64)
    D = np.diag(A.sum(axis=1))
    I = np.eye(A.shape[0])
    match matrix_choice:
        case "adjacency" | "A":
            S = A
        case "laplacian" | "L":
            S = D - A
        case "signless_laplacian" | "Q":
            S = D + A
        case "normalized_laplacian" | "L_normal":
            deg = A.sum(axis=1)
            D_recip = np.diag(1.0 / np.sqrt(deg))
            S = I - D_recip @ A @ D_recip
        case "distance_normalized_laplacian" | "L_dist_normal":
            dist = shortest_path(A, directed=nx.is_directed(G))
            t = dist.sum(axis=1)
            T_inv_sqrt = np.diag(1.0 / np.sqrt(t))
            S = I - T_inv_sqrt @ dist @ T_inv_sqrt
        case _:
            raise ValueError(f"Matrix choice {matrix_choice} unsupported.")
    return S


def get_input(G, options, B_old=None, modified_matrix=False):
    n = len(G)
    match options['input']:
        case 'all_ones':
            B = np.ones((n, 1))
        case 'identity':
            B = np.eye(n)
        case 'zfs':
            if modified_matrix and (B_old is not None):
                B = B_old
            else:
                B = zero_forcing_set_greedy(G)
        case 'zfs_new':
            B = zero_forcing_set_greedy(G)
        case _:
            raise ValueError(f'Unsupported input {options['input']}.')
    
    return B


def rank_edges_based_on_toggling_single_edge(options, rng):
    t = options['t_horizon']
    edge_score_choice = options['edge_score_choice']
    G = get_graph(graph_choice=options['graph_choice'], rng=rng)
    A = get_system_matrix_from_graph(G, options['graph_matrix_choice'])
    B = get_input(G, options)
    W = finite_time_gramian(A, B, t=t)

    if options['input'] == 'zfs':
        laplacian = get_system_matrix_from_graph(G, matrix_choice="laplacian")
        W_test = finite_time_gramian(laplacian, B, t=t)
        if np.linalg.matrix_rank(W_test) < A.shape[0]:
            raise RuntimeError("System is not controllable with zero forcing set as input.")

    A_eigvals = eigvalsh(A)
    W_eigvals = eigvalsh(W)
    other_results = {}
    other_results['A_lambda_min'] = float(np.min(A_eigvals))
    other_results['A_lambda_max'] = float(np.max(A_eigvals))

    nodes = list(G.nodes())
    n = len(nodes)
    results_per_edge = {}

    for i in tqdm(range(n)):
        for j in range(n):
            u, v = nodes[i], nodes[j]
            if i == j:
                if G.has_edge(u, v):
                    raise ValueError(f"Non-simple graphs not supported.")
                else:
                    continue
            
            results_per_edge[(u,v)] = {}

            # Modify the graph: toggle edge presence
            G_mod = G.copy()
            if G_mod.has_edge(u, v):
                G_mod.remove_edge(u, v)
            else:
                G_mod.add_edge(u, v)

            A_mod = get_system_matrix_from_graph(G_mod, options['graph_matrix_choice'])

            match edge_score_choice:
                case 'sys_mat_spec_dist':
                    score = spectral_distance(A, A_mod, M1_eigvals=A_eigvals)
                # case 'trace_W':
                #     tr_mod = float(np.trace(W_mod))
                #     scores[(u, v)] = tr_full - tr_mod
                case _:
                    raise ValueError(f'Edge score choice {edge_score_choice} not supported.')
            
            results_per_edge[(u, v)][edge_score_choice] = score

            B_mod = get_input(G, options, B_old=B, modified_matrix=True)
            W_mod = finite_time_gramian(A_mod, B_mod, t=t)
            results_per_edge[(u, v)]['Wc_spec_dist'] = spectral_distance(W, W_mod, M1_eigvals=W_eigvals)

    return results_per_edge, other_results


def spectral_distance(M1, M2, M1_eigvals=None, M2_eigvals=None, ord_norm=1):
    """
    || lambda(M1)) - lambda(M2)) ||_{ord_norm},
    where M1 and M2 are symmetric and eigenvalues are sorted.
    """
    if M1_eigvals is None:
        M1_eigvals = eigvalsh(M1)
    if M2_eigvals is None:
        M2_eigvals = eigvalsh(M2)
    cs = np.linalg.norm(np.sort(M1_eigvals) - np.sort(M2_eigvals), ord=ord_norm)
    return cs


def safe_det_psd(W, tol=1e-12):
    ev = eigvalsh(W)
    full = np.min(ev) > tol
    return (float(np.prod(ev)) if full else 0.0), full

def logdet_psd(W, tol=1e-12):
    ev = eigvalsh(W)
    kept = ev[ev > tol]
    if kept.size == 0:
        return -np.inf, 0, False
    return float(np.sum(np.log(kept))), kept.size, (kept.size == ev.size)


def plot_scatter(x, y, *, title=None, xlabel=None, ylabel=None):
    plt.scatter(x, y, s=24, alpha=0.7)
    plt.xlabel(xlabel or "x")
    plt.ylabel(ylabel or "y")
    plt.title(title or "")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    #     L = get_system_matrix_from_graph(G, matrix_choice="laplacian")
    #     rank, n, is_controllable = compute_controllability_rank(L, B)
    #     if not is_controllable:
    #         raise RuntimeError("Computed zero forcing set does not yield controllable system with laplacian as system matrix.")

    return B



    