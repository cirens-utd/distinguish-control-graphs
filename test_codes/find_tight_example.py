import sys
import os
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(dirname(abspath(__file__))), 'modules'))

from graphs import get_system_matrix_from_graph, real_eigval_for_potentially_nonsymmetric_matrix

import math
import numpy as np
import networkx as nx


RED = '\033[91m'
RESET = '\033[0m'


def flip_edges_from_selected_nodes(
    G_original: nx.Graph,
    rng: np.random.Generator,
    nodelist=None,
):
    """
    Starting from G_original, do one random perturbation experiment:

    1. Pick random k with 1 <= k <= n
    2. Pick k distinct nodes
    3. For each selected node u, consider all n-1 other nodes in random order
    4. Flip edge (u, v) if it has not already been flipped
       - remove if present
       - add if absent

    Returns
    -------
    G_new : nx.Graph
        Perturbed graph
    k : int
        Number of selected nodes
    E : int
        Number of unique edges flipped
    selected_nodes : np.ndarray
        Selected node labels
    flipped_edges : list[tuple]
        List of flipped undirected edges (normalized to (min, max) order by nodelist index)
    """
    if nodelist is None:
        nodelist = sorted(G_original.nodes())

    n = len(nodelist)
    if n < 2:
        raise ValueError("Graph must have at least 2 nodes.")

    # Work on a fresh copy so every trial starts from the original graph
    G_new = G_original.copy()

    # Random k with 1 <= k <= n
    k = int(rng.integers(1, n + 1))

    # Randomly select k distinct nodes
    selected_nodes = rng.choice(nodelist, size=k, replace=False)

    # For consistent undirected-edge bookkeeping, map node -> index
    node_to_idx = {node: i for i, node in enumerate(nodelist)}

    flipped_edges = set()

    for u in selected_nodes:
        others = [v for v in nodelist if v != u]
        k_others = int(rng.integers(0, len(others) + 1))
        if k_others == 0:
            k -= 1
            continue
        selected_others = rng.choice(others, size=k_others, replace=False)

        for v in selected_others:
            iu = node_to_idx[u]
            iv = node_to_idx[v]
            edge_key = (u, v) if iu < iv else (v, u)

            # Do not flip an edge more than once
            if edge_key in flipped_edges:
                continue

            if G_new.has_edge(u, v):
                G_new.remove_edge(u, v)
            else:
                G_new.add_edge(u, v)

            flipped_edges.add(edge_key)

    E = len(flipped_edges)
    flipped_edges = sorted(flipped_edges, key=lambda e: (node_to_idx[e[0]], node_to_idx[e[1]]))

    # Prune selected_nodes while preserving coverage of all flipped edges:
    # for every flipped edge, at least one endpoint must remain selected.
    selected_nodes_set = set(selected_nodes.tolist())

    for u in list(selected_nodes_set):
        candidate_selected = selected_nodes_set - {u}

        can_remove_u = True
        for a, b in flipped_edges:
            if a not in candidate_selected and b not in candidate_selected:
                can_remove_u = False
                break

        if can_remove_u:
            selected_nodes_set.remove(u)

    selected_nodes = np.array(sorted(selected_nodes_set, key=lambda node: node_to_idx[node]))
    k = len(selected_nodes)

    return G_new, k, E, selected_nodes, flipped_edges


def find_violation(
    n: int,
    p: float,
    graph_seed: int = 0,
    experiment_seed: int = 1,
    max_trials: int = 10000,
    verbose: bool = False,
    k_greater_than: int = 0,
    matrix_choice: str = "adjacency",
):
    """
    Generate one ER graph G(n, p), then repeatedly perturb it starting from the
    original graph each time.

    In each trial:
      - compute eigenvalues of original and perturbed matrices, where
        matrix_choice is either "adjacency" or "laplacian"
      - compute abs difference vector
      - for adjacency, compare max(abs_diff) to sqrt(2k) * diff_frob_norm
      - for laplacian, compare max(abs_diff) to sqrt(n) * diff_frob_norm

    Stop on the first trial where the chosen abs_diff exceeds the chosen bound.

    Returns
    -------
    result : dict
        Dictionary with experiment results.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1.")
    if matrix_choice not in ("adjacency", "laplacian"):
        raise ValueError('matrix_choice must be either "adjacency" or "laplacian".')

    # Fixed original graph
    G_original = nx.erdos_renyi_graph(n=n, p=p, seed=graph_seed)

    # Fix a consistent node order for all adjacency matrices / eigenvalue vectors
    nodelist = sorted(G_original.nodes())

    # Original eigenvalues (computed once)
    A_orig = get_system_matrix_from_graph(G_original, matrix_choice=matrix_choice)
    eig_orig = real_eigval_for_potentially_nonsymmetric_matrix(A_orig)

    rng = np.random.default_rng(experiment_seed)

    closest_diff = float('inf')
    for trial in range(1, max_trials + 1):
        G_new, k, E, selected_nodes, flipped_edges = flip_edges_from_selected_nodes(
            G_original=G_original,
            rng=rng,
            nodelist=nodelist,
        )

        if k <= k_greater_than:
            continue

        A_new = get_system_matrix_from_graph(G_new, matrix_choice=matrix_choice)
        eig_new = real_eigval_for_potentially_nonsymmetric_matrix(A_new)
        
        abs_diff = np.sum(np.abs(eig_orig - eig_new))
        diff_frob_norm = np.linalg.norm(A_orig - A_new, ord='fro')

        if matrix_choice == "laplacian":
            bound = math.sqrt(n) * diff_frob_norm
            bound_label = "sqrt(n) * diff_frob_norm"
        elif matrix_choice == "adjacency":
            # bound = math.sqrt(2*k) * diff_frob_norm
            # bound_label = "sqrt(2k) * diff_frob_norm"
            bound = 2 * math.sqrt(k * E)
            bound_label = "2 * sqrt(k * E)"
        else:
            raise ValueError(f"Unknown matrix_choice: {matrix_choice}")

        if verbose or trial % 1000 == 0:
            print(
                f"trial={trial:5d} | k={k:3d} | E={E:5d} | "
                f"max|Δλ|={abs_diff:.6f} | bound={bound:.6f} | closest_diff={closest_diff:.6f} | diff_frob_norm={diff_frob_norm:.6f}"
            )

        if abs(abs_diff - bound) < closest_diff or abs_diff - bound > -1e-5*bound:
            closest_diff = abs_diff - bound
            best = {
                "trial": trial,
                "k": k,
                "E": E,
                "bound": bound,
                "bound_formula": bound_label,
                "matrix_choice": matrix_choice,
                "abs_diff": abs_diff,
                "selected_nodes": np.array(selected_nodes),
                "flipped_edges": flipped_edges,
                "eig_original": eig_orig,
                "eig_new": eig_new,
                "abs_diff": abs_diff,
                "A_original": nx.to_numpy_array(G_original, nodelist=nodelist, dtype=int),
                "A_new": nx.to_numpy_array(G_new, nodelist=nodelist, dtype=int),
                "G_original": G_original,
                "G_new": G_new,
            }
            if abs_diff - bound > -1e-5*bound:
                print(f"{RED}Found example where max(|Δλ|) >= {bound_label} at trial {trial}{RESET}")
                return best

    return {
        "found": False,
        **best
    }


if __name__ == "__main__":
    # Example parameters
    n = 4
    p = 0.33
    graph_seed = np.random.randint(0, 1000000)
    experiment_seed = np.random.randint(0, 1000000)
    max_trials = 50000
    matrix_choice = "adjacency"  # Use "laplacian" for the 2*n*sqrt(E) bound

    result = find_violation(
        n=n,
        p=p,
        graph_seed=graph_seed,
        experiment_seed=experiment_seed,
        max_trials=max_trials,
        verbose=False,
        k_greater_than=1,
        matrix_choice=matrix_choice,
    )

    print("\n--- RESULT ---")
    print(result)
    # if result["found"]:
    #     print("Found an example where max(|Δλ|) >= 2*sqrt(kE)")
    #     print(f"trial         = {result['trial']}")
    #     print(f"k             = {result['k']}")
    #     print(f"E             = {result['E']}")
    #     print(f"bound         = {result['bound']:.6f}")
    #     print(f"|Δλ|          = {result['abs_diff']:.6f}")
    #     print(f"selected_nodes= {result['selected_nodes']}")
    #     print("new adjacency matrix:")
    #     print(result["A_new"].astype(int))
    # else:
    #     print("No example found within the given number of trials.")
    #     print(f"closest_diff  = {result['closest_diff']:.6f}")
