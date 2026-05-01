import random
from typing import Callable, Dict, List, Optional, Tuple
import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def adjacency_matrix_builder(G: nx.Graph) -> np.ndarray:
    """Default matrix builder: adjacency matrix."""
    return nx.to_numpy_array(G, dtype=float)


def laplacian_matrix_builder(G: nx.Graph) -> np.ndarray:
    """Alternative matrix builder: Laplacian matrix."""
    return nx.laplacian_matrix(G).toarray().astype(float)


def compute_sorted_eigenvalues(
    matrix: np.ndarray,
    descending: bool = False,
    assume_symmetric: bool = True,
) -> np.ndarray:
    """
    Compute eigenvalues of a matrix and return them sorted.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix whose eigenvalues will be computed.
    descending : bool
        If True, sort eigenvalues in descending order.
        If False, sort in ascending order.
    assume_symmetric : bool
        If True, use eigvalsh (recommended for symmetric/Hermitian matrices such as
        adjacency of undirected graphs or Laplacian). Otherwise use eigvals.

    Returns
    -------
    np.ndarray
        Sorted eigenvalues.
    """
    if assume_symmetric:
        eigvals = np.linalg.eigvalsh(matrix)
    else:
        eigvals = np.linalg.eigvals(matrix)
        eigvals = np.real_if_close(eigvals)
        if np.iscomplexobj(eigvals):
            raise ValueError(
                "The chosen matrix has genuinely complex eigenvalues. "
                "This code expects a real spectrum for ranking/correlation."
            )

    eigvals = np.asarray(eigvals, dtype=float)
    eigvals.sort()

    if descending:
        eigvals = eigvals[::-1]

    return eigvals


def build_kth_largest_timeseries(
    eigenvalue_history: np.ndarray,
    storage_order_descending: bool = False,
) -> np.ndarray:
    """
    Convert stored eigenvalue history into time series of k-th largest eigenvalues.

    Parameters
    ----------
    eigenvalue_history : np.ndarray
        Shape: (num_iterations, n), where each row contains the sorted eigenvalues
        at one iteration.
    storage_order_descending : bool
        Whether eigenvalues were stored in descending order.

    Returns
    -------
    np.ndarray
        Shape: (n, num_iterations), where row k-1 is the time series of the k-th largest
        eigenvalue over time.
    """
    if storage_order_descending:
        # Stored as [largest, ..., smallest] already
        kth_largest = eigenvalue_history.T
    else:
        # Stored as [smallest, ..., largest], so reverse columns to get largest-first
        kth_largest = eigenvalue_history[:, ::-1].T

    return kth_largest


def default_correlation_functions() -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """
    Return default correlation functions. You can extend this dictionary later
    with additional coefficients if you want.
    """
    return {
        "spearman": lambda x, y: spearmanr(x, y).statistic,
        "kendall": lambda x, y: kendalltau(x, y).statistic,
        "pearson": lambda x, y: pearsonr(x, y).statistic,
    }


def simulate_random_graph_spectral_evolution(
    n: int,
    matrix_builder: Optional[Callable[[nx.Graph], np.ndarray]] = None,
    descending: bool = False,
    assume_symmetric: bool = True,
    seed: Optional[int] = None,
    include_initial_state: bool = False,
    correlation_functions: Optional[
        Dict[str, Callable[[np.ndarray, np.ndarray], float]]
    ] = None,
) -> Dict[str, object]:
    """
    Build an empty graph on n nodes, add edges one-by-one in random order until complete,
    and track sorted eigenvalues and density at each iteration.

    Parameters
    ----------
    n : int
        Number of nodes.
    matrix_builder : callable, optional
        Function that takes a graph G and returns a matrix (numpy array).
        Defaults to adjacency matrix.
    descending : bool
        If True, store eigenvalues in descending order.
        If False, store eigenvalues in ascending order.
    assume_symmetric : bool
        If True, use eigvalsh. Recommended for undirected adjacency/Laplacian matrices.
    seed : int, optional
        Random seed for reproducibility.
    include_initial_state : bool
        If True, also compute/store the initial empty-graph state before any edge is added.
    correlation_functions : dict, optional
        Dictionary mapping metric names to functions f(x, y) -> scalar.
        If None, defaults to Spearman/Kendall/Pearson.

    Returns
    -------
    dict
        A dictionary containing the graph evolution results.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if matrix_builder is None:
        matrix_builder = adjacency_matrix_builder

    if correlation_functions is None:
        correlation_functions = default_correlation_functions()

    rng = random.Random(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_possible_edges)

    eigenvalue_rows: List[np.ndarray] = []
    density_history: List[float] = []
    edge_count_history: List[int] = []

    def record_current_state() -> None:
        matrix = matrix_builder(G)
        eigvals = compute_sorted_eigenvalues(
            matrix,
            descending=descending,
            assume_symmetric=assume_symmetric,
        )
        eigenvalue_rows.append(eigvals)
        density_history.append(nx.density(G))
        edge_count_history.append(G.number_of_edges())

    if include_initial_state:
        record_current_state()

    for u, v in all_possible_edges:
        G.add_edge(u, v)
        record_current_state()

    eigenvalue_history = np.vstack(eigenvalue_rows)  # shape: (T, n)
    density_history = np.asarray(density_history, dtype=float)  # shape: (T,)
    edge_count_history = np.asarray(edge_count_history, dtype=int)  # shape: (T,)

    kth_largest_timeseries = build_kth_largest_timeseries(
        eigenvalue_history,
        storage_order_descending=descending,
    )  # shape: (n, T)

    # Correlation of each eigenvalue time series with density
    correlation_results: Dict[str, List[float]] = {}
    for metric_name, metric_fn in correlation_functions.items():
        values = []
        for k in range(n):
            coeff = metric_fn(kth_largest_timeseries[k], density_history)
            values.append(coeff)
        correlation_results[metric_name] = values

    return {
        "graph": G,
        "n": n,
        "num_iterations": eigenvalue_history.shape[0],
        "eigenvalue_history": eigenvalue_history,            # stored in chosen sort order
        "kth_largest_timeseries": kth_largest_timeseries,    # always largest-first
        "density_history": density_history,
        "edge_count_history": edge_count_history,
        "correlation_results": correlation_results,
        "descending_storage_order": descending,
        "include_initial_state": include_initial_state,
    }


def rank_eigenvalue_timeseries(
    correlation_results: Dict[str, List[float]],
    rank_by: str = "spearman",
    descending: bool = True,
) -> List[Tuple[int, Dict[str, float]]]:
    """
    Rank eigenvalue time series by a chosen correlation metric.

    Parameters
    ----------
    correlation_results : dict
        Output dictionary["correlation_results"] from the simulation.
    rank_by : str
        Which metric to use for ranking (e.g. 'spearman', 'kendall', 'pearson').
    descending : bool
        If True, rank from largest coefficient to smallest.
        If False, rank from smallest to largest.

    Returns
    -------
    list of tuples
        Each item is:
        (
            eigenvalue_timeseries_number,   # 1 means largest eigenvalue series
            {"spearman": ..., "kendall": ..., ...}
        )
    """
    if rank_by not in correlation_results:
        raise ValueError(
            f"rank_by='{rank_by}' is not available. Choices: {list(correlation_results)}"
        )

    n = len(correlation_results[rank_by])
    ranked = []

    for k in range(n):
        metrics_for_k = {name: correlation_results[name][k] for name in correlation_results}
        ranked.append((k + 1, metrics_for_k))  # k+1 => k-th largest eigenvalue series

    ranked.sort(key=lambda item: item[1][rank_by], reverse=descending)
    return ranked


def print_ranked_correlations(
    ranked_results: List[Tuple[int, Dict[str, float]]],
    rank_by: str = "spearman",
) -> None:
    """
    Print the ranked eigenvalue series and their coefficients.
    """
    print(f"\nRanking eigenvalue time series by '{rank_by}' coefficient:\n")
    for series_number, metrics in ranked_results:
        metric_str = ", ".join(f"{name}={value:.6f}" for name, value in metrics.items())
        print(f"Eigenvalue time series #{series_number}: {metric_str}")


def plot_eigenvalue_timeseries(
    kth_largest_timeseries: np.ndarray,
    edge_count_history: np.ndarray,
    density_history: np.ndarray,
    title: str = "Eigenvalue Time Series During Random Edge Addition",
    show_legend: bool = True,
    alpha: float = 0.8,
    linewidth: float = 1.5,
) -> None:
    """
    Plot all eigenvalue time series on the same axes.

    Parameters
    ----------
    kth_largest_timeseries : np.ndarray
        Shape: (n, T), where row k-1 is the k-th largest eigenvalue time series.
    edge_count_history : np.ndarray
        Shape: (T,), number of edges present at each iteration.
    density_history : np.ndarray
        Shape: (T,), graph density at each iteration.
    """
    n, T = kth_largest_timeseries.shape

    plt.figure(figsize=(12, 7))

    x = np.arange(T)
    for k in range(n):
        plt.plot(
            x,
            kth_largest_timeseries[k],
            label=f"{k + 1}-th largest eigenvalue",
            alpha=alpha,
            linewidth=linewidth,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if show_legend:
        # For large n the legend can get crowded; you can disable it if needed.
        plt.legend(loc="best", fontsize=9, ncol=2)

    # Optional secondary x-axis showing density
    def iter_to_density(i):
        i = np.asarray(i)
        i = np.clip(np.round(i).astype(int), 0, T - 1)
        return density_history[i]

    def density_to_iter(d):
        # Rough inverse map for display only
        d = np.asarray(d)
        return d * (T - 1)

    secax = plt.gca().secondary_xaxis("top", functions=(iter_to_density, density_to_iter))
    secax.set_xlabel("Approx. density")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # =========================
    # Example usage
    # =========================

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--matrix_type", type=str, default="adjacency")
    args = parser.parse_args()
    n = args.n
    matrix_type = args.matrix_type

    # Choose the matrix builder:
    if matrix_type == "adjacency":
        matrix_builder = adjacency_matrix_builder
    elif matrix_type == "laplacian":
        matrix_builder = laplacian_matrix_builder
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


    results = simulate_random_graph_spectral_evolution(
        n=n,
        matrix_builder=matrix_builder,
        descending=False,          # store eigenvalues ascending; set True for descending
        assume_symmetric=True,     # adjacency/laplacian for undirected graphs are symmetric
        seed=42,
        include_initial_state=False,  # set True if you want the empty graph included
    )

    kth_largest_timeseries = results["kth_largest_timeseries"]
    density_history = results["density_history"]
    edge_count_history = results["edge_count_history"]
    correlation_results = results["correlation_results"]

    ranked = rank_eigenvalue_timeseries(
        correlation_results=correlation_results,
        rank_by="spearman",   # change to "kendall" or "pearson" if desired
        descending=True,
    )

    print_ranked_correlations(
        ranked_results=ranked,
        rank_by="spearman",
    )

    plot_eigenvalue_timeseries(
        kth_largest_timeseries=kth_largest_timeseries,
        edge_count_history=edge_count_history,
        density_history=density_history,
        title=f"Spectral Evolution for Random Edge Addition (n={n})",
        show_legend=True,
    )