import numpy as np
from scipy.linalg import expm
import control as ct



def finite_time_gramian(A, B, t=1.0):
    """
    Compute the finite-time controllability Gramian

        W_t = ∫_0^t e^{A τ} B B^T e^{A^T τ} dτ

    using the Van Loan block-matrix exponential identity.

    Parameters
    ----------
    A : (n, n) ndarray
        System / graph matrix (e.g., adjacency, Laplacian, etc.).
    B : (n, m) ndarray
        Input matrix.
    t : float
        Time horizon.

    Returns
    -------
    W_t : (n, n) ndarray
        Finite-time controllability Gramian.
    """
    n = A.shape[0]
    top = np.hstack([-A, B @ B.T])
    bot = np.hstack([np.zeros((n, n)), A.T])
    M = np.vstack([top, bot])
    T = expm(M * t)
    F22 = T[n:2*n, n:2*n]
    F12 = T[0:n,   n:2*n]
    return F22.T @ F12  # symmetric PSD


def compute_controllability_rank(A, B):
    """
    Computes the rank of the controllability matrix for a given system (A, B).

    Args:
        A (np.ndarray): The state matrix (n x n).
        B (np.ndarray): The input matrix (n x m).

    Returns:
        int: The rank of the controllability matrix.
        int: The number of states (n).
        bool: True if the system is controllable, False otherwise.
    """
    # Compute the controllability matrix
    # The result 'C' is the controllability matrix [B, AB, A^2B, ..., A^(n-1)B]
    controllability_matrix = ct.ctrb(A, B)
    
    # Compute the rank of the controllability matrix
    rank = np.linalg.matrix_rank(controllability_matrix)
    
    # Determine the number of states (n) from matrix A
    n = A.shape[0]
    
    # Check for controllability: the system is controllable if the rank equals n
    is_controllable = rank == n
    
    return rank, n, is_controllable