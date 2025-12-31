import numpy as np
from scipy.linalg import expm



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


