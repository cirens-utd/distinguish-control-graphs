import numpy as np
from scipy.linalg import expm, null_space
import control as ct
from scipy.integrate import solve_ivp
import cvxpy as cp
from numpy.linalg import eigvals



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
    T = expm((M * t).astype(np.float64))
    F22 = T[n:2*n, n:2*n]
    F12 = T[0:n,   n:2*n]
    return F22.T @ F12  # symmetric PSD


def finite_time_discrete_gramian(A, B, t_as_fraction_of_n=2.0):
    """
    Compute the finite-time discrete-time controllability Gramian

        W_t = sum_{k=0}^{t_total-1} A^k B B^T {A^k}^T

    Parameters
    ----------
    A : (n, n) ndarray
        System / graph matrix (e.g., adjacency, Laplacian, etc.).
    B : (n, m) ndarray
        Input matrix.
    t_as_fraction_of_n : float
        Fraction of n to use as time horizon.

    Returns
    -------
    W_t : (n, n) ndarray
        Finite-time discrete-time controllability Gramian.
    """
    n = A.shape[0]
    Wc_discrete = np.zeros((n, n))
    t_total = int(round(t_as_fraction_of_n * n))
    Ak = np.eye(n)
    for k in range(t_total):
        Wc_discrete += Ak @ B @ B.T @ Ak.T
        Ak = Ak @ A
    return Wc_discrete


def finite_horizon_gramian_through_integration(A, B, t=1.0):
    """
    Computes the finite horizon controllability Gramian Wc(T) for a system (A, B).

    Args:
        A (np.ndarray): The system state matrix (n x n).
        B (np.ndarray): The system input matrix (n x m).
        T (float): The final time horizon.

    Returns:
        np.ndarray: The controllability Gramian Wc(T) (n x n).
    """
    n = A.shape[0]
    
    def gramian_ode(t, W_flat):
        W = W_flat.reshape((n, n))
        # The differential equation: dW/dt = A*W + W*A.T + B*B.T
        dWdt = A @ W + W @ A.T + B @ B.T
        return dWdt.flatten()

    # Initial condition: Wc(0) = 0
    W0_flat = np.zeros(n * n)
    
    # Time span for integration
    t_span = [0, t]
    
    # Solve the ODE
    solution = solve_ivp(gramian_ode, t_span, W0_flat, dense_output=True)
    
    if not solution.success:
        print(f"Integration failed: {solution.message}")
    
    # Extract the Gramian at time T
    Wc_T = solution.y[:, -1].reshape((n, n))
    return Wc_T


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


def pseudo_gramian_for_semistable_A_inf_horizon(A, B, t=np.inf, tol=1e-4, opt_tol=1e-5):

    if any(eigvals(A) > tol):
        raise ValueError("A is not semistable (NSD).")
    
    V0 = null_space(A)
    J = V0 @ np.linalg.pinv(V0.T @ V0) @ V0.T   # projection onto nullspace
    # alternative method for J: J = expm(A*t) with t large enough (based on eigenvalues of A)
    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)

    I = np.eye(n)
    Qc = (I-J) @ B @ B.T @ (I-J).T

    constraints = [
        A @ P + P @ A.T + Qc == 0,   # Lyapunov equality
        # A @ P + P @ A.T + Qc <= tol,   # Lyapunov equality
        # A @ P + P @ A.T + Qc >= -tol,   # Lyapunov equality
        J @ P @ J.T == 0,            # nullspace constraint
        # P >> tol                       # optional: P is PSD
        P >> 0                       # optional: P is PSD
    ]

    # print(cp.installed_solvers())
    prob = cp.Problem(cp.Minimize(0), constraints=constraints)
    prob.solve(solver=cp.MOSEK)#, FeasibilityTol=opt_tol, OptimalityTol=opt_tol)   # or solver=cp.CVXOPT etc.
    # prob.solve(solver=cp.SCS, eps=opt_tol)   # or solver=cp.CVXOPT etc.
        # , feastol=opt_tol, reltol=opt_tol, feastol_inacc=opt_tol, reltol_inacc=opt_tol, epstol=opt_tol)
    P_opt = P.value

    if np.linalg.norm(J @ P_opt @ J.T) > tol or np.linalg.norm(A @ P_opt + P_opt @ A.T + Qc) > tol or not np.all(np.linalg.eigvals(P_opt) > -tol):
        # print(f"P_opt = {P_opt}", end="\n\n")
        print(f"tol = {tol}")
        print(f"norm(J @ P_opt @ J.T) = {np.linalg.norm(J @ P_opt @ J.T, ord=1):.3g}")
        print(f"norm(A @ P_opt + P_opt @ A.T + Qc) = {np.linalg.norm(A @ P_opt + P_opt @ A.T + Qc, ord=1):.3g}")
        print(f"P_opt >> 0: {np.all(np.linalg.eigvals(P_opt) > -tol)}")
        print(f"prob.status = {prob.status}")
        raise ValueError("Problem with P_opt")

    return P_opt


def compute_energy_transfer_edge_centrality(A, T):
    """
    Compute c_ij from adjacency matrix A up to time T.
    
    Parameters
    ----------
    A : numpy.ndarray (n x n)
        Adjacency matrix
    T : int
        Time horizon
    
    Returns
    -------
    C_T : numpy.ndarray (n x n)
        Matrix C_T where C_T[i,j] = c_ij^{(T)}
    """
    n = A.shape[0]
    H_t = np.zeros((n, n))
    C_t = np.zeros((n, n))
    
    A_power = np.eye(n)  # A^0

    if T < 2:
        raise ValueError(f"T must be at least 2; recommended value is T = 2*n (= {2*n}) or T = n (= {n})")
    
    for t in range(1, T):
        # accumulate H^{(t)} = sum_{k=0}^{t-1} (A^k)^2 elementwise
        H_t += A_power**2
        
        # compute p^{(t)} and q^{(t)}
        p_t = H_t.sum(axis=0)  # column sums
        q_t = H_t.sum(axis=1)  # row sums
        
        # accumulate c_ij^{(t)}
        C_t += np.outer(q_t, p_t)
        
        # update A^k
        A_power = A_power @ A
    
    C_T = C_t
    
    return C_T
