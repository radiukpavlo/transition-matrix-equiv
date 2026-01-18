"""
Equivariant transition matrix implementation (New Approach).

Implements Algorithm 1 from the manuscript: "Computation of the Structurally-
Consistent Transition Matrix". This finds T that balances fidelity and
equivariance via the combined objective:

    L(T) = ||B^T - T A^T||_F^2 + λ Σ_i ||T J_i^A - J_i^B T||_F^2
"""

import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import LinearOperator, lsqr
from typing import List, Tuple, Optional, Union

from .core import vec, devec, svd_solve, check_dimensions


def build_fidelity_system(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the fidelity part of the system (Step 1 of Algorithm 1).
    
    Forms M_fid and Y_fid from the vectorized equation:
        (A ⊗ I_ℓ) vec(T) = vec(B^T)
    
    Parameters
    ----------
    A : np.ndarray
        FM features of shape (m, k)
    B : np.ndarray
        MM features of shape (m, ℓ)
    
    Returns
    -------
    M_fid : np.ndarray
        Fidelity matrix of shape (m·ℓ, k·ℓ)
    Y_fid : np.ndarray
        Target vector of shape (m·ℓ,)
    
    Notes
    -----
    Dimension check:
    - M_fid = A ⊗ I_ℓ: (m, k) ⊗ (ℓ, ℓ) = (m·ℓ, k·ℓ)
    - Y_fid = vec(B^T): (ℓ, m) → (m·ℓ,)
    """
    m, k = A.shape
    _, l = B.shape
    
    # Form M_fid = A ⊗ I_ℓ explicitly (feasible for small problems)
    I_l = np.eye(l)
    M_fid = np.kron(A, I_l)
    
    # Form Y_fid = vec(B^T)
    Y_fid = vec(B.T)
    
    # Dimension checks
    assert M_fid.shape == (m * l, k * l), f"M_fid shape {M_fid.shape}"
    assert Y_fid.shape == (m * l,), f"Y_fid shape {Y_fid.shape}"
    
    return M_fid, Y_fid


def build_symmetry_system(J_A: np.ndarray, J_B: np.ndarray,
                          k: int, l: int) -> np.ndarray:
    """
    Build the symmetry constraint matrix for one generator (Step 2).
    
    Forms K_i from the vectorized intertwining equation:
        T J^A - J^B T = 0
    becomes:
        ((J^A)^T ⊗ I_ℓ - I_k ⊗ J^B) vec(T) = 0
    
    Parameters
    ----------
    J_A : np.ndarray
        FM generator of shape (k, k)
    J_B : np.ndarray  
        MM generator of shape (ℓ, ℓ)
    k : int
        FM feature dimension
    l : int
        MM feature dimension
    
    Returns
    -------
    K : np.ndarray
        Constraint matrix of shape (k·ℓ, k·ℓ)
    
    Notes
    -----
    The intertwining condition T J^A = J^B T can be vectorized:
        vec(T J^A) = ((J^A)^T ⊗ I_ℓ) vec(T)
        vec(J^B T) = (I_k ⊗ J^B) vec(T)
    
    So T J^A - J^B T = 0 becomes K vec(T) = 0 where:
        K = (J^A)^T ⊗ I_ℓ - I_k ⊗ J^B
    """
    I_l = np.eye(l)
    I_k = np.eye(k)
    
    # K = (J^A)^T ⊗ I_ℓ - I_k ⊗ J^B
    K = np.kron(J_A.T, I_l) - np.kron(I_k, J_B)
    
    # Dimension check
    assert K.shape == (k * l, k * l), f"K shape {K.shape}"
    
    return K


def compute_equivariant_T(A: np.ndarray, B: np.ndarray,
                          J_A: Union[np.ndarray, List[np.ndarray]],
                          J_B: Union[np.ndarray, List[np.ndarray]],
                          lambda_: float = 0.5,
                          tau: float = 1e-10) -> np.ndarray:
    """
    Compute equivariant transition matrix T_new (Algorithm 1).
    
    Minimizes the combined objective:
        L(T) = ||B^T - T A^T||_F^2 + λ Σ_i ||T J_i^A - J_i^B T||_F^2
    
    Parameters
    ----------
    A : np.ndarray
        FM features of shape (m, k)
    B : np.ndarray
        MM features of shape (m, ℓ)
    J_A : np.ndarray or list
        FM generator(s), shape (k, k) or list of such
    J_B : np.ndarray or list
        MM generator(s), shape (ℓ, ℓ) or list of such
    lambda_ : float
        Weighting coefficient for symmetry constraints. Default 0.5.
    tau : float
        SVD regularization threshold. Default 1e-10.
    
    Returns
    -------
    np.ndarray
        Equivariant transition matrix T_new of shape (ℓ, k)
    
    Notes
    -----
    This is the explicit (direct) solver suitable for small problems.
    For MNIST-scale problems, use compute_equivariant_T_iterative().
    
    When λ=0, this reduces to the baseline (fidelity-only) solution.
    """
    dims = check_dimensions(A, B)
    m, k, l = dims['m'], dims['k'], dims['l']
    
    # Ensure J_A, J_B are lists
    if isinstance(J_A, np.ndarray) and J_A.ndim == 2:
        J_A = [J_A]
        J_B = [J_B]
    
    r = len(J_A)  # number of generators
    
    # Step 1: Build fidelity system
    M_fid, Y_fid = build_fidelity_system(A, B)
    
    # Step 2: Build symmetry constraints for each generator
    K_blocks = []
    for i in range(r):
        K_i = build_symmetry_system(J_A[i], J_B[i], k, l)
        K_blocks.append(lambda_ * K_i)
    
    # Step 3: Assemble combined system
    M_sym = np.vstack(K_blocks)
    Y_sym = np.zeros(r * k * l)
    
    M = np.vstack([M_fid, M_sym])
    Y = np.concatenate([Y_fid, Y_sym])
    
    # Dimension check for combined system
    n_eq_fid = m * l
    n_eq_sym = r * k * l
    n_unknowns = k * l
    assert M.shape == (n_eq_fid + n_eq_sym, n_unknowns), f"M shape {M.shape}"
    
    # Step 4: Solve via SVD
    u = svd_solve(M, Y, tau=tau)
    
    # Step 5: Reshape to transition matrix
    T = devec(u, (l, k))
    
    return T


def compute_equivariant_T_iterative(A: np.ndarray, B: np.ndarray,
                                    J_A: Union[np.ndarray, List[np.ndarray]],
                                    J_B: Union[np.ndarray, List[np.ndarray]],
                                    lambda_: float = 0.5,
                                    atol: float = 1e-8,
                                    btol: float = 1e-8,
                                    iter_lim: int = 500,
                                    show: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Memory-efficient equivariant solver using LSQR with implicit operators.
    
    For MNIST-scale problems where explicit Kronecker products are infeasible,
    this solver uses LinearOperator to define matrix-vector products implicitly.
    
    Parameters
    ----------
    A : np.ndarray
        FM features of shape (m, k)
    B : np.ndarray
        MM features of shape (m, ℓ)
    J_A : np.ndarray or list
        FM generator(s), shape (k, k)
    J_B : np.ndarray or list
        MM generator(s), shape (ℓ, ℓ)
    lambda_ : float
        Weighting coefficient. Default 0.5.
    atol, btol : float
        LSQR convergence tolerances. Default 1e-8.
    iter_lim : int
        Maximum iterations. Default 500.
    show : bool
        Print LSQR progress. Default False.
    
    Returns
    -------
    T : np.ndarray
        Transition matrix of shape (ℓ, k)
    info : dict
        Solver information (iterations, residual, etc.)
    
    Notes
    -----
    Memory usage: O(m·k + m·ℓ + k² + ℓ² + k·ℓ) instead of O((m·ℓ + k·ℓ)·k·ℓ)
    
    The implicit operator computes:
    - Fidelity: (A ⊗ I_ℓ) @ u via reshape/matmul
    - Symmetry: ((J^A)^T ⊗ I_ℓ - I_k ⊗ J^B) @ u via reshape/matmul
    """
    dims = check_dimensions(A, B)
    m, k, l = dims['m'], dims['k'], dims['l']
    
    # Ensure J_A, J_B are lists
    if isinstance(J_A, np.ndarray) and J_A.ndim == 2:
        J_A = [J_A]
        J_B = [J_B]
    
    r = len(J_A)
    
    # Precompute transposes
    A_T = A.T  # (k, m)
    J_A_T = [J.T for J in J_A]
    
    def matvec_fidelity(u):
        """Compute (A ⊗ I_ℓ) @ u without forming Kronecker."""
        T = u.reshape((l, k), order='F')
        result = T @ A_T  # (l, k) @ (k, m) = (l, m)
        return result.ravel(order='F')
    
    def matvec_symmetry_i(u, i):
        """Compute K_i @ u for generator i."""
        T = u.reshape((l, k), order='F')
        term1 = T @ J_A_T[i]      # (l, k) @ (k, k) = (l, k)
        term2 = J_B[i] @ T        # (l, l) @ (l, k) = (l, k)
        return lambda_ * (term1 - term2).ravel(order='F')
    
    def combined_matvec(u):
        """Combined matrix-vector product for full system."""
        result_fid = matvec_fidelity(u)
        result_sym = np.concatenate([matvec_symmetry_i(u, i) for i in range(r)])
        return np.concatenate([result_fid, result_sym])
    
    def matvec_fidelity_T(v):
        """Compute (A ⊗ I_ℓ)^T @ v = (A^T ⊗ I_ℓ) @ v."""
        V = v.reshape((l, m), order='F')
        result = V @ A  # (l, m) @ (m, k) = (l, k)
        return result.ravel(order='F')
    
    def matvec_symmetry_T_i(v, i):
        """Compute K_i^T @ v."""
        V = v.reshape((l, k), order='F')
        term1 = V @ J_A[i]        # (l, k) @ (k, k) = (l, k)
        term2 = J_B[i].T @ V      # (l, l) @ (l, k) = (l, k)
        return lambda_ * (term1 - term2).ravel(order='F')
    
    def combined_rmatvec(v):
        """Combined transpose matrix-vector product."""
        v_fid = v[:m * l]
        v_sym = v[m * l:]
        
        result = matvec_fidelity_T(v_fid)
        for i in range(r):
            v_i = v_sym[i * k * l:(i + 1) * k * l]
            result += matvec_symmetry_T_i(v_i, i)
        return result
    
    # Define implicit linear operator
    n_rows = m * l + r * k * l
    n_cols = k * l
    
    M_op = LinearOperator(
        shape=(n_rows, n_cols),
        matvec=combined_matvec,
        rmatvec=combined_rmatvec,
        dtype=np.float64
    )
    
    # Form right-hand side Y
    Y_fid = vec(B.T)
    Y_sym = np.zeros(r * k * l)
    Y = np.concatenate([Y_fid, Y_sym])
    
    # Solve using LSQR
    result = lsqr(M_op, Y, atol=atol, btol=btol, iter_lim=iter_lim, show=show)
    
    u = result[0]
    T = devec(u, (l, k))
    
    info = {
        'iterations': result[2],
        'residual_norm': result[3],
        'atol_satisfied': result[4],
        'btol_satisfied': result[5],
        'condition_estimate': result[6],
    }
    
    return T, info


def compute_equivariant_T_auto(A: np.ndarray, B: np.ndarray,
                               J_A: Union[np.ndarray, List[np.ndarray]],
                               J_B: Union[np.ndarray, List[np.ndarray]],
                               lambda_: float = 0.5,
                               memory_threshold_mb: float = 500) -> np.ndarray:
    """
    Automatically choose solver based on problem size.
    
    Uses explicit SVD for small problems, iterative LSQR for large.
    
    Parameters
    ----------
    A, B, J_A, J_B, lambda_ : 
        Same as compute_equivariant_T
    memory_threshold_mb : float
        Switch to iterative solver if explicit matrix would exceed this. Default 500 MB.
    
    Returns
    -------
    np.ndarray
        Transition matrix T
    """
    m, k = A.shape
    _, l = B.shape
    r = 1 if isinstance(J_A, np.ndarray) and J_A.ndim == 2 else len(J_A)
    
    # Estimate memory for explicit M matrix
    n_rows = m * l + r * k * l
    n_cols = k * l
    memory_bytes = n_rows * n_cols * 8  # float64
    memory_mb = memory_bytes / (1024 ** 2)
    
    if memory_mb < memory_threshold_mb:
        return compute_equivariant_T(A, B, J_A, J_B, lambda_)
    else:
        T, _ = compute_equivariant_T_iterative(A, B, J_A, J_B, lambda_)
        return T
