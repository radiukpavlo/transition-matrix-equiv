"""
Core mathematical operations for equivariant transition matrices.

This module provides fundamental operations for matrix vectorization,
Kronecker products, and SVD-based solving.
"""

import numpy as np
from numpy.linalg import svd, pinv
from typing import Tuple, Optional


def vec(T: np.ndarray) -> np.ndarray:
    """
    Vectorize a matrix by stacking columns.
    
    For matrix T ∈ ℝ^(ℓ × k), returns vec(T) ∈ ℝ^(k·ℓ).
    Uses column-major (Fortran) order as per standard mathematical convention.
    
    Parameters
    ----------
    T : np.ndarray
        Matrix of shape (ℓ, k)
    
    Returns
    -------
    np.ndarray
        Vectorized form of shape (k·ℓ,)
    
    Example
    -------
    >>> T = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2
    >>> vec(T)
    array([1, 3, 5, 2, 4, 6])  # column-major
    """
    return T.ravel(order='F')


def devec(u: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    De-vectorize (reshape) a vector back to matrix form.
    
    Inverse of vec(): devec(vec(T), T.shape) == T
    
    Parameters
    ----------
    u : np.ndarray
        Vectorized matrix of shape (k·ℓ,)
    shape : tuple of int
        Target shape (ℓ, k)
    
    Returns
    -------
    np.ndarray
        Matrix of shape (ℓ, k)
    """
    l, k = shape
    return u.reshape((l, k), order='F')


def kron_mvp(A: np.ndarray, B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute Kronecker matrix-vector product (A ⊗ B) @ x without forming A ⊗ B.
    
    Uses the identity: (A ⊗ B) vec(X) = vec(B X A^T)
    where x = vec(X) and X has appropriate shape.
    
    For A ∈ ℝ^(m × n) and B ∈ ℝ^(p × q), x should have length n·q,
    and the result has length m·p.
    
    Parameters
    ----------
    A : np.ndarray
        Left matrix of shape (m, n)
    B : np.ndarray
        Right matrix of shape (p, q)
    x : np.ndarray
        Vector of shape (n·q,)
    
    Returns
    -------
    np.ndarray
        Result of (A ⊗ B) @ x, shape (m·p,)
    
    Memory Complexity
    -----------------
    O(m·n + p·q + n·q) instead of O(m·n·p·q) for explicit Kronecker
    """
    m, n = A.shape
    p, q = B.shape
    
    # Reshape x to matrix X of shape (q, n): x = vec(X)
    X = x.reshape((q, n), order='F')
    
    # Compute B @ X @ A^T
    result = B @ X @ A.T
    
    # Return vec(result)
    return result.ravel(order='F')


def svd_solve(M: np.ndarray, Y: np.ndarray, tau: float = 1e-10) -> np.ndarray:
    """
    Solve the least-squares problem M @ u = Y using SVD with regularization.
    
    Computes u = M^+ @ Y where M^+ is the Moore-Penrose pseudoinverse,
    with small singular values (< tau) set to zero for numerical stability.
    
    Parameters
    ----------
    M : np.ndarray
        System matrix of shape (n_equations, n_unknowns)
    Y : np.ndarray
        Right-hand side vector of shape (n_equations,)
    tau : float, optional
        Regularization threshold for singular values. Default 1e-10.
    
    Returns
    -------
    np.ndarray
        Solution vector u of shape (n_unknowns,)
    
    Notes
    -----
    This is the key solver for Algorithm 1 (Step 4) in the manuscript.
    Using SVD is critically important because:
    - Handles rank-deficient matrices M
    - Truncates noise via singular value thresholding
    - Provides unique optimal solution in least-squares sense
    """
    U, sigma, Vt = svd(M, full_matrices=False)
    
    # Compute pseudoinverse of singular values with threshold
    sigma_inv = np.zeros_like(sigma)
    mask = sigma > tau
    sigma_inv[mask] = 1.0 / sigma[mask]
    
    # u = V @ Σ^+ @ U^T @ Y
    u = Vt.T @ (sigma_inv * (U.T @ Y))
    
    return u


def check_dimensions(A: np.ndarray, B: np.ndarray, 
                     J_A: Optional[np.ndarray] = None,
                     J_B: Optional[np.ndarray] = None) -> dict:
    """
    Validate and report dimensions as required by the manuscript.
    
    Parameters
    ----------
    A : np.ndarray
        FM feature matrix of shape (m, k)
    B : np.ndarray
        MM feature matrix of shape (m, ℓ)
    J_A : np.ndarray, optional
        FM generator of shape (k, k)
    J_B : np.ndarray, optional
        MM generator of shape (ℓ, ℓ)
    
    Returns
    -------
    dict
        Dictionary with dimension information
    
    Raises
    ------
    ValueError
        If dimensions are inconsistent
    """
    m_A, k = A.shape
    m_B, l = B.shape
    
    if m_A != m_B:
        raise ValueError(f"Sample count mismatch: A has {m_A}, B has {m_B}")
    
    m = m_A
    
    dims = {
        'm': m,      # number of samples
        'k': k,      # FM feature dimension
        'l': l,      # MM feature dimension
        'kl': k * l, # size of vec(T)
    }
    
    if J_A is not None:
        if J_A.shape != (k, k):
            raise ValueError(f"J_A shape {J_A.shape} != ({k}, {k})")
    
    if J_B is not None:
        if J_B.shape != (l, l):
            raise ValueError(f"J_B shape {J_B.shape} != ({l}, {l})")
    
    return dims
