"""
Baseline transition matrix implementation (Old Approach).

Implements the method from reference [4] in the manuscript:
"Explainable Deep Learning: A Visual Analytics Approach with Transition Matrices"

The baseline approach finds T by minimizing fidelity only:
    min_T ||B - A T^T||_F^2
    
Solution: T = (A^+ B)^T = B^T (A^T)^+ = B^T A (A^T A)^+ 
        or equivalently: T^T = pinv(A) @ B
"""

import numpy as np
from numpy.linalg import pinv, svd
from typing import Tuple


def compute_baseline_T(A: np.ndarray, B: np.ndarray, 
                       tau: float = 1e-10) -> np.ndarray:
    """
    Compute baseline transition matrix T_old using pseudoinverse.
    
    Solves the fidelity-only objective:
        min_T ||B - A T^T||_F^2
    
    The solution is T^T = A^+ @ B, so T = B^T @ (A^+)^T
    
    Parameters
    ----------
    A : np.ndarray
        FM feature matrix of shape (m, k)
    B : np.ndarray
        MM feature matrix of shape (m, ℓ)
    tau : float, optional
        Regularization threshold for SVD. Default 1e-10.
    
    Returns
    -------
    np.ndarray
        Baseline transition matrix T_old of shape (ℓ, k)
    
    Notes
    -----
    Dimension check (per manuscript requirement):
    - A: (m, k) where m = samples, k = FM features
    - B: (m, ℓ) where ℓ = MM features  
    - T: (ℓ, k) maps FM → MM via b ≈ T @ a
    
    The equation B ≈ A T^T can be rewritten as:
    - For all samples: B^T ≈ T @ A^T
    - Per sample: b_j ≈ T @ a_j
    """
    m, k = A.shape
    m_b, l = B.shape
    
    assert m == m_b, f"Sample count mismatch: {m} vs {m_b}"
    
    # Compute T^T = pinv(A) @ B
    # pinv(A) has shape (k, m)
    # pinv(A) @ B has shape (k, ℓ)
    T_transpose = pinv(A, rcond=tau) @ B
    
    # T = (T^T)^T has shape (ℓ, k)
    T = T_transpose.T
    
    # Dimension check
    assert T.shape == (l, k), f"T shape {T.shape} != ({l}, {k})"
    
    return T


def compute_baseline_T_svd(A: np.ndarray, B: np.ndarray,
                           tau: float = 1e-10) -> Tuple[np.ndarray, dict]:
    """
    Compute baseline transition matrix with SVD decomposition details.
    
    Same as compute_baseline_T but returns additional diagnostic info
    about the SVD decomposition for analysis.
    
    Parameters
    ----------
    A : np.ndarray
        FM feature matrix of shape (m, k)
    B : np.ndarray
        MM feature matrix of shape (m, ℓ)
    tau : float, optional
        Regularization threshold. Default 1e-10.
    
    Returns
    -------
    T : np.ndarray
        Transition matrix of shape (ℓ, k)
    info : dict
        SVD diagnostic information:
        - 'singular_values': array of singular values
        - 'condition_number': ratio of max/min singular value
        - 'effective_rank': number of singular values > tau
        - 'residual_norm': ||B - A T^T||_F
    """
    m, k = A.shape
    _, l = B.shape
    
    # SVD of A
    U, sigma, Vt = svd(A, full_matrices=False)
    
    # Compute pseudoinverse via SVD
    sigma_inv = np.zeros_like(sigma)
    mask = sigma > tau
    sigma_inv[mask] = 1.0 / sigma[mask]
    
    # pinv(A) = V @ Σ^+ @ U^T
    A_pinv = (Vt.T * sigma_inv) @ U.T
    
    # T^T = pinv(A) @ B
    T_transpose = A_pinv @ B
    T = T_transpose.T
    
    # Compute diagnostics
    B_pred = A @ T.T
    residual = np.linalg.norm(B - B_pred, 'fro')
    
    info = {
        'singular_values': sigma,
        'condition_number': sigma[0] / sigma[mask][-1] if mask.any() else np.inf,
        'effective_rank': int(mask.sum()),
        'residual_norm': residual,
    }
    
    return T, info


def predict_mm_features(A: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Predict MM features from FM features using transition matrix.
    
    Computes B* = A @ T^T
    
    Parameters
    ----------
    A : np.ndarray
        FM feature matrix of shape (m, k) or single sample (k,)
    T : np.ndarray
        Transition matrix of shape (ℓ, k)
    
    Returns
    -------
    np.ndarray
        Predicted MM features of shape (m, ℓ) or (ℓ,)
    """
    if A.ndim == 1:
        # Single sample
        return T @ A
    else:
        # Batch of samples: B* = A @ T^T
        return A @ T.T
