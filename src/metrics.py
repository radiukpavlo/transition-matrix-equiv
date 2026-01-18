"""
Evaluation metrics for transition matrix quality.

Implements all metrics required by the manuscript:
- Fidelity error (MSE)
- Symmetry defect
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
"""

import numpy as np
from typing import Tuple, Optional


def fidelity_mse(B: np.ndarray, B_pred: np.ndarray) -> float:
    """
    Compute fidelity error as Mean Squared Error.
    
    MSE_fid = (1 / (m·ℓ)) ||B - B*||_F^2
    
    Parameters
    ----------
    B : np.ndarray
        True MM features of shape (m, ℓ)
    B_pred : np.ndarray
        Predicted MM features of shape (m, ℓ)
    
    Returns
    -------
    float
        Mean squared error
    """
    diff = B - B_pred
    frobenius_sq = np.sum(diff ** 2)
    mse = frobenius_sq / B.size
    return mse


def fidelity_frobenius(B: np.ndarray, B_pred: np.ndarray) -> float:
    """
    Compute fidelity error as squared Frobenius norm.
    
    ||B - B*||_F^2
    
    Parameters
    ----------
    B : np.ndarray
        True MM features of shape (m, ℓ)
    B_pred : np.ndarray
        Predicted MM features of shape (m, ℓ)
    
    Returns
    -------
    float
        Squared Frobenius norm of residual
    """
    return np.sum((B - B_pred) ** 2)


def symmetry_defect(T: np.ndarray, J_A: np.ndarray, J_B: np.ndarray) -> float:
    """
    Compute symmetry defect (intertwining violation).
    
    Sym_err = ||T J^A - J^B T||_F^2
    
    Measures how well T respects the Lie algebra generators.
    A value of 0 means T is a perfect intertwiner.
    
    Parameters
    ----------
    T : np.ndarray
        Transition matrix of shape (ℓ, k)
    J_A : np.ndarray
        FM generator of shape (k, k)
    J_B : np.ndarray
        MM generator of shape (ℓ, ℓ)
    
    Returns
    -------
    float
        Symmetry defect (squared Frobenius norm)
    
    Notes
    -----
    Dimension check:
    - T J^A: (ℓ, k) @ (k, k) = (ℓ, k)
    - J^B T: (ℓ, ℓ) @ (ℓ, k) = (ℓ, k)
    - Difference: (ℓ, k)
    """
    left = T @ J_A    # (l, k) @ (k, k) = (l, k)
    right = J_B @ T   # (l, l) @ (l, k) = (l, k)
    defect = left - right
    return np.sum(defect ** 2)


def symmetry_defect_normalized(T: np.ndarray, J_A: np.ndarray, 
                               J_B: np.ndarray) -> float:
    """
    Compute normalized symmetry defect.
    
    Normalized by the norms of T, J_A, J_B for scale-invariance.
    """
    raw_defect = symmetry_defect(T, J_A, J_B)
    norm_factor = (np.linalg.norm(T, 'fro') * 
                   np.linalg.norm(J_A, 'fro') * 
                   np.linalg.norm(J_B, 'fro'))
    if norm_factor < 1e-10:
        return 0.0
    return raw_defect / (norm_factor ** 2)


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                 data_range: Optional[float] = None,
                 C1: float = 0.01, C2: float = 0.03) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    SSIM = (2μ₁μ₂ + c₁)(2σ₁₂ + c₂) / ((μ₁² + μ₂² + c₁)(σ₁² + σ₂² + c₂))
    
    Parameters
    ----------
    img1, img2 : np.ndarray
        Images of same shape (H, W)
    data_range : float, optional
        Dynamic range of images. If None, uses max - min of img1.
    C1, C2 : float
        Stability constants as fractions of data_range
    
    Returns
    -------
    float
        SSIM value in [-1, 1], with 1 being identical
    """
    if data_range is None:
        data_range = img1.max() - img1.min()
        if data_range == 0:
            data_range = 1.0
    
    c1 = (C1 * data_range) ** 2
    c2 = (C2 * data_range) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    return numerator / denominator


def compute_psnr(img1: np.ndarray, img2: np.ndarray,
                 data_range: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    PSNR = 10 log₁₀(MAX² / MSE)
    
    Parameters
    ----------
    img1, img2 : np.ndarray
        Images of same shape
    data_range : float, optional
        Maximum pixel value. If None, uses max of img1.
    
    Returns
    -------
    float
        PSNR in dB. Higher is better. Returns inf for identical images.
    """
    if data_range is None:
        data_range = img1.max()
        if data_range == 0:
            data_range = 1.0
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10(data_range ** 2 / mse)


def compute_reconstruction_metrics(B_true: np.ndarray, B_pred: np.ndarray,
                                    image_shape: Optional[Tuple[int, int]] = None
                                    ) -> dict:
    """
    Compute all reconstruction metrics.
    
    Parameters
    ----------
    B_true : np.ndarray
        True MM features of shape (m, ℓ)
    B_pred : np.ndarray
        Predicted MM features of shape (m, ℓ)
    image_shape : tuple, optional
        If MM features are images, reshape to (H, W) for SSIM/PSNR
    
    Returns
    -------
    dict
        Dictionary with MSE, and optionally SSIM/PSNR per sample
    """
    m = B_true.shape[0]
    
    results = {
        'mse': fidelity_mse(B_true, B_pred),
        'frobenius_sq': fidelity_frobenius(B_true, B_pred),
    }
    
    if image_shape is not None:
        ssim_vals = []
        psnr_vals = []
        
        for i in range(m):
            img1 = B_true[i].reshape(image_shape)
            img2 = B_pred[i].reshape(image_shape)
            
            ssim_vals.append(compute_ssim(img1, img2))
            psnr_vals.append(compute_psnr(img1, img2))
        
        results['ssim'] = np.array(ssim_vals)
        results['ssim_mean'] = np.mean(ssim_vals)
        results['ssim_std'] = np.std(ssim_vals)
        results['psnr'] = np.array(psnr_vals)
        results['psnr_mean'] = np.mean(psnr_vals)
        results['psnr_std'] = np.std(psnr_vals)
    
    return results


def robustness_error(B_target: np.ndarray, B_pred: np.ndarray) -> float:
    """
    Compute robustness error on transformed data.
    
    Used in Scenario 3 (Robustness Test) to compare predictions
    on rotated data against ideal rotated targets.
    
    Parameters
    ----------
    B_target : np.ndarray
        Ideal MM features after transformation (m, ℓ)
    B_pred : np.ndarray
        Predicted MM features on transformed FM data (m, ℓ)
    
    Returns
    -------
    float
        MSE between target and prediction
    """
    return fidelity_mse(B_target, B_pred)


def print_comparison_table(metrics_old: dict, metrics_new: dict,
                           title: str = "Results Comparison") -> str:
    """
    Format comparison of old vs new approach as ASCII table.
    
    Parameters
    ----------
    metrics_old : dict
        Metrics for baseline approach
    metrics_new : dict
        Metrics for equivariant approach
    title : str
        Table title
    
    Returns
    -------
    str
        Formatted table string
    """
    lines = [
        f"\n{title}",
        "-" * 50,
        f"{'Metric':<30} {'Old':>8} {'New':>8}",
        "-" * 50,
    ]
    
    for key in metrics_old:
        if key in metrics_new:
            old_val = metrics_old[key]
            new_val = metrics_new[key]
            if isinstance(old_val, float):
                lines.append(f"{key:<30} {old_val:>8.4f} {new_val:>8.4f}")
    
    lines.append("-" * 50)
    
    return "\n".join(lines)
