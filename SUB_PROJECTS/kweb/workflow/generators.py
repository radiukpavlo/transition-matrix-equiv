"""
Generator Estimation for Equivariant Representation Learning

This module implements generator estimation as described in Section 3.1 of the manuscript.
Generators capture the infinitesimal transformations of representations under group actions
(e.g., rotations of images).

Key concepts:
- J^A: Generator for Formal Model features (penultimate layer of CNN)
- J^B: Generator for Mental Model features (pixel space)
- Symmetry Defect: ||T J^A - J^B T||_F measures how well T preserves symmetry

Author: K-Dense Coding Agent
Date: 2026-01-18
"""

import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.linalg import lstsq
from typing import Tuple, Optional


def estimate_generators(
    model: torch.nn.Module,
    data: torch.Tensor,
    epsilon: float = 0.01,
    device: str = 'cpu',
    batch_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate generators J^A and J^B using finite differences.

    This implements the generator estimation from Section 3.1 of the manuscript.
    We apply a small rotation epsilon to input images and compute the resulting
    changes in both the formal model features (A) and mental model features (B).

    The generators are estimated by solving:
        A J^T ≈ ΔA  (formal model generator)
        B J^T ≈ ΔB  (mental model generator)

    using least squares.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network (e.g., CNN) with a get_penultimate_features method
    data : torch.Tensor
        Input images of shape (N, C, H, W)
    epsilon : float, default=0.01
        Small rotation angle in radians for finite difference approximation
    device : str, default='cpu'
        Device for PyTorch computation ('cpu' or 'cuda')
    batch_size : int, default=100
        Batch size for processing to avoid memory issues

    Returns
    -------
    J_A : np.ndarray
        Generator for formal model features, shape (d_A, d_A) where d_A is feature dimension
    J_B : np.ndarray
        Generator for mental model features, shape (d_B, d_B) where d_B is pixel dimension

    Notes
    -----
    - The rotation is applied around the center of the image
    - We use scipy.ndimage.rotate with order=1 (bilinear interpolation)
    - The generators are antisymmetric matrices representing infinitesimal rotations
    """
    model.eval()
    model = model.to(device)

    n_samples = data.shape[0]
    epsilon_deg = epsilon * 180 / np.pi  # Convert radians to degrees for scipy

    # Storage for features
    A_original_list = []
    A_rotated_list = []
    B_original_list = []
    B_rotated_list = []

    print(f"Estimating generators with epsilon={epsilon:.4f} rad ({epsilon_deg:.4f} deg)")
    print(f"Processing {n_samples} samples in batches of {batch_size}")

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            if i % (batch_size * 10) == 0:
                print(f"  Progress: {i}/{n_samples} samples processed")

            batch_end = min(i + batch_size, n_samples)
            batch_data = data[i:batch_end]

            # Original features
            batch_original = batch_data.to(device)
            A_original = model.get_penultimate_features(batch_original).cpu().numpy()
            B_original = batch_data.cpu().numpy().reshape(batch_data.shape[0], -1)

            # Apply rotation to images
            batch_rotated_np = np.zeros_like(batch_data.cpu().numpy())
            for j, img in enumerate(batch_data.cpu().numpy()):
                # img shape: (C, H, W)
                for c in range(img.shape[0]):
                    batch_rotated_np[j, c] = rotate(
                        img[c],
                        epsilon_deg,
                        reshape=False,
                        order=1,  # Bilinear interpolation
                        mode='constant'
                    )

            # Rotated features
            batch_rotated = torch.from_numpy(batch_rotated_np).float().to(device)
            A_rotated = model.get_penultimate_features(batch_rotated).cpu().numpy()
            B_rotated = batch_rotated_np.reshape(batch_data.shape[0], -1)

            A_original_list.append(A_original)
            A_rotated_list.append(A_rotated)
            B_original_list.append(B_original)
            B_rotated_list.append(B_rotated)

    print(f"  Progress: {n_samples}/{n_samples} samples processed")

    # Concatenate all batches
    A_original = np.vstack(A_original_list)  # Shape: (N, d_A)
    A_rotated = np.vstack(A_rotated_list)
    B_original = np.vstack(B_original_list)  # Shape: (N, d_B)
    B_rotated = np.vstack(B_rotated_list)

    # Compute finite differences
    Delta_A = (A_rotated - A_original) / epsilon  # Shape: (N, d_A)
    Delta_B = (B_rotated - B_original) / epsilon  # Shape: (N, d_B)

    print(f"Computing generators via least squares...")
    print(f"  A shape: {A_original.shape}, ΔA shape: {Delta_A.shape}")
    print(f"  B shape: {B_original.shape}, ΔB shape: {Delta_B.shape}")

    # Solve A J^T ≈ ΔA for J^A using least squares
    # This is equivalent to solving (A^T A) J^T = A^T ΔA
    # Or: J = (ΔA^T A) (A^T A)^{-1}
    # We use lstsq to solve A J^T = ΔA directly
    J_A_T, residuals_A, rank_A, s_A = lstsq(A_original, Delta_A)
    J_A = J_A_T.T  # Shape: (d_A, d_A)

    # Solve B J^T ≈ ΔB for J^B
    J_B_T, residuals_B, rank_B, s_B = lstsq(B_original, Delta_B)
    J_B = J_B_T.T  # Shape: (d_B, d_B)

    print(f"Generator estimation complete!")
    print(f"  J^A shape: {J_A.shape}, rank: {rank_A}/{A_original.shape[1]}")
    print(f"  J^B shape: {J_B.shape}, rank: {rank_B}/{B_original.shape[1]}")
    print(f"  Residual norms: ||A J^A^T - ΔA||_F = {np.sqrt(residuals_A.sum()):.6f}, "
          f"||B J^B^T - ΔB||_F = {np.sqrt(residuals_B.sum()):.6f}")

    return J_A, J_B


def compute_symmetry_defect(
    T: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray
) -> float:
    """
    Compute the Symmetry Defect: ||T J^A - J^B T||_F.

    This measures how well the transition matrix T preserves the symmetry
    encoded in the generators. A lower defect indicates better equivariance.

    Parameters
    ----------
    T : np.ndarray
        Transition matrix of shape (d_B, d_A)
    J_A : np.ndarray
        Generator for formal model features, shape (d_A, d_A)
    J_B : np.ndarray
        Generator for mental model features, shape (d_B, d_B)

    Returns
    -------
    defect : float
        Frobenius norm of the symmetry defect

    Notes
    -----
    The symmetry defect measures the violation of the equivariance condition:
        T ∘ g_A = g_B ∘ T
    where g_A and g_B are group actions on the two representations.
    """
    # Compute T J^A - J^B T
    left = T @ J_A
    right = J_B @ T
    defect_matrix = left - right

    # Frobenius norm
    defect = np.linalg.norm(defect_matrix, 'fro')

    return defect


def estimate_generators_bridge(
    A: np.ndarray,
    B: np.ndarray,
    T: np.ndarray,
    epsilon: float = 0.01,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate generators using the "bridge" method described in Algorithm 2 (Section 3.4.1).

    This is a fallback method for synthetic data where we don't have actual transformations.
    We use MDS to create a low-dimensional embedding and then decode back to estimate
    the tangent space structure.

    Parameters
    ----------
    A : np.ndarray
        Formal model features, shape (N, d_A)
    B : np.ndarray
        Mental model features, shape (N, d_B)
    T : np.ndarray
        Transition matrix, shape (d_B, d_A)
    epsilon : float, default=0.01
        Perturbation scale for finite differences
    n_components : int, default=2
        Number of MDS components (dimensionality of latent space)

    Returns
    -------
    J_A : np.ndarray
        Estimated generator for A, shape (d_A, d_A)
    J_B : np.ndarray
        Estimated generator for B, shape (d_B, d_B)

    Notes
    -----
    This is primarily used for synthetic data validation where we don't have
    natural group actions (like rotations) available.
    """
    from sklearn.manifold import MDS

    print(f"Estimating generators via bridge method (MDS + decoder)")
    print(f"  Input shapes: A {A.shape}, B {B.shape}, T {T.shape}")

    # Step 1: MDS on A to get low-dimensional representation
    mds = MDS(n_components=n_components, random_state=42, dissimilarity='euclidean')
    Z = mds.fit_transform(A)  # Shape: (N, n_components)

    print(f"  MDS embedding: {Z.shape}")

    # Step 2: Create decoder from Z to A and Z to B
    # Using linear decoder: A ≈ Z W_A, B ≈ Z W_B
    W_A, _, _, _ = lstsq(Z, A)  # Shape: (n_components, d_A)
    W_B, _, _, _ = lstsq(Z, B)  # Shape: (n_components, d_B)

    # Step 3: Define a rotation in the latent space Z
    # For 2D, a rotation generator is the antisymmetric matrix [[0, -1], [1, 0]]
    if n_components == 2:
        J_Z = np.array([[0, -1], [1, 0]], dtype=float)
    else:
        # For higher dimensions, use the first 2D plane
        J_Z = np.zeros((n_components, n_components))
        J_Z[0, 1] = -1
        J_Z[1, 0] = 1

    # Step 4: Lift the generator to A and B spaces
    # J^A = W_A^T J_Z W_A (via chain rule)
    # J^B = W_B^T J_Z W_B
    J_A = W_A.T @ J_Z @ W_A  # Shape: (d_A, d_A)
    J_B = W_B.T @ J_Z @ W_B  # Shape: (d_B, d_B)

    print(f"  Bridge method complete!")
    print(f"  J^A shape: {J_A.shape}")
    print(f"  J^B shape: {J_B.shape}")

    return J_A, J_B


if __name__ == "__main__":
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing generator estimation module")
    print("=" * 60)

    # Load synthetic data
    import os
    session_dir = "/app/sandbox/session_20260118_175817_da8f96a1d029"
    data = np.load(os.path.join(session_dir, "workflow/data/synthetic.npz"))
    A = data['A']
    B = data['B']

    print(f"\nSynthetic data: A {A.shape}, B {B.shape}")

    # Compute simple transition matrix
    T_simple = np.linalg.pinv(A.T) @ B.T
    print(f"T_simple shape: {T_simple.T.shape}")

    # Test bridge method
    J_A, J_B = estimate_generators_bridge(A, B, T_simple.T, n_components=2)

    # Compute symmetry defect
    defect = compute_symmetry_defect(T_simple.T, J_A, J_B)
    print(f"\nSymmetry defect: {defect:.6f}")

    print("\n" + "=" * 60)
    print("Generator estimation module test complete!")
    print("=" * 60)
