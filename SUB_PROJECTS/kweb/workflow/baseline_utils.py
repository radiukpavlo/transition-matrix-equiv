"""
Baseline Utilities for Transition Matrix Computation and Reconstruction Metrics

This module implements the baseline approach from the manuscript:
- Compute transition matrix T using pseudoinverse method
- Reconstruct mental model features from formal model features
- Calculate reconstruction quality metrics (MSE, SSIM, PSNR)
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_transition_matrix(A, B):
    """
    Compute transition matrix T using the pseudoinverse method.

    Solves: B ≈ A T^T
    Solution: T = (A^+ B)^T

    where A^+ is the Moore-Penrose pseudoinverse of A.

    Parameters
    ----------
    A : np.ndarray
        Formal Model features, shape (n_samples, n_features_A)
        For MNIST: (n_samples, 490) from CNN penultimate layer
    B : np.ndarray
        Mental Model features, shape (n_samples, n_features_B)
        For MNIST: (n_samples, 784) from flattened images

    Returns
    -------
    T : np.ndarray
        Transition matrix, shape (n_features_B, n_features_A)
        For MNIST: (784, 490)

    Notes
    -----
    The transition matrix T maps from the Formal Model feature space
    to the Mental Model feature space via: B* = A T^T
    """
    print(f"Computing transition matrix...")
    print(f"  A shape: {A.shape} (Formal Model features)")
    print(f"  B shape: {B.shape} (Mental Model features)")

    # Compute pseudoinverse of A: A^+ = (A^T A)^{-1} A^T
    A_pinv = np.linalg.pinv(A)
    print(f"  A pseudoinverse shape: {A_pinv.shape}")

    # Compute T = (A^+ B)^T
    T = (A_pinv @ B).T
    print(f"  T shape: {T.shape}")

    # Verify dimensions
    assert T.shape == (B.shape[1], A.shape[1]), \
        f"Expected T shape ({B.shape[1]}, {A.shape[1]}), got {T.shape}"

    return T


def reconstruct_mental_features(A, T):
    """
    Reconstruct Mental Model features from Formal Model features.

    Computes: B* = A T^T

    Parameters
    ----------
    A : np.ndarray
        Formal Model features, shape (n_samples, n_features_A)
    T : np.ndarray
        Transition matrix, shape (n_features_B, n_features_A)

    Returns
    -------
    B_reconstructed : np.ndarray
        Reconstructed Mental Model features, shape (n_samples, n_features_B)

    Notes
    -----
    This function applies the transition matrix to reconstruct
    the Mental Model representation from the Formal Model features.
    """
    print(f"Reconstructing mental features...")
    print(f"  A shape: {A.shape}")
    print(f"  T shape: {T.shape}")

    # Compute B* = A T^T
    B_reconstructed = A @ T.T
    print(f"  B* shape: {B_reconstructed.shape}")

    return B_reconstructed


def calculate_metrics(original, reconstructed):
    """
    Calculate reconstruction quality metrics.

    Computes:
    - MSE (Mean Squared Error): Lower is better
    - SSIM (Structural Similarity Index): Higher is better (range [0, 1])
    - PSNR (Peak Signal-to-Noise Ratio): Higher is better (in dB)

    Parameters
    ----------
    original : np.ndarray
        Original Mental Model features, shape (n_samples, n_features)
        For image-based data, each row can be reshaped to image dimensions
    reconstructed : np.ndarray
        Reconstructed Mental Model features, shape (n_samples, n_features)

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mse': Mean Squared Error (scalar)
        - 'ssim_mean': Mean SSIM across all samples (scalar)
        - 'ssim_std': Standard deviation of SSIM (scalar)
        - 'psnr_mean': Mean PSNR across all samples (scalar)
        - 'psnr_std': Standard deviation of PSNR (scalar)

    Notes
    -----
    For MNIST data (784 features = 28×28 images):
    - Each feature vector is reshaped to 28×28 for SSIM and PSNR computation
    - These metrics are computed per-image and then averaged
    """
    print(f"Calculating reconstruction metrics...")
    print(f"  Original shape: {original.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")

    # Compute MSE (overall)
    mse = mean_squared_error(original, reconstructed)
    print(f"  MSE: {mse:.6f}")

    # For MNIST: reshape to images for SSIM and PSNR
    # Assuming 784 features = 28×28 images
    if original.shape[1] == 784:
        img_size = 28
        n_samples = original.shape[0]

        ssim_scores = []
        psnr_scores = []

        # Compute SSIM and PSNR per image
        for i in range(n_samples):
            # Reshape to 28×28
            orig_img = original[i].reshape(img_size, img_size)
            recon_img = reconstructed[i].reshape(img_size, img_size)

            # Normalize to [0, 1] range for SSIM and PSNR
            orig_img_norm = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-10)
            recon_img_norm = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-10)

            # Compute SSIM (data_range=1.0 for normalized [0,1] images)
            ssim_val = ssim(orig_img_norm, recon_img_norm, data_range=1.0)
            ssim_scores.append(ssim_val)

            # Compute PSNR (data_range=1.0)
            psnr_val = psnr(orig_img_norm, recon_img_norm, data_range=1.0)
            psnr_scores.append(psnr_val)

            # Print progress every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{n_samples} images...")

        ssim_scores = np.array(ssim_scores)
        psnr_scores = np.array(psnr_scores)

        ssim_mean = np.mean(ssim_scores)
        ssim_std = np.std(ssim_scores)
        psnr_mean = np.mean(psnr_scores)
        psnr_std = np.std(psnr_scores)

        print(f"  SSIM: {ssim_mean:.6f} ± {ssim_std:.6f}")
        print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")

        metrics = {
            'mse': float(mse),
            'ssim_mean': float(ssim_mean),
            'ssim_std': float(ssim_std),
            'psnr_mean': float(psnr_mean),
            'psnr_std': float(psnr_std)
        }
    else:
        # For non-image data, only MSE is meaningful
        print(f"  Warning: Non-image data detected ({original.shape[1]} features)")
        print(f"  SSIM and PSNR are not computed (require image structure)")

        metrics = {
            'mse': float(mse),
            'ssim_mean': None,
            'ssim_std': None,
            'psnr_mean': None,
            'psnr_std': None
        }

    return metrics


def test_utilities():
    """
    Test the utility functions with synthetic data.
    """
    print("=" * 60)
    print("Testing Baseline Utilities")
    print("=" * 60)

    # Create synthetic data
    n_samples = 100
    n_features_A = 50
    n_features_B = 784  # MNIST-like

    np.random.seed(42)
    A = np.random.randn(n_samples, n_features_A)
    B = np.random.randn(n_samples, n_features_B)

    print(f"\nSynthetic data:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")

    # Test 1: Compute transition matrix
    print(f"\n{'='*60}")
    print("Test 1: Compute Transition Matrix")
    print("=" * 60)
    T = compute_transition_matrix(A, B)
    print(f"✓ Transition matrix computed successfully")

    # Test 2: Reconstruct features
    print(f"\n{'='*60}")
    print("Test 2: Reconstruct Mental Features")
    print("=" * 60)
    B_reconstructed = reconstruct_mental_features(A, T)
    print(f"✓ Features reconstructed successfully")

    # Test 3: Calculate metrics
    print(f"\n{'='*60}")
    print("Test 3: Calculate Metrics")
    print("=" * 60)
    metrics = calculate_metrics(B, B_reconstructed)
    print(f"✓ Metrics calculated successfully")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return T, B_reconstructed, metrics


if __name__ == "__main__":
    test_utilities()
