"""
Step 5.3: MNIST Experiment Evaluation and Visualization

This script evaluates the robustness of T_old (baseline) vs T_new (equivariant)
against rotational transformations on MNIST images. It implements the experimental
protocol from Section 4 of the manuscript.

Key steps:
1. Load T_old and T_new transition matrices
2. Rotate MNIST test set images by angles in [-15°, 15°]
3. Reconstruct using both T_old and T_new
4. Compute SSIM and PSNR metrics
5. Generate visualizations comparing reconstruction quality
6. Save metrics to results/mnist_experiment.json

Author: K-Dense Coding Agent
Date: 2026-01-18
"""

import numpy as np
import torch
import sys
import os
import time
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from typing import Tuple

# Add workflow directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MNISTCNN
from mnist_loader import load_mnist

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Rotation experiment parameters
ROTATION_ANGLES = np.linspace(-15, 15, 15)  # Degrees
N_TEST_SAMPLES = 1000  # Number of test samples to evaluate
N_VIS_SAMPLES = 5  # Number of samples to visualize

# Create output directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("STEP 5.3: MNIST Experiment Evaluation and Visualization")
print("=" * 80)
print()


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a single image by the specified angle.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W) or (C, H, W)
    angle_deg : float
        Rotation angle in degrees

    Returns
    -------
    np.ndarray
        Rotated image with same shape as input
    """
    if img.ndim == 2:
        return rotate(img, angle_deg, reshape=False, order=1, mode='constant')
    else:
        # Multiple channels
        rotated = np.zeros_like(img)
        for c in range(img.shape[0]):
            rotated[c] = rotate(img[c], angle_deg, reshape=False, order=1, mode='constant')
        return rotated


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images to compare (same shape)
    max_val : float
        Maximum possible pixel value

    Returns
    -------
    float
        PSNR in decibels
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Parameters
    ----------
    img1, img2 : np.ndarray
        Images to compare (same shape, assumed to be 2D)

    Returns
    -------
    float
        SSIM value in [0, 1]
    """
    # Simple SSIM implementation (windowed version is more robust but requires skimage)
    # We'll try to import skimage, fall back to simple version if not available
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2, data_range=1.0)
    except ImportError:
        # Simple SSIM without windowing
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        return ssim


print("Loading transition matrices...")
t_old_path = DATA_DIR / "t_old_mnist.npy"
t_new_path = DATA_DIR / "t_new_mnist.npy"

if not t_old_path.exists():
    raise FileNotFoundError(f"T_old not found at {t_old_path}")
if not t_new_path.exists():
    raise FileNotFoundError(f"T_new not found at {t_new_path}")

T_old = np.load(t_old_path)
T_new = np.load(t_new_path)

print(f"✓ Transition matrices loaded")
print(f"  T_old shape: {T_old.shape}")
print(f"  T_new shape: {T_new.shape}")
print()

# Load trained CNN model
print("Loading trained CNN model...")
model = MNISTCNN()
model_path = DATA_DIR / "cnn_mnist.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"✓ Model loaded")
print()

# Load MNIST test data
print("Loading MNIST test set...")
train_data, train_labels, test_data, test_labels = load_mnist()
print(f"✓ MNIST test set loaded ({test_data.shape[0]} samples)")
print()

# Run rotation experiment
print(f"Running rotation robustness experiment...")
print(f"  Rotation angles: {ROTATION_ANGLES[0]:.1f}° to {ROTATION_ANGLES[-1]:.1f}° ({len(ROTATION_ANGLES)} steps)")
print(f"  Test samples: {N_TEST_SAMPLES}")
print()

# Storage for metrics
metrics_old = {'ssim': [], 'psnr': [], 'mse': []}
metrics_new = {'ssim': [], 'psnr': [], 'mse': []}
metrics_by_angle_old = {angle: {'ssim': [], 'psnr': []} for angle in ROTATION_ANGLES}
metrics_by_angle_new = {angle: {'ssim': [], 'psnr': []} for angle in ROTATION_ANGLES}

# Storage for visualizations
vis_originals = []
vis_rotated_inputs = []
vis_recon_old = []
vis_recon_new = []
vis_angles = []

start_time = time.time()
n_processed = 0

with torch.no_grad():
    for sample_idx in range(N_TEST_SAMPLES):
        if sample_idx % 100 == 0:
            print(f"  Progress: {sample_idx}/{N_TEST_SAMPLES} samples processed")

        # Get original image
        img_np = test_data[sample_idx].squeeze()  # Shape: (28, 28)

        # For each rotation angle
        for angle in ROTATION_ANGLES:
            # Rotate image
            img_rotated_np = rotate_image(img_np, angle)
            img_rotated = torch.from_numpy(img_rotated_np).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 28, 28)

            # Extract CNN features from rotated image
            A_rotated = model.get_penultimate_features(img_rotated.to(DEVICE)).cpu().numpy()  # (1, 490)

            # Reconstruct using T_old and T_new
            B_recon_old = (T_old @ A_rotated.T).T  # (1, 784)
            B_recon_new = (T_new @ A_rotated.T).T  # (1, 784)

            # Reshape to images
            img_recon_old = B_recon_old.reshape(28, 28)
            img_recon_new = B_recon_new.reshape(28, 28)

            # Compute metrics vs original
            ssim_old = compute_ssim(img_np, img_recon_old)
            ssim_new = compute_ssim(img_np, img_recon_new)
            psnr_old = compute_psnr(img_np, img_recon_old)
            psnr_new = compute_psnr(img_np, img_recon_new)
            mse_old = np.mean((img_np - img_recon_old) ** 2)
            mse_new = np.mean((img_np - img_recon_new) ** 2)

            # Store metrics
            metrics_old['ssim'].append(ssim_old)
            metrics_old['psnr'].append(psnr_old)
            metrics_old['mse'].append(mse_old)
            metrics_new['ssim'].append(ssim_new)
            metrics_new['psnr'].append(psnr_new)
            metrics_new['mse'].append(mse_new)

            metrics_by_angle_old[angle]['ssim'].append(ssim_old)
            metrics_by_angle_old[angle]['psnr'].append(psnr_old)
            metrics_by_angle_new[angle]['ssim'].append(ssim_new)
            metrics_by_angle_new[angle]['psnr'].append(psnr_new)

        n_processed += 1

        # Save samples for visualization (mid-range rotation)
        if sample_idx < N_VIS_SAMPLES:
            angle_vis = 10.0  # 10 degree rotation for visualization
            img_rotated_vis = rotate_image(img_np, angle_vis)
            img_rotated_tensor = torch.from_numpy(img_rotated_vis).unsqueeze(0).unsqueeze(0).float()
            A_vis = model.get_penultimate_features(img_rotated_tensor.to(DEVICE)).cpu().numpy()
            B_recon_old_vis = (T_old @ A_vis.T).T.reshape(28, 28)
            B_recon_new_vis = (T_new @ A_vis.T).T.reshape(28, 28)

            vis_originals.append(img_np)
            vis_rotated_inputs.append(img_rotated_vis)
            vis_recon_old.append(B_recon_old_vis)
            vis_recon_new.append(B_recon_new_vis)
            vis_angles.append(angle_vis)

print(f"  Progress: {N_TEST_SAMPLES}/{N_TEST_SAMPLES} samples processed")
elapsed = time.time() - start_time
print(f"✓ Rotation experiment complete in {elapsed:.2f} seconds")
print()

# Compute summary statistics
print("Computing summary statistics...")
results = {
    'experiment': 'mnist_rotation_robustness',
    'n_samples': N_TEST_SAMPLES,
    'rotation_angles': ROTATION_ANGLES.tolist(),
    'baseline': {
        'ssim_mean': float(np.mean(metrics_old['ssim'])),
        'ssim_std': float(np.std(metrics_old['ssim'])),
        'psnr_mean': float(np.mean(metrics_old['psnr'])),
        'psnr_std': float(np.std(metrics_old['psnr'])),
        'mse_mean': float(np.mean(metrics_old['mse'])),
        'mse_std': float(np.std(metrics_old['mse']))
    },
    'equivariant': {
        'ssim_mean': float(np.mean(metrics_new['ssim'])),
        'ssim_std': float(np.std(metrics_new['ssim'])),
        'psnr_mean': float(np.mean(metrics_new['psnr'])),
        'psnr_std': float(np.std(metrics_new['psnr'])),
        'mse_mean': float(np.mean(metrics_new['mse'])),
        'mse_std': float(np.std(metrics_new['mse']))
    },
    'by_angle': {
        f'{angle:.1f}deg': {
            'baseline_ssim': float(np.mean(metrics_by_angle_old[angle]['ssim'])),
            'baseline_psnr': float(np.mean(metrics_by_angle_old[angle]['psnr'])),
            'equivariant_ssim': float(np.mean(metrics_by_angle_new[angle]['ssim'])),
            'equivariant_psnr': float(np.mean(metrics_by_angle_new[angle]['psnr']))
        }
        for angle in ROTATION_ANGLES
    }
}

print("✓ Summary statistics computed")
print()
print(f"Baseline (T_old):")
print(f"  SSIM: {results['baseline']['ssim_mean']:.4f} ± {results['baseline']['ssim_std']:.4f}")
print(f"  PSNR: {results['baseline']['psnr_mean']:.2f} ± {results['baseline']['psnr_std']:.2f} dB")
print(f"  MSE:  {results['baseline']['mse_mean']:.6f} ± {results['baseline']['mse_std']:.6f}")
print()
print(f"Equivariant (T_new):")
print(f"  SSIM: {results['equivariant']['ssim_mean']:.4f} ± {results['equivariant']['ssim_std']:.4f}")
print(f"  PSNR: {results['equivariant']['psnr_mean']:.2f} ± {results['equivariant']['psnr_std']:.2f} dB")
print(f"  MSE:  {results['equivariant']['mse_mean']:.6f} ± {results['equivariant']['mse_std']:.6f}")
print()

# Save metrics
output_json = RESULTS_DIR / "mnist_experiment.json"
print(f"Saving metrics to {output_json}...")
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Metrics saved")
print()

# Generate visualizations
print("Generating visualizations...")

# Figure 1: Side-by-side reconstruction comparison
fig, axes = plt.subplots(N_VIS_SAMPLES, 4, figsize=(12, 3 * N_VIS_SAMPLES))
if N_VIS_SAMPLES == 1:
    axes = axes.reshape(1, -1)

for i in range(N_VIS_SAMPLES):
    # Original
    axes[i, 0].imshow(vis_originals[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f'Original (Sample {i+1})')
    axes[i, 0].axis('off')

    # Rotated input
    axes[i, 1].imshow(vis_rotated_inputs[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_title(f'Rotated (+{vis_angles[i]:.0f}°)')
    axes[i, 1].axis('off')

    # T_old reconstruction
    axes[i, 2].imshow(vis_recon_old[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_title('Baseline Recon')
    axes[i, 2].axis('off')

    # T_new reconstruction
    axes[i, 3].imshow(vis_recon_new[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 3].set_title('Equivariant Recon')
    axes[i, 3].axis('off')

plt.tight_layout()
fig_path = FIGURES_DIR / "mnist_reconstruction_comparison.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {fig_path}")

# Figure 2: Difference maps
fig, axes = plt.subplots(N_VIS_SAMPLES, 3, figsize=(9, 3 * N_VIS_SAMPLES))
if N_VIS_SAMPLES == 1:
    axes = axes.reshape(1, -1)

for i in range(N_VIS_SAMPLES):
    # Rotated input
    axes[i, 0].imshow(vis_rotated_inputs[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f'Rotated Input (+{vis_angles[i]:.0f}°)')
    axes[i, 0].axis('off')

    # T_old error
    diff_old = np.abs(vis_originals[i] - vis_recon_old[i])
    im1 = axes[i, 1].imshow(diff_old, cmap='hot', vmin=0, vmax=0.5)
    axes[i, 1].set_title(f'Baseline Error (MAE={np.mean(diff_old):.4f})')
    axes[i, 1].axis('off')
    plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

    # T_new error
    diff_new = np.abs(vis_originals[i] - vis_recon_new[i])
    im2 = axes[i, 2].imshow(diff_new, cmap='hot', vmin=0, vmax=0.5)
    axes[i, 2].set_title(f'Equivariant Error (MAE={np.mean(diff_new):.4f})')
    axes[i, 2].axis('off')
    plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

plt.tight_layout()
fig_path = FIGURES_DIR / "mnist_difference_maps.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {fig_path}")

# Figure 3: Box plots of SSIM/PSNR distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SSIM box plot
data_ssim = [metrics_old['ssim'], metrics_new['ssim']]
bp1 = axes[0].boxplot(data_ssim, labels=['Baseline', 'Equivariant'], patch_artist=True)
for patch, color in zip(bp1['boxes'], ['lightblue', 'lightgreen']):
    patch.set_facecolor(color)
axes[0].set_ylabel('SSIM', fontsize=12)
axes[0].set_title('Structural Similarity (SSIM)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# PSNR box plot
data_psnr = [metrics_old['psnr'], metrics_new['psnr']]
bp2 = axes[1].boxplot(data_psnr, labels=['Baseline', 'Equivariant'], patch_artist=True)
for patch, color in zip(bp2['boxes'], ['lightblue', 'lightgreen']):
    patch.set_facecolor(color)
axes[1].set_ylabel('PSNR (dB)', fontsize=12)
axes[1].set_title('Peak Signal-to-Noise Ratio (PSNR)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = FIGURES_DIR / "mnist_robustness_boxplots.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {fig_path}")

# Figure 4: Metrics vs rotation angle
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

angles = list(sorted(metrics_by_angle_old.keys()))
ssim_old_by_angle = [np.mean(metrics_by_angle_old[a]['ssim']) for a in angles]
ssim_new_by_angle = [np.mean(metrics_by_angle_new[a]['ssim']) for a in angles]
psnr_old_by_angle = [np.mean(metrics_by_angle_old[a]['psnr']) for a in angles]
psnr_new_by_angle = [np.mean(metrics_by_angle_new[a]['psnr']) for a in angles]

axes[0].plot(angles, ssim_old_by_angle, 'o-', label='Baseline', color='blue', linewidth=2)
axes[0].plot(angles, ssim_new_by_angle, 's-', label='Equivariant', color='green', linewidth=2)
axes[0].set_xlabel('Rotation Angle (degrees)', fontsize=12)
axes[0].set_ylabel('SSIM', fontsize=12)
axes[0].set_title('SSIM vs Rotation Angle', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(angles, psnr_old_by_angle, 'o-', label='Baseline', color='blue', linewidth=2)
axes[1].plot(angles, psnr_new_by_angle, 's-', label='Equivariant', color='green', linewidth=2)
axes[1].set_xlabel('Rotation Angle (degrees)', fontsize=12)
axes[1].set_ylabel('PSNR (dB)', fontsize=12)
axes[1].set_title('PSNR vs Rotation Angle', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig_path = FIGURES_DIR / "mnist_robustness_vs_angle.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {fig_path}")

print()
print("=" * 80)
print("STEP 5.3: COMPLETE")
print("=" * 80)
print(f"Metrics saved to: {output_json}")
print()
print("Generated figures:")
print(f"  1. {FIGURES_DIR / 'mnist_reconstruction_comparison.png'}")
print(f"  2. {FIGURES_DIR / 'mnist_difference_maps.png'}")
print(f"  3. {FIGURES_DIR / 'mnist_robustness_boxplots.png'}")
print(f"  4. {FIGURES_DIR / 'mnist_robustness_vs_angle.png'}")
print()
print("=" * 80)
print("STEP 5: MNIST EXPERIMENT EXECUTION - ALL COMPLETE")
print("=" * 80)
