"""
Step 5.1: Compute MNIST Generators

This script implements generator estimation for the full MNIST dataset as described
in Section 3.1 of the manuscript. It computes the infinitesimal generators J^A (490×490)
and J^B (784×784) representing rotational transformations in the CNN feature space
and pixel space, respectively.

Key steps:
1. Load trained CNN model and MNIST data
2. Select subset of images (1000 samples for efficiency)
3. Apply small rotation (ε=0.01 rad) and compute finite differences
4. Solve for generators using least squares
5. Save J^A and J^B for use in equivariant solver

Author: K-Dense Coding Agent
Date: 2026-01-18
"""

import numpy as np
import torch
import sys
import os
import time
from pathlib import Path

# Add workflow directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MNISTCNN
from mnist_loader import load_mnist
from generators import estimate_generators

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
EPSILON = 0.01  # Small rotation angle in radians
N_SAMPLES = 1000  # Number of samples to use for generator estimation (subset)
BATCH_SIZE = 100  # Batch size for processing
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("STEP 5.1: MNIST Generator Estimation")
print("=" * 80)
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
print(f"✓ Model loaded from {model_path}")
print(f"  Using device: {DEVICE}")
print()

# Load MNIST data
print("Loading MNIST dataset...")
train_data, train_labels, test_data, test_labels = load_mnist()
print(f"✓ MNIST loaded")
print(f"  Training samples: {train_data.shape[0]}")
print(f"  Test samples: {test_data.shape[0]}")
print()

# Select subset for generator estimation
print(f"Selecting {N_SAMPLES} samples for generator estimation...")
# Use training set for generator estimation
subset_data = torch.from_numpy(train_data[:N_SAMPLES]).float()
print(f"✓ Selected {len(subset_data)} samples")
print(f"  Shape: {subset_data.shape} (N, C, H, W)")
print()

# Estimate generators using finite differences
print("Estimating generators J^A and J^B...")
print(f"  Rotation angle: ε = {EPSILON:.4f} rad ({EPSILON * 180 / np.pi:.4f}°)")
print(f"  Expected dimensions:")
print(f"    J^A: (490, 490) - CNN feature space generator")
print(f"    J^B: (784, 784) - Pixel space generator")
print()

start_time = time.time()
J_A, J_B = estimate_generators(
    model=model,
    data=subset_data,
    epsilon=EPSILON,
    device=DEVICE,
    batch_size=BATCH_SIZE
)
elapsed = time.time() - start_time

print()
print(f"✓ Generator estimation complete in {elapsed:.2f} seconds")
print(f"  J^A shape: {J_A.shape}")
print(f"  J^B shape: {J_B.shape}")
print()

# Verify antisymmetry (generators should be antisymmetric matrices)
print("Verifying generator properties...")
antisym_A = np.linalg.norm(J_A + J_A.T, 'fro')
antisym_B = np.linalg.norm(J_B + J_B.T, 'fro')
print(f"  Antisymmetry check (should be small):")
print(f"    ||J^A + J^A^T||_F = {antisym_A:.6f}")
print(f"    ||J^B + J^B^T||_F = {antisym_B:.6f}")

# Check norms
norm_A = np.linalg.norm(J_A, 'fro')
norm_B = np.linalg.norm(J_B, 'fro')
print(f"  Frobenius norms:")
print(f"    ||J^A||_F = {norm_A:.6f}")
print(f"    ||J^B||_F = {norm_B:.6f}")
print()

# Save generators
output_file = DATA_DIR / "mnist_generators.npz"
print(f"Saving generators to {output_file}...")
np.savez(
    output_file,
    J_A=J_A,
    J_B=J_B,
    epsilon=EPSILON,
    n_samples=N_SAMPLES,
    antisym_A=antisym_A,
    antisym_B=antisym_B,
    norm_A=norm_A,
    norm_B=norm_B
)
print(f"✓ Generators saved successfully")
print()

print("=" * 80)
print("STEP 5.1: COMPLETE")
print("=" * 80)
print(f"Output: {output_file}")
print(f"  J^A: {J_A.shape} (CNN feature generator)")
print(f"  J^B: {J_B.shape} (Pixel space generator)")
print()
print("Next step: Run 06_mnist_solver.py to compute T_new")
print("=" * 80)
