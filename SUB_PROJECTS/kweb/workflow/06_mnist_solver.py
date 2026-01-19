"""
Step 5.2: Solve for Equivariant Transition Matrix

This script solves for the equivariant transition matrix T_new using the
methodology from Section 3.2-3.3 of the manuscript. It minimizes the combined
objective:
    L(T) = ||B^T - T A^T||_F^2 + λ ||T J^A - J^B T||_F^2

The first term enforces reconstruction accuracy, while the second term enforces
equivariance with respect to rotations.

Key steps:
1. Load trained CNN and extract features from MNIST training set
2. Load generators J^A and J^B from previous step
3. Use EquivariantSolver to find T_new with λ=0.5
4. Save T_new for evaluation

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
from solver import EquivariantSolver
from generators import compute_symmetry_defect

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
LAMBDA_REG = 0.5  # Regularization weight for symmetry constraint
N_SAMPLES = 10000  # Number of samples to use for solving (subset for efficiency)
BATCH_SIZE = 500  # Batch size for feature extraction
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("STEP 5.2: Solve for Equivariant Transition Matrix T_new")
print("=" * 80)
print()

# Load generators
print("Loading generators J^A and J^B...")
generators_path = DATA_DIR / "mnist_generators.npz"
if not generators_path.exists():
    raise FileNotFoundError(f"Generators not found at {generators_path}")

generators_data = np.load(generators_path)
J_A = generators_data['J_A']
J_B = generators_data['J_B']
epsilon = generators_data['epsilon']
n_gen_samples = generators_data['n_samples']

print(f"✓ Generators loaded from {generators_path}")
print(f"  J^A shape: {J_A.shape}")
print(f"  J^B shape: {J_B.shape}")
print(f"  Estimated from {n_gen_samples} samples with ε={epsilon:.4f} rad")
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
print()

# Extract features from training set (use subset for efficiency)
print(f"Extracting features from {N_SAMPLES} training samples...")
print(f"  Processing in batches of {BATCH_SIZE}")

A_list = []  # CNN features
B_list = []  # Pixel features

start_time = time.time()
with torch.no_grad():
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        if i % (BATCH_SIZE * 5) == 0:
            print(f"  Progress: {i}/{N_SAMPLES} samples processed")

        batch_end = min(i + BATCH_SIZE, N_SAMPLES)
        batch_data = torch.from_numpy(train_data[i:batch_end]).float()

        # Extract CNN features (formal model)
        batch_data_device = batch_data.to(DEVICE)
        A_batch = model.get_penultimate_features(batch_data_device).cpu().numpy()
        A_list.append(A_batch)

        # Extract pixel features (mental model)
        B_batch = batch_data.numpy().reshape(batch_data.shape[0], -1)
        B_list.append(B_batch)

print(f"  Progress: {N_SAMPLES}/{N_SAMPLES} samples processed")

# Concatenate batches
A = np.vstack(A_list)  # Shape: (N, 490)
B = np.vstack(B_list)  # Shape: (N, 784)
elapsed = time.time() - start_time

print(f"✓ Feature extraction complete in {elapsed:.2f} seconds")
print(f"  A (CNN features) shape: {A.shape}")
print(f"  B (pixel features) shape: {B.shape}")
print()

# Initialize equivariant solver
print("Initializing EquivariantSolver...")
solver = EquivariantSolver(
    A=A,
    B=B,
    J_A=J_A,
    J_B=J_B,
    lambda_reg=LAMBDA_REG
)
print()

# Solve for T_new
print("Solving for equivariant transition matrix T_new...")
print(f"  Objective: L(T) = ||B^T - T A^T||_F^2 + λ ||T J^A - J^B T||_F^2")
print(f"  Regularization: λ = {LAMBDA_REG}")
print()

T_new, info = solver.solve(
    method='lsqr',
    max_iter=1000,
    tol=1e-6,
    verbose=True
)
print()

print(f"✓ Solver complete")
print(f"  T_new shape: {T_new.shape}")
print(f"  Iterations: {info['iterations']}")
print(f"  Status: {info['status']} (1=converged, 2=least squares solution)")
print()

# Compute metrics for T_new
print("Computing metrics for T_new...")

# Reconstruction error
reconstruction = B.T - T_new @ A.T
recon_error_new = np.linalg.norm(reconstruction, 'fro') ** 2 / (A.shape[0] * B.shape[1])

# Symmetry defect
symm_defect_new = compute_symmetry_defect(T_new, J_A, J_B)

# Total objective
total_obj_new = recon_error_new + LAMBDA_REG * (symm_defect_new ** 2)

print(f"  Reconstruction error: {recon_error_new:.6f}")
print(f"  Symmetry defect: ||T J^A - J^B T||_F = {symm_defect_new:.6f}")
print(f"  Total objective: {total_obj_new:.6f}")
print()

# Compare with baseline T_old
print("Loading baseline T_old for comparison...")
t_old_path = DATA_DIR / "t_old_mnist.npy"
if t_old_path.exists():
    T_old = np.load(t_old_path)

    # Compute metrics for T_old
    reconstruction_old = B.T - T_old @ A.T
    recon_error_old = np.linalg.norm(reconstruction_old, 'fro') ** 2 / (A.shape[0] * B.shape[1])
    symm_defect_old = compute_symmetry_defect(T_old, J_A, J_B)
    total_obj_old = recon_error_old + LAMBDA_REG * (symm_defect_old ** 2)

    print(f"✓ Baseline T_old metrics:")
    print(f"  Reconstruction error: {recon_error_old:.6f}")
    print(f"  Symmetry defect: ||T J^A - J^B T||_F = {symm_defect_old:.6f}")
    print(f"  Total objective: {total_obj_old:.6f}")
    print()

    # Compute improvements
    symm_improvement = (symm_defect_old - symm_defect_new) / symm_defect_old * 100
    obj_improvement = (total_obj_old - total_obj_new) / total_obj_old * 100

    print(f"Improvements:")
    print(f"  Symmetry defect reduction: {symm_improvement:.2f}%")
    print(f"  Total objective improvement: {obj_improvement:.2f}%")
    print()
else:
    print(f"⚠ Baseline T_old not found at {t_old_path}")
    print()

# Save T_new
output_file = DATA_DIR / "t_new_mnist.npy"
print(f"Saving T_new to {output_file}...")
np.save(output_file, T_new)
print(f"✓ T_new saved successfully")
print()

print("=" * 80)
print("STEP 5.2: COMPLETE")
print("=" * 80)
print(f"Output: {output_file}")
print(f"  T_new shape: {T_new.shape}")
print(f"  Reconstruction error: {recon_error_new:.6f}")
print(f"  Symmetry defect: {symm_defect_new:.6f}")
print()
print("Next step: Run 07_mnist_evaluation.py to evaluate robustness")
print("=" * 80)
