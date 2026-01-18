"""
MNIST Experiment Runner (Section 3.5).

Stages:
1. Train/Load CNN Model
2. Extract Features (FM and MM)
3. Estimate Generators (J_A, J_B) via Rotation
4. Compute Transition Matrices (Baseline vs Equivariant)
5. Evaluate (MSE, SSIM, PSNR, Symmetry Defect)
6. Robustness Test (Rotated Test Set)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baseline import compute_baseline_T, predict_mm_features
from src.equivariant import compute_equivariant_T_auto, compute_equivariant_T_iterative
from src.generators import estimate_generator
from src.metrics import (
    fidelity_mse, symmetry_defect, compute_reconstruction_metrics, 
    robustness_error
)
from experiments.mnist.model import MNIST_CNN
from experiments.mnist.features import (
    get_mnist_data, extract_features, get_rotated_features
)

def train_model(model, train_loader, device='cpu', epochs=1):
    """Simple training loop to ensure meaningful features."""
    print(f"Training model for {epochs} epochs...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}: Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

def run_mnist_experiment(
    n_samples: int = 1000,
    lambda_: float = 0.5,
    epsilon: float = 0.01,
    epochs: int = 1,
    seed: int = 42,
    device: str = 'cpu'
):
    print(f"Running MNIST Experiment with n={n_samples}, lambda={lambda_}...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Setup Data and Model
    train_data = get_mnist_data(train=True, download=True)
    test_data = get_mnist_data(train=False, download=True)
    
    model = MNIST_CNN().to(device)
    
    # Train briefly (or load)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    train_model(model, train_loader, device=device, epochs=epochs)
    
    # 2. Extract Features (Training Set Subset)
    # We use these to solve for T
    print("Extracting features...")
    A_train, B_train, labels_train = extract_features(
        model, train_data, num_samples=n_samples, device=device
    )
    print(f"Features shape: A={A_train.shape}, B={B_train.shape}")
    
    # 3. Estimate Generators
    # Use the same subset (indices 0..n_samples-1)
    indices = np.arange(n_samples) # effectively since we took first n samples from subset
    # But extract_features with num_samples does a random choice with seed 42.
    # To be consistent for Rotation, we need the exact same images.
    # Let's rely on extract_features returning what we need, but for rotation we need dataset access.
    # Refactoring slightly: extract_features uses random choice internally.
    # Let's re-implement indices selection here to be explicit.
    
    dataset_indices = np.random.RandomState(42).choice(len(train_data), n_samples, replace=False)
    
    # Re-extract to be sure (loading overhead is minimal compared to correctness)
    # Note: features.py/extract_features uses seed 42 internally so it should be deterministic
    # We need the indices for get_rotated_features
    
    print("Estimating generators via rotation...")
    A_rot, B_rot = get_rotated_features(
        model, train_data, dataset_indices, angle_rad=epsilon, device=device
    )
    
    # Explicitly calculate generator
    # We learned: delta = (rot - orig)/eps ; A J^T = delta
    J_A = estimate_generator(A_train, A_rot, epsilon)
    J_B = estimate_generator(B_train, B_rot, epsilon)
    
    print(f"Generators estimated. J_A: {J_A.shape}, J_B: {J_B.shape}")
    
    # 4. Compute Transition Matrices
    
    # Baseline
    print("Computing Baseline T_old...")
    T_old = compute_baseline_T(A_train, B_train)
    
    # Equivariant
    print("Computing Equivariant T_new (Iterative LSQR)...")
    # This uses the memory-efficient solver automatically if needed
    # But for n=1000, 490*784 is big enough to prefer iterative usually?
    # 1000 * 784 * 490 float64 is ~3GB. Explicit matrix M is huge.
    # compute_equivariant_T_iterative is mandatory.
    
    T_new, info = compute_equivariant_T_iterative(
        A_train, B_train, J_A, J_B, lambda_=lambda_, show=True
    )
    print(f"LSQR Info: {info['iterations']} iterations, residual: {info['residual_norm']:.4e}")
    
    # 5. Evaluation metrics on TRAIN
    print("\n--- Training Evaluation ---")
    B_pred_old = predict_mm_features(A_train, T_old)
    B_pred_new = predict_mm_features(A_train, T_new)
    
    mse_old = fidelity_mse(B_train, B_pred_old)
    sym_old = symmetry_defect(T_old, J_A, J_B)
    
    mse_new = fidelity_mse(B_train, B_pred_new)
    sym_new = symmetry_defect(T_new, J_A, J_B)
    
    print(f"Old: MSE={mse_old:.6f}, SymDefect={sym_old:.6f}")
    print(f"New: MSE={mse_new:.6f}, SymDefect={sym_new:.6f}")
    
    # 6. Robustness Test (on TEST set)
    print("\n--- Robustness Test (Rotated Test Set) ---")
    n_test = 200 # smaller for speed
    test_indices = np.random.RandomState(99).choice(len(test_data), n_test, replace=False)
    
    # Get original features for test
    A_test, B_test_static, _ = extract_features(model, test_data, num_samples=n_test)
    
    # Random angle rotation for test set
    test_angle = np.pi / 6 # 30 degrees
    # Ideally should be random per sample, but block rotation is easier to implement first
    
    A_test_rot, B_test_target_rot = get_rotated_features(
        model, test_data, test_indices, angle_rad=test_angle, device=device
    )
    
    # Predict
    B_pred_old_rot = predict_mm_features(A_test_rot, T_old)
    B_pred_new_rot = predict_mm_features(A_test_rot, T_new)
    
    # Error against "Ideal" B_test_target_rot
    rob_old = robustness_error(B_test_target_rot, B_pred_old_rot)
    rob_new = robustness_error(B_test_target_rot, B_pred_new_rot)
    
    print(f"Robustness MSE (30 deg): Old={rob_old:.6f}, New={rob_new:.6f}")
    
    # SSIM/PSNR on reconstruction
    metrics_old = compute_reconstruction_metrics(
        B_test_target_rot, B_pred_old_rot, image_shape=(28,28)
    )
    metrics_new = compute_reconstruction_metrics(
        B_test_target_rot, B_pred_new_rot, image_shape=(28,28)
    )
    
    print(f"SSIM: Old={metrics_old['ssim_mean']:.4f}, New={metrics_new['ssim_mean']:.4f}")
    print(f"PSNR: Old={metrics_old['psnr_mean']:.2f}, New={metrics_new['psnr_mean']:.2f}")

    # Save results
    results = {
        'train_mse_old': mse_old, 'train_mse_new': mse_new,
        'sym_old': sym_old, 'sym_new': sym_new,
        'rob_old': rob_old, 'rob_new': rob_new,
        'ssim_old': metrics_old['ssim_mean'], 'ssim_new': metrics_new['ssim_mean'],
        'n_samples': n_samples, 'lambda': lambda_
    }
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    run_mnist_experiment(n_samples=args.samples, epochs=args.epochs, device=device)
