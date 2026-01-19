#!/usr/bin/env python3
"""
Step 4: Synthetic Experiment - Robustness Test (CORRECTED VERSION)

This corrected version addresses the scientific interpretation issues identified
by the Science Methodology Review Agent:
1. Honest interpretation: Baseline outperforms equivariant method on this synthetic task
2. Ground truth visualization: Shows that the "chaotic" baseline matches the target
3. Scientific context: Explains why baseline works better (linear data generation)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("STEP 4: SYNTHETIC EXPERIMENT - ROBUSTNESS TEST (CORRECTED)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading synthetic data and computed matrices...")

# Load original data (A, B)
data_path_1 = Path(__file__).parent / "data" / "synthetic.npz"
data_1 = np.load(data_path_1)
A = data_1['A']  # Original features (15 samples × 5 dimensions)
B = data_1['B']  # Target features (15 samples × 4 dimensions)

# Load computed matrices (T_old, T_new)
data_path_2 = Path(__file__).parent / "data" / "synthetic_generators.npz"
data_2 = np.load(data_path_2)
T_old = data_2['T_old']  # Baseline transfer matrix (4 × 5)
T_new = data_2['T_new']  # Equivariant transfer matrix (4 × 5)

n_samples, d_A = A.shape
_, d_B = B.shape

print(f"  A shape: {A.shape}")
print(f"  B shape: {B.shape}")
print(f"  T_old shape: {T_old.shape}")
print(f"  T_new shape: {T_new.shape}")

# ============================================================================
# 2. RE-IMPLEMENT THE BRIDGE
# ============================================================================
print("\n[2/6] Re-implementing the Bridge (MDS + Linear Decoders)...")

# Fit MDS on A to obtain 2D latent points Z
print("  Fitting MDS on A to obtain 2D latent space...")
mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
Z = mds.fit_transform(A)
print(f"  Z shape: {Z.shape}")
print(f"  MDS stress: {mds.stress_:.4f}")

# Train Linear Regression decoder: Z → A
print("  Training decoder Z → A...")
decoder_A = LinearRegression()
decoder_A.fit(Z, A)
A_reconstructed = decoder_A.predict(Z)
reconstruction_error_A = np.mean((A - A_reconstructed) ** 2)
print(f"  Decoder A reconstruction MSE: {reconstruction_error_A:.6f}")

# Train Linear Regression decoder: Z → B (for ground truth generation)
print("  Training decoder Z → B...")
decoder_B = LinearRegression()
decoder_B.fit(Z, B)
B_reconstructed = decoder_B.predict(Z)
reconstruction_error_B = np.mean((B - B_reconstructed) ** 2)
print(f"  Decoder B reconstruction MSE: {reconstruction_error_B:.6f}")

# ============================================================================
# 3. GENERATE ROTATED DATA
# ============================================================================
print("\n[3/6] Generating rotated data...")

# Apply random rotations to each sample in Z
n_rotations = n_samples  # One rotation per sample
rotations = []
Z_rotated_list = []
A_rot_list = []
B_rot_list = []

print(f"  Applying {n_rotations} random rotations (alpha in [-15°, 15°])...")
for i in range(n_rotations):
    # Random rotation angle in radians
    alpha = np.random.uniform(-15, 15) * np.pi / 180  # Convert to radians

    # 2D rotation matrix
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])

    # Rotate the latent point
    z_original = Z[i:i+1, :]  # Keep as 2D array (1 × 2)
    z_rotated = z_original @ R.T

    # Decode rotated latent point back to feature spaces
    a_rot = decoder_A.predict(z_rotated)
    b_rot = decoder_B.predict(z_rotated)

    rotations.append(alpha * 180 / np.pi)  # Store in degrees
    Z_rotated_list.append(z_rotated[0])
    A_rot_list.append(a_rot[0])
    B_rot_list.append(b_rot[0])

    if (i + 1) % 5 == 0:
        print(f"    Progress: {i + 1}/{n_rotations} rotations applied")

A_rot = np.array(A_rot_list)  # Shape: (15, 5)
B_rot = np.array(B_rot_list)  # Shape: (15, 4) - Ground truth for rotated samples

print(f"  A_rot shape: {A_rot.shape}")
print(f"  B_rot shape: {B_rot.shape}")
print(f"  Rotation angles (degrees): min={min(rotations):.2f}, max={max(rotations):.2f}")

# ============================================================================
# 4. COMPUTE PREDICTIONS
# ============================================================================
print("\n[4/6] Computing predictions with both methods...")

# Baseline approach: B*_old = A_rot @ T_old^T
B_pred_old = A_rot @ T_old.T
print(f"  B_pred_old shape: {B_pred_old.shape}")

# Equivariant approach: B*_new = A_rot @ T_new^T
B_pred_new = A_rot @ T_new.T
print(f"  B_pred_new shape: {B_pred_new.shape}")

# ============================================================================
# 5. CALCULATE METRICS
# ============================================================================
print("\n[5/6] Calculating metrics...")

# MSE comparing predictions to ground truth B_rot
mse_old = np.mean((B_pred_old - B_rot) ** 2)
mse_new = np.mean((B_pred_new - B_rot) ** 2)

# Per-sample MSE for detailed analysis
mse_old_per_sample = np.mean((B_pred_old - B_rot) ** 2, axis=1)
mse_new_per_sample = np.mean((B_pred_new - B_rot) ** 2, axis=1)

print(f"  MSE (Baseline/Old): {mse_old:.6f}")
print(f"  MSE (Equivariant/New): {mse_new:.6f}")
print(f"  Difference: {(mse_new - mse_old):.6f} (positive = new is worse)")
print(f"  Ratio (New/Old): {mse_new / mse_old:.2f}x")

# ============================================================================
# 6. GENERATE VISUALIZATION WITH GROUND TRUTH
# ============================================================================
print("\n[6/6] Generating visualization with ground truth...")

# Use PCA to project 4D B* vectors to 2D for plotting
pca = PCA(n_components=2, random_state=42)

# Fit PCA on combined data to ensure same projection space
B_combined = np.vstack([B_rot, B_pred_old, B_pred_new])
pca.fit(B_combined)

# Project all data to 2D
B_rot_2d = pca.transform(B_rot)
B_pred_old_2d = pca.transform(B_pred_old)
B_pred_new_2d = pca.transform(B_pred_new)

print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")

# Define class labels (0-4: Class 1, 5-9: Class 2, 10-14: Class 3)
class_labels = np.array([0]*5 + [1]*5 + [2]*5)
class_names = ['Class 1', 'Class 2', 'Class 3']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Create figure with THREE subplots (Ground Truth, Baseline, Equivariant)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Ground Truth (B_rot)
ax = axes[0]
for class_id in range(3):
    mask = class_labels == class_id
    ax.scatter(
        B_rot_2d[mask, 0],
        B_rot_2d[mask, 1],
        c=colors[class_id],
        label=class_names[class_id],
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=1.5
    )
ax.set_title('Ground Truth (B_rot)\nIdeal Target Pattern', fontsize=13, fontweight='bold')
ax.set_xlabel('PCA Component 1', fontsize=10)
ax.set_ylabel('PCA Component 2', fontsize=10)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, 'Target Pattern',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
        fontsize=9)

# Subplot 2: Baseline (Old Approach)
ax = axes[1]
for class_id in range(3):
    mask = class_labels == class_id
    ax.scatter(
        B_pred_old_2d[mask, 0],
        B_pred_old_2d[mask, 1],
        c=colors[class_id],
        label=class_names[class_id],
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=1.5
    )
ax.set_title('Baseline (Unconstrained Linear)\nMatches Ground Truth', fontsize=13, fontweight='bold')
ax.set_xlabel('PCA Component 1', fontsize=10)
ax.set_ylabel('PCA Component 2', fontsize=10)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f'MSE: {mse_old:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
        fontsize=9)

# Subplot 3: Equivariant (New Approach)
ax = axes[2]
for class_id in range(3):
    mask = class_labels == class_id
    ax.scatter(
        B_pred_new_2d[mask, 0],
        B_pred_new_2d[mask, 1],
        c=colors[class_id],
        label=class_names[class_id],
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=1.5
    )
ax.set_title('Equivariant (Rotation-Constrained)\nFails on Linear Data', fontsize=13, fontweight='bold')
ax.set_xlabel('PCA Component 1', fontsize=10)
ax.set_ylabel('PCA Component 2', fontsize=10)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f'MSE: {mse_new:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
        fontsize=9)

plt.suptitle('Robustness Test: Baseline Outperforms Equivariant on Linearly-Generated Synthetic Data',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
figures_dir = Path(__file__).parent.parent / "figures"
figures_dir.mkdir(exist_ok=True)
output_path = figures_dir / "synthetic_robustness_corrected.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  Figure saved to: {output_path}")
plt.close()

# ============================================================================
# 7. SAVE CORRECTED METRICS
# ============================================================================
print("\nSaving corrected metrics to JSON...")

results = {
    "experiment": "synthetic_robustness_test_corrected",
    "description": "Robustness test on rotated synthetic data - CORRECTED INTERPRETATION",
    "data": {
        "n_samples": int(n_samples),
        "d_A": int(d_A),
        "d_B": int(d_B),
        "n_rotations": int(n_rotations),
        "rotation_range_degrees": [-15, 15]
    },
    "bridge": {
        "mds_stress": float(mds.stress_),
        "decoder_A_mse": float(reconstruction_error_A),
        "decoder_B_mse": float(reconstruction_error_B)
    },
    "metrics": {
        "mse_baseline": float(mse_old),
        "mse_equivariant": float(mse_new),
        "error_increase": float(mse_new - mse_old),
        "error_ratio": float(mse_new / mse_old)
    },
    "per_sample_mse": {
        "baseline": mse_old_per_sample.tolist(),
        "equivariant": mse_new_per_sample.tolist()
    },
    "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    "scientific_interpretation": {
        "result": "Baseline OUTPERFORMS equivariant method on this synthetic task",
        "evidence": f"Baseline MSE = {mse_old:.4f}, Equivariant MSE = {mse_new:.4f} ({mse_new/mse_old:.1f}x worse)",
        "explanation": "The synthetic data generation uses LINEAR decoders from MDS coordinates, creating a perfect linear relationship B_rot ≈ A_rot * W. The baseline (unconstrained linear regression) easily captures this. The equivariant method imposes rotation symmetry constraints that do NOT exist in the linearly-generated data, causing it to fail.",
        "visualization": "Ground truth added to show baseline matches target pattern, while equivariant does not",
        "conclusion": "This result validates that the experiment setup itself favors unconstrained linear models. For real-world data with actual rotational structure, equivariant method may perform better."
    },
    "limitations": [
        "Synthetic data generation is inherently linear, which favors baseline",
        "Ground truth B_rot generated via same linear decoder mechanism",
        "Does not test actual rotational equivariance in natural data",
        "Results are specific to this synthetic setup and may not generalize"
    ]
}

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)
results_path = results_dir / "synthetic_experiment_corrected.json"

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to: {results_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE (CORRECTED INTERPRETATION)")
print("=" * 80)
print(f"✓ MSE (Baseline): {mse_old:.6f} ← BETTER (lower error)")
print(f"✓ MSE (Equivariant): {mse_new:.6f} ← WORSE (higher error)")
print(f"✓ Error Ratio: {mse_new / mse_old:.1f}x (equivariant is ~{mse_new / mse_old:.0f}x worse)")
print(f"\n✓ SCIENTIFIC INTERPRETATION:")
print(f"  - Baseline outperforms equivariant on this linearly-generated synthetic data")
print(f"  - This is expected: linear data generation favors unconstrained linear models")
print(f"  - Equivariant constraints don't match the linear generation process")
print(f"  - Visualization shows baseline matches ground truth, equivariant does not")
print(f"\n✓ Figure (3 panels): {output_path}")
print(f"✓ Corrected Metrics: {results_path}")
print("=" * 80)
