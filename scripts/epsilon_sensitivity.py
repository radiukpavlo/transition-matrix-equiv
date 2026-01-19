"""
Epsilon sensitivity analysis for generator estimation.

Tests the stability of generator estimation across different epsilon values.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators import SyntheticBridge
from src.baseline import compute_baseline_T, predict_mm_features
from src.equivariant import compute_equivariant_T
from src.metrics import fidelity_mse, symmetry_defect
from experiments.synthetic.data import get_matrix_A, get_matrix_B


def run_epsilon_sensitivity(epsilons=None, lambda_=0.5, seed=42):
    """Run epsilon sensitivity analysis."""
    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    A = get_matrix_A()
    B = get_matrix_B()
    
    results = []
    
    for eps in epsilons:
        print(f"\nEpsilon = {eps}")
        
        # Estimate generators
        bridge_A = SyntheticBridge(random_state=seed)
        J_A = bridge_A.estimate_generator(A, epsilon=eps)
        
        bridge_B = SyntheticBridge(random_state=seed)
        J_B = bridge_B.estimate_generator(B, epsilon=eps)
        
        # Compute equivariant T
        T_new = compute_equivariant_T(A, B, J_A, J_B, lambda_=lambda_)
        T_old = compute_baseline_T(A, B)
        
        # Metrics
        B_pred_new = predict_mm_features(A, T_new)
        B_pred_old = predict_mm_features(A, T_old)
        
        mse_new = fidelity_mse(B, B_pred_new)
        sym_new = symmetry_defect(T_new, J_A, J_B)
        sym_old = symmetry_defect(T_old, J_A, J_B)
        
        # Generator norm (stability check)
        j_a_norm = np.linalg.norm(J_A, 'fro')
        j_b_norm = np.linalg.norm(J_B, 'fro')
        
        results.append({
            'epsilon': eps,
            'mse_new': mse_new,
            'sym_new': sym_new,
            'sym_old': sym_old,
            'j_a_norm': j_a_norm,
            'j_b_norm': j_b_norm,
        })
        
        print(f"  MSE_new: {mse_new:.6f}")
        print(f"  Sym_new: {sym_new:.6f}")
        print(f"  ||J_A||: {j_a_norm:.4f}, ||J_B||: {j_b_norm:.4f}")
    
    return results


def plot_epsilon_sensitivity(results, save_path=None):
    """Plot epsilon sensitivity results."""
    epsilons = [r['epsilon'] for r in results]
    sym_new = [r['sym_new'] for r in results]
    sym_old = [r['sym_old'] for r in results]
    mse_new = [r['mse_new'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Symmetry defect vs epsilon
    ax1.loglog(epsilons, sym_old, 'o--', label='Old Approach', color='#d62728', linewidth=2, markersize=8)
    ax1.loglog(epsilons, sym_new, 's-', label='New Approach', color='#2ca02c', linewidth=2, markersize=8)
    ax1.set_xlabel('ε (radians)', fontsize=12)
    ax1.set_ylabel('Symmetry Defect', fontsize=12)
    ax1.set_title('Symmetry Defect vs ε', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(0.01, ax1.get_ylim()[1]*0.8, 'ε=0.01\n(default)', fontsize=9, ha='center', color='gray')
    
    # Plot 2: MSE vs epsilon
    ax2.semilogx(epsilons, mse_new, 's-', color='#1f77b4', linewidth=2, markersize=8)
    ax2.set_xlabel('ε (radians)', fontsize=12)
    ax2.set_ylabel('Fidelity MSE', fontsize=12)
    ax2.set_title('Fidelity MSE vs ε (New Approach)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.suptitle('Epsilon Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Running Epsilon Sensitivity Analysis...")
    results = run_epsilon_sensitivity()
    
    # Print summary table
    print("\n" + "="*70)
    print("EPSILON SENSITIVITY RESULTS")
    print("="*70)
    print(f"{'Epsilon':<10} {'MSE_new':<12} {'Sym_new':<15} {'Sym_old':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['epsilon']:<10.3f} {r['mse_new']:<12.6f} {r['sym_new']:<15.6f} {r['sym_old']:<15.2f}")
    
    # Generate figure
    fig = plot_epsilon_sensitivity(results, save_path="figures/fig4_epsilon_sensitivity.png")
    print("\nAnalysis complete!")
