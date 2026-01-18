"""
Figure generation script.

Generates all required figures from the manuscript:
- Fig. 2: Robustness test scatter plots (Editor's requirement)
- Fig. 3: λ trade-off curve
- Additional visualizations for the report
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_robustness_scatter(B_pred_old_rot: np.ndarray,
                            B_pred_new_rot: np.ndarray,
                            labels: np.ndarray,
                            save_path: str = None,
                            title: str = "Robustness Test: Old vs New Approach") -> plt.Figure:
    """
    Generate the Editor's requested scatter plot (Section 3.4.4).
    
    Two side-by-side plots:
    - Left: B*_old_rot (chaotic, class structure collapses)
    - Right: B*_new_rot (preserves clustered structure)
    
    Uses MDS to reduce to 2D for visualization.
    """
    from sklearn.manifold import MDS
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color map for 3 classes
    colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green
    cmap = ListedColormap(colors)
    
    # MDS reduction to 2D for visualization
    mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
    
    # Left plot: Old approach (chaotic)
    B_old_2d = mds.fit_transform(B_pred_old_rot)
    ax1 = axes[0]
    scatter1 = ax1.scatter(B_old_2d[:, 0], B_old_2d[:, 1], 
                           c=labels, cmap=cmap, s=100, edgecolors='black', 
                           linewidth=1.5, alpha=0.8)
    ax1.set_title("Old Approach: $B^*_{old\\_rot}$\n(Chaotic - Class Structure Lost)", 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("MDS Dimension 1")
    ax1.set_ylabel("MDS Dimension 2")
    ax1.grid(True, alpha=0.3)
    
    # Right plot: New approach (structured)
    B_new_2d = mds.fit_transform(B_pred_new_rot)
    ax2 = axes[1]
    scatter2 = ax2.scatter(B_new_2d[:, 0], B_new_2d[:, 1], 
                           c=labels, cmap=cmap, s=100, edgecolors='black',
                           linewidth=1.5, alpha=0.8)
    ax2.set_title("New Approach: $B^*_{new\\_rot}$\n(Structured - Class Clusters Preserved)", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("MDS Dimension 1")
    ax2.set_ylabel("MDS Dimension 2")
    ax2.grid(True, alpha=0.3)
    
    # Legend
    legend_labels = [f'Class {i}' for i in range(len(colors))]
    legend_handles = [plt.scatter([], [], c=colors[i], s=100, edgecolors='black') 
                      for i in range(len(colors))]
    fig.legend(legend_handles, legend_labels, loc='upper center', 
               ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_lambda_tradeoff(A: np.ndarray, B: np.ndarray,
                         J_A: np.ndarray, J_B: np.ndarray,
                         lambda_range: np.ndarray = None,
                         save_path: str = None) -> plt.Figure:
    """
    Generate λ trade-off curve (fidelity vs symmetry defect).
    """
    from src.equivariant import compute_equivariant_T
    from src.metrics import fidelity_mse, symmetry_defect
    from src.baseline import predict_mm_features
    
    if lambda_range is None:
        lambda_range = np.logspace(-3, 2, 20)
    
    mse_values = []
    sym_values = []
    
    for lam in lambda_range:
        T = compute_equivariant_T(A, B, J_A, J_B, lambda_=lam)
        B_pred = predict_mm_features(A, T)
        
        mse = fidelity_mse(B, B_pred)
        sym = symmetry_defect(T, J_A, J_B)
        
        mse_values.append(mse)
        sym_values.append(sym)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#1f77b4'
    ax1.set_xlabel('λ (Weighting Coefficient)', fontsize=12)
    ax1.set_ylabel('Fidelity Error (MSE)', color=color1, fontsize=12)
    ax1.semilogx(lambda_range, mse_values, 'o-', color=color1, 
                 linewidth=2, markersize=8, label='Fidelity (MSE)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Symmetry Defect', color=color2, fontsize=12)
    ax2.semilogx(lambda_range, sym_values, 's--', color=color2,
                 linewidth=2, markersize=8, label='Symmetry Defect')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Mark λ = 0.5 (manuscript default)
    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(0.5, ax1.get_ylim()[1] * 0.95, 'λ=0.5', fontsize=10, 
             ha='center', color='gray')
    
    plt.title('Trade-off: Fidelity vs Equivariance\n(Effect of λ)', 
              fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_mds_visualization(A: np.ndarray, B: np.ndarray, 
                           labels: np.ndarray,
                           save_path: str = None) -> plt.Figure:
    """
    Generate MDS visualization of A and B matrices (Fig. 1 from manuscript).
    """
    from sklearn.manifold import MDS
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    cmap = ListedColormap(colors)
    
    mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
    
    # Plot A
    A_2d = mds.fit_transform(A)
    axes[0].scatter(A_2d[:, 0], A_2d[:, 1], c=labels, cmap=cmap, 
                    s=150, edgecolors='black', linewidth=1.5)
    axes[0].set_title(f'(a) Matrix $A \\in \\mathbb{{R}}^{{{A.shape[0]} \\times {A.shape[1]}}}$\n(FM Features)', 
                      fontsize=11)
    axes[0].set_xlabel('MDS Dimension 1')
    axes[0].set_ylabel('MDS Dimension 2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot B
    B_2d = mds.fit_transform(B)
    axes[1].scatter(B_2d[:, 0], B_2d[:, 1], c=labels, cmap=cmap,
                    s=150, edgecolors='black', linewidth=1.5)
    axes[1].set_title(f'(b) Matrix $B \\in \\mathbb{{R}}^{{{B.shape[0]} \\times {B.shape[1]}}}$\n(MM Features)',
                      fontsize=11)
    axes[1].set_xlabel('MDS Dimension 1')
    axes[1].set_ylabel('MDS Dimension 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('MDS Visualization of Feature Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def generate_all_synthetic_figures(output_dir: str = "figures") -> dict:
    """
    Generate all figures for the synthetic experiment.
    
    Returns paths to generated figures.
    """
    from experiments.synthetic.run_experiment import run_full_experiment
    from experiments.synthetic.data import get_matrix_A, get_matrix_B, get_class_labels
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Running synthetic experiment...")
    results = run_full_experiment(verbose=False)
    
    A = get_matrix_A()
    B = get_matrix_B()
    labels = get_class_labels()
    
    figures = {}
    
    # Fig 1: MDS visualization
    print("Generating Fig 1: MDS visualization...")
    fig1 = plot_mds_visualization(A, B, labels, 
                                   save_path=str(output_path / "fig1_mds_visualization.png"))
    figures['fig1'] = fig1
    
    # Fig 2: Robustness scatter (Editor's requirement)
    print("Generating Fig 2: Robustness scatter plots...")
    fig2 = plot_robustness_scatter(
        results['scenario_3']['B_pred_old_rot'],
        results['scenario_3']['B_pred_new_rot'],
        labels,
        save_path=str(output_path / "fig2_robustness_scatter.png")
    )
    figures['fig2'] = fig2
    
    # Fig 3: Lambda trade-off
    print("Generating Fig 3: Lambda trade-off curve...")
    fig3 = plot_lambda_tradeoff(A, B, results['J_A'], results['J_B'],
                                 save_path=str(output_path / "fig3_lambda_tradeoff.png"))
    figures['fig3'] = fig3
    
    print(f"\nAll figures saved to: {output_path.absolute()}")
    
    return figures


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment figures")
    parser.add_argument("--output", "-o", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively")
    
    args = parser.parse_args()
    
    figures = generate_all_synthetic_figures(output_dir=args.output)
    
    if args.show:
        plt.show()
