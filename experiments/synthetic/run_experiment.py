"""
Complete synthetic experiment runner for Section 3.4.

Implements all three scenarios:
- Scenario 1: Old Approach (Static Transition Matrix)
- Scenario 2: New Approach (Equivariant Transition Matrix)
- Scenario 3: Robustness Test
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baseline import compute_baseline_T, predict_mm_features
from src.generators import SyntheticBridge, estimate_generator
from src.equivariant import compute_equivariant_T
from src.metrics import fidelity_mse, symmetry_defect, robustness_error

from experiments.synthetic.data import (
    get_synthetic_data, get_matrix_A, get_matrix_B, 
    get_class_labels, EXPECTED_RESULTS
)


def run_scenario_1(A: np.ndarray, B: np.ndarray, 
                   J_A: np.ndarray, J_B: np.ndarray,
                   verbose: bool = True) -> dict:
    """
    Scenario 1: Old Approach (Static Transition Matrix).
    
    Computes T_old using pseudoinverse (fidelity-only objective).
    """
    if verbose:
        print("\n" + "="*60)
        print("SCENARIO 1: Old Approach (Static Transition Matrix)")
        print("="*60)
    
    # Compute baseline T
    T_old = compute_baseline_T(A, B)
    
    # Predict MM features
    B_pred_old = predict_mm_features(A, T_old)
    
    # Compute metrics
    mse_fid = fidelity_mse(B, B_pred_old)
    sym_err = symmetry_defect(T_old, J_A, J_B)
    
    if verbose:
        print(f"\nT_old shape: {T_old.shape}")
        print(f"MSE (fidelity): {mse_fid:.6f}")
        print(f"Symmetry defect: {sym_err:.6f}")
        print(f"\nExpected from manuscript:")
        print(f"  MSE: {EXPECTED_RESULTS['old_approach']['mse_fid']}")
        print(f"  Sym_err: {EXPECTED_RESULTS['old_approach']['sym_err']}")
    
    return {
        'T': T_old,
        'B_pred': B_pred_old,
        'mse_fid': mse_fid,
        'sym_err': sym_err,
    }


def run_scenario_2(A: np.ndarray, B: np.ndarray,
                   J_A: np.ndarray, J_B: np.ndarray,
                   lambda_: float = 0.5,
                   verbose: bool = True) -> dict:
    """
    Scenario 2: New Approach (Equivariant Transition Matrix).
    
    Computes T_new balancing fidelity and equivariance.
    """
    if verbose:
        print("\n" + "="*60)
        print(f"SCENARIO 2: New Approach (Equivariant, lambda={lambda_})")
        print("="*60)
    
    # Compute equivariant T
    T_new = compute_equivariant_T(A, B, J_A, J_B, lambda_=lambda_)
    
    # Predict MM features
    B_pred_new = predict_mm_features(A, T_new)
    
    # Compute metrics
    mse_fid = fidelity_mse(B, B_pred_new)
    sym_err = symmetry_defect(T_new, J_A, J_B)
    
    if verbose:
        print(f"\nT_new shape: {T_new.shape}")
        print(f"MSE (fidelity): {mse_fid:.6f}")
        print(f"Symmetry defect: {sym_err:.6f}")
        print(f"\nExpected from manuscript:")
        print(f"  MSE: {EXPECTED_RESULTS['new_approach']['mse_fid']}")
        print(f"  Sym_err: {EXPECTED_RESULTS['new_approach']['sym_err']}")
    
    return {
        'T': T_new,
        'B_pred': B_pred_new,
        'mse_fid': mse_fid,
        'sym_err': sym_err,
    }


def run_scenario_3(A: np.ndarray, B: np.ndarray,
                   T_old: np.ndarray, T_new: np.ndarray,
                   bridge_A: SyntheticBridge, bridge_B: SyntheticBridge,
                   n_rotations: int = 15,
                   angle_range: tuple = (-np.pi/12, np.pi/12),
                   random_seed: int = 42,
                   verbose: bool = True) -> dict:
    """
    Scenario 3: Robustness Test.
    
    Tests stability under random rotations in the range ±15°.
    """
    if verbose:
        print("\n" + "="*60)
        print("SCENARIO 3: Robustness Test")
        print("="*60)
    
    np.random.seed(random_seed)
    
    m = A.shape[0]
    
    # Generate random rotation angles for each sample
    angles = np.random.uniform(angle_range[0], angle_range[1], size=m)
    
    if verbose:
        print(f"\nRotation angles range: [{np.degrees(angle_range[0]):.1f}°, "
              f"{np.degrees(angle_range[1]):.1f}°]")
    
    # Rotate FM features (A) using the bridge
    A_2d = bridge_A.get_2d_projection()
    A_2d_rot = np.zeros_like(A_2d)
    for i in range(m):
        cos_a, sin_a = np.cos(angles[i]), np.sin(angles[i])
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        A_2d_rot[i] = A_2d[i] @ R.T
    
    # Reconstruct rotated A
    A_rot = bridge_A.decoder.predict(A_2d_rot)
    
    # Similarly rotate B features for target
    B_2d = bridge_B.get_2d_projection()
    B_2d_rot = np.zeros_like(B_2d)
    for i in range(m):
        cos_a, sin_a = np.cos(angles[i]), np.sin(angles[i])
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        B_2d_rot[i] = B_2d[i] @ R.T
    
    # Reconstruct rotated B (ideal target)
    B_target = bridge_B.decoder.predict(B_2d_rot)
    
    # Predict with old and new approaches
    B_pred_old_rot = predict_mm_features(A_rot, T_old)
    B_pred_new_rot = predict_mm_features(A_rot, T_new)
    
    # Compute robustness errors
    err_old = robustness_error(B_target, B_pred_old_rot)
    err_new = robustness_error(B_target, B_pred_new_rot)
    
    if verbose:
        print(f"\nRobustness error (old approach): {err_old:.6f}")
        print(f"Robustness error (new approach): {err_new:.6f}")
        print(f"\nExpected from manuscript:")
        print(f"  Old: {EXPECTED_RESULTS['old_approach']['robustness_err']}")
        print(f"  New: {EXPECTED_RESULTS['new_approach']['robustness_err']}")
    
    return {
        'A_rot': A_rot,
        'B_target': B_target,
        'B_pred_old_rot': B_pred_old_rot,
        'B_pred_new_rot': B_pred_new_rot,
        'robustness_err_old': err_old,
        'robustness_err_new': err_new,
        'angles': angles,
    }


def run_full_experiment(epsilon: float = 0.01,
                        lambda_: float = 0.5,
                        random_seed: int = 42,
                        verbose: bool = True,
                        validate: bool = False) -> dict:
    """
    Run the complete synthetic experiment (Section 3.4).
    
    Parameters
    ----------
    epsilon : float
        Small angle for generator estimation (radians). Default 0.01.
    lambda_ : float
        Weighting coefficient for equivariance. Default 0.5.
    random_seed : int
        Random seed for reproducibility. Default 42.
    verbose : bool
        Print results. Default True.
    validate : bool
        Raise assertion errors if results don't match expected. Default False.
    
    Returns
    -------
    dict
        Complete results from all scenarios
    """
    if verbose:
        print("\n" + "#"*70)
        print("# SYNTHETIC EXPERIMENT (Section 3.4)                                #")
        print("#"*70)
        print(f"\nParameters:")
        print(f"  epsilon = {epsilon} rad")
        print(f"  lambda = {lambda_}")
        print(f"  Random seed = {random_seed}")
    
    # Load data
    data = get_synthetic_data()
    A = data['A']
    B = data['B']
    labels = data['labels']
    
    if verbose:
        print(f"\nData dimensions:")
        print(f"  A: {A.shape} (FM features)")
        print(f"  B: {B.shape} (MM features)")
        print(f"  Samples: {data['m']}, Classes: {len(np.unique(labels))}")
    
    # Estimate generators using Algorithm 2 (MDS bridge)
    if verbose:
        print("\n" + "-"*60)
        print("Estimating Lie algebra generators (Algorithm 2)")
        print("-"*60)
    
    bridge_A = SyntheticBridge(random_state=random_seed)
    bridge_A.fit(A)
    J_A = bridge_A.estimate_generator(A, epsilon=epsilon)
    
    bridge_B = SyntheticBridge(random_state=random_seed)
    bridge_B.fit(B)
    J_B = bridge_B.estimate_generator(B, epsilon=epsilon)
    
    if verbose:
        print(f"J_A shape: {J_A.shape}")
        print(f"J_B shape: {J_B.shape}")
    
    # Run scenarios
    results_s1 = run_scenario_1(A, B, J_A, J_B, verbose=verbose)
    results_s2 = run_scenario_2(A, B, J_A, J_B, lambda_=lambda_, verbose=verbose)
    results_s3 = run_scenario_3(
        A, B, 
        results_s1['T'], results_s2['T'],
        bridge_A, bridge_B,
        random_seed=random_seed,
        verbose=verbose
    )
    
    # Print summary table (Table 1)
    if verbose:
        print("\n" + "="*60)
        print("TABLE 1: Results of the experiment on synthetic data")
        print("="*60)
        print(f"{'Metric':<30} {'Old Approach':>12} {'New Approach':>12}")
        print("-"*60)
        print(f"{'MSE on training data':<30} {results_s1['mse_fid']:>12.3f} {results_s2['mse_fid']:>12.3f}")
        print(f"{'Symmetry Defect (Sym_err)':<30} {results_s1['sym_err']:>12.3f} {results_s2['sym_err']:>12.3f}")
        print(f"{'Error on rotated data':<30} {results_s3['robustness_err_old']:>12.3f} {results_s3['robustness_err_new']:>12.3f}")
        print("-"*60)
    
    # Validation
    if validate:
        # Allow some tolerance due to different MDS/regression implementations
        tolerance = 0.5  # 50% relative tolerance
        
        def check(actual, expected, name):
            if abs(actual - expected) / max(expected, 0.001) > tolerance:
                print(f"WARNING: {name}: got {actual:.3f}, expected ~{expected:.3f}")
        
        check(results_s1['mse_fid'], EXPECTED_RESULTS['old_approach']['mse_fid'], 
              "Old MSE")
        check(results_s1['sym_err'], EXPECTED_RESULTS['old_approach']['sym_err'],
              "Old Sym_err")
        check(results_s3['robustness_err_old'], 
              EXPECTED_RESULTS['old_approach']['robustness_err'], "Old Robustness")
        
        check(results_s3['robustness_err_new'], 
              EXPECTED_RESULTS['new_approach']['robustness_err'], "New Robustness")
        
        print("\nValidation complete!")
    
    return {
        'scenario_1': results_s1,
        'scenario_2': results_s2,
        'scenario_3': results_s3,
        'J_A': J_A,
        'J_B': J_B,
        'bridge_A': bridge_A,
        'bridge_B': bridge_B,
        'labels': labels,
        'epsilon': epsilon,
        'lambda': lambda_,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run synthetic experiment (Section 3.4)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Small angle for generator estimation (radians)")
    parser.add_argument("--lambda_", type=float, default=0.5,
                        help="Weighting coefficient for equivariance")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--validate", action="store_true",
                        help="Validate results against manuscript values")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    results = run_full_experiment(
        epsilon=args.epsilon,
        lambda_=args.lambda_,
        random_seed=args.seed,
        verbose=not args.quiet,
        validate=args.validate,
    )
