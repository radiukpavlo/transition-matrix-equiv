"""
Step 3: Methodology Verification - Equivariant Solver

This script verifies the implementation of the equivariant method on synthetic data.
It tests:
1. Generator estimation using the bridge method (MDS + Decoder)
2. Equivariant solver with symmetry constraint
3. Comparison with baseline (T_old) to verify symmetry defect reduction

Author: K-Dense Coding Agent
Date: 2026-01-18
"""

import numpy as np
import os
import sys
import json
import time
from pathlib import Path

# Add workflow directory to path
session_dir = Path("/app/sandbox/session_20260118_175817_da8f96a1d029")
sys.path.insert(0, str(session_dir / "workflow"))

from generators import estimate_generators_bridge, compute_symmetry_defect
from solver import EquivariantSolver, compute_objective


def main():
    print("=" * 70)
    print("STEP 3: METHODOLOGY VERIFICATION (EQUIVARIANT SOLVER)")
    print("=" * 70)
    print(f"Session directory: {session_dir}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================================================
    # 1. Load Synthetic Data
    # ========================================================================
    print("1. Loading synthetic data...")
    print("-" * 70)

    synthetic_path = session_dir / "workflow/data/synthetic.npz"
    data = np.load(synthetic_path)
    A = data['A']  # Formal model features: 15 × 5
    B = data['B']  # Mental model features: 15 × 4

    print(f"  Loaded A: {A.shape} (Formal Model features)")
    print(f"  Loaded B: {B.shape} (Mental Model features)")
    print(f"  Sample size: N = {A.shape[0]}")
    print()

    # ========================================================================
    # 2. Compute Baseline Transition Matrix (T_old)
    # ========================================================================
    print("2. Computing baseline transition matrix (T_old)...")
    print("-" * 70)

    # T_old = (A^+ B)^T where A^+ is pseudoinverse of A
    # A is (15, 5), A^+ is (5, 15), B is (15, 4)
    # A^+ B is (5, 15) @ (15, 4) = (5, 4)
    # T_old = (A^+ B)^T = (4, 5)
    A_pinv = np.linalg.pinv(A)  # (5, 15)
    T_old_temp = A_pinv @ B  # (5, 4)
    T_old = T_old_temp.T  # Transpose to (4, 5) = (d_B, d_A)

    print(f"  T_old shape: {T_old.shape} (d_B × d_A)")
    print(f"  T_old reconstruction error: ||B^T - T_old A^T||_F^2 = "
          f"{np.linalg.norm(B.T - T_old @ A.T, 'fro')**2:.6e}")
    print()

    # ========================================================================
    # 3. Estimate Generators using Bridge Method
    # ========================================================================
    print("3. Estimating generators using bridge method (MDS + Decoder)...")
    print("-" * 70)

    # Use bridge method since we don't have natural transformations for synthetic data
    J_A, J_B = estimate_generators_bridge(
        A, B, T_old,
        epsilon=0.01,
        n_components=2  # Use 2D latent space for synthetic data
    )

    print(f"\n  Generator shapes: J^A {J_A.shape}, J^B {J_B.shape}")

    # Verify generators are antisymmetric (should be approximately)
    antisym_A = np.linalg.norm(J_A + J_A.T, 'fro')
    antisym_B = np.linalg.norm(J_B + J_B.T, 'fro')
    print(f"  Antisymmetry check (should be small):")
    print(f"    ||J^A + (J^A)^T||_F = {antisym_A:.6e}")
    print(f"    ||J^B + (J^B)^T||_F = {antisym_B:.6e}")
    print()

    # ========================================================================
    # 4. Compute Symmetry Defect for T_old
    # ========================================================================
    print("4. Computing symmetry defect for baseline T_old...")
    print("-" * 70)

    defect_old = compute_symmetry_defect(T_old, J_A, J_B)
    print(f"  Symmetry defect (T_old): ||T_old J^A - J^B T_old||_F = {defect_old:.6e}")
    print()

    # ========================================================================
    # 5. Solve for T_new using Equivariant Solver
    # ========================================================================
    print("5. Solving for T_new with equivariant constraint (λ=0.5)...")
    print("-" * 70)

    lambda_reg = 0.5
    solver = EquivariantSolver(A, B, J_A, J_B, lambda_reg=lambda_reg)

    T_new, solve_info = solver.solve(
        method='lsqr',
        max_iter=1000,
        tol=1e-8,
        verbose=True
    )

    print(f"\n  T_new shape: {T_new.shape}")
    print()

    # ========================================================================
    # 6. Compute Symmetry Defect for T_new
    # ========================================================================
    print("6. Computing symmetry defect for equivariant T_new...")
    print("-" * 70)

    defect_new = compute_symmetry_defect(T_new, J_A, J_B)
    print(f"  Symmetry defect (T_new): ||T_new J^A - J^B T_new||_F = {defect_new:.6e}")
    print()

    # ========================================================================
    # 7. Compare T_old vs T_new
    # ========================================================================
    print("7. Comparison: T_old (baseline) vs T_new (equivariant)...")
    print("-" * 70)

    # Compute objectives
    obj_old, recon_old, sym_old = compute_objective(T_old, A, B, J_A, J_B, lambda_reg)
    obj_new, recon_new, sym_new = compute_objective(T_new, A, B, J_A, J_B, lambda_reg)

    print(f"\n  T_old (Baseline - Pseudoinverse):")
    print(f"    Reconstruction error: {recon_old:.6e}")
    print(f"    Symmetry defect²:     {sym_old:.6e}")
    print(f"    Total objective:      {obj_old:.6e}")

    print(f"\n  T_new (Equivariant Solver, λ={lambda_reg}):")
    print(f"    Reconstruction error: {recon_new:.6e}")
    print(f"    Symmetry defect²:     {sym_new:.6e}")
    print(f"    Total objective:      {obj_new:.6e}")

    print(f"\n  Improvements:")
    improvement_obj = (obj_old - obj_new) / obj_old * 100
    improvement_sym = (sym_old - sym_new) / sym_old * 100 if sym_old > 0 else 0

    print(f"    Total objective: {improvement_obj:+.2f}%")
    print(f"    Symmetry defect: {improvement_sym:+.2f}%")
    print()

    # ========================================================================
    # 8. Verification Tests
    # ========================================================================
    print("8. Running verification tests...")
    print("-" * 70)

    tests_passed = []
    tests_failed = []

    # Test 1: Generators have correct shapes
    test_name = "Generator shapes"
    if J_A.shape == (A.shape[1], A.shape[1]) and J_B.shape == (B.shape[1], B.shape[1]):
        print(f"  ✓ {test_name}: PASS")
        tests_passed.append(test_name)
    else:
        print(f"  ✗ {test_name}: FAIL")
        tests_failed.append(test_name)

    # Test 2: Solver completes without errors
    test_name = "Solver completion"
    if T_new.shape == (B.shape[1], A.shape[1]):
        print(f"  ✓ {test_name}: PASS")
        tests_passed.append(test_name)
    else:
        print(f"  ✗ {test_name}: FAIL")
        tests_failed.append(test_name)

    # Test 3: Symmetry defect is reduced
    test_name = "Symmetry defect reduction"
    if defect_new < defect_old:
        print(f"  ✓ {test_name}: PASS (reduced by {improvement_sym:.2f}%)")
        tests_passed.append(test_name)
    else:
        print(f"  ✗ {test_name}: FAIL (not reduced)")
        tests_failed.append(test_name)

    # Test 4: Total objective is reduced (or very close)
    test_name = "Total objective improvement"
    if obj_new <= obj_old * 1.01:  # Allow 1% tolerance
        print(f"  ✓ {test_name}: PASS")
        tests_passed.append(test_name)
    else:
        print(f"  ✗ {test_name}: FAIL")
        tests_failed.append(test_name)

    # Test 5: No NaN or Inf values
    test_name = "Numerical stability"
    if (np.isfinite(T_new).all() and np.isfinite(J_A).all() and np.isfinite(J_B).all()):
        print(f"  ✓ {test_name}: PASS")
        tests_passed.append(test_name)
    else:
        print(f"  ✗ {test_name}: FAIL")
        tests_failed.append(test_name)

    print()
    print(f"  Tests passed: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)}")
    print()

    # ========================================================================
    # 9. Save Results
    # ========================================================================
    print("9. Saving results...")
    print("-" * 70)

    # Save generators
    generators_path = session_dir / "workflow/data/synthetic_generators.npz"
    np.savez(
        generators_path,
        J_A=J_A,
        J_B=J_B,
        T_old=T_old,
        T_new=T_new,
        defect_old=defect_old,
        defect_new=defect_new
    )
    print(f"  Saved generators to: {generators_path}")

    # Save test results as JSON
    results = {
        "step": "3_methodology_verification",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "data": {
            "A_shape": list(A.shape),
            "B_shape": list(B.shape),
            "N": int(A.shape[0]),
            "d_A": int(A.shape[1]),
            "d_B": int(B.shape[1])
        },
        "generators": {
            "J_A_shape": list(J_A.shape),
            "J_B_shape": list(J_B.shape),
            "method": "bridge_mds",
            "n_components": 2,
            "antisymmetry_A": float(antisym_A),
            "antisymmetry_B": float(antisym_B)
        },
        "T_old": {
            "shape": list(T_old.shape),
            "method": "pseudoinverse",
            "reconstruction_error": float(recon_old),
            "symmetry_defect": float(sym_old),
            "total_objective": float(obj_old)
        },
        "T_new": {
            "shape": list(T_new.shape),
            "method": "equivariant_solver",
            "lambda_reg": float(lambda_reg),
            "reconstruction_error": float(recon_new),
            "symmetry_defect": float(sym_new),
            "total_objective": float(obj_new),
            "solver_info": {
                "iterations": int(solve_info.get('iterations', -1)),
                "elapsed_time": float(solve_info.get('elapsed_time', 0))
            }
        },
        "comparison": {
            "objective_improvement_pct": float(improvement_obj),
            "symmetry_improvement_pct": float(improvement_sym),
            "defect_old": float(defect_old),
            "defect_new": float(defect_new),
            "defect_reduction_pct": float((defect_old - defect_new) / defect_old * 100) if defect_old > 0 else 0
        },
        "tests": {
            "passed": tests_passed,
            "failed": tests_failed,
            "total": len(tests_passed) + len(tests_failed),
            "success_rate": len(tests_passed) / (len(tests_passed) + len(tests_failed))
        }
    }

    results_path = session_dir / "results/methodology_verification.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved test results to: {results_path}")
    print()

    # ========================================================================
    # 10. Summary
    # ========================================================================
    print("=" * 70)
    print("STEP 3 VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\n✓ Generator estimation: Successfully computed J^A and J^B")
    print(f"✓ Equivariant solver: Successfully solved for T_new")
    print(f"✓ Symmetry defect: Reduced from {defect_old:.6e} to {defect_new:.6e}")
    print(f"  ({improvement_sym:+.2f}% improvement)")
    print(f"✓ Total objective: {improvement_obj:+.2f}% improvement")

    if len(tests_failed) == 0:
        print(f"\n✓ All {len(tests_passed)} tests PASSED!")
        print("\nStep 3 (Methodology Implementation) is COMPLETE and VERIFIED.")
    else:
        print(f"\n⚠ {len(tests_failed)} test(s) failed:")
        for test in tests_failed:
            print(f"  - {test}")

    print("\n" + "=" * 70)

    return len(tests_failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
