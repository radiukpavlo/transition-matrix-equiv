"""
Data Setup and Verification Script

This script:
1. Imports and validates synthetic matrices A and B
2. Loads and validates MNIST dataset
3. Saves synthetic matrices to workflow/data/synthetic.npz
4. Prints a comprehensive summary
"""

import sys
import os
import numpy as np

# Add workflow directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_data import A, B
from mnist_loader import load_mnist, get_mnist_summary

def main():
    print("=" * 70)
    print("STEP 1: Data & Environment Setup")
    print("=" * 70)

    # =========================================================================
    # Part 1: Verify Synthetic Matrices
    # =========================================================================
    print("\n1. SYNTHETIC MATRICES VERIFICATION")
    print("-" * 70)

    print(f"\nMatrix A (Formal Model - FM):")
    print(f"  Shape: {A.shape}")
    print(f"  Expected: (15, 5)")
    print(f"  Data type: {A.dtype}")

    # Assert correct shape
    assert A.shape == (15, 5), f"Matrix A should be 15×5, got {A.shape}"
    print("  ✓ Shape verification passed!")

    print(f"\nMatrix B (Mental Model - MM):")
    print(f"  Shape: {B.shape}")
    print(f"  Expected: (15, 4)")
    print(f"  Data type: {B.dtype}")

    # Assert correct shape
    assert B.shape == (15, 4), f"Matrix B should be 15×4, got {B.shape}"
    print("  ✓ Shape verification passed!")

    # Display sample data
    print(f"\nMatrix A - First 3 rows:")
    for i in range(3):
        print(f"  Row {i}: {A[i]}")

    print(f"\nMatrix B - First 3 rows:")
    for i in range(3):
        print(f"  Row {i}: {B[i]}")

    # =========================================================================
    # Part 2: Load and Verify MNIST Dataset
    # =========================================================================
    print("\n\n2. MNIST DATASET VERIFICATION")
    print("-" * 70)
    print("Loading MNIST dataset (this may take a moment)...")

    train_data, train_labels, test_data, test_labels = load_mnist()

    print(f"\nTraining Data:")
    print(f"  Shape: {train_data.shape}")
    print(f"  Expected: (60000, 1, 28, 28)")
    print(f"  Data type: {train_data.dtype}")
    print(f"  Value range: [{train_data.min():.3f}, {train_data.max():.3f}]")

    # Assert correct shape
    assert train_data.shape == (60000, 1, 28, 28), \
        f"Expected (60000, 1, 28, 28), got {train_data.shape}"
    print("  ✓ Shape verification passed!")

    print(f"\nTraining Labels:")
    print(f"  Shape: {train_labels.shape}")
    print(f"  Unique classes: {np.unique(train_labels)}")

    print(f"\nTest Data:")
    print(f"  Shape: {test_data.shape}")
    print(f"  Expected: (10000, 1, 28, 28)")
    print(f"  Data type: {test_data.dtype}")
    print(f"  Value range: [{test_data.min():.3f}, {test_data.max():.3f}]")

    # Assert correct shape
    assert test_data.shape == (10000, 1, 28, 28), \
        f"Expected (10000, 1, 28, 28), got {test_data.shape}"
    print("  ✓ Shape verification passed!")

    print(f"\nTest Labels:")
    print(f"  Shape: {test_labels.shape}")
    print(f"  Unique classes: {np.unique(test_labels)}")

    # Get detailed summary
    summary = get_mnist_summary(train_data, train_labels, test_data, test_labels)

    print("\nClass Distribution (Training Set):")
    for label, count in summary['train_label_distribution'].items():
        print(f"  Class {label}: {count:5d} samples ({100*count/summary['train_size']:.1f}%)")

    print("\nClass Distribution (Test Set):")
    for label, count in summary['test_label_distribution'].items():
        print(f"  Class {label}: {count:5d} samples ({100*count/summary['test_size']:.1f}%)")

    # =========================================================================
    # Part 3: Save Synthetic Matrices
    # =========================================================================
    print("\n\n3. SAVING SYNTHETIC MATRICES")
    print("-" * 70)

    output_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'synthetic.npz'
    )

    np.savez(
        output_path,
        A=A,
        B=B,
        description="Synthetic matrices from manuscript Appendix 1.1"
    )

    print(f"Synthetic matrices saved to: {output_path}")

    # Verify saved file
    loaded = np.load(output_path)
    assert np.array_equal(loaded['A'], A), "Saved matrix A doesn't match!"
    assert np.array_equal(loaded['B'], B), "Saved matrix B doesn't match!"
    print("  ✓ Save verification passed!")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ All verifications passed successfully!")
    print(f"\nDatasets prepared:")
    print(f"  • Synthetic matrices: A {A.shape}, B {B.shape}")
    print(f"  • MNIST training: {train_data.shape}")
    print(f"  • MNIST test: {test_data.shape}")
    print(f"\nOutput files:")
    print(f"  • {output_path}")
    print("\nNext steps:")
    print("  • Matrices are ready for subsequent experiments")
    print("  • Use 'np.load()' to load synthetic matrices")
    print("  • Use 'mnist_loader.load_mnist()' to reload MNIST data")
    print("=" * 70)


if __name__ == "__main__":
    main()
