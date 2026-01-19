"""
Synthetic Data Module

This module defines the exact matrices A (15×5) and B (15×4) from the manuscript
(Appendix 1.1 and Section 3.1.2 - Illustrative Numerical Example).

Matrix A: Formal Model (FM) feature vectors - 15 samples × 5 features
Matrix B: Mental Model (MM) feature vectors - 15 samples × 4 features
"""

import numpy as np

# Matrix A (15×5) - From Section 3.1.2, equation (15)
# Formal Model feature vectors
# Class 1: rows 0-4, Class 2: rows 5-9, Class 3: rows 10-14
A = np.array([
    # Class 1
    [2.8, -1.8, -2.8, 1.3, 0.4],
    [2.9, -1.9, -2.9, 1.4, 0.5],
    [3.0, -2.0, -3.0, 1.5, 0.6],
    [3.1, -2.1, -3.1, 1.6, 0.7],
    [3.2, -2.2, -3.2, 1.7, 0.8],
    # Class 2
    [1.6, -2.5, -1.5, 0.2, 0.6],
    [1.3, -2.7, -1.3, 0.4, 0.8],
    [1.0, -3.0, -1.5, 0.6, 1.0],
    [0.7, -3.2, -1.7, 0.8, 1.2],
    [0.5, -3.5, -1.9, 1.0, 1.4],
    # Class 3
    [1.2, -1.2, -0.7, 0.3, 2.8],
    [1.1, -1.1, -0.8, 0.4, 2.9],
    [1.0, -1.0, -0.844444, 0.44444, 3.0],
    [0.9, -0.9, -0.85, 0.45, 3.1],
    [0.8, -0.8, -0.9, 0.5, 3.2]
])

# Matrix B (15×4) - From the manuscript matrix display (lines 1667-1694)
# Mental Model feature vectors (transformed/reduced dimensionality)
B = np.array([
    # Class 1
    [1.959307524, -1.381119943, -1.729640979, -1.97939],
    [1.99818664, -1.912855282, -1.97511053, -1.84391],
    [1.998896097, -1.999605076, -1.9989167665, -1.99937],
    [1.997776, -1.844000202, 1.660111333, -1.373532039],
    [1.992024, -1.923804827, 0.706593926, -1.543784398],
    # Class 2
    [1.997854, -1.999410881, -0.243400633, -1.827587263],
    [1.574201387, 1.581026838, 0.851626, 1.573934081],
    [1.615475549, 1.723582196, 1.107744, 1.807614602],
    [1.695289797, 1.953503509, 1.290406, 1.946250271],
    [-1.97939, 1.959307524, -1.381119943, -1.729640979],
    # Class 3
    [-1.84391, 1.99818664, -1.912855282, -1.97511053],
    [-1.99937, 1.998896097, -1.999605076, -1.9989167665],
    [1.660111333, -1.373532039, 1.997776, -1.844000202],
    [0.706593926, -1.543784398, 1.992024, -1.923804827],
    [-0.243400633, -1.827587263, 1.997854, -1.999410881]
])

# Verify shapes
assert A.shape == (15, 5), f"Matrix A should be 15×5, got {A.shape}"
assert B.shape == (15, 4), f"Matrix B should be 15×4, got {B.shape}"

if __name__ == "__main__":
    print("Synthetic Data Matrices")
    print("=" * 50)
    print(f"Matrix A shape: {A.shape} (15 samples × 5 features)")
    print(f"Matrix B shape: {B.shape} (15 samples × 4 features)")
    print("\nMatrix A (Formal Model):")
    print(A)
    print("\nMatrix B (Mental Model):")
    print(B)
