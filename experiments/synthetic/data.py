"""
Synthetic data for Section 3.4 experiments.

Data from manuscript Appendix 1.1:
- Matrix A ∈ ℝ^(15×5): FM features (5 dimensions)
- Matrix B ∈ ℝ^(15×4): MM features (4 dimensions)
- 15 samples divided into 3 classes (5 samples each)
"""

import numpy as np


import json
from pathlib import Path

def load_synthetic_data_from_json() -> dict:
    """
    Load synthetic feature matrices and labels from JSON.
    
    Returns
    -------
    dict
        Dictionary containing 'A', 'B', 'T_old', and 'labels'.
    """
    data_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic_matrices.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return {
        'A': np.array(data['A']),
        'B': np.array(data['B']),
        'T_old': np.array(data['T_old']),
        'labels': np.array(data['labels'])
    }


def get_matrix_A() -> np.ndarray:
    """Get synthetic FM feature matrix A."""
    return load_synthetic_data_from_json()['A']


def get_matrix_B() -> np.ndarray:
    """Get synthetic MM feature matrix B."""
    return load_synthetic_data_from_json()['B']


def get_matrix_T_old() -> np.ndarray:
    """Get baseline transition matrix T_old."""
    return load_synthetic_data_from_json()['T_old']


def get_class_labels() -> np.ndarray:
    """Get class labels."""
    return load_synthetic_data_from_json()['labels']


def get_synthetic_data() -> dict:
    """
    Get all synthetic data as a dictionary.
    
    Returns
    -------
    dict
        Dictionary with keys: 'A', 'B', 'T_old', 'labels', 'm', 'k', 'l'
    """
    raw_data = load_synthetic_data_from_json()
    A = raw_data['A']
    B = raw_data['B']
    T_old = raw_data['T_old']
    labels = raw_data['labels']
    
    m, k = A.shape
    _, l = B.shape
    
    return {
        'A': A,
        'B': B,
        'T_old': T_old,
        'labels': labels,
        'm': m,
        'k': k,
        'l': l,
    }


# Expected results from manuscript Table 1
EXPECTED_RESULTS = {
    'old_approach': {
        'mse_fid': 0.002,
        'sym_err': 1.450,
        'robustness_err': 0.850,
    },
    'new_approach': {
        'mse_fid': 0.005,
        'sym_err': 0.080,
        'robustness_err': 0.120,
    },
}
