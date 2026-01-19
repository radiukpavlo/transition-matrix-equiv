
import numpy as np
import sys
from pathlib import Path

def print_matrix(name, M):
    print(f"\n{name} {M.shape}:")
    with np.printoptions(precision=3, suppress=True, linewidth=120):
        print(M)

def inspect_synthetic():
    path = Path('outputs/synthetic/matrices.npz')
    if not path.exists():
        print("File not found")
        return

    data = np.load(path)
    print("Keys:", data.files)
    
    # Print small matrices directly
    print_matrix("A (Input Features)", data['A'])
    print_matrix("B (Target Features)", data['B'])
    print_matrix("T_old (Baseline)", data['T_old'])
    print_matrix("T_new (Equivariant)", data['T_new'])
    print_matrix("J_A (Generator A)", data['J_A'])
    print_matrix("J_B (Generator B)", data['J_B'])

if __name__ == "__main__":
    inspect_synthetic()
