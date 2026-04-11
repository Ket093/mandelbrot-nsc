"""
E1: Find machine epsilon experimentally
"""

import numpy as np

def find_machine_epsilon(dtype=np.float64):
    """
    Find machine epsilon by successive halving.
    
    Start with eps = 1.0
    Keep dividing by 2 until 1.0 + eps/2 == 1.0
    The last eps that changed the result is machine epsilon.
    """
    eps = dtype(1.0)
    count = 0
    
    while dtype(1.0) + eps / dtype(2.0) != dtype(1.0):
        eps = eps / dtype(2.0)
        count = count + 1
    
    return eps, count

# Test for different data types
dtypes = [np.float16, np.float32, np.float64]


print("E1: Finding Machine Epsilon")


for dtype in dtypes:
    computed, steps = find_machine_epsilon(dtype)
    reference = np.finfo(dtype).eps
    
    print(f"\n{dtype.__name__}:")
    print(f"  Computed epsilon: {float(computed):.4e}")
    print(f"  Reference epsilon: {float(reference):.4e}")
    print(f"  Number of halving steps: {steps}")
    print(f"  Match: {'✓' if abs(float(computed) - reference) < 1e-10 else '✗'}")