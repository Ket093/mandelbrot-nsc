"""
MILESTONE 3: Numba JIT Compilation
Author: Akondeng Atengong Ketawah
Course: Numerical Scientific Computing 2026
"""

# Check if Numba is installed
import numba
print(f"Numba version: {numba.__version__}")
print("Numba is ready to use!")

import time
import statistics
import numpy as np
from numba import njit
import sys
import os

# Add parent directory for mandelbrot.py import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from mandelbrot.py file
from mandelbrot import compute_mandelbrot, numpy_mandelbrot

print("All imports completed successfully!")

# Create Numba version of naive function
@njit
def mandelbrot_numba(xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, width=1024, height=1024, max_iter=100):
    # Create coordinate arrays
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    # Result array
    result = np.zeros((height, width), dtype=np.int32)

    # Triple nested loops 
    for i in range(height):          # Loop over rows
        y_val = y[i]
        for j in range(width):       # Loop over columns
            x_val = x[j]
            c = x_val + 1j * y_val   # Create complex number

            z = 0j                    # Initialize z
            n = 0                     # Iteration counter

            # OPTIMIZATION: squared magnitude used so as to avoid sqrt
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z*z + c
                n += 1

            result[i, j] = n

    return result