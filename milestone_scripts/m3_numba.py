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