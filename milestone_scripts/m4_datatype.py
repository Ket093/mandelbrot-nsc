"""
MILESTONE 4: Data Type Optimization (float32 vs float64)
Author: Akondeng Atengong Ketawah
Course: Numerical Scientific Computing 2026
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import sys
import os

# Add parent directory for mandelbrot.py import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("M4: Setup complete - ready to implement precision functions")