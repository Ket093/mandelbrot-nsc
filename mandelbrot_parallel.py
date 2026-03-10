"""
Mandelbrot Set Generator - Parallel Version
Author : Akondeng Atengong Ketawah
Course : Numerical Scientific Computing 2026
"""

import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import os
import statistics
import matplotlib.pyplot as plt
from pathlib import Path

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    """
    Calculate escape time for a single pixel.
    
    Parameters
    ----------
    c_real, c_imag : float
        Real and imaginary parts of complex number c
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    int
        Iteration count when escaped, or max_iter if in set
    """
    z_real = 0.0
    z_imag = 0.0
    
    for i in range(max_iter):
        # Check if point has escaped (|z| > 2)
        if z_real*z_real + z_imag*z_imag > 4.0:
            return i
        
        # Update z = z² + c (using real and imaginary parts)
        new_real = z_real*z_real - z_imag*z_imag + c_real
        new_imag = 2.0*z_real*z_imag + c_imag
        z_real = new_real
        z_imag = new_imag
    
    return max_iter