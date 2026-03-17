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

@njit
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):
    """
    Compute Mandelbrot for a range of rows.
    """
    # Create empty array to store results for this chunk
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    
    # Calculate how much to move between pixels
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    
    # For each row in this chunk
    for r in range(row_end - row_start):
        # Calculate y coordinate (imaginary part) for this row
        c_imag = y_min + (r + row_start) * dy
        
        # For each column in this row
        for col in range(N):
            # Calculate x coordinate (real part) for this column
            c_real = x_min + col * dx
            
            # Compute the pixel value and store it
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)
    
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    """
    Compute entire Mandelbrot set using a single chunk.
    
    This is a wrapper around mandelbrot_chunk that processes all rows at once.
    It gives us a baseline to compare against parallel versions.
    """
    # Call chunk function for ALL rows (from row 0 to row N)
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)
def _worker(args):
    """
    Unpack arguments for mandelbrot_chunk.
    
    This helper is needed because pool.map() only passes one argument.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter)
    
    Returns
    -------
    numpy.ndarray
        2D array from mandelbrot_chunk
    """
    # The * operator "unpacks" the tuple into separate arguments
    return mandelbrot_chunk(*args)