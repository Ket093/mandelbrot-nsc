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

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, 
                        max_iter=100, n_workers=4, n_chunks=None, pool=None):
    """
    Compute Mandelbrot set in parallel with configurable chunks.
    
    Parameters
    ----------
    N : int
        Grid size (N x N pixels)
    x_min, x_max, y_min, y_max : float
        Coordinate boundaries
    max_iter : int
        Maximum iterations
    n_workers : int
        Number of worker processes
    n_chunks : int or None
        Number of chunks to split work into (if None, uses n_workers)
    pool : multiprocessing.Pool or None
        Existing pool to reuse (if None, creates new pool)
    
    Returns
    -------
    numpy.ndarray
        2D array of shape (N, N) with iteration counts
    """
    # If n_chunks not specified, use n_workers (original behavior)
    if n_chunks is None:
        n_chunks = n_workers
    
    print(f"  Using {n_workers} workers, {n_chunks} chunks")
    
    # Calculate chunk size (at least 1 row per chunk)
    chunk_size = max(1, N // n_chunks)
    
    # Create list of chunks
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    print(f"  Created {len(chunks)} chunks of work")
    
    # Use existing pool if provided
    if pool is not None:
        return np.vstack(pool.map(_worker, chunks))
    
    # Create new pool (with warm-up)
    with Pool(processes=n_workers) as new_pool:
        # Tiny warm-up to load Numba cache
        tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
        new_pool.map(_worker, tiny)
        
        # Actual computation
        results = new_pool.map(_worker, chunks)
        return np.vstack(results)

if __name__ == "__main__":
    N = 1024
    max_iter = 100
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    
    print("-" * 40)
    print("MP2 M3: Parallel Mandelbrot Benchmark")
    print("=" * 40)
    print(f"\nGrid: {N} x {N}")
    print(f"Max iterations: {max_iter}")
    print(f"Region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # === SERIAL BASELINE ===
    print("\n" + "-" * 40)
    print("Serial baseline")
    print("-" * 40)
    
    _ = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    
    times = []
    for _ in range(3):
        start = time.time()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times.append(time.time() - start)
    t_serial = statistics.median(times)
    print(f"Serial time: {t_serial:.3f} seconds")
    
    # === PARALLEL BENCHMARK ===
    print("\n" + "-" * 40)
    print("Parallel benchmark")
    print("-" * 40)
    print(f"{'Workers':>8} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 50)
    
    cpu_count = os.cpu_count()
    
    for n_workers in range(1, cpu_count + 1):
        chunk_size = max(1, N // n_workers)
        chunks = []
        row = 0
        while row < N:
            row_end = min(row + chunk_size, N)
            chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
            row = row_end
        
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)  # warm-up
            
            times = []
            for _ in range(3):
                start = time.time()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.time() - start)
            t_par = statistics.median(times)
        
        speedup = t_serial / t_par
        efficiency = (speedup / n_workers) * 100
        
        print(f"{n_workers:8d} {t_par:10.3f} {speedup:10.2f}x {efficiency:11.1f}%")
    
    print("\n" + "-" * 40)
    print("Benchmark complete")
    print("-" * 40)