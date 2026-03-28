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

@njit(cache=True)
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
def mandelbrot_point(c, max_iter=100):
    """Test if a point is in the Mandelbrot set."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter
def numpy_mandelbrot(xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, width=1024, height=1024, max_iter=100):
    """NumPy vectorized Mandelbrot implementation - much faster!"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M
def compute_mandelbrot(x_min, x_max, y_min, y_max, width=100, height=100, max_iter=100):
    """Create a grid of Mandelbrot values."""
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)

    results = np.zeros((height, width), dtype=int)

    for row in range(height):
        y = y_vals[row]
        for col in range(width):
            x = x_vals[col]
            c = x + 1j * y
            results[row, col] = mandelbrot_point(c, max_iter)

    return results

@njit(cache=True)
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
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # Serial baseline (Numba already warm after M1 warm-up)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    for n_workers in range(1, os.cpu_count() + 1):
        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)  # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
            t_par = statistics.median(times)

            speedup = t_serial / t_par
            lif = (n_workers * t_par / t_serial) - 1
            print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, LIF={lif:.3f}")
    
    print("\n" + "-" * 60)
    print("M2 COMPLETE - Starting M3 Analysis")
    print("=" * 60)
    print("MP2 M3: Comprehensive Performance Analysis")
    print("=" * 60)
    print(f"\nGrid: {N} x {N}")
    print(f"Max iterations: {max_iter}")
    print(f"Region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # === WARM-UP ===
    print("\n" + "-" * 60)
    print("WARM-UP")
    print("-" * 60)
    print("Running warm-up to compile Numba code...")
    _ = mandelbrot_serial(64, x_min, x_max, y_min, y_max, max_iter)
    print("Warm-up complete")

    print("\n" + "-" * 60)
    print("IMPLEMENTATION 0: Naive")
    print("-" * 60)

    times = []
    for _ in range(5):
        start = time.time()
        compute_mandelbrot(-2.5, 1.0, -1.25, 1.25, 1024, 1024, 100)
        times.append(time.time() - start)
    t_naive = statistics.median(times)
    print(f"Time: {t_naive:.3f} seconds")

    # === IMPLEMENTATION 0: NUMPY ===

    print("\n" + "-" * 60)
    print("IMPLEMENTATION 0: Numpy")
    print("-" * 60)

    times = []
    for _ in range(5):
        start = time.time()
        numpy_mandelbrot(-2.5, 1.0, -1.25, 1.25, 1024, 1024, 100)
        times.append(time.time() - start)
    t_numpy = statistics.median(times)
    numpy_speedup = t_naive / t_numpy
    print(f"Time: {t_numpy:.3f} seconds")
    print(f"Speedup: {numpy_speedup:.2f}x over naive")
    
    # === IMPLEMENTATION 1: SERIAL BASELINE ===
    print("\n" + "-" * 60)
    print("IMPLEMENTATION 1: Serial Numba")
    print("-" * 60)
    
    times = []
    for _ in range(5):
        start = time.time()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times.append(time.time() - start)
    t_serial = statistics.median(times)
    t_serial_speedup = t_naive / t_serial
    print(f"Time: {t_serial:.3f} seconds")
    print(f"Speedup: {t_serial_speedup:.2f}x over naive")
    
    # === IMPLEMENTATION 2: PARALLEL WITH OPTIMAL CHUNKS ===
    print("\n" + "-" * 60)
    print("IMPLEMENTATION 2: Parallel (2 workers, 2 chunks)")
    print("-" * 60)
    
    n_workers = 2
    n_chunks = 2  # 1x multiplier
    
    # Create chunks
    chunk_size = max(1, N // n_chunks)
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    # Time parallel version
    with Pool(processes=n_workers) as pool:
        # Warm-up
        tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
        pool.map(_worker, tiny)
        
        # Timed runs
        times = []
        for _ in range(5):
            start = time.time()
            np.vstack(pool.map(_worker, chunks))
            times.append(time.time() - start)
        t_par = statistics.median(times)
    
    speedup = t_naive / t_par
    lif = (n_workers * t_par / t_serial) - 1
    
    print(f"Time: {t_par:.3f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"LIF: {lif:.3f}")
    
    # === IMPLEMENTATION 3: PARALLEL WITH MORE CHUNKS ===
    print("\n" + "-" * 60)
    print("IMPLEMENTATION 3: Parallel (2 workers, 32 chunks)")
    print("-" * 60)
    
    n_chunks = 32  # 16x multiplier
    
    # Create chunks
    chunk_size = max(1, N // n_chunks)
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    # Time parallel version
    with Pool(processes=n_workers) as pool:
        # Warm-up
        tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
        pool.map(_worker, tiny)
        
        # Timed runs
        times = []
        for _ in range(5):
            start = time.time()
            np.vstack(pool.map(_worker, chunks))
            times.append(time.time() - start)
        t_par2 = statistics.median(times)
    
    speedup2 = t_naive / t_par2
    lif2 = (n_workers * t_par2 / t_serial) - 1

    
    print(f"Time: {t_par2:.3f} seconds")
    print(f"Speedup: {speedup2:.2f}x")
    print(f"LIF: {lif2:.3f}")
    
    # === SUMMARY TABLE ===
    print("\n" + "=" * 60)
    print("SUMMARY: Performance Comparison")
    print("=" * 60)
    print(f"\n{'Implementation':<30} {'Time (s)':>10} {'Speedup':>10} {'LIF':>10}")
    print("-" * 80)
    print(f"{'Naive':<30} {t_naive:>10.3f} {1.00:>10.2f}x {'-':>12} {'-':>10}")
    print(f"{'Serial Numba':<30} {t_serial:>10.3f} {t_serial_speedup:>10.2f}x {'-':>12} {'-':>10}")
    print(f"{'Parallel (2 workers, 2 chunks)':<30} {t_par:>10.3f} {speedup:>10.2f}x {lif:>10.3f}")
    print(f"{'Parallel (2 workers, 32 chunks)':<30} {t_par2:>10.3f} {speedup2:>10.2f}x {lif2:>10.3f}")
    print(f"{'Numpy Vectorized':<30} {t_numpy:>10.3f} {numpy_speedup:>10.2f}x {'-':>12} {'-':>10}")
    
    # === IMPLIED SERIAL FRACTION ===
    print("\n" + "-" * 60)
    print("AMDAHL'S LAW ANALYSIS")
    print("-" * 60)
    
    # Using best parallel result
    best_speedup = max(speedup, speedup2)
    best_workers = n_workers
    
    # Back-solve for implied serial fraction: s = (1/Sp - 1/p) / (1 - 1/p)
    implied_s = ((1/best_speedup) - (1/best_workers)) / (1 - (1/best_workers))
    
    print(f"Best speedup achieved: {best_speedup:.2f}x with {best_workers} workers")
    print(f"Implied serial fraction s = {implied_s:.3f}")
    print(f"This means approximately {implied_s*100:.1f}% of the code is effectively serial")
    print(f"(due to overhead, not actual algorithm serial fraction)")
    
    print("\n" + "=" * 60)
    print("MP2 M3: Analysis Complete")
    print("=" * 60)


    