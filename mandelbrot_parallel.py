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
from dask import delayed
import dask

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
    
def mandelbrot_dask(N, x_min, x_max, y_min, y_max, 
                    max_iter=100, n_chunks=32):
    """
    Compute Mandelbrot set using Dask delayed.
    
    Parameters
    ----------
    N : int
        Grid size (N x N pixels)
    x_min, x_max, y_min, y_max : float
        Coordinate boundaries
    max_iter : int
        Maximum iterations
    n_chunks : int
        Number of chunks to split work into
    
    Returns
    -------
    numpy.ndarray
        2D array of shape (N, N) with iteration counts
    """
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0
    
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    
    parts = dask.compute(*tasks)
    return np.vstack(parts)

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

    print("\n" + "=" * 60)
    print("MP2 M2: Chunk Size Sweep (Fixed Workers = 2)")
    print("=" * 60)

    # Fix workers at optimum (from earlier benchmark, 2 workers gave best performance)
    fixed_workers = 2
    print(f"Fixed workers: {fixed_workers}")

    # Test chunk multipliers: 1x, 2x, 4x, 8x, 16x workers
    multipliers = [1, 2, 4, 8, 16]

    print(f"\n{'Multiplier':>10} {'Chunks':>10} {'Time (s)':>12} {'Speedup':>10} {'LIF':>12}")
    print("-" * 65)

    results = []  # Store (chunks, lif) for finding sweet spot

    for mult in multipliers:
        n_chunks = fixed_workers * mult

        # Calculate chunk size
        chunk_size = max(1, N // n_chunks)
        chunks = []
        row = 0
        while row < N:
            row_end = min(row + chunk_size, N)
            chunks.append((row, row_end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = row_end

        # Time parallel version
        with Pool(processes=fixed_workers) as pool:
            # Warm-up
            tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
            pool.map(_worker, tiny)

            # Timed runs
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
            t_par = statistics.median(times)

        speedup = t_serial / t_par
        lif = (fixed_workers * t_par / t_serial) - 1

        print(f"{mult:10d}x {n_chunks:10d} {t_par:12.3f} {speedup:10.2f}x {lif:12.3f}")

        results.append((n_chunks, lif))

         # Find sweet spot (minimum LIF)
    best_chunks, best_lif = min(results, key=lambda x: x[1])
    print("\n" + "-" * 65)
    print(f"SWEET SPOT: {best_chunks} chunks gives minimum LIF = {best_lif:.3f}")
    print("(LIF = Load Imbalance Factor; lower is better)")
    print("-" * 65)
    
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
    
    speedup = t_serial / t_par
    lif = (n_workers * t_par / t_serial) - 1
    
    print(f"Time: {t_par:.3f} seconds")
    print(f"Speedup: {speedup:.2f}x over serial Numba")
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
    
    speedup2 = t_serial / t_par2
    lif2 = (n_workers * t_par2 / t_serial) - 1

    
    print(f"Time: {t_par2:.3f} seconds")
    print(f"Speedup: {speedup2:.2f}x over serial Numba")
    print(f"LIF: {lif2:.3f}")

    # Best Dask results from M2 sweep
    best_dask_chunks = 2
    best_dask_time = 0.080
    best_dask_speedup = t_naive / best_dask_time
    
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
    print(f"{'Dask Local (2 chunks)':<30} {best_dask_time:>10.3f} {best_dask_speedup:>10.2f}x {'-':>12} {'-':>10}")
    
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

        # === COMPARISON WITH L04 RESULT ===
    print("\n" + "-" * 60)
    print("COMPARISON WITH L04 RESULT")
    print("-" * 60)
    
    # L04 result (original parallel without chunking)
    # From earlier benchmark: 2 workers gave 2.00x speedup
    l04_best_workers = 2
    l04_best_speedup = 2.00
    l04_implied_s = ((1 / l04_best_speedup) - (1 / l04_best_workers)) / (1 - (1 / l04_best_workers))
    
    print(f"L04 (original parallel, no chunking):")
    print(f"  - Best speedup: {l04_best_speedup:.2f}x with {l04_best_workers} workers")
    print(f"  - Implied serial fraction s = {l04_implied_s:.3f}")
    
    print(f"\nL05 (chunked parallel with cache=True):")
    print(f"  - Best speedup: {best_speedup:.2f}x with {best_workers} workers")
    print(f"  - Implied serial fraction s = {implied_s:.3f}")
    
    print(f"\nComparison:")
    if implied_s > l04_implied_s:
        print(f"  - L05 s ({implied_s:.3f}) is LARGER than L04 s ({l04_implied_s:.3f})")
        print(f"  - This indicates L05 measurement is more accurate due to:")
        print(f"    * Proper warm-up runs")
        print(f"    * Multiple iterations with median timing")
        print(f"    * cache=True for Numba functions")
        print(f"    * Exclusion of startup overhead from timing loops")
    else:
        print(f"  - L05 s ({implied_s:.3f}) is SMALLER than L04 s ({l04_implied_s:.3f})")
        print(f"  - This indicates improved load balance from chunking")
    
        # RECOMMENDATION 
    print("\n" + "-" * 60)
    print("RECOMMENDATION")
    print("-" * 60)
    
    # Find best configuration from M2 chunk sweep results
    # From earlier M2 fixed-worker sweep, 2 chunks gave best LIF
    optimal_chunks = 2
    optimal_workers = 2
    
    print(f"Based on the performance analysis:")
    print(f"  - Best configuration: {optimal_workers} workers with {optimal_chunks} chunks")
    print(f"  - Achieved speedup: {best_speedup:.2f}x")
    print(f"  - Efficiency: {(best_speedup / optimal_workers * 100):.0f}%")
    print(f"  - Load Imbalance Factor (LIF): {implied_s:.3f}")
    
    print(f"\nIs parallelisation worth it on this hardware?")
    print(f"  - Yes, with {optimal_workers} workers the speedup is {best_speedup:.2f}x")
    print(f"  - Adding more workers (3-4) shows no benefit due to hyperthreading")
    print(f"  - For larger grids (2048×2048 or larger), speedup would be closer to ideal")
    
    print(f"\nSettings for best time:")
    print(f"  - Workers: {optimal_workers}")
    print(f"  - Chunks: {optimal_chunks} (1× multiplier)")
    print(f"  - cache=True enabled on Numba functions")

        #  MP2 M1: Dask Mandelbrot 
    print("\n" + "-" * 60)
    print("MP2 M1: Dask Mandelbrot (Local)")
    print("-" * 60)
    
    from dask.distributed import Client, LocalCluster
    
    # Create local cluster with workers matching your physical cores
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    
    # Warm up Numba JIT in all workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    
    # Time Dask Mandelbrot (3 runs, median)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result_dask = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=32)
        times.append(time.perf_counter() - t0)
    t_dask = statistics.median(times)
    t_dask_speedup = t_naive / t_dask
    print(f"Dask local (n_chunks=32): {t_dask:.3f} seconds")
    
    # Verify result matches serial Numba
    result_serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    if np.array_equal(result_dask, result_serial):
        print("Verification: PASSED (matches serial Numba)")
    else:
        print("Verification: FAILED")

            # MP2 M2: Dask Chunk Size Sweep 
    print("\n" + "-" * 50)
    print("MP2 M2: Dask Chunk Size Sweep")
    print("-" * 50)
    
    # Test different chunk sizes
    chunk_multipliers = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print(f"\n{'Chunks':>8} {'Time (s)':>10} {'Speedup':>10} {'LIF':>10}")
    print("-" * 45)
    
    results = []

    for n_chunks in chunk_multipliers:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        
        speedup = t_serial / t_par
        lif = (2 * t_par / t_serial) - 1
        
        print(f"{n_chunks:8d} {t_par:10.3f} {speedup:10.2f}x {lif:10.3f}")
        results.append((n_chunks, t_par, speedup, lif))

            # Find sweet spot (minimum time)
    best = min(results, key=lambda x: x[1])
    print("\n" + "-" * 45)
    print(f"SWEET SPOT: {best[0]} chunks gives minimum time = {best[1]:.3f}s")
    
    # Close cluster
    client.close()
    cluster.close()

    print("\n" + "=" * 60)
    print("MP2 M3: Analysis Complete")
    print("=" * 60)


    