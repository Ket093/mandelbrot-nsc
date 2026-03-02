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

# Create a benchmark function for functions that take 7 arguments (NumPy and Numba)
def bench(func, *args, runs=5):
    print(f"  Warming up {func.__name__}...")
    func(*args)  # (Not timed)

    times = []
    for i in range(runs):
        t0 = time.perf_counter()  # Start timer
        func(*args)                # Run the function
        t1 = time.perf_counter()   # Stop timer
        times.append(t1 - t0)      # Record time in seconds
        print(f"    Run {i+1}: {times[-1]*1000:.2f} ms")

    median = statistics.median(times)
    return median

# Special benchmark function for naive version (takes 3 arguments)
def bench_naive(func, width, height, max_iter, runs=5):
    """Benchmark for compute_mandelbrot which takes 3 arguments"""
    print(f"  Warming up {func.__name__}...")
    func(width=width, height=height, max_iter=max_iter)  # Warm-up

    times = []
    for i in range(runs):
        t0 = time.perf_counter()  # Start timer
        func(width=width, height=height, max_iter=max_iter)  # Run with keyword args
        t1 = time.perf_counter()   # Stop timer
        times.append(t1 - t0)      # Record time in seconds
        print(f"    Run {i+1}: {times[-1]*1000:.2f} ms")

    median = statistics.median(times)
    return median

# Run the benchmarks
print("=" * 70)
print("MILESTONE 3: Numba JIT Compilation")
print("-" * 70)

# Warm up Numba (compilation happens here - Not timed!)
print("\nWarming up Numba (compilation)...")
_ = mandelbrot_numba(-2, 1, -1.5, 1.5, 64, 64, 100)
print("Numba compiled successfully!")
    
# Arguments for vectorized functions (NumPy and Numba take 7 args)
args_vectorized = (-2, 1, -1.5, 1.5, 1024, 1024, 100)

print("\nBenchmarking all versions (5 runs each)...")
print("-" * 40)

# Benchmark Naive version (uses special function with 3 arguments)
print("\n1. Benchmarking Naive version...")
print("-" * 40)
t_naive = bench_naive(compute_mandelbrot, 1024, 1024, 100)

# Benchmark NumPy version (uses 7 arguments)
print("\n2. Benchmarking NumPy version...")
print("-" * 40)
t_numpy = bench(numpy_mandelbrot, *args_vectorized)