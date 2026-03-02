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

# Separate Numba functions for each precision type
@njit
def mandelbrot_float64(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """Mandelbrot with float64 precision"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    result = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        y_val = y[i]
        for j in range(width):
            x_val = x[j]
            c = x_val + 1j * y_val  # complex128
            
            z = 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z*z + c
                n += 1
            
            result[i, j] = n
    
    return result

print("M4: Setup complete - ready to implement precision functions")

@njit
def mandelbrot_float32(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """Mandelbrot with float32 precision"""
    x = np.linspace(xmin, xmax, width).astype(np.float32)
    y = np.linspace(ymin, ymax, height).astype(np.float32)
    
    result = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        y_val = y[i]
        for j in range(width):
            x_val = x[j]
            c = np.complex64(x_val + 1j * y_val)
            
            z = np.complex64(0j)
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z*z + c
                n += 1
            
            result[i, j] = n
    
    return result

print("+" * 40)
print("MILESTONE 4: Data Type Optimization")
print("+" * 40)

# Configuration (using 512x512 for faster testing)
width, height = 512, 512
max_iter = 100
runs = 3

print(f"\nConfiguration:")
print(f"  Resolution: {width} x {height}")
print(f"  Max iterations: {max_iter}")
print(f"  Runs per type: {runs}")
print("+" * 40)

# Test each precision
results = {}
times = {}

# Test float64
print(f"\nTesting float64...")
funcs = [mandelbrot_float64, mandelbrot_float32]
names = ['float64', 'float32']

for func, name in zip(funcs, names):
    print(f"\nTesting {name}...")
    
    # Warm-up run
    _ = func(-2, 1, -1.5, 1.5, width, height, max_iter)
    
    # Timed runs
    run_times = []
    for i in range(runs):
        t0 = time.perf_counter()
        img = func(-2, 1, -1.5, 1.5, width, height, max_iter)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        run_times.append(elapsed_ms)
        print(f"  Run {i+1}: {elapsed_ms:.2f} ms")
    
    median_ms = np.median(run_times)
    times[name] = median_ms
    results[name] = img
    print(f"  Median: {median_ms:.2f} ms")