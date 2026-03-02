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
# Accuracy comparison
print("\n" + "+" * 60)
print("ACCURACY COMPARISON (float32 vs float64)")
print("+" * 60)

ref = results['float64']
comp = results['float32']
total_pixels = width * height

diff = np.abs(comp - ref)
max_diff = diff.max()
diff_pixels = (diff > 0).sum()
percent_diff = (diff_pixels / total_pixels) * 100

print(f"\nfloat32 vs float64:")
print(f"  Max difference: {max_diff}")
print(f"  Different pixels: {diff_pixels} / {total_pixels} ({percent_diff:.4f}%)")
if max_diff == 0:
    print(f"  Identical to float64")
elif max_diff < 5:
    print(f"  Very minor differences (still acceptable for Mandelbrot)")

# Visual comparison
print("\n" + "+" * 60)
print("GENERATING VISUAL COMPARISON")
print("+" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot float64
axes[0].imshow(results['float64'], cmap='hot', extent=[-2, 1, -1.5, 1.5])
axes[0].set_title(f"float64\n{times['float64']:.1f} ms", fontsize=12)
axes[0].set_xlabel('Real')
axes[0].set_ylabel('Imaginary')

# Plot float32
axes[1].imshow(results['float32'], cmap='hot', extent=[-2, 1, -1.5, 1.5])
axes[1].set_title(f"float32\n{times['float32']:.1f} ms", fontsize=12)
axes[1].set_xlabel('Real')
axes[1].set_ylabel('Imaginary')

plt.tight_layout()
plt.savefig('m4_precision_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: m4_precision_comparison.png")

# Final summary
print("\n" + "+" * 50)
print("RESULTS - To be used for report")
print("+" * 50)
print(f"\n{'Data Type':<12} {'Median Time (ms)':<18} {'Speedup':<12}")
print("+" * 50)

for name in ['float64', 'float32']:
    if name in times:
        if name == 'float64':
            print(f"{name:<12} {times[name]:<18.2f} {'1.00x':<12}")
        else:
            speedup = times['float64'] / times[name]
            print(f"{name:<12} {times[name]:<18.2f} {speedup:<12.2f}x")

# Recommendation
print("\n" + "+" * 50)
print("RECOMMENDATION")
print("+" * 50)

speedup = times['float64'] / times['float32']
diff = np.abs(results['float32'] - results['float64']).max()

if diff == 0:
    print(f"float32 is {speedup:.2f}x faster with no quality loss - USE THIS")
elif diff < 5 and speedup > 1.05:
    print(f"float32 is {speedup:.2f}x faster with very minor differences - RECOMMENDED")
elif diff < 5 and speedup <= 1.05:
    print(f"float32 shows no significant speedup ({speedup:.2f}x) with very minor differences - stick with float64")
else:
    print(f"float32 has visible differences ({diff} max diff) - stick with float64")

print("\nMilestone 4 is now complete!")