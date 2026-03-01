"""
MILESTONE 1: Function-level profiling with cProfile
Author: Akondeng Atengong Ketawah
Course: Numerical Scientific Computing 2026
"""

import cProfile
import pstats
import sys
import os

# Add parent directory to import mandelbrot.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions
from mandelbrot import compute_mandelbrot, numpy_mandelbrot

print("=" * 60)
print("MILESTONE 1: cProfile Analysis")
print("=" * 60)

# Profile naive version
print("\n1. Profiling Naive version (512x512)...")
cProfile.run('compute_mandelbrot(width=512, height=512, max_iter=100)', 'naive_profile.prof')

# Profile NumPy version
print("2. Profiling NumPy version (512x512)...")
cProfile.run('numpy_mandelbrot(width=512, height=512, max_iter=100)', 'numpy_profile.prof')

# Display naive results
print("\n" + "=" * 50)
print("NAIVE VERSION - Top 10 functions by cumulative time")
print("=" * 50)
stats = pstats.Stats('naive_profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)

# Display NumPy results
print("\n" + "=" * 50)
print("NUMPY VERSION - Top 10 functions by cumulative time")
print("=" * 50)
stats = pstats.Stats('numpy_profile.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)

print("\nDone! Milestone 1 complete!")
print("Copy output above to report.")