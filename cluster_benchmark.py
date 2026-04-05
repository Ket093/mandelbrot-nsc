"""
Dask Cluster Benchmark at 4096x4096 - Self-contained
"""

import time
import statistics
import numpy as np
from numba import njit
from dask.distributed import Client
from mandelbrot_parallel import mandelbrot_chunk

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0:
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk_njit(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            c_real = x_min + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)
    return out

def mandelbrot_dask_simple(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    return np.vstack([mandelbrot_chunk_njit(*t) for t in tasks])

# Connect to cluster
client = Client("tcp://130.225.39.197:8786")
print(f"Connected: {client}")

N = 4096
max_iter = 100
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

print("=" * 50)
print("DASK CLUSTER BENCHMARK at 4096x4096")
print("=" * 50)

# Warm up
print("\nWarming up...")
client.run(lambda: None)

# Test chunk sizes
chunk_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

print(f"\n{'Chunks':>8} {'Time (s)':>10} {'Speedup':>10}")
print("-" * 35)

for n_chunks in chunk_sizes:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        future = client.submit(mandelbrot_dask_simple, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks)
        result = future.result()
        times.append(time.perf_counter() - t0)
    t_par = statistics.median(times)
    speedup = 1.574 / t_par
    print(f"{n_chunks:8d} {t_par:10.3f} {speedup:9.2f}x")

client.close()
