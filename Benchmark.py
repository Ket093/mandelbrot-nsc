import time
import numpy as np
from mandelbrot import mandelbrot_point, compute_mandelbrot


def numpy_mandelbrot(width, height, max_iter=100):
    x = np.linspace(-2, 1, width)
    y = np.linspace(-1.5, 1.5, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

# Measurement of naive version 3 times
print("Naive version:")
naive_times = []
for i in range(3):
    start = time.perf_counter()
    compute_mandelbrot(1024, 1024, 100)
    naive_times.append((time.perf_counter() - start) * 1000)
    print(f"  Run {i+1}: {naive_times[-1]:.2f} ms")

naive_times.sort()
naive_median = naive_times[1]
print(f"Median: {naive_median:.2f} ms\n")

# Measurement of numpy version 3 times
print("NumPy version:")
numpy_times = []
for i in range(3):
    start = time.perf_counter()
    numpy_mandelbrot(1024, 1024, 100)
    numpy_times.append((time.perf_counter() - start) * 1000)
    print(f"  Run {i+1}: {numpy_times[-1]:.2f} ms")

numpy_times.sort()
numpy_median = numpy_times[1]
print(f"Median: {numpy_median:.2f} ms")

print("\n" + "="*40)
print("MOODLE SUBMISSION:")
print(f"Naive: {naive_median:.2f} ms")
print(f"NumPy: {numpy_median:.2f} ms")
print("="*40)