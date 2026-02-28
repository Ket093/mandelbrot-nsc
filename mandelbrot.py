"""
Mandelbrot Set Generator
Author : [Akondeng Atengong Ketawah]
Course : Numerical Scientific Computing 2026
"""

def mandelbrot_point(c, max_iter=100):
    """Test if a point is in the Mandelbrot set."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

import numpy as np
import time

def compute_mandelbrot(width=100, height=100, max_iter=100):
    """Create a grid of Mandelbrot values."""
    x_vals = np.linspace(-2, 1, width)
    y_vals = np.linspace(-1.5, 1.5, height)

    results = np.zeros((height, width), dtype=int)

    for row in range(height):
        y = y_vals[row]
        for col in range(width):
            x = x_vals[col]
            c = x + 1j * y
            results[row, col] = mandelbrot_point(c, max_iter)

    return results

 # NUMPY VECTORIZED VERSION (FAST!)
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
     #SIMPLE VISUALIZATION 
import matplotlib.pyplot as plt

print("Creating Mandelbrot image...")
grid = compute_mandelbrot(width=300, height=200, max_iter=50)

plt.figure(figsize=(8, 6))
plt.imshow(grid, cmap='hot', extent=[-2, 1, -1.5, 1.5])
plt.colorbar()
plt.title('Mandelbrot Set')
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.savefig('mandelbrot.png')
plt.show()

print("Image saved as mandelbrot.png")