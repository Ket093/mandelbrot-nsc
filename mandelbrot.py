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

# ===== SIMPLE VISUALIZATION =====
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