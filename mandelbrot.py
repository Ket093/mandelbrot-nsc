"""
Mandelbrot Set Generator
Author : [ Akondeng Atengong Ketawah ]
Course : Numerical Scientific Computing 2026
"""
def mandelbrot_point(c, max_iter=100):
    """
    Test if a point c is in the Mandelbrot set.

    Parameters
    ----------
    c : complex
        Complex number to test (like 0.5 + 0.3j)
    max_iter : int
        Maximum iterations to try

    Returns
    -------
    int
        Iteration count when escaped, or max_iter if in set
    """
    z = 0  # Start here

    # Try max_iter times
    for n in range(max_iter):
        # Check if number got too big
        if abs(z) > 2:
            return n  # Escaped after n tries

        # Calculate new z
        z = z*z + c

    # If we get here, never escaped
    return max_iter


# Simple test (after file execution)
if __name__ == "__main__":
    print("Testing mandelbrot_point function:")
    print("-" * 40)

    # Test 1: c = 0
    result1 = mandelbrot_point(0 + 0j, 100)
    print(f"c = 0: {result1} iterations")

    # Test 2: c = 1
    result2 = mandelbrot_point(1 + 0j, 100)
    print(f"c = 1: {result2} iterations")

    # Test 3: c = -1
    result3 = mandelbrot_point(-1 + 0j, 100)
    print(f"c = -1: {result3} iterations")

    print("\nFunction works! Ready for Step 3.")

import numpy as np

def compute_mandelbrot(xmin=-2.0, xmax=1.0,
                       ymin=-1.5, ymax=1.5,
                       width=100, height=100,
                       max_iter=100):
    """
    Create a grid of complex numbers and compute Mandelbrot for each.

    Parameters
    ----------
    xmin, xmax : float
        Left and right boundaries
    ymin, ymax : float
        Bottom and top boundaries
    width, height : int
        Grid size (columns, rows)
    max_iter : int
        Maximum iterations per point

    Returns
    -------
    numpy.ndarray
        2D array of iteration counts
    """
    # Create x values (real part)
    x_values = np.linspace(xmin, xmax, width)

    # Create y values (imaginary part)
    y_values = np.linspace(ymin, ymax, height)

    # Create empty 2D array for results
    results = np.zeros((height, width), dtype=int)

    # Loop through all points
    for row in range(height):
        y = y_values[row]

        for col in range(width):
            x = x_values[col]

            # Create complex number: c = x + yi
            c = x + 1j * y

            # Test this point
            iterations = mandelbrot_point(c, max_iter)

            # Store result (row, column)
            results[row, col] = iterations

    return results

import time

def test_grid_performance():
    """Test the grid function and measure performance."""
    print("\n" + "="*60)
    print("Step 3 & 4: Testing compute_mandelbrot and measuring time")
    print("="*60)

    # Test 1: Small grid (for quick testing)
    print("\nTest 1: Small grid (10x10)")
    start = time.time()
    small_grid = compute_mandelbrot(width=10, height=10, max_iter=100)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Grid shape: {small_grid.shape}")
    print(f"Min value: {small_grid.min()}, Max value: {small_grid.max()}")

    # Test 2: Medium grid
    print("\nTest 2: Medium grid (100x100)")
    start = time.time()
    medium_grid = compute_mandelbrot(width=100, height=100, max_iter=100)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Grid shape: {medium_grid.shape}")

    # Test 3: Large grid (as per exercise: 1024x1024)
    print("\nTest 3: Large grid (256x256) - smaller for testing")
    print("Note: 1024x1024 might take a while with this simple code")
    start = time.time()
    large_grid = compute_mandelbrot(width=256, height=256, max_iter=100)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Grid shape: {large_grid.shape}")

    return small_grid, medium_grid, large_grid


# Run the tests
if __name__ == "__main__":
    print("Testing mandelbrot_point function:")
    print("-" * 40)

    # Test single point function
    result1 = mandelbrot_point(0 + 0j, 100)
    print(f"c = 0: {result1} iterations")

    result2 = mandelbrot_point(1 + 0j, 100)
    print(f"c = 1: {result2} iterations")

    result3 = mandelbrot_point(-1 + 0j, 100)
    print(f"c = -1: {result3} iterations")

    # Test grid function and measure performance
    test_grid_performance()

    print("\n" + "="*60)
    print("Steps 3 & 4 completed successfully!")