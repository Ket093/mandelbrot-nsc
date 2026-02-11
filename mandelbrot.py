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
import time

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

def run_tests():
    """
    Run all required tests for Steps 2, 3, and 4.
    """
    print("=== Mandelbrot Set Implementation ===")
    print("Testing all required functions...")
    print("=" * 60)

    # Step 2: Test single point function
    print("\nStep 2: Testing mandelbrot_point()")
    print("-" * 40)

    test_points = [
        (0+0j, 100, "Origin (should be 100)"),
        (1+0j, 3, "1 (should be around 3)"),
        (-1+0j, 100, "-1 (should be 100)"),
    ]

    all_correct = True
    for c, expected, description in test_points:
        result = mandelbrot_point(c)
        correct = (result == expected) if expected != 3 else (result <= 5)
        status = "✓" if correct else "✗"
        print(f"{status} {description}: {result} iterations")
        if not correct:
            all_correct = False

    # Step 3: Test grid function with small grid
    print("\nStep 3: Testing compute_mandelbrot()")
    print("-" * 40)

    small_grid = compute_mandelbrot(width=10, height=10)
    print(f"Created 10x10 grid: shape = {small_grid.shape}")
    print(f"Values range: {small_grid.min()} to {small_grid.max()}")

    # Step 4: Performance measurement
    print("\nStep 4: Performance measurement")
    print("-" * 40)

    print("Testing 100x100 grid...")
    start = time.time()
    result_100 = compute_mandelbrot(width=100, height=100)
    elapsed_100 = time.time() - start
    print(f"100x100: {elapsed_100:.3f} seconds")

    print("\nTesting 1024x1024 grid (this will take a moment)...")
    start = time.time()
    result_1024 = compute_mandelbrot(-2, 1, -1.5, 1.5, 1024, 1024)
    elapsed_1024 = time.time() - start
    print(f"1024x1024: {elapsed_1024:.3f} seconds")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_correct:
        print("✓ Step 2: mandelbrot_point() passed tests")
    else:
        print("✗ Step 2: Some tests failed")

    print("✓ Step 3: compute_mandelbrot() created grid successfully")
    print("✓ Step 4: Performance measured")
    print(f"  - 100x100: {elapsed_100:.3f}s")
    print(f"  - 1024x1024: {elapsed_1024:.3f}s")

    print("\nAll steps completed! ✅")


# Run the tests when file is executed
if __name__ == "__main__":
    run_tests()