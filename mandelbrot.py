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
