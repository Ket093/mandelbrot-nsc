"""
MILESTONE 2: Line-level profiling with line_profiler
Author: Akondeng Atengong Ketawah
Course: Numerical Scientific Computing 2026
"""

import numpy as np

# naive function with @profile decorator
@profile
def naive_mandelbrot_profiled(xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, width=512, height=512, max_iter=100):
    """naive Mandelbrot function for line profiling"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    results = np.zeros((height, width), dtype=int)
    
    for row in range(height):
        y_val = y[row]
        for col in range(width):
            x_val = x[col]
            c = x_val + 1j * y_val
            
            z = 0 + 0j
            for n in range(max_iter):
                if abs(z) > 2:
                    results[row, col] = n
                    break
                z = z*z + c
            else:
                results[row, col] = max_iter
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 2: Line Profiler")
    print("=" * 60)
    print("\nRunning function once...")
    print("(This will be slower as the profiler is observing)")
    
    # Run function once
    result = naive_mandelbrot_profiled(width=512, height=512, max_iter=50)
    
    print("\nFunction executed successfully!")
    print("\nProceed to run the command below:")
    print("   kernprof -l -v m2_lineprofile.py")