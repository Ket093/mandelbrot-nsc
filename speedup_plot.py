"""
Speedup Plot for MP2
"""

import matplotlib.pyplot as plt

# Data from your M2 worker sweep
workers = [1, 2, 3, 4]
speedup = [0.86, 2.00, 1.50, 2.00]  # From your earlier benchmark

# Ideal speedup (perfect scaling)
ideal = [1, 2, 3, 4]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(workers, speedup, 'o-', label='Actual speedup', linewidth=2, markersize=8, color='blue')
plt.plot(workers, ideal, '--', label='Ideal speedup (perfect scaling)', linewidth=2, color='red')

# Labels and title
plt.xlabel('Number of workers')
plt.ylabel('Speedup (times faster than serial)')
plt.title('Mandelbrot Parallel Speedup: Actual vs Ideal')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Set axis limits
plt.xlim(0.5, 4.5)
plt.ylim(0, 5)

# Save the plot
plt.savefig('speedup_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Speedup plot saved as 'speedup_plot.png'")