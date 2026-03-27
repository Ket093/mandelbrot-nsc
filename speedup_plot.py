"""
Speedup Plot for MP2
"""

import matplotlib.pyplot as plt

# Data from M2 worker sweep
workers = [1, 2, 3, 4]
speedup = [0.86, 2.00, 1.50, 2.00]

# Ideal speedup (perfect scaling)
ideal = [1, 2, 3, 4]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(workers, speedup, 'o-', label='Actual speedup', linewidth=2, markersize=8, color='blue')
plt.plot(workers, ideal, '--', label='Ideal speedup (perfect scaling)', linewidth=2, color='red')