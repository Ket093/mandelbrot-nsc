print("--- TEST COURSE PACKAGES----")
import numpy as np
import matplotlib.pyplot as plt
import scipy
import numba
import pytest
import dask
import distributed

print("✅ All packages imported successfully!")
print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Numba: {numba.__version__}")
print(f"Pytest: {pytest.__version__}")
print(f"Dask: {dask.__version__}")
print(f"Distributed: {distributed.__version__}")