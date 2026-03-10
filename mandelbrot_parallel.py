"""
Mandelbrot Set Generator - Parallel Version
Author : Akondeng Atengong Ketawah
Course : Numerical Scientific Computing 2026
"""

import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import os
import statistics
import matplotlib.pyplot as plt
from pathlib import Path