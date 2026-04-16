"""
Test suite for Mandelbrot implementation
"""

import pytest
import numpy as np
from mandelbrot_parallel import mandelbrot_pixel, mandelbrot_serial

# Known test cases: (c_real, c_imag, max_iter, expected)
KNOWN_CASES = [
    (0.0, 0.0, 100, 100),     # origin: never escapes, returns max_iter
    (5.0, 0.0, 100, 1),       # far outside: escapes on iteration 1
    (-2.5, 0.0, 100, 1),      # left tip of Mandelbrot set
    (0.25, 0.0, 100, 100),    # inside the set
]

@pytest.mark.parametrize("c_real, c_imag, max_iter, expected", KNOWN_CASES)
def test_mandelbrot_pixel(c_real, c_imag, max_iter, expected):
    """Test mandelbrot_pixel against known values."""
    result = mandelbrot_pixel(c_real, c_imag, max_iter)
    assert result == expected, f"c={c_real}+{c_imag}j: expected {expected}, got {result}"


def test_serial_vs_known():
    """Test that serial grid matches known pixel values."""
    N = 32
    result = mandelbrot_serial(N, -2.5, 1.0, -1.25, 1.25, 100)
    
    assert result[0, 0] == 1
    assert result[N-1, N-1] == 1
    assert result[N//2, N//2] > 0
