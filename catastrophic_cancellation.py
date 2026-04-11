"""
E2: Catastrophic Cancellation - Quadratic Formula
"""

import numpy as np

def quadratic_naive(a, b, c):
    t = type(a)
    disc = t(np.sqrt(b*b - t(4)*a*c))
    x1 = (-b + disc) / (t(2)*a)
    x2 = (-b - disc) / (t(2)*a)
    return x1, x2

def quadratic_stable(a, b, c):
    t = type(a)
    disc = t(np.sqrt(b*b - t(4)*a*c))
    if b > 0:
        x1 = (-b - disc) / (t(2)*a)
    else:
        x1 = (-b + disc) / (t(2)*a)
    x2 = c / (a * x1)
    return x1, x2

# Test polynomial: x^2 - 10000.0001*x + 1 = 0
# True roots: x1 ≈ 10000.0001, x2 ≈ 1e-4

true_small = 1.0 / 10000.0001

print("*" * 40)
print("E2: Catastrophic Cancellation")
print("*" * 40)

print("\n1. Naive Quadratic Formula:")
print("-" * 40)
for dtype in [np.float32, np.float64]:
    a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
    x1, x2 = quadratic_naive(a, b, c)
    print(f"{dtype.__name__}: x1 = {float(x1):.4f}, x2 = {float(x2):.10f}")

print("\n2. Stable Quadratic Formula (Vieta):")
print("-" * 40)
for dtype in [np.float32, np.float64]:
    a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
    x1, x2 = quadratic_stable(a, b, c)
    print(f"{dtype.__name__}: x1 = {float(x1):.4f}, x2 = {float(x2):.10f}")

print("\n3. Relative Error Comparison:")
print("-" * 40)
for dtype in [np.float32, np.float64]:
    a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
    _, x2_naive = quadratic_naive(a, b, c)
    _, x2_stable = quadratic_stable(a, b, c)
    err_naive = abs(float(x2_naive) - true_small) / true_small
    err_stable = abs(float(x2_stable) - true_small) / true_small
    print(f"{dtype.__name__}:")
    print(f"  Naive error:  {err_naive:.2e}")
    print(f"  Stable error: {err_stable:.2e}")
