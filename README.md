# Mandelbrot Set Explorer

Name:  Akondeng Atengong Ketawah
Course: Numerical Scientific Computing 2026

# Implementations
- Naive Python
- NumPy
- Numba (float32, float64)
- Multiprocessing
- Dask Local
- Dask Cluster (Strato)
- GPU (PyOpenCL, float32 + float64)

# Run
```bash
conda activate nsc2026
python mandelbrot_parallel.py
python mandelbrot_opencl.py
pytest test_mandelbrot.py -v
