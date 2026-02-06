print('Course Package Check')
print('-------------------')
checks = [
    ('numpy', 'import numpy'),
    ('matplotlib', 'import matplotlib'),
    ('scipy', 'import scipy'),
    ('numba', 'import numba'),
    ('pytest', 'import pytest'),
    ('dask', 'import dask'),
    ('distributed', 'import distributed')
]
all_ok = True
for package_name, import_command in checks:
    try:
        exec(import_command)
        print(f'OK - {package_name}')
    except:
        print(f'FAIL - {package_name}')
        all_ok = False
print('-------------------')
if all_ok:
    print('PASS: All packages installed')
else:
    print('FAIL: Some packages missing')
