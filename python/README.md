## GPUtils API

### Installation

As simple as...
```bash
pip install gputils-api
```
of course, preferably from within a virtual environment.

### Write to file

```python
import numpy as np
import gputils_api as g
a = np.eye(3)
g.write_array_to_gputils_binary_file(a, 'my_data.bt')
```

### Read from file

```python
import numpy as np
import gputils_api as g
x = g.read_array_from_gputils_binary_file('my_data.bt')
```

### Tests

From the repository root, run all Python tests with:

```bash
./scripts/run_python_tests.sh
```

The script uses `build/test-venv/bin/python` if available, otherwise `python3`.
Set `PYTHON` to choose an interpreter with NumPy and pytest installed. It honours
`GPUTILS_BUILD_DIR` and `GPUTILS_FILE_EXCHANGE` and accepts extra pytest arguments,
for example `./scripts/run_python_tests.sh -k round_trip`.

To invoke pytest directly:

```bash
PYTHONPATH=python python -m pytest python/test -v
```

Tests use temporary directories, deterministic values, and full-array comparisons.
They cover shapes and precisions, non-contiguous inputs, independent binary-layout
fixtures, malformed files, and filesystem errors. Empty and zero-dimensional arrays
are expected to raise `ValueError`; one-element vectors remain valid inputs.

The reader validates the complete 24-byte header, positive dimensions, NumPy size
limits, and exact payload byte count before loading array data. It rejects both
truncated payloads and trailing bytes with `ValueError`. The writer validates inputs
before opening the output file, so invalid inputs do not create or overwrite files.
Object-containing and zero-size dtypes are rejected by both functions.

Actual C++/Python file-exchange tests need the test-only `gputils_file_exchange`
executable. Build with `GPUTILS_BUILD_TEST=ON`; the tests find it automatically at
`build/test/gputils_file_exchange`. For another build directory, set:

```bash
GPUTILS_FILE_EXCHANGE=/absolute/path/to/build/test/gputils_file_exchange \
  PYTHONPATH=python python -m pytest python/test -v
```

Without a built helper, only the C++ exchange tests are skipped. An explicitly
configured missing helper is an error. `scripts/build_and_test.sh` builds the helper
and sets its path, so these tests run as part of the combined suite.

The format stores three little-endian `uint64` dimensions, followed by consecutive
column-major matrices. It does not store a dtype: pass the matching `dt` to the
reader. Same-width dtype mistakes cannot reliably be detected. Python can preserve
explicit payload byte orders when supplied the matching dtype; C++ interoperability
tests use little-endian float32/float64 data on a little-endian host. Read results
always have three axes, padding vector and matrix shapes with singleton dimensions.


