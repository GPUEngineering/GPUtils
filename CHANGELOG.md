# Changelog 

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- ---------------------
      v1.10.0
     --------------------- -->
## v1.10.0 - Unreleased

### Added

- `DTensor::getRowsInto(destination, rowsFrom, rowsTo, matIdx)` copies rows into a
  preallocated tensor without allocating or resizing; validates indices, destination shape,
  and non-overlapping storage, and enqueues the copy on the source tensor's stream.
- `DTensor::leastSquaresBatched(b, infoArray)` accepts a reusable device status buffer,
  avoiding GPUtils's per-call workspace allocation while retaining the original overload.
- `Session::numStreams()` exposes the initialized stream count.
- Standalone Release-mode GPU benchmarks for pointer-based versus strided batched GEMM
  and row-loop, 2D-copy, kernel, and actual `getRows` extraction paths. Includes configurable
  dimensions, correctness checks, preallocated and allocation-inclusive measurements,
  host/CUDA-event timings, CSV percentiles, and build/run documentation.
- Regression tests for setup validation, malformed files, dimension mismatches in checked
  operations, non-default GEMM coefficients, stream ordering, move semantics, reusable
  least-squares status buffers, and preallocated row extraction.

### Changed

- Column-major uploads copy directly from the supplied host vector; only row-major uploads
  allocate a conversion buffer.
- `setStreamIdx` returns a reference for lvalues and moves rvalues into the returned tensor,
  eliminating the previous implicit device allocation and deep copy.
- `getRows` uses one asynchronous 2D device copy instead of a cuBLAS call per row and
  returns a tensor with the source's stream index. It shares its copy implementation with
  `getRowsInto`; caller-supplied destinations retain their own stream index.
- Explicit tensor allocations require positive dimensions that fit CUDA's integer dimensions
  and reject allocation-size overflow; default construction remains available for placeholders.
  Vector constructors validate data length and storage mode before device allocation.
- Added validation for reshape sizes, slice axes/ranges, random bounds, storage modes,
  row extraction, debug element access, empty factoriser inputs, and QR output batch shapes.
- Stream setup rejects zero streams and reconfiguration after session initialization;
  stream selection validates against the actual initialized count.
- Documented unchecked numerical-operation preconditions, slice and asynchronous-buffer
  lifetimes, stream dependencies, reusable workspace usage, and binary-format assumptions.
- Ignore generated CMake/CTest files, fetched dependencies, build products, and temporary
  file-I/O test artifacts, including those produced by in-source builds.

### Fixed

- Enable testing at the project root so `ctest --test-dir build` discovers the test suite.
- Make the Python-format binary-reader test self-contained, removing its dependency on
  separately generated Python test files and a particular working directory.
- Reject malformed text headers/data and incomplete or incorrectly sized binary files before
  constructing tensors; reject extra payload data and report file-write failures. Handle short
  file names safely and avoid writes into empty text-data vectors and uninitialized binary reads.
- Avoid null-pointer arithmetic when slicing a single-matrix tensor along the matrix axis.
- QR workspace queries, factorisation, and least-squares/Q extraction use the matrix-selected
  cuSolver handle consistently with cuBLAS. The debugging `getQR` method synchronizes that
  stream before reading the triangular factor on the host.
- Givens coefficient computation runs on the selected cuBLAS stream before the rotation,
  checks kernel launch errors, and restores the caller's previous cuBLAS pointer mode.
- Configure test streams independently of test execution order and add nonblocking-stream
  regressions with delayed inputs and downstream consumers for QR, Givens, and row extraction.
- Correct the row-slicing test to select a column before slicing its rows, rather than using
  flat storage offsets as row indices.


<!-- ---------------------
      v1.9.0
     --------------------- -->
## v1.9.0 - 28-03-2025

### Added

- Multiple streams supported


<!-- ---------------------
      v1.8.0
     --------------------- -->
## v1.8.0 - 21-03-2025

### Added

- Support for counting total allocated memory via `Session::getInstance().totalAllocatedBytes()` 


<!-- ---------------------
      v1.7.2
     --------------------- -->
## v1.7.2 - 6-12-2024

### Fixed

- Using alias instead of email in `setup.py`


<!-- ---------------------
      v1.7.1
     --------------------- -->
## v1.7.1 - 4-12-2024

### Fixed

- Compatibility between Python and C++ in how the data is stored in bt files


<!-- ---------------------
      v1.7.0
     --------------------- -->
## v1.7.0 - 3-12-2024

### Added

- The project now uses the GNU General Public License (GPL) v3; license file added
- Introduces new Python package for storing and loading numpy arrays; in can be installed with `pip install gputils-api`; 
  unit tests and documentation

### Fixed

- When compiling with `cmake`, the unit tests will not be compiled by default unless the flag `GPUTILS_BUILD_TEST` is set
- Clang clippy recommendations applied
- Proper error handling when binary tensor file is not found  


<!-- ---------------------
      v1.6.0
     --------------------- -->
## v1.6.0 - 2-12-2024

### Added

- Method `saveToFile` becomes `DTensor<T>::saveToFile(std::string pathToFile, Serialisation ser)`, i.e., the user can 
  choose whether to save the file as a text (ASCII) file, or a binary one

### Changed

 - Method `parseFromTextFile` renamed to `parseFromFile` (supports text and binary formats)


<!-- ---------------------
      v1.5.2
     --------------------- -->
## v1.5.2 - 1-12-2024

### Fixed

- Quick bug bix in `DTensor::parseFromTextFile` (passing storage mode to `vectorFromTextFile`)


<!-- ---------------------
      v1.5.1
     --------------------- -->
## v1.5.1 - 29-11-2024

### Fixed

- Set precision in `DTensor::saveToFile` properly
- `DTensor<T>::parseFromFile` throws `std::invalid_argument` if `T` is unsupported

<!-- ---------------------
      v1.5.0
     --------------------- -->
## v1.5.0 - 27-11-2024

### Added

- Created methods for serialising and deserialising `DTensor` objects


<!-- ---------------------
      v1.4.0
     --------------------- -->
## v1.4.0 - 25-11-2024

### Changed

- (Breaking change) The methods `CholeskyFactoriser::factorise`, `CholeskyFactoriser::solve`, `QRFactoriser::factorise`,
  `QRFactoriser::leastSquares`, and `QRFactoriser::getQR` are now `void` and do not
  return a status code. Instead, a status code is returned by called `statusCode`.
  This change leads to a reduction in data being downloaded from the GPU.
- In `Svd` a status code (`bool`) is returned from `Svd<double>::factorise` only if the
  `#GPUTILS_DEBUG_MODE` is defined, otherwise, the method returns always `true`.
- New base class `IStatus` used for a universal implementation of `info()`


<!-- ---------------------
      v1.3.2
     --------------------- -->
## v1.3.2 - 8-11-2024

### Fixed

- When slicing a `DTensor` along `axis=2`, update the pointer to matrices
- We got rid of warning `DTensor<T>::createRandomTensor` 


<!-- ---------------------
      v1.3.1
     --------------------- -->
## v1.3.1 - 7-11-2024

### Fixed

- Memory management improvements: we got rid of `pointerToMatrices`, which would unnecessarily allocate memory and `addAB` does not allocate any new memory internally.

<!-- ---------------------
      v1.3.0
     --------------------- -->
## v1.3.0 - 11-10-2024 

### Added

- Left/right Givens rotations
- `GivensAnnihilator` implemented


<!-- ---------------------
      v1.2.1
     --------------------- -->
## v1.2.1 - 07-10-2024

### Added

- Patch initialisation of Q in QR decomposition.
- Add test for tall skinny matrices.

<!-- ---------------------
      v1.2.0
     --------------------- -->
## v1.2.0 - 04-10-2024

### Added

- Implementation and test of QR factorisation for tall or square matrices.
- Solve least-square problems with QR factorisation.
- Improve documentation.

<!-- ---------------------
      v1.1.0
     --------------------- -->
## v1.1.0 - 03-08-2024

### Added

- Implementation and test of methods `.maxAbs()` and `.minAbs()` for any tensor.

<!-- ---------------------
      v1.0.0
     --------------------- -->
## v1.0.0 - 29-05-2024

### Added

- Support for random tensors
- Implementation of `CholeskyMultiFactoriser` which performs multiple Cholesky factorisations in parallel

### Changed

- Using a function `numBlocks` instead of the macro `DIM2BLOCKS`
- Using `TEMPLATE_WITH_TYPE_T` and `TEMPLATE_CONSTRAINT_REQUIRES_FPX` for the code to run on both C++17 and C++20

<!-- ---------------------
      v0.1.0
     --------------------- -->
## v0.1.0 - 23-04-2024

### Added

- Implementation and test of `Nullspace(DTensor A)` method `.project(DTensor b)`
- `project` will project in place each `bi` onto the nullspace of `Ai`

<!-- ---------------------
      v0.0.0
     --------------------- -->
## v0.0.0 - 20-04-2024

### Added

- Implementation of `DTensor<T>`, which is our basic entity for data storage and maniputation (supports basic linear algebra using `cublas` and `cusolver`); implementation of `=+`, `-=`, `*=` (for scalars and other tensors), `+`, `-`, `*` (scalars and tensors), printing (using `std::cout <<`), computation of norms (Frobenius and sum of absolute values of all elements); device vectors and matrices are tensors  
- Singular value decomposition using `cublas`
- Least-squares on tensors
- Computation of nullspace matrices (on tensor objects)
- Cholesky factorisation 
- Set up unit tests, CI, and CHANGELOG
