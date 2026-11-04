# Changelog 

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- ---------------------
      v1.10.0
     --------------------- -->
## v1.10.0 - Unreleased

### Added

- `LeastSquaresBatchedWorkspace<T>` preallocates reusable status storage and one-entry matrix-pointer arrays only for batches of one.
- `Svd::rankRelative(rtol, atol = 0)` adds GPU rank calculation using per-matrix thresholds `max(atol, rtol * sigmaMax)` and validated tolerances.
- Build/test and CUDA memory-check scripts run C++/Python tests, repeated leak checks, and save sanitizer logs.
- Size-safe `Session::recordAllocation(size_t)`, `recordDeallocation(size_t)`, and counted `cudaRelease` methods.
- `DTensor::getRowsInto(destination, rowsFrom, rowsTo, matIdx)` validates and copies rows into preallocated storage on the source stream.
- `DTensor::leastSquaresBatched(b, infoArray)` supports reusable device status storage without per-call allocation.
- `Session::numStreams()` exposes the initialized stream count.
- Configurable GPU benchmarks compare batched GEMM and row-extraction methods, with correctness checks, host/GPU timings, and CSV results.
- Regression tests cover validation, files, GEMM coefficients, stream ordering, ownership, rank tolerances, and reusable buffers.

### Changed

- **Breaking:** replace `incrementAllocatedBytes(int)` with separate size-safe allocation/deallocation methods.
- **Breaking:** copy assignment deep-copies; move assignment transfers storage and empties the source. Use slices for aliases and `deviceCopyTo` for preallocated copies.
- Reshape retains and reuses owned matrix-pointer capacity where possible.
- Column-major uploads use the supplied host vector directly; only row-major uploads allocate a conversion buffer.
- `setStreamIdx` returns a reference for lvalues and moves rvalues, avoiding implicit allocation and deep copying.
- `getRows` uses one asynchronous 2D copy and inherits the source stream; `getRowsInto` preserves the destination's stream index.
- Tensor constructors validate dimensions, allocation overflow, vector length, and storage mode before allocating.
- Validate reshape sizes, slices, random bounds, row extraction, debug access, empty factoriser inputs, and QR output shapes.
- Reject zero streams and reconfiguration after initialization; validate stream indices against the initialized count.
- Document numerical preconditions, buffer lifetimes, stream dependencies, reusable workspaces, and binary-format assumptions.
- Ignore generated build files, dependencies, binaries, and temporary test artifacts, including in-source builds.

### Fixed

- Batched Cholesky and least-squares support single matrices using constructor-allocated pointer storage without allocations during reusable solves.
- Size QR workspace for the largest requirement across factorisation, least-squares, and Q extraction; add large square/tall regressions for both precisions.
- Share stream configuration across translation units using an inline static `Session` member, with a two-file regression test.
- Prevent SVD constructor exception leaks using a `unique_ptr` owner and raw observer; verify cleanup with allocation-failure tests.
- Tensor constructors release GPU storage on subsequent host-allocation failures; fault-injection tests verify cleanup for float, double, and int.
- Session shutdown completes pending work, releases library handles, and destroys owned CUDA streams.
- Single and batched Cholesky queries, factorisations, solves, and status tensors use the matrix-selected stream.
- Host transfers, scalar reads, initialization, and pointer uploads complete on their selected stream; transpose results inherit the source stream.
- Device copies wait for prior destination work and complete on the source stream before returning.
- QR debugging-helper uploads complete on the selected solver stream before returning, fixing intermittent stream-ordering failures.
- Track owned bytes and pointer capacity independently of shape, count successful allocations/frees, and preserve accounting across reshape, views, and moves.
- Reset `Svd::rank()` counters on each call; SVD factorisation, rank kernels, and results use the input stream.
- Assignment releases previous storage and safely handles metadata, self-assignment, and empty tensors; deep copies complete on the source stream.
- Enable testing at the project root so `ctest --test-dir build` discovers the test suite.
- Make the Python-format binary-reader test independent of generated Python files and the working directory.
- Reject malformed text and incomplete or incorrectly sized binary files; report write failures and handle short filenames and empty data safely.
- Avoid null-pointer arithmetic when slicing a single-matrix tensor along the matrix axis.
- QR operations use the matrix-selected cuSolver/cuBLAS stream; `getQR` synchronizes before reading R on the host.
- Givens coefficient kernels use the selected cuBLAS stream, check launch errors, and restore the previous pointer mode.
- Make test-stream setup independent of execution order and add nonblocking-stream regressions for QR, Givens, and row extraction.
- Correct the row-slicing test to select a column before slicing its rows.


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
- Python package for storing and loading NumPy arrays, with tests and documentation; install with `pip install gputils-api`.

### Fixed

- When compiling with `cmake`, the unit tests will not be compiled by default unless the flag `GPUTILS_BUILD_TEST` is set
- Clang clippy recommendations applied
- Proper error handling when binary tensor file is not found  


<!-- ---------------------
      v1.6.0
     --------------------- -->
## v1.6.0 - 2-12-2024

### Added

- `DTensor<T>::saveToFile(std::string pathToFile, Serialisation ser)` supports text (ASCII) and binary output.

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

- **Breaking:** Cholesky factorisation/solve and QR factorisation/least-squares/extraction return `void`; query status separately to reduce GPU downloads.
- `Svd<double>::factorise` returns a status only with `GPUTILS_DEBUG_MODE`; otherwise it always returns `true`.
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

- Remove unnecessary `pointerToMatrices` allocations; `addAB` no longer allocates memory internally.

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

- `DTensor<T>` provides GPU tensor storage, arithmetic, printing, and norms using cuBLAS and cuSolver; includes device vectors and matrices.
- Singular value decomposition using `cublas`
- Least-squares on tensors
- Computation of nullspace matrices (on tensor objects)
- Cholesky factorisation 
- Set up unit tests, CI, and CHANGELOG
