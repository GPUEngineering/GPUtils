# GPUtils

## 1. DTensor

The `DTensor` class is for manipulating data on a GPU. 
It manages their memory and facilitates various algebraic operations.

A tensor has three axes: `[rows (m) x columns (n) x matrices (k)]`.
An (m,n,1)-tensor stores a _matrix_, and an (m,1,1)-tensor stores a _vector_.

Explicit tensor dimensions must be positive and fit in the integer dimensions
used by CUDA's BLAS and solver interfaces. A default-constructed tensor is an
uninitialized placeholder. Tensor sizes and allocation sizes are checked for
overflow. Slices are non-owning views: their source storage must remain alive
and must not be reallocated while the view is used.

Allocating constructors and the deep-copy constructor clean up acquired device
storage if a C++ exception occurs during construction, then rethrow the original
exception. For example, a host conversion-buffer allocation failure propagates
`std::bad_alloc` without retaining the constructor's GPU allocations. This does
not change `gpuErrChk`'s existing process-termination policy for CUDA errors.

Numerical operations intended for repeated use may leave operand and index
validation to the caller. In particular, `addAB` requires compatible matrix
and batch dimensions and correctly sized output, `+=` and `-=` require matching
tensor shapes, and Givens rotation indices must be in range. Validate these
preconditions during application setup before entering time-critical code.

We first need to decide on a data type between `float` or `double`.
We will use `float` in the following examples.

### 1.1. Vectors

The simplest way to create an empty `DTensor` object is by constructing a vector:

```c++
size_t n = 100;
DTensor myTensor(n);
```

> [!IMPORTANT] 
> This creates an n-dimensional vector as an (n,1,1)-tensor on the device.

A `DTensor` can be instantiated from host memory:

```c++
std::vector<float> h_a{4., -5., 6., 9., 8., 5., 9., -10.2, 9., 11.};
DTensor<float> myTensor(h_a, h_a.size());
std::cout << myTensor << "\n";
```

> [!CAUTION]
> Printing a `DTensor` to `std::cout` will slow down your program
> (it requires the data to be downloaded from the device).
> Printing was designed for quick debugging.

We will often need to create slices (or shallow copies) of a `DTensor` 
given a range of values. We can then do:

```c++
size_t axis = 0;  // rows=0, cols=1, mats=2
size_t from = 3;
size_t to = 5;
DTensor<float> mySlice(myTensor, axis, from, to);
std::cout << mySlice << "\n";
```

Sometimes we need to reuse an already allocated `DTensor` by uploading 
new data from the host by using the method `upload`. Here is a short example:

```c++
std::vector<float> h_a{1., 2., 3.};  // host data a
DTensor<float> myVec(h_a, 3);  // create vector in tensor on device
std::vector<float> h_b{4., -5., 6.};  // host data b
myVec.upload(h_b);
std::cout << myVec << "\n";
```

We can upload some host data to a particular position of a `DTensor` as follows:

```c++
std::vector<float> hostData{1., 2., 3.};
// here, `true` tells the constructor to set all allocated elements to zero
DTensor<float> x(7, 1, 1, true);  // x = [0, 0, 0, 0, 0, 0, 0]'
DTensor<float> mySlice(x, 0, 3, 5); 
mySlice.upload(hostData);
std::cout << x << "\n";  // x = [0, 0, 0, 1, 2, 3, 0]'
```

If necessary, the data can be downloaded from the device to the host using 
`download`.

Very often we will also need to copy data from an existing `DTensor`
to another `DTensor` (without passing through the host).
To do this we can use `deviceCopyTo`. Here is an example:

```c++
DTensor<float> x(10);
DTensor<float> y(10);
x.deviceCopyTo(y);  // x ---> y (device memory to device memory)
```

Copy construction and copy assignment create independent owning storage:

```c++
DTensor<float> myCopy(existingTensor);
myCopy = existingTensor;             // deep copy, replacing myCopy's allocation
myCopy = std::move(existingTensor);  // transfer storage; existingTensor becomes empty
```

**Compatibility change:** copy assignment previously aliased the source. It now
matches the copy constructor's deep-copy semantics. Copies preserve the source's
shape and stream index, and complete on the source's stream before returning.
Self-copy and self-move assignment are safe no-ops; empty and moved-from tensors
can be assigned again. Moving an owning tensor transfers its allocation without
copying device data. Replacing an owning destination releases its old allocation
and invalidates views referring to it.

For intentional aliasing, use the slice constructor explicitly. Slices remain
non-owning: their backing allocation must outlive them. Copying a slice produces
independent owning storage; moving a slice preserves its non-owning nature and
does not extend the backing allocation's lifetime. In particular, do not move a
slice into the tensor that owns its backing allocation; copy it instead.
Assigning into a slice replaces that view rather than writing through it.

For repeated operations requiring no allocation, preallocate the destination
and use `existingTensor.deviceCopyTo(myCopy)` instead of copy assignment.
Ensure previous asynchronous uses of a destination have completed before
replacing its storage, and establish dependencies for source writes from other
streams. Move assignment can free old storage; it is not a guarantee of bounded
real-time latency.

Lastly, a not so efficient method that should only be used for 
debugging, if at all, is the `()` operator (e.g., `x(i, j, k)`), which fetches
one element of the `DTensor` to the host.
This cannot be used to set a value, so don't do anything like `x(0, 0, 0) = 4.5`!
> [!CAUTION]
> For the love of god, do not put this `()` operator in a loop.

### 1.2. Computation of scalar quantities

The following scalar quantities can be computed (internally, 
we use `cublas` functions):

- `.normF()`: the Frobenius norm of a tensor $x$, using `nrm2` (i.e., the 2-norm, or Euclidean norm, if $x$ is a vector)
- `.sumAbs()`: the sum of the absolute of all the elements, using `asum` (i.e., the 1-norm if $x$ is a vector)

### 1.3. Some cool operators

We can element-wise add `DTensor`s on the device as follows:

```c++
std::vector<float> host_x{1., 2., 3., 4., 5., 6.,  7.};
std::vector<float> host_y{1., 3., 5., 7., 9., 11., 13.};
DTensor<float> x(host_x, host_x.size());
DTensor<float> y(host_y, host_y.size());
x += y;  // x = [2, 5, 8, 11, 14, 17, 20]'
std::cout << x << "\n";
```

To element-wise subtract `y` from `x` we can use `x -= y`.

We can also scale a `DTensor` by a scalar with `*=` (e.g, `x *= 5.0f`). 
To negate the values of a `DTensor` we can do `x *= -1.0f`.

We can also compute the inner product (as a (1,1,1)-tensor) of two vectors as follows:

```c++
std::vector<float> host_x{1., 2., 3., 4., 5., 6.,  7.};
std::vector<float> host_y{1., 3., 5., 7., 9., 11., 13.};
DTensor<float> xtr(host_x, 1, host_x.size());  // column vector
DTensor<float> y(host_y, host_y.size());  // row vector
DTensor<float> innerProduct = x * y;
```

If necessary, we can also use the following element-wise operations

```c++
DTensor<float> x(host_x, host_x.size());  // row vector
auto sum = x + y;
auto diff = x - y;
auto scaledX = 3.0f * x;
```

### 1.4. Matrices

To store a matrix in a `DTensor` we need to provide the data in an array; 
we can use either column-major (default) or row-major format.
```TODO implement row-major```
Suppose we need to store the matrix 

$$A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9 \\
10 & 11 & 12 \\
13 & 14 & 15
\end{bmatrix},$$

where this data is stored in row-major format.
Then, we do
```c++
size_t rows = 5;
size_t cols = 3;
std::vector<float> h_data{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f,
                          7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f,
                          13.0f, 14.0f, 15.0f};
DTensor<float> myTensor(h_data, rows, cols, 1, rowMajor);
```

Choose `rowMajor` or `columnMajor` as appropriate.

We can also preallocate memory for a `DTensor` as follows:

```c++
DTensor<float> a(rows, cols, 1);
```

Then, we can upload the data as follows:

```c++
a.upload(h_data, rowMajor);
```

The copy constructor has also been implemented; 
to hard-copy a vector just do 
`DTensor<float> myCopy(existingTensor)`.

The number of rows and columns of a `DTensor` can be 
retrieved using the methods `.numRows()` and `.numCols()` respectively.

### 1.5. More operations

The operators `+=` are `-=` supported for device matrices.

Matrix-matrix multiplication is as simple as:

```c++
size_t m = 2, k = 3, n=5;
std::vector<float> aData{1.0f,  2.0f,  3.0f,
                         4.0f,  5.0f,  6.0f};
std::vector<float> bData{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
                         6.0f,  7.0f,  8.0f,  9.0f, 10.0f,
                         11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
DTensor<float> A(aData, m, k, 1, rowMajor);
DTensor<float> B(bData, k, n, 1, rowMajor);
auto X = A * B;
std::cout << A << B << X << "\n";
```

### 1.6. Tensors

As you would expect, all operations mentioned so far are supported by actual tensors
as batched operations (that is, (m,n)-matrix-wise).

Also, we can create the transposes of a `DTensor` using `.tr()`.
This transposes each (m,n)-matrix and stores it in a new `DTensor`
at the same k-index.
Transposition in-place is not possible.

### 1.7. Least squares

The solution of least squares has been implmented as a tensor method.
Say we want to solve `A\b` using least squares.
We first create $A$ and $b$

```c++
size_t m = 4;
size_t n = 3;
std::vector<float> aData{1.0f, 2.0f, 4.0f,
                         2.0f, 13.0f, 23.0f,
                         4.0f, 23.0f, 77.0f,
                         6.0f, 7.0f, 8.0f};
std::vector<float> bData{1.0f, 2.0f, 3.0f, 4.0f};
DTensor<float> A(aData, m, n, 1, rowMajor);
DTensor<float> B(bData, m);
```

Then, we can solve the system by

```c++
A.leastSquaresBatched(B);
```

For repeated batched solves, allocate the per-matrix device status buffer once
during setup and pass it explicitly:

```c++
DTensor<int> info(A.numMats()); // setup: one integer per matrix
A.leastSquaresBatched(B, info); // no GPUtils workspace allocation in this call
Session::getInstance().synchronizeStream(A.streamIdx());
std::vector<int> status;
info.download(status);
```

The original one-argument overload remains available and allocates its own
temporary status buffer. The supplied buffer may be larger than the batch;
only its first `A.numMats()` entries are written. Both overloads overwrite A
and B; restore the original inputs before solving a new problem. Keep the
inputs and supplied buffer alive until A's stream completes. Reuse on that
stream is ordered; use separate buffers or synchronization for concurrent
streams. The buffer's own stream index does not select the solve stream.

The `DTensor` `B` will be overwritten with the solution.

> [!IMPORTANT]
> This particular example demonstrates how the solution may 
> overwrite only part of the given `B`, as `B` is a
> (4,1,1)-tensor and the solution is a (3,1,1)-tensor.

### 1.8. Saving and loading tensors

Tensor data can be stored in simple text files or binary files. 
The text-based format has the following structure

```text
number_of_rows
number_of_columns
number_of_matrices
data (one entry per line)
```

To save a tensor in a file, simply call `DTensor::saveToFile(filename)`.

If the file extension is `.bt` (binary tensor), the data will be stored in binary format.
The structure of the binary encoding is similar to that of the text encoding:
the first three `uint64_t`-sized positions correspond to the number of rows, columns
and matrices, followed by the elements of the tensor.
The dimensions and payload are stored in the host's native byte order and the
payload uses the native representation of `T`; the file does not record its
element type. The Python API uses little-endian dimensions, so interoperability
requires a little-endian host and a matching NumPy dtype. Readers reject missing,
malformed, truncated, or extra data.

To load a tensor from a file, the static function `DTensor<T>::parseFromFile(filename)` can be used. For example:

```c++
auto z = DTensor<double>::parseFromFile("path/to/my.dtensor")
```

If necessary, you can provide a second argument to `parseFromFile` to specify the order in which the data are stored (the `StorageMode`).

Soon we will release a Python API for reading and serialising (numpy) arrays to `.bt` files.  

## 2. Cholesky factorisation and system solution

> [!WARNING]
> This factorisation only works with positive-definite matrices.

Here is an example:

$$A = \begin{bmatrix}
1 & 2 & 4 \\
2 & 13 & 23 \\
4 & 23 & 77
\end{bmatrix}.$$

This is how to perform a Cholesky factorisation:

```c++
size_t n = 3;
std::vector<float> aData{1.0f, 2.0f, 4.0f,
                         2.0f, 13.0f, 23.0f,
                         4.0f, 23.0f, 77.0f};
DTensor<float> A(aData, n, n, 1, rowMajor);
CholeskyFactoriser<float> cfEngine(A);
status = cfEngine.factorise();
```

Then, you can solve the system `A\b`

```c++
std::vector<float> bData{1.0f, 2.0f, 3.0f};
DTensor<float> B(bData, n);
cfEngine.solve(B);
```

The `DTensor` `B` will be overwritten with the solution. 

## 3. Singular Value Decomposition

> [!WARNING] 
> This implementation only works with square or tall matrices. 

Here is an example with the 4-by-3 matrix

$$B = \begin{bmatrix}
1 & 2 & 3 \\
6 & 7 & 8 \\
6 & 7 & 8 \\
6 & 7 & 8 
\end{bmatrix}.$$

Evidently, the rank of $B$ is 2, so there will be two nonzero singular values.

This is how to perform an SVD decomposition:

```c++
size_t m = 4;
size_t n = 3;
std::vector<float> bData{1.0f, 2.0f, 3.0f,
                         6.0f, 7.0f, 8.0f,
                         6.0f, 7.0f, 8.0f,
                         6.0f, 7.0f, 8.0f};
DTensor<float> B(bData, m, n, 1, rowMajor);
SvdFactoriser<float> svdEngine(B);
status = svdEngine.factorise();
```

By default, `SvdFactoriser` will not compute matrix $U$. If you need it,
create an instance of `SvdFactoriser` as follows

```c++
SvdFactoriser<float> svdEngine(B, true); // computes U
```

Note that the default behaviour of `.factorise()` is to destroy
the given matrix $B$. If you want the factoriser to keep your 
matrix, you need to set the third argument of the above constructor
to `false`.

With `destroyMatrix=false`, SVD owns an independent input copy through a
`std::unique_ptr`; that copy is released automatically on normal destruction or
if later constructor setup throws. With `destroyMatrix=true`, SVD only borrows
the caller's tensor and never frees it. The caller must keep borrowed input
alive and retain its shape/stream binding while the solver uses it.

After you have factorised the matrix, you can access $S$, $V'$ and, perhaps, $U$.
You can do:

```c++
std::cout << "S = " << svdEngine.singularValues() << "\n";
std::cout << "V' = " << svdEngine.rightSingularVectors() << "\n";
```

Note that $U$ can be obtained, if it is computed 
in the first place, by the method 
`.leftSingularVectors()` which returns an object 
of type [`std::optional<DeviceMatrix<TElement>>`](https://dev.to/delta456/modern-c-stdoptional-58ga).
Here is an example:

```c++
auto U = svdEngine.leftSingularVectors();
if (U) std::cout << "U = " << U.value();
```

### Numerical rank tolerances

After a successful factorisation, `svdEngine.rank(epsilon)` retains its existing
absolute-cutoff behavior. To use a scale-aware cutoff explicitly:

```c++
const auto& ranks = svdEngine.rankRelative(1e-6f);       // atol defaults to zero
const auto& noisyRanks = svdEngine.rankRelative(1e-6f, 1e-8f);
```

`rankRelative(rtol, atol)` counts singular values **strictly greater than**
`max(atol, rtol * sigmaMax)`. Each matrix in a batch uses its own largest
singular value. Both tolerances must be finite and nonnegative. Zero matrices
have rank zero, including when both tolerances are zero. Computation and counter
reset stay on the selected stream without downloading singular values or
allocating new workspace.

With `atol = 0`, rescaling a matrix rescales its threshold, preserving numerical
rank within floating-point limits. A positive `atol` intentionally imposes an
absolute cutoff, useful when data have a known noise scale. `rtol` is required;
one conventional choice is `max(rows, cols) * std::numeric_limits<T>::epsilon()`,
but applications with noisy data may need a larger tolerance.

Both rank methods return the same reusable internal device buffer. Download or
deep-copy a result before making another rank call if you need a snapshot, and
order consumers on other streams explicitly. For example, the second call above
overwrites the result referenced by `ranks`. `Nullspace` continues to use its
existing absolute rank policy; this new method does not change its behavior.

## 4. Projection onto a nullspace

The nullspace of a matrix is computed by SVD.
The user provides a `DTensor` made of (padded) matrices.
Then, `Nullspace` computes, possibly pads, and returns the
nullspace matrices `N = (N1, ..., Nk)` in another `DTensor`.

```c++
DTensor<float> paddedMatrices(m, n, k);
Nullspace N(paddedMatrices);  // computes N and NN'
DTensor<float> ns = N.nullspace();  // returns N
```

Each padded nullspace matrix `Ni` is orthogonal, 
and `Nullspace` further computes and stores the
nullspace projection operators `NN' = (N1N1', ..., NkNk')`.
This allows the user to project-in-place onto the nullspace.

```c++
DTensor<float> vectors(m, 1, k);
N.project(vectors);
std::cout << vectors << "\n";
```

## 5. Other

We can get the total allocated bytes (on the GPU) with
```c++
size_t allocatedBytes = 
    Session::getInstance().totalAllocatedBytes();
```

This counts live allocations managed by GPUtils, including retained matrix-pointer
capacity after reshaping, rather than only the bytes implied by the current shape.
Shrinking preserves pointer capacity; growing within that capacity reuses it.
Slices do not count borrowed element storage a second time. A reshaped slice may
own a separate pointer array, which is counted and released independently.
Moves and swaps transfer allocation sizes along with ownership.

For manual accounting, `recordAllocation(size_t)` and `recordDeallocation(size_t)`
replace `incrementAllocatedBytes(int)`. They check overflow/underflow without
narrowing byte counts. Prefer `Session::cudaAllocate` and
`Session::cudaRelease(pointer, allocatedBytes)` so accounting follows successful
CUDA calls automatically. Do not also manually account for those wrapper calls.

GPUtils supports multiple streams. By default a single stream is created, 
but you can set the number of streams you need with 
```c++
/* This needs to be the first line in your code */
Session::setStreams(4); // create 4 strems
```
Call `setStreams` with a positive count before the first `Session` or tensor
allocation initializes the session. Changing the count afterward is rejected.
Then, you can use `setStreamIdx` to select a stream to go with your instance of `DTensor`
```c++
auto a = DTensor<double>::createRandomTensor(3, 6, 4, -1, 1).setStreamIdx(0);
auto b = DTensor<double>::createRandomTensor(3, 6, 4, -1, 1).setStreamIdx(1);
// do stuff...
Session::getInstance().synchronizeAllStreams();
```

Stream ordering rules:

- Single and batched Cholesky, QR, and SVD execute on their bound matrix's selected
  stream. Select that stream before constructing the solver. Keep solver workspaces
  and operands alive until completion; operands produced on another stream require
  an explicit dependency before the solve. Solver `info()` tensors follow the
  matrix stream, as do SVD outputs and transpose results.
- Existing `upload`, `download`, debug element access, and deep copies wait for
  completion on the tensor/source stream. They do not depend on legacy default-stream
  synchronization. The row-major upload conversion buffer stays alive until transfer
  completion. No new public asynchronous host-transfer API is introduced.
- `source.deviceCopyTo(destination)` waits for prior work on the destination's
  selected stream, then completes the copy on the source stream before returning.
  Neither stream index changes, so subsequent destination-stream consumers are safe.
  This is a blocking convenience operation, not a transfer-overlap API.
- Arithmetic involving several tensors does not automatically connect all operand
  streams. Callers must synchronize or establish CUDA-event dependencies between
  different producer and consumer streams. A selected stream records an execution
  convention, not a record of every external access to a tensor's raw pointer.

## 6. Build, test, and check CUDA memory

The scripts in `scripts/` can be launched from any working directory. They require
CMake, a supported CUDA toolkit/GPU, a C++ compiler, and Python with NumPy. For an
isolated Python environment (from the repository root):

```bash
python3 -m venv build/test-venv
build/test-venv/bin/python -m pip install numpy
```

Run all C++ and Python unit tests:

```bash
./scripts/build_and_test.sh -DSM_ARCH=120
```

Use `SM_ARCH=86` for A40 or `SM_ARCH=87 -DCPPVERSION=17` for Orin instead of `120`.
Additional arguments are forwarded to CMake; the default build type is
`RelWithDebInfo`. `GPUTILS_BUILD_DIR`, `JOBS`, and `PYTHON` override the build
directory, build parallelism (default 4), and Python interpreter. Without `PYTHON`,
the script uses `test-venv/bin/python` in the build directory if available, then
`python3`. Python fixtures stay in the build tree.

Build/test and run the complete CUDA suite under Compute Sanitizer memcheck:

```bash
./scripts/check_memory.sh -DSM_ARCH=120
```

This checks invalid device memory accesses and device/heap leaks, runs the full
suite three times in shuffled order within one process, and checks `gputils_main`
separately. It enables CUDA line information in the default build configuration,
preserves failure exit codes, and requires clean error/leak summaries. Complete
logs and generated fixtures are stored under `build/sanitizer/`. Set
`COMPUTE_SANITIZER` to override its executable path and `SANITIZER_REPEATS` to
change the repeat count. Compute Sanitizer checks CUDA memory, not arbitrary
C++ host-memory leaks. Do not use sanitizer runs for performance measurements.

## Happy number crunching!
