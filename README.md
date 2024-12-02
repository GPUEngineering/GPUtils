# GPUtils

## 1. DTensor

The `DTensor` class is for manipulating data on a GPU. 
It manages their memory and facilitates various algebraic operations.

A tensor has three axes: `[rows (m) x columns (n) x matrices (k)]`.
An (m,n,1)-tensor stores a _matrix_, and an (m,1,1)-tensor stores a _vector_.

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

The copy constructor has also been implemented; to hard-copy a `DTensor` just
do `DTensor<float> myCopy(existingTensor)`.

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

## Happy number crunching!
