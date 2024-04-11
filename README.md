# GPUtils

## 1. DeviceVector

The `DeviceVector` class is for manipulating vectors on the GPU. It manages their
memory and facilitates various simple algebraic operations.

In what follows, we first need to create a `Context` object by doing

```c++
Context context;
```

### 1.1. Memory management

The simplest way to create an empty `DeviceVector` object is by defining its capacity:

```c++
size_t n = 100;
DeviceVector myVector(&context, n);
```

A `DeviceVector` can be instantiated from host memory:

```c++
std::vector<float> h_a{4., -5., 6., 9., 8., 5., 9., -10.2, 9., 11.};
DeviceVector<float> myVector(&context, h_a);
```

We will often need to create slices (or shallow copies) of a `DeviceVector` 
given a range of values. We can then do:


```c++
size_t from = 3;
size_t to = 5;
DeviceVector<float> mySlice(myVector, from, to);
```

The method `allocateOnDevice` can be used to allocate memory on the GPU.
Note that this will destroy and clear any previously allocated memory.
In principle, it is not wise to call this repeatedly on the same object.
The method returns true if the allocation is successful. Note that this 
method is not used to resize the allocated memory. For example, have a look
at this snippet:

```c++
DeviceVector<float> z(&context, 100);  // size of z = 100
z.allocateOnDevice(80); // nothing happens
std::cout << "size of x = " << z.capacity() << std::endl;  // size is stil 100
```

Sometimes we need to reuse an already allocated device vector by uploading 
new data from the host by using the method `upload`. Here is a short example:

```c++
std::vector<float> h_a{4., -5., 6., 9., 8., 5., 9., -10.2, 9., 11.};  // host data
DeviceVector<float> z(&context, 100);  // device vector of length 100
z.upload(h_a);  // upload 10 values only
std::cout << "#z = "<< z.capacity() << std::endl; // the size is still 100
```

We can upload some host data to a particular position of the device vector as follows:

```c++
std::vector<float> hostData{1., 2., 3.};
std::vector<float> hostZeros(7);
DeviceVector<float> x(&context, hostZeros);  // x = [0, 0, 0, 0, 0, 0, 0]
DeviceVector<float> mySlice(x, 3, 5); 
mySlice.upload(hostData);
std::cout << x << std::endl;  // x = [0, 0, 0, 1, 2, 3, 0]
```

If necessary, the data can be downloaded from the device to the host using 
`download`.

Very often we will also need to copy data from a device vector 
to another device vector (without passing through the host).
To do this we can use `deviceCopyTo`. Here is an example:

```c++
DeviceVector<float> x(&context, 10);
DeviceVector<float> y(&context, 10);
x.deviceCopyTo(y); // x ---> y
```


Lasly, a not so efficient method that should only be used for 
debugging, if at all, is `fetchElementFromDevice`, which fetches
one element of the vector to the host; for the love of god, do 
not put this in a loop.

### 1.2. Printing vectors

Printing vectors is as easy as 

```c++
std::cout << myVector;
```

The `<<` operator downloads the data from the device to the host and 
prints them (which makes it useful for quick debugging). 


### 1.3. Computation of scalar quantities

The following scalar quantities can be computed (internally, 
we use `cublas` functions):

- The Euclidean norm of $x$, using `norm2`
- The sum of all values of the vector, using `sum` 

### 1.4. Some cool operators

:warning: Note that these operations are currently supported for 
`float`-based device vectors only (the extension other types is trivial).

We can add device vectors on the device as follows:

```c++
Context context;
std::vector<float> host_x{1., 2., 3., 4., 5., 6.,  7.};
std::vector<float> host_y{1., 3., 5., 7., 9., 11., 13.};
DeviceVector<float> x(&context, host_x);
DeviceVector<float> y(&context, host_x);
x += y;  // x = [2, 4, 6, 8, 10, 12, 14]
```

To subtract `y` from `x` we can use `x -= y`.

We can also scale a device vector by a scalar with `*=` (e.g, `x *= 5.0f`). 
To negate the values of a device vector we can be `x *= -1.0`.

We can also compute the inner product of two vectors as follows:

```c++
float innerProduct = x * y;
```

If necessary, we can also use the following operations

```c++
auto sum = x + y;
auto diff = x - y;
auto scaledX = 3.0f * x;
```


## 2. DeviceMatrix

To construct a device matrix we need to provide the data in 
an array; we can use either a column-major or a row-major format,
the former being the preferred and default one.
Suppose we need to construct the matrix 

$$A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9 \\
10 & 11 & 12 \\
13 & 14 & 15
\end{bmatrix},$$

where the data is stored, say, in row-major format.
Then, we do
```c++
Context context;
size_t n_rows = 5;
std::vector<float> h_data{1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f,
                          7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f,
                          13.0f, 14.0f, 15.0f};
DeviceMatrix<float> mat(&context,
                        n_rows,
                        h_data,
                        MatrixStorageMode::RowMajor);
```
