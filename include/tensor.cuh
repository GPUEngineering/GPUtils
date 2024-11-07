#include <random>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <memory>
#include <optional>
#include <cassert>

#ifndef TENSOR_CUH
#define TENSOR_CUH

/**
 * Define defaults
 */
#define DEFAULT_FPX double
#define THREADS_PER_BLOCK 512
#if (__cplusplus >= 201703L)  ///< if c++17 or above
#define TEMPLATE_WITH_TYPE_T template<typename T = DEFAULT_FPX>
#else
#define TEMPLATE_WITH_TYPE_T template<typename T>
#endif
#if (__cplusplus >= 202002L)  ///< if c++20 or above
#define TEMPLATE_CONSTRAINT_REQUIRES_FPX requires std::floating_point<T>
#else
#define TEMPLATE_CONSTRAINT_REQUIRES_FPX
#endif

static std::random_device RND_DEVICE;


/**
 * Generate vector of random elements
 * @tparam T
 * @param n
 * @param low
 * @param hi
 * @return
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
std::vector<T> generateRealRandomVector(size_t n, T low, T hi) {
    std::mt19937_64 mersenne_engine(RND_DEVICE());
    std::uniform_real_distribution<T> dist(low, hi);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    std::vector<T> vec(n);
    generate(begin(vec), end(vec), gen);
    return vec;
}

inline std::vector<int> generateIntRandomVector(size_t n, int low, int hi) {
    std::mt19937_64 mersenne_engine(RND_DEVICE());
    std::uniform_int_distribution dist(low, hi);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    std::vector<int> vec(n);
    generate(begin(vec), end(vec), gen);
    return vec;
}

/**
 * Determines the number of blocks needed for a given number of tasks, n,
 * and number of threads per block
 *
 * @param n problem size
 * @param threads_per_block threads per block (defaults to THREADS_PER_BLOCK)
 * @return number of blocks
 */
constexpr size_t numBlocks(size_t n, size_t threads_per_block = THREADS_PER_BLOCK) {
    return (n / threads_per_block + (n % threads_per_block != 0));
}


/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), __FILE__, __LINE__); } while(false)

TEMPLATE_WITH_TYPE_T inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublas error. Name: " << cublasGetStatusName(code)
                      << ", string: " << cublasGetStatusString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cusolverStatus_t>) {
        if (code != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "cusolver error. Status: " << code
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}


/* ================================================================================================
 *  SESSION
 * ================================================================================================ */

/**
 * Singleton for Cuda library handles.
 * Cuda library functions require a handle.
 * A project requires exactly one handle per Cuda library.
 * This class is created as a singleton, and contains a unique handle for each library.
 * The cuBlas handle can be accessed anywhere by `Session::getInstance().cuBlasHandle()`
 * The cuSolver handle can be accessed anywhere by `Session::getInstance().cuSolverHandle()`
 */
class Session {
public:

    static Session &getInstance() {
        static Session instance;
        return instance;
    }

private:
    Session() {
        gpuErrChk(cublasCreate(&m_cublasHandle));
        gpuErrChk(cusolverDnCreate(&m_cusolverHandle));
    }

    ~Session() {
        gpuErrChk(cublasDestroy(m_cublasHandle));
        gpuErrChk(cusolverDnDestroy(m_cusolverHandle));
    }

    cublasHandle_t m_cublasHandle;
    cusolverDnHandle_t m_cusolverHandle;


public:
    Session(Session const &) = delete;

    void operator=(Session const &) = delete;

    cublasHandle_t &cuBlasHandle() { return m_cublasHandle; }

    cusolverDnHandle_t &cuSolverHandle() { return m_cusolverHandle; }
};


/* ================================================================================================
 *  TENSOR
 * ================================================================================================ */

/**
 * Storage mode for the data of a matrix
 */
enum StorageMode {
    columnMajor,  ///< column major storage (default)
    rowMajor,  ///< row major storage
    defaultMajor = columnMajor
};

/**
 * This library uses tensors to store and manipulate data on a GPU device.
 * A tensor has three axes: [rows (m) x columns (n) x matrices (k)].
 * An (m,n,1)-tensor is a matrix, and an (m,1,1)-tensor is a vector.
 * Tensors can be used to do a batched operation on many similar-sized matrices or vectors in parallel.
 * @tparam T type of data stored in tensor
 */
TEMPLATE_WITH_TYPE_T
class DTensor {

private:
    T *m_d_data = nullptr;  ///< Pointer to device data
    T **m_d_ptrMatrices = nullptr; ///< Pointer to matrices in tensor
    size_t m_numRows = 0;  ///< Number of rows
    size_t m_numCols = 0;  ///< Number of columns
    size_t m_numMats = 0;  ///< Number of matrices
    bool m_doDestroyData = false;  ///< Whether to destroy memory
    bool m_doDestroyPtrMatrices = false;  ///< Whether to destroy memory

    void destroy() {
        if (m_doDestroyData) {
            if (m_d_data) gpuErrChk(cudaFree(m_d_data));
            m_d_data = nullptr;
        }
        if (m_doDestroyPtrMatrices) {
            if (m_d_ptrMatrices) gpuErrChk(cudaFree(m_d_ptrMatrices));
            m_d_ptrMatrices = nullptr;
        }
    }

    /**
     * Allocate `size` number of `T` data on the device.
     * @param size number of data elements to allocate
     * @param zero sets allocated data to `0`
     * @return
     */
    bool allocateOnDevice(size_t size, bool zero = false);

    /**
     * Create column-major `std::vector` from a row-major one.
     * @param rm row-major stored data
     * @param cm column-major storage
     */
    void rm2cm(const std::vector<T> &rm, std::vector<T> &cm) {
        size_t n = m_numRows * m_numCols;
        for (size_t k = 0; k < m_numMats; k++) {
            size_t idx = k * n;
            for (size_t r = 0; r < m_numRows; r++) {
                for (size_t c = 0; c < m_numCols; c++) {
                    cm[idx + (r + c * m_numRows)] = rm[idx + (c + r * m_numCols)];
                }
            }
        }
    }

    /**
     * Appends this tensor to `std::ostream` object, in the format it represents.
     * @param out `std::ostream` object for appending data to be printed
     * @return tensor in `std::ostream` data format
     */
    std::ostream &print(std::ostream &out) const;

    void initialisePointersToMatricesData() {
        /* Make sure m_d_ptrMatrices has been allocated */
        if (!m_d_ptrMatrices || !m_doDestroyPtrMatrices) {
            throw std::runtime_error("Unallocated memory (m_d_ptrMatrices)");
        }
        /* Host-based vector of pointers */
        std::vector<T *> h_pointers(m_numMats);
        size_t numelMat = m_numRows * m_numCols;
        h_pointers[0] = m_d_data;
        for (size_t i = 1; i < m_numMats; i++) {
            h_pointers[i] = m_d_data + i * numelMat;
        }
        /* Upload data to m_d_ptrMatrices */
        size_t buffer_size = m_numMats * sizeof(T *);
        gpuErrChk(cudaMemcpy(m_d_ptrMatrices, h_pointers.data(), buffer_size, cudaMemcpyHostToDevice));
    }

public:

    /**
     * Create a tensor with random elements
     * @param numRows
     * @param numCols
     * @param numMats
     * @param low
     * @param hi
     */
    static DTensor<T> createRandomTensor(size_t numRows, size_t numCols, size_t numMats, T low, T hi);

    /**
    * Constructs a DTensor object.
    */
    DTensor() = default;

    /**
    * Destroys a DTensor object.
    */
    ~DTensor() {
        destroy();
    }

    /**
     * Constructs (m,n,k)-tensor and allocates memory.
     * @param m number of rows
     * @param n number of columns
     * @param k number of matrices
     * @param zero sets allocated data to `0`
     */
    DTensor(size_t m, size_t n = 1, size_t k = 1, bool zero = false);

    /**
     * Constructs (m,n,k)-tensor and uploads data to device.
     * @param data `std::vector` of data to upload to the device
     * @param m number of rows
     * @param n number of columns
     * @param k number of matrices
     */
    DTensor(const std::vector<T> &data, size_t m, size_t n = 1, size_t k = 1,
            StorageMode mode = StorageMode::defaultMajor);

    /**
     * Copy constructor.
     * @param other tensor to copy to newly constructed tensor
     */
    DTensor(const DTensor &other);

    /**
     * Move constructor.
     * @param other tensor to move to newly constructed tensor
     */
    DTensor(DTensor &&other);

    /**
     * Slice constructor.
     * @param other other tensor with the same template type
     * @param axis axis to slice (0=rows, 1=columns, 2=matrices)
     * @param from index to slice axis from (zero-indexed)
     * @param to index to slice axis to (inclusive)
     */
    DTensor(const DTensor &other, size_t axis, size_t from, size_t to);

    /**
     * @return raw pointer to the first element of this tensor on the device
     */
    T *raw() const;

    /**
     * @return number of rows
     */
    size_t numRows() const;

    /**
     * @return number of columns
     */
    size_t numCols() const;

    /**
     * @return number of matrices
     */
    size_t numMats() const;

    /**
     * @return number of elements
     */
    size_t numEl() const;

    /**
     * Upload from `std::vector` to device.
     * @param vec data source to upload
     * @return true iff upload is successful
     */
    bool upload(const std::vector<T> &vec, StorageMode mode = StorageMode::defaultMajor);

    /**
     * Download from device to `std::vector`.
     * @param vec destination vector
     */
    void download(std::vector<T> &vec) const;

    /**
     * Device-to-device copy.
     * @param other target tensor
     */
    void deviceCopyTo(DTensor<T> &other) const;

    /**
     * Creates a vector of pointers to the matrices of this tensor.
     * The vector is an (n,1,1)-tensor, where n is the number of matrices in this tensor.
     * @return vector of pointers to the first element of each matrix
     */
    DTensor<T *> pointersToMatrices() const;

    /**
     * Slices rows from specified matrix.
     * @param rowsFrom index to slice rows from (zero-indexed)
     * @param rowsTo index to slice rows to (inclusive)
     * @param matIdx index of matrix to slice rows from (zero-indexed)
     * @return slice of rows
     */
    DTensor<T> getRows(size_t rowsFrom, size_t rowsTo, size_t matIdx) const;

    /**
     * Transposes each (m,n)-matrix of this tensor.
     * Each transposed matrix is stored at same k-index in new tensor.
     * @return tensor of transposed matrices
     */
    DTensor<T> tr() const;

    /**
     * Frobenius dot product.
     * @param other other tensor of compatible dimensions
     * @return value of Frobenius dot product
     */
    T dotF(const DTensor &other);

    /**
     * Frobenius norm.
     * The square root of the sum of squares of all elements.
     * A.k.a. the Euclidean norm, if this is a vector.
     * @return norm as same data type
     */
    T normF() const;

    /**
     * Sum of absolute of all elements.
     * @return sum as same data type
     */
    T sumAbs() const;

    /**
     * Maximum of absolute of all elements.
     * Equivalent to inf-norm, max(|x_i|) for all i.
     * @return max of absolute as same data type
     */
    T maxAbs() const;

    /**
     * Minimum of absolute of all elements, min(|x_i|) for all i.
     * @return min of absolute as same data type
     */
    T minAbs() const;

    /**
     * Applied the Givens rotation G(i, j, c, s) on row i and column j,
     * with cos θ = c, and sin θ = s. The rotation is applied in-place
     * to all slices of the tensor. Recall that the Givens rotation is
     *
     *                  i     j
     *         |1  0             ...  0 |
     *         |0  1             ...  0 |
     *         |                        |
     *         |      1                 |
     *       i |        c     s         |
     * G' =    '                        '
     *       j |       -s     c         |
     *         |                1       |
     *         |                        |
     *         |                      0 |
     *
     * The right Givens rotation consists in multiplying from the right
     * by G.
     *
     * Equivalently, the right Givens transformation performs the following
     * operation on the i and j columns of this matrix:
     *
     * A[:, i] <-- c * A[:, i] - s * A[:, j]
     * A[:, j] <-- s * A[:, i] + c * A[:, j]
     *
     * @param i first column index
     * @param j second column index
     * @param c cos θ
     * @param minus_s minus sin θ
     * @throws std::invalid_argument if either i or j is greater or equal ncols
     */
    void applyRightGivensRotation(size_t i, size_t j, const T *c, const T *minus_s);

    /**
     * Performs a Givens rotation (left multiplication by G')
     *
     * @param i first row index
     * @param j second row index
     * @param c cos θ
     * @param minus_s minus sin θ
     * @throws std::invalid_argument if either i or j is greater or equal nrows
     */
    void applyLeftGivensRotation(size_t i, size_t j, const T *c, const T *minus_s);

    /**
     * Batch solves `A \ b`.
     * Solves `bi <- Ai \ bi` for each k-index `i`.
     * A is this (m,n,k)-tensor and b is the provided (m,1,k)-tensor.
     * A and b must have compatible dimensions (same number of rows and matrices).
     * A must be a square or tall matrix (m>=n).
     * @param b provided tensor
     * @return least squares solutions (overwrites (n,1,k)-part of b)
     */
    void leastSquaresBatched(DTensor &b);

    /**
     * Batched `C <- bC + a*A*B`.
     * Performs the operation `Ci <- bCi + a*Ai*Bi` for each k-index `i`.
     * C is this tensor.
     * A and B are tensors of compatible dimensions.
     * @param A tensor A
     * @param B tensor B
     * @param alpha scalar to scale AB
     * @param beta scalar to scale C
     */
    void addAB(const DTensor<T> &A, const DTensor<T> &B, T alpha = 1, T beta = 0);

    void reshape(size_t newNumRows, size_t newNumCols, size_t newNumMats = 1);

    /* ------------- OPERATORS ------------- */

    DTensor &operator=(const DTensor &other);

    T operator()(size_t i, size_t j = 0, size_t k = 0) const;

    DTensor &operator*=(T scalar);

    DTensor &operator+=(const DTensor &rhs);

    DTensor &operator-=(const DTensor &rhs);

    /* ------------- FRIENDS ------------- */

    friend DTensor operator+(DTensor &first, const DTensor &second) {
        DTensor result(first);
        result += second;
        return result;
    }

    friend DTensor operator-(DTensor &first, const DTensor &second) {
        DTensor result(first);
        result -= second;
        return result;
    }

    friend DTensor<T> operator*(DTensor &A, DTensor &B) {
        size_t nrA = A.m_numRows, ncB = B.m_numCols, nmB = B.m_numMats;
        DTensor<T> result(nrA, ncB, nmB);
        result.addAB(A, B);
        return result;
    }

    friend DTensor<T> operator*(T a, DTensor &B) {
        DTensor<T> result(B);
        result *= a;
        return result;
    }

    friend std::ostream &operator<<(std::ostream &out, const DTensor<T> &data) {
        return data.print(out);
    }

}; /* END OF DTENSOR */

template<typename T>
DTensor<T> DTensor<T>::createRandomTensor(size_t numRows, size_t numCols, size_t numMats, T low, T hi) {
    if constexpr (std::is_floating_point<T>::value) {
        auto randVec = generateRealRandomVector(numRows * numCols * numMats, low, hi);
        DTensor<T> a(randVec, numRows, numCols, numMats);
        return a;
    } else if constexpr (std::is_same_v<T, int>) {
        auto randVec = generateIntRandomVector(numRows * numCols * numMats, low, hi);
        DTensor<T> a(randVec, numRows, numCols, numMats);
        return a;
    } else {
        throw std::invalid_argument("[createRandomTensor] unsupported type T");
    }
}

template<typename T>
void DTensor<T>::reshape(size_t newNumRows, size_t newNumCols, size_t newNumMats) {
    size_t newNumElements = newNumRows * newNumCols * newNumMats;
    if (numEl() != newNumElements) {
        char errMessage[256];
        sprintf(errMessage,
                "DTensor[%lu x %lu x %lu] with %lu elements cannot be reshaped into DTensor[%lu x %lu x %lu] (%lu elements)",
                numRows(), numRows(), numMats(), numEl(), newNumRows, newNumCols, newNumMats, newNumElements);
        throw std::invalid_argument(errMessage);
    }
    m_numRows = newNumRows;
    m_numCols = newNumCols;
    m_numMats = newNumMats;
}

template<typename T>
DTensor<T>::DTensor(size_t m, size_t n, size_t k, bool zero) {
    m_numRows = m;
    m_numCols = n;
    m_numMats = k;
    size_t size = m * n * k;
    allocateOnDevice(size, zero);
}

template<typename T>
DTensor<T>::DTensor(const std::vector<T> &data, size_t m, size_t n, size_t k, StorageMode mode) {
    m_numRows = m;
    m_numCols = n;
    m_numMats = k;
    size_t size = m * n * k;
    allocateOnDevice(size);
    upload(data, mode);
}

template<typename T>
DTensor<T>::DTensor(const DTensor<T> &other) {
    m_numMats = other.m_numMats;
    m_numRows = other.m_numRows;
    m_numCols = other.m_numCols;

    allocateOnDevice(m_numRows * m_numCols * m_numMats);
    gpuErrChk(cudaMemcpy(m_d_data, other.raw(), m_numRows * m_numCols * m_numMats * sizeof(T),
                         cudaMemcpyDeviceToDevice));
}

template<typename T>
DTensor<T>::DTensor(const DTensor<T> &other, size_t axis, size_t from, size_t to) {
    if (from > to) throw std::invalid_argument("from > to");
    size_t offset = 0, len = to - from + 1;
    if (axis == 2) {
        offset = other.m_numRows * other.m_numCols * from;
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;
        m_numMats = len;
    } else if (axis == 1) {
        offset = other.m_numRows * from;
        m_numRows = other.m_numRows;
        m_numCols = len;
        m_numMats = 1;
    } else if (axis == 0) {
        offset = from;
        m_numRows = to - from + 1;
        m_numCols = 1;
        m_numMats = 1;
    }
    m_d_data = other.m_d_data + offset;
    m_doDestroyData = false;
}

template<typename T>
DTensor<T>::DTensor(DTensor<T> &&other) {
    m_numCols = other.m_numCols;
    m_numRows = other.m_numRows;
    m_numMats = other.m_numMats;
    m_d_data = other.m_d_data;
    m_doDestroyData = true;
    other.m_doDestroyData = false;
    other.m_d_data = nullptr;
    other.m_numCols = 0;
    other.m_numRows = 0;
    other.m_numMats = 0;
}

template<typename T>
inline size_t DTensor<T>::numRows() const {
    return m_numRows;
}

template<typename T>
inline size_t DTensor<T>::numCols() const {
    return m_numCols;
}

template<typename T>
inline size_t DTensor<T>::numMats() const {
    return m_numMats;
}

template<typename T>
inline size_t DTensor<T>::numEl() const {
    return m_numRows * m_numCols * m_numMats;
}

template<>
inline double DTensor<double>::dotF(const DTensor<double> &other) {
    if (m_numRows != other.m_numRows || m_numCols != other.m_numCols || m_numMats != other.m_numMats)
        throw std::invalid_argument("[dotF] incompatible dimensions");
    size_t n = numEl();
    double result;
    gpuErrChk(cublasDdot(Session::getInstance().cuBlasHandle(), n,
                         raw(), 1,
                         other.raw(), 1,
                         &result));
    return result;
}

template<>
inline float DTensor<float>::dotF(const DTensor<float> &other) {
    if (m_numRows != other.m_numRows || m_numCols != other.m_numCols || m_numMats != other.m_numMats)
        throw std::invalid_argument("[dotF] incompatible dimensions");
    size_t n = numEl();
    float result;
    gpuErrChk(cublasSdot(Session::getInstance().cuBlasHandle(), n,
                         raw(), 1,
                         other.raw(), 1,
                         &result));
    return result;
}

template<>
inline double DTensor<double>::normF() const {
    double the_norm;
    gpuErrChk(cublasDnrm2(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &the_norm));
    return the_norm;
}

template<>
inline float DTensor<float>::normF() const {
    float the_norm;
    gpuErrChk(cublasSnrm2(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &the_norm));
    return the_norm;
}

template<>
inline float DTensor<float>::sumAbs() const {
    float sumAbsAllElements;
    gpuErrChk(cublasSasum(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &sumAbsAllElements));
    return sumAbsAllElements;
}

template<>
inline double DTensor<double>::sumAbs() const {
    double sumAbsAllElements;
    gpuErrChk(cublasDasum(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &sumAbsAllElements));
    return sumAbsAllElements;
}

template<>
inline float DTensor<float>::maxAbs() const {
    int idx;
    float hostDst;
    gpuErrChk(cublasIsamax(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                           &idx));
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + idx - 1, sizeof(float), cudaMemcpyDeviceToHost));
    return std::signbit(hostDst) ? -hostDst : hostDst;
}

template<>
inline double DTensor<double>::maxAbs() const {
    int idx;
    double hostDst;
    gpuErrChk(cublasIdamax(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                           &idx));
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + idx - 1, sizeof(double), cudaMemcpyDeviceToHost));
    return std::signbit(hostDst) ? -hostDst : hostDst;
}

template<>
inline float DTensor<float>::minAbs() const {
    int idx;
    float hostDst;
    gpuErrChk(cublasIsamin(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                           &idx));
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + idx - 1, sizeof(float), cudaMemcpyDeviceToHost));
    return std::signbit(hostDst) ? -hostDst : hostDst;
}

template<>
inline double DTensor<double>::minAbs() const {
    int idx;
    double hostDst;
    gpuErrChk(cublasIdamin(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                           &idx));
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + idx - 1, sizeof(double), cudaMemcpyDeviceToHost));
    return std::signbit(hostDst) ? -hostDst : hostDst;
}

template<typename T>
void DTensor<T>::applyRightGivensRotation(size_t i, size_t j, const T *c, const T *minus_s) {
    if (m_numMats > 1) throw std::invalid_argument("[applyRightGivensRotation] tensors (nMat>1) not supported");
    T *col_i = m_d_data + i * m_numRows;
    T *col_j = m_d_data + j * m_numRows;
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDrot(Session::getInstance().cuBlasHandle(), m_numRows,
                             col_i, 1, col_j, 1, c, minus_s));
    } else if constexpr (std::is_same_v<T, float>) {
        gpuErrChk(cublasSrot(Session::getInstance().cuBlasHandle(), m_numRows,
                             col_i, 1, col_j, 1, c, minus_s));
    }
}

template<typename T>
void DTensor<T>::applyLeftGivensRotation(size_t i, size_t j, const T *c, const T *minus_s) {
    if (m_numMats > 1) throw std::invalid_argument("[applyLeftGivensRotation] tensors (nMat>1) not supported");
    if constexpr (std::is_same_v<T, double>) {
        gpuErrChk(cublasDrot(Session::getInstance().cuBlasHandle(), m_numCols,
                             m_d_data + i, m_numRows,
                             m_d_data + j, m_numRows,
                             c, minus_s));
    } else if constexpr (std::is_same_v<T, float>) {
        gpuErrChk(cublasSrot(Session::getInstance().cuBlasHandle(), m_numCols,
                             m_d_data + i, m_numRows,
                             m_d_data + j, m_numRows,
                             c, minus_s));
    } else {
        throw std::invalid_argument("[applyLeftGivensRotation] Unsupported type T");
    }
}

template<typename T>
inline bool DTensor<T>::allocateOnDevice(size_t size, bool zero) {
    if (size <= 0) return false;
    destroy();
    m_doDestroyData = true;
    size_t buffer_size = size * sizeof(T);
    bool cudaStatus = cudaMalloc(&m_d_data, buffer_size);
    if (cudaStatus != cudaSuccess) return false;
    if (zero) gpuErrChk(cudaMemset(m_d_data, 0, buffer_size)); // set to zero all elements

    m_doDestroyPtrMatrices = true;
    cudaStatus = cudaMalloc(&m_d_ptrMatrices, numMats() * sizeof(T *));

    /* Initialise m_d_ptrMatrices */
    initialisePointersToMatricesData();

    return (cudaStatus != cudaSuccess);
}

template<typename T>
inline bool DTensor<T>::upload(const std::vector<T> &vec, StorageMode mode) {
    size_t size = vec.size();
    size_t thisSize = m_numRows * m_numCols * m_numMats;
    // make sure vec is of right size
    if (size != thisSize) throw std::invalid_argument("[upload] vec has wrong size");
    std::vector<T> vecCm(thisSize);
    if (mode == StorageMode::rowMajor) {
        rm2cm(vec, vecCm);
    } else {
        vecCm = vec;
    }
    if (size <= thisSize) {
        size_t buffer_size = size * sizeof(T);
        gpuErrChk(cudaMemcpy(m_d_data, vecCm.data(), buffer_size, cudaMemcpyHostToDevice));
    }
    return true;
}

template<typename T>
inline void DTensor<T>::download(std::vector<T> &vec) const {
    vec.resize(m_numRows * m_numCols * m_numMats);
    gpuErrChk(cudaMemcpy(vec.data(),
                         m_d_data,
                         m_numRows * m_numCols * m_numMats * sizeof(T),
                         cudaMemcpyDeviceToHost));
}

template<typename T>
inline T *DTensor<T>::raw() const {
    return m_d_data;
}

template<>
inline DTensor<float> DTensor<float>::tr() const {
    DTensor<float> transposes(m_numCols, m_numRows, m_numMats);
    float alpha = 1.0f, beta = 0;
    size_t numElMat = m_numCols * m_numRows;
    for (size_t i = 0; i < m_numMats; i++) {
        gpuErrChk(cublasSgeam(Session::getInstance().cuBlasHandle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              m_numCols, m_numRows,
                              &alpha, raw() + numElMat * i, m_numRows,
                              &beta, nullptr, m_numCols,
                              transposes.raw() + numElMat * i, m_numCols));
    }
    return transposes;
}

template<>
inline DTensor<double> DTensor<double>::tr() const {
    DTensor<double> transposes(m_numCols, m_numRows, m_numMats);
    double alpha = 1.0f, beta = 0;
    size_t numElMat = m_numCols * m_numRows;
    for (size_t i = 0; i < m_numMats; i++) {
        gpuErrChk(cublasDgeam(Session::getInstance().cuBlasHandle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              m_numCols, m_numRows,
                              &alpha, raw() + numElMat * i, m_numRows,
                              &beta, nullptr, m_numCols,
                              transposes.raw() + numElMat * i, m_numCols));
    }
    return transposes;
}

template<typename T>
inline void DTensor<T>::deviceCopyTo(DTensor<T> &elsewhere) const {
    if (elsewhere.numEl() < numEl()) {
        throw std::invalid_argument("[deviceCopyTo] tensor does not fit into destination");
    }
    gpuErrChk(cudaMemcpy(elsewhere.raw(),
                         m_d_data,
                         m_numRows * m_numCols * m_numMats * sizeof(T),
                         cudaMemcpyDeviceToDevice));
}

template<>
inline DTensor<double> &DTensor<double>::operator*=(double scalar) {
    double alpha = scalar;
    gpuErrChk(
            cublasDscal(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, m_d_data, 1));
    return *this;
}

template<typename T>
DTensor<T> &DTensor<T>::operator=(const DTensor<T> &other) {
    m_numMats = other.m_numMats;
    m_numRows = other.m_numRows;
    m_numCols = other.m_numCols;
    m_doDestroyData = false;
    m_d_data = other.m_d_data;
    return *this;
}

template<>
inline DTensor<float> &DTensor<float>::operator*=(float scalar) {
    float alpha = scalar;
    gpuErrChk(
            cublasSscal(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, m_d_data, 1));
    return *this;
}

template<>
inline DTensor<double> &DTensor<double>::operator+=(const DTensor<double> &rhs) {
    const double alpha = 1.;
    gpuErrChk(
            cublasDaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data,
                        1, m_d_data, 1));
    return *this;
}

template<>
inline DTensor<float> &DTensor<float>::operator+=(const DTensor<float> &rhs) {
    const float alpha = 1.;
    gpuErrChk(
            cublasSaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data,
                        1, m_d_data, 1));
    return *this;
}

template<>
inline DTensor<float> &DTensor<float>::operator-=(const DTensor<float> &rhs) {
    const float alpha = -1.;
    cublasSaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data, 1,
                m_d_data, 1);
    return *this;
}

template<>
inline DTensor<double> &DTensor<double>::operator-=(const DTensor<double> &rhs) {
    const double alpha = -1.;
    gpuErrChk(
            cublasDaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data,
                        1, m_d_data, 1));
    return *this;
}

template<typename T>
inline T DTensor<T>::operator()(size_t i, size_t j, size_t k) const {
    T hostDst;
    size_t offset = i + m_numRows * (j + m_numCols * k);
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + offset, sizeof(T), cudaMemcpyDeviceToHost));
    return hostDst;
}

template<typename T>
inline DTensor<T *> DTensor<T>::pointersToMatrices() const {
    std::vector<T *> h_pointers(m_numMats);
    size_t numelMat = m_numRows * m_numCols;
    h_pointers[0] = m_d_data;
    for (size_t i = 1; i < m_numMats; i++) {
        h_pointers[i] = m_d_data + i * numelMat;
    }
    DTensor<T *> t(h_pointers, m_numMats, 1, 1);
    return t;
}

template<>
inline void DTensor<double>::addAB(const DTensor<double> &A, const DTensor<double> &B, double alpha, double beta) {
    size_t nMat = A.numMats();
    size_t nRA = A.numRows();
    size_t nCA = A.numCols();
    size_t nCB = B.numCols();
    DTensor<double *> ptrA = A.pointersToMatrices();
    DTensor<double *> ptrB = B.pointersToMatrices();
    DTensor<double *> ptr = pointersToMatrices();
    double _alpha = alpha, _beta = beta;
    gpuErrChk(cublasDgemmBatched(Session::getInstance().cuBlasHandle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 nRA, nCB, nCA, &_alpha,
                                 ptrA.raw(), nRA,
                                 ptrB.raw(), nCA,
                                 &_beta,
                                 ptr.raw(), nRA,
                                 nMat));
}

template<>
inline void DTensor<float>::addAB(const DTensor<float> &A, const DTensor<float> &B, float alpha, float beta) {
    size_t nMat = A.numMats();
    size_t nRA = A.numRows();
    size_t nCA = A.numCols();
    size_t nCB = B.numCols();
    DTensor<float *> ptrA = A.pointersToMatrices();
    DTensor<float *> ptrB = B.pointersToMatrices();
    DTensor<float *> ptr = pointersToMatrices();
    float _alpha = alpha, _beta = beta;
    gpuErrChk(cublasSgemmBatched(Session::getInstance().cuBlasHandle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 nRA, nCB, nCA, &_alpha,
                                 ptrA.raw(), nRA,
                                 ptrB.raw(), nCA,
                                 &_beta,
                                 ptr.raw(), nRA,
                                 nMat));
}

template<>
inline void DTensor<double>::leastSquaresBatched(DTensor &B) {
    size_t batchSize = numMats();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows)
        throw std::invalid_argument("[Least squares batched] rhs rows does not equal lhs rows");
    if (nColsB != 1)
        throw std::invalid_argument("[Least squares batched] rhs are not vectors");
    if (B.numMats() != batchSize)
        throw std::invalid_argument("[Least squares batched] rhs numMats does not equal lhs numMats");
    if (m_numCols > m_numRows)
        throw std::invalid_argument("[Least squares batched] supports square or tall matrices only");
    int info = 0;
    DTensor<int> infoArray(batchSize);
    DTensor<double *> As = pointersToMatrices();
    DTensor<double *> Bs = B.pointersToMatrices();
    gpuErrChk(cublasDgelsBatched(Session::getInstance().cuBlasHandle(),
                                 CUBLAS_OP_N,
                                 m_numRows,
                                 m_numCols,
                                 nColsB,
                                 As.raw(),
                                 m_numRows,
                                 Bs.raw(),
                                 m_numRows,
                                 &info,
                                 infoArray.raw(),
                                 batchSize));
}

template<>
inline void DTensor<float>::leastSquaresBatched(DTensor &B) {
    size_t batchSize = numMats();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows)
        throw std::invalid_argument("[Least squares batched] rhs rows does not equal lhs rows");
    if (nColsB != 1)
        throw std::invalid_argument("[Least squares batched] rhs are not vectors");
    if (B.numMats() != batchSize)
        throw std::invalid_argument("[Least squares batched] rhs numMats does not equal lhs numMats");
    if (m_numCols > m_numRows)
        throw std::invalid_argument("[Least squares batched] supports square or tall matrices only");
    int info = 0;
    DTensor<int> infoArray(batchSize);
    DTensor<float *> As = pointersToMatrices();
    DTensor<float *> Bs = B.pointersToMatrices();
    gpuErrChk(cublasSgelsBatched(Session::getInstance().cuBlasHandle(),
                                 CUBLAS_OP_N,
                                 m_numRows,
                                 m_numCols,
                                 nColsB,
                                 As.raw(),
                                 m_numRows,
                                 Bs.raw(),
                                 m_numRows,
                                 &info,
                                 infoArray.raw(),
                                 batchSize));
}

template<>
inline DTensor<double> DTensor<double>::getRows(size_t rowsFrom, size_t rowsTo, size_t matIdx) const {
    size_t rowsRangeLength = rowsTo - rowsFrom + 1;
    size_t n = numCols(), m = numRows();
    DTensor<double> rowsOnly(rowsRangeLength, numCols(), 1);
    for (size_t i = 0; i < rowsRangeLength; i++) {
        gpuErrChk(cublasDcopy(Session::getInstance().cuBlasHandle(),
                              n, // # values to copy
                              raw() + rowsFrom + i + matIdx * n * m, m,
                              rowsOnly.raw() + i,
                              rowsRangeLength));
    }
    return rowsOnly;
}

template<>
inline DTensor<float> DTensor<float>::getRows(size_t rowsFrom, size_t rowsTo, size_t matIdx) const {
    size_t rowsRangeLength = rowsTo - rowsFrom + 1;
    size_t n = numCols(), m = numRows();
    DTensor<float> rowsOnly(rowsRangeLength, numCols(), 1);
    for (size_t i = 0; i < rowsRangeLength; i++) {
        gpuErrChk(cublasScopy(Session::getInstance().cuBlasHandle(),
                              n, // # values to copy
                              raw() + rowsFrom + i + matIdx * n * m, m,
                              rowsOnly.raw() + i,
                              rowsRangeLength));
    }
    return rowsOnly;
}

template<typename T>
std::ostream &DTensor<T>::print(std::ostream &out) const {
    size_t nr = m_numRows, nc = m_numCols, nm = m_numMats;
    out << "Tensor [" << m_numRows << " x "
        << m_numCols << " x "
        << m_numMats << "]:" << std::endl;
    std::vector<T> temp;
    download(temp);
    for (size_t k = 0; k < nm; k++) {
        out << ">> layer: " << k << std::endl;
        for (size_t i = 0; i < nr; i++) {
            for (size_t j = 0; j < nc; j++) {
                out << std::setw(10) << temp[nr * (nc * k + j) + i] << ", ";
            }
            out << std::endl;
        }
    }
    return out;
}


/* ================================================================================================
 *  SINGULAR VALUE DECOMPOSITION (SVD)
 * ================================================================================================ */

/**
 * Kernel that counts the number of elements of a vector that are higher than epsilon.
 * @tparam T either float or double
 * @param d_array device array
 * @param n length of device array
 * @param d_count on exit, count of elements (int on device)
 * @param epsilon threshold
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
__global__ void k_countNonzeroSingularValues(const T *d_array, size_t n, unsigned int *d_count, T epsilon) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_array[idx] > epsilon) {
        atomicAdd(d_count, 1);
    }
}

/**
 * Singular value decomposition (SVD) needs a workspace to be setup for cuSolver before factorisation.
 * This object can be setup for a specific type and size of (m,n,1)-tensor (i.e., a matrix).
 * Then, many same-type-(m,n,1)-tensor can be factorised using this object's workspace.
 * @tparam T data type of (m,n,1)-tensor to be factorised (must be float or double)
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
class Svd {

private:

    int m_lwork = -1;  ///< Size of workspace needed for SVD
    DTensor<T> *m_tensor = nullptr;  ///< Pointer to original matrix to be factorised
    std::shared_ptr<DTensor<T>> m_Vtr;  ///< Matrix V' or right singular vectors
    std::shared_ptr<DTensor<T>> m_S;  ///< Diagonal matrix S or singular values
    std::shared_ptr<DTensor<T>> m_U;  ///< Matrix U or left singular vectors
    std::unique_ptr<DTensor<T>> m_workspace;  ///< Workspace for SVD
    std::unique_ptr<DTensor<int>> m_info;  ///< Status code of computation
    std::shared_ptr<DTensor<unsigned int>> m_rank;  ///< Rank of original matrix
    bool m_computeU = false;  ///< Whether to compute U
    bool m_destroyMatrix = true; ///< Whether to sacrifice original matrix

    /**
     * Ensures tensor to factorise contains exactly one matrix, and that matrix is tall.
     * @param mat matrix to be factorised
     */
    void checkMatrix(DTensor<T> &tensor) const {
        if (tensor.numRows() < tensor.numCols()) {
            throw std::invalid_argument("[svd] your matrix is fat (no offence)");
        }
    };

    /**
     * Computes the workspace size required by cuSolver.
     * @param m number of rows of matrix to be factorised
     * @param n number of columns of matrix to be factorised
     */
    void computeWorkspaceSize(size_t m, size_t n);

public:

    /**
     * Constructor.
     * @param mat matrix to be factorised
     * @param computeU whether to compute U (default=false)
     * @param destroyMatrix whether to overwrite original matrix (default=true)
     */
    Svd(DTensor<T> &mat,
        bool computeU = false,
        bool destroyMatrix = true) {
        checkMatrix(mat);
        m_destroyMatrix = destroyMatrix;
        m_tensor = (destroyMatrix) ? &mat : new DTensor<T>(mat);
        m_computeU = computeU;
        size_t m = mat.numRows();
        size_t n = mat.numCols();
        size_t minMN = std::min(m, n);
        size_t nMats = mat.numMats();
        computeWorkspaceSize(m, n);
        m_workspace = std::make_unique<DTensor<T>>(m_lwork, 1, 1);  ///< Allocates required workspace memory
        m_Vtr = std::make_shared<DTensor<T>>(n, n, nMats);
        m_S = std::make_shared<DTensor<T>>(minMN, 1, nMats);
        m_info = std::make_unique<DTensor<int>>(1, 1, nMats);
        m_rank = std::make_unique<DTensor<unsigned int>>(1, 1, nMats, true);
        if (computeU) m_U = std::make_shared<DTensor<T>>(m, m, nMats);
    }

    /**
     * Perform factorisation.
     * Warning: the original matrix is destroyed by default!
     * @return true if factorisation is successful
     */
    bool factorise();

    /**
     * @return diagonal matrix S, or singular values
     */
    DTensor<T> &singularValues() const {
        return *m_S;
    }

    /**
     * @return matrix V', or right singular vectors
     */
    DTensor<T> const &rightSingularVectors() const {
        return *m_Vtr;
    }

    /**
     * @return matrix U, or left singular vectors
     */
    std::optional<std::shared_ptr<DTensor<T>>> leftSingularVectors() const {
        if (!m_computeU) return std::nullopt;
        return m_U;
    }

    /**
     * Destroyer.
     */
    ~Svd() {
        m_lwork = -1;
        if (!m_destroyMatrix && m_tensor) delete m_tensor;
    }


    /**
     * Computes the rank of the original matrices and returns
     * a (1, 1, nMats)-tensor.
     * @param epsilon any numerical value less than epsilon is considered zero
     * @return rank of original matrices
     */
    DTensor<unsigned int> const &rank(T epsilon = 1e-6) const {
        size_t numElS = m_S->numCols() * m_S->numRows();
        for (size_t i = 0; i < m_rank->numMats(); i++) {
            DTensor<T> Si(*m_S, 2, i, i);
            DTensor<unsigned int> rankI(*m_rank, 2, i, i);
            k_countNonzeroSingularValues<T><<<numBlocks(numElS), THREADS_PER_BLOCK>>>(Si.raw(), numElS,
                                                                                      rankI.raw(), epsilon);
        }
        return *m_rank;
    }

};


template<>
inline void Svd<float>::computeWorkspaceSize(size_t m, size_t n) {
    gpuErrChk(cusolverDnSgesvd_bufferSize(Session::getInstance().cuSolverHandle(), m, n, &m_lwork));
}

template<>
inline void Svd<double>::computeWorkspaceSize(size_t m, size_t n) {
    gpuErrChk(cusolverDnDgesvd_bufferSize(Session::getInstance().cuSolverHandle(), m, n, &m_lwork));
}


template<>
inline bool Svd<double>::factorise() {
    size_t m = m_tensor->numRows();
    size_t n = m_tensor->numCols();
    size_t nMats = m_tensor->numMats();
    bool info = true;
    std::unique_ptr<DTensor<double>> Ui;
    for (size_t i = 0; i < nMats; i++) {
        DTensor<double> Ai(*m_tensor, 2, i, i); // tensor A[:, :, i]
        DTensor<double> Si(*m_S, 2, i, i); // S[:, :, i]
        DTensor<double> Vtri(*m_Vtr, 2, i, i); // Vtr[:, :, i]
        if (m_computeU)
            Ui = std::make_unique<DTensor<double>>(*m_U, 2, i, i);
        gpuErrChk(
                cusolverDnDgesvd(Session::getInstance().cuSolverHandle(),
                                 (m_computeU) ? 'A' : 'N', 'A',
                                 m, n,
                                 Ai.raw(), m,
                                 Si.raw(),
                                 (m_computeU) ? Ui->raw() : nullptr, m,
                                 Vtri.raw(), n,
                                 m_workspace->raw(),
                                 m_lwork,
                                 nullptr,  // rwork (used only if SVD fails)
                                 m_info->raw()));
        info = info && ((*m_info)(0, 0, 0) == 0);
    }
    return info;
}

template<>
inline bool Svd<float>::factorise() {
    size_t m = m_tensor->numRows();
    size_t n = m_tensor->numCols();
    size_t nMats = m_tensor->numMats();
    bool info = true;
    std::unique_ptr<DTensor<float>> Ui;
    for (size_t i = 0; i < nMats; i++) {
        DTensor<float> Ai(*m_tensor, 2, i, i); // tensor A[:, :, i]
        DTensor<float> Si(*m_S, 2, i, i); // S[:, :, i]
        DTensor<float> Vtri(*m_Vtr, 2, i, i); // Vtr[:, :, i]
        if (m_computeU)
            Ui = std::make_unique<DTensor<float>>(*m_U, 2, i, i);
        gpuErrChk(
                cusolverDnSgesvd(Session::getInstance().cuSolverHandle(),
                                 (m_computeU) ? 'A' : 'N', 'A',
                                 m, n,
                                 Ai.raw(), m,
                                 Si.raw(),
                                 (m_computeU) ? Ui->raw() : nullptr, m,
                                 Vtri.raw(), n,
                                 m_workspace->raw(),
                                 m_lwork,
                                 nullptr,  // rwork (used only if SVD fails)
                                 m_info->raw()));
        info = info && ((*m_info)(0, 0, 0) == 0);
    }
    return info;
}


/* ================================================================================================
 *  CHOLESKY FACTORISATION (CF)
 * ================================================================================================ */

/**
 * Cholesky factorisation (CF) needs a workspace to be setup for cuSolver before factorisation.
 * This object can be setup for a specific type and size of (m,n,1)-tensor (i.e., a matrix).
 * Then, many same-type-(m,n,1)-tensor can be factorised using this object's workspace
 * @tparam T data type of (m,n,1)-tensor to be factorised (must be float or double)
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
class CholeskyFactoriser {

private:
    int m_workspaceSize = 0;  ///< Size of workspace needed for CF
    std::unique_ptr<DTensor<int>> m_info;  ///< Status code of computation
    std::unique_ptr<DTensor<T>> m_workspace;  ///< Workspace for CF
    DTensor<T> *m_matrix;  ///< Matrix to factorise. Do not destroy!

    /**
     * Computes the workspace size required by cuSolver.
     */
    void computeWorkspaceSize();

public:

    CholeskyFactoriser(DTensor<T> &A) {
        if (A.numMats() > 1) throw std::invalid_argument("[Cholesky] 3D tensors require `CholeskyBatchFactoriser`");
        if (A.numRows() != A.numCols()) throw std::invalid_argument("[Cholesky] Matrix A must be square");
        m_matrix = &A;
        computeWorkspaceSize();
        m_workspace = std::make_unique<DTensor<T>>(m_workspaceSize);
        m_info = std::make_unique<DTensor<int>>(1);
    }

    /**
     * Factorise matrix.
     * @return status code of computation
     */
    int factorise();

    /**
     * Solves for the solution of A \ b using the CF of A.
     * A is the matrix that is factorised and b is the provided matrix.
     * A and b must have compatible dimensions (same number of rows and matrices=1).
     * A must be square (m=n).
     * @param b provided matrix
     * @return status code of computation
     */
    int solve(DTensor<T> &b);

};

template<>
inline void CholeskyFactoriser<double>::computeWorkspaceSize() {
    size_t n = m_matrix->numRows();
    gpuErrChk(cusolverDnDpotrf_bufferSize(Session::getInstance().cuSolverHandle(),
                                          CUBLAS_FILL_MODE_LOWER, n,
                                          nullptr, n, &m_workspaceSize));
}

template<>
inline void CholeskyFactoriser<float>::computeWorkspaceSize() {
    size_t n = m_matrix->numRows();
    gpuErrChk(cusolverDnSpotrf_bufferSize(Session::getInstance().cuSolverHandle(),
                                          CUBLAS_FILL_MODE_LOWER, n,
                                          nullptr, n, &m_workspaceSize));
}

template<>
inline int CholeskyFactoriser<double>::factorise() {
    size_t n = m_matrix->numRows();
    gpuErrChk(cusolverDnDpotrf(Session::getInstance().cuSolverHandle(), CUBLAS_FILL_MODE_LOWER, n,
                               m_matrix->raw(), n,
                               m_workspace->raw(),
                               m_workspaceSize,
                               m_info->raw()));
    return (*m_info)(0);
}


template<>
inline int CholeskyFactoriser<float>::factorise() {
    size_t n = m_matrix->numRows();
    gpuErrChk(cusolverDnSpotrf(Session::getInstance().cuSolverHandle(), CUBLAS_FILL_MODE_LOWER, n,
                               m_matrix->raw(), n,
                               m_workspace->raw(),
                               m_workspaceSize,
                               m_info->raw()));
    return (*m_info)(0);
}

template<>
inline int CholeskyFactoriser<double>::solve(DTensor<double> &rhs) {
    size_t n = m_matrix->numRows();
    gpuErrChk(cusolverDnDpotrs(Session::getInstance().cuSolverHandle(),
                               CUBLAS_FILL_MODE_LOWER,
                               n, 1,
                               m_matrix->raw(), n,
                               rhs.raw(), n,
                               m_info->raw()));
    return (*m_info)(0);
}

template<>
inline int CholeskyFactoriser<float>::solve(DTensor<float> &rhs) {
    size_t n = m_matrix->numRows();
    gpuErrChk(cusolverDnSpotrs(Session::getInstance().cuSolverHandle(),
                               CUBLAS_FILL_MODE_LOWER,
                               n, 1,
                               m_matrix->raw(), n,
                               rhs.raw(), n,
                               m_info->raw()));
    return (*m_info)(0);
}


/* ================================================================================================
 *  QR DECOMPOSITION (QR)
 * ================================================================================================ */

/**
 * QR decomposition (QR) needs a workspace to be setup for cuSolver before factorisation.
 * This object can be setup for a specific type and size of (m,n,1)-tensor (i.e., a matrix).
 * Then, many same-type-(m,n,1)-tensor can be factorised using this object's workspace
 * @tparam T data type of (m,n,1)-tensor to be factorised (must be float or double)
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
class QRFactoriser {

private:
    int m_workspaceSize = 0;  ///< Size of workspace needed for LS
    std::unique_ptr<DTensor<int>> m_info;  ///< Status code of computation
    std::unique_ptr<DTensor<T>> m_householder;  ///< For storing householder reflectors
    std::unique_ptr<DTensor<T>> m_workspace;  ///< Workspace for LS
    DTensor<T> *m_matrix;  ///< Lhs matrix template. Do not destroy!

    /**
     * Computes the workspace size required by cuSolver.
     */
    void computeWorkspaceSize();

public:

    QRFactoriser(DTensor<T> &A) {
        if (A.numMats() > 1) throw std::invalid_argument("[QR] 3D tensors require `leastSquaresBatched`");
        if (A.numRows() < A.numCols()) throw std::invalid_argument("[QR] Matrix A must be tall or square");
        m_matrix = &A;
        computeWorkspaceSize();
        m_workspace = std::make_unique<DTensor<T>>(m_workspaceSize);
        m_householder = std::make_unique<DTensor<T>>(m_matrix->numCols());
        m_info = std::make_unique<DTensor<int>>(1);
    }

    /**
     * Factorise matrix.
     * @return status code of computation
     */
    int factorise();

    /**
     * Solves A \ b using the QR of A.
     * A is the matrix that is factorised and b is the provided matrix.
     * A and b must have compatible dimensions (same number of rows and matrices=1).
     * A must be tall or square (m>=n).
     * @param b provided matrix
     * @return status code of computation
     */
    int leastSquares(DTensor<T> &b);

    /**
     * Populate the given tensors with Q and R.
     * Caution! This is an inefficient method: only to be used for debugging.
     *
     * @param Q matrix Q (preallocated)
     * @param R matrix R (preallocated)
     * @return status code
     *
     * @throws std::invalid_argument if Q or R have invalid dimensions
     */
    int getQR(DTensor<T> &Q, DTensor<T> &R);

};

template<>
inline void QRFactoriser<double>::computeWorkspaceSize() {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    gpuErrChk(cusolverDnDgeqrf_bufferSize(Session::getInstance().cuSolverHandle(),
                                          m, n,
                                          nullptr, m,
                                          &m_workspaceSize));
}

template<>
inline void QRFactoriser<float>::computeWorkspaceSize() {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    gpuErrChk(cusolverDnSgeqrf_bufferSize(Session::getInstance().cuSolverHandle(),
                                          m, n,
                                          nullptr, m,
                                          &m_workspaceSize));
}

template<>
inline int QRFactoriser<double>::factorise() {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    gpuErrChk(cusolverDnDgeqrf(Session::getInstance().cuSolverHandle(),
                               m, n,
                               m_matrix->raw(), m,
                               m_householder->raw(),
                               m_workspace->raw(), m_workspaceSize,
                               m_info->raw()));
    return (*m_info)(0);
}


template<>
inline int QRFactoriser<float>::factorise() {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    gpuErrChk(cusolverDnSgeqrf(Session::getInstance().cuSolverHandle(),
                               m, n,
                               m_matrix->raw(), m,
                               m_householder->raw(),
                               m_workspace->raw(), m_workspaceSize,
                               m_info->raw()));
    return (*m_info)(0);
}

template<>
inline int QRFactoriser<double>::leastSquares(DTensor<double> &rhs) {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    double alpha = 1.;
    gpuErrChk(cusolverDnDormqr(Session::getInstance().cuSolverHandle(),
                               CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 1, n,
                               m_matrix->raw(), m,
                               m_householder->raw(),
                               rhs.raw(), m,
                               m_workspace->raw(), m_workspaceSize,
                               m_info->raw()));
    gpuErrChk(cublasDtrsm(Session::getInstance().cuBlasHandle(),
                          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 1,
                          &alpha,
                          m_matrix->raw(), m,
                          rhs.raw(), m));
    return (*m_info)(0);
}

template<>
inline int QRFactoriser<float>::leastSquares(DTensor<float> &rhs) {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    float alpha = 1.;
    gpuErrChk(cusolverDnSormqr(Session::getInstance().cuSolverHandle(),
                               CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 1, n,
                               m_matrix->raw(), m,
                               m_householder->raw(),
                               rhs.raw(), m,
                               m_workspace->raw(), m_workspaceSize,
                               m_info->raw()));
    gpuErrChk(cublasStrsm(Session::getInstance().cuBlasHandle(),
                          CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 1,
                          &alpha,
                          m_matrix->raw(), m,
                          rhs.raw(), m));
    return (*m_info)(0);
}

template<>
inline int QRFactoriser<double>::getQR(DTensor<double> &Q, DTensor<double> &R) {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    if (Q.numRows() != m || Q.numCols() != n)
        throw std::invalid_argument("[QR] invalid shape of Q.");
    if (R.numRows() != n || R.numCols() != n)
        throw std::invalid_argument("[QR] invalid shape of R.");
    // Initialize Q to 1's on diagonal
    std::vector<double> vecQ(m * n, 0.);
    for (size_t r = 0; r < m; r++) {
        for (size_t c = 0; c < n; c++) {
            if (r == c) { vecQ[r * n + c] = 1.; }
        }
    }
    Q.upload(vecQ, rowMajor);
    // Apply Householder reflectors to compute Q
    gpuErrChk(cusolverDnDormqr(Session::getInstance().cuSolverHandle(),
                               CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, n,
                               m_matrix->raw(), m,
                               m_householder->raw(),
                               Q.raw(), m,
                               m_workspace->raw(), m_workspaceSize,
                               m_info->raw()));
    // Extract upper triangular R
    std::vector<double> vecR(n * n, 0.);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = r; c < n; c++) {
            vecR[r * n + c] = (*m_matrix)(r, c);
        }
    }
    R.upload(vecR, rowMajor);
    return (*m_info)(0);
}

template<>
inline int QRFactoriser<float>::getQR(DTensor<float> &Q, DTensor<float> &R) {
    size_t m = m_matrix->numRows();
    size_t n = m_matrix->numCols();
    if (Q.numRows() != m || Q.numCols() != n)
        throw std::invalid_argument("[QR] invalid shape of Q.");
    if (R.numRows() != n || R.numCols() != n)
        throw std::invalid_argument("[QR] invalid shape of R.");
    // Initialize Q to 1's on diagonal
    std::vector<float> vecQ(m * n, 0.);
    for (size_t r = 0; r < m; r++) {
        for (size_t c = 0; c < n; c++) {
            if (r == c) { vecQ[r * n + c] = 1.; }
        }
    }
    Q.upload(vecQ, rowMajor);
    // Apply Householder reflectors to compute Q
    gpuErrChk(cusolverDnSormqr(Session::getInstance().cuSolverHandle(),
                               CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, n,
                               m_matrix->raw(), m,
                               m_householder->raw(),
                               Q.raw(), m,
                               m_workspace->raw(), m_workspaceSize,
                               m_info->raw()));
    // Extract upper triangular R
    std::vector<float> vecR(n * n, 0.);
    for (size_t r = 0; r < n; r++) {
        for (size_t c = r; c < n; c++) {
            vecR[r * n + c] = (*m_matrix)(r, c);
        }
    }
    R.upload(vecR, rowMajor);
    return (*m_info)(0);
}


/* ================================================================================================
 *  Nullspace (N)
 * ================================================================================================ */

/**
 * The nullspace (N) of a matrix is computed by SVD.
 * The user provides a tensor made of (padded) matrices.
 * Nullspace computes, pads, and stores the nullspace matrices.
 * @tparam T data type (must be float or double)
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
class Nullspace {

private:

    std::unique_ptr<DTensor<T>> m_nullspace;  ///< Stores all nullspace matrices (N)
    std::unique_ptr<DTensor<T>> m_projOp;  ///< Stores all projection operators (N*N')

public:

    /**
     * Constructor (computes nullspace)
     * @param a device tensor
     */
    Nullspace(DTensor<T> &a);

    /**
     * For a given tensor A = (A1, ..., Ak), this returns a
     * tensor NN' = (N1*N1', ..., Nk*Nk'), where the columns of Ni span
     * the kernel of Ai; the matrices Ni are padded with
     * zero columns where necessary.
     * @return NN' = (N1*N1', ..., Nk*Nk')
     */
    DTensor<T> const &nullspace() const {
        return *m_nullspace;
    }

    /**
     * Uses the stored tensor NN' = (N1*N1', ..., Nk*Nk'),
     * of orthogonal matrices, to project the given tensor
     * b = (b1, ..., bk) onto the nullspace of the original tensor.
     * That is, projection zi = Ni * Ni' * bi.
     * The projection zi is stored in bi.
     */
    void project(DTensor<T> &b);
};


template<typename T>
TEMPLATE_CONSTRAINT_REQUIRES_FPX
inline Nullspace<T>::Nullspace(DTensor<T> &a) {
    size_t m = a.numRows(), n = a.numCols(), nMats = a.numMats();
    if (m > n) throw std::invalid_argument("[nullspace] I was expecting a square or fat matrix");
    m_nullspace = std::make_unique<DTensor<T>>(n, n, nMats, true);
    m_projOp = std::make_unique<DTensor<T>>(n, n, nMats, true);
    auto aTranspose = a.tr();
    Svd<T> svd(aTranspose, true);
    svd.factorise();

    DTensor<unsigned int> devRankA = svd.rank();
    std::vector<unsigned int> hostRankA;
    devRankA.download(hostRankA);

    std::optional<std::shared_ptr<DTensor<T>>> leftSingValsOptional = svd.leftSingularVectors();
    assert(leftSingValsOptional); // make sure left SVs were computed
    std::shared_ptr<DTensor<T>> leftSingVals = leftSingValsOptional.value();
    for (size_t i = 0; i < nMats; i++) { // for each matrix
        // Slice the matrix of left SVs to get the matrix that spans
        // the nullspace of a[:, :, i]
        unsigned int rankAi = hostRankA[i];
        unsigned int nullityI = n - rankAi; // nullity(A[:, :, i])
        if (nullityI == 0) continue;
        DTensor<T> Ui(*leftSingVals, 2, i, i); // leftSingVals[:, :, i]
        DTensor<T> nullityMatrixI(Ui, 1, n - nullityI, n - 1); // leftSingVals[:, <range>, i]
        // Copy to destination
        DTensor<T> currNsDst(*m_nullspace, 2, i, i);
        DTensor<T> currNsColSlice(currNsDst, 1, 0, nullityI - 1);
        nullityMatrixI.deviceCopyTo(currNsColSlice);
        DTensor<T> currProjOpDst(*m_projOp, 2, i, i);
        DTensor<T> Ntr = currNsDst.tr();
        currProjOpDst.addAB(currNsDst, Ntr);
    }
}

template<typename T>
TEMPLATE_CONSTRAINT_REQUIRES_FPX
inline void Nullspace<T>::project(DTensor<T> &b) {
    b.addAB(*m_projOp, b, 1, 0);
}


/* ================================================================================================
 *  CHOLESKY BATCH FACTORISATION (CBF)
 * ================================================================================================ */

/**
 * Cholesky Factorisation (CF) for batches of matrices (tensors).
 * This object can factorise and then solve,
 * or you can provide pre-computed Cholesky lower-triangular matrices and then solve.
 * @tparam T data type of tensor (must be float or double)
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
class CholeskyBatchFactoriser {

private:
    DTensor<T> *m_matrix;  ///< Matrix to factorise or lower-triangular decomposed matrix
    size_t m_numRows = 0;  ///< Number of rows in `m_matrix`
    size_t m_numMats = 0;  ///< Number of matrices in `m_matrix`
    std::unique_ptr<DTensor<int>> m_deviceInfo;  ///< Info from cusolver functions
    bool m_factorisationDone = false;  ///< Whether `m_matrix` holds original or lower-triangular matrix

public:
    CholeskyBatchFactoriser() = delete;

    /**
     * Constructor
     * @param A either matrices to be factorised or lower-triangular Cholesky decomposition matrices
     * @param factorised whether A is the original matrices or the factorised ones (default=original matrices)
     */
    CholeskyBatchFactoriser(DTensor<T> &A, bool factorised = false) : m_factorisationDone(factorised) {
        if (A.numRows() != A.numCols()) throw std::invalid_argument("[CholeskyBatch] A must be square");
        m_matrix = &A;
        m_numRows = A.numRows();
        m_numMats = A.numMats();
        m_deviceInfo = std::make_unique<DTensor<int>>(m_numMats, 1, 1, true);
    }

    /**
     * Factorise all matrices in tensor (batched).
     */
    void factorise();

    /**
     * Solve the system A\b using Cholesky lower-triangular matrix of A (batched).
     * @param b vector in system Ax=b to be solved where x is the solution
     */
    void solve(DTensor<T> &b);

};

template<>
inline void CholeskyBatchFactoriser<double>::factorise() {
    if (m_factorisationDone) return;
    DTensor<double *> ptrA = m_matrix->pointersToMatrices();
    gpuErrChk(cusolverDnDpotrfBatched(Session::getInstance().cuSolverHandle(),
                                      CUBLAS_FILL_MODE_LOWER,
                                      m_numRows,
                                      ptrA.raw(),
                                      m_numRows,
                                      m_deviceInfo->raw(),
                                      m_numMats));
    m_factorisationDone = true;
}

template<>
inline void CholeskyBatchFactoriser<float>::factorise() {
    if (m_factorisationDone) return;
    DTensor<float *> ptrA = m_matrix->pointersToMatrices();
    gpuErrChk(cusolverDnSpotrfBatched(Session::getInstance().cuSolverHandle(),
                                      CUBLAS_FILL_MODE_LOWER,
                                      m_numRows,
                                      ptrA.raw(),
                                      m_numRows,
                                      m_deviceInfo->raw(),
                                      m_numMats));
    m_factorisationDone = true;
}

template<>
inline void CholeskyBatchFactoriser<double>::solve(DTensor<double> &b) {
    if (!m_factorisationDone) throw std::logic_error("[CholeskyBatchSolve] no factor to solve with");
    if (m_numRows != b.numRows() || m_numMats != b.numMats()) {
        throw std::invalid_argument("[CholeskyBatchSolve] A and b incompatible");
    }
    if (b.numCols() != 1) throw std::invalid_argument("[CholeskyBatchSolve] only supports `b` with one column");
    DTensor<double *> ptrA = m_matrix->pointersToMatrices();
    DTensor<double *> ptrB = b.pointersToMatrices();
    gpuErrChk(cusolverDnDpotrsBatched(Session::getInstance().cuSolverHandle(),
                                      CUBLAS_FILL_MODE_LOWER,
                                      m_numRows,
                                      1,  ///< only supports rhs = 1
                                      ptrA.raw(),
                                      m_numRows,
                                      ptrB.raw(),
                                      m_numRows,
                                      m_deviceInfo->raw(),
                                      m_numMats));
}

template<>
inline void CholeskyBatchFactoriser<float>::solve(DTensor<float> &b) {
    if (!m_factorisationDone) throw std::logic_error("[CholeskyBatchSolve] no factor to solve with");
    if (m_numRows != b.numRows() || m_numMats != b.numMats()) {
        throw std::invalid_argument("[CholeskyBatchSolve] A and b incompatible");
    }
    if (b.numCols() != 1) throw std::invalid_argument("[CholeskyBatchSolve] only supports `b` with one column");
    DTensor<float *> ptrA = m_matrix->pointersToMatrices();
    DTensor<float *> ptrB = b.pointersToMatrices();
    gpuErrChk(cusolverDnSpotrsBatched(Session::getInstance().cuSolverHandle(),
                                      CUBLAS_FILL_MODE_LOWER,
                                      m_numRows,
                                      1,  ///< only supports rhs = 1
                                      ptrA.raw(),
                                      m_numRows,
                                      ptrB.raw(),
                                      m_numRows,
                                      m_deviceInfo->raw(),
                                      m_numMats));
}



/* ================================================================================================
 *  GIVENS ANNIHILATOR
 * ================================================================================================ */

/**
 * GivensAnnihilator is used to apply a left Givens rotation that
 * makes a particular element (k, j) of a matrix zero by applying
 * an appropriate Givens rotation G(i, k, c, s).
 *
 * @tparam T data type of tensor (must be float or double)
 */
TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
class GivensAnnihilator {

private:
    DTensor<T> *m_matrix;
    /**
     * Auxiliary memory on the device of length 3 used to store
     * rhypot(xij, xkj), cos θ, and sin θ.
     */
    std::unique_ptr<DTensor<T>> m_d_rhyp_cos_sin;

    void init() {
        m_d_rhyp_cos_sin = std::make_unique<DTensor<T>>(3);
    }

public:

    GivensAnnihilator() {
        init();
    }

    /**
     * Constructor of GivensAnnihilator
     * @param a matrix
     * @throws std::invalid_argument if a.numMats() > 1
     */
    GivensAnnihilator(DTensor<T> &a) {
        if (a.numMats() > 1) {
            throw std::invalid_argument("[GivensAnnihilator] tensors (numMats > 1) not supported");
        }
        m_matrix = &a;
        init();
    }

    /**
     * Set the reference to a matrix; this way the current
     * object can be reused
     *
     * @param a
     */
    void setMatrix(DTensor<T> &a) {
        if (a.numMats() > 1) {
            throw std::invalid_argument("[GivensAnnihilator] tensors (numMats > 1) not supported");
        }
        m_matrix = &a;
    }

    /**
     * Applies a left Givens rotation G(i, k, c, s) that eliminates
     * the (k, j) element of the given matrix.
     *
     * @param i row index i
     * @param k row index k
     * @param j column index j
     *
     * @throws std::invalid_argument if i, k, or j are out of bounds
     */
    void annihilate(size_t i, size_t k, size_t j);

};

TEMPLATE_WITH_TYPE_T TEMPLATE_CONSTRAINT_REQUIRES_FPX
__global__ void k_givensAnnihilateRHypot(const T *data,
                                         T *res,
                                         size_t i, size_t k, size_t j,
                                         size_t nRows) {
    T xij = data[i + j * nRows];
    T xkj = data[k + j * nRows];
    res[0] = rhypot(xij, xkj);
    res[1] = xij * (*res); // cos
    res[2] = xkj * (*res); // -sin
}

template<typename T>
inline void GivensAnnihilator<T>::annihilate(size_t i, size_t k, size_t j) {
    /* A few checks */
    size_t nR = m_matrix->numRows(), nC = m_matrix->numCols();
    if (i >= nR or k >= nR) throw std::invalid_argument("[GivensAnnihilator::annihilate] invalid row index");
    if (j >= nC) std::invalid_argument("[GivensAnnihilator::annihilate] invalid column index j");

    /*
     * Pass cosine and sine as device pointers
     * (Avoid having to download first)
     */
    gpuErrChk(cublasSetPointerMode(Session::getInstance().cuBlasHandle(), CUBLAS_POINTER_MODE_DEVICE));

    /* Useful definitions */
    T *aux = m_d_rhyp_cos_sin->raw();
    T *matData = m_matrix->raw();

    /* Call kernel to determine 1/sqrt(Ai^2 + Ak^2) */
    k_givensAnnihilateRHypot<<<1, 1>>>(m_matrix->raw(), aux, i, k, j, nR);

    /* Apply Givens rotation */
    m_matrix->applyLeftGivensRotation(i, k, aux + 1, aux + 2);

    /* Change back to default behaviour */
    gpuErrChk(cublasSetPointerMode(Session::getInstance().cuBlasHandle(), CUBLAS_POINTER_MODE_HOST));
}


#endif
