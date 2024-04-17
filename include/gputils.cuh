#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <memory>
#include <optional>
#include <source_location>

#ifndef GPUTILS_CUH
#define GPUTILS_CUH

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), std::source_location::current()); }

template<typename T>
inline void gpuAssert(T code, std::source_location loc, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << loc.file_name() << ", line: " << loc.line() << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublas error. Name: " << cublasGetStatusName(code)
                      << ", string: " << cublasGetStatusString(code)
                      << ", file: " << loc.file_name() << ", line: " << loc.line() << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cusolverStatus_t>) {
        if (code != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "cusolver error. Status: " << code
                      << ", file: " << loc.file_name() << ", line: " << loc.line() << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}

#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

/* ------------------------------------------------------------------------------------
 *  Context
 * ------------------------------------------------------------------------------------ */


class Context {
public:
    static Context &getInstance() {
        static Context instance;
        return instance;
    }

private:
    Context() {
        gpuErrChk(cublasCreate(&m_cublasHandle));
        gpuErrChk(cusolverDnCreate(&m_cusolverHandle));
    }

    ~Context() {
        gpuErrChk(cublasDestroy(m_cublasHandle));
        gpuErrChk(cusolverDnDestroy(m_cusolverHandle));
    }

    cublasHandle_t m_cublasHandle;
    cusolverDnHandle_t m_cusolverHandle;


public:
    Context(Context const &) = delete;

    void operator=(Context const &) = delete;

    cublasHandle_t &cuBlasHandle() { return m_cublasHandle; }

    cusolverDnHandle_t &cuSolverHandle() { return m_cusolverHandle; }
};


/* ------------------------------------------------------------------------------------
*  Convert between row- and column-major ordering of vector-stored matrices
* ------------------------------------------------------------------------------------ */

template<typename T>
static void row2col(const std::vector<T> &srcRow, std::vector<T> &dstCol, size_t numRows, size_t numCols) {
    if (numRows * numCols != srcRow.size()) std::cerr << "row2col dimension mismatch" << "\n";
    dstCol.resize(srcRow.size());
    std::vector<T> copySrc(srcRow);
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            dstCol[c * numRows + r] = copySrc[r * numCols + c];
        }
    }
}

template<typename T>
static void col2row(const std::vector<T> &srcCol, std::vector<T> &dstRow, size_t numRows, size_t numCols) {
    if (numRows * numCols != srcCol.size()) std::cerr << "col2row dimension mismatch" << "\n";
    dstRow.resize(srcCol.size());
    std::vector<T> copySrc(srcCol);
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            dstRow[r * numCols + c] = copySrc[c * numRows + r];
        }
    }
}

/* ------------------------------------------------------------------------------------
 *  Device Vector
 * ------------------------------------------------------------------------------------ */

/**
 * DeviceVector is a unique_ptr-type entity for device data.
 */
template<typename TElement>
class DeviceVector {

private:
    /** Pointer to device data */
    TElement *m_d_data = nullptr;
    /** Number of allocated elements */
    size_t m_numAllocatedElements = 0;
    bool m_doDestroy = false;

    bool destroy() {
        if (!m_doDestroy) return false;
        if (m_d_data) cudaFree(m_d_data);
        m_numAllocatedElements = 0;
        m_d_data = nullptr;
        return true;
    }

    /**
     * Fetches just one value from the device
     *
     * Use sparingly
     *
     * @param i index
     * @return entry of array at index i
     */
    TElement fetchElementFromDevice(size_t i);

public:

    /**
     * Constructs a DeviceVector object
     */
    DeviceVector() = default;

    /**
     * Constructs a DeviceVector object and allocates
     * memory on the device for n elements
     */
    DeviceVector(size_t n) {
        allocateOnDevice(n);
    }

    /**
     * Take a slice of another DeviceVector
     *
     * @param other other device vector
     * @param from start (index)
     * @param to end (index)
     */
    DeviceVector(DeviceVector &other, size_t from, size_t to) {
        m_doDestroy = false;
        m_numAllocatedElements = to - from + 1;
        m_d_data = other.m_d_data + from;
    }

    /**
     * Copy constructor
     * @param other
     */
    DeviceVector(DeviceVector &other) {
        allocateOnDevice(other.m_numAllocatedElements);
        cudaMemcpy(m_d_data,
                   other.raw(),
                   m_numAllocatedElements * sizeof(TElement),
                   cudaMemcpyDeviceToDevice);
        m_doDestroy = true;
    }

    /**
     * Create device vector from host vector.
     * This allocates memory on the device and copies the host data.
     *
     * @param vec host vector
     */
    DeviceVector(const std::vector<TElement> &vec) {
        allocateOnDevice(vec.size());
        upload(vec);
    }

    /**
     * Destructor - frees the device memory
     */
    ~DeviceVector() {
        destroy();
    }

    /**
     * Allocates memory on the device for `size` elements
     *
     * @param size number of elements
     * @return true if and only if no errors occurred during
     *         the memory allocation
     */
    bool allocateOnDevice(size_t size);

    /**
     * Size of allocated memory space on the device
     */
    size_t capacity() const {
        return m_numAllocatedElements;
    }


    /**
     * Upload array of data to device
     *
     * Note that if the allocated memory is insufficient,
     * it will be attempted to allocate new memory on the
     * device after freeing the previously allocated memory.
     *
     * @param dataArray pointer to array of data
     * @param size size of array
     * @return true iff the uploading is successful
     */
    bool upload(const TElement *dataArray, size_t size);


    /**
     * Uploads a the data of vector to the device
     *
     * @param vec vector to be uploaded
     * @return true iff the uploading is successful
     */
    bool upload(const std::vector<TElement> &vec) {
        return upload(vec.data(), vec.size());
    }

    /**
     * Returns the raw pointer to the device data
     */
    TElement *raw() {
        return m_d_data;
    }

    /**
     * Downloads the device data to a provided host
     * memory position. It is assumed the memory position
     * on the host is appropriately allocated for the
     * device data to be copied.
     *
     * @param hostData destination memory position on host
     */
    void download(TElement *hostData) const;

    /**
     * Download the device data to a vector
     *
     * @param vec
     */
    void download(std::vector<TElement> &vec) const;


    /**
     * Copy data to another memory position on the device.
     *
     * @param elsewhere destination
     */
    void deviceCopyTo(DeviceVector<TElement> &elsewhere);

    /**
     * Prints device vector to an output stream
     * @param out
     * @param data
     * @return
     */
    friend std::ostream &operator<<(std::ostream &out, const DeviceVector<TElement> &data) {
        out << "DeviceVector [" << data.m_numAllocatedElements << "]:" << std::endl;
        std::vector<TElement> temp(data.m_numAllocatedElements);
        cudaMemcpy(temp.data(),
                   data.m_d_data,
                   data.m_numAllocatedElements * sizeof(TElement),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < data.m_numAllocatedElements - 1; i++) {
            out << std::setw(10) << temp[i] << std::endl;
        }
        out << std::setw(10) << temp[data.m_numAllocatedElements - 1] << std::endl;
        return out;
    }

    TElement operator()(size_t i) {
        return fetchElementFromDevice(i);
    }

    /**
     * Add another vector to the current vector (element-wise)
     * @param rhs
     * @return
     */
    DeviceVector &operator+=(const DeviceVector &rhs);

    /**
     * Subtract from the current vector another vector (element-wise)
     * @param rhs
     * @return
     */
    DeviceVector &operator-=(const DeviceVector &rhs);

    /**
     * Inner product between two vectors
     * @param rhs
     * @return
     */
    TElement operator*(const DeviceVector &rhs) const;

    /**
     * Scalar multiplication
     * @param scalar
     * @return
     */
    DeviceVector &operator*=(TElement scalar);  // CLion warns `not implemented`, but it is

    friend DeviceVector operator+(DeviceVector &firstVector, const DeviceVector &secondVector) {
        DeviceVector resultVec(firstVector.capacity());
        firstVector.deviceCopyTo(resultVec);
        resultVec += secondVector;
        return resultVec;
    }

    friend DeviceVector operator-(DeviceVector &firstVector, const DeviceVector &secondVector) {
        DeviceVector resultVec(firstVector.capacity());
        firstVector.deviceCopyTo(resultVec);
        resultVec -= secondVector;
        return resultVec;
    }

    /**
     * Scalar multiplication
     * @param firstVector
     * @param secondVector
     * @return
     */
    friend DeviceVector operator*(const TElement firstVector, DeviceVector &secondVector) {
        DeviceVector resultVec(secondVector.capacity());
        secondVector.deviceCopyTo(resultVec);
        resultVec *= firstVector;
        return resultVec;
    }

    /**
     * Euclidean norm
     * @return
     */
    TElement norm2() const;

    /**
     * Sum of the elements of the vector
     * @return
     */
    TElement norm1() const;


}; /* end of class */

template<>
inline DeviceVector<float> &DeviceVector<float>::operator+=(const DeviceVector<float> &rhs) {
    const float alpha = 1.;
    gpuErrChk(cublasSaxpy(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1));
    return *this;
}

template<>
inline DeviceVector<double> &DeviceVector<double>::operator+=(const DeviceVector<double> &rhs) {
    const double alpha = 1.;
    gpuErrChk(cublasDaxpy(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1));
    return *this;
}

template<>
inline DeviceVector<float> &DeviceVector<float>::operator*=(float scalar) {
    float alpha = scalar;
    gpuErrChk(cublasSscal(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, m_d_data, 1));
    return *this;
}

template<>
inline DeviceVector<double> &DeviceVector<double>::operator*=(double scalar) {
    double alpha = scalar;
    gpuErrChk(cublasDscal(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, m_d_data, 1));
    return *this;
}

template<>
inline DeviceVector<float> &DeviceVector<float>::operator-=(const DeviceVector<float> &rhs) {
    const float alpha = -1.;
    gpuErrChk(cublasSaxpy(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1,
                          m_d_data, 1));
    return *this;
}

template<>
inline DeviceVector<double> &DeviceVector<double>::operator-=(const DeviceVector<double> &rhs) {
    const double alpha = -1.;
    gpuErrChk(cublasDaxpy(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1,
                          m_d_data, 1));
    return *this;
}

template<>
inline float DeviceVector<float>::operator*(const DeviceVector<float> &rhs) const {
    float inn_prod;
    gpuErrChk(cublasSdot(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, rhs.m_d_data, 1,
                         &inn_prod));
    return inn_prod;
}

template<>
inline double DeviceVector<double>::operator*(const DeviceVector<double> &rhs) const {
    double inn_prod;

    gpuErrChk(cublasDdot(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, rhs.m_d_data, 1,
                         &inn_prod));
    return inn_prod;
}

template<>
inline float DeviceVector<float>::norm2() const {
    float the_norm;

    gpuErrChk(cublasSnrm2(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_norm));
    return the_norm;
}

template<>
inline double DeviceVector<double>::norm2() const {
    double the_norm;

    gpuErrChk(cublasDnrm2(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_norm));
    return the_norm;
}

template<>
inline float DeviceVector<float>::norm1() const {
    float the_sum;

    gpuErrChk(cublasSasum(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_sum));
    return the_sum;
}

template<>
inline double DeviceVector<double>::norm1() const {
    double nrm1;

    gpuErrChk(cublasDasum(Context::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &nrm1));
    return nrm1;
}

template<typename TElement>
TElement DeviceVector<TElement>::fetchElementFromDevice(size_t i) {
    if (i >= capacity()) throw std::out_of_range("Uh oh! Index out of bounds");
    DeviceVector<TElement> d_element(*this, i, i);
    TElement xi[1];
    d_element.download(xi);
    return xi[0];
}

template<typename TElement>
bool DeviceVector<TElement>::allocateOnDevice(size_t size) {
    if (size <= 0) return false;
    if (size <= m_numAllocatedElements) return true;
    destroy();
    m_doDestroy = true;
    size_t buffer_size = size * sizeof(TElement);
    bool cudaStatus = cudaMalloc(&m_d_data, buffer_size);
    if (cudaStatus != cudaSuccess) return false;
    m_numAllocatedElements = size;
    return true;
}

template<typename TElement>
bool DeviceVector<TElement>::upload(const TElement *dataArray, size_t size) {
    if (!this->allocateOnDevice(size)) return false;
    if (size <= m_numAllocatedElements) {
        size_t buffer_size = size * sizeof(TElement);
        gpuErrChk(cudaMemcpy(m_d_data, dataArray, buffer_size, cudaMemcpyHostToDevice));
    }
    return true;
}

template<typename TElement>
void DeviceVector<TElement>::deviceCopyTo(DeviceVector<TElement> &elsewhere) {
    elsewhere.allocateOnDevice(m_numAllocatedElements);
    gpuErrChk(cudaMemcpy(elsewhere.raw(),
                         m_d_data,
                         m_numAllocatedElements * sizeof(TElement),
                         cudaMemcpyDeviceToDevice));
}

template<typename TElement>
void DeviceVector<TElement>::download(TElement *hostData) const {
    gpuErrChk(cudaMemcpy(hostData,
                         m_d_data,
                         m_numAllocatedElements * sizeof(TElement),
                         cudaMemcpyDeviceToHost));
}

template<typename TElement>
void DeviceVector<TElement>::download(std::vector<TElement> &vec) const {
    vec.reserve(m_numAllocatedElements);
    gpuErrChk(cudaMemcpy(vec.data(),
                         m_d_data,
                         m_numAllocatedElements * sizeof(TElement),
                         cudaMemcpyDeviceToHost));
}

/* ------------------------------------------------------------------------------------
 *  Device Matrix
 * ------------------------------------------------------------------------------------ */

/**
 * Storage mode for the data of a matrix
 */
enum MatrixStorageMode {
    columnMajor, ///< column major storage (preferred/default)
    rowMajor ///< row major storage
};


/**
 * Device matrix
 * @tparam TElement
 */
template<typename TElement>
class DeviceMatrix {

private:
    // the data is always stored in CM format
    DeviceVector<TElement> *m_vec = nullptr;  ///< stores all useful memory
    size_t m_numRows = 0;  ///< number of rows
    bool m_doDeleteVecMemory = true;

    /**
     *
     */
    void destroy() {
        m_numRows = 0;
        if (m_vec && m_doDeleteVecMemory) delete m_vec;
    }

public:

    /**
     *
     * @param numRows
     * @param numCols
     */
    DeviceMatrix(size_t numRows, size_t numCols) {
        m_numRows = numRows;
        m_vec = new DeviceVector<TElement>(numRows * numCols);
    }

    /**
     *
     * @param numRows
     * @param vec
     * @param mode
     */
    DeviceMatrix(size_t numRows,
                 const std::vector<TElement> &vec,
                 MatrixStorageMode mode = MatrixStorageMode::columnMajor) {
        size_t numel = vec.size();
        m_numRows = numRows;
        if (numel % numRows != 0) throw std::invalid_argument("impossible dimensions");
        size_t numCols = numel / numRows;
        if (mode == MatrixStorageMode::rowMajor) {
            std::vector<TElement> vec_cm(numel);
            row2col(vec, vec_cm, numRows, numCols);  // to column-major
            m_vec = new DeviceVector<TElement>(vec_cm);
        } else {
            m_vec = new DeviceVector<TElement>(vec);
        }
    }

    DeviceMatrix(const DeviceMatrix &other) {
        m_numRows = other.m_numRows;
        m_vec = new DeviceVector<TElement>(*other.m_vec);
    }

    // SLICE!
    DeviceMatrix(DeviceMatrix &other, size_t colFrom, size_t colTo) {
        m_numRows = other.m_numRows;
        size_t start = colFrom * m_numRows;
        size_t finish = (colTo + 1) * m_numRows - 1;
        m_vec = new DeviceVector<TElement>(*other.m_vec, start, finish);
    }

    DeviceMatrix(DeviceVector<TElement> &other) {
        m_numRows = other.capacity();
        m_doDeleteVecMemory = false;
        m_vec = &other;
    }

    DeviceMatrix getRows(size_t rowsFrom, size_t rowsTo);


    /**
     *
     * @param vec
     * @param numRows
     * @param mode
     */
    void upload(const std::vector<TElement> &vec,
                size_t numRows,
                MatrixStorageMode mode = MatrixStorageMode::columnMajor) {
        size_t n = vec.size();
        if (n % numRows != 0) throw std::invalid_argument("impossible dimensions");
        size_t numCols = n / numRows;
        if (mode == MatrixStorageMode::rowMajor) {
            std::vector<TElement> vec_cm(n);
            row2col(vec, vec_cm, numRows, numCols);  // to column-major
            m_vec->upload(vec_cm);
        } else {
            m_vec->upload(vec);
        }
    }

    TElement *raw() {
        return m_vec->raw();
    }

    /**
     *
     */
    ~DeviceMatrix() {
        destroy();
    }

    /**
     * Returns a vector, which is a shallow copy of the matrix elements
     * as a vector. This means that editing this vector, will change the
     * elements of the matrix. Note that the data is stored in column-major
     * format.
     *
     * @return
     */
    DeviceVector<TElement> asVector() {
        size_t numel = m_vec->capacity();
        DeviceVector<TElement> vectorisation(*m_vec, 0, numel - 1);
        return vectorisation;
    }

    /**
     * Number of rows
     * @return
     */
    size_t numRows() const {
        return m_numRows;
    }

    /**
     * Number of columns
     * @return
     */
    size_t numCols() const {
        return m_vec->capacity() / m_numRows;
    }

    DeviceMatrix<TElement> tr();


    TElement operator()(size_t i, size_t j) {
        size_t m = numRows();
        if (i >= m) throw std::out_of_range("Uh oh! i >= number of rows");
        if (j >= numCols()) throw std::out_of_range("Uh oh! j >= number of columns");
        return (*m_vec)(j * m + i);
    }

    /**
     *
     * @param rhs
     * @return
     */
    DeviceMatrix &operator+=(const DeviceMatrix &rhs);

    /**
     *
     * @param rhs
     * @return
     */
    DeviceMatrix &operator-=(const DeviceMatrix &rhs);

    /**
     *
     * @param scalar
     * @return
     */
    DeviceMatrix &operator*=(TElement scalar);  // CLion warns `not implemented`, but it is

    /**
     * Matrix-vector multiplication (vec = A * b)
     * @param A LHS matrix
     * @param b RHS vector
     * @return resulting vector (allocates fresh memory)
     */
    friend DeviceVector<float> operator*(DeviceMatrix &A, DeviceVector<float> &b) {
        size_t nRowsA = A.numRows();
        size_t nColsA = b.capacity();
        float alpha = 1.;
        float beta = 0.;
        DeviceVector<float> resultVector(nRowsA);
        // TODO use addAB in this implementation

        gpuErrChk(cublasSgemv(Context::getInstance().cuBlasHandle(),
                              CUBLAS_OP_N,
                              nRowsA,
                              nColsA,
                              &alpha,
                              A.m_vec->raw(),
                              nRowsA,
                              b.raw(),
                              1,
                              &beta,
                              resultVector.raw(),
                              1));
        return resultVector;
    }

    friend DeviceVector<double> operator*(DeviceMatrix &A, DeviceVector<double> &b) {
        size_t nRowsA = A.numRows();
        size_t nColsA = b.capacity();
        double alpha = 1.;
        double beta = 0.;

        DeviceVector<double> resultVector(nRowsA);
        gpuErrChk(cublasDgemv(Context::getInstance().cuBlasHandle(),
                              CUBLAS_OP_N,
                              nRowsA,
                              nColsA,
                              &alpha,
                              A.m_vec->raw(),
                              nRowsA,
                              b.raw(),
                              1,
                              &beta,
                              resultVector.raw(),
                              1));
        return resultVector;
    }

    friend DeviceMatrix operator+(DeviceMatrix &first, const DeviceMatrix &second) {
        DeviceMatrix resultVec(first.numRows(), first.numCols());
        first.m_vec->deviceCopyTo(*resultVec.m_vec);
        resultVec += second;
        return resultVec;
    }

    friend DeviceMatrix operator-(DeviceMatrix &first, const DeviceMatrix &second) {
        DeviceMatrix resultVec(first.numRows(), first.numCols());
        first.m_vec->deviceCopyTo(*resultVec.m_vec);
        resultVec -= second;
        return resultVec;
    }

    /**
     * C = AB
     */
    friend DeviceMatrix operator*(DeviceMatrix &A, const DeviceMatrix &B) {
        size_t nRowsA = A.numRows();
        size_t nColsA = A.numCols();
        size_t nColsB = B.numCols();
        DeviceMatrix resultMatrix(nRowsA, nColsB);
        resultMatrix.addAB(A, B, 1., 0.);
        return resultMatrix;
    }

    /**
     * C <- beta C + alpha AB
     */
    void addAB(const DeviceMatrix &A, const DeviceMatrix &B, TElement alpha = 1., TElement beta = 1.);

    /**
     *
     * @param out
     * @param data
     * @return
     */
    friend std::ostream &operator<<(std::ostream &out, const DeviceMatrix<TElement> &data) {
        size_t numel = data.m_vec->capacity();
        size_t nr = data.m_numRows;
        size_t nc = numel / data.m_numRows;
        out << "DeviceMatrix [" << nr << " x " << nc << "]:" << std::endl;
        std::vector<TElement> temp;
        data.m_vec->download(temp);
        for (size_t i = 0; i < nr; i++) {
            for (size_t j = 0; j < nc; j++) {
                out << std::setw(10) << temp[j * nr + i] << ", ";
            }
            out << std::endl;
        }
        return out;
    }

};

template<>
DeviceMatrix<float> DeviceMatrix<float>::tr() {
    size_t m = numRows();
    size_t n = numCols();
    DeviceMatrix<float> transpose(n, m);
    float alpha = 1.0f, beta = 0;

    gpuErrChk(cublasSgeam(Context::getInstance().cuBlasHandle(),
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          n, m,
                          &alpha, m_vec->raw(), m,
                          &beta, nullptr, n,
                          transpose.raw(), n));
    return transpose;
}

template<>
DeviceMatrix<double> DeviceMatrix<double>::tr() {
    size_t m = numRows();
    size_t n = numCols();
    DeviceMatrix<double> transpose(n, m);
    double alpha = 1.0f, beta = 0;

    gpuErrChk(cublasDgeam(Context::getInstance().cuBlasHandle(),
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          n, m,
                          &alpha, m_vec->raw(), m,
                          &beta, nullptr, n,
                          transpose.raw(), n));
    return transpose;
}

template<>
inline DeviceMatrix<double> DeviceMatrix<double>::getRows(size_t rowsFrom, size_t rowsTo) {
    size_t rowsRangeLength = rowsTo - rowsFrom + 1;
    size_t n = numCols(), m = numRows();
    DeviceMatrix<double> rowsOnly(rowsRangeLength, numCols());

    for (size_t i = 0; i < rowsRangeLength; i++) {
        gpuErrChk(cublasDcopy(Context::getInstance().cuBlasHandle(),
                              n, // # values to copy
                              m_vec->raw() + rowsFrom + i, m,
                              rowsOnly.raw() + i,
                              rowsRangeLength));
    }
    return rowsOnly;
}

template<>
inline DeviceMatrix<float> DeviceMatrix<float>::getRows(size_t rowsFrom, size_t rowsTo) {
    size_t rowsRangeLength = rowsTo - rowsFrom + 1;
    size_t n = numCols(), m = numRows();

    DeviceMatrix<float> rowsOnly(rowsRangeLength, numCols());
    for (size_t i = 0; i < rowsRangeLength; i++) {
        gpuErrChk(cublasScopy(Context::getInstance().cuBlasHandle(),
                              n, // # values to copy
                              m_vec->raw() + rowsFrom + i, m,
                              rowsOnly.raw() + i,
                              rowsRangeLength));
    }
    return rowsOnly;
}

template<>
inline DeviceMatrix<float> &DeviceMatrix<float>::operator+=(const DeviceMatrix<float> &rhs) {
    *m_vec += *rhs.m_vec;
    return *this;
}

template<>
inline DeviceMatrix<double> &DeviceMatrix<double>::operator+=(const DeviceMatrix<double> &rhs) {
    *m_vec += *rhs.m_vec;
    return *this;
}

template<>
inline DeviceMatrix<float> &DeviceMatrix<float>::operator-=(const DeviceMatrix<float> &rhs) {
    *m_vec -= *rhs.m_vec;
    return *this;
}

template<>
inline DeviceMatrix<double> &DeviceMatrix<double>::operator-=(const DeviceMatrix<double> &rhs) {
    *m_vec -= *rhs.m_vec;
    return *this;
}

template<>
inline DeviceMatrix<float> &DeviceMatrix<float>::operator*=(float scalar) {
    *m_vec *= scalar;
    return *this;
}

template<>
inline DeviceMatrix<double> &DeviceMatrix<double>::operator*=(double scalar) {
    *m_vec *= scalar;
    return *this;
}

template<>
inline void DeviceMatrix<float>::addAB(const DeviceMatrix &A, const DeviceMatrix &B, float alpha, float beta) {
    size_t nColsC = this->numCols();
    size_t nColsA = A.numCols();
    if (A.numRows() != m_numRows || B.numCols() != nColsC || nColsA != B.numRows()) {
        throw std::invalid_argument("impossible dimensions");
    }
    float _alpha = alpha;
    float _beta = beta;

    gpuErrChk(cublasSgemm(Context::getInstance().cuBlasHandle(),
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          m_numRows,
                          nColsC,
                          nColsA,
                          &_alpha,
                          A.m_vec->raw(),
                          m_numRows,
                          B.m_vec->raw(),
                          nColsA,
                          &_beta,
                          m_vec->raw(),
                          m_numRows));
}

template<>
inline void DeviceMatrix<double>::addAB(const DeviceMatrix &A, const DeviceMatrix &B, double alpha, double beta) {
    size_t nColsC = this->numCols();
    size_t nColsA = A.numCols();
    if (A.numRows() != m_numRows || B.numCols() != nColsC || nColsA != B.numRows()) {
        throw std::invalid_argument("impossible dimensions");
    }
    double _alpha = alpha;
    double _beta = beta;

    gpuErrChk(cublasDgemm(Context::getInstance().cuBlasHandle(),
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          m_numRows,
                          nColsC,
                          nColsA,
                          &_alpha,
                          A.m_vec->raw(),
                          m_numRows,
                          B.m_vec->raw(),
                          nColsA,
                          &_beta,
                          m_vec->raw(),
                          m_numRows));
}


template<typename TElement>
class DeviceTensor {

private:
    size_t m_numRows = 0;  ///< number of rows of each matrix
    size_t m_numCols = 0;  ///< number of columns of each matrix
    /**
     * Host-side vector of device-side matrices
     */
    std::vector<std::reference_wrapper<DeviceMatrix<TElement>>> m_cacheDevMatrix;
    /**
     * Device-side vector of device-side pointers; this is a bunch of pointers
     * {p1, p2, ..., pk} (on the device) that point to other device-side memory
     * locations. It's needed for some cuda functions.
     */
    std::unique_ptr<DeviceVector<TElement *>> m_d_bunchPointers;

public:

    DeviceTensor(size_t numRows, size_t numCols = 1, size_t capacity = 0) {
        m_numRows = numRows;
        m_numCols = numCols;
        m_cacheDevMatrix.reserve(capacity);
        m_d_bunchPointers = std::make_unique<DeviceVector<TElement *>>(capacity);
    }

    size_t numMatrices() const {
        return m_cacheDevMatrix.size();
    }

    size_t numRows() const {
        return m_numRows;
    }

    size_t numCols() const {
        return m_numCols;
    }

    void pushBack(DeviceMatrix<TElement> &o) {
        if (o.numRows() != m_numRows || o.numCols() != m_numCols) {
            throw std::invalid_argument("Given matrix has incompatible dimensions");
        }
        m_cacheDevMatrix.push_back(o);
    }

    DeviceVector<TElement *> devicePointersToMatrices() {
        size_t n = m_cacheDevMatrix.size();
        std::vector<TElement *> rawVecPointers(n);
        for (size_t i = 0; i < n; i++) {
            rawVecPointers[i] = m_cacheDevMatrix[i].get().raw();
        }
        m_d_bunchPointers->upload(rawVecPointers);
        return *m_d_bunchPointers;
    }

    void leastSquares(DeviceTensor &b);


//    void project(DeviceTensor &rhs) {
//        leastSquares(rhs);
//        // TODO tensor batch multiplication
//    }

    void addAB(DeviceTensor<TElement> &A, DeviceTensor<TElement> &B, TElement alpha = 1, TElement beta = 0);

    friend std::ostream &operator<<(std::ostream &out, const DeviceTensor<TElement> &data) {
        out << "DeviceTensor [" << data.m_numRows << " x " << data.m_numCols << " x " << data.m_cacheDevMatrix.size()
            << "]:" << std::endl;
        size_t i = 0;
        for (auto mat: data.m_cacheDevMatrix) {
            out << "Matrix " << i << ":\n" << mat;
            i++;
        }
        return out;
    }

};

template<>
inline void DeviceTensor<float>::addAB(DeviceTensor<float> &A, DeviceTensor<float> &B, float alpha, float beta) {
    size_t nMat = A.numMatrices();
    size_t nRA = A.numRows();
    size_t nCA = A.numCols();
    size_t nCB = B.numCols();
    float _alpha = alpha, _beta = beta;

    gpuErrChk(cublasSgemmBatched(Context::getInstance().cuBlasHandle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 nRA, nCB, nCA, &_alpha,
                                 A.devicePointersToMatrices().raw(), nRA,
                                 B.devicePointersToMatrices().raw(), nCA,
                                 &_beta,
                                 devicePointersToMatrices().raw(), nRA,
                                 nMat));
}

template<>
inline void DeviceTensor<double>::addAB(DeviceTensor<double> &A, DeviceTensor<double> &B, double alpha, double beta) {
    size_t nMat = A.numMatrices();
    size_t nRA = A.numRows();
    size_t nCA = A.numCols();
    size_t nCB = B.numCols();
    double _alpha = alpha, _beta = beta;

    gpuErrChk(cublasDgemmBatched(Context::getInstance().cuBlasHandle(),
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 nRA, nCB, nCA, &_alpha,
                                 A.devicePointersToMatrices().raw(), nRA,
                                 B.devicePointersToMatrices().raw(), nCA,
                                 &_beta,
                                 devicePointersToMatrices().raw(), nRA,
                                 nMat));
}

template<>
inline void DeviceTensor<float>::leastSquares(DeviceTensor &B) {
    size_t batchSize = numMatrices();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows || nColsB != 1 || B.numMatrices() != batchSize) {
        throw std::invalid_argument("Least squares rhs size does not equal lhs size");
    }
    int info = 0;
    DeviceVector<int> infoArray(batchSize);
    DeviceVector<float *> As = devicePointersToMatrices();
    DeviceVector<float *> Bs = B.devicePointersToMatrices();

    gpuErrChk(cublasSgelsBatched(Context::getInstance().cuBlasHandle(),
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
inline void DeviceTensor<double>::leastSquares(DeviceTensor &B) {
    size_t batchSize = numMatrices();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows || nColsB != 1 || B.numMatrices() != batchSize) {
        throw std::invalid_argument("Least squares rhs size does not equal lhs size");
    }
    int info = 0;
    DeviceVector<int> infoArray(batchSize);
    DeviceVector<double *> As = devicePointersToMatrices();
    DeviceVector<double *> Bs = B.devicePointersToMatrices();

    gpuErrChk(cublasDgelsBatched(Context::getInstance().cuBlasHandle(),
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


/* ------------------------------------------------------------------------------------
 *  SVD Factoriser
 * ------------------------------------------------------------------------------------ */

/**
 * Kernel that counts the number of elements of a vector that are higher than epsilon
 * @tparam TElement either float or double
 * @param d_array device array
 * @param n length of device array
 * @param d_count on exit, count of elements (int on device)
 * @param epsilon threshold
 */
template<typename TElement>
requires std::floating_point<TElement>
__global__ void k_countNonzeroSingularValues(TElement *d_array, size_t n, unsigned int *d_count, TElement epsilon) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_array[idx] > epsilon) {
        atomicAdd(d_count, 1);
    }
}

template<typename TElement> requires std::floating_point<TElement>
class SvdFactoriser {

private:

    int m_lwork = -1; /**< size of workspace needed for SVD */
    DeviceMatrix<TElement> *m_mat = nullptr;  /**< pointer to original matrix to be factorised */
    std::unique_ptr<DeviceMatrix<TElement>> m_Vtr;  /**< matrix V' or right singular vectors */
    std::unique_ptr<DeviceVector<TElement>> m_S;
    std::unique_ptr<DeviceMatrix<TElement>> m_U;  /**< matrix U or left singular vectors*/
    std::unique_ptr<DeviceVector<TElement>> m_workspace;  /**< workspace vector */
    std::unique_ptr<DeviceVector<int>> m_info;  /**< status code of computation */
    std::unique_ptr<DeviceVector<unsigned int>> m_rank;
    bool m_computeU = false;  /**< whether to compute U */
    bool m_destroyMatrix = true; /**< whether to sacrifice original matrix */

    /**
     * Checks whether matrix is tall; throws invalid_argument if not
     * @param mat given matrix
     */
    void checkMatrix(DeviceMatrix<TElement> &mat) {
        if (mat.numRows() < mat.numCols()) {
            throw std::invalid_argument("your matrix is fat (no offence)");
        }
    };

    void computeWorkspaceSize(size_t m, size_t n);

public:

    /**
     * Constructor
     * @param mat matrix to be factorised
     * @param computeU whether to compute U (default is false)
     */
    SvdFactoriser(DeviceMatrix<TElement> &mat,
                  bool computeU = false,
                  bool destroyMatrix = true) {
        checkMatrix(mat);
        m_destroyMatrix = destroyMatrix;
        m_mat = (destroyMatrix) ? &mat : new DeviceMatrix<TElement>(mat);
        m_computeU = computeU;
        size_t m = mat.numRows();
        size_t n = mat.numCols();
        size_t k = std::min(m, n);
        computeWorkspaceSize(m, n);
        m_workspace = std::make_unique<DeviceVector<TElement>>(m_lwork);
        m_Vtr = std::make_unique<DeviceMatrix<TElement>>(n, n);
        m_S = std::make_unique<DeviceVector<TElement>>(k);
        m_info = std::make_unique<DeviceVector<int>>(1);
        m_rank = std::make_unique<DeviceVector<unsigned int>>(1);
        if (computeU) m_U = std::make_unique<DeviceMatrix<TElement>>(m, m);

    }

    /**
     * Update matrix reusing allocated memory; the currect object can be
     * reused for matrices of the same dimensions
     * @param mat
     */
    void updateMatrix(DeviceMatrix<TElement> &mat) {
        checkMatrix(mat);
        size_t m = mat.numRows();
        size_t n = mat.numCols();
        if (m != m_mat->numRows() || n != m_mat->numCols()) {
            throw std::invalid_argument("wrong matrix dimensions");
        }
        m_mat = &mat;
    }

    /**
     * Perform factorisation
     * @return status code
     *
     * Warning: the given matrix is destroyed
     */
    int factorise();

    DeviceVector<TElement> singularValues() const {
        return *m_S;
    }

    DeviceMatrix<TElement> rightSingularVectors() const {
        return *m_Vtr;
    }

    std::optional<DeviceMatrix<TElement>> leftSingularVectors() const {
        if (!m_computeU) return std::nullopt;
        return *m_U;
    }

    ~SvdFactoriser() {
        m_lwork = -1;
        if (!m_destroyMatrix && m_mat) delete m_mat;
    }

    unsigned int rank(TElement epsilon = 1e-6) {
        int k = m_S->capacity();
        k_countNonzeroSingularValues<TElement><<<DIM2BLOCKS(k), THREADS_PER_BLOCK>>>(m_S->raw(), k,
                                                                                     m_rank->raw(),
                                                                                     epsilon);
        return (*m_rank)(0);
    }

};

template<>
inline int SvdFactoriser<float>::factorise() {
    size_t m = m_mat->numRows();
    size_t n = m_mat->numCols();

    gpuErrChk(
            cusolverDnSgesvd(Context::getInstance().cuSolverHandle(),
                             (m_computeU) ? 'A' : 'N', 'A',
                             m, n,
                             m_mat->raw(), m,
                             m_S->raw(),
                             (m_computeU) ? m_U->raw() : nullptr, m,
                             m_Vtr->raw(), n,
                             m_workspace->raw(),
                             m_lwork,
                             nullptr,  // rwork (used only if SVD fails)
                             m_info->raw()));
    int info = (*m_info)(0);
    return info;
}


template<>
inline int SvdFactoriser<double>::factorise() {
    size_t m = m_mat->numRows();
    size_t n = m_mat->numCols();

    gpuErrChk(
            cusolverDnDgesvd(Context::getInstance().cuSolverHandle(),
                             (m_computeU) ? 'A' : 'N', 'A',
                             m, n,
                             m_mat->raw(), m,
                             m_S->raw(),
                             (m_computeU) ? m_U->raw() : nullptr, m,
                             m_Vtr->raw(), n,
                             m_workspace->raw(),
                             m_lwork,
                             nullptr,  // rwork (used only if SVD fails)
                             m_info->raw()));
    int info = (*m_info)(0);
    return info;
}

template<>
inline void SvdFactoriser<float>::computeWorkspaceSize(size_t m, size_t n) {

    gpuErrChk(cusolverDnSgesvd_bufferSize(Context::getInstance().cuSolverHandle(), m, n, &m_lwork));
}

template<>
inline void SvdFactoriser<double>::computeWorkspaceSize(size_t m, size_t n) {

    gpuErrChk(cusolverDnDgesvd_bufferSize(Context::getInstance().cuSolverHandle(), m, n, &m_lwork));
}


/* ------------------------------------------------------------------------------------
 *  Cholesky Factoriser
 * ------------------------------------------------------------------------------------ */

template<typename TElement> requires std::floating_point<TElement>
class CholeskyFactoriser {

private:
    int m_workspaceSize = 0;
    std::unique_ptr<DeviceVector<int>> m_d_info;
    std::unique_ptr<DeviceVector<TElement>> m_d_workspace;
    DeviceMatrix<TElement> *m_d_matrix; // do not destroy

    void computeWorkspaceSize();

public:

    CholeskyFactoriser(DeviceMatrix<TElement> &A) {
        if (A.numRows() != A.numCols()) throw std::invalid_argument("Matrix A must be square");
        m_d_matrix = &A;
        computeWorkspaceSize();
        m_d_workspace = std::make_unique<DeviceVector<TElement>>(m_workspaceSize);
        m_d_info = std::make_unique<DeviceVector<int>>(1);
    }

    int factorise();

    // TODO do we need to allow rhs to be a matrix?
    int solve(DeviceVector<TElement> &rhs);

};

template<>
void CholeskyFactoriser<double>::computeWorkspaceSize() {
    size_t n = m_d_matrix->numRows();

    gpuErrChk(cusolverDnDpotrf_bufferSize(Context::getInstance().cuSolverHandle(),
                                          CUBLAS_FILL_MODE_LOWER, n,
                                          nullptr, n, &m_workspaceSize));
}

template<>
void CholeskyFactoriser<float>::computeWorkspaceSize() {
    size_t n = m_d_matrix->numRows();

    gpuErrChk(cusolverDnSpotrf_bufferSize(Context::getInstance().cuSolverHandle(),
                                          CUBLAS_FILL_MODE_LOWER, n,
                                          nullptr, n, &m_workspaceSize));
}

template<>
inline int CholeskyFactoriser<double>::factorise() {
    size_t n = m_d_matrix->numRows();

    gpuErrChk(cusolverDnDpotrf(Context::getInstance().cuSolverHandle(), CUBLAS_FILL_MODE_LOWER, n,
                               m_d_matrix->raw(), n,
                               m_d_workspace->raw(),
                               m_workspaceSize,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}


template<>
inline int CholeskyFactoriser<float>::factorise() {
    size_t n = m_d_matrix->numRows();

    gpuErrChk(cusolverDnSpotrf(Context::getInstance().cuSolverHandle(), CUBLAS_FILL_MODE_LOWER, n,
                               m_d_matrix->raw(), n,
                               m_d_workspace->raw(),
                               m_workspaceSize,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}

template<>
inline int CholeskyFactoriser<double>::solve(DeviceVector<double> &rhs) {
    size_t n = m_d_matrix->numRows();
    size_t k = rhs.capacity();

    gpuErrChk(cusolverDnDpotrs(Context::getInstance().cuSolverHandle(),
                               CUBLAS_FILL_MODE_LOWER,
                               n, 1,
                               m_d_matrix->raw(), n,
                               rhs.raw(), n,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}

template<>
inline int CholeskyFactoriser<float>::solve(DeviceVector<float> &rhs) {
    size_t n = m_d_matrix->numRows();
    size_t k = rhs.capacity();

    gpuErrChk(cusolverDnSpotrs(Context::getInstance().cuSolverHandle(),
                               CUBLAS_FILL_MODE_LOWER,
                               n, 1,
                               m_d_matrix->raw(), n,
                               rhs.raw(), n,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}


/* ------------------------------------------------------------------------------------
 *  Nullspace Project
 * ------------------------------------------------------------------------------------ */

template<typename TElement> requires std::floating_point<TElement>
class Nullspace {

private:
    DeviceMatrix<TElement> *m_matrix = nullptr;
    size_t m_rankA;
    DeviceMatrix<TElement> *m_N = nullptr;

public:

    Nullspace(DeviceMatrix<TElement> &A, bool destroyA = true) {
        m_matrix = &A;
        SvdFactoriser<TElement> svd(A, false, destroyA);
        svd.factorise();
        m_rankA = svd.rank();
        DeviceMatrix<TElement> Vtr = svd.rightSingularVectors();
        size_t k = Vtr.numRows();
        DeviceMatrix<TElement> Ntr = Vtr.getRows(m_rankA, k - 1);
        DeviceMatrix<TElement> N = Ntr.tr();
        m_N = new DeviceMatrix<TElement>(N);
    }

    ~Nullspace() {
        if (m_N) delete m_N;
    }

    DeviceMatrix<TElement> get() {
        return *m_N;
    }


};

#endif