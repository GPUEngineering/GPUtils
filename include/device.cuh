#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <memory>
#include <optional>

#ifndef DEVICE_VECTOR_CUH__
#define DEVICE_VECTOR_CUH__

#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

/* ------------------------------------------------------------------------------------
 *  Context
 * ------------------------------------------------------------------------------------ */

class Context {

private:
    cublasHandle_t m_cublasHandle;
    cusolverDnHandle_t m_cusolverHandle;

public:
    explicit Context() noexcept {
        cublasStatus_t stat = cublasCreate(&m_cublasHandle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS initialization failed\n");
        }
        cusolverStatus_t stat2 = cusolverDnCreate(&m_cusolverHandle);
        if (stat2 != CUSOLVER_STATUS_SUCCESS) {
            printf("cuSOLVER initialization failed\n");
        }
    }

    virtual ~Context() noexcept {
        cublasDestroy(m_cublasHandle);
        cusolverDnDestroy(m_cusolverHandle);
    }

    cublasHandle_t &cuBlasHandle() { return m_cublasHandle; }

    cusolverDnHandle_t &cuSolverHandle() { return m_cusolverHandle; }

};

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
    Context *m_context = nullptr;
    bool m_doDestroy = false;

    bool destroy() {
        if (!m_doDestroy) return false;
        if (m_d_data) cudaFree(m_d_data);
        m_numAllocatedElements = 0;
        m_d_data = nullptr;
        m_context = nullptr;
        return true;
    }

public:

    /**
     * Constructs a DeviceVector object
     */
    DeviceVector() = default;

    /**
     * Constructs a DeviceVector object and allocates
     * memory on the device for n elements
     */
    DeviceVector(Context &context, size_t n) {
        m_context = &context;
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
        m_context = other.m_context;
        m_doDestroy = false;
        m_numAllocatedElements = to - from + 1;
        m_d_data = other.m_d_data + from;
    }

    /**
     * Copy constructor
     * @param other
     */
    DeviceVector(DeviceVector &other) {
        m_context = other.m_context;
        allocateOnDevice(other.m_numAllocatedElements);
        cudaMemcpy(m_d_data,
                   other.get(),
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
    DeviceVector(Context &context, const std::vector<TElement> &vec) {
        m_context = &context;
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
     * @return true if and only if no errors occured during
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
    TElement *get() {
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
     * Fetches just one value from the device
     *
     * Use sparingly
     *
     * @param i index
     * @return entry of array at index i
     */
    TElement fetchElementFromDevice(size_t i);

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
        DeviceVector resultVec(firstVector.m_context, firstVector.capacity());
        firstVector.deviceCopyTo(resultVec);
        resultVec += secondVector;
        return resultVec;
    }

    friend DeviceVector operator-(DeviceVector &firstVector, const DeviceVector &secondVector) {
        DeviceVector resultVec(firstVector.m_context, firstVector.capacity());
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
        DeviceVector resultVec(secondVector.m_context, secondVector.capacity());
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
    TElement sum() const;


}; /* end of class */

template<>
inline DeviceVector<float> &DeviceVector<float>::operator+=(const DeviceVector<float> &rhs) {
    const float alpha = 1.;
    cublasSaxpy(m_context->cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
inline DeviceVector<double> &DeviceVector<double>::operator+=(const DeviceVector<double> &rhs) {
    const double alpha = 1.;
    cublasDaxpy(m_context->cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
inline DeviceVector<float> &DeviceVector<float>::operator*=(float scalar) {
    float alpha = scalar;
    cublasSscal(m_context->cuBlasHandle(), m_numAllocatedElements, &alpha, m_d_data, 1);
    return *this;
}

template<>
inline DeviceVector<double> &DeviceVector<double>::operator*=(double scalar) {
    double alpha = scalar;
    cublasDscal(m_context->cuBlasHandle(), m_numAllocatedElements, &alpha, m_d_data, 1);
    return *this;
}

template<>
inline DeviceVector<float> &DeviceVector<float>::operator-=(const DeviceVector<float> &rhs) {
    const float alpha = -1.;
    cublasSaxpy(m_context->cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
inline DeviceVector<double> &DeviceVector<double>::operator-=(const DeviceVector<double> &rhs) {
    const double alpha = -1.;
    cublasDaxpy(m_context->cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
inline float DeviceVector<float>::operator*(const DeviceVector<float> &rhs) const {
    float inn_prod;
    cublasSdot(m_context->cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, rhs.m_d_data, 1, &inn_prod);
    return inn_prod;
}

template<>
inline double DeviceVector<double>::operator*(const DeviceVector<double> &rhs) const {
    double inn_prod;
    cublasDdot(m_context->cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, rhs.m_d_data, 1, &inn_prod);
    return inn_prod;
}

template<>
inline float DeviceVector<float>::norm2() const {
    float the_norm;
    cublasSnrm2(m_context->cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_norm);
    return the_norm;
}

template<>
inline double DeviceVector<double>::norm2() const {
    double the_norm;
    cublasDnrm2(m_context->cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_norm);
    return the_norm;
}

template<>
inline float DeviceVector<float>::sum() const {
    float the_sum;
    cublasSasum(m_context->cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_sum);
    return the_sum;
}

template<>
inline double DeviceVector<double>::sum() const {
    double the_sum;
    cublasDasum(m_context->cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_sum);
    return the_sum;
}

template<typename TElement>
TElement DeviceVector<TElement>::fetchElementFromDevice(size_t i) {
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
    if (!allocateOnDevice(size)) return false;
    if (size <= m_numAllocatedElements) {
        size_t buffer_size = size * sizeof(TElement);
        cudaMemcpy(m_d_data, dataArray, buffer_size, cudaMemcpyHostToDevice);
    }
    return true;
}

template<typename TElement>
void DeviceVector<TElement>::deviceCopyTo(DeviceVector<TElement> &elsewhere) {
    elsewhere.allocateOnDevice(m_numAllocatedElements);
    cudaMemcpy(elsewhere.get(),
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToDevice);
}

template<typename TElement>
void DeviceVector<TElement>::download(TElement *hostData) const {
    cudaMemcpy(hostData,
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToHost);
}

template<typename TElement>
void DeviceVector<TElement>::download(std::vector<TElement> &vec) const {
    vec.reserve(m_numAllocatedElements);
    cudaMemcpy(vec.data(),
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToHost);
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
    Context *m_context = nullptr;
    DeviceVector<TElement> *m_vec = nullptr;  ///< stores all useful memory
    size_t m_numRows = 0;  ///< number of rows

    /**
     *
     */
    void destroy() {
        m_numRows = 0;
        if (m_vec) delete m_vec;
    }

    /**
     *
     * @param vec_rm
     * @param vec_cm
     * @param n_rows
     * @param n_cols
     */
    void rm2cm(std::vector<TElement> vec_rm,
               std::vector<TElement> &vec_cm,
               size_t n_rows,
               size_t n_cols) {
        for (size_t i = 0; i < n_rows; i++) {
            for (size_t j = 0; j < n_cols; j++) {
                TElement c = vec_rm[j + i * n_cols];
                vec_cm[i + j * n_rows] = c;
            }
        }
    }


public:

    /**
     *
     * @param context
     * @param n_rows
     * @param n_cols
     */
    DeviceMatrix(Context &context, size_t n_rows, size_t n_cols) {
        m_context = &context;
        m_numRows = n_rows;
        m_vec = new DeviceVector<TElement>(context, n_rows * n_cols);
    }

    /**
     *
     * @param context
     * @param n_rows
     * @param vec
     * @param mode
     */
    DeviceMatrix(Context &context,
                 size_t n_rows,
                 const std::vector<TElement> &vec,
                 MatrixStorageMode mode = MatrixStorageMode::columnMajor) {
        m_context = &context;
        size_t numel = vec.size();
        m_numRows = n_rows;

        if (numel % n_rows != 0) throw std::invalid_argument("impossible dimensions");
        size_t n_cols = numel / n_rows;
        if (mode == MatrixStorageMode::rowMajor) {
            std::vector<TElement> vec_cm(numel);
            rm2cm(vec, vec_cm, n_rows, n_cols);  // to column-major
            m_vec = new DeviceVector<TElement>(context, vec_cm);
        } else {
            m_vec = new DeviceVector<TElement>(context, vec);
        }
    }

    DeviceMatrix(const DeviceMatrix &other) {
        m_context = other.m_context;
        m_numRows = other.m_numRows;
        m_vec = new DeviceVector<TElement>(*other.m_vec);
    }

    // SLICE!
    DeviceMatrix(DeviceMatrix &other, size_t colFrom, size_t colTo) {
        m_context = other.m_context;
        m_numRows = other.m_numRows;
        size_t start = colFrom * m_numRows;
        size_t finish = (colTo + 1) * m_numRows - 1;
        m_vec = new DeviceVector<TElement>(*other.m_vec, start, finish);
    }

    /**
     *
     * @param vec
     * @param n_rows
     * @param mode
     */
    void upload(const std::vector<TElement> &vec,
                size_t n_rows,
                MatrixStorageMode mode = MatrixStorageMode::columnMajor) {
        size_t n = vec.size();
        if (n % n_rows != 0) throw std::invalid_argument("impossible dimensions");
        size_t n_cols = n / n_rows;
        if (mode == MatrixStorageMode::rowMajor) {
            std::vector<TElement> vec_cm(n);
            rm2cm(vec, vec_cm, n_rows, n_cols);  // to column-major
            m_vec->upload(vec_cm);
        } else {
            m_vec->upload(vec);
        }
    }

    TElement *get() {
        return m_vec->get();
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
     * elements of the matrix.
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

    DeviceMatrix<TElement> tr() {
        size_t m = numRows();
        size_t n = numCols();
        DeviceMatrix<TElement> transpose(*m_context, n, m);
        float alpha = 1.0f, beta = 0;
        cublasSgeam(m_context->cuBlasHandle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m,
                    &alpha, m_vec->get(), m,
                    &beta, nullptr, n,
                    transpose.get(), n);
        return transpose;
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
        DeviceVector<float> resultVector(*A.m_context, nRowsA);
        cublasSgemv(A.m_context->cuBlasHandle(),
                    CUBLAS_OP_N,
                    nRowsA,
                    nColsA,
                    &alpha,
                    A.m_vec->get(),
                    nRowsA,
                    b.get(),
                    1,
                    &beta,
                    resultVector.get(),
                    1);
        return resultVector;
    }
    friend DeviceVector<double> operator*(DeviceMatrix &A, DeviceVector<double> &b) {
        size_t nRowsA = A.numRows();
        size_t nColsA = b.capacity();
        double alpha = 1.;
        double beta = 0.;
        DeviceVector<double> resultVector(*A.m_context, nRowsA);
        cublasDgemv(A.m_context->cuBlasHandle(),
                    CUBLAS_OP_N,
                    nRowsA,
                    nColsA,
                    &alpha,
                    A.m_vec->get(),
                    nRowsA,
                    b.get(),
                    1,
                    &beta,
                    resultVector.get(),
                    1);
        return resultVector;
    }

    friend DeviceMatrix operator*(DeviceMatrix &A, const DeviceMatrix &b) {
        size_t nRowsA = A.numRows();
        size_t nColsA = A.numCols();
        size_t nColsB = b.numCols();
        float alpha = 1.;
        float beta = 1.;
        DeviceMatrix resultMatrix(A.m_context, nRowsA, nColsB);
        cublasSgemm(A.m_context->cuBlasHandle(),
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    nRowsA,
                    nColsB,
                    nColsA,
                    &alpha,
                    A.m_vec->get(),
                    nRowsA,
                    b.m_vec->get(),
                    nColsA,
                    &beta,
                    resultMatrix.m_vec->get(),
                    nRowsA);
        return resultMatrix;
    }

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

template<typename TElement>
requires std::floating_point<TElement>
class SvdFactoriser {

private:

    Context *m_context = nullptr;
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
     * @param context context
     * @param mat matrix to be factorised
     * @param computeU whether to compute U (default is false)
     */
    SvdFactoriser(Context &context,
                  DeviceMatrix<TElement> &mat,
                  bool computeU = false,
                  bool destroyMatrix = true) {
        checkMatrix(mat);
        m_context = &context;
        m_destroyMatrix = destroyMatrix;
        m_mat = (destroyMatrix) ? &mat : new DeviceMatrix<TElement>(mat);
        m_computeU = computeU;
        size_t m = mat.numRows();
        size_t n = mat.numCols();
        size_t k = std::min(m, n);
        computeWorkspaceSize(m, n);
        m_workspace = std::make_unique<DeviceVector<TElement>>(context, m_lwork);
        m_Vtr = std::make_unique<DeviceMatrix<TElement>>(context, n, n);
        m_S = std::make_unique<DeviceVector<TElement>>(context, k);
        m_info = std::make_unique<DeviceVector<int>>(context, 1);
        m_rank = std::make_unique<DeviceVector<unsigned int>>(context, 1);
        if (computeU) m_U = std::make_unique<DeviceMatrix<TElement>>(context, m, m);

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
        k_countNonzeroSingularValues<TElement><<<DIM2BLOCKS(k), THREADS_PER_BLOCK>>>(m_S->get(), k,
                                                                                     m_rank->get(),
                                                                                     epsilon);
        return m_rank->fetchElementFromDevice(0);
    }

};

template<>
inline int SvdFactoriser<float>::factorise() {
    size_t m = m_mat->numRows();
    size_t n = m_mat->numCols();
    cusolverDnSgesvd(m_context->cuSolverHandle(),
                     (m_computeU) ? 'A' : 'N', 'A',
                     m, n,
                     m_mat->get(), m,
                     m_S->get(),
                     (m_computeU) ? m_U->get() : nullptr, m,
                     m_Vtr->get(), n,
                     m_workspace->get(),
                     m_lwork,
                     nullptr,  // rwork (used only if SVD fails)
                     m_info->get());
    int info = m_info->fetchElementFromDevice(0);
    return info;
}


template<>
inline int SvdFactoriser<double>::factorise() {
    size_t m = m_mat->numRows();
    size_t n = m_mat->numCols();
    cusolverDnDgesvd(m_context->cuSolverHandle(),
                     (m_computeU) ? 'A' : 'N', 'A',
                     m, n,
                     m_mat->get(), m,
                     m_S->get(),
                     (m_computeU) ? m_U->get() : nullptr, m,
                     m_Vtr->get(), n,
                     m_workspace->get(),
                     m_lwork,
                     nullptr,  // rwork (used only if SVD fails)
                     m_info->get());
    int info = m_info->fetchElementFromDevice(0);
    return info;
}

template<>
inline void SvdFactoriser<float>::computeWorkspaceSize(size_t m, size_t n) {
    cusolverDnSgesvd_bufferSize(m_context->cuSolverHandle(), m, n, &m_lwork);
}

template<>
inline void SvdFactoriser<double>::computeWorkspaceSize(size_t m, size_t n) {
    cusolverDnDgesvd_bufferSize(m_context->cuSolverHandle(), m, n, &m_lwork);
}

#endif
