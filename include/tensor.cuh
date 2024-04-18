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

#ifndef TENSOR_CUH
#define TENSOR_CUH

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


#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))
#endif



/* ================================================================================================
 *  SESSION
 * ================================================================================================ */

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

template<typename T>
class DTensor {

private:
    /** Pointer to device data */
    T *m_d_data = nullptr;
    /** Number of allocated elements */
    size_t m_numRows = 0;
    size_t m_numCols = 0;
    size_t m_numMats = 0;
    bool m_doDestroy = false;

    bool destroy() {
        if (!m_doDestroy) return false;
        if (m_d_data) cudaFree(m_d_data);
        m_d_data = nullptr;
        return true;
    }

    bool allocateOnDevice(size_t size, bool zero = false);

public:
    /**
    * Constructs a DeviceVector object
    */
    DTensor() = default;

    ~DTensor() {
        destroy();
    }

    /**
     * Allocates (m, n, k)-tensor
     * @param n
     */
    DTensor(size_t m, size_t n = 1, size_t k = 1, bool zero = false) {
        m_numRows = m;
        m_numCols = n;
        m_numMats = k;
        size_t size = m * n * k;
        allocateOnDevice(size, zero);
    }

    DTensor(const std::vector<T> &data, size_t m, size_t n = 1, size_t k = 1) {
        m_numRows = m;
        m_numCols = n;
        m_numMats = k;
        size_t size = m * n * k;
        allocateOnDevice(size);
        upload(data);
    }

    /**
     * Copy constructor
     */
    DTensor(const DTensor &other) {
        m_numMats = other.m_numMats;
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;

        allocateOnDevice(m_numRows * m_numCols * m_numMats);
        gpuErrChk(cudaMemcpy(m_d_data, other.raw(), m_numRows * m_numCols * m_numMats * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

    DTensor(DTensor&& other) {
        m_numCols = other.m_numCols;
        m_numRows = other.m_numRows;
        m_numMats = other.m_numMats;
        m_d_data = other.m_d_data;
        m_doDestroy = true;
        other.m_doDestroy = false;
        other.m_d_data = nullptr;
    }

    /**
     * Slicing constructor
     * @param other
     * @param axis
     * @param from
     * @param to
     */
    DTensor(const DTensor &other, size_t axis, size_t from, size_t to) {
        if (from > to) throw std::invalid_argument("from > to");
        size_t offset = 0, len = to - from + 1;
        if (axis == 2) {
            offset = other.m_numRows * other.m_numCols * from;
            m_numRows = other.m_numRows;
            m_numCols = other.m_numCols;
            m_numMats = len;
        } else if (axis == 1) {
            offset = other.m_numCols * from;
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
        m_doDestroy = false;
    }

    T *raw() const;

    size_t numRows() const;

    size_t numCols() const;

    size_t numMats() const;

    size_t numel() const;

    bool upload(const std::vector<T> &vec);

    void download(std::vector<T> &vec) const;

    void deviceCopyTo(DTensor<T> &other) const;

    T normF() const;

    T sumAbs() const;

    void leastSquares(DTensor &b);

    /* ------------- OPERATORS ------------- */

    DTensor &operator=(const DTensor &other);

    T operator()(size_t i, size_t j=0, size_t k=0);

    DTensor &operator*=(T scalar);

    DTensor &operator+=(const DTensor &rhs);

    DTensor &operator-=(const DTensor &rhs);

    DTensor<T *> pointersToMatrices();

    DTensor<T> getRows(size_t rowsFrom, size_t rowsTo, size_t matIdx);

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

    void addAB(DTensor<T> &A, DTensor<T> &B, T alpha = 1, T beta = 0);

    friend DTensor<T> operator*(DTensor &A, DTensor &B) {
        size_t nrA = A.m_numRows, ncB = B.m_numCols, nmB = B.m_numMats;
        DTensor<T> result(nrA, ncB, nmB);
        result.addAB(A, B);
        return result;
    }

    friend std::ostream &operator<<(std::ostream &out, const DTensor<T> &data) {
        size_t nr = data.m_numRows, nc = data.m_numCols, nm = data.m_numMats;
        out << "Tensor [" << data.m_numRows << " x "
            << data.m_numCols << " x "
            << data.m_numMats << "]:" << std::endl;
        std::vector<T> temp;
        data.download(temp);
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

}; /* END OF TENZOR */


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
inline size_t DTensor<T>::numel() const {
    return m_numRows * m_numCols * m_numMats;
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

template<typename T>
inline bool DTensor<T>::allocateOnDevice(size_t size, bool zero) {
    if (size <= 0) return false;
    destroy();
    m_doDestroy = true;
    size_t buffer_size = size * sizeof(T);
    bool cudaStatus = cudaMalloc(&m_d_data, buffer_size);
    if (cudaStatus != cudaSuccess) return false;
    if (zero) gpuErrChk(cudaMemset(m_d_data, 0, buffer_size)); // set to zero all elements
    return true;
}

template<typename T>
inline bool DTensor<T>::upload(const std::vector<T> &vec) {
    size_t size = vec.size();
    // make sure vec is of right size
    if (size != m_numRows * m_numCols * m_numMats) throw std::invalid_argument("vec has wrong size");
    if (size <= m_numRows * m_numCols * m_numMats) {
        size_t buffer_size = size * sizeof(T);
        gpuErrChk(cudaMemcpy(m_d_data, vec.data(), buffer_size, cudaMemcpyHostToDevice));
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

template<typename T>
inline void DTensor<T>::deviceCopyTo(DTensor<T> &elsewhere) const {
    if (elsewhere.numel() < numel()) {
        throw std::invalid_argument("tensor does not fit into destination");
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
    m_doDestroy = false;
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
inline T DTensor<T>::operator()(size_t i, size_t j, size_t k) {
    T hostDst;
    size_t offset = i + m_numRows * (j + m_numCols * k);
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + offset, sizeof(T), cudaMemcpyDeviceToHost));
    return hostDst;
}

template<typename T>
inline DTensor<T *> DTensor<T>::pointersToMatrices() {
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
inline void DTensor<double>::addAB(DTensor<double> &A, DTensor<double> &B, double alpha, double beta) {
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
inline void DTensor<float>::addAB(DTensor<float> &A, DTensor<float> &B, float alpha, float beta) {
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
inline void DTensor<double>::leastSquares(DTensor &B) {
    size_t batchSize = numMats();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows || nColsB != 1 || B.numMats() != batchSize)
        throw std::invalid_argument("Least squares rhs size does not equal lhs size");
    if (m_numCols > m_numRows)
        throw std::invalid_argument("Least squares supports tall matrices only");
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
inline void DTensor<float>::leastSquares(DTensor &B) {
    size_t batchSize = numMats();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows || nColsB != 1 || B.numMats() != batchSize)
        throw std::invalid_argument("Least squares rhs size does not equal lhs size");
    if (m_numCols > m_numRows)
        throw std::invalid_argument("Least squares supports tall matrices only");
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
DTensor<double> DTensor<double>::getRows(size_t rowsFrom, size_t rowsTo, size_t matIdx) {
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

/* ================================================================================================
 *  SINGULAR VALUE DECOMPOSITION (SVD)
 * ================================================================================================ */

/**
 * Kernel that counts the number of elements of a vector that are higher than epsilon
 * @tparam TElement either float or double
 * @param d_array device array
 * @param n length of device array
 * @param d_count on exit, count of elements (int on device)
 * @param epsilon threshold
 */
template<typename T>
requires std::floating_point<T>
__global__ void k_countNonzeroSingularValues(T *d_array, size_t n, unsigned int *d_count, T epsilon) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && d_array[idx] > epsilon) {
        atomicAdd(d_count, 1);
    }
}


template<typename T> requires std::floating_point<T>
class Svd {

private:

    int m_lwork = -1; /**< size of workspace needed for SVD */
    DTensor<T> *m_tensor = nullptr;  /**< pointer to original matrix to be factorised */
    std::unique_ptr<DTensor<T>> m_Vtr;  /**< matrix V' or right singular vectors */
    std::unique_ptr<DTensor<T>> m_S;
    std::unique_ptr<DTensor<T>> m_U;  /**< matrix U or left singular vectors*/
    std::unique_ptr<DTensor<T>> m_workspace;  /**< workspace vector */
    std::unique_ptr<DTensor<int>> m_info;  /**< status code of computation */
    std::unique_ptr<DTensor<unsigned int>> m_rank;
    bool m_computeU = false;  /**< whether to compute U */
    bool m_destroyMatrix = true; /**< whether to sacrifice original matrix */

    /**
     * Checks whether matrix is tall; throws invalid_argument if not
     * @param mat given matrix
     */
    void checkMatrix(DTensor<T> &tenz) {
        if (tenz.numMats() > 1) {
            throw std::invalid_argument("Only (m, n, 1) tensors are supported for now");
        }
        if (tenz.numRows() < tenz.numCols()) {
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
    Svd(DTensor<T> &mat,
        bool computeU = false,
        bool destroyMatrix = true) {
        checkMatrix(mat);
        m_destroyMatrix = destroyMatrix;
        m_tensor = (destroyMatrix) ? &mat : new DTensor<T>(mat);
        m_computeU = computeU;
        size_t m = mat.numRows();
        size_t n = mat.numCols();
        size_t k = std::min(m, n);
        computeWorkspaceSize(m, n);
        m_workspace = std::make_unique<DTensor<T>>(m_lwork, 1, 1);
        m_Vtr = std::make_unique<DTensor<T>>(n, n, 1);
        m_S = std::make_unique<DTensor<T>>(k, 1, 1);
        m_info = std::make_unique<DTensor<int>>(1, 1, 1);
        m_rank = std::make_unique<DTensor<unsigned int>>(1, 1, 1);
        if (computeU) m_U = std::make_unique<DTensor<T>>(m, m, 1);
    }

    /**
     * Perform factorisation
     * @return status code
     *
     * Warning: the given matrix is destroyed
     */
    int factorise();

    DTensor<T> singularValues() const {
        return *m_S;
    }

    DTensor<T> rightSingularVectors() const {
        return *m_Vtr;
    }

    std::optional<DTensor<T>> leftSingularVectors() const {
        if (!m_computeU) return std::nullopt;
        return *m_U;
    }

    ~Svd() {
        m_lwork = -1;
        if (!m_destroyMatrix && m_tensor) delete m_tensor;
    }

    unsigned int rank(T epsilon = 1e-6) {
        int k = m_S->numel();
        k_countNonzeroSingularValues<T><<<DIM2BLOCKS(k), THREADS_PER_BLOCK>>>(m_S->raw(), k,
                m_rank->raw(),
                epsilon);
        return (*m_rank)(0);
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
inline int Svd<double>::factorise() {
    size_t m = m_tensor->numRows();
    size_t n = m_tensor->numCols();
    gpuErrChk(
            cusolverDnDgesvd(Session::getInstance().cuSolverHandle(),
                             (m_computeU) ? 'A' : 'N', 'A',
                             m, n,
                             m_tensor->raw(), m,
                             m_S->raw(),
                             (m_computeU) ? m_U->raw() : nullptr, m,
                             m_Vtr->raw(), n,
                             m_workspace->raw(),
                             m_lwork,
                             nullptr,  // rwork (used only if SVD fails)
                             m_info->raw()));
    int info = (*m_info)(0, 0, 0);
    return info;
}

template<>
inline int Svd<float>::factorise() {
    size_t m = m_tensor->numRows();
    size_t n = m_tensor->numCols();
    gpuErrChk(
            cusolverDnSgesvd(Session::getInstance().cuSolverHandle(),
                             (m_computeU) ? 'A' : 'N', 'A',
                             m, n,
                             m_tensor->raw(), m,
                             m_S->raw(),
                             (m_computeU) ? m_U->raw() : nullptr, m,
                             m_Vtr->raw(), n,
                             m_workspace->raw(),
                             m_lwork,
                             nullptr,  // rwork (used only if SVD fails)
                             m_info->raw()));
    int info = (*m_info)(0, 0, 0);
    return info;
}



/* ================================================================================================
 *  CHOLESKY FACTORISATION
 * ================================================================================================ */

template<typename T> requires std::floating_point<T>
class CholeskyFactoriser {

private:
    int m_workspaceSize = 0;
    std::unique_ptr<DTensor<int>> m_d_info;
    std::unique_ptr<DTensor<T>> m_d_workspace;
    DTensor<T> *m_d_matrix; // do not destroy

    void computeWorkspaceSize();

public:

    CholeskyFactoriser(DTensor<T> &A) {
        if (A.numMats() > 1) throw std::invalid_argument("3D Tensors are not supported (for now); only matrices");
        if (A.numRows() != A.numCols()) throw std::invalid_argument("Matrix A must be square");
        m_d_matrix = &A;
        computeWorkspaceSize();
        m_d_workspace = std::make_unique<DTensor<T>>(m_workspaceSize);
        m_d_info = std::make_unique<DTensor<int>>(1);
    }

    int factorise();

    // TODO do we need to allow rhs to be a matrix?
    int solve(DTensor<T> &rhs);

};

template<>
void CholeskyFactoriser<double>::computeWorkspaceSize() {
    size_t n = m_d_matrix->numRows();

    gpuErrChk(cusolverDnDpotrf_bufferSize(Session::getInstance().cuSolverHandle(),
                                          CUBLAS_FILL_MODE_LOWER, n,
                                          nullptr, n, &m_workspaceSize));
}

template<>
void CholeskyFactoriser<float>::computeWorkspaceSize() {
    size_t n = m_d_matrix->numRows();

    gpuErrChk(cusolverDnSpotrf_bufferSize(Session::getInstance().cuSolverHandle(),
                                          CUBLAS_FILL_MODE_LOWER, n,
                                          nullptr, n, &m_workspaceSize));
}

template<>
inline int CholeskyFactoriser<double>::factorise() {
    size_t n = m_d_matrix->numRows();
    gpuErrChk(cusolverDnDpotrf(Session::getInstance().cuSolverHandle(), CUBLAS_FILL_MODE_LOWER, n,
                               m_d_matrix->raw(), n,
                               m_d_workspace->raw(),
                               m_workspaceSize,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}


template<>
inline int CholeskyFactoriser<float>::factorise() {
    size_t n = m_d_matrix->numRows();
    gpuErrChk(cusolverDnSpotrf(Session::getInstance().cuSolverHandle(), CUBLAS_FILL_MODE_LOWER, n,
                               m_d_matrix->raw(), n,
                               m_d_workspace->raw(),
                               m_workspaceSize,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}

template<>
inline int CholeskyFactoriser<double>::solve(DTensor<double> &rhs) {
    size_t n = m_d_matrix->numRows();
    size_t k = rhs.numel();
    gpuErrChk(cusolverDnDpotrs(Session::getInstance().cuSolverHandle(),
                               CUBLAS_FILL_MODE_LOWER,
                               n, 1,
                               m_d_matrix->raw(), n,
                               rhs.raw(), n,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}

template<>
inline int CholeskyFactoriser<float>::solve(DTensor<float> &rhs) {
    size_t n = m_d_matrix->numRows();
    size_t k = rhs.numel();
    gpuErrChk(cusolverDnSpotrs(Session::getInstance().cuSolverHandle(),
                               CUBLAS_FILL_MODE_LOWER,
                               n, 1,
                               m_d_matrix->raw(), n,
                               rhs.raw(), n,
                               m_d_info->raw()));
    return (*m_d_info)(0);
}
#endif
