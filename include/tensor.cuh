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

#ifndef gpuErrChk
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

#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))
#endif

/* ------------------------------------------------------------------------------------
 *  Session
 * ------------------------------------------------------------------------------------ */


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


template<typename T>
class Tenzor {

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
    Tenzor() = default;

    ~Tenzor() {
        destroy();
    }

    /**
     * Allocates (m, n, k)-tensor
     * @param n
     */
    Tenzor(size_t m, size_t n = 1, size_t k = 1, bool zero = false) {
        m_numRows = m;
        m_numCols = n;
        m_numMats = k;
        size_t size = m * n * k;
        allocateOnDevice(size, zero);
    }

    Tenzor(const std::vector<T> &data, size_t m, size_t n = 1, size_t k = 1) {
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
    Tenzor(const Tenzor &other) {
        m_numMats = other.m_numMats;
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;

        allocateOnDevice(m_numRows * m_numCols * m_numMats);
        gpuErrChk(cudaMemcpy(m_d_data, other.raw(), m_numRows * m_numCols * m_numMats * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

    /**
     * Slicing constructor
     * @param other
     * @param axis
     * @param from
     * @param to
     */
    Tenzor(const Tenzor &other, size_t axis, size_t from, size_t to) {
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

    void deviceCopyTo(Tenzor<T> &other) const;

    T normF() const;

    T sumAbs() const;

    void leastSquares(Tenzor &b);

    /* ------------- OPERATORS ------------- */

    Tenzor &operator=(const Tenzor &other);

    T operator()(size_t i, size_t j=0, size_t k=0);

    Tenzor &operator*=(T scalar);

    Tenzor &operator+=(const Tenzor &rhs);

    Tenzor &operator-=(const Tenzor &rhs);

    Tenzor<T *> pointersToMatrices();

    friend Tenzor operator+(Tenzor &first, const Tenzor &second) {
        Tenzor result(first);
        result += second;
        return result;
    }

    friend Tenzor operator-(Tenzor &first, const Tenzor &second) {
        Tenzor result(first);
        result -= second;
        return result;
    }

    void addAB(Tenzor<T> &A, Tenzor<T> &B, T alpha = 1, T beta = 0);

    friend std::ostream &operator<<(std::ostream &out, const Tenzor<T> &data) {
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
inline size_t Tenzor<T>::numRows() const {
    return m_numRows;
}

template<typename T>
inline size_t Tenzor<T>::numCols() const {
    return m_numCols;
}

template<typename T>
inline size_t Tenzor<T>::numMats() const {
    return m_numMats;
}

template<typename T>
inline size_t Tenzor<T>::numel() const {
    return m_numRows * m_numCols * m_numMats;
}

template<>
inline double Tenzor<double>::normF() const {
    double the_norm;
    gpuErrChk(cublasDnrm2(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &the_norm));
    return the_norm;
}

template<>
inline float Tenzor<float>::normF() const {
    float the_norm;
    gpuErrChk(cublasSnrm2(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &the_norm));
    return the_norm;
}


template<>
inline float Tenzor<float>::sumAbs() const {
    float sumAbsAllElements;
    gpuErrChk(cublasSasum(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &sumAbsAllElements));
    return sumAbsAllElements;
}

template<>
inline double Tenzor<double>::sumAbs() const {
    double sumAbsAllElements;
    gpuErrChk(cublasDasum(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, m_d_data, 1,
                          &sumAbsAllElements));
    return sumAbsAllElements;
}

template<typename T>
inline bool Tenzor<T>::allocateOnDevice(size_t size, bool zero) {
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
inline bool Tenzor<T>::upload(const std::vector<T> &vec) {
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
inline void Tenzor<T>::download(std::vector<T> &vec) const {
    vec.resize(m_numRows * m_numCols * m_numMats);
    gpuErrChk(cudaMemcpy(vec.data(),
                         m_d_data,
                         m_numRows * m_numCols * m_numMats * sizeof(T),
                         cudaMemcpyDeviceToHost));
}

template<typename T>
inline T *Tenzor<T>::raw() const {
    return m_d_data;
}

template<typename T>
inline void Tenzor<T>::deviceCopyTo(Tenzor<T> &elsewhere) const {
    if (elsewhere.numel() < numel()) {
        throw std::invalid_argument("tensor does not fit into destination");
    }
    gpuErrChk(cudaMemcpy(elsewhere.raw(),
                         m_d_data,
                         m_numRows * m_numCols * m_numMats * sizeof(T),
                         cudaMemcpyDeviceToDevice));
}

template<>
inline Tenzor<double> &Tenzor<double>::operator*=(double scalar) {
    double alpha = scalar;
    gpuErrChk(
            cublasDscal(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, m_d_data, 1));
    return *this;
}

template<typename T>
Tenzor<T> &Tenzor<T>::operator=(const Tenzor<T> &other) {
    m_numMats = other.m_numMats;
    m_numRows = other.m_numRows;
    m_numCols = other.m_numCols;
    m_doDestroy = false;
    m_d_data = other.m_d_data;
    return *this;
}

template<>
inline Tenzor<float> &Tenzor<float>::operator*=(float scalar) {
    float alpha = scalar;
    gpuErrChk(
            cublasSscal(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, m_d_data, 1));
    return *this;
}

template<>
inline Tenzor<double> &Tenzor<double>::operator+=(const Tenzor<double> &rhs) {
    const double alpha = 1.;
    gpuErrChk(
            cublasDaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data,
                        1, m_d_data, 1));
    return *this;
}

template<>
inline Tenzor<float> &Tenzor<float>::operator+=(const Tenzor<float> &rhs) {
    const float alpha = 1.;
    gpuErrChk(
            cublasSaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data,
                        1, m_d_data, 1));
    return *this;
}

template<>
inline Tenzor<float> &Tenzor<float>::operator-=(const Tenzor<float> &rhs) {
    const float alpha = -1.;
    cublasSaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data, 1,
                m_d_data, 1);
    return *this;
}

template<>
inline Tenzor<double> &Tenzor<double>::operator-=(const Tenzor<double> &rhs) {
    const double alpha = -1.;
    gpuErrChk(
            cublasDaxpy(Session::getInstance().cuBlasHandle(), m_numRows * m_numCols * m_numMats, &alpha, rhs.m_d_data,
                        1, m_d_data, 1));
    return *this;
}

template<typename T>
inline T Tenzor<T>::operator()(size_t i, size_t j, size_t k) {
    T hostDst;
    size_t offset = i + m_numRows * (j + m_numCols * k);
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + offset, sizeof(T), cudaMemcpyDeviceToHost));
    return hostDst;
}

template<typename T>
inline Tenzor<T *> Tenzor<T>::pointersToMatrices() {
    std::vector<T *> h_pointers(m_numMats);
    size_t numelMat = m_numRows * m_numCols;
    h_pointers[0] = m_d_data;
    for (size_t i = 1; i < m_numMats; i++) {
        h_pointers[i] = m_d_data + i * numelMat;
    }
    Tenzor<T *> t(h_pointers, m_numMats, 1, 1);
    return t;
}

template<>
inline void Tenzor<double>::addAB(Tenzor<double> &A, Tenzor<double> &B, double alpha, double beta) {
    size_t nMat = A.numMats();
    size_t nRA = A.numRows();
    size_t nCA = A.numCols();
    size_t nCB = B.numCols();
    Tenzor<double *> ptrA = A.pointersToMatrices();
    Tenzor<double *> ptrB = B.pointersToMatrices();
    Tenzor<double *> ptr = pointersToMatrices();
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
inline void Tenzor<float>::addAB(Tenzor<float> &A, Tenzor<float> &B, float alpha, float beta) {
    size_t nMat = A.numMats();
    size_t nRA = A.numRows();
    size_t nCA = A.numCols();
    size_t nCB = B.numCols();
    Tenzor<float *> ptrA = A.pointersToMatrices();
    Tenzor<float *> ptrB = B.pointersToMatrices();
    Tenzor<float *> ptr = pointersToMatrices();
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
inline void Tenzor<double>::leastSquares(Tenzor &B) {
    size_t batchSize = numMats();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows || nColsB != 1 || B.numMats() != batchSize)
        throw std::invalid_argument("Least squares rhs size does not equal lhs size");
    if (m_numCols > m_numRows)
        throw std::invalid_argument("Least squares supports tall matrices only");
    int info = 0;
    Tenzor<int> infoArray(batchSize);
    Tenzor<double *> As = pointersToMatrices();
    Tenzor<double *> Bs = B.pointersToMatrices();
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
inline void Tenzor<float>::leastSquares(Tenzor &B) {
    size_t batchSize = numMats();
    size_t nColsB = B.numCols();
    if (B.numRows() != m_numRows || nColsB != 1 || B.numMats() != batchSize)
        throw std::invalid_argument("Least squares rhs size does not equal lhs size");
    if (m_numCols > m_numRows)
        throw std::invalid_argument("Least squares supports tall matrices only");
    int info = 0;
    Tenzor<int> infoArray(batchSize);
    Tenzor<float *> As = pointersToMatrices();
    Tenzor<float *> Bs = B.pointersToMatrices();
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
    Tenzor<T> *m_tensor = nullptr;  /**< pointer to original matrix to be factorised */
    std::unique_ptr<Tenzor<T>> m_Vtr;  /**< matrix V' or right singular vectors */
    std::unique_ptr<Tenzor<T>> m_S;
    std::unique_ptr<Tenzor<T>> m_U;  /**< matrix U or left singular vectors*/
    std::unique_ptr<Tenzor<T>> m_workspace;  /**< workspace vector */
    std::unique_ptr<Tenzor<int>> m_info;  /**< status code of computation */
    std::unique_ptr<Tenzor<unsigned int>> m_rank;
    bool m_computeU = false;  /**< whether to compute U */
    bool m_destroyMatrix = true; /**< whether to sacrifice original matrix */

    /**
     * Checks whether matrix is tall; throws invalid_argument if not
     * @param mat given matrix
     */
    void checkMatrix(Tenzor<T> &tenz) {
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
    Svd(Tenzor<T> &mat,
        bool computeU = false,
        bool destroyMatrix = true) {
        checkMatrix(mat);
        m_destroyMatrix = destroyMatrix;
        m_tensor = (destroyMatrix) ? &mat : new Tenzor<T>(mat);
        m_computeU = computeU;
        size_t m = mat.numRows();
        size_t n = mat.numCols();
        size_t k = std::min(m, n);
        computeWorkspaceSize(m, n);
        m_workspace = std::make_unique<Tenzor<T>>(m_lwork, 1, 1);
        m_Vtr = std::make_unique<Tenzor<T>>(n, n, 1);
        m_S = std::make_unique<Tenzor<T>>(k, 1, 1);
        m_info = std::make_unique<Tenzor<int>>(1, 1, 1);
        m_rank = std::make_unique<Tenzor<unsigned int>>(1, 1, 1);
        if (computeU) m_U = std::make_unique<Tenzor<T>>(m, m, 1);
    }

    /**
     * Perform factorisation
     * @return status code
     *
     * Warning: the given matrix is destroyed
     */
    int factorise();

    Tenzor<T> singularValues() const {
        return *m_S;
    }

    Tenzor<T> rightSingularVectors() const {
        return *m_Vtr;
    }

    std::optional<Tenzor<T>> leftSingularVectors() const {
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

#endif
