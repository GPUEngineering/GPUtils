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

    /* ------------- OPERATORS ------------- */

    Tenzor &operator=(const Tenzor &other) {
        m_numMats = other.m_numMats;
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;
        m_doDestroy = false;
        m_d_data = other.m_d_data;
        return *this;
    }

    T operator()(size_t i, size_t j, size_t k);

    Tenzor &operator*=(T scalar);

    Tenzor &operator+=(const Tenzor &rhs);

    Tenzor &operator-=(const Tenzor &rhs);

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
bool Tenzor<T>::upload(const std::vector<T> &vec) {
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
void Tenzor<T>::download(std::vector<T> &vec) const {
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
void Tenzor<T>::deviceCopyTo(Tenzor<T> &elsewhere) const {
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
T Tenzor<T>::operator()(size_t i, size_t j, size_t k) {
    T hostDst;
    size_t offset = i + m_numRows * (j + m_numCols * k);
    gpuErrChk(cudaMemcpy(&hostDst, m_d_data + offset, sizeof(T), cudaMemcpyDeviceToHost));
    return hostDst;
}


#endif
