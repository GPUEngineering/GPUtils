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
        cublasCreate(&m_cublasHandle);
        cusolverDnCreate(&m_cusolverHandle);
    }

    ~Session() {
        cublasDestroy(m_cublasHandle);
        cusolverDnDestroy(m_cusolverHandle);
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
    size_t m_numAllocatedElements = 0;
    size_t m_numRows = 0;
    size_t m_numCols = 0;
    size_t m_numMats = 0;
    bool m_doDestroy = false;

    bool destroy() {
        if (!m_doDestroy) return false;
        if (m_d_data) cudaFree(m_d_data);
        m_numAllocatedElements = 0;
        m_d_data = nullptr;
        return true;
    }

    bool allocateOnDevice(size_t size);

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
    Tenzor(size_t m, size_t n = 1, size_t k = 1) {
        m_numRows = m;
        m_numCols = n;
        m_numMats = k;
        size_t size = m * n * k;
        allocateOnDevice(size);
    }

    /**
     * Copy constructor
     */
    Tenzor(const Tenzor &other) {
        m_numMats = other.m_numMats;
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;
        allocateOnDevice(other.m_numAllocatedElements);
        cudaMemcpy(m_d_data, other.raw(), m_numAllocatedElements * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    Tenzor(const Tenzor &other, size_t axis, size_t from, size_t to) {
        if (from > to) throw std::invalid_argument("from > to");
        size_t offset = 0;
        if (axis == 2) {
            offset = other.m_numRows * other.m_numCols * from;
            m_numRows = other.m_numRows;
            m_numCols = other.m_numCols;
            m_numMats = to - from + 1;
        } else if (axis == 1) {
            offset = other.m_numCols * from;
            m_numRows = other.m_numRows;
            m_numCols = to - from + 1;
            m_numMats = 1;
        } else if (axis == 0) {
            offset = from;
            m_numRows = to - from + 1;
            m_numCols = 1;
            m_numMats = 1;
        }
        m_d_data = other.m_d_data + offset;
        m_numAllocatedElements = m_numRows * m_numCols * m_numMats;
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

    /* OPERATORS */
    Tenzor &operator=(const Tenzor &other) {
        m_numMats = other.m_numMats;
        m_numRows = other.m_numRows;
        m_numCols = other.m_numCols;
        m_doDestroy = false;
        m_d_data = other.m_d_data;
        m_numAllocatedElements = other.m_numAllocatedElements;
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
    return m_numAllocatedElements;
}

template<>
inline double Tenzor<double>::normF() const {
    double the_norm;
    cublasDnrm2(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_norm);
    return the_norm;
}

template<>
inline float Tenzor<float>::normF() const {
    float the_norm;
    cublasSnrm2(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, m_d_data, 1, &the_norm);
    return the_norm;
}

template<typename T>
inline bool Tenzor<T>::allocateOnDevice(size_t size) {
    if (size <= 0) return false;
    if (size <= m_numAllocatedElements) return true;
    destroy();
    m_doDestroy = true;
    size_t buffer_size = size * sizeof(T);
    bool cudaStatus = cudaMalloc(&m_d_data, buffer_size);
    if (cudaStatus != cudaSuccess) return false;
    m_numAllocatedElements = size;
    return true;
}

template<typename T>
bool Tenzor<T>::upload(const std::vector<T> &vec) {
    size_t size = vec.size();
    // make sure vec is of right size
    if (size != m_numAllocatedElements) throw std::invalid_argument("vec has wrong size");
    if (size <= m_numAllocatedElements) {
        size_t buffer_size = size * sizeof(T);
        cudaMemcpy(m_d_data, vec.data(), buffer_size, cudaMemcpyHostToDevice);
    }
    return true;
}

template<typename T>
void Tenzor<T>::download(std::vector<T> &vec) const {
    vec.reserve(m_numAllocatedElements);
    cudaMemcpy(vec.data(),
               m_d_data,
               m_numAllocatedElements * sizeof(T),
               cudaMemcpyDeviceToHost);
}

template<typename T>
inline T *Tenzor<T>::raw() const {
    return m_d_data;
}

template<typename T>
void Tenzor<T>::deviceCopyTo(Tenzor<T> &elsewhere) const {
    elsewhere.allocateOnDevice(m_numAllocatedElements);
    cudaMemcpy(elsewhere.raw(),
               m_d_data,
               m_numAllocatedElements * sizeof(T),
               cudaMemcpyDeviceToDevice);
}

template<>
inline Tenzor<double> &Tenzor<double>::operator*=(double scalar) {
    double alpha = scalar;
    cublasDscal(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, m_d_data, 1);
    return *this;
}

template<>
inline Tenzor<float> &Tenzor<float>::operator*=(float scalar) {
    float alpha = scalar;
    cublasSscal(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, m_d_data, 1);
    return *this;
}

template<>
inline Tenzor<double> &Tenzor<double>::operator+=(const Tenzor<double> &rhs) {
    const double alpha = 1.;
    cublasDaxpy(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
inline Tenzor<float> &Tenzor<float>::operator+=(const Tenzor<float> &rhs) {
    const float alpha = 1.;
    cublasSaxpy(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
inline Tenzor<float> &Tenzor<float>::operator-=(const Tenzor<float> &rhs) {
    const float alpha = -1.;
    cublasSaxpy(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1,
                m_d_data, 1);
    return *this;
}

template<>
inline Tenzor<double> &Tenzor<double>::operator-=(const Tenzor<double> &rhs) {
    const double alpha = -1.;
    cublasDaxpy(Session::getInstance().cuBlasHandle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1,
                m_d_data, 1);
    return *this;
}

template<typename T>
T Tenzor<T>::operator()(size_t i, size_t j, size_t k) {
    T hostDst;
    size_t offset = i + m_numRows * (j + m_numCols * k);
    cudaMemcpy(&hostDst, m_d_data + offset, sizeof(T), cudaMemcpyDeviceToHost);
    return hostDst;
}


#endif
