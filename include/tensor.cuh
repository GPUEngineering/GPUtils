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
 *  Context
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

public:
    /**
    * Constructs a DeviceVector object
    */
    Tenzor() = default;

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

    bool allocateOnDevice(size_t size);

    T *raw();

    bool upload(const T *dataArray, size_t size);

    bool upload(const std::vector<T> &vec);

    void download(T *hostData) const;

    void download(std::vector<T> &vec) const;

    friend std::ostream &operator<<(std::ostream &out, const Tenzor<T> &data) {
        size_t nr = data.m_numRows, nc = data.m_numCols, nm = data.m_numMats;
        out << "Tensor [" << data.m_numRows << " x "
            << data.m_numCols << " x "
            << data.m_numMats << "]:" << std::endl;
        std::vector<T> temp;
        data.download(temp);
        for (size_t k = 0; k < nm; k++) {
            out << ">> layer: " << k << std::endl;
            for (size_t i = 0; i < nr ; i++) {
                for (size_t j = 0; j < nc; j++) {
                    out << std::setw(10) << temp[nr * (nc * k + j) + i] << ", ";
                }
                out << std::endl;
            }
        }
//        for (size_t i = 0; i < data.m_numAllocatedElements - 1; i++) {
//            out << std::setw(10) << temp[i] << std::endl;
//        }
//        out << std::setw(10) << temp[data.m_numAllocatedElements - 1] << std::endl;
        return out;
    }

}; /* END OF TENZOR */


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
    return upload(vec.data(), vec.size());
}

template<typename T>
bool Tenzor<T>::upload(const T *dataArray, size_t size) {
    if (!this->allocateOnDevice(size)) return false;
    if (size <= m_numAllocatedElements) {
        size_t buffer_size = size * sizeof(T);
        cudaMemcpy(m_d_data, dataArray, buffer_size, cudaMemcpyHostToDevice);
    }
    return true;
}

template<typename T>
void Tenzor<T>::download(T *hostData) const {
    cudaMemcpy(hostData,
               m_d_data,
               m_numAllocatedElements * sizeof(T),
               cudaMemcpyDeviceToHost);
}

template<typename T>
void Tenzor<T>::download(std::vector<T> &vec) const {
    vec.reserve(m_numAllocatedElements);
    download(vec.data());
}

template<typename T>
inline T *Tenzor<T>::raw() {
    return m_d_data;
}

#endif
