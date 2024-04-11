#include <vector>
#include <iostream>
#include <cublas_v2.h>

#ifndef DEVICE_VECTOR_CUH__
#define DEVICE_VECTOR_CUH__


class Context {

private:
    cublasHandle_t cublasHandle;

public:
    explicit Context()

    noexcept {
        cublasStatus_t stat = cublasCreate(&cublasHandle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
        }
    }

    virtual ~Context()

    noexcept { cublasDestroy(cublasHandle); }

    cublasHandle_t &handle() { return cublasHandle; }

};

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
    DeviceVector(Context *context, size_t n) {
        m_context = context;
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
     * Create device vector from host vector.
     * This allocates memory on the device and copies the host data.
     *
     * @param vec host vector
     */
    DeviceVector(Context *context, const std::vector <TElement> &vec) {
        m_context = context;
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
    size_t capacity() {
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
    bool upload(const std::vector <TElement> &vec) {
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
    void download(TElement *hostData);

    /**
     * Download the device data to a vector
     *
     * @param vec
     */
    void download(std::vector <TElement> &vec);

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
        std::vector <TElement> temp(data.m_numAllocatedElements);
        cudaMemcpy(temp.data(),
                   data.m_d_data,
                   data.m_numAllocatedElements * sizeof(TElement),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < data.m_numAllocatedElements; i++) {
            out << "@(" << i << ") = " << temp[i] << std::endl;
        }
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
    DeviceVector &operator*=(float scalar);


    friend DeviceVector operator+(DeviceVector &a, const DeviceVector &b) {
        DeviceVector n(a.m_context, a.capacity());
        a.deviceCopyTo(n);
        n += b;
        return n;
    }

    friend DeviceVector operator-(DeviceVector &a, const DeviceVector &b) {
        DeviceVector n(a.m_context, a.capacity());
        a.deviceCopyTo(n);
        n -= b;
        return n;
    }

    friend DeviceVector operator*(const float b, DeviceVector &a) {
        DeviceVector n(a.m_context, a.capacity());
        a.deviceCopyTo(n);
        n *= b;
        return n;
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
DeviceVector<float> &DeviceVector<float>::operator+=(const DeviceVector<float> &rhs) {
    const float alpha = 1.;
    cublasSaxpy(m_context->handle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
DeviceVector<float> &DeviceVector<float>::operator*=(float scalar) {
    float alpha = scalar;
    cublasSscal(m_context->handle(), m_numAllocatedElements, &alpha, m_d_data, 1);
    return *this;
}


template<>
DeviceVector<float> &DeviceVector<float>::operator-=(const DeviceVector<float> &rhs) {
    const float alpha = -1.;
    cublasSaxpy(m_context->handle(), m_numAllocatedElements, &alpha, rhs.m_d_data, 1, m_d_data, 1);
    return *this;
}

template<>
float DeviceVector<float>::operator*(const DeviceVector<float> &rhs) const {
    float inn_prod;
    cublasSdot(m_context->handle(), m_numAllocatedElements, m_d_data, 1, rhs.m_d_data, 1, &inn_prod);
    return inn_prod;
}

template<>
float DeviceVector<float>::norm2() const {
    float the_norm;
    cublasSnrm2(m_context->handle(), m_numAllocatedElements, m_d_data, 1, &the_norm);
    return the_norm;
}

template<>
float DeviceVector<float>::sum() const {
    float the_sum;
    cublasSasum(m_context->handle(), m_numAllocatedElements, m_d_data, 1, &the_sum);
    return the_sum;
}

template<typename TElement>
TElement DeviceVector<TElement>::fetchElementFromDevice(size_t i) {
    DeviceVector<TElement> d_element(*this, i, i);
    float xi[1];
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
void DeviceVector<TElement>::download(TElement *hostData) {
    cudaMemcpy(hostData,
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToHost);
}

template<typename TElement>
void DeviceVector<TElement>::download(std::vector <TElement> &vec) {
    vec.reserve(m_numAllocatedElements);
    cudaMemcpy(vec.data(),
               m_d_data,
               m_numAllocatedElements * sizeof(TElement),
               cudaMemcpyDeviceToHost);
}

/**
 * Storage mode for the data of a matrix
 */
enum MatrixStorageMode {
    ColumnMajor, /**< column major storage (preferred/default) */
    RowMajor /**< row major storage */
};

template<typename TElement>
class DeviceMatrix {

private:
    // the data is always stored in CM format
    DeviceVector<TElement> *m_vec = nullptr;
    size_t m_num_rows = 0;

    void destroy() {
        m_num_rows = 0;
        if (m_vec) delete m_vec;
    }

    void rm2cm(std::vector <TElement> vec_rm,
               std::vector <TElement> &vec_cm,
               size_t n_rows,
               size_t n_cols) {
        for (size_t i = 0; i < n_rows; i++) {
            for (size_t j = 0; j < n_cols; j++) {
                float c = vec_rm[j + i * n_cols];
                vec_cm[i + j * n_rows] = c;
            }
        }
    }


public:

    DeviceMatrix(Context *context, size_t n_rows, size_t n_cols) {
        m_num_rows = n_rows;
        m_vec = new DeviceVector<TElement>(context, n_rows * n_cols);
    }

    DeviceMatrix(Context *context,
                 size_t n_rows,
                 const std::vector <TElement> &vec,
                 MatrixStorageMode mode = MatrixStorageMode::ColumnMajor) {
        size_t numel = vec.size();
        m_num_rows = n_rows;
        size_t n_cols = numel / n_rows;
        if (mode == MatrixStorageMode::RowMajor) {
            std::vector <TElement> vec_cm(numel);
            rm2cm(vec, vec_cm, n_rows, n_cols);  // to column-major
            m_vec = new DeviceVector<TElement>(context, vec_cm);
        } else {
            m_vec = new DeviceVector<TElement>(context, vec);
        }
    }

    void upload(const std::vector <TElement> &vec,
                size_t n_rows,
                MatrixStorageMode mode = MatrixStorageMode::ColumnMajor) {
        // TODO implement this
        // make sure enough memory has been reserved

    }

    ~DeviceMatrix() {
        destroy();
    }

    size_t n_rows() {
        return m_num_rows;
    }

    size_t n_cols() {
        return m_vec->capacity() / m_num_rows;
    }

    DeviceMatrix &operator+=(const DeviceMatrix &rhs);

    DeviceMatrix &operator-=(const DeviceMatrix &rhs);

    DeviceMatrix &operator*=(float scalar);

    friend std::ostream &operator<<(std::ostream &out, const DeviceMatrix<TElement> &data) {
        size_t numel = data.m_vec->capacity();
        size_t nr = data.m_num_rows;
        size_t nc = numel / data.m_num_rows;
        out << "DeviceMatrix [" << nr << " x " << nc << "]:" << std::endl;
        std::vector <TElement> temp;
        data.m_vec->download(temp);
        for (size_t i = 0; i < nr; i++) {
            for (size_t j = 0; j < nc; j++) {
                out << temp[j * nr + i] << ", ";
            }
            out << std::endl;
        }
        return out;
    }

};


template<>
DeviceMatrix<float> &DeviceMatrix<float>::operator+=(const DeviceMatrix<float> &rhs) {
    *m_vec += *rhs.m_vec;
    return *this;
}

template<>
DeviceMatrix<float> &DeviceMatrix<float>::operator-=(const DeviceMatrix<float> &rhs) {
    *m_vec -= *rhs.m_vec;
    return *this;
}


template<>
DeviceMatrix<float> &DeviceMatrix<float>::operator*=(float scalar) {
    *m_vec *= scalar;
    return *this;
}

#endif
