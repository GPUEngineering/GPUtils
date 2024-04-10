#include <vector>
#include <iostream>
#include <memory>
#include <cublas_v2.h>
#include "include/device_vector.cuh"

#define THREADS_PER_BLOCK 512
#define DIM2BLOCKS(n) ((n) / THREADS_PER_BLOCK + ((n) % THREADS_PER_BLOCK != 0))

static __global__ void maxWithZero(float *x, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = max(0., x[i]);
}

static __global__ void projectOnSOC(float *x, float scaling, float norm, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) x[i] *= scaling;
    if (i == n - 1) x[i] = scaling * norm;
}



class Set {

protected:
    Context &m_context;
    size_t m_dimension = 0;

    explicit Set(Context &context, size_t dim) : m_context(context), m_dimension(dim) {};

public:

    virtual void project(DeviceVector<float> &x) = 0;

    size_t dimension() {
        return m_dimension;
    }
};

class NonnegativeOrthant : public Set {
public:
    NonnegativeOrthant(Context &context, size_t dim) : Set(context, dim) {}

    void project(DeviceVector<float> &x) {
        maxWithZero<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(x.get(), m_dimension);
    }
};

class SOC : public Set {

public:
    explicit SOC(Context &context, size_t dim) : Set(context, dim) {}

    void project(DeviceVector<float> &x) {
        /* Determine the norm of the first n-1 elements of x */
        float nrm;
        size_t n = x.capacity();
        cublasStatus_t status = cublasSnrm2(m_context.handle(), n - 1, x.get(), 1, &nrm);
        if (status != CUBLAS_STATUS_SUCCESS) printf("Oops! Computation of norm failed!");
        float xLastElement = x.fetchElementFromDevice(n - 1);
        if (nrm <= xLastElement) return;
        float scaling = (nrm + xLastElement) / (2. * nrm);
        projectOnSOC<<<DIM2BLOCKS(m_dimension), THREADS_PER_BLOCK>>>(x.get(), scaling, nrm, m_dimension);
    }
};

class Cartesian : public Set {
private:
    std::vector<Set *> m_cones;
public:
    explicit Cartesian(Context &context) : Set(context, 0) {}

    void addCone(Set &cone) {
        m_cones.push_back(&cone);
        m_dimension += cone.dimension();
    }

    void project(DeviceVector<float> &x) {
        if (x.capacity() != m_dimension)
            throw std::invalid_argument("dim x is wrong!");
        size_t start = 0;
        for (Set *set: m_cones) {
            size_t ni = set->dimension();
            size_t end = start + ni - 1;
            DeviceVector<float> xSlice(x, start, end);
            set->project(xSlice);
            start += ni;
        }
    }
};



void t_operations_with_vectors(){
    Context context;
    // Host data
    std::vector<float> h_a{4., -5., 6., 9., 8., 5., 9., -10.2, 9., 11.};
    std::vector<float> h_b{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};

    // Device data
    DeviceVector<float> a(&context, h_a);
    DeviceVector<float> b(&context, h_b);

    a += b;

    float x = a * b;

    std::cout << a;
    std::cout << b;
    std::cout << "a * b = " << x << std::endl;
    std::cout << "||a|| = " << a.norm2() << std::endl;

    a -= b;
    std::cout << a;

    std::cout << "Σ a_i = " << a.sum() << std::endl;

    DeviceVector<float> c = a + b;
    std::cout << c;
}


void t_operations_with_matrices() {
    Context context;
    std::vector<float> h_data{1., 2., 3.,
                              4., 5., 6.};

    // Experiments with DeviceMatrix
    DeviceMatrix<float> mat(&context, 2, h_data, MatrixStorageMode::ColumnMajor);
    std::cout << mat << std::endl;

}
int main() {

    t_operations_with_vectors();
//    t_operations_with_matrices();





//    std::vector<float> xHost{4., -5., 6., 9., 8., 5., 9., -10., 9., 11.};
//
//    /* PosOrth(2)*/
//    NonnegativeOrthant orthant(context, 3);
//
//    /* SOC(4) */
//    SOC soc(context, 4);
//
//    /* Cartesian product: X = PosOrth(2) x SOC(4) x PosOrth(2) */
//    Cartesian cartesian(context);
//    cartesian.addCone(orthant);
//    cartesian.addCone(soc);
//    cartesian.addCone(orthant);
//
//    std::unique_ptr<Set> soc2 = std::make_unique<NonnegativeOrthant>(context, 2);
//
//
//    DeviceVector<float> x(xHost);
//    cartesian.project(x);
//
//    x.download(xHost);
//    printf("\nVector x after projection:\n");
//    for (float xi: xHost) {
//        printf("xi = %g\n", xi);
//    }

    return 0;
}
