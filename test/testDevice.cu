#include <gtest/gtest.h>
#include "../include/device.cuh"


class DeviceTest : public testing::Test {
protected:
    Context m_context;  ///< Create one context only
    DeviceTest() {}

    virtual ~DeviceTest() {}
};

/* =======================================
 * DeviceVector<T>
 * ======================================= */

/* ---------------------------------------
 * Vector capacity and allocateOnDevice
 * --------------------------------------- */
template<typename T>
void VectorCapacity(Context &context) {
    DeviceVector<T> four(context, 4);
    EXPECT_EQ(4, four.capacity());
    DeviceVector<T> five(context, 0);
    five.allocateOnDevice(5);
    EXPECT_EQ(5, five.capacity());
}

TEST_F(DeviceTest, VectorCapacity) {
    VectorCapacity<float>(m_context);
    VectorCapacity<double>(m_context);
}

/* ---------------------------------------
 * Basic constructor (extreme cases)
 * --------------------------------------- */
template<typename T>
void DeviceVectorBasicConstructor(Context &context) {
    DeviceVector<T> empty(context, 0);
    EXPECT_EQ(0, empty.capacity());
    DeviceVector<T> big(context, 24000);
    EXPECT_EQ(24000, big.capacity());
}

TEST_F(DeviceTest, DeviceVectorBasicConstructor) {
    DeviceVectorBasicConstructor<float>(m_context);
    DeviceVectorBasicConstructor<double>(m_context);
    DeviceVectorBasicConstructor<int>(m_context);
}


/* ---------------------------------------
 * Slice constructor
 * --------------------------------------- */

template<typename T>
void DeviceVectorSliceConstructor(Context &context) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> xSlice(x, 1, 3);
    EXPECT_EQ(3, xSlice.capacity());
    EXPECT_EQ(2, xSlice(0));
    EXPECT_EQ(3, xSlice(1));
}

TEST_F(DeviceTest, DeviceVectorSliceConstructor) {
    DeviceVectorSliceConstructor<float>(m_context);
    DeviceVectorSliceConstructor<double>(m_context);
    DeviceVectorSliceConstructor<int>(m_context);
}

/* ---------------------------------------
 * Copy constructor
 * --------------------------------------- */

template<typename T>
void DeviceVectorCopyConstructor(Context &context) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> xCopy(x);
    EXPECT_EQ(5, xCopy.capacity());
    EXPECT_EQ(1, xCopy(0));
    EXPECT_EQ(5, xCopy(4));
}

TEST_F(DeviceTest, DeviceVectorCopyConstructor) {
    DeviceVectorCopyConstructor<float>(m_context);
    DeviceVectorCopyConstructor<double>(m_context);
    DeviceVectorCopyConstructor<int>(m_context);
}

/* ---------------------------------------
 * Operator * (dot product)
 * --------------------------------------- */
template<typename T>
void DeviceVectorDotProduct(Context &context) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    std::vector<T> dataY{-1, 4, -6, 9, 10};
    DeviceVector<T> y(context, dataY);
    T dotProduct = x * y;
    EXPECT_EQ(75, dotProduct);
}

TEST_F(DeviceTest, DeviceVectorDotProduct) {
    DeviceVectorDotProduct<float>(m_context);
    DeviceVectorDotProduct<double>(m_context);
}


/* ---------------------------------------
 * Norm2
 * --------------------------------------- */
template<typename T>
void DeviceVectorEuclideanNorm(Context &context, T epsilon) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    T nrmX = x.norm2();
    EXPECT_NEAR(7.416198487095663, nrmX, epsilon);
}

TEST_F(DeviceTest, DeviceVectorEuclideanNorm) {
    DeviceVectorEuclideanNorm<float>(m_context, 1e-4);
    DeviceVectorEuclideanNorm<double>(m_context, 1e-12);
}

/* ---------------------------------------
 * Norm-1
 * --------------------------------------- */
template<typename T>
void DeviceVectorNorm1(Context &context) {
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> y(context, dataY);
    T nrmX = y.norm1();
    EXPECT_EQ(30, nrmX);
}

TEST_F(DeviceTest, DeviceVectorNorm1) {
    DeviceVectorNorm1<float>(m_context);
    DeviceVectorNorm1<double>(m_context);
}

/* ---------------------------------------
 * Sum of vectors (operator +)
 * --------------------------------------- */
template<typename T>
void DeviceVectorSum(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    auto sum = x + y;
    EXPECT_EQ(9, sum(0));
    EXPECT_EQ(60, sum(4));
}

TEST_F(DeviceTest, DeviceVectorSum) {
    DeviceVectorSum<float>(m_context);
    DeviceVectorSum<double>(m_context);
}


/* ---------------------------------------
 * Scalar product (scaling)
 * --------------------------------------- */
template<typename T>
void DeviceVectorScaling(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., -50.};
    DeviceVector<T> x(context, dataX);
    T alpha = 2.;
    auto scaledX = alpha * x;
    EXPECT_EQ(20, scaledX(0));
    EXPECT_EQ(-100, scaledX(4));
}

TEST_F(DeviceTest, DeviceVectorScaling) {
    DeviceVectorScaling<float>(m_context);
    DeviceVectorScaling<double>(m_context);
}

/* ---------------------------------------
 * Scalar product (scaling in place)
 * --------------------------------------- */
template<typename T>
void DeviceVectorScalingInPlace(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., -50.};
    DeviceVector<T> x(context, dataX);
    T alpha = 2.;
    x *= alpha;
    EXPECT_EQ(20, x(0));
    EXPECT_EQ(-100, x(4));
}

TEST_F(DeviceTest, DeviceVectorScalingInPlace) {
    DeviceVectorScalingInPlace<float>(m_context);
    DeviceVectorScalingInPlace<double>(m_context);
}

/* ---------------------------------------
 * Difference of vectors (operator -)
 * --------------------------------------- */
template<typename T>
void DeviceVectorDiff(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    auto sum = x - y;
    EXPECT_EQ(11, sum(0));
    EXPECT_EQ(40, sum(4));
}

TEST_F(DeviceTest, DeviceVectorDiff) {
    DeviceVectorDiff<float>(m_context);
    DeviceVectorDiff<double>(m_context);
}

/* ---------------------------------------
 * Device-to-device copy with slicing
 * --------------------------------------- */
template<typename T>
void DeviceVectorDeviceToDeviceCopy(Context &context) {
    std::vector<T> dataX{-1., 2., 3., 4., 6.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> xSlice(x, 1, 3);

    std::vector<T> dataY{5, 5, 5};
    DeviceVector<T> y(context, dataY);

    y.deviceCopyTo(xSlice);

    std::vector<T> xExpected{-1., 5., 5., 5., 6.};
    std::vector<T> h_x(5);
    x.download(h_x);
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(xExpected[i], h_x[i]);
    }
}

TEST_F(DeviceTest, DeviceVectorDeviceToDeviceCopy) {
    DeviceVectorDeviceToDeviceCopy<float>(m_context);
    DeviceVectorDeviceToDeviceCopy<double>(m_context);
}


/* ---------------------------------------
 * Operator +=
 * --------------------------------------- */
template<typename T>
void DeviceVectorOpPlusEq(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    x += y;
    EXPECT_EQ(9, x(0));
    EXPECT_EQ(49, x(3));
    EXPECT_EQ(60, x(4));
}

TEST_F(DeviceTest, DeviceVectorOpPlusEq) {
    DeviceVectorOpPlusEq<float>(m_context);
    DeviceVectorOpPlusEq<double>(m_context);
}

/* ---------------------------------------
 * Operator -=
 * --------------------------------------- */
template<typename T>
void DeviceVectorOpMinusEq(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    x -= y;
    EXPECT_EQ(11, x(0));
    EXPECT_EQ(16, x(1));
    EXPECT_EQ(36, x(2));
}

TEST_F(DeviceTest, DeviceVectorOpMinusEq) {
    DeviceVectorOpMinusEq<float>(m_context);
    DeviceVectorOpMinusEq<double>(m_context);
}


/* ---------------------------------------
 * Upload data
 * --------------------------------------- */
template<typename T>
void DeviceVectorUploadData(Context &context) {
    std::vector<T> data = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(data.size());
    DeviceVector<T> vec(context, data.size());
    vec.upload(data);
    vec.download(result);
    EXPECT_EQ(data, result);
}

TEST_F(DeviceTest, DeviceVectorUploadData) {
    DeviceVectorUploadData<float>(m_context);
    DeviceVectorUploadData<double>(m_context);
}



/* =======================================
 * DeviceMatrix<T>
 * ======================================= */


/* ---------------------------------------
 * Matrix Dimensions
 * .numRows and .numCols
 * --------------------------------------- */

template<typename T>
void MatrixDimensions(Context &context) {
    DeviceMatrix<T> fourByThree(context, 4, 3);
    EXPECT_EQ(4, fourByThree.numRows());
    EXPECT_EQ(3, fourByThree.numCols());
}

TEST_F(DeviceTest, MatrixDimensions) {
    MatrixDimensions<float>(m_context);
    MatrixDimensions<double>(m_context);
}

/* ---------------------------------------
 * DeviceVector data transfer
 * .upload and .download
 * --------------------------------------- */

template<typename T>
void Transfer(Context &context) {
    std::vector<T> data = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(data.size());

    DeviceVector<T> vec(context, data.size());
    vec.upload(data);
    vec.download(result);
    EXPECT_EQ(data, result);

    DeviceMatrix<T> mat(context, 3, 2);
    mat.upload(data, MatrixStorageMode::rowMajor);
    mat.asVector().download(result);
    EXPECT_EQ(data, result);
}

TEST_F(DeviceTest, Transfer) {
    Transfer<float>(m_context);
    Transfer<double>(m_context);
}


/* ---------------------------------------
 * Matrix-vector multiplication
 * --------------------------------------- */

template<typename T>
void MatrixVectorOperatorAsterisk(Context &context) {
    size_t rows = 4;
    std::vector<T> mat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<T> vec = {1, 2, 3};
    std::vector<T> result(rows);

    DeviceMatrix<T> d_mat(context, rows, mat, MatrixStorageMode::rowMajor);
    DeviceVector<T> d_vec(context, vec);
    auto d_result = d_mat * d_vec;
    d_result.download(result);
    std::vector<T> expected = {14, 32, 50, 68};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, MatrixVectorOperatorAsterisk) {
    MatrixVectorOperatorAsterisk<float>(m_context);
    MatrixVectorOperatorAsterisk<double>(m_context);
}


/* ---------------------------------------
 * Indexing Vectors
 * --------------------------------------- */

template<typename T>
void IndexingVectors(Context &context) {
    std::vector<T> xData{1.0, 2.0, 3.0, 6.0, 7.0, 8.0};
    DeviceVector<T> x(context, xData);
    EXPECT_EQ(1., x(0));
    EXPECT_EQ(7., x(4));
}

TEST_F(DeviceTest, IndexingVectors) {
    IndexingVectors<float>(m_context);
    IndexingVectors<double>(m_context);
}


/* ---------------------------------------
 * Indexing Matrices
 * --------------------------------------- */

template<typename T>
void IndexingMatrices(Context &context) {
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0};
    DeviceMatrix<T> B(context, 2, bData, MatrixStorageMode::rowMajor);
    EXPECT_EQ(2., B(0, 1));
    EXPECT_EQ(7., B(1, 1));
}

TEST_F(DeviceTest, IndexingVectorsMatrices) {
    IndexingMatrices<float>(m_context);
    IndexingMatrices<double>(m_context);
}


/* ---------------------------------------
 * Get Matrix Rows
 * --------------------------------------- */

template<typename T>
void GetMatrixRows(Context &context) {
    size_t k = 6;
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0,
                         12.0, 13.0, 14.0,
                         15.0, 16.0, 17.0,
                         18.0, 19.0, 20.0};
    DeviceMatrix<T> B(context, k, bData, MatrixStorageMode::rowMajor);
    auto copiedRows = B.getRows(1, 4);
    EXPECT_EQ(6., copiedRows(0, 0));
    EXPECT_EQ(10., copiedRows(1, 1));
    EXPECT_EQ(16., copiedRows(3, 1));
    EXPECT_EQ(17., copiedRows(3, 2));
}

TEST_F(DeviceTest, GetMatrixRows) {
    GetMatrixRows<float>(m_context);
    GetMatrixRows<double>(m_context);
}


/* =======================================
 * SVD
 * ======================================= */

/* ---------------------------------------
 * Computation of singular values
 * and matrix rank
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void SingularValuesComputation(Context &context, float epsilon) {
    size_t k = 8;
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,
                         6.0, 7.0, 8.0,};
    DeviceMatrix<T> B(context, k, bData, rowMajor);
    SvdFactoriser<T> svdEngine(context, B, true, false);
    EXPECT_EQ(0, svdEngine.factorise());
    auto S = svdEngine.singularValues();
    unsigned int r = svdEngine.rank();
    EXPECT_EQ(2, r);
    EXPECT_NEAR(32.496241123753592, S(0), epsilon); // value from MATLAB
    EXPECT_NEAR(0.997152358903242, S(1), epsilon); // value from MATLAB
}

TEST_F(DeviceTest, SingularValues) {
    SingularValuesComputation<float>(m_context, 1e-4);
    SingularValuesComputation<double>(m_context, 1e-7);
}