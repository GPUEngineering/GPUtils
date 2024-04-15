#include <gtest/gtest.h>
#include "../include/device.cuh"


class DeviceTest : public testing::Test {
protected:
    Context m_context;  ///< Create one context only
    DeviceTest() {}

    virtual ~DeviceTest() {}
};

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