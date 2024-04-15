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
void row2Col() {
    size_t rows = 4;
    size_t cols = 3;
    std::vector<T> expectedRowOrder = {1, 2, 3,
                                       4, 5, 6,
                                       7, 8, 9,
                                       10, 11, 12};
    std::vector<T> expectedColOrder = {1, 4, 7, 10,
                                       2, 5, 8, 11,
                                       3, 6, 9, 12};
    std::vector<T> matRow(expectedRowOrder);
    std::vector<T> matCol;
    row2col(matRow, matCol, rows, cols);
    ASSERT_EQ(expectedColOrder, matCol);
    // now test using same storage
    std::vector<T> mat(expectedRowOrder);
    row2col(mat, mat, rows, cols);
    ASSERT_EQ(expectedColOrder, mat);
}

TEST_F(DeviceTest, row2Col) {
    row2Col<float>();
    row2Col<double>();
}

template<typename T>
void col2Row() {
    size_t rows = 4;
    size_t cols = 3;
    std::vector<T> expectedRowOrder = {1, 2, 3,
                                       4, 5, 6,
                                       7, 8, 9,
                                       10, 11, 12};
    std::vector<T> expectedColOrder = {1, 4, 7, 10,
                                       2, 5, 8, 11,
                                       3, 6, 9, 12};
    std::vector<T> matCol(expectedColOrder);
    std::vector<T> matRow;
    col2row(matCol, matRow, rows, cols);
    ASSERT_EQ(expectedRowOrder, matRow);
    // now test using same storage
    std::vector<T> mat(expectedColOrder);
    col2row(mat, mat, rows, cols);
    ASSERT_EQ(expectedRowOrder, mat);
}

TEST_F(DeviceTest, col2Row) {
    col2Row<float>();
    col2Row<double>();
}

template<typename T>
void vectorCapacity(Context &context) {
    DeviceVector<T> four(context, 4);
    EXPECT_EQ(4, four.capacity());
    DeviceVector<T> five(context, 0);
    five.allocateOnDevice(5);
    EXPECT_EQ(5, five.capacity());
}

TEST_F(DeviceTest, vectorCapacity) {
    vectorCapacity<float>(m_context);
    vectorCapacity<double>(m_context);
}

/* ---------------------------------------
 * Matrix Dimensions
 * .numRows and .numCols
 * --------------------------------------- */

template<typename T>
void matrixDimensions(Context &context) {
    DeviceMatrix<T> fourByThree(context, 4, 3);
    EXPECT_EQ(4, fourByThree.numRows());
    EXPECT_EQ(3, fourByThree.numCols());
}

TEST_F(DeviceTest, matrixDimensions) {
    matrixDimensions<float>(m_context);
    matrixDimensions<double>(m_context);
}

/* ---------------------------------------
 * DeviceVector data transfer
 * .upload and .download
 * --------------------------------------- */

template<typename T>
void transfer(Context &context) {
    size_t rows = 3;
    size_t cols = 2;
    size_t n = rows * cols;
    std::vector<T> data = {1, 2,
                           3, 4,
                           5, 6};
    std::vector<T> dataCM = {1, 3, 5,
                             2, 4, 6};
    std::vector<T> resultVec(n);
    std::vector<T> resultCM(n);
    std::vector<T> resultRM(n);
    DeviceVector<T> vec(context, n);
    vec.upload(data);
    vec.download(resultVec);
    EXPECT_EQ(data, resultVec);
    DeviceMatrix<T> mat(context, rows, cols);
    mat.upload(dataCM, rows, MatrixStorageMode::columnMajor);
    mat.asVector().download(resultCM);
    EXPECT_EQ(resultCM, dataCM);
    mat.upload(data, rows, MatrixStorageMode::rowMajor);
    mat.asVector().download(resultRM);
    EXPECT_EQ(resultRM, dataCM);
}

TEST_F(DeviceTest, transfer) {
    transfer<float>(m_context);
    transfer<double>(m_context);
}


/* ---------------------------------------
 * Matrix-vector multiplication
 * --------------------------------------- */

template<typename T>
void matrixVectorOperatorAsterisk(Context &context) {
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

TEST_F(DeviceTest, matrixVectorOperatorAsterisk) {
    matrixVectorOperatorAsterisk<float>(m_context);
    matrixVectorOperatorAsterisk<double>(m_context);
}

template<typename T>
void addAB(Context &context) {
    size_t nRowsC = 4;
    size_t nColsC = 3;
    size_t k = 2;
    std::vector<T> matC = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<T> matA = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<T> matB = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(nRowsC * nColsC);
    DeviceMatrix<T> d_matC(context, nRowsC, matC, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matA(context, nRowsC, matA, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matB(context, k, matB, MatrixStorageMode::rowMajor);
    d_matC.addAB(d_matA, d_matB);
    d_matC.asVector().download(result);
    std::vector<T> expected = {10, 23, 36, 49, 14, 31, 48, 65, 18, 39, 60, 81};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, addAB) {
    addAB<float>(m_context);
    addAB<double>(m_context);
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