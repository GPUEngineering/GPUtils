#include <gtest/gtest.h>
#include "../include/device.cuh"


class DeviceTest : public testing::Test {
protected:
    Context m_context;  ///< Create one context only
    DeviceTest() {}

    virtual ~DeviceTest() {}
};

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
