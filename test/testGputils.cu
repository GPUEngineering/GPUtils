#include <gtest/gtest.h>
#include "../include/gputils.cuh"
#define PRECISION 1e-6


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
 * Row-major <-> column-major storage
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

/* ---------------------------------------
 * DeviceVector
 * .capacity and .allocateOnDevice
 * --------------------------------------- */

template<typename T>
void deviceVectorCapacity(Context &context) {
    DeviceVector<T> four(context, 4);
    EXPECT_EQ(4, four.capacity());
    DeviceVector<T> five(context, 0);
    five.allocateOnDevice(5);
    EXPECT_EQ(5, five.capacity());
}

TEST_F(DeviceTest, deviceVectorCapacity) {
    deviceVectorCapacity<float>(m_context);
    deviceVectorCapacity<double>(m_context);
}

/* ---------------------------------------
 * Basic constructor (extreme cases)
 * --------------------------------------- */
template<typename T>
void deviceVectorBasicConstructor(Context &context) {
    DeviceVector<T> empty(context, 0);
    EXPECT_EQ(0, empty.capacity());
    DeviceVector<T> big(context, 24000);
    EXPECT_EQ(24000, big.capacity());
}

TEST_F(DeviceTest, deviceVectorBasicConstructor) {
    deviceVectorBasicConstructor<float>(m_context);
    deviceVectorBasicConstructor<double>(m_context);
    deviceVectorBasicConstructor<int>(m_context);
}


/* ---------------------------------------
 * Slice constructor
 * --------------------------------------- */

template<typename T>
void deviceVectorSliceConstructor(Context &context) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> xSlice(x, 1, 3);
    EXPECT_EQ(3, xSlice.capacity());
    EXPECT_EQ(2, xSlice(0));
    EXPECT_EQ(3, xSlice(1));
}

TEST_F(DeviceTest, deviceVectorSliceConstructor) {
    deviceVectorSliceConstructor<float>(m_context);
    deviceVectorSliceConstructor<double>(m_context);
    deviceVectorSliceConstructor<int>(m_context);
}

/* ---------------------------------------
 * Copy constructor
 * --------------------------------------- */

template<typename T>
void deviceVectorCopyConstructor(Context &context) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> xCopy(x);
    EXPECT_EQ(5, xCopy.capacity());
    EXPECT_EQ(1, xCopy(0));
    EXPECT_EQ(5, xCopy(4));
}

TEST_F(DeviceTest, deviceVectorCopyConstructor) {
    deviceVectorCopyConstructor<float>(m_context);
    deviceVectorCopyConstructor<double>(m_context);
    deviceVectorCopyConstructor<int>(m_context);
}

/* ---------------------------------------
 * Operator * (dot product)
 * --------------------------------------- */
template<typename T>
void deviceVectorDotProduct(Context &context) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    std::vector<T> dataY{-1, 4, -6, 9, 10};
    DeviceVector<T> y(context, dataY);
    T dotProduct = x * y;
    EXPECT_EQ(75, dotProduct);
}

TEST_F(DeviceTest, deviceVectorDotProduct) {
    deviceVectorDotProduct<float>(m_context);
    deviceVectorDotProduct<double>(m_context);
}


/* ---------------------------------------
 * Norm-2
 * --------------------------------------- */
template<typename T>
void deviceVectorEuclideanNorm(Context &context, T epsilon) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(context, dataX);
    T nrmX = x.norm2();
    EXPECT_NEAR(7.416198487095663, nrmX, epsilon);
}

TEST_F(DeviceTest, deviceVectorEuclideanNorm) {
    deviceVectorEuclideanNorm<float>(m_context, 1e-4);
    deviceVectorEuclideanNorm<double>(m_context, 1e-12);
}

/* ---------------------------------------
 * Norm-1
 * --------------------------------------- */
template<typename T>
void deviceVectorNorm1(Context &context) {
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> y(context, dataY);
    T nrmX = y.norm1();
    EXPECT_EQ(30, nrmX);
}

TEST_F(DeviceTest, deviceVectorNorm1) {
    deviceVectorNorm1<float>(m_context);
    deviceVectorNorm1<double>(m_context);
}

/* ---------------------------------------
 * Sum of vectors (operator +)
 * --------------------------------------- */
template<typename T>
void deviceVectorSum(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    auto sum = x + y;
    EXPECT_EQ(9, sum(0));
    EXPECT_EQ(60, sum(4));
}

TEST_F(DeviceTest, deviceVectorSum) {
    deviceVectorSum<float>(m_context);
    deviceVectorSum<double>(m_context);
}


/* ---------------------------------------
 * Scalar product (scaling)
 * --------------------------------------- */
template<typename T>
void deviceVectorScaling(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., -50.};
    DeviceVector<T> x(context, dataX);
    T alpha = 2.;
    auto scaledX = alpha * x;
    EXPECT_EQ(20, scaledX(0));
    EXPECT_EQ(-100, scaledX(4));
}

TEST_F(DeviceTest, deviceVectorScaling) {
    deviceVectorScaling<float>(m_context);
    deviceVectorScaling<double>(m_context);
}

/* ---------------------------------------
 * Scalar product (scaling in place)
 * --------------------------------------- */
template<typename T>
void deviceVectorScalingInPlace(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., -50.};
    DeviceVector<T> x(context, dataX);
    T alpha = 2.;
    x *= alpha;
    EXPECT_EQ(20, x(0));
    EXPECT_EQ(-100, x(4));
}

TEST_F(DeviceTest, deviceVectorScalingInPlace) {
    deviceVectorScalingInPlace<float>(m_context);
    deviceVectorScalingInPlace<double>(m_context);
}

/* ---------------------------------------
 * Difference of vectors (operator -)
 * --------------------------------------- */
template<typename T>
void deviceVectorDiff(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    auto sum = x - y;
    EXPECT_EQ(11, sum(0));
    EXPECT_EQ(40, sum(4));
}

TEST_F(DeviceTest, deviceVectorDiff) {
    deviceVectorDiff<float>(m_context);
    deviceVectorDiff<double>(m_context);
}

/* ---------------------------------------
 * Device-to-device copy with slicing
 * --------------------------------------- */
template<typename T>
void deviceVectorDeviceToDeviceCopy(Context &context) {
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

TEST_F(DeviceTest, deviceVectorDeviceToDeviceCopy) {
    deviceVectorDeviceToDeviceCopy<float>(m_context);
    deviceVectorDeviceToDeviceCopy<double>(m_context);
}

/* ---------------------------------------
 * Operator +=
 * --------------------------------------- */
template<typename T>
void deviceVectorOpPlusEq(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    x += y;
    EXPECT_EQ(9, x(0));
    EXPECT_EQ(49, x(3));
    EXPECT_EQ(60, x(4));
}

TEST_F(DeviceTest, deviceVectorOpPlusEq) {
    deviceVectorOpPlusEq<float>(m_context);
    deviceVectorOpPlusEq<double>(m_context);
}

/* ---------------------------------------
 * Operator -=
 * --------------------------------------- */
template<typename T>
void deviceVectorOpMinusEq(Context &context) {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(context, dataX);
    DeviceVector<T> y(context, dataY);
    x -= y;
    EXPECT_EQ(11, x(0));
    EXPECT_EQ(16, x(1));
    EXPECT_EQ(36, x(2));
}

TEST_F(DeviceTest, deviceVectorOpMinusEq) {
    deviceVectorOpMinusEq<float>(m_context);
    deviceVectorOpMinusEq<double>(m_context);
}


/* ---------------------------------------
 * Upload data
 * --------------------------------------- */
template<typename T>
void deviceVectorUploadData(Context &context) {
    std::vector<T> data = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(data.size());
    DeviceVector<T> vec(context, data.size());
    vec.upload(data);
    vec.download(result);
    EXPECT_EQ(data, result);
}

TEST_F(DeviceTest, deviceVectorUploadData) {
    deviceVectorUploadData<float>(m_context);
    deviceVectorUploadData<double>(m_context);
}


/* =======================================
 * DeviceMatrix<T>
 * ======================================= */


/* ---------------------------------------
 * Basic constructor and matrix dimensions
 * .numRows and .numCols
 * --------------------------------------- */

template<typename T>
void deviceMatrixDimensions(Context &context) {
    DeviceMatrix<T> fourByThree(context, 4, 3);
    EXPECT_EQ(4, fourByThree.numRows());
    EXPECT_EQ(3, fourByThree.numCols());
}

TEST_F(DeviceTest, deviceMatrixDimensions) {
    deviceMatrixDimensions<float>(m_context);
    deviceMatrixDimensions<double>(m_context);
}

/* ---------------------------------------
 * Copy constructor
 * .numRows and .numCols
 * --------------------------------------- */

template<typename T>
void deviceMatrixCopyConstructor(Context &context) {
    std::vector<T> data{1, 2,
                        3, 4,
                        5, 6};
    DeviceMatrix<T> X(context, 3, data, rowMajor);
    DeviceMatrix<T> XCopy(X);
    EXPECT_EQ(3, XCopy.numRows());
    EXPECT_EQ(2, XCopy.numCols());
    X *= 0;
    EXPECT_EQ(2, XCopy(0, 1));
}

TEST_F(DeviceTest, deviceMatrixCopyConstructor) {
    deviceMatrixCopyConstructor<float>(m_context);
    deviceMatrixCopyConstructor<double>(m_context);
}

/* ---------------------------------------
 * Column range (shallow copy)
 * --------------------------------------- */

template<typename T>
void deviceMatrixColumnRangeShallow(Context &context) {
    std::vector<T> data{1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10};
    DeviceMatrix<T> X(context, 2, data, rowMajor);
    DeviceMatrix<T> XColSlice(X, 2, 3);
    EXPECT_EQ(2, XColSlice.numRows());
    EXPECT_EQ(2, XColSlice.numCols());
    XColSlice *= 2;
    EXPECT_EQ(2, X(0, 1));
    EXPECT_EQ(16, X(1, 2));
}

TEST_F(DeviceTest, deviceMatrixColumnRangeShallow) {
    deviceMatrixColumnRangeShallow<float>(m_context);
    deviceMatrixColumnRangeShallow<double>(m_context);
}

/* ---------------------------------------
 * Constructor from DeviceVector
 * This does not allocate new memory
 * --------------------------------------- */

template<typename T>
void deviceMatrixConstructorFromVector(Context &context) {
    std::vector<T> od{1, 1, 1, 3, 4};
    DeviceVector<T> o(context, od);
    DeviceMatrix<T> p(context, o);
    EXPECT_EQ(1, p.numCols());
    EXPECT_EQ(5, p.numRows());
    EXPECT_EQ(1, p(0, 0));
    EXPECT_EQ(3, p(3, 0));
    EXPECT_EQ(4, p(4, 0));
}

TEST_F(DeviceTest, deviceMatrixConstructorFromVector) {
    deviceMatrixConstructorFromVector<float>(m_context);
    deviceMatrixConstructorFromVector<double>(m_context);
    deviceMatrixConstructorFromVector<int>(m_context);
}

/* ---------------------------------------
 * Matrix as vector (shallow copy)
 * --------------------------------------- */

template<typename T>
void deviceMatrixAsVector(Context &context) {
    std::vector<T> data{1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                        10, 11, 12};
    DeviceMatrix<T> X(context, 4, data, rowMajor);
    auto x = X.asVector();
    EXPECT_EQ(12, x.capacity());
    EXPECT_EQ(1, x(0));
    EXPECT_EQ(4, x(1));
    EXPECT_EQ(12, x(11));
}

TEST_F(DeviceTest, deviceMatrixAsVector) {
    deviceMatrixAsVector<float>(m_context);
    deviceMatrixAsVector<double>(m_context);
}

/* ---------------------------------------
 * Scalar multiplication (*=)
 * --------------------------------------- */

template<typename T>
void deviceMatrixScalarTimeEq(Context &context) {
    std::vector<T> data{1, 2, 3,
                        4, 5, 6};
    DeviceMatrix<T> X(context, 2, data, rowMajor);
    X *= 2.;
    EXPECT_EQ(4, X(0, 1));
    EXPECT_EQ(12, X(1, 2));
}

TEST_F(DeviceTest, deviceMatrixScalarTimeEq) {
    deviceMatrixScalarTimeEq<float>(m_context);
    deviceMatrixScalarTimeEq<double>(m_context);
}

/* ---------------------------------------
 * DeviceVector/Matrix data transfer
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
 * operator* and .addAB
 * --------------------------------------- */

template<typename T>
void matrixVectorOpAst(Context &context) {
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

TEST_F(DeviceTest, matrixVectorOpAst) {
    matrixVectorOpAst<float>(m_context);
    matrixVectorOpAst<double>(m_context);
}


/* ---------------------------------------
 * Matrix: addAB
 * C += AB
 * --------------------------------------- */

template<typename T>
void deviceMatrixAddAB(Context &context) {
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

TEST_F(DeviceTest, deviceMatrixAddAB) {
    deviceMatrixAddAB<float>(m_context);
    deviceMatrixAddAB<double>(m_context);
}


/* ---------------------------------------
 * Matrix: addAB complete
 * C = beta C + alpha AB
 * --------------------------------------- */

template<typename T>
void deviceMatrixAddABComplete(Context &context) {
    size_t nRowsC = 4;
    size_t nColsC = 3;
    size_t k = 2;
    std::vector<T> matC = {1, 2, 3,
                           4, 5, 6,
                           7, 8, 9,
                           10, 11, 12};
    std::vector<T> matA = {1, 2,
                           3, 4,
                           5, 6,
                           7, 8};
    std::vector<T> matB = {1, 2, 3,
                           4, 5, 6};
    std::vector<T> result(nRowsC * nColsC);
    DeviceMatrix<T> d_matC(context, nRowsC, matC, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matA(context, nRowsC, matA, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matB(context, k, matB, MatrixStorageMode::rowMajor);
    d_matC.addAB(d_matA, d_matB, -4., 2.);
    d_matC.asVector().download(result);
    std::vector<T> expected = {-34, -68, -102, -136, -44, -94, -144, -194, -54, -120, -186, -252};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixAddABComplete) {
    deviceMatrixAddABComplete<float>(m_context);
    deviceMatrixAddABComplete<double>(m_context);
}

/* ---------------------------------------
 * Matvec (*=)
 * --------------------------------------- */

template<typename T>
void deviceMatrixMatvec(Context &context) {
    std::vector<T> aData = {1, 2, 3,
                            4, 5, 6};
    std::vector<T> xData = {10, -20, 30};
    std::vector<T> result(2);
    DeviceMatrix<T> A(context, 2, aData, rowMajor);
    DeviceVector<T> x(context, xData);
    auto res = A * x;
    EXPECT_EQ(2, res.capacity());
    res.download(result);
    std::vector<T> expected = {60, 120};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixMatvec) {
    deviceMatrixMatvec<float>(m_context);
    deviceMatrixMatvec<double>(m_context);
}

/* ---------------------------------------
 * Matrix addition
 * --------------------------------------- */

template<typename T>
void deviceMatrixAddition(Context &context) {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6};
    std::vector<T> bData = {10, 20, 30, 40, 50, 60};
    std::vector<T> result(6);
    DeviceMatrix<T> A(context, 2, aData, rowMajor);
    DeviceMatrix<T> B(context, 2, bData, rowMajor);
    DeviceMatrix<T> res = A + B;
    res.asVector().download(result);
    std::vector<T> expected = {11, 44, 22, 55, 33, 66};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixAddition) {
    deviceMatrixAddition<float>(m_context);
    deviceMatrixAddition<double>(m_context);
}


/* ---------------------------------------
 * Matrix subtraction
 * --------------------------------------- */

template<typename T>
void deviceMatrixSubtraction(Context &context) {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6};
    std::vector<T> bData = {10, 20, 30, 40, 50, 60};
    std::vector<T> result(6);
    DeviceMatrix<T> A(context, 2, aData, rowMajor);
    DeviceMatrix<T> B(context, 2, bData, rowMajor);
    DeviceMatrix<T> res = B - A;
    res.asVector().download(result);
    std::vector<T> expected = {9, 36, 18, 45, 27, 54};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixSubtraction) {
    deviceMatrixSubtraction<float>(m_context);
    deviceMatrixSubtraction<double>(m_context);
}

/* ---------------------------------------
 * Indexing Vectors
 * --------------------------------------- */

template<typename T>
void indexingVectors(Context &context) {
    std::vector<T> xData{1.0, 2.0, 3.0, 6.0, 7.0, 8.0};
    DeviceVector<T> x(context, xData);
    EXPECT_EQ(1., x(0));
    EXPECT_EQ(7., x(4));
}

TEST_F(DeviceTest, indexingVectors) {
    indexingVectors<float>(m_context);
    indexingVectors<double>(m_context);
}

/* ---------------------------------------
 * Indexing Matrices
 * --------------------------------------- */

template<typename T>
void indexingMatrices(Context &context) {
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0};
    DeviceMatrix<T> B(context, 2, bData, MatrixStorageMode::rowMajor);
    EXPECT_EQ(2., B(0, 1));
    EXPECT_EQ(7., B(1, 1));
}

TEST_F(DeviceTest, indexingMatrices) {
    indexingMatrices<float>(m_context);
    indexingMatrices<double>(m_context);
}

/* ---------------------------------------
 * Get Matrix Rows
 * .getRows
 * --------------------------------------- */

template<typename T>
void getMatrixRows(Context &context) {
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

TEST_F(DeviceTest, getMatrixRows) {
    getMatrixRows<float>(m_context);
    getMatrixRows<double>(m_context);
}


/* =======================================
 * DeviceTensor<T>
 * ======================================= */

/* ---------------------------------------
 * Constructors and .pushBack
 * --------------------------------------- */

template<typename T>
void deviceTensorConstructPush(Context &context) {
    size_t nRows = 2, nCols = 3, nMats = 3;
    std::vector<T> aData = {1, 2, 3,
                            4, 5, 6};
    std::vector<T> bData = {7, 8, 9,
                            10, 11, 12};
    DeviceMatrix<T> matrixA(context, nRows, aData, rowMajor);
    DeviceMatrix<T> matrixB(context, nRows, bData, rowMajor);
    T *rawA = matrixA.raw();
    T *rawB = matrixB.raw();
    DeviceTensor<T> myTensor(context, nRows, nCols, nMats);
    myTensor.pushBack(matrixA);
    myTensor.pushBack(matrixA);
    myTensor.pushBack(matrixB);
    DeviceVector<T *> pointersToMatrices = myTensor.devicePointersToMatrices();
    EXPECT_EQ(rawA, pointersToMatrices(0));
    EXPECT_EQ(rawA, pointersToMatrices(1));
    EXPECT_EQ(rawB, pointersToMatrices(2));
}

TEST_F(DeviceTest, deviceTensorConstructPush) {
    deviceTensorConstructPush<float>(m_context);
    deviceTensorConstructPush<double>(m_context);
    deviceTensorConstructPush<int>(m_context);
}

/* ---------------------------------------
 * Batched least squares
 * .leastSquares
 * --------------------------------------- */

template<typename T>
void deviceTensorLeastSquares(Context &context) {
    size_t rows = 3;
    size_t cols = 2;
    std::vector<T> A1 = {1, 0, 0, 3, 2, 6};
    DeviceMatrix<T> d_A1(context, rows, A1, MatrixStorageMode::rowMajor);
    std::vector<T> b1 = {1, 2, 3};
    DeviceVector<T> d_b1(context, b1);
    std::vector<T> A2 = {1, 3, 3, 2, 2, 1};
    DeviceMatrix<T> d_A2(context, rows, A2, MatrixStorageMode::rowMajor);
    std::vector<T> b2 = {1, 2, 3};
    DeviceVector<T> d_b2(context, b2);
    DeviceTensor<T> d_As(context, rows, cols, 2);
    DeviceTensor<T> d_bs(context, rows, 1, 2);
    d_As.pushBack(d_A1);
    d_As.pushBack(d_A2);
    DeviceMatrix<T> db1_mat(context, d_b1);
    d_bs.pushBack(db1_mat);
    DeviceMatrix<T> d_B2(context, d_b2);
    d_bs.pushBack(d_B2);
    d_As.leastSquares(d_bs);
    std::vector<T> hostData(cols);
    size_t from = 0;
    size_t to = cols - 1;
    DeviceVector<T> d_x1(d_b1, from, to);
    d_x1.download(hostData);
    std::vector<T> expectedResult1{0.33333333333333, 0.444444444444444};
    for (size_t i = 0; i < cols; i++) {
        EXPECT_NEAR(hostData[i], expectedResult1[i], PRECISION);
    }
    DeviceVector<T> d_x2(d_b2, from, to);
    d_x2.download(hostData);
    std::vector<T> expectedResult2{0.96, -0.04};
    for (size_t i = 0; i < cols; i++) {
        EXPECT_NEAR(hostData[i], expectedResult2[i], PRECISION);
    }
}

TEST_F(DeviceTest, deviceTensorLeastSquares) {
    deviceTensorLeastSquares<float>(m_context);
    deviceTensorLeastSquares<double>(m_context);
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
void singularValuesComputation(Context &context, float epsilon) {
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

TEST_F(DeviceTest, singularValuesComputation) {
    singularValuesComputation<float>(m_context, 1e-4);
    singularValuesComputation<double>(m_context, 1e-7);
}


/* =======================================
 * Cholesky factorisation and solution
 * of linear systems
 * ======================================= */

/* ---------------------------------------
 * Factorisation of matrix
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyFactoriserFactorise(Context &context, float epsilon) {
    std::vector<T> aData{10.0, 2.0, 3.0,
                         3.0, 20.0, -1.0,
                         3.0, -1.0, 30.0};
    DeviceMatrix<T> A(context, 3, aData, rowMajor);
    CholeskyFactoriser<T> chol(context, A);
    EXPECT_EQ(0, chol.factorise());
    EXPECT_NEAR(3.162277660168380, A(0, 0), epsilon);
}

TEST_F(DeviceTest, choleskyFactoriserFactorise) {
    choleskyFactoriserFactorise<float>(m_context, 1e-4);
    choleskyFactoriserFactorise<double>(m_context, 1e-7);
}

/* ---------------------------------------
 * Solve linear system via Cholesky
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyFactoriserSolve(Context &context, float epsilon) {
    std::vector<T> aData = {10., 2., 3.,
                            2., 20., -1.,
                            3., -1., 30.};
    DeviceMatrix<T> A(context, 3, aData, rowMajor);
    DeviceMatrix<T> L(A); // L = A
    CholeskyFactoriser<T> chol(context, L);
    EXPECT_EQ(0, chol.factorise());

    std::vector<T> bData = {-1., -3., 5.};
    DeviceVector<T> b(context, bData);
    DeviceVector<T> sol(b); // b = x
    chol.solve(sol);

    auto error = A * sol;
    error -= b;
    EXPECT_TRUE(error.norm2() < epsilon);
}

TEST_F(DeviceTest, choleskyFactoriserSolve) {
    choleskyFactoriserSolve<float>(m_context, 1e-6);
    choleskyFactoriserSolve<double>(m_context, 1e-12);
}