#include <gtest/gtest.h>
#include "../include/gputils.cuh"

#define PRECISION 1e-6


class DeviceTest : public testing::Test {
protected:
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
void deviceVectorCapacity() {
    DeviceVector<T> four(4);
    EXPECT_EQ(4, four.capacity());
    DeviceVector<T> five(0);
    five.allocateOnDevice(5);
    EXPECT_EQ(5, five.capacity());
}

TEST_F(DeviceTest, deviceVectorCapacity) {
    deviceVectorCapacity<float>();
    deviceVectorCapacity<double>();
}

/* ---------------------------------------
 * Basic constructor (extreme cases)
 * --------------------------------------- */
template<typename T>
void deviceVectorBasicConstructor() {
    DeviceVector<T> empty(0);
    EXPECT_EQ(0, empty.capacity());
    DeviceVector<T> big(24000);
    EXPECT_EQ(24000, big.capacity());
}

TEST_F(DeviceTest, deviceVectorBasicConstructor) {
    deviceVectorBasicConstructor<float>();
    deviceVectorBasicConstructor<double>();
    deviceVectorBasicConstructor<int>();
}


/* ---------------------------------------
 * Slice constructor
 * --------------------------------------- */

template<typename T>
void deviceVectorSliceConstructor() {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(dataX);
    DeviceVector<T> xSlice(x, 1, 3);
    EXPECT_EQ(3, xSlice.capacity());
    EXPECT_EQ(2, xSlice(0));
    EXPECT_EQ(3, xSlice(1));
}

TEST_F(DeviceTest, deviceVectorSliceConstructor) {
    deviceVectorSliceConstructor<float>();
    deviceVectorSliceConstructor<double>();
    deviceVectorSliceConstructor<int>();
}

/* ---------------------------------------
 * Copy constructor
 * --------------------------------------- */

template<typename T>
void deviceVectorCopyConstructor() {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(dataX);
    DeviceVector<T> xCopy(x);
    EXPECT_EQ(5, xCopy.capacity());
    EXPECT_EQ(1, xCopy(0));
    EXPECT_EQ(5, xCopy(4));
}

TEST_F(DeviceTest, deviceVectorCopyConstructor) {
    deviceVectorCopyConstructor<float>();
    deviceVectorCopyConstructor<double>();
    deviceVectorCopyConstructor<int>();
}

/* ---------------------------------------
 * Operator * (dot product)
 * --------------------------------------- */
template<typename T>
void deviceVectorDotProduct() {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(dataX);
    std::vector<T> dataY{-1, 4, -6, 9, 10};
    DeviceVector<T> y(dataY);
    T dotProduct = x * y;
    EXPECT_EQ(75, dotProduct);
}

TEST_F(DeviceTest, deviceVectorDotProduct) {
    deviceVectorDotProduct<float>();
    deviceVectorDotProduct<double>();
}


/* ---------------------------------------
 * Norm-2
 * --------------------------------------- */
template<typename T>
void deviceVectorEuclideanNorm(T epsilon) {
    std::vector<T> dataX{1, 2, 3, 4, 5};
    DeviceVector<T> x(dataX);
    T nrmX = x.norm2();
    EXPECT_NEAR(7.416198487095663, nrmX, epsilon);
}

TEST_F(DeviceTest, deviceVectorEuclideanNorm) {
    deviceVectorEuclideanNorm<float>(1e-4);
    deviceVectorEuclideanNorm<double>(1e-12);
}

/* ---------------------------------------
 * Norm-1
 * --------------------------------------- */
template<typename T>
void deviceVectorNorm1() {
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> y(dataY);
    T nrmX = y.norm1();
    EXPECT_EQ(30, nrmX);
}

TEST_F(DeviceTest, deviceVectorNorm1) {
    deviceVectorNorm1<float>();
    deviceVectorNorm1<double>();
}

/* ---------------------------------------
 * Sum of vectors (operator +)
 * --------------------------------------- */
template<typename T>
void deviceVectorSum() {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(dataX);
    DeviceVector<T> y(dataY);
    auto sum = x + y;
    EXPECT_EQ(9, sum(0));
    EXPECT_EQ(60, sum(4));
}

TEST_F(DeviceTest, deviceVectorSum) {
    deviceVectorSum<float>();
    deviceVectorSum<double>();
}


/* ---------------------------------------
 * Scalar product (scaling)
 * --------------------------------------- */
template<typename T>
void deviceVectorScaling() {
    std::vector<T> dataX{10., 20., 30., 40., -50.};
    DeviceVector<T> x(dataX);
    T alpha = 2.;
    auto scaledX = alpha * x;
    EXPECT_EQ(20, scaledX(0));
    EXPECT_EQ(-100, scaledX(4));
}

TEST_F(DeviceTest, deviceVectorScaling) {
    deviceVectorScaling<float>();
    deviceVectorScaling<double>();
}

/* ---------------------------------------
 * Scalar product (scaling in place)
 * --------------------------------------- */
template<typename T>
void deviceVectorScalingInPlace() {
    std::vector<T> dataX{10., 20., 30., 40., -50.};
    DeviceVector<T> x(dataX);
    T alpha = 2.;
    x *= alpha;
    EXPECT_EQ(20, x(0));
    EXPECT_EQ(-100, x(4));
}

TEST_F(DeviceTest, deviceVectorScalingInPlace) {
    deviceVectorScalingInPlace<float>();
    deviceVectorScalingInPlace<double>();
}

/* ---------------------------------------
 * Difference of vectors (operator -)
 * --------------------------------------- */
template<typename T>
void deviceVectorDiff() {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(dataX);
    DeviceVector<T> y(dataY);
    auto sum = x - y;
    EXPECT_EQ(11, sum(0));
    EXPECT_EQ(40, sum(4));
}

TEST_F(DeviceTest, deviceVectorDiff) {
    deviceVectorDiff<float>();
    deviceVectorDiff<double>();
}

/* ---------------------------------------
 * Device-to-device copy with slicing
 * --------------------------------------- */
template<typename T>
void deviceVectorDeviceToDeviceCopy() {
    std::vector<T> dataX{-1., 2., 3., 4., 6.};
    DeviceVector<T> x(dataX);
    DeviceVector<T> xSlice(x, 1, 3);
    std::vector<T> dataY{5, 5, 5};
    DeviceVector<T> y(dataY);
    y.deviceCopyTo(xSlice);
    std::vector<T> xExpected{-1., 5., 5., 5., 6.};
    std::vector<T> h_x(5);
    x.download(h_x);
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(xExpected[i], h_x[i]);
    }
}

TEST_F(DeviceTest, deviceVectorDeviceToDeviceCopy) {
    deviceVectorDeviceToDeviceCopy<float>();
    deviceVectorDeviceToDeviceCopy<double>();
}

/* ---------------------------------------
 * Operator +=
 * --------------------------------------- */
template<typename T>
void deviceVectorOpPlusEq() {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(dataX);
    DeviceVector<T> y(dataY);
    x += y;
    EXPECT_EQ(9, x(0));
    EXPECT_EQ(49, x(3));
    EXPECT_EQ(60, x(4));
}

TEST_F(DeviceTest, deviceVectorOpPlusEq) {
    deviceVectorOpPlusEq<float>();
    deviceVectorOpPlusEq<double>();
}

/* ---------------------------------------
 * Operator -=
 * --------------------------------------- */
template<typename T>
void deviceVectorOpMinusEq() {
    std::vector<T> dataX{10., 20., 30., 40., 50.};
    std::vector<T> dataY{-1., 4., -6., 9., 10.};
    DeviceVector<T> x(dataX);
    DeviceVector<T> y(dataY);
    x -= y;
    EXPECT_EQ(11, x(0));
    EXPECT_EQ(16, x(1));
    EXPECT_EQ(36, x(2));
}

TEST_F(DeviceTest, deviceVectorOpMinusEq) {
    deviceVectorOpMinusEq<float>();
    deviceVectorOpMinusEq<double>();
}


/* ---------------------------------------
 * Upload data
 * --------------------------------------- */
template<typename T>
void deviceVectorUploadData() {
    std::vector<T> data = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(data.size());
    DeviceVector<T> vec(data.size());
    vec.upload(data);
    vec.download(result);
    EXPECT_EQ(data, result);
}

TEST_F(DeviceTest, deviceVectorUploadData) {
    deviceVectorUploadData<float>();
    deviceVectorUploadData<double>();
}


/* =======================================
 * DeviceMatrix<T>
 * ======================================= */


/* ---------------------------------------
 * Basic constructor and matrix dimensions
 * .numRows and .numCols
 * --------------------------------------- */

template<typename T>
void deviceMatrixDimensions() {
    DeviceMatrix<T> fourByThree(4, 3);
    EXPECT_EQ(4, fourByThree.numRows());
    EXPECT_EQ(3, fourByThree.numCols());
}

TEST_F(DeviceTest, deviceMatrixDimensions) {
    deviceMatrixDimensions<float>();
    deviceMatrixDimensions<double>();
}

/* ---------------------------------------
 * Copy constructor
 * .numRows and .numCols
 * --------------------------------------- */

template<typename T>
void deviceMatrixCopyConstructor() {
    std::vector<T> data{1, 2,
                        3, 4,
                        5, 6};
    DeviceMatrix<T> X(3, data, rowMajor);
    DeviceMatrix<T> XCopy(X);
    EXPECT_EQ(3, XCopy.numRows());
    EXPECT_EQ(2, XCopy.numCols());
    X *= 0;
    EXPECT_EQ(2, XCopy(0, 1));
}

TEST_F(DeviceTest, deviceMatrixCopyConstructor) {
    deviceMatrixCopyConstructor<float>();
    deviceMatrixCopyConstructor<double>();
}

/* ---------------------------------------
 * Column range (shallow copy)
 * --------------------------------------- */

template<typename T>
void deviceMatrixColumnRangeShallow() {
    std::vector<T> data{1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10};
    DeviceMatrix<T> X(2, data, rowMajor);
    DeviceMatrix<T> XColSlice(X, 2, 3);
    EXPECT_EQ(2, XColSlice.numRows());
    EXPECT_EQ(2, XColSlice.numCols());
    XColSlice *= 2;
    EXPECT_EQ(2, X(0, 1));
    EXPECT_EQ(16, X(1, 2));
}

TEST_F(DeviceTest, deviceMatrixColumnRangeShallow) {
    deviceMatrixColumnRangeShallow<float>();
    deviceMatrixColumnRangeShallow<double>();
}

/* ---------------------------------------
 * Constructor from DeviceVector
 * This does not allocate new memory
 * --------------------------------------- */

template<typename T>
void deviceMatrixConstructorFromVector() {
    std::vector<T> od{1, 1, 1, 3, 4};
    DeviceVector<T> o(od);
    DeviceMatrix<T> p(o);
    EXPECT_EQ(1, p.numCols());
    EXPECT_EQ(5, p.numRows());
    EXPECT_EQ(1, p(0, 0));
    EXPECT_EQ(3, p(3, 0));
    EXPECT_EQ(4, p(4, 0));
}

TEST_F(DeviceTest, deviceMatrixConstructorFromVector) {
    deviceMatrixConstructorFromVector<float>();
    deviceMatrixConstructorFromVector<double>();
    deviceMatrixConstructorFromVector<int>();
}

/* ---------------------------------------
 * Matrix as vector (shallow copy)
 * --------------------------------------- */

template<typename T>
void deviceMatrixAsVector() {
    std::vector<T> data{1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                        10, 11, 12};
    DeviceMatrix<T> X(4, data, rowMajor);
    auto x = X.asVector();
    EXPECT_EQ(12, x.capacity());
    EXPECT_EQ(1, x(0));
    EXPECT_EQ(4, x(1));
    EXPECT_EQ(12, x(11));
}

TEST_F(DeviceTest, deviceMatrixAsVector) {
    deviceMatrixAsVector<float>();
    deviceMatrixAsVector<double>();
}

/* ---------------------------------------
 * Scalar multiplication (*=)
 * --------------------------------------- */

template<typename T>
void deviceMatrixScalarTimeEq() {
    std::vector<T> data{1, 2, 3,
                        4, 5, 6};
    DeviceMatrix<T> X(2, data, rowMajor);
    X *= 2.;
    EXPECT_EQ(4, X(0, 1));
    EXPECT_EQ(12, X(1, 2));
}

TEST_F(DeviceTest, deviceMatrixScalarTimeEq) {
    deviceMatrixScalarTimeEq<float>();
    deviceMatrixScalarTimeEq<double>();
}

/* ---------------------------------------
 * DeviceVector/Matrix data transfer
 * .upload and .download
 * --------------------------------------- */

template<typename T>
void transfer() {
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
    DeviceVector<T> vec(n);
    vec.upload(data);
    vec.download(resultVec);
    EXPECT_EQ(data, resultVec);
    DeviceMatrix<T> mat(rows, cols);
    mat.upload(dataCM, rows, MatrixStorageMode::columnMajor);
    mat.asVector().download(resultCM);
    EXPECT_EQ(resultCM, dataCM);
    mat.upload(data, rows, MatrixStorageMode::rowMajor);
    mat.asVector().download(resultRM);
    EXPECT_EQ(resultRM, dataCM);
}

TEST_F(DeviceTest, transfer) {
    transfer<float>();
    transfer<double>();
}

/* ---------------------------------------
 * Transposition
 * --------------------------------------- */

template<typename T>
void deviceMatrixTranspose() {
    std::vector<T> aData = {1, 2, 3,
                            4, 5, 6};
    DeviceMatrix<T> A(2, aData, rowMajor);
    DeviceMatrix<T> At = A.tr();

    EXPECT_EQ(3, At.numRows());
    EXPECT_EQ(2, At.numCols());
    EXPECT_EQ(2, At(1, 0));
}

TEST_F(DeviceTest, deviceMatrixTranspose) {
    deviceMatrixTranspose<float>();
    deviceMatrixTranspose<double>();
}

/* ---------------------------------------
 * Matrix-vector multiplication
 * operator* and .addAB
 * --------------------------------------- */

template<typename T>
void matrixVectorOpAst() {
    size_t rows = 4;
    std::vector<T> mat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<T> vec = {1, 2, 3};
    std::vector<T> result(rows);
    DeviceMatrix<T> d_mat(rows, mat, MatrixStorageMode::rowMajor);
    DeviceVector<T> d_vec(vec);
    auto d_result = d_mat * d_vec;
    d_result.download(result);
    std::vector<T> expected = {14, 32, 50, 68};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, matrixVectorOpAst) {
    matrixVectorOpAst<float>();
    matrixVectorOpAst<double>();
}


/* ---------------------------------------
 * Matrix: addAB
 * C += AB
 * --------------------------------------- */

template<typename T>
void deviceMatrixAddAB() {
    size_t nRowsC = 4;
    size_t nColsC = 3;
    size_t k = 2;
    std::vector<T> matC = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<T> matA = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<T> matB = {1, 2, 3, 4, 5, 6};
    std::vector<T> result(nRowsC * nColsC);
    DeviceMatrix<T> d_matC(nRowsC, matC, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matA(nRowsC, matA, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matB(k, matB, MatrixStorageMode::rowMajor);
    d_matC.addAB(d_matA, d_matB);
    d_matC.asVector().download(result);
    std::vector<T> expected = {10, 23, 36, 49, 14, 31, 48, 65, 18, 39, 60, 81};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixAddAB) {
    deviceMatrixAddAB<float>();
    deviceMatrixAddAB<double>();
}


/* ---------------------------------------
 * Matrix: addAB complete
 * C = beta C + alpha AB
 * --------------------------------------- */

template<typename T>
void deviceMatrixAddABComplete() {
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
    DeviceMatrix<T> d_matC(nRowsC, matC, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matA(nRowsC, matA, MatrixStorageMode::rowMajor);
    DeviceMatrix<T> d_matB(k, matB, MatrixStorageMode::rowMajor);
    d_matC.addAB(d_matA, d_matB, -4., 2.);
    d_matC.asVector().download(result);
    std::vector<T> expected = {-34, -68, -102, -136, -44, -94, -144, -194, -54, -120, -186, -252};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixAddABComplete) {
    deviceMatrixAddABComplete<float>();
    deviceMatrixAddABComplete<double>();
}

/* ---------------------------------------
 * Matvec (*=)
 * --------------------------------------- */

template<typename T>
void deviceMatrixMatvec() {
    std::vector<T> aData = {1, 2, 3,
                            4, 5, 6};
    std::vector<T> xData = {10, -20, 30};
    std::vector<T> result(2);
    DeviceMatrix<T> A(2, aData, rowMajor);
    DeviceVector<T> x(xData);
    auto res = A * x;
    EXPECT_EQ(2, res.capacity());
    res.download(result);
    std::vector<T> expected = {60, 120};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixMatvec) {
    deviceMatrixMatvec<float>();
    deviceMatrixMatvec<double>();
}

/* ---------------------------------------
 * Matrix addition
 * --------------------------------------- */

template<typename T>
void deviceMatrixAddition() {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6};
    std::vector<T> bData = {10, 20, 30, 40, 50, 60};
    std::vector<T> result(6);
    DeviceMatrix<T> A(2, aData, rowMajor);
    DeviceMatrix<T> B(2, bData, rowMajor);
    DeviceMatrix<T> res = A + B;
    res.asVector().download(result);
    std::vector<T> expected = {11, 44, 22, 55, 33, 66};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixAddition) {
    deviceMatrixAddition<float>();
    deviceMatrixAddition<double>();
}


/* ---------------------------------------
 * Matrix subtraction
 * --------------------------------------- */

template<typename T>
void deviceMatrixSubtraction() {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6};
    std::vector<T> bData = {10, 20, 30, 40, 50, 60};
    std::vector<T> result(6);
    DeviceMatrix<T> A(2, aData, rowMajor);
    DeviceMatrix<T> B(2, bData, rowMajor);
    DeviceMatrix<T> res = B - A;
    res.asVector().download(result);
    std::vector<T> expected = {9, 36, 18, 45, 27, 54};
    EXPECT_EQ(expected, result);
}

TEST_F(DeviceTest, deviceMatrixSubtraction) {
    deviceMatrixSubtraction<float>();
    deviceMatrixSubtraction<double>();
}

/* ---------------------------------------
 * Indexing Vectors
 * --------------------------------------- */

template<typename T>
void indexingVectors() {
    std::vector<T> xData{1.0, 2.0, 3.0, 6.0, 7.0, 8.0};
    DeviceVector<T> x(xData);
    EXPECT_EQ(1., x(0));
    EXPECT_EQ(7., x(4));
}

TEST_F(DeviceTest, indexingVectors) {
    indexingVectors<float>();
    indexingVectors<double>();
}

/* ---------------------------------------
 * Indexing Matrices
 * --------------------------------------- */

template<typename T>
void indexingMatrices() {
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0};
    DeviceMatrix<T> B(2, bData, MatrixStorageMode::rowMajor);
    EXPECT_EQ(2., B(0, 1));
    EXPECT_EQ(7., B(1, 1));
}

TEST_F(DeviceTest, indexingMatrices) {
    indexingMatrices<float>();
    indexingMatrices<double>();
}

/* ---------------------------------------
 * Get Matrix Rows
 * .getRows
 * --------------------------------------- */

template<typename T>
void getMatrixRows() {
    size_t k = 6;
    std::vector<T> bData{1.0, 2.0, 3.0,
                         6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0,
                         12.0, 13.0, 14.0,
                         15.0, 16.0, 17.0,
                         18.0, 19.0, 20.0};
    DeviceMatrix<T> B(k, bData, MatrixStorageMode::rowMajor);
    auto copiedRows = B.getRows(1, 4);
    EXPECT_EQ(6., copiedRows(0, 0));
    EXPECT_EQ(10., copiedRows(1, 1));
    EXPECT_EQ(16., copiedRows(3, 1));
    EXPECT_EQ(17., copiedRows(3, 2));
}

TEST_F(DeviceTest, getMatrixRows) {
    getMatrixRows<float>();
    getMatrixRows<double>();
}


/* =======================================
 * DeviceTensor<T>
 * ======================================= */

/* ---------------------------------------
 * Constructors and .pushBack
 * --------------------------------------- */

template<typename T>
void deviceTensorConstructPush() {
    size_t nRows = 2, nCols = 3, nMats = 3;
    std::vector<T> aData = {1, 2, 3,
                            4, 5, 6};
    std::vector<T> bData = {7, 8, 9,
                            10, 11, 12};
    DeviceMatrix<T> matrixA(nRows, aData, rowMajor);
    DeviceMatrix<T> matrixB(nRows, bData, rowMajor);
    T *rawA = matrixA.raw();
    T *rawB = matrixB.raw();
    DeviceTensor<T> myTensor(nRows, nCols, nMats);
    myTensor.pushBack(matrixA);
    myTensor.pushBack(matrixA);
    myTensor.pushBack(matrixB);
    DeviceVector<T *> pointersToMatrices = myTensor.devicePointersToMatrices();
    EXPECT_EQ(rawA, pointersToMatrices(0));
    EXPECT_EQ(rawA, pointersToMatrices(1));
    EXPECT_EQ(rawB, pointersToMatrices(2));
}

TEST_F(DeviceTest, deviceTensorConstructPush) {
    deviceTensorConstructPush<float>();
    deviceTensorConstructPush<double>();
    deviceTensorConstructPush<int>();
}


/* ---------------------------------------
 * Tensor-tensor multiplication
 * .addAB
 * --------------------------------------- */

template<typename T>
void deviceTensorAddAB() {
    std::vector<T> a1Data = {1.0, 2.0, 3.0,
                             6.0, 7.0, 8.0,
                             9.0, -10.0, -1.0,
                             6.0, 6.6, 6.0};
    std::vector<T> a2Data = {5.0, 2.0, 3.0,
                             6.0, 7.0, 9.0,
                             6.0, -7.0, 8.0,
                             -1.0, 0.0, 8.0};
    std::vector<T> b1Data = {5.0, 12.0,
                             7.0, 17.0,
                             11.0, 97.0};
    std::vector<T> b2Data = {-2.0, 12.0,
                             7.0, 17.0,
                             11.0, -0.1};

    // Tensor A = (A1, A2)
    DeviceMatrix<T> A1( 4, a1Data, rowMajor);
    DeviceMatrix<T> A2( 4, a2Data, rowMajor);
    DeviceTensor<T> A( 4, 3, 2);
    A.pushBack(A1);
    A.pushBack(A2);

    // Tensor B = (B1, B2)
    DeviceMatrix<T> B1( 3, b1Data, rowMajor);
    DeviceMatrix<T> B2( 3, b2Data, rowMajor);
    DeviceTensor<T> B( 3, 2, 2);
    B.pushBack(B1);
    B.pushBack(B2);

    // Tensor C = (C1, C2)
    DeviceMatrix<T> C1( 4, 2);
    DeviceMatrix<T> C2( 4, 2);
    DeviceTensor<T> C( 4, 2, 2);
    C.pushBack(C1);
    C.pushBack(C2);
    C.addAB(A, B); // C = A * B

    DeviceMatrix<T> A1B1 = A1 * B1;
    DeviceMatrix<T> error1 = C1 - A1B1;
    auto errVec1 = error1.asVector();
    EXPECT_TRUE(errVec1.norm2() < PRECISION);

    DeviceMatrix<T> A2B2 = A2 * B2;
    DeviceMatrix<T> error2 = C2 - A2B2;
    auto errVec2 = error1.asVector();
    EXPECT_TRUE(errVec2.norm2() < PRECISION);
}

TEST_F(DeviceTest, deviceTensorAddAB) {
    deviceTensorAddAB<float>();
    deviceTensorAddAB<double>();
}

/* ---------------------------------------
 * Batched least squares
 * .leastSquares
 * --------------------------------------- */

template<typename T>
void deviceTensorLeastSquares() {
    size_t rows = 3;
    size_t cols = 2;
    std::vector<T> A1 = {1, 0, 0, 3, 2, 6};
    DeviceMatrix<T> d_A1( rows, A1, MatrixStorageMode::rowMajor);
    std::vector<T> b1 = {1, 2, 3};
    DeviceVector<T> d_b1( b1);
    std::vector<T> A2 = {1, 3, 3, 2, 2, 1};
    DeviceMatrix<T> d_A2( rows, A2, MatrixStorageMode::rowMajor);
    std::vector<T> b2 = {1, 2, 3};
    DeviceVector<T> d_b2( b2);
    DeviceTensor<T> d_As( rows, cols, 2);
    DeviceTensor<T> d_bs( rows, 1, 2);
    d_As.pushBack(d_A1);
    d_As.pushBack(d_A2);
    DeviceMatrix<T> db1_mat( d_b1);
    d_bs.pushBack(db1_mat);
    DeviceMatrix<T> d_B2( d_b2);
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
    deviceTensorLeastSquares<float>();
    deviceTensorLeastSquares<double>();
}


/* =======================================
 * SVD
 * ======================================= */

/* ---------------------------------------
 * Computation of singular values
 * and matrix rank
 * --------------------------------------- */

template<typename T> requires std::floating_point<T>
void singularValuesComputation(float epsilon) {
size_t k = 8;
std::vector<T> bData{1.0, 2.0, 3.0,
                     6.0, 7.0, 8.0,
                     6.0, 7.0, 8.0,
                     6.0, 7.0, 8.0,
                     6.0, 7.0, 8.0,
                     6.0, 7.0, 8.0,
                     6.0, 7.0, 8.0,
                     6.0, 7.0, 8.0,};
DeviceMatrix<T> B( k, bData, rowMajor);
SvdFactoriser<T> svdEngine( B, true, false);
EXPECT_EQ(0, svdEngine.factorise());
auto S = svdEngine.singularValues();
unsigned int r = svdEngine.rank();
EXPECT_EQ(2, r);
EXPECT_NEAR(32.496241123753592, S(0), epsilon); // value from MATLAB
EXPECT_NEAR(0.997152358903242, S(1), epsilon); // value from MATLAB
}

TEST_F(DeviceTest, singularValuesComputation) {
    singularValuesComputation<float>(1e-4);
    singularValuesComputation<double>(1e-7);
}


/* =======================================
 * Cholesky factorisation and solution
 * of linear systems
 * ======================================= */

/* ---------------------------------------
 * Factorisation of matrix
 * --------------------------------------- */

template<typename T> requires std::floating_point<T>
void choleskyFactoriserFactorise(float epsilon) {
std::vector<T> aData{10.0, 2.0, 3.0,
                     3.0, 20.0, -1.0,
                     3.0, -1.0, 30.0};
DeviceMatrix<T> A( 3, aData, rowMajor);
CholeskyFactoriser<T> chol( A);
EXPECT_EQ(0, chol.factorise());
EXPECT_NEAR(3.162277660168380, A(0, 0), epsilon);
}

TEST_F(DeviceTest, choleskyFactoriserFactorise) {
    choleskyFactoriserFactorise<float>( 1e-4);
    choleskyFactoriserFactorise<double>( 1e-7);
}

/* ---------------------------------------
 * Solve linear system via Cholesky
 * --------------------------------------- */

template<typename T> requires std::floating_point<T>
void choleskyFactoriserSolve(float epsilon) {
std::vector<T> aData = {10., 2., 3.,
                        2., 20., -1.,
                        3., -1., 30.};
DeviceMatrix<T> A( 3, aData, rowMajor);
DeviceMatrix<T> L(A); // L = A
CholeskyFactoriser<T> chol( L);
EXPECT_EQ(0, chol.factorise());

std::vector<T> bData = {-1., -3., 5.};
DeviceVector<T> b( bData);
DeviceVector<T> sol(b); // b = x
chol.
solve(sol);

auto error = A * sol;
error -=
b;
EXPECT_TRUE(error.norm2() < epsilon);
}

TEST_F(DeviceTest, choleskyFactoriserSolve) {
    choleskyFactoriserSolve<float>( 1e-6);
    choleskyFactoriserSolve<double>( 1e-12);
}