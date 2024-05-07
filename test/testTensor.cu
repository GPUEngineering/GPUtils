#include <gtest/gtest.h>
#include "../include/tensor.cuh"

#define PRECISION_LOW 1e-4
#define PRECISION_HIGH 1e-10


/* ================================================================================================
 *  TENSOR<T> TESTS
 * ================================================================================================ */
class TensorTest : public testing::Test {
protected:
    TensorTest() {}

    virtual ~TensorTest() {}
};

#define TENSOR_DATA_234A {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 10, 5, 4, 3, 2, 1, -1, 4, 3, 4, 3, 4, 8}
#define TENSOR_DATA_234B {7, -6, 9, 2, 1, 11, 34, -1, -4, -3, 12, 7, 9, 9, 2, 9, -9, -3, 2, 5, 4, -5, 4, 5}
#define TENSOR_DATA_234APB {8, -4, 12, 6, 6, 17, 41, 7, 5, 5, 19, 17, 14, 13, 5, 11, -8, -4, 6, 8, 8, -2, 8, 13}
#define TENSOR_DATA_234AMB {-6, 8, -6, 2, 4, -5, -27, 9, 13, 11, -5, 3, -4, -5, 1, -7, 10, 2, 2, -2, 0, 8, 0, 3};

/* ---------------------------------------
 * Zero Tensor (Constructor)
 * --------------------------------------- */

template<typename T>
void tensorConstructionZero() {
    DTensor<T> zero(2, 3, 4, true);
    EXPECT_EQ(2, zero.numRows());
    EXPECT_EQ(3, zero.numCols());
    EXPECT_EQ(4, zero.numMats());
    std::vector<T> expectedResult(2 * 3 * 4, 0);
    std::vector<T> zeroDown(2 * 3 * 4);
    zero.download(zeroDown);
    EXPECT_EQ(expectedResult, zeroDown);
}

TEST_F(TensorTest, tensorConstructionZero) {
    tensorConstructionZero<float>();
    tensorConstructionZero<double>();
    tensorConstructionZero<int>();
}

/* ---------------------------------------
 * Row- and column-major data
 * --------------------------------------- */

template<typename T>
void tensorConstructionStorageMode() {
    size_t rows = 3;
    size_t cols = 2;
    size_t mats = 2;
    std::vector<T> aCm = {1, 3, 5,
                          2, 4, 6};
    std::vector<T> bCm = {7, 9, 11,
                          8, 10, 12};
    const std::vector<T> Cm = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
    std::vector<T> aRm = {1, 2,
                          3, 4,
                          5, 6};
    std::vector<T> bRm = {7, 8,
                          9, 10,
                          11, 12};
    std::vector<T> Rm = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<T> hostData(rows * cols * mats);
    // test constructor
    DTensor<T> testCm(Cm, rows, cols, mats, columnMajor);
    DTensor<T> testRm(Rm, rows, cols, mats, rowMajor);
    testCm.download(hostData);
    EXPECT_EQ(Cm, hostData);
    testRm.download(hostData);
    EXPECT_EQ(Cm, hostData);
    // test .upload()
    DTensor<T> testSplitCm(rows, cols, mats);
    DTensor<T> ACm(testSplitCm, 2, 0, 0);
    DTensor<T> BCm(testSplitCm, 2, 1, 1);
    ACm.upload(aCm, columnMajor);
    BCm.upload(bCm, columnMajor);
    DTensor<T> testSplitRm(rows, cols, mats);
    DTensor<T> ARm(testSplitRm, 2, 0, 0);
    DTensor<T> BRm(testSplitRm, 2, 1, 1);
    ARm.upload(aRm, rowMajor);
    BRm.upload(bRm, rowMajor);
    testSplitCm.download(hostData);
    EXPECT_EQ(Cm, hostData);
    testSplitRm.download(hostData);
    EXPECT_EQ(Cm, hostData);
}

TEST_F(TensorTest, tensorConstructionStorageMode) {
    tensorConstructionStorageMode<float>();
    tensorConstructionStorageMode<double>();
    tensorConstructionStorageMode<int>();
}

/* ---------------------------------------
 * Move constructor
 * --------------------------------------- */

template<typename T>
void tensorMoveConstructor() {
    DTensor<T> zero(2, 3, 4, true);
    DTensor<T> x(std::move(zero));
    DTensor<T> y(DTensor<T>{100, 10, 1000});
}

TEST_F(TensorTest, tensorMoveConstructor) {
    tensorMoveConstructor<float>();
    tensorMoveConstructor<double>();
    tensorMoveConstructor<int>();
    tensorMoveConstructor<int *>();
    tensorMoveConstructor<double *>();
}

/* ---------------------------------------
 * New tensor from data (std::vector)
 * Constructor
 * --------------------------------------- */

template<typename T>
void tensorConstructionFromVector() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    EXPECT_EQ(2, tenz.numRows());
    EXPECT_EQ(3, tenz.numCols());
    EXPECT_EQ(4, tenz.numMats());
    EXPECT_EQ(2 * 3 * 4, tenz.numEl());
}

TEST_F(TensorTest, tensorConstructionFromVector) {
    tensorConstructionFromVector<float>();
    tensorConstructionFromVector<double>();
    tensorConstructionFromVector<int>();
}

/* ---------------------------------------
 * Tensor: Copy constructor
 * --------------------------------------- */

template<typename T>
void tensorCopyConstructor() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> tenzCp(tenz);
    EXPECT_EQ(2, tenzCp.numRows());
    EXPECT_EQ(3, tenzCp.numCols());
    EXPECT_EQ(4, tenzCp.numMats());
    EXPECT_EQ(2 * 3 * 4, tenzCp.numEl());
    std::vector<T> tenzDown(2 * 3 * 4);
    tenzCp.download(tenzDown);
    EXPECT_EQ(data, tenzDown);
    EXPECT_NE(tenz.raw(), tenzCp.raw());
}

TEST_F(TensorTest, tensorCopyConstructor) {
    tensorCopyConstructor<float>();
    tensorCopyConstructor<double>();
    tensorCopyConstructor<int>();
}

/* ---------------------------------------
 * Tensor: Slicing constructor
 * axis = 2 (matrices)
 * --------------------------------------- */

template<typename T>
void tensorSlicingConstructorAxis2() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tens(data, 2, 3, 4);
    DTensor<T> tensSlice(tens, 2, 0, 1); // matrices #0 and #1
    EXPECT_EQ(2, tensSlice.numRows());
    EXPECT_EQ(3, tensSlice.numCols());
    EXPECT_EQ(2, tensSlice.numMats());
    EXPECT_EQ(tens.raw(), tensSlice.raw()); // it is indeed a slice
}

TEST_F(TensorTest, tensorSlicingConstructorAxis2) {
    tensorSlicingConstructorAxis2<float>();
    tensorSlicingConstructorAxis2<double>();
    tensorSlicingConstructorAxis2<int>();
}

/* ---------------------------------------
 * Tensor: Slicing constructor
 * axis = 1 (columns)
 * --------------------------------------- */

template<typename T>
void tensorSlicingConstructorAxis1() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> tenzSlice(tenz, 1, 1, 2); // columns from 1 to 2
    EXPECT_EQ(2, tenzSlice.numRows());
    EXPECT_EQ(2, tenzSlice.numCols());
    EXPECT_EQ(1, tenzSlice.numMats());
    std::vector<T> expected = {4, 5, 6, 7};
    std::vector<T> tenzSliceDown(4);
    tenzSlice.download(tenzSliceDown);
    EXPECT_EQ(expected, tenzSliceDown);
}

TEST_F(TensorTest, tensorSlicingConstructorAxis1) {
    tensorSlicingConstructorAxis1<float>();
    tensorSlicingConstructorAxis1<double>();
    tensorSlicingConstructorAxis1<int>();
}

/* ---------------------------------------
 * Tensor: Slicing constructor
 * axis = 0 (columns)
 * --------------------------------------- */

template<typename T>
void tensorSlicingConstructorAxis0() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> tenzSlice(tenz, 0, 2, 3); // elements 2..3
    EXPECT_EQ(2, tenzSlice.numRows());
    EXPECT_EQ(1, tenzSlice.numCols());
    EXPECT_EQ(1, tenzSlice.numMats());
    std::vector<T> expected = {3, 4};
    std::vector<T> tenzSliceDown(2);
    tenzSlice.download(tenzSliceDown);
    EXPECT_EQ(expected, tenzSliceDown);
}

TEST_F(TensorTest, tensorSlicingConstructorAxis0) {
    tensorSlicingConstructorAxis0<float>();
    tensorSlicingConstructorAxis0<double>();
    tensorSlicingConstructorAxis0<int>();
}

/* ---------------------------------------
 * Tensor: Upload data
 * --------------------------------------- */

template<typename T>
void tensorUpload() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(2, 3, 4);
    tenz.upload(data);
    EXPECT_EQ(2, tenz.numRows());
    EXPECT_EQ(3, tenz.numCols());
    EXPECT_EQ(4, tenz.numMats());
    EXPECT_EQ(2 * 3 * 4, tenz.numEl());
    EXPECT_EQ(4, tenz.numMats());
    EXPECT_EQ(8, tenz(1, 2, 3));
}

TEST_F(TensorTest, tensorUpload) {
    tensorUpload<float>();
    tensorUpload<double>();
    tensorUpload<int>();
}

/* ---------------------------------------
 * Tensor: deviceCopyTo
 * --------------------------------------- */

template<typename T>
void tensorDeviceCopyTo() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> other(2, 3, 5, true);
    DTensor<T> z(other, 2, 1, 4);
    tenz.deviceCopyTo(z);
    std::vector<T> expected = {0, 0, 0, 0, 0, 0,
                               1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 10, 5, 4, 3, 2, 1, -1, 4, 3, 4, 3, 4, 8};
    std::vector<T> actual(2 * 3 * 5);
    other.download(actual);
    EXPECT_EQ(expected, actual);
}

TEST_F(TensorTest, tensorDeviceCopyTo) {
    tensorDeviceCopyTo<float>();
    tensorDeviceCopyTo<double>();
    tensorDeviceCopyTo<int>();
}

/* ---------------------------------------
 * Tensor: Frobenius dot product
 * --------------------------------------- */

template<typename T>
void tensorDotF(T epsilon) {
    // as vectors
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> vecA(dataA, dataA.size());
    DTensor<T> vecB(dataB, dataB.size());
    T dotVector = vecA.dotF(vecB);
    EXPECT_EQ(604, dotVector);  // from MATLAB
    // as matrices
    DTensor<T> tenA(dataA, 2, 3, 4);
    DTensor<T> tenB(dataB, 2, 3, 4);
    T dotTensor = tenA.dotF(tenB);
    EXPECT_EQ(604, dotTensor);  // from MATLAB
}

TEST_F(TensorTest, tensorDotF) {
    tensorDotF<float>(PRECISION_LOW);
    tensorDotF<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Tensor: Frobenius norm
 * --------------------------------------- */

template<typename T>
void tensorNormF(T epsilon) {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    EXPECT_NEAR(26.153393661244042, tenz.normF(), epsilon); // from MATLAB
}

TEST_F(TensorTest, tensorNormF) {
    tensorNormF<float>(PRECISION_LOW);
    tensorNormF<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Tensor: sum of absolute value of
 * all elements
 * --------------------------------------- */

template<typename T>
void tensorSumAbs() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    EXPECT_NEAR(112, tenz.sumAbs(), PRECISION_HIGH); // from MATLAB
}

TEST_F(TensorTest, tensorNormFtensorSumAbs) {
    tensorSumAbs<float>();
    tensorSumAbs<double>();
}

/* ---------------------------------------
 * Tensor operator() to access element
 * e.g., t(2, 3, 4)
 * --------------------------------------- */

template<typename T>
void tensorBracketOperator() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    EXPECT_EQ(1, tenz(0, 0, 0));
    EXPECT_EQ(3, tenz(0, 1, 2));
    EXPECT_EQ(8, tenz(1, 2, 3));
}

TEST_F(TensorTest, tensorBracketOperator) {
    tensorBracketOperator<float>();
    tensorBracketOperator<double>();
    tensorBracketOperator<int>();
}

/* ---------------------------------------
 * Tensor assignment operator
 * --------------------------------------- */

template<typename T>
void tensorAssignmentOperator() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> other;
    other = tenz;
    EXPECT_EQ(tenz.raw(), other.raw());
    EXPECT_EQ(2, other.numRows());
    EXPECT_EQ(3, other.numCols());
    EXPECT_EQ(4, other.numMats());
}

TEST_F(TensorTest, tensorAssignmentOperator) {
    tensorAssignmentOperator<float>();
    tensorAssignmentOperator<double>();
    tensorAssignmentOperator<int>();
}

/* ---------------------------------------
 * Tensor times-equals scalar
 * --------------------------------------- */

template<typename T>
void tensorTimesEqualsScalar() {
    std::vector<T> data = TENSOR_DATA_234A;
    std::vector<T> dataTimes3 = {3, 6, 9, 12, 15, 18, 21, 24, 27, 24, 21, 30, 15, 12, 9, 6, 3, -3, 12, 9, 12, 9, 12,
                                 24};
    DTensor<T> tenz(data, 2, 3, 4);
    tenz *= 3.0;
    std::vector<T> actual;
    tenz.download(actual);
    EXPECT_EQ(dataTimes3, actual);
}

TEST_F(TensorTest, tensorTimesEqualsScalar) {
    tensorTimesEqualsScalar<float>();
    tensorTimesEqualsScalar<double>();
}

/* ---------------------------------------
 * Scalar times tensor
 * --------------------------------------- */

template<typename T>
void tensorTimesScalar() {
    std::vector<T> data = TENSOR_DATA_234A;
    std::vector<T> dataTimes3 = {3, 6, 9, 12, 15, 18, 21, 24, 27, 24, 21, 30, 15, 12, 9, 6, 3, -3, 12, 9, 12, 9, 12,
                                 24};
    DTensor<T> tenz(data, 2, 3, 4);
    auto tripleTensor = 3.0 * tenz;
    std::vector<T> actual;
    tripleTensor.download(actual);
    EXPECT_EQ(dataTimes3, actual);
}

TEST_F(TensorTest, tensorTimesScalar) {
    tensorTimesScalar<float>();
    tensorTimesScalar<double>();
}

/* ---------------------------------------
 * Tensor plus-equals tensor
 * --------------------------------------- */

template<typename T>
void tensorPlusEqualsTensor() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> A(dataA, 2, 3, 4);
    DTensor<T> B(dataB, 2, 3, 4);
    A += B;
    std::vector<T> expected = TENSOR_DATA_234APB;
    std::vector<T> actual;
    A.download(actual);
    EXPECT_EQ(expected, actual);
}

TEST_F(TensorTest, tensorPlusEqualsTensor) {
    tensorPlusEqualsTensor<float>();
    tensorPlusEqualsTensor<double>();
}

/* ---------------------------------------
 * Tensor minus-equals tensor
 * --------------------------------------- */

template<typename T>
void tensorMinusEqualsTensor() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> A(dataA, 2, 3, 4);
    DTensor<T> B(dataB, 2, 3, 4);
    A -= B;
    std::vector<T> expected = TENSOR_DATA_234AMB;
    std::vector<T> actual;
    A.download(actual);
    EXPECT_EQ(expected, actual);
}

TEST_F(TensorTest, tensorMinusEqualsTensor) {
    tensorMinusEqualsTensor<float>();
    tensorMinusEqualsTensor<double>();
}

/* ---------------------------------------
 * Tensor + Tensor
 * --------------------------------------- */

template<typename T>
void tensorPlusTensor() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> A(dataA, 2, 3, 4);
    DTensor<T> B(dataB, 2, 3, 4);
    DTensor<T> C = A + B;
    std::vector<T> expected = TENSOR_DATA_234APB;
    std::vector<T> actual;
    C.download(actual);
    EXPECT_EQ(expected, actual);
}

TEST_F(TensorTest, tensorPlusTensor) {
    tensorPlusTensor<float>();
    tensorPlusTensor<double>();
}

/* ---------------------------------------
 * Tensor - Tensor
 * --------------------------------------- */

template<typename T>
void tensorMinusTensor() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> A(dataA, 2, 3, 4);
    DTensor<T> B(dataB, 2, 3, 4);
    DTensor<T> C = A - B;
    std::vector<T> expected = TENSOR_DATA_234AMB;
    std::vector<T> actual;
    C.download(actual);
    EXPECT_EQ(expected, actual);
}

TEST_F(TensorTest, tensorMinusTensor) {
    tensorMinusTensor<float>();
    tensorMinusTensor<double>();
}

/* ---------------------------------------
 * Tensor: pointers to matrices (on device)
 * --------------------------------------- */

template<typename T>
void tensorPointersToMatrices() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    DTensor<T> A(dataA, 2, 3, 4);
    DTensor<T *> pointers = A.pointersToMatrices();
    EXPECT_EQ(4, pointers.numRows());
    EXPECT_EQ(1, pointers.numCols());
    EXPECT_EQ(1, pointers.numMats());
    T *p1 = pointers(1, 0, 0); // pointer to matrix #1
    T hostDst; // let's see what's there...
    cudaMemcpy(&hostDst, p1, sizeof(T), cudaMemcpyDeviceToHost);
    EXPECT_EQ(dataA[6], hostDst);
}

TEST_F(TensorTest, tensorPointersToMatrices) {
    tensorPointersToMatrices<float>();
    tensorPointersToMatrices<double>();
    tensorPointersToMatrices<int>();
}

/* ---------------------------------------
 * Tensor: C = AB
 * --------------------------------------- */

template<typename T>
void tensorAddAB() {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6,
                            7, 8, 9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18};
    std::vector<T> bData = {6, 5, 4, 3, 2, 1,
                            7, 6, 5, 4, 3, 2,
                            1, 2, 1, 5, -6, 8};
    DTensor<T> A(aData, 2, 3, 3);
    DTensor<T> B(bData, 3, 2, 3);
    DTensor<T> C(2, 2, 3, true);
    C.addAB(A, B);
    std::vector<T> expected = {41, 56, 14, 20, 158, 176, 77, 86, 60, 64, 111, 118};
    std::vector<T> actual;
    C.download(actual);
    EXPECT_EQ(expected, actual);
}

TEST_F(TensorTest, tensorAddAB) {
    tensorAddAB<double>();
    tensorAddAB<float>();
}

/* ---------------------------------------
 * Tensor: getRows
 * --------------------------------------- */

template<typename T>
void tensorGetRows() {
    std::vector<T> aData = {10.5, 25.0, 60.0,
                            -21.0, 720.0, -1.0,
                            11.0, -1.0, 30.0,
                            5., 6., 7.,
                            8., 9., 10.,
                            11., 12., 13};
    DTensor<T> A(aData, 3, 3, 2);
    DTensor<T> Ar0 = A.getRows(1, 1, 0);
    std::vector<T> expected0 = {25., 720., -1.};
    std::vector<T> actual0(3);
    Ar0.download(actual0);
    EXPECT_EQ(expected0, actual0);

    DTensor<T> Ar1 = A.getRows(1, 2, 1);
    std::vector<T> expected1 = {6., 7., 9., 10., 12., 13.};
    std::vector<T> actual1(6);
    Ar1.download(actual1);
    EXPECT_EQ(expected1, actual1);
}

TEST_F(TensorTest, tensorGetRows) {
    tensorGetRows<float>();
    tensorGetRows<double>();
}


/* ---------------------------------------
 * Tensor: transpose
 * --------------------------------------- */

template<typename T>
void tensorTranspose() {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    DTensor<T> A(aData, 3, 2, 2);
    DTensor<T> Atranspose = A.tr();
    EXPECT_EQ(2, Atranspose.numRows());
    EXPECT_EQ(3, Atranspose.numCols());
    EXPECT_EQ(2, Atranspose.numMats());
    std::vector<T> expected = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};
    std::vector<T> actual;
    Atranspose.download(actual);
    EXPECT_EQ(expected, actual);

}

TEST_F(TensorTest, tensorTranspose) {
    tensorTranspose<float>();
    tensorTranspose<double>();
}

/* ================================================================================================
 *  LEAST SQUARES TESTS
 * ================================================================================================ */
class LeastSquaresTest : public testing::Test {
protected:
    LeastSquaresTest() {}

    virtual ~LeastSquaresTest() {}
};

/* ---------------------------------------
 * Tensor: Least squares
 * --------------------------------------- */

template<typename T>
void tensorLeastSquares1(T epsilon) {
    // TODO test with tall matrices too
    std::vector<T> aData = {1, 2,
                            3, 4,
                            7, 8,
                            9, 10,
                            6, 8,
                            -9, 20};
    std::vector<T> bData = {1, 1, -1, 2, 30, -80};
    DTensor<T> A0(aData, 2, 2, 3);
    DTensor<T> A(A0);
    DTensor<T> B(bData, 2, 1, 3);
    DTensor<T> sol(B);
    A0.leastSquares(sol);
    DTensor<T> C(2, 1, 3);
    C.addAB(A, sol);
    C -= B;
    T nrmErr = C.normF();
    EXPECT_LT(nrmErr, epsilon);
}

TEST_F(LeastSquaresTest, tensorLS1) {
    tensorLeastSquares1<float>(PRECISION_LOW);
    tensorLeastSquares1<double>(PRECISION_HIGH);
}


/* ================================================================================================
 *  SVD TESTS
 * ================================================================================================ */
class SvdTest : public testing::Test {
protected:
    SvdTest() {}

    virtual ~SvdTest() {}
};

/* ---------------------------------------
 * Computation of singular values
 * and matrix rank
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void singularValuesComputation(float epsilon) {
    std::vector<T> bData = {1, 6, 6, 6, 6, 6, 6, 6,
                            2, 7, 7, 7, 7, 7, 7, 7,
                            3, 8, 8, 8, 8, 8, 8, 8,};
    DTensor<T> B(bData, 8, 3);
    Svd<T> svd(B, true, false);
    EXPECT_EQ(true, svd.factorise());
    auto S = svd.singularValues();
    EXPECT_NEAR(32.496241123753592, S(0), epsilon); // value from MATLAB
    EXPECT_NEAR(0.997152358903242, S(1), epsilon); // value from MATLAB

    auto U = svd.leftSingularVectors();
    EXPECT_TRUE(U.has_value());
}

TEST_F(SvdTest, singularValuesComputation) {
    singularValuesComputation<float>(PRECISION_LOW);
    singularValuesComputation<double>(PRECISION_HIGH);
}


/* ---------------------------------------
 * Singular values - memory mgmt
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void singularValuesMemory(float epsilon) {
    std::vector<T> bData = {1, 6, 6, 6, 6, 6, 6, 6,
                            2, 7, 7, 7, 7, 7, 7, 7,
                            3, 8, 8, 8, 8, 8, 8, 8,};
    DTensor<T> B(bData, 8, 3);
    Svd<T> svd(B, true, false);
    EXPECT_EQ(true, svd.factorise());
    DTensor<T> const &v1 = svd.rightSingularVectors();
    DTensor<T> const &v2 = svd.rightSingularVectors();
    EXPECT_EQ(&v1, &v2);
    EXPECT_EQ(v1.raw(), v2.raw());
    DTensor<T> const &s1 = svd.singularValues();
    DTensor<T> const &s2 = svd.singularValues();
    EXPECT_EQ(&s1, &s2);
    EXPECT_EQ(s1.raw(), s2.raw());
    auto u1 = svd.leftSingularVectors().value();
    auto u2 = svd.leftSingularVectors().value();
    EXPECT_EQ(u1, u2);
    EXPECT_EQ(u1->raw(), u2->raw());
}

TEST_F(SvdTest, singularValuesMemory) {
    singularValuesMemory<float>(PRECISION_LOW);
    singularValuesMemory<double>(PRECISION_HIGH);
}


/* ---------------------------------------
 * SVD with multiple matrices
 * --------------------------------------- */
template<typename T>
requires std::floating_point<T>
void singularValuesMultipleMatrices(float epsilon) {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1};
    DTensor<T> A(aData, 3, 2, 3);
    Svd<T> svd(A, true); // do compute U (A will be destroyed)
    svd.factorise();
    DTensor<T> const &S = svd.singularValues();
    DTensor<T> const &V = svd.rightSingularVectors();
    auto Uopt = svd.leftSingularVectors();
    auto U = Uopt.value();
    std::vector<T> expected_v = {-0.386317703118612, -0.922365780077058, -0.922365780077058, 0.386317703118612,
                                 -0.447213595499958, -0.894427190999916, 0.894427190999916, -0.447213595499958,
                                 0, -1, 1, 0};
    std::vector<T> actual_v(12);
    V.download(actual_v);
    for (size_t i = 0; i < 4; i++) EXPECT_NEAR(expected_v[i], actual_v[i], epsilon);
    std::vector<T> expected_s = {9.508032000695726, 0.772869635673484, 3.872983346207417, 0, 1, 0};
    std::vector<T> actual_s(6);
    S.download(actual_s);
    for (size_t i = 0; i < 6; i++) EXPECT_NEAR(expected_s[i], actual_s[i], epsilon);
    std::vector<T> expected_u = {
        -0.428667133548626, -0.566306918848035, -0.703946704147444,
        0.805963908589298, 0.112382414096594, -0.581199080396110,
        0.408248290463863, -0.816496580927726, 0.408248290463863,
        -0.577350269189626, -0.577350269189626, -0.577350269189626,
        0.816496580927726, -0.408248290463863, -0.408248290463863,
        0.000000000000000, -0.707106781186548, 0.707106781186547,
        0, 0, -1,
        1, 0, 0,
        0, -1, 0,
    };
    std::vector<T> actual_u(27);
    U->download(actual_u);
    for (size_t i = 0; i < 27; i++) EXPECT_NEAR(expected_u[i], actual_u[i], epsilon);

}

TEST_F(SvdTest, singularValuesMultipleMatrices) {
    singularValuesMultipleMatrices<float>(10 * PRECISION_LOW); // SVD with float performs quite poorly
    singularValuesMultipleMatrices<double>(PRECISION_HIGH);
}


/* ---------------------------------------
 * SVD for rank computation of multiple
 * matrices
 * --------------------------------------- */
template<typename T>
requires std::floating_point<T>
void singularValuesRankMultipleMatrices(float epsilon) {
    std::vector<T> aData = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 0,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12};
    DTensor<T> A(aData, 4, 3, 3);

    Svd<T> svd(A);
    svd.factorise();
    auto rank = svd.rank(epsilon);
    EXPECT_EQ(3, rank(0, 0, 0));
    EXPECT_EQ(2, rank(0, 0, 1));
    EXPECT_EQ(1, rank(0, 0, 2));
}

TEST_F(SvdTest, singularValuesRankMultipleMatrices) {
    singularValuesRankMultipleMatrices<float>(PRECISION_LOW); // SVD with float performs quite poorly
    singularValuesRankMultipleMatrices<double>(PRECISION_HIGH);
}

/* ================================================================================================
 *  CHOLESKY TESTS
 * ================================================================================================ */
class CholeskyTest : public testing::Test {
protected:
    CholeskyTest() {}

    virtual ~CholeskyTest() {}
};


/* ---------------------------------------
 * Cholesky factorisation
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyFactorisation(T epsilon) {
    std::vector<T> aData = {10.0, 2.0, 3.0,
                            2.0, 20.0, -1.0,
                            3.0, -1.0, 30.0};
    DTensor<T> A(aData, 3, 3, 1);
    CholeskyFactoriser<T> chol(A);
    chol.factorise();
    EXPECT_NEAR(3.162277660168380, A(0, 0), epsilon);
    EXPECT_NEAR(-0.361403161162101, A(2, 1), epsilon);
    EXPECT_NEAR(5.382321781081287, A(2, 2), epsilon);
}

TEST_F(CholeskyTest, choleskyFactorisation) {
    choleskyFactorisation<float>(PRECISION_LOW);
    choleskyFactorisation<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Cholesky factorisation: solve system
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyFactorisationSolution(T epsilon) {
    std::vector<T> aData = {10.0, 2.0, 3.0,
                            2.0, 20.0, -1.0,
                            3.0, -1.0, 30.0};
    DTensor<T> A(aData, 3, 3, 1);
    DTensor<T> L(A); // L = A
    CholeskyFactoriser<T> chol(L);
    chol.factorise();

    std::vector<T> bData = {-1., -3., 5.};
    DTensor<T> rhs(bData, 3, 1, 1);
    DTensor<T> sol(rhs);
    chol.solve(sol);

    std::vector<T> expected = {-0.126805213103205, -0.128566396618528, 0.175061641423036};
    std::vector<T> actual(3);
    sol.download(actual);
    for (size_t i = 0; i < 3; i++) EXPECT_NEAR(expected[i], actual[i], epsilon);

    DTensor<T> error = A * sol;
    error -= rhs;
    EXPECT_TRUE(error.normF() < epsilon);

}

TEST_F(CholeskyTest, choleskyFactorisationSolution) {
    choleskyFactorisationSolution<float>(PRECISION_LOW);
    choleskyFactorisationSolution<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Batched Cholesky factorisation
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyBatchFactorisation(T epsilon) {
    std::vector<T> aData = {10.0, 2.0, 3.0,
                            2.0, 20.0, -1.0,
                            3.0, -1.0, 30.0};
    DTensor<T> A(3, 3, 2);
    DTensor<T> A0(A, 2, 0, 0);
    DTensor<T> A1(A, 2, 1, 1);
    A0.upload(aData);
    A1.upload(aData);
    CholeskyBatchFactoriser<T> chol(A);
    chol.factorise();
    // 0
    EXPECT_NEAR(3.162277660168380, A(0, 0, 0), epsilon);
    EXPECT_NEAR(-0.361403161162101, A(2, 1, 0), epsilon);
    EXPECT_NEAR(5.382321781081287, A(2, 2, 0), epsilon);
    // 1
    EXPECT_NEAR(3.162277660168380, A(0, 0, 1), epsilon);
    EXPECT_NEAR(-0.361403161162101, A(2, 1, 1), epsilon);
    EXPECT_NEAR(5.382321781081287, A(2, 2, 1), epsilon);
}

TEST_F(CholeskyTest, choleskyBatchFactorisation) {
    choleskyBatchFactorisation<float>(PRECISION_LOW);
    choleskyBatchFactorisation<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Batched Cholesky solve
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyBatchFactorSolve(T epsilon) {
    std::vector<T> aData = {10.0, 2.0, 3.0,
                            2.0, 20.0, -1.0,
                            3.0, -1.0, 30.0};
    DTensor<T> A(3, 3, 2);
    DTensor<T> A0(A, 2, 0, 0);
    DTensor<T> A1(A, 2, 1, 1);
    A0.upload(aData);
    A1.upload(aData);
    DTensor<T> L(A); // L = A
    CholeskyBatchFactoriser<T> chol(L);
    chol.factorise();
    std::vector<T> bData = {-1., -3., 5.};
    DTensor<T> rhs(3, 1, 2);
    DTensor<T> rhs0(rhs, 2, 0, 0);
    DTensor<T> rhs1(rhs, 2, 1, 1);
    rhs0.upload(bData);
    rhs1.upload(bData);
    DTensor<T> sol(rhs);
    chol.solve(sol);
    std::vector<T> expected = {-0.126805213103205, -0.128566396618528, 0.175061641423036};
    std::vector<T> actual(6);
    sol.download(actual);
    for (size_t i = 0; i < 3; i++) EXPECT_NEAR(expected[i], actual[i], epsilon);  // 0
    for (size_t i = 0; i < 3; i++) EXPECT_NEAR(expected[i], actual[i + 3], epsilon);  // 1
    DTensor<T> error = A * sol;
    error -= rhs;
    EXPECT_TRUE(error.normF() < epsilon);
}

TEST_F(CholeskyTest, choleskyBatchFactorSolve) {
    choleskyBatchFactorSolve<float>(PRECISION_LOW);
    choleskyBatchFactorSolve<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Batched Cholesky solve (factor provided)
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void choleskyBatchSolve(T epsilon) {
    /* originalMatrix = {10.0, 2.0, 3.0,
                         2.0, 20.0, -1.0,
                         3.0, -1.0, 30.0}; */
    std::vector<T> lowData = {3.162277660168380, 0, 0,
                              0.632455532033676, 4.427188724235731, 0,
                              0.948683298050514, -0.361403161162101, 5.382321781081287};  // from matlab
    DTensor<T> low(3, 3, 2);
    DTensor<T> low0(low, 2, 0, 0);
    DTensor<T> low1(low, 2, 1, 1);
    low0.upload(lowData);
    low1.upload(lowData);
    CholeskyBatchFactoriser<T> chol(low, true);
    std::vector<T> bData = {-1., -3., 5.};
    DTensor<T> rhs(3, 1, 2);
    DTensor<T> rhs0(rhs, 2, 0, 0);
    DTensor<T> rhs1(rhs, 2, 1, 1);
    rhs0.upload(bData);
    rhs1.upload(bData);
    DTensor<T> sol(rhs);
    chol.solve(sol);
    std::vector<T> expected = {-0.126805213103205, -0.128566396618528, 0.175061641423036};
    std::vector<T> actual(6);
    sol.download(actual);
    for (size_t i = 0; i < 3; i++) EXPECT_NEAR(expected[i], actual[i], epsilon);  // 0
    for (size_t i = 0; i < 3; i++) EXPECT_NEAR(expected[i], actual[i + 3], epsilon);  // 1
    DTensor<T> error = low * sol;
    error -= rhs;
    EXPECT_TRUE(error.normF() < epsilon);
}

TEST_F(CholeskyTest, choleskyBatchSolve) {
    choleskyBatchSolve<float>(PRECISION_LOW);
    choleskyBatchSolve<double>(PRECISION_HIGH);
}


/* ================================================================================================
 *  NULLSPACE TESTS
 * ================================================================================================ */
class NullspaceTest : public testing::Test {
protected:
    NullspaceTest() {}

    virtual ~NullspaceTest() {}
};


/* ---------------------------------------
 * Basic nullspace test
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void computeNullspaceTensor(T epsilon) {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9,
                            1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    DTensor<T> A(aData, 3, 4, 5);
    Nullspace<T> ns(A);
    DTensor<T> nA = ns.nullspace();
    size_t nMats = nA.numMats();
    EXPECT_EQ(nMats, 5);
    for (size_t i = 0; i < nMats; i++) {
        DTensor<T> nAi(nA, 2, i, i);
        DTensor<T> Ai(A, 2, i, i);
        DTensor<T> mustBeZero = Ai * nAi;
        EXPECT_LT(mustBeZero.normF(), epsilon);

        DTensor<T> nAiTr = nAi.tr();
        DTensor<T> mustBeEye = nAiTr * nAi;
        EXPECT_NEAR(1, mustBeEye(0, 0, 0), epsilon);
        for (size_t ir = 0; ir < mustBeEye.numRows(); ir++) {
            for (size_t ic = 0; ic < mustBeEye.numCols(); ic++) {
                if (ir != ic) {
                    EXPECT_NEAR(0, mustBeEye(ir, ic, 0), epsilon);
                }
            }
        }
    }
}

TEST_F(NullspaceTest, computeNullspaceTensor) {
    computeNullspaceTensor<float>(PRECISION_LOW);
    computeNullspaceTensor<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Nullspace is trivial
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void computeNullspaceTrivial(T epsilon) {
    std::vector<T> data{4, 5, 7,
                        4, 1, 8,
                        4, 5, 0,
                        1, 1, 1,
                        5, 6, 7,
                        9, 0, 3};
    DTensor<T> A(data, 3, 3, 2, rowMajor);
    Nullspace<T> nullA(A);
    DTensor<T> N = nullA.nullspace();
    EXPECT_EQ(N.normF(), 0);
}

TEST_F(NullspaceTest, computeNullspaceTrivial) {
    computeNullspaceTrivial<float>(PRECISION_LOW);
    computeNullspaceTrivial<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Project onto nullspace
 * --------------------------------------- */

template<typename T>
requires std::floating_point<T>
void projectOnNullspaceTensor(T epsilon) {
    // offline
    size_t m = 3;
    size_t n = 7;
    std::vector<T> mat{1, -2, 3, 4, -1, -1, -1,
                       1, 2, -3, 4, -1, -1, -1,
                       -1, 3, 5, -7, -1, -1, -1};
    DTensor<T> A(m, n, 1);
    A.upload(mat, rowMajor);
    Nullspace<T> ns = Nullspace(A);
    DTensor<T> N = ns.nullspace();

    // online
    std::vector<T> vec{1, 2, 3, 4, 5, 6, 7};
    DTensor<T> x(vec, n);
    DTensor<T> proj(x);
    ns.project(proj);

    // Testing that proj is indeed in ker A
    DTensor<T> error(m, 1, 1, true);
    error.addAB(A, proj);
    EXPECT_TRUE(error.normF() < epsilon);

    // Orthogonality test (other - p) † (p - x)
    std::vector<T> h_other{1, -2, 5, 4, 0, 0, 0};
    DTensor<T> other(h_other, n);
    DTensor<T> y = N * other;
    DTensor<T> delta1 = y - proj;
    DTensor<T> delta2 = proj - x;
    EXPECT_LT(delta1.dotF(delta2), epsilon);
}

TEST_F(NullspaceTest, projectOnNullspaceTensor) {
    projectOnNullspaceTensor<float>(PRECISION_LOW);
    projectOnNullspaceTensor<double>(PRECISION_HIGH);
}