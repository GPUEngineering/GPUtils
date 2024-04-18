#include <gtest/gtest.h>
#include "../include/tensor.cuh"

#define PRECISION 1e-6


class TensorTest : public testing::Test {
protected:
    TensorTest() {}

    virtual ~TensorTest() {}
};

#define TENSOR_DATA_234A {1, 2,  3, 4, 5, 6, 7, 8, 9, 8, 7, 10, 5, 4, 3, 2, 1, -1, 4, 3, 4, 3, 4, 8}
#define TENSOR_DATA_234B {7, -6, 9, 2, 1, 11, 34, -1, -4, -3, 12, 7, 9, 9, 2, 9, -9, -3, 2, 5, 4, -5, 4, 5}
#define TENSOR_DATA_234APB {8, -4, 12, 6, 6, 17, 41, 7, 5, 5, 19, 17, 14, 13, 5, 11, -8, -4, 6, 8, 8, -2, 8, 13}
#define TENSOR_DATA_234AMB {-6, 8, -6, 2, 4, -5, -27, 9, 13, 11, -5, 3, -4, -5, 1, -7, 10, 2, 2, -2, 0, 8, 0, 3};

/* ---------------------------------------
 * Zero Tensor (Constructor)
 * --------------------------------------- */

template<typename T>
void tensorConstructionZero() {
    Tenzor<T> zero(2, 3, 4, true);
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
 * New tensor from data (std::vector)
 * Constructor
 * --------------------------------------- */

template<typename T>
void tensorConstructionFromVector() {
    std::vector<T> data = TENSOR_DATA_234A;
    Tenzor<T> tenz(data, 2, 3, 4);
    EXPECT_EQ(2, tenz.numRows());
    EXPECT_EQ(3, tenz.numCols());
    EXPECT_EQ(4, tenz.numMats());
    EXPECT_EQ(2 * 3 * 4, tenz.numel());
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
    Tenzor<T> tenz(data, 2, 3, 4);
    Tenzor<T> tenzCp(tenz);
    EXPECT_EQ(2, tenzCp.numRows());
    EXPECT_EQ(3, tenzCp.numCols());
    EXPECT_EQ(4, tenzCp.numMats());
    EXPECT_EQ(2 * 3 * 4, tenzCp.numel());
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
    Tenzor<T> tenz(data, 2, 3, 4);
    Tenzor<T> tenzSlice(tenz, 2, 0, 1); // matrices #0 and #1
    EXPECT_EQ(2, tenzSlice.numRows());
    EXPECT_EQ(3, tenzSlice.numCols());
    EXPECT_EQ(2, tenzSlice.numMats());
    EXPECT_EQ(tenz.raw(), tenzSlice.raw()); // it is indeed a slice
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
    Tenzor<T> tenz(data, 2, 3, 4);
    Tenzor<T> tenzSlice(tenz, 1, 1, 2); // matrices #0 and #1
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
    Tenzor<T> tenz(data, 2, 3, 4);
    Tenzor<T> tenzSlice(tenz, 0, 2, 3); // matrices #0 and #1
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
    Tenzor<T> tenz(2, 3, 4);
    tenz.upload(data);
    EXPECT_EQ(2, tenz.numRows());
    EXPECT_EQ(3, tenz.numCols());
    EXPECT_EQ(4, tenz.numMats());
    EXPECT_EQ(2 * 3 * 4, tenz.numel());
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
    Tenzor<T> tenz(data, 2, 3, 4);
    Tenzor<T> other(2, 3, 5, true);
    Tenzor<T> z(other, 2, 1, 4);
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
 * Tensor: Frobenius norm
 * --------------------------------------- */

template<typename T>
void tensorNormF(T epsilon) {
    std::vector<T> data = TENSOR_DATA_234A;
    Tenzor<T> tenz(data, 2, 3, 4);
    EXPECT_NEAR(26.153393661244042, tenz.normF(), epsilon); // from MATLAB
}

TEST_F(TensorTest, tensorNormF) {
    tensorNormF<float>(1e-6);
    tensorNormF<double>(1e-12);
}

/* ---------------------------------------
 * Tensor operator() to access element
 * e.g., t(2, 3, 4)
 * --------------------------------------- */

template<typename T>
void tensorBracketOperator() {
    std::vector<T> data = TENSOR_DATA_234A;
    Tenzor<T> tenz(data, 2, 3, 4);
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
    Tenzor<T> tenz(data, 2, 3, 4);
    Tenzor<T> other;
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
    Tenzor<T> tenz(data, 2, 3, 4);
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
 * Tensor plus-equals tensor
 * --------------------------------------- */

template<typename T>
void tensorPlusEqualsTensor() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    Tenzor<T> A(dataA, 2, 3, 4);
    Tenzor<T> B(dataB, 2, 3, 4);
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
    Tenzor<T> A(dataA, 2, 3, 4);
    Tenzor<T> B(dataB, 2, 3, 4);
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
    Tenzor<T> A(dataA, 2, 3, 4);
    Tenzor<T> B(dataB, 2, 3, 4);
    Tenzor<T> C = A + B;
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
    Tenzor<T> A(dataA, 2, 3, 4);
    Tenzor<T> B(dataB, 2, 3, 4);
    Tenzor<T> C = A - B;
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
    Tenzor<T> A(dataA, 2, 3, 4);
    Tenzor<T *> pointers = A.pointersToMatrices();
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
    Tenzor<T> A(aData, 2, 3, 3);
    Tenzor<T> B(bData, 3, 2, 3);
    Tenzor<T> C(2, 2, 3, true);
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



