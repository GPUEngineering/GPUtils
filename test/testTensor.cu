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

#define TENSOR_DATA_234A {1, 2,  3, 4, 5, 6, 7, 8, 9, 8, 7, 10, 5, 4, 3, 2, 1, -1, 4, 3, 4, 3, 4, 8}
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
    std::vector<T> aData{10.5, 25.0, 60.0,
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
    std::vector<T> bData{1, 6, 6, 6, 6, 6, 6, 6,
                         2, 7, 7, 7, 7, 7, 7, 7,
                         3, 8, 8, 8, 8, 8, 8, 8,};
    DTensor<T> B(bData, 8, 3);
    Svd<T> svd(B, true, false);
    EXPECT_EQ(true, svd.factorise());
    auto S = svd.singularValues();
    unsigned int r = svd.rank();
    EXPECT_EQ(2, r);
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
 * SVD with multiple matrices
 * --------------------------------------- */
template<typename T>
requires std::floating_point<T>
void singularValuesMutlipleMatrices(float epsilon) {
    std::vector<T> aData{1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1};
    DTensor<T> A(aData, 3, 2, 3);

    Svd<T> svd(A, true); // do compute U (A will be destroyed)
    svd.factorise();
    auto S = svd.singularValues();
    auto V = svd.rightSingularVectors();
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
    U.download(actual_u);
    for (size_t i = 0; i < 27; i++) EXPECT_NEAR(expected_u[i], actual_u[i], epsilon);

}

TEST_F(SvdTest, singularValuesMutlipleMatrices) {
    singularValuesMutlipleMatrices<float>(10 * PRECISION_LOW); // SVD with float performs quite poorly
    singularValuesMutlipleMatrices<double>(PRECISION_HIGH);
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
    std::vector<T> aData{10.0, 2.0, 3.0,
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
    std::vector<T> aData{10.0, 2.0, 3.0,
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