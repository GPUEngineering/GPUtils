#include <gtest/gtest.h>
#include "../include/tensor.cuh"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

#define PRECISION_LOW 1e-4
#define PRECISION_HIGH 1e-10


/* ================================================================================================
 *  TENSOR<T> TESTS
 * ================================================================================================ */
class TensorTest : public testing::Test {
protected:
    TensorTest() {
    }

    virtual ~TensorTest() {
    }
};

#define TENSOR_DATA_234A {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 10, 5, 4, 3, 2, 1, -1, 4, 3, 4, 3, 4, 8}
#define TENSOR_DATA_234B {7, -6, 9, 2, 1, 11, 34, -1, -4, -3, 12, 7, 9, 9, 2, 9, -9, -3, 2, 5, 4, -5, 4, 5}
#define TENSOR_DATA_234APB {8, -4, 12, 6, 6, 17, 41, 7, 5, 5, 19, 17, 14, 13, 5, 11, -8, -4, 6, 8, 8, -2, 8, 13}
#define TENSOR_DATA_234AMB {-6, 8, -6, 2, 4, -5, -27, 9, 13, 11, -5, 3, -4, -5, 1, -7, 10, 2, 2, -2, 0, 8, 0, 3};

/* ---------------------------------------
 * Session streams
 *
 * Configure before main, without initializing CUDA, so filtered/shuffled tests
 * have the same stream setup and do not depend on test execution order.
 * --------------------------------------- */

static const bool testStreamsConfigured = [] { Session::setStreams(2); return true; }();

TEST(SessionTest, streamSelectionAndExecution) {
    EXPECT_THROW(Session::setStreams(0), std::invalid_argument);

    DTensor<double> A({1, 2, 3, 4}, 2, 2);
    DTensor<double> B({5, 6, 7, 8}, 2, 2);
    DTensor<double> C(2, 2, 1, true);
    A.setStreamIdx(1);
    B.setStreamIdx(1);
    C.setStreamIdx(1);

    EXPECT_EQ(1, A.streamIdx());
    EXPECT_EQ(1, B.streamIdx());
    EXPECT_EQ(1, C.streamIdx());
    EXPECT_THROW(A.setStreamIdx(2), std::invalid_argument);
    EXPECT_THROW(Session::setStreams(3), std::logic_error);

    C.addAB(A, B);
    auto selectedRow = C.getRows(1, 1, 0);
    EXPECT_EQ(1, selectedRow.streamIdx());
    Session::getInstance().synchronizeStream(1);
    std::vector<double> rowResult;
    selectedRow.download(rowResult);
    EXPECT_EQ((std::vector<double>{34, 46}), rowResult);
    std::vector<double> result;
    C.download(result);
    EXPECT_EQ((std::vector<double>{23, 34, 31, 46}), result);

    DTensor<double> lhs({1, 0, 0, 1, 1, 0, 0, 1}, 2, 2, 2);
    DTensor<double> rhs(std::vector<double>{2, 3, 4, 5}, 2, 1, 2);
    DTensor<int> status(2); // The solver uses lhs's stream, not the buffer's.
    lhs.setStreamIdx(1);
    rhs.setStreamIdx(1);
    lhs.leastSquaresBatched(rhs, status);
    Session::getInstance().synchronizeStream(1);
    rhs.download(result);
    EXPECT_EQ((std::vector<double>{2, 3, 4, 5}), result);
    std::vector<int> codes;
    status.download(codes);
    EXPECT_EQ((std::vector<int>{0, 0}), codes);
}

// Temporarily replace handle 1's streams with a nonblocking stream to remove
// implicit legacy-default-stream ordering from the regression tests.
class NonblockingTestStream {
public:
    cudaStream_t stream, oldBlasStream, oldSolverStream;
    cublasHandle_t blas = Session::getInstance().cuBlasHandle(1);
    cusolverDnHandle_t solver = Session::getInstance().cuSolverHandle(1);
    cublasPointerMode_t oldMode;
    NonblockingTestStream() {
        gpuErrChk(cublasGetStream(blas, &oldBlasStream));
        gpuErrChk(cusolverDnGetStream(solver, &oldSolverStream));
        gpuErrChk(cublasGetPointerMode(blas, &oldMode));
        gpuErrChk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        gpuErrChk(cublasSetStream(blas, stream));
        gpuErrChk(cusolverDnSetStream(solver, stream));
    }
    ~NonblockingTestStream() {
        gpuErrChk(cudaStreamSynchronize(stream));
        gpuErrChk(cublasSetPointerMode(blas, oldMode));
        gpuErrChk(cublasSetStream(blas, oldBlasStream));
        gpuErrChk(cusolverDnSetStream(solver, oldSolverStream));
        gpuErrChk(cudaStreamDestroy(stream));
    }
};

static void CUDART_CB delayTestInput(void *) {
    // No CUDA API calls are permitted inside a CUDA host callback.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

template<typename T>
void qrUsesSelectedStream(T epsilon) {
    NonblockingTestStream execution;
    DTensor<T> source(std::vector<T>{1,0,1,0,1,1}, 3, 2);
    DTensor<T> sourceB(std::vector<T>{2,3,5}, 3);
    DTensor<T> A(3,2,1,true), B(3,1,1,true), captured(3), Q(3,2), R(2,2);
    A.setStreamIdx(1);
    QRFactoriser<T> qr(A);
    gpuErrChk(cudaLaunchHostFunc(execution.stream, delayTestInput, nullptr));
    gpuErrChk(cudaMemcpyAsync(A.raw(), source.raw(), 6*sizeof(T), cudaMemcpyDeviceToDevice, execution.stream));
    gpuErrChk(cudaMemcpyAsync(B.raw(), sourceB.raw(), 3*sizeof(T), cudaMemcpyDeviceToDevice, execution.stream));
    qr.factorise();
    qr.leastSquares(B);
    // A downstream consumer must see the completed solution on this stream.
    gpuErrChk(cudaMemcpyAsync(captured.raw(), B.raw(), 3*sizeof(T), cudaMemcpyDeviceToDevice, execution.stream));
    gpuErrChk(cudaStreamSynchronize(execution.stream));
    std::vector<T> solution;
    captured.download(solution);
    EXPECT_NEAR(2, solution[0], epsilon);
    EXPECT_NEAR(3, solution[1], epsilon);
    EXPECT_EQ(0, qr.info()(0));

    qr.getQR(Q,R);
    std::vector<T> q, r, expected;
    Q.download(q); R.download(r); source.download(expected);
    for (size_t c=0;c<2;++c) for (size_t row=0;row<3;++row) {
        T actual=0;
        for (size_t k=0;k<2;++k) actual += q[row+3*k]*r[k+2*c];
        EXPECT_NEAR(expected[row+3*c], actual, epsilon);
    }
}

TEST_F(TensorTest, qrOrdersWorkOnSelectedStream) {
    qrUsesSelectedStream<float>(PRECISION_LOW);
    qrUsesSelectedStream<double>(PRECISION_HIGH);
}

template<typename T>
void givensUsesSelectedStream(T epsilon) {
    NonblockingTestStream execution;
    DTensor<T> source(std::vector<T>{3,4,0,5},2,2);
    DTensor<T> A(std::vector<T>{1,0,0,1},2,2), captured(2,2);
    A.setStreamIdx(1);
    GivensAnnihilator<T> rotation(A);
    for (auto mode : {CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE}) {
        A.upload(std::vector<T>{1,0,0,1});
        gpuErrChk(cublasSetPointerMode(execution.blas,mode));
        gpuErrChk(cudaLaunchHostFunc(execution.stream,delayTestInput,nullptr));
        gpuErrChk(cudaMemcpyAsync(A.raw(),source.raw(),4*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
        rotation.annihilate(0,1,0);
        gpuErrChk(cudaMemcpyAsync(captured.raw(),A.raw(),4*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
        gpuErrChk(cudaStreamSynchronize(execution.stream));
        cublasPointerMode_t actualMode;
        gpuErrChk(cublasGetPointerMode(execution.blas,&actualMode));
        EXPECT_EQ(mode,actualMode);
        std::vector<T> actual;
        captured.download(actual);
        const std::vector<T> expected{5,0,4,3};
        for(size_t i=0;i<expected.size();++i) EXPECT_NEAR(expected[i],actual[i],epsilon);
    }
}

TEST_F(TensorTest, givensOrdersWorkOnSelectedStream) {
    givensUsesSelectedStream<float>(PRECISION_LOW);
    givensUsesSelectedStream<double>(PRECISION_HIGH);
}

template<typename T>
void streamSelectionPreservesStorage() {
    static_assert(std::is_same_v<decltype(std::declval<DTensor<T> &>().setStreamIdx(0)), DTensor<T> &>);
    static_assert(std::is_same_v<decltype(std::declval<DTensor<T> &&>().setStreamIdx(0)), DTensor<T>>);
    auto &session = Session::getInstance();
    const size_t before = session.totalAllocatedBytes();
    const std::vector<T> data{1, 2, 3, 4, 5, 6, 7, 8};
    {
        DTensor<T> source(data, 2, 2, 2);
        auto *raw = source.raw();
        auto *matrices = source.ptrMatrices();
        const size_t allocated = session.totalAllocatedBytes();
        auto &reference = source.setStreamIdx(0);
        EXPECT_EQ(&source, &reference);
        EXPECT_EQ(raw, source.raw());
        EXPECT_EQ(matrices, source.ptrMatrices());
        EXPECT_EQ(allocated, session.totalAllocatedBytes());

        EXPECT_THROW(std::move(source).setStreamIdx(session.numStreams()), std::invalid_argument);
        EXPECT_EQ(raw, source.raw());
        EXPECT_EQ(0, source.streamIdx());

        auto moved = std::move(source).setStreamIdx(0);
        EXPECT_EQ(raw, moved.raw());
        EXPECT_EQ(matrices, moved.ptrMatrices());
        EXPECT_EQ(nullptr, source.raw());
        EXPECT_EQ(0, source.numEl());
        EXPECT_EQ(allocated, session.totalAllocatedBytes());
        std::vector<T> downloaded;
        moved.download(downloaded);
        EXPECT_EQ(data, downloaded);
    }
    EXPECT_EQ(before, session.totalAllocatedBytes());
    {
        auto chained = DTensor<T>(data, 2, 2, 2).setStreamIdx(0).setStreamIdx(0);
        EXPECT_EQ(2, chained.numRows());
        EXPECT_EQ(2, chained.numCols());
        EXPECT_EQ(2, chained.numMats());
        EXPECT_EQ(0, chained.streamIdx());
        std::vector<T> downloaded;
        chained.download(downloaded);
        EXPECT_EQ(data, downloaded);
    }
    EXPECT_EQ(before, session.totalAllocatedBytes());
}

TEST_F(TensorTest, streamSelectionPreservesStorage) {
    streamSelectionPreservesStorage<float>();
    streamSelectionPreservesStorage<double>();
    streamSelectionPreservesStorage<int>();
}

/* ---------------------------------------
 * Zero Tensor (Constructor)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
void tensorConstructionStorageMode() {
    size_t rows = 3;
    size_t cols = 2;
    size_t mats = 2;
    std::vector<T> aCm = {
        1, 3, 5,
        2, 4, 6
    };
    std::vector<T> bCm = {
        7, 9, 11,
        8, 10, 12
    };
    const std::vector<T> Cm = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
    std::vector<T> aRm = {
        1, 2,
        3, 4,
        5, 6
    };
    std::vector<T> bRm = {
        7, 8,
        9, 10,
        11, 12
    };
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
 * Create random tensors
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void randomTensorCreation() {
    // 20 x 40 x 60-tensor; elements drawn from U[-1, 1]
    auto r = DTensor<T>::createRandomTensor(20, 40, 60, -1, 1);
    EXPECT_EQ(20, r.numRows());
    EXPECT_EQ(40, r.numCols());
    EXPECT_EQ(60, r.numMats());
    auto rEle = r(19, 39, 59);
    EXPECT_TRUE(rEle >= -1 && rEle <= 1);
}

TEST_F(TensorTest, randomTensorCreation) {
    randomTensorCreation<float>();
    randomTensorCreation<double>();
    randomTensorCreation<int>();
}

/* ---------------------------------------
 * Save to file and parse (text)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void parseTensorFromFile() {
    size_t n_runs = 10;
    for (size_t i = 0; i < n_runs; i++) {
        size_t nR = 20, nC = 40, nM = 6;
        auto r = DTensor<T>::createRandomTensor(nR, nC, nM, -1, 1);
        std::string fName = "myTest.dtensor";
        r.saveToFile(fName);
        auto a = DTensor<T>::parseFromFile(fName);
        EXPECT_EQ(nR, a.numRows());
        EXPECT_EQ(nC, a.numCols());
        EXPECT_EQ(nM, a.numMats());
        auto diff = a - r;
        T err = diff.maxAbs();
        EXPECT_LT(err, 2 * std::numeric_limits<T>::epsilon());
    }
}

TEST_F(TensorTest, parseTensorFromFile) {
    parseTensorFromFile<float>();
    parseTensorFromFile<double>();
}

TEST_F(TensorTest, parseTensorUnsupportedDataType) {
    size_t nR = 20, nC = 40, nM = 60;
    auto r = DTensor<double>::createRandomTensor(nR, nC, nM, -1, 1);
    std::string fName = "myTest.dtensor";
    r.saveToFile(fName);
    EXPECT_THROW(DTensor<char>::parseFromFile(fName), std::invalid_argument);
}

/* ---------------------------------------
 * Save to file and parse (binary)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void parseTensorFromFileBinary() {
    size_t nR = 20, nC = 40, nM = 6;
    auto r = DTensor<T>::createRandomTensor(nR, nC, nM, -1, 1);
    std::string fName = "myTest.bt";
    r.saveToFile(fName);
    auto a = DTensor<T>::parseFromFile(fName);
    EXPECT_EQ(nR, a.numRows());
    EXPECT_EQ(nC, a.numCols());
    EXPECT_EQ(nM, a.numMats());
    auto diff = a - r;
    T err = diff.maxAbs();
    EXPECT_LT(err, 2 * std::numeric_limits<T>::epsilon());
}

TEST_F(TensorTest, parseTensorFromFileBinary) {
    parseTensorFromFileBinary<float>();
    parseTensorFromFileBinary<double>();
}


/* ---------------------------------------
 * Parse files generated by Python
 * --------------------------------------- */

TEST_F(TensorTest, parseTensorFromBinaryPython) {
    // Write the binary layout emitted by write_array_to_gputils_binary_file()
    // directly, so this test does not depend on the Python test suite having
    // run first or on its working directory.
    const auto fName = std::filesystem::path(testing::TempDir()) / "gputils_python_fixture.bt";
    const uint64_t numRows = 3, numCols = 3, numMats = 2;
    const std::vector<double> pythonBinaryData = {
        1, 7, 13, 3, 9, 15, 5, 11, 17,
        2, 8, 14, 4, 10, 16, 6, 12, 18
    };
    {
        std::ofstream file(fName, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.write(reinterpret_cast<const char *>(&numRows), sizeof(numRows));
        file.write(reinterpret_cast<const char *>(&numCols), sizeof(numCols));
        file.write(reinterpret_cast<const char *>(&numMats), sizeof(numMats));
        file.write(reinterpret_cast<const char *>(pythonBinaryData.data()),
                   static_cast<std::streamsize>(pythonBinaryData.size() * sizeof(double)));
        ASSERT_TRUE(file.good());
    }

    DTensor<double> b = DTensor<double>::parseFromFile(fName.string());
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_NEAR(1 + 2*j + 6*i, b(i, j, 0), PRECISION_HIGH);
            EXPECT_NEAR(2 + 2*j + 6*i, b(i, j, 1), PRECISION_HIGH);
        }
    }
    std::filesystem::remove(fName);
}


/* ---------------------------------------
 * Parse not existing file
 * --------------------------------------- */

TEST_F(TensorTest, parseTensorFromNonexistentFile) {
    std::string fName = "../../python/whatever.bt";
    EXPECT_THROW(DTensor<double> b = DTensor<double>::parseFromFile(fName, rowMajor), std::invalid_argument);
    std::string fName2 = "../../python/whatever.txt";
    EXPECT_THROW(DTensor<double> b = DTensor<double>::parseFromFile(fName2, rowMajor), std::invalid_argument);
}


/* ---------------------------------------
 * Move constructor
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
void tensorSlicingConstructorAxis2() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tens(data, 2, 3, 4);
    DTensor<T> tensSlice(tens, 2, 0, 1); // matrices #0 and #1
    EXPECT_EQ(2, tensSlice.numRows());
    EXPECT_EQ(3, tensSlice.numCols());
    EXPECT_EQ(2, tensSlice.numMats());
    EXPECT_EQ(tens.raw(), tensSlice.raw()); // it is indeed a slice
    EXPECT_TRUE(tensSlice.ptrMatrices() != nullptr);
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

TEMPLATE_WITH_TYPE_T
void tensorSlicingConstructorAxis1() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> tenzSlice(tenz, 1, 1, 2); // columns from 1 to 2
    EXPECT_EQ(2, tenzSlice.numRows());
    EXPECT_EQ(2, tenzSlice.numCols());
    EXPECT_EQ(1, tenzSlice.numMats());
    EXPECT_TRUE(tenzSlice.ptrMatrices() == nullptr);
    std::vector<T> expected = {3, 4, 5, 6};
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
 * axis = 0 (rows)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorSlicingConstructorAxis0() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> column(tenz, 1, 1, 1);
    DTensor<T> tenzSlice(column, 0, 0, 1); // both rows of the second column
    EXPECT_EQ(2, tenzSlice.numRows());
    EXPECT_EQ(1, tenzSlice.numCols());
    EXPECT_EQ(1, tenzSlice.numMats());
    EXPECT_TRUE(tenzSlice.ptrMatrices() == nullptr);
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
void tensorDeviceCopyTo() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    DTensor<T> other(2, 3, 5, true);
    DTensor<T> z(other, 2, 1, 4);
    tenz.deviceCopyTo(z);
    std::vector<T> expected = {
        0, 0, 0, 0, 0, 0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 10, 5, 4, 3, 2, 1, -1, 4, 3, 4, 3, 4, 8
    };
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
 * Tensor: Reshape
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorReshape() {
    size_t m = 5, n = 10, k = 3;
    DTensor<T> a = DTensor<T>::createRandomTensor(m, n, k, -1, 1); // dim = (m, n, k)
    T lastElement = a(m - 1, n - 1, k - 1); // last element
    T firstElement = a(0, 0, 0);
    ASSERT_EQ(m, a.numRows());
    ASSERT_EQ(n, a.numCols());
    ASSERT_EQ(k, a.numMats());
    a.reshape(m, k, n); // dim = (m, k, n)
    ASSERT_EQ(m, a.numRows());
    ASSERT_EQ(k, a.numCols());
    ASSERT_EQ(n, a.numMats());
    a.reshape(k, n, m); // dim = (k, n, m)
    ASSERT_EQ(k, a.numRows());
    ASSERT_EQ(n, a.numCols());
    ASSERT_EQ(m, a.numMats());
    a.reshape(k * n, m, 1); // dim = (k*n, m, 1)
    ASSERT_EQ(k * n, a.numRows());
    ASSERT_EQ(m, a.numCols());
    ASSERT_EQ(1, a.numMats());
    a.reshape(m, k * n, 1); // dim = (m, k*n, 1)
    ASSERT_EQ(m, a.numRows());
    ASSERT_EQ(k * n, a.numCols());
    ASSERT_EQ(1, a.numMats());
    a.reshape(m * k * n, 1, 1); // dim = (m*k*n, 1, 1)
    ASSERT_EQ(m * k * n, a.numRows());
    ASSERT_EQ(1, a.numCols());
    ASSERT_EQ(1, a.numMats());
    ASSERT_EQ(lastElement, a(m * n * k - 1, 0, 0));
    ASSERT_EQ(firstElement, a(0, 0, 0));
}

TEST_F(TensorTest, tensorReshape) {
    tensorReshape<float>();
    tensorReshape<double>();
    tensorReshape<int>();
}

/* ---------------------------------------
 * Tensor: Slice, reshape and add/multiply
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorSliceAndReshape(T epsilon) {
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> a(dataA, 2, 3, 4);
    DTensor<T> b(dataB, 2, 3, 4);

    /* ---- Slicing axis = 2 ---- */
    DTensor<T> aSlice(a, 2, 1, 3);
    DTensor<T> bSlice(b, 2, 1, 3);
    aSlice.reshape(2, 9, 1);
    bSlice.reshape(2, 9, 1);
    aSlice += bSlice;

    std::vector<T> dataAExpected = {
        1, 2, 3, 4, 5, 6, 41, 7, 5, 5,
        19, 17, 14, 13, 5, 11, -8, -4,
        6, 8, 8, -2, 8, 13
    };
    DTensor<T> aExpected(dataAExpected, 2, 3, 4);

    DTensor<T> err = aExpected - a;
    ASSERT_LT(err.normF(), epsilon);
}

TEST_F(TensorTest, tensorSliceAndReshape) {
    tensorSliceAndReshape<float>(PRECISION_LOW);
    tensorSliceAndReshape<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Tensor: Frobenius dot product
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorDotF(T epsilon) {
    // as vectors
    std::vector<T> dataA = TENSOR_DATA_234A;
    std::vector<T> dataB = TENSOR_DATA_234B;
    DTensor<T> vecA(dataA, dataA.size());
    DTensor<T> vecB(dataB, dataB.size());
    T dotVector = vecA.dotF(vecB);
    EXPECT_EQ(604, dotVector); // from MATLAB
    // as matrices
    DTensor<T> tenA(dataA, 2, 3, 4);
    DTensor<T> tenB(dataB, 2, 3, 4);
    T dotTensor = tenA.dotF(tenB);
    EXPECT_EQ(604, dotTensor); // from MATLAB
}

TEST_F(TensorTest, tensorDotF) {
    tensorDotF<float>(PRECISION_LOW);
    tensorDotF<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Tensor: Frobenius norm
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
void tensorSumAbs() {
    std::vector<T> data = TENSOR_DATA_234A;
    DTensor<T> tenz(data, 2, 3, 4);
    EXPECT_NEAR(112, tenz.sumAbs(), PRECISION_HIGH); // from MATLAB
}

TEST_F(TensorTest, tensorSumAbs) {
    tensorSumAbs<float>();
    tensorSumAbs<double>();
}

/* ---------------------------------------
 * Tensor: max of absolute of all elements
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorMax() {
    std::vector<T> data = TENSOR_DATA_234AMB;
    DTensor<T> tenz(data, 2, 3, 4);
    T m = tenz.maxAbs();
    EXPECT_EQ(27, m);
}

TEST_F(TensorTest, tensorMax) {
    tensorMax<float>();
    tensorMax<double>();
}

/* ---------------------------------------
 * Tensor: min of absolute of all elements
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorMin() {
    std::vector<T> data = TENSOR_DATA_234AMB;
    DTensor<T> tenz(data, 2, 3, 4);
    T m = tenz.minAbs();
    EXPECT_EQ(0, m);
}

TEST_F(TensorTest, tensorMin) {
    tensorMin<float>();
    tensorMin<double>();
}

/* ---------------------------------------
 * Tensor: right Givens rotation
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorRightGivens(T epsilon) {
    // Construct matrix A
    size_t m = 10;
    size_t n = 6;
    std::vector<T> v(m * n);
    v.reserve(m * n);
    std::iota(v.begin(), v.end(), 1);
    auto a = DTensor<T>(v, m, n, 1);

    // Apply right Givens rotation G
    size_t i_givens = 1, j_givens = 4;
    T c = 0.1;
    T minus_s = sqrt(1 - c * c);
    a.applyRightGivensRotation(i_givens, j_givens, &c, &minus_s);

    // Check the result
    for (size_t i = 0; i < m; i++) {
        EXPECT_NEAR(1 + i, a(i, 0), epsilon);
        EXPECT_NEAR(21 + i, a(i, 2), epsilon);
        EXPECT_NEAR(31 + i, a(i, 3), epsilon);
        EXPECT_NEAR((11 + i) * c - (41 + i) * (-minus_s), a(i, i_givens), epsilon);
        EXPECT_NEAR((11 + i) * (-minus_s) + (41 + i) * c, a(i, j_givens), epsilon);
    }
}

TEST_F(TensorTest, tensorRightGivens) {
    tensorRightGivens<float>(PRECISION_LOW);
    tensorRightGivens<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Tensor: left Givens rotation
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorLeftGivens(T epsilon) {
    // Construct matrix A
    size_t m = 10;
    size_t n = 6;
    std::vector<double> v(m * n);
    v.reserve(m * n);
    std::iota(v.begin(), v.end(), 1);
    auto a = DTensor<double>(v, m, n, 1);

    // Apply right Givens rotation G
    size_t i_givens = 1, j_givens = 9;
    double c = 0.1;
    double minus_s = -sqrt(1 - c * c);
    a.applyLeftGivensRotation(i_givens, j_givens, &c, &minus_s);


    // Check the result
    for (size_t j = 0; j < n; j++) {
        EXPECT_NEAR(1 + 10 * j, a(0, j), epsilon);
        for (size_t i = 2; i < m - 1; i++) {
            EXPECT_NEAR(1 + i + 10 * j, a(i, j), epsilon);
        }
        EXPECT_NEAR((2 + 10 * j) * c + (10 + 10 * j) * minus_s, a(i_givens, j), epsilon);
        EXPECT_NEAR((2 + 10 * j) * (-minus_s) + (10 + 10 * j) * c, a(j_givens, j), epsilon);
    }
}

TEST_F(TensorTest, tensorLeftGivens) {
    tensorLeftGivens<float>(1e-10);
    tensorLeftGivens<double>(1e-14);
}

/* ---------------------------------------
 * Tensor operator() to access element
 * e.g., t(2, 3, 4)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
void tensorTimesEqualsScalar() {
    std::vector<T> data = TENSOR_DATA_234A;
    std::vector<T> dataTimes3 = {
        3, 6, 9, 12, 15, 18, 21, 24, 27, 24, 21, 30, 15, 12, 9, 6, 3, -3, 12, 9, 12, 9, 12,
        24
    };
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

TEMPLATE_WITH_TYPE_T
void tensorTimesScalar() {
    std::vector<T> data = TENSOR_DATA_234A;
    std::vector<T> dataTimes3 = {
        3, 6, 9, 12, 15, 18, 21, 24, 27, 24, 21, 30, 15, 12, 9, 6, 3, -3, 12, 9, 12, 9, 12,
        24
    };
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
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

TEMPLATE_WITH_TYPE_T
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
 * Tensor: C = AB
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorAddAB() {
    std::vector<T> aData = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18
    };
    std::vector<T> bData = {
        6, 5, 4, 3, 2, 1,
        7, 6, 5, 4, 3, 2,
        1, 2, 1, 5, -6, 8
    };
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
 * Tensor: slice ptrMatrices
 * axis = 2 (matrices)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorSliceAxis2PtrMatrices() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    DTensor<T> d_A(dataA, 2, 3, 4);
    DTensor<T> d_ASlice(d_A, 2, 2, 3);
    EXPECT_TRUE(d_ASlice.ptrMatrices() == d_A.ptrMatrices() + 2);
}

TEST_F(TensorTest, tensorSliceAxis2PtrMatrices) {
    tensorSliceAxis2PtrMatrices<float>();
    tensorSliceAxis2PtrMatrices<double>();
    tensorSliceAxis2PtrMatrices<int>();
}

/* ---------------------------------------
 * Tensor: slice ptrMatrices
 * axis = 0 and 1
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorSliceAxis01PtrMatrices() {
    std::vector<T> dataA = TENSOR_DATA_234A;
    DTensor<T> d_A(dataA, 2, 3, 4);
    DTensor<T> d_ASlice0(d_A, 0, 0, 1);
    EXPECT_TRUE(!d_ASlice0.ptrMatrices());
    DTensor<T> d_ASlice1(d_A, 1, 0, 2);
    EXPECT_TRUE(!d_ASlice0.ptrMatrices());
}

TEST_F(TensorTest, tensorSliceAxis01PtrMatrices) {
    tensorSliceAxis01PtrMatrices<float>();
    tensorSliceAxis01PtrMatrices<double>();
    tensorSliceAxis01PtrMatrices<int>();
}

/* ---------------------------------------
 * Tensor: getRows
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorGetRows() {
    std::vector<T> aData = {
        10.5, 25.0, 60.0,
        -21.0, 720.0, -1.0,
        11.0, -1.0, 30.0,
        5., 6., 7.,
        8., 9., 10.,
        11., 12., 13
    };
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

    auto fullMatrix = A.getRows(0, 2, 1);
    std::vector<T> actualFull;
    fullMatrix.download(actualFull);
    EXPECT_EQ((std::vector<T>{5, 6, 7, 8, 9, 10, 11, 12, 13}), actualFull);
}

TEST_F(TensorTest, tensorGetRows) {
    tensorGetRows<float>();
    tensorGetRows<double>();
}

template<typename T>
void tensorGetRowsInto() {
    std::vector<T> data(24);
    DTensor<T> source(4,3,2);
    for (size_t count : {size_t(1),size_t(2),size_t(4)}) {
        DTensor<T> destination(count,3);
        auto *pointer = destination.raw();
        const size_t allocated = Session::getInstance().totalAllocatedBytes();
        const size_t first = count == 4 ? 0 : 1;
        for (size_t repeat=0;repeat<3;++repeat) {
            std::iota(data.begin(),data.end(),T(10*repeat));
            source.upload(data);
            source.getRowsInto(destination,first,first+count-1,repeat%2);
            Session::getInstance().synchronizeStream(source.streamIdx());
            EXPECT_EQ(pointer,destination.raw());
            EXPECT_EQ(allocated,Session::getInstance().totalAllocatedBytes());
            std::vector<T> actual;
            destination.download(actual);
            for (size_t col=0;col<3;++col) for(size_t row=0;row<count;++row)
                EXPECT_EQ(data[12*(repeat%2)+4*col+first+row],actual[count*col+row]);
        }
    }
    DTensor<T> destination(2,3), wrongRows(1,3), wrongColumns(2,2), wrongBatch(2,3,2), empty;
    EXPECT_THROW(source.getRowsInto(destination,2,1,0),std::invalid_argument);
    EXPECT_THROW(source.getRowsInto(destination,0,4,0),std::invalid_argument);
    EXPECT_THROW(source.getRowsInto(destination,0,1,2),std::invalid_argument);
    EXPECT_THROW(source.getRowsInto(wrongRows,0,1,0),std::invalid_argument);
    EXPECT_THROW(source.getRowsInto(wrongColumns,0,1,0),std::invalid_argument);
    EXPECT_THROW(source.getRowsInto(wrongBatch,0,1,0),std::invalid_argument);
    EXPECT_THROW(source.getRowsInto(empty,0,1,0),std::invalid_argument);
    DTensor<T> overlapping(source,2,0,0);
    EXPECT_THROW(source.getRowsInto(overlapping,0,3,0),std::invalid_argument);
    EXPECT_THROW(overlapping.getRowsInto(overlapping,0,3,0),std::invalid_argument);
}

TEST_F(TensorTest, getRowsIntoReusesDestination) {
    tensorGetRowsInto<float>();
    tensorGetRowsInto<double>();
    tensorGetRowsInto<int>();
}

template<typename T>
void getRowsIntoUsesSelectedStream() {
    NonblockingTestStream execution;
    DTensor<T> producer(std::vector<T>{1,2,3,4,5,6},3,2);
    DTensor<T> source(3,2,1,true), destination(2,2), captured(2,2);
    source.setStreamIdx(1);
    gpuErrChk(cudaLaunchHostFunc(execution.stream,delayTestInput,nullptr));
    gpuErrChk(cudaMemcpyAsync(source.raw(),producer.raw(),6*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    source.getRowsInto(destination,1,2,0);
    gpuErrChk(cudaMemcpyAsync(captured.raw(),destination.raw(),4*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    gpuErrChk(cudaStreamSynchronize(execution.stream));
    EXPECT_EQ(0,destination.streamIdx()); // Caller configuration is preserved.
    std::vector<T> actual;
    captured.download(actual);
    EXPECT_EQ((std::vector<T>{2,3,5,6}),actual);
}

TEST_F(TensorTest, getRowsIntoOrdersWorkOnSelectedStream) {
    getRowsIntoUsesSelectedStream<float>();
    getRowsIntoUsesSelectedStream<double>();
}


/* ---------------------------------------
 * Tensor: transpose
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
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

/* ---------------------------------------
 * Tensor: total bytes allocated
 * --------------------------------------- */
TEMPLATE_WITH_TYPE_T
void tensorBytesAllocated() {
    const size_t previouslyAllocatedBytes = Session::getInstance().totalAllocatedBytes();
    size_t m = 10, n = 10, k = 20;
    DTensor<T> A(m, n, k);
    const size_t allocatedBytes = Session::getInstance().totalAllocatedBytes();
    const size_t currentAllocatedBytes = m * n * k * sizeof(T) + k * sizeof(T *);
    const size_t expectedBytes = currentAllocatedBytes + previouslyAllocatedBytes;
    EXPECT_EQ(expectedBytes, allocatedBytes);
}

TEST_F(TensorTest, tensorBytesAllocated) {
    tensorBytesAllocated<float>();
    tensorBytesAllocated<double>();
    tensorBytesAllocated<int>();
}


/* ---------------------------------------
 * Tensor: total bytes allocated
 * --------------------------------------- */
TEMPLATE_WITH_TYPE_T
void tensorBytesAllocatedDeallocated() {
    const size_t previouslyAllocatedBytes = Session::getInstance().totalAllocatedBytes();
    size_t m = 10, n = 10, k = 20;
    auto *A = new DTensor<T>(m, n, k); // new allocation (increments session)
    const size_t allocatedBytes = Session::getInstance().totalAllocatedBytes();
    const size_t expectedBytes = previouslyAllocatedBytes + m * n * k * sizeof(T) + k * sizeof(T *);
    EXPECT_EQ(expectedBytes, allocatedBytes);
    delete A;
    EXPECT_EQ(previouslyAllocatedBytes, Session::getInstance().totalAllocatedBytes());
}

TEST_F(TensorTest, tensorBytesAllocatedDeallocated) {
    tensorBytesAllocatedDeallocated<float>();
    tensorBytesAllocatedDeallocated<double>();
    tensorBytesAllocatedDeallocated<int>();
}

/* ---------------------------------------
 * Input validation
 * --------------------------------------- */

TEST_F(TensorTest, sliceConstructorRejectsInvalidAxisAndRanges) {
    DTensor<double> tensor(2, 3, 2);

    EXPECT_THROW(DTensor<double>(tensor, 3, 0, 0), std::invalid_argument);
    EXPECT_THROW(DTensor<double>(tensor, 0, 1, 0), std::invalid_argument);
    EXPECT_THROW(DTensor<double>(tensor, 0, 0, 2), std::invalid_argument);
    EXPECT_THROW(DTensor<double>(tensor, 1, 0, 3), std::invalid_argument);
    EXPECT_THROW(DTensor<double>(tensor, 2, 0, 2), std::invalid_argument);
}

TEST_F(TensorTest, givensRotationsRejectBatches) {
    const double c = 1.0;
    const double minusS = 0.0;
    DTensor<double> batchedMatrix(2, 3, 2);

    EXPECT_THROW(batchedMatrix.applyRightGivensRotation(0, 1, &c, &minusS), std::invalid_argument);
    EXPECT_THROW(batchedMatrix.applyLeftGivensRotation(0, 1, &c, &minusS), std::invalid_argument);
}

TEST_F(TensorTest, tensorOperationsRejectIncompatibleDimensions) {
    DTensor<double> twoByTwo(2, 2);
    DTensor<double> twoByThree(2, 3);
    DTensor<double> tooSmall(3);

    EXPECT_THROW(twoByTwo.dotF(twoByThree), std::invalid_argument);
    EXPECT_THROW(twoByTwo.deviceCopyTo(tooSmall), std::invalid_argument);
}

TEST_F(TensorTest, tensorSetupRejectsInvalidDimensionsAndRanges) {
    EXPECT_THROW(DTensor<double>(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(DTensor<double>(static_cast<size_t>(INT_MAX) + 1, 1, 1), std::invalid_argument);
    EXPECT_THROW(DTensor<double>(static_cast<size_t>(INT_MAX), static_cast<size_t>(INT_MAX), 2), std::invalid_argument);
    EXPECT_THROW(DTensor<double>((std::vector<double>{1}), 2), std::invalid_argument);
    EXPECT_THROW(DTensor<double>::createRandomTensor(2, 1, 1, 2.0, 1.0), std::invalid_argument);
    DTensor<double> tensor(2, 3, 2);
    EXPECT_THROW(tensor.reshape(static_cast<size_t>(INT_MAX), static_cast<size_t>(INT_MAX), 2), std::invalid_argument);
    EXPECT_THROW(tensor.getRows(1, 0, 0), std::invalid_argument);
    EXPECT_THROW(tensor.getRows(0, 2, 0), std::invalid_argument);
    EXPECT_THROW(tensor.getRows(0, 1, 2), std::invalid_argument);
    EXPECT_THROW(tensor(2, 0, 0), std::invalid_argument);
    EXPECT_THROW(tensor(0, 3, 0), std::invalid_argument);
    EXPECT_THROW(tensor(0, 0, 2), std::invalid_argument);
    EXPECT_THROW(tensor.upload(std::vector<double>(12), static_cast<StorageMode>(99)), std::invalid_argument);
}

TEST_F(TensorTest, factoriserSetupRejectsEmptyAndIncompatibleMatrices) {
    DTensor<double> empty;
    DTensor<double> wide(2, 3);
    DTensor<double> nonsquare(3, 2, 2);
    EXPECT_THROW(Svd<double> svd(empty), std::invalid_argument);
    EXPECT_THROW(CholeskyFactoriser<double> cholesky(empty), std::invalid_argument);
    EXPECT_THROW(QRFactoriser<double> qr(wide), std::invalid_argument);
    EXPECT_THROW(CholeskyBatchFactoriser<double> cholesky(nonsquare), std::invalid_argument);
    EXPECT_THROW(GivensAnnihilator<double> givens(empty), std::invalid_argument);
}

TEST_F(TensorTest, fileParsingRejectsMalformedDataAndHeaders) {
    const auto path = std::filesystem::path(testing::TempDir()) / "gputils_invalid_input.dtensor";
    for (const auto &content : {"2\n1\n1\n1\n", "2\n1\n1\n1\ninvalid\n", "1\n1\n1\n1\n2\n", "2\n"}) {
        { std::ofstream file(path); ASSERT_TRUE(file.is_open()); file << content; }
        EXPECT_THROW(DTensor<double>::parseFromFile(path.string()), std::invalid_argument);
    }
    std::filesystem::remove(path);
    const auto binaryPath = std::filesystem::path(testing::TempDir()) / "gputils_invalid_header.bt";
    { std::ofstream file(binaryPath, std::ios::binary); ASSERT_TRUE(file.is_open()); file.put('\1'); }
    EXPECT_THROW(DTensor<double>::parseFromFile(binaryPath.string()), std::invalid_argument);
    std::filesystem::remove(binaryPath);
}

TEST_F(TensorTest, leastSquaresRejectsIncompatibleDimensions) {
    DTensor<double> tallMatrix(3, 2, 2);
    DTensor<double> wrongRows(2, 1, 2);
    DTensor<double> wrongColumns(3, 2, 2);
    DTensor<double> wrongBatchSize(3, 1, 1);
    DTensor<double> wideMatrix(2, 3, 2);
    DTensor<double> compatibleRhs(2, 1, 2);

    EXPECT_THROW(tallMatrix.leastSquaresBatched(wrongRows), std::invalid_argument);
    EXPECT_THROW(tallMatrix.leastSquaresBatched(wrongColumns), std::invalid_argument);
    EXPECT_THROW(tallMatrix.leastSquaresBatched(wrongBatchSize), std::invalid_argument);
    EXPECT_THROW(wideMatrix.leastSquaresBatched(compatibleRhs), std::invalid_argument);
}

/* ---------------------------------------
 * Matrix multiplication coefficients
 * --------------------------------------- */

TEST_F(TensorTest, tensorAddABUsesAlphaBetaForASingleMatrix) {
    DTensor<double> A({1, 2, 3, 4}, 2, 2);
    DTensor<double> B({5, 6, 7, 8}, 2, 2);
    DTensor<double> C({1, 2, 3, 4}, 2, 2);

    C.addAB(A, B, 2.0, -1.0);

    std::vector<double> result;
    C.download(result);
    EXPECT_EQ((std::vector<double>{45, 66, 59, 88}), result);
}

/* ---------------------------------------
 * Malformed input files
 * --------------------------------------- */

TEST_F(TensorTest, parseTensorRejectsTruncatedBinaryFile) {
    const auto path = std::filesystem::path(testing::TempDir()) / "gputils_truncated_fixture.bt";
    const uint64_t rows = 2, cols = 2, mats = 1;
    const double onlyOneElement = 1.0;
    {
        std::ofstream file(path, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        file.write(reinterpret_cast<const char *>(&mats), sizeof(mats));
        file.write(reinterpret_cast<const char *>(&onlyOneElement), sizeof(onlyOneElement));
    }

    EXPECT_THROW(DTensor<double>::parseFromFile(path.string()), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST_F(TensorTest, parseTensorRejectsMalformedTextFile) {
    const auto path = std::filesystem::path(testing::TempDir()) / "gputils_malformed_fixture.dtensor";
    {
        std::ofstream file(path);
        ASSERT_TRUE(file.is_open());
        file << "not-a-dimension\n2\n1\n1\n2\n3\n4\n";
    }

    EXPECT_THROW(DTensor<double>::parseFromFile(path.string()), std::invalid_argument);
    std::filesystem::remove(path);
}



/* ================================================================================================
 *  LEAST SQUARES TESTS
 * ================================================================================================ */
class LeastSquaresTest : public testing::Test {
protected:
    LeastSquaresTest() {
    }

    virtual ~LeastSquaresTest() {
    }
};

/* ---------------------------------------
 * Tensor: Least squares
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
void tensorLeastSquares1(T epsilon) {
    // TODO test with tall matrices too
    std::vector<T> aData = {
        1, 2,
        3, 4,
        7, 8,
        9, 10,
        6, 8,
        -9, 20
    };
    std::vector<T> bData = {1, 1, -1, 2, 30, -80};
    DTensor<T> A0(aData, 2, 2, 3);
    DTensor<T> A(A0);
    DTensor<T> B(bData, 2, 1, 3);
    DTensor<T> sol(B);
    A0.leastSquaresBatched(sol);
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

template<typename T>
void leastSquaresReusesStatusBuffer(T epsilon) {
    auto &session = Session::getInstance();
    for (size_t rows : {size_t(2), size_t(3)}) {
        const std::vector<T> matrix = rows == 2
            ? std::vector<T>{1,0,0,1, 2,0,0,2}
            : std::vector<T>{1,0,1,0,1,1, 2,0,2,0,2,2};
        DTensor<T> A(matrix, rows, 2, 2);
        DTensor<T> B(rows, 1, 2);
        DTensor<int> status(3); // Excess capacity is allowed and left untouched.
        DTensor<int> tooSmall(1);
        DTensor<int> empty;
        EXPECT_THROW(A.leastSquaresBatched(B, tooSmall), std::invalid_argument);
        EXPECT_THROW(A.leastSquaresBatched(B, empty), std::invalid_argument);
        const auto *statusPointer = status.raw();
        const size_t allocated = session.totalAllocatedBytes();
        for (int repeat = 0; repeat < 3; ++repeat) {
            T x = T(2+repeat), y = T(3-repeat);
            std::vector<T> rhs = rows == 2
                ? std::vector<T>{x,y,2*x,2*y}
                : std::vector<T>{x,y,x+y,2*x,2*y,2*(x+y)};
            A.upload(matrix); // gels overwrites its inputs, so restore each run.
            B.upload(rhs);
            status.upload(std::vector<int>{-1,-1,123});
            A.leastSquaresBatched(B, status);
            session.synchronizeStream(A.streamIdx());
            EXPECT_EQ(allocated, session.totalAllocatedBytes());
            EXPECT_EQ(statusPointer, status.raw());
            std::vector<T> solution;
            B.download(solution);
            for (size_t batch = 0; batch < 2; ++batch) {
                EXPECT_NEAR(x, solution[batch*rows], epsilon);
                EXPECT_NEAR(y, solution[batch*rows+1], epsilon);
            }
            std::vector<int> codes;
            status.download(codes);
            EXPECT_EQ((std::vector<int>{0,0,123}), codes);
        }
    }
}

TEST_F(LeastSquaresTest, reusableStatusBuffer) {
    leastSquaresReusesStatusBuffer<float>(PRECISION_LOW);
    leastSquaresReusesStatusBuffer<double>(PRECISION_HIGH);
}


/* ================================================================================================
 *  SVD TESTS
 * ================================================================================================ */
class SvdTest : public testing::Test {
protected:
    SvdTest() {
    }

    virtual ~SvdTest() {
    }
};

/* ---------------------------------------
 * Computation of singular values
 * and matrix rank
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void singularValuesComputation(float epsilon) {
    std::vector<T> bData = {
        1, 6, 6, 6, 6, 6, 6, 6,
        2, 7, 7, 7, 7, 7, 7, 7,
        3, 8, 8, 8, 8, 8, 8, 8,
    };
    DTensor<T> B(bData, 8, 3);
    Svd<T> svd(B, true, false);
    svd.factorise();
    for (size_t i = 0; i < B.numMats(); i++) {
        EXPECT_EQ(0, svd.info()(i));
    }
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

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void singularValuesMemory(float epsilon) {
    std::vector<T> bData = {
        1, 6, 6, 6, 6, 6, 6, 6,
        2, 7, 7, 7, 7, 7, 7, 7,
        3, 8, 8, 8, 8, 8, 8, 8,
    };
    DTensor<T> B(bData, 8, 3);
    Svd<T> svd(B, true, false);
    svd.factorise();
    EXPECT_EQ(0, svd.info()(0));
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
TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void singularValuesMultipleMatrices(float epsilon) {
    std::vector<T> aData = {1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1};
    DTensor<T> A(aData, 3, 2, 3);
    Svd<T> svd(A, true); // do compute U (A will be destroyed)
    svd.factorise();
    DTensor<T> const &S = svd.singularValues();
    DTensor<T> const &V = svd.rightSingularVectors();
    auto Uopt = svd.leftSingularVectors();
    auto U = Uopt.value();
    std::vector<T> expected_v = {
        -0.386317703118612, -0.922365780077058, -0.922365780077058, 0.386317703118612,
        -0.447213595499958, -0.894427190999916, 0.894427190999916, -0.447213595499958,
        0, -1, 1, 0
    };
    std::vector<T> actual_v(12);
    V.download(actual_v);
    for (size_t i = 0; i < 4; i++)
        EXPECT_NEAR(expected_v[i], actual_v[i], epsilon);
    std::vector<T> expected_s = {9.508032000695726, 0.772869635673484, 3.872983346207417, 0, 1, 0};
    std::vector<T> actual_s(6);
    S.download(actual_s);
    for (size_t i = 0; i < 6; i++)
        EXPECT_NEAR(expected_s[i], actual_s[i], epsilon);
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
    for (size_t i = 0; i < 27; i++)
        EXPECT_NEAR(expected_u[i], actual_u[i], epsilon);
}

TEST_F(SvdTest, singularValuesMultipleMatrices) {
    singularValuesMultipleMatrices<float>(10 * PRECISION_LOW); // SVD with float performs quite poorly
    singularValuesMultipleMatrices<double>(PRECISION_HIGH);
}


/* ---------------------------------------
 * SVD for rank computation of multiple
 * matrices
 * --------------------------------------- */
TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void singularValuesRankMultipleMatrices(float epsilon) {
    std::vector<T> aData = {
        1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 0,
        1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
        1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12
    };
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
    CholeskyTest() {
    }

    virtual ~CholeskyTest() {
    }
};


/* ---------------------------------------
 * Cholesky factorisation
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void choleskyFactorisation(T epsilon) {
    std::vector<T> aData = {
        10.0, 2.0, 3.0,
        2.0, 20.0, -1.0,
        3.0, -1.0, 30.0
    };
    DTensor<T> A(aData, 3, 3, 1);
    CholeskyFactoriser<T> chol(A);
    chol.factorise();
    EXPECT_NEAR(3.162277660168380, A(0, 0), epsilon);
    EXPECT_NEAR(-0.361403161162101, A(2, 1), epsilon);
    EXPECT_NEAR(5.382321781081287, A(2, 2), epsilon);
    EXPECT_EQ(0, chol.info()(0));
}

TEST_F(CholeskyTest, choleskyFactorisation) {
    choleskyFactorisation<float>(PRECISION_LOW);
    choleskyFactorisation<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Cholesky factorisation: solve system
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void choleskyFactorisationSolution(T epsilon) {
    std::vector<T> aData = {
        10.0, 2.0, 3.0,
        2.0, 20.0, -1.0,
        3.0, -1.0, 30.0
    };
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
    for (size_t i = 0; i < 3; i++)
        EXPECT_NEAR(expected[i], actual[i], epsilon);

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

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void choleskyBatchFactorisation(T epsilon) {
    std::vector<T> aData = {
        10.0, 2.0, 3.0,
        2.0, 20.0, -1.0,
        3.0, -1.0, 30.0
    };
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

    EXPECT_EQ(0, chol.info()(0));
}

TEST_F(CholeskyTest, choleskyBatchFactorisation) {
    choleskyBatchFactorisation<float>(PRECISION_LOW);
    choleskyBatchFactorisation<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Batched Cholesky solve
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void choleskyBatchFactorSolve(T epsilon) {
    std::vector<T> aData = {
        10.0, 2.0, 3.0,
        2.0, 20.0, -1.0,
        3.0, -1.0, 30.0
    };
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
    for (size_t i = 0; i < 3; i++)
        EXPECT_NEAR(expected[i], actual[i], epsilon); // 0
    for (size_t i = 0; i < 3; i++)
        EXPECT_NEAR(expected[i], actual[i + 3], epsilon); // 1
    DTensor<T> error = A * sol;
    error -= rhs;
    EXPECT_TRUE(error.normF() < epsilon);
    EXPECT_EQ(0, chol.info()(0));
}

TEST_F(CholeskyTest, choleskyBatchFactorSolve) {
    choleskyBatchFactorSolve<float>(PRECISION_LOW);
    choleskyBatchFactorSolve<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * Batched Cholesky solve (factor provided)
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void choleskyBatchSolve(T epsilon) {
    std::vector<T> aData = {
        10.0, 2.0, 3.0,
        2.0, 20.0, -1.0,
        3.0, -1.0, 30.0
    };
    DTensor<T> A(3, 3, 2);
    DTensor<T> A0(A, 2, 0, 0);
    DTensor<T> A1(A, 2, 1, 1);
    A0.upload(aData);
    A1.upload(aData);
    std::vector<T> lowData = {
        3.162277660168380, 0, 0,
        0.632455532033676, 4.427188724235731, 0,
        0.948683298050514, -0.361403161162101, 5.382321781081287
    }; // from matlab
    DTensor<T> low(3, 3, 2);
    DTensor<T> low0(low, 2, 0, 0);
    DTensor<T> low1(low, 2, 1, 1);
    low0.upload(lowData, rowMajor);
    low1.upload(lowData, rowMajor);
    DTensor<T> L(low);
    CholeskyBatchFactoriser<T> chol(L, true);
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
    for (size_t i = 0; i < 3; i++)
        EXPECT_NEAR(expected[i], actual[i], epsilon); // 0
    for (size_t i = 0; i < 3; i++)
        EXPECT_NEAR(expected[i], actual[i + 3], epsilon); // 1
    DTensor<T> error = A * sol;
    error -= rhs;
    EXPECT_TRUE(error.normF() < epsilon);
    EXPECT_EQ(0, chol.info()(0));
}

TEST_F(CholeskyTest, choleskyBatchSolve) {
    choleskyBatchSolve<float>(PRECISION_LOW);
    choleskyBatchSolve<double>(PRECISION_HIGH);
}


/* ================================================================================================
 *  QR TESTS
 * ================================================================================================ */
class QRTest : public testing::Test {
protected:
    QRTest() {
    }

    virtual ~QRTest() {
    }
};


/* ---------------------------------------
 * QR factorisation
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void qrFactorisation(T epsilon) {
    size_t nR = 4;
    size_t nC = 3;
    DTensor<T> temp(nR, nC);
    DTensor<T> A = DTensor<T>::createRandomTensor(nR, nC, 1, -100, 100);
    QRFactoriser<T> qr(temp);
    A.deviceCopyTo(temp);
    qr.factorise();
    EXPECT_EQ(0, qr.info()(0));
    DTensor<T> Q(nR, nC);
    DTensor<T> R(nC, nC, 1, true);
    DTensor<T> QR(nR, nC);
    qr.getQR(Q, R);
    EXPECT_EQ(0, qr.info()(0));
    QR.addAB(Q, R);
    QR -= A;
    T nrm = QR.normF();
    EXPECT_NEAR(nrm, 0., epsilon);
}

TEST_F(QRTest, qrFactorisation) {
    qrFactorisation<float>(PRECISION_LOW);
    qrFactorisation<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * QR factorisation
 * - tall and skinny matrix
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void qrFactorisationTall(T epsilon) {
    size_t nR = 20;
    size_t nC = 3;
    DTensor<T> temp(nR, nC);
    DTensor<T> A = DTensor<T>::createRandomTensor(nR, nC, 1, -100, 100);
    QRFactoriser<T> qr(temp);
    A.deviceCopyTo(temp);
    qr.factorise();
    EXPECT_EQ(0, qr.info()(0));
    DTensor<T> Q(nR, nC);
    DTensor<T> R(nC, nC, 1, true);
    DTensor<T> QR(nR, nC);
    qr.getQR(Q, R);
    EXPECT_EQ(0, qr.info()(0));
    QR.addAB(Q, R);
    QR -= A;
    T nrm = QR.normF();
    EXPECT_NEAR(nrm, 0., epsilon);
}

TEST_F(QRTest, qrFactorisationTall) {
    qrFactorisationTall<float>(PRECISION_LOW);
    qrFactorisationTall<double>(PRECISION_HIGH);
}

/* ---------------------------------------
 * QR factorisation: solve least squares
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void qrLeastSquares(T epsilon) {
    size_t nR = 4;
    size_t nC = 3;
    DTensor<T> temp(nR, nC);
    std::vector<T> vecA = {
        85.5638, -59.4001, -80.1992,
        99.9464, 5.51393, 5.17935,
        6.87488, -26.7536, 36.0914,
        -44.3857, -32.1268, 54.8915
    }; // Random matrix
    std::vector<T> vecB = {
        -23.3585,
        -48.5744,
        43.4229,
        -56.5081
    }; // Random vector
    DTensor<T> A(vecA, nR, nC, 1, rowMajor);
    DTensor<T> b(vecB, nR);
    DTensor<T> xFull(nR);
    DTensor<T> x(xFull, 0, 0, nC - 1);
    DTensor<T> Ax(nR);
    QRFactoriser<T> qr(temp);
    A.deviceCopyTo(temp);
    qr.factorise();
    EXPECT_EQ(0, qr.info()(0));
    b.deviceCopyTo(xFull);
    qr.leastSquares(xFull);
    EXPECT_EQ(0, qr.info()(0));
    Ax.addAB(A, x);
    Ax -= b;
    T nrm = Ax.normF();
    EXPECT_NEAR(nrm, 80.003169364198072, epsilon); // From MatLab
}

TEST_F(QRTest, qrLeastSquares) {
    qrLeastSquares<float>(PRECISION_LOW);
    qrLeastSquares<double>(PRECISION_HIGH);
}


/* ================================================================================================
 *  NULLSPACE TESTS
 * ================================================================================================ */
class NullspaceTest : public testing::Test {
protected:
    NullspaceTest() {
    }

    virtual ~NullspaceTest() {
    }
};


/* ---------------------------------------
 * Basic nullspace test
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void computeNullspaceTensor(T epsilon) {
    std::vector<T> aData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9,
        1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
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

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void computeNullspaceTrivial(T epsilon) {
    std::vector<T> data{
        4, 5, 7,
        4, 1, 8,
        4, 5, 0,
        1, 1, 1,
        5, 6, 7,
        9, 0, 3
    };
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

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void projectOnNullspaceTensor(T epsilon) {
    // offline
    size_t m = 3;
    size_t n = 7;
    std::vector<T> mat{
        1, -2, 3, 4, -1, -1, -1,
        1, 2, -3, 4, -1, -1, -1,
        -1, 3, 5, -7, -1, -1, -1
    };
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


/* ================================================================================================
 *  GIVENSANNIHILATOR TESTS
 * ================================================================================================ */
class GivensAnnihilatorTest : public testing::Test {
protected:
    GivensAnnihilatorTest() {
    }

    virtual ~GivensAnnihilatorTest() {
    }
};


/* ---------------------------------------
 * GivensAnnihilator works
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void givensAnnihilateElement(T epsilon) {
    size_t m = 10;
    size_t n = 6;
    std::vector<T> v(m * n);
    v.reserve(m * n);
    std::iota(v.begin(), v.end(), 1);

    auto a = DTensor<T>(v, m, n, 1);
    auto ga = GivensAnnihilator<T>(a);
    size_t i = 0;
    for (size_t k = 1; k < m; k++) {
        for (size_t j = 0; j < n; j++) {
            ga.annihilate(i, k, j);
            EXPECT_NEAR(0.0, a(k, j), epsilon);
        }
    }
}

TEST_F(GivensAnnihilatorTest, givensAnnihilateElement) {
    givensAnnihilateElement<float>(PRECISION_LOW);
    givensAnnihilateElement<double>(PRECISION_HIGH);
}


/* ---------------------------------------
 * GivensAnnihilator: correctness
 * --------------------------------------- */

TEMPLATE_WITH_TYPE_T
TEMPLATE_CONSTRAINT_REQUIRES_FPX
void givensAnnihilateCorrectness(T epsilon) {
    size_t m = 10, n = 6;
    std::vector<double> v(m * n);
    v.reserve(m * n);
    std::iota(v.begin(), v.end(), 1);
    DTensor<double> a = DTensor<double>(v, m, n);

    auto ga = GivensAnnihilator<double>(a);
    ga.annihilate(0, 1, 2);

    EXPECT_NEAR(0.0, a(1, 2), epsilon);
    EXPECT_NEAR(2.137186834969645, a(0, 0), epsilon);
    EXPECT_NEAR(44.552125559751822, a(0, 3), epsilon);
    EXPECT_NEAR(-0.328797974610715, a(1, 3), epsilon);
}

TEST_F(GivensAnnihilatorTest, givensAnnihilateCorrectness) {
    givensAnnihilateCorrectness<double>(1e-14);
    givensAnnihilateCorrectness<float>(1e-12);
}
