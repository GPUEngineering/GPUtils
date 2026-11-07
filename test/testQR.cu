#include <gtest/gtest.h>
#include "../include/tensor.cuh"
#include <chrono>
#include <thread>
#include <utility>

namespace {
// Use the same handle index as the other tests, but remove any implicit
// ordering with the legacy default stream. Restore both handles on exit.
class QRTestStream {
    cudaStream_t m_oldBlas, m_oldSolver;
public:
    cudaStream_t stream;
    QRTestStream() {
        auto &session = Session::getInstance();
        gpuErrChk(cublasGetStream(session.cuBlasHandle(1), &m_oldBlas));
        gpuErrChk(cusolverDnGetStream(session.cuSolverHandle(1), &m_oldSolver));
        gpuErrChk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        gpuErrChk(cublasSetStream(session.cuBlasHandle(1), stream));
        gpuErrChk(cusolverDnSetStream(session.cuSolverHandle(1), stream));
    }
    ~QRTestStream() {
        gpuErrChk(cudaStreamSynchronize(stream));
        auto &session = Session::getInstance();
        gpuErrChk(cublasSetStream(session.cuBlasHandle(1), m_oldBlas));
        gpuErrChk(cusolverDnSetStream(session.cuSolverHandle(1), m_oldSolver));
        gpuErrChk(cudaStreamDestroy(stream));
    }
};

template<typename T>
class QRComponentsTest : public testing::Test {};
using QRPrecisions = testing::Types<float, double>;
TYPED_TEST_SUITE(QRComponentsTest, QRPrecisions);

template<typename T>
double tolerance() { return std::is_same_v<T, float> ? 2e-5 : 2e-12; }

template<typename T>
std::vector<T> matrixData(size_t m, size_t n, int pass = 0) {
    std::vector<T> a(m * n);
    for (size_t c = 0; c < n; ++c)
        for (size_t r = 0; r < m; ++r)
            a[r + c * m] = T(std::sin(0.17 * (r + 1) * (c + 1) + pass) + (r == c ? 3 : 0));
    return a;
}

// Verify identities independently on the CPU, including the full-vector norm
// and thin-Q^T rhs. Q's orthogonal complement need not be explicitly constructed.
template<typename T>
void checkComponents(const std::vector<T> &a, const std::vector<T> &b,
                     const DTensor<T> &Q, const DTensor<T> &R, const DTensor<T> &transformed) {
    const size_t m = Q.numRows(), n = Q.numCols();
    std::vector<T> q, r, qt;
    Q.download(q); R.download(r); transformed.download(qt);
    const double eps = tolerance<T>();
    for (size_t c = 0; c < n; ++c) {
        for (size_t row = 0; row < m; ++row) {
            double reconstructed = 0;
            for (size_t k = 0; k < n; ++k) reconstructed += double(q[row + k * m]) * r[k + c * n];
            EXPECT_NEAR(a[row + c * m], reconstructed, eps * 10);
        }
        for (size_t j = 0; j < n; ++j) {
            double product = 0;
            for (size_t row = 0; row < m; ++row) product += double(q[row + c * m]) * q[row + j * m];
            EXPECT_NEAR(c == j ? 1 : 0, product, eps);
            if (j > c) EXPECT_EQ(T(0), r[j + c * n]);
        }
        double product = 0;
        for (size_t row = 0; row < m; ++row) product += double(q[row + c * m]) * b[row];
        EXPECT_NEAR(product, qt[c], eps * 10);
    }
    double before = 0, after = 0;
    for (size_t row = 0; row < m; ++row) {
        before += double(b[row]) * b[row];
        after += double(qt[row]) * qt[row];
    }
    EXPECT_NEAR(before, after, eps * std::max(1.0, before));
}

TYPED_TEST(QRComponentsTest, RepeatedComponentsPreserveFactorsAndStorage) {
    using T = TypeParam;
    const std::pair<size_t, size_t> shapes[] = {{1,1}, {5,5}, {37,4}, {4097,3}};
    for (const auto &shape : shapes) {
        const auto [m,n] = shape;
        SCOPED_TRACE(::testing::Message() << m << 'x' << n);
        DTensor<T> A(m,n), Q(m,n), R(n,n), rhs(m);
        A.setStreamIdx(1); Q.setStreamIdx(1); R.setStreamIdx(1); rhs.setStreamIdx(1);
        QRFactoriser<T> qr(A);
        const auto bytes = Session::getInstance().totalAllocatedBytes();
        const auto qPointer = Q.raw(), rPointer = R.raw(), bPointer = rhs.raw();
        for (int pass = 0; pass < 3; ++pass) {
            const auto a = matrixData<T>(m,n,pass);
            std::vector<T> b(m);
            for (size_t row = 0; row < m; ++row) b[row] = T(std::cos(0.23 * row + pass));
            A.upload(a); rhs.upload(b);
            R.upload(std::vector<T>(n*n,T(-93))); // Extraction must write every entry.
            qr.factorise();
            std::vector<T> savedFactors;
            A.download(savedFactors);
            qr.extractR(R); qr.applyQt(rhs); qr.formThinQ(Q);
            Session::getInstance().synchronizeStream(1);
            ASSERT_EQ(0, qr.info()(0));
            checkComponents(a,b,Q,R,rhs);
            std::vector<T> factorsAfter;
            A.download(factorsAfter);
            EXPECT_EQ(savedFactors, factorsAfter);
            // Outputs can be regenerated without destroying the reflectors.
            rhs.upload(b);
            qr.formThinQ(Q); qr.applyQt(rhs); qr.extractR(R);
            Session::getInstance().synchronizeStream(1);
            checkComponents(a,b,Q,R,rhs);
            EXPECT_EQ(bytes, Session::getInstance().totalAllocatedBytes());
            EXPECT_EQ(qPointer, Q.raw()); EXPECT_EQ(rPointer, R.raw()); EXPECT_EQ(bPointer, rhs.raw());
            EXPECT_EQ(1, Q.streamIdx()); EXPECT_EQ(1, R.streamIdx()); EXPECT_EQ(1, rhs.streamIdx());
        }
    }
}

TYPED_TEST(QRComponentsTest, RankDeficientAndZeroMatrices) {
    using T = TypeParam;
    constexpr size_t m = 11, n = 3;
    for (bool zero : {false,true}) {
        auto a = matrixData<T>(m,n);
        for (size_t row = 0; row < m; ++row) a[row + 2*m] = a[row];
        if (zero) std::fill(a.begin(),a.end(),T(0));
        std::vector<T> b(m,T(2));
        DTensor<T> A(a,m,n), Q(m,n), R(n,n), rhs(b,m);
        QRFactoriser<T> qr(A);
        qr.factorise(); qr.formThinQ(Q); qr.extractR(R); qr.applyQt(rhs);
        Session::getInstance().synchronizeStream();
        ASSERT_EQ(0, qr.info()(0));
        checkComponents(a,b,Q,R,rhs);
        EXPECT_NEAR(0, R(n-1,n-1), tolerance<T>()*10);
    }
}

TYPED_TEST(QRComponentsTest, ColumnSpaceRhsHasZeroOrthogonalComplement) {
    using T = TypeParam;
    constexpr size_t m = 13, n = 3;
    const auto a = matrixData<T>(m,n);
    const std::vector<T> x = {T(2),T(-3),T(0.5)};
    std::vector<T> b(m,T(0));
    for (size_t row = 0; row < m; ++row)
        for (size_t c = 0; c < n; ++c) b[row] += a[row+c*m]*x[c];
    DTensor<T> A(a,m,n), R(n,n), rhs(b,m);
    QRFactoriser<T> qr(A);
    qr.factorise(); qr.applyQt(rhs); qr.extractR(R);
    Session::getInstance().synchronizeStream();
    std::vector<T> r, transformed;
    R.download(r); rhs.download(transformed);
    for (size_t row = 0; row < m; ++row) {
        double expected = 0;
        if (row < n) for (size_t c = row; c < n; ++c) expected += double(r[row+c*n])*x[c];
        EXPECT_NEAR(expected, transformed[row], tolerance<T>()*20);
    }
}

TYPED_TEST(QRComponentsTest, RejectsInvalidUseAndOverlappingStorage) {
    using T = TypeParam;
    DTensor<T> backing(3,4), A(backing,1,0,2), partial(backing,1,1,3);
    A.upload(matrixData<T>(3,3));
    DTensor<T> Q(3,3), R(3,3), rhs(3), badRows(2), badCols(3,2), badBatch(3,1,2);
    QRFactoriser<T> qr(A);
    EXPECT_THROW(qr.applyQt(rhs),std::logic_error);
    EXPECT_THROW(qr.extractR(R),std::logic_error);
    EXPECT_THROW(qr.formThinQ(Q),std::logic_error);
    EXPECT_THROW(qr.getQR(Q,R),std::logic_error);
    EXPECT_THROW(qr.leastSquares(rhs),std::logic_error);
    qr.factorise();
    EXPECT_THROW(qr.applyQt(badRows),std::invalid_argument);
    EXPECT_THROW(qr.applyQt(badCols),std::invalid_argument);
    EXPECT_THROW(qr.applyQt(badBatch),std::invalid_argument);
    EXPECT_THROW(qr.leastSquares(badCols),std::invalid_argument);
    EXPECT_THROW(qr.formThinQ(badCols),std::invalid_argument);
    EXPECT_THROW(qr.extractR(badCols),std::invalid_argument);
    DTensor<T> column(A,1,0,0);
    EXPECT_THROW(qr.applyQt(column),std::invalid_argument);
    EXPECT_THROW(qr.leastSquares(column),std::invalid_argument);
    EXPECT_THROW(qr.formThinQ(A),std::invalid_argument);
    EXPECT_THROW(qr.formThinQ(partial),std::invalid_argument);
    EXPECT_THROW(qr.extractR(A),std::invalid_argument);
    EXPECT_THROW(qr.extractR(partial),std::invalid_argument);
    EXPECT_THROW(qr.getQR(Q,Q),std::invalid_argument);
    EXPECT_THROW(qr.getQR(A,R),std::invalid_argument);
    EXPECT_THROW(qr.getQR(Q,A),std::invalid_argument);
    DTensor<T> outputs(3,4), first(outputs,1,0,2), second(outputs,1,1,3);
    EXPECT_THROW(qr.getQR(first,second),std::invalid_argument);
    // A rejected call must not damage the valid factorisation.
    qr.getQR(Q,R);
    EXPECT_EQ(0,qr.info()(0));
}

TYPED_TEST(QRComponentsTest, RejectsChangedBoundMatrix) {
    using T = TypeParam;
    DTensor<T> A(matrixData<T>(6,3),6,3), rhs(6);
    QRFactoriser<T> qr(A);
    qr.factorise();
    Session::getInstance().synchronizeStream();
    A.reshape(9,2);
    EXPECT_THROW(qr.factorise(),std::invalid_argument);
    EXPECT_THROW(qr.applyQt(rhs),std::invalid_argument);
    A.reshape(6,3);
    A.setStreamIdx(1);
    EXPECT_THROW(qr.factorise(),std::invalid_argument);
    A.setStreamIdx(0);
    // Hold the original allocation so the replacement cannot reuse its address.
    DTensor<T> old(std::move(A));
    A = DTensor<T>(6,3);
    EXPECT_THROW(qr.factorise(),std::invalid_argument);
}

static void CUDART_CB delayQRInput(void *) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

TYPED_TEST(QRComponentsTest, OrdersKernelsAndSolverOnTheBoundNonblockingStream) {
    using T = TypeParam;
    QRTestStream execution;
    const auto a = matrixData<T>(7,3);
    const std::vector<T> b = {1,2,3,4,5,6,7};
    DTensor<T> source(a,7,3), sourceB(b,7), A(7,3), rhs(7), Q(7,3), R(3,3);
    DTensor<T> observedQ(7,3), observedR(3,3), observedB(7);
    A.setStreamIdx(1); // Deliberately leave outputs' metadata on stream 0.
    QRFactoriser<T> qr(A);
    gpuErrChk(cudaLaunchHostFunc(execution.stream,delayQRInput,nullptr));
    gpuErrChk(cudaMemcpyAsync(A.raw(),source.raw(),A.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    gpuErrChk(cudaMemcpyAsync(rhs.raw(),sourceB.raw(),rhs.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    qr.factorise(); qr.extractR(R); qr.formThinQ(Q); qr.applyQt(rhs);
    // Same-stream consumers must observe every output, including custom kernels.
    gpuErrChk(cudaMemcpyAsync(observedQ.raw(),Q.raw(),Q.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    gpuErrChk(cudaMemcpyAsync(observedR.raw(),R.raw(),R.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    gpuErrChk(cudaMemcpyAsync(observedB.raw(),rhs.raw(),rhs.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
    gpuErrChk(cudaStreamSynchronize(execution.stream));
    EXPECT_EQ(0,qr.info()(0));
    checkComponents(a,b,observedQ,observedR,observedB);
    EXPECT_EQ(0,Q.streamIdx()); EXPECT_EQ(0,R.streamIdx()); EXPECT_EQ(0,rhs.streamIdx());
}

TYPED_TEST(QRComponentsTest, CapturesAndReplaysWithChangedInputAndStableStorage) {
    using T = TypeParam;
    QRTestStream execution;
    constexpr size_t m = 17, n = 3;
    DTensor<T> source(matrixData<T>(m,n),m,n), sourceB(std::vector<T>(m,T(1)),m);
    DTensor<T> A(m,n), rhs(m), Q(m,n), R(n,n);
    A.setStreamIdx(1); source.setStreamIdx(1); sourceB.setStreamIdx(1);
    QRFactoriser<T> qr(A);
    auto enqueue = [&] {
        gpuErrChk(cudaMemcpyAsync(A.raw(),source.raw(),A.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
        gpuErrChk(cudaMemcpyAsync(rhs.raw(),sourceB.raw(),rhs.numEl()*sizeof(T),cudaMemcpyDeviceToDevice,execution.stream));
        qr.factorise(); qr.formThinQ(Q); qr.extractR(R); qr.applyQt(rhs);
    };
    enqueue(); // Warm up the library before capture.
    gpuErrChk(cudaStreamSynchronize(execution.stream));
    const auto bytes = Session::getInstance().totalAllocatedBytes();
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    gpuErrChk(cudaStreamBeginCapture(execution.stream,cudaStreamCaptureModeGlobal));
    enqueue();
    gpuErrChk(cudaStreamEndCapture(execution.stream,&graph));
    gpuErrChk(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for (int pass = 0; pass < 3; ++pass) {
        const auto a = matrixData<T>(m,n,pass);
        std::vector<T> b(m);
        for (size_t row = 0; row < m; ++row) b[row] = T(row+pass);
        source.upload(a); sourceB.upload(b);
        gpuErrChk(cudaGraphLaunch(executable,execution.stream));
        gpuErrChk(cudaStreamSynchronize(execution.stream));
        EXPECT_EQ(0,qr.info()(0));
        checkComponents(a,b,Q,R,rhs);
        EXPECT_EQ(bytes,Session::getInstance().totalAllocatedBytes());
    }
    gpuErrChk(cudaGraphExecDestroy(executable));
    gpuErrChk(cudaGraphDestroy(graph));
}
} // namespace
