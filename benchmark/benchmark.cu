#include "tensor.cuh"
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>

// Standalone comparisons: no production implementations are replaced here.
struct Buffer {
    void *data = nullptr;
    explicit Buffer(size_t bytes) { gpuErrChk(cudaMalloc(&data, bytes)); }
    ~Buffer() { gpuErrChk(cudaFree(data)); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    template<class T> T *as() const { return static_cast<T *>(data); }
};

struct Events {
    cudaEvent_t start, stop;
    Events() { gpuErrChk(cudaEventCreate(&start)); gpuErrChk(cudaEventCreate(&stop)); }
    ~Events() { gpuErrChk(cudaEventDestroy(start)); gpuErrChk(cudaEventDestroy(stop)); }
};

struct GemmShape { int m, n, k, batch; };
struct RowShape { int rows, cols, selected; };
struct Options {
    int samples = 200, warmup = 30;
    std::vector<GemmShape> gemms{{8,8,8,128}, {32,32,32,128}, {64,32,48,64}, {128,128,128,16}};
    std::vector<RowShape> rows{{32,32,1}, {128,128,8}, {512,256,64}, {1024,512,256}};
};

template<class T>
void stridedGemm(T *a, T *b, T *c, GemmShape s, T beta = 0) {
    const T alpha = T(1.25);
    auto handle = Session::getInstance().cuBlasHandle();
    if constexpr (std::is_same_v<T, float>) {
        gpuErrChk(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            s.m, s.n, s.k, &alpha, a, s.m, 1LL*s.m*s.k, b, s.k, 1LL*s.k*s.n,
            &beta, c, s.m, 1LL*s.m*s.n, s.batch));
    } else {
        gpuErrChk(cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            s.m, s.n, s.k, &alpha, a, s.m, 1LL*s.m*s.k, b, s.k, 1LL*s.k*s.n,
            &beta, c, s.m, 1LL*s.m*s.n, s.batch));
    }
}

template<class T>
__global__ void extractRows(const T *src, T *dst, int rows, int cols, int first, int count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_t(count)*cols) dst[idx] = src[first + idx%count + (idx/count)*rows];
}

template<class T>
void copyRows(const T *src, T *dst, RowShape s, int first, int method, cudaStream_t stream) {
    if (method == 0) {
        for (int row = 0; row < s.selected; ++row) {
            auto handle = Session::getInstance().cuBlasHandle();
            if constexpr (std::is_same_v<T, float>) {
                gpuErrChk(cublasScopy(handle, s.cols, src+first+row, s.rows, dst+row, s.selected));
            } else {
                gpuErrChk(cublasDcopy(handle, s.cols, src+first+row, s.rows, dst+row, s.selected));
            }
        }
    } else if (method == 1) {
        gpuErrChk(cudaMemcpy2DAsync(dst, s.selected*sizeof(T), src+first, s.rows*sizeof(T),
            s.selected*sizeof(T), s.cols, cudaMemcpyDeviceToDevice, stream));
    } else {
        extractRows<<<numBlocks(size_t(s.selected)*s.cols), THREADS_PER_BLOCK, 0, stream>>>(
            src, dst, s.rows, s.cols, first, s.selected);
        gpuErrChk(cudaGetLastError());
    }
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(std::ceil(fraction*values.size()))-1];
}

void measure(const Options &options, const std::string &key, cudaStream_t stream,
             bool deviceTiming, const std::function<void()> &operation) {
    for (int i = 0; i < options.warmup; ++i) { operation(); gpuErrChk(cudaStreamSynchronize(stream)); }
    std::vector<double> host, device;
    Events events;
    for (int i = 0; i < options.samples; ++i) {
        // No other work is in flight in this standalone process.
        auto start = std::chrono::steady_clock::now();
        operation();
        gpuErrChk(cudaStreamSynchronize(stream));
        auto stop = std::chrono::steady_clock::now();
        host.push_back(std::chrono::duration<double, std::micro>(stop-start).count());
    }
    // Separate pass: event instrumentation must not inflate host latency.
    if (deviceTiming) {
        for (int i = 0; i < options.samples; ++i) {
            gpuErrChk(cudaEventRecord(events.start, stream));
            operation();
            gpuErrChk(cudaEventRecord(events.stop, stream));
            gpuErrChk(cudaEventSynchronize(events.stop));
            float ms;
            gpuErrChk(cudaEventElapsedTime(&ms, events.start, events.stop));
            device.push_back(ms*1000.0);
        }
    }
    auto print = [&](const char *clock, const std::vector<double> &values) {
        std::cout << key << ',' << clock << ',' << options.samples << ','
                  << percentile(values, .5) << ',' << percentile(values, .95) << ','
                  << percentile(values, .99) << ',' << *std::max_element(values.begin(), values.end()) << '\n';
    };
    print("host", host);
    if (deviceTiming) print("cuda_event", device);
}

template<class T>
void checkValues(const T *data, const std::vector<T> &expected) {
    std::vector<T> actual(expected.size());
    gpuErrChk(cudaMemcpy(actual.data(), data, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i]) > T(1e-5)*std::max(T(1), std::abs(expected[i])))
            throw std::runtime_error("benchmark correctness check failed at element " + std::to_string(i));
    }
}

template<class T>
void benchmarkGemm(const Options &options, GemmShape s, cudaStream_t stream) {
    std::vector<T> a(size_t(s.m)*s.k*s.batch, T(1)), b(size_t(s.k)*s.n*s.batch, T(2));
    std::vector<T> c(size_t(s.m)*s.n*s.batch, T(3));
    DTensor<T> A(a,s.m,s.k,s.batch), B(b,s.k,s.n,s.batch), C(c,s.m,s.n,s.batch);
    C.addAB(A,B,T(1.25),T(.5));
    gpuErrChk(cudaStreamSynchronize(stream));
    checkValues(C.raw(), std::vector<T>(c.size(), T(2.5*s.k+1.5)));
    C.upload(c);
    stridedGemm(A.raw(),B.raw(),C.raw(),s,T(.5));
    gpuErrChk(cudaStreamSynchronize(stream));
    checkValues(C.raw(), std::vector<T>(c.size(), T(2.5*s.k+1.5)));
    std::string key = std::string("gemm,") + (sizeof(T)==4?"float":"double") + ',' +
        std::to_string(s.m)+','+std::to_string(s.n)+','+std::to_string(s.k)+','+std::to_string(s.batch);
    measure(options,key+",preallocated,pointer",stream,true,[&] { C.addAB(A,B,T(1.25),T(0)); });
    measure(options,key+",preallocated,strided",stream,true,[&] { stridedGemm(A.raw(),B.raw(),C.raw(),s); });
    // Includes all three allocations, pointer-array initialization, D2D input
    // initialization, GEMM completion, and destruction. No host uploads here.
    measure(options,key+",allocate,pointer",stream,false,[&] {
        DTensor<T> x(s.m,s.k,s.batch), y(s.k,s.n,s.batch), z(s.m,s.n,s.batch);
        gpuErrChk(cudaMemcpyAsync(x.raw(),A.raw(),a.size()*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        gpuErrChk(cudaMemcpyAsync(y.raw(),B.raw(),b.size()*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        z.addAB(x,y,T(1.25),T(0));
        gpuErrChk(cudaStreamSynchronize(stream));
    });
    measure(options,key+",allocate,strided",stream,false,[&] {
        Buffer x(a.size()*sizeof(T)), y(b.size()*sizeof(T)), z(c.size()*sizeof(T));
        gpuErrChk(cudaMemcpyAsync(x.data,A.raw(),a.size()*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        gpuErrChk(cudaMemcpyAsync(y.data,B.raw(),b.size()*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        stridedGemm(x.as<T>(),y.as<T>(),z.as<T>(),s);
        gpuErrChk(cudaStreamSynchronize(stream));
    });
}

template<class T>
void benchmarkRows(const Options &options, RowShape s, cudaStream_t stream) {
    std::vector<T> source(size_t(s.rows)*s.cols);
    for (size_t i=0; i<source.size(); ++i) source[i] = T(int(i%127)-63);
    DTensor<T> A(source,s.rows,s.cols), result(s.selected,s.cols);
    int first = (s.rows-s.selected)/2;
    std::vector<T> expected(size_t(s.selected)*s.cols);
    for (int c=0;c<s.cols;++c) for (int r=0;r<s.selected;++r)
        expected[r+size_t(c)*s.selected] = source[first+r+size_t(c)*s.rows];
    auto apiResult = A.getRows(first,first+s.selected-1,0);
    gpuErrChk(cudaStreamSynchronize(stream));
    checkValues(apiResult.raw(),expected);
    std::string key = std::string("rows,")+(sizeof(T)==4?"float":"double")+','+
        std::to_string(s.rows)+','+std::to_string(s.cols)+','+std::to_string(s.selected)+",1";
    measure(options,key+",allocate,getRows",stream,false,[&] {
        auto output = A.getRows(first,first+s.selected-1,0);
        gpuErrChk(cudaStreamSynchronize(stream));
    });
    for (int method=0;method<3;++method) {
        const std::string name = method==0?"row_loop":method==1?"copy2d":"kernel";
        copyRows(A.raw(),result.raw(),s,first,method,stream);
        gpuErrChk(cudaStreamSynchronize(stream));
        checkValues(result.raw(),expected);
        measure(options,key+",preallocated,"+name,stream,true,[&] { copyRows(A.raw(),result.raw(),s,first,method,stream); });
        measure(options,key+",allocate,"+name,stream,false,[&] {
            DTensor<T> output(s.selected,s.cols);
            copyRows(A.raw(),output.raw(),s,first,method,stream);
            gpuErrChk(cudaStreamSynchronize(stream));
        });
    }
}

int main(int argc, char **argv) {
    try {
        Options options;
        auto integer = [&](int &i) {
            if (++i>=argc) throw std::invalid_argument("missing option value");
            std::string value(argv[i]);
            size_t used=0;
            long n=std::stol(value,&used);
            if (used!=value.size() || n<=0 || n>INT_MAX) throw std::invalid_argument("expected positive integer");
            return static_cast<int>(n);
        };
        for (int i=1;i<argc;++i) {
            std::string arg(argv[i]);
            if (arg=="--samples") options.samples=integer(i);
            else if (arg=="--warmup") options.warmup=integer(i);
            else if (arg=="--gemm") {
                int m=integer(i), n=integer(i), k=integer(i), batch=integer(i);
                options.gemms={{m,n,k,batch}};
            } else if (arg=="--rows") {
                int rows=integer(i), cols=integer(i), selected=integer(i);
                if (selected>rows) throw std::invalid_argument("selected rows exceed source rows");
                options.rows={{rows,cols,selected}};
            } else throw std::invalid_argument("options: --samples N --warmup N --gemm M N K BATCH --rows ROWS COLS SELECTED");
        }
        if (std::string(GPUTILS_BENCH_CONFIG)!="Release" && std::string(GPUTILS_BENCH_CONFIG)!="RelWithDebInfo")
            throw std::runtime_error("use a Release or RelWithDebInfo build");
        cudaDeviceProp props{};
        int device=0, driver=0, runtime=0;
        gpuErrChk(cudaGetDevice(&device)); gpuErrChk(cudaGetDeviceProperties(&props,device));
        gpuErrChk(cudaDriverGetVersion(&driver)); gpuErrChk(cudaRuntimeGetVersion(&runtime));
        std::cerr << "GPU: " << props.name << "; compute capability: " << props.major << '.' << props.minor
                  << "; driver API: " << driver << "; runtime: " << runtime << "; build: " << GPUTILS_BENCH_CONFIG
                  << "; warmup: " << options.warmup << "; samples: " << options.samples << '\n';
        cudaStream_t stream;
        gpuErrChk(cublasGetStream(Session::getInstance().cuBlasHandle(),&stream));
        std::cout << "operation,dtype,dim1,dim2,dim3,batch,scope,variant,clock,samples,median_us,p95_us,p99_us,max_us\n";
        std::cout << std::fixed << std::setprecision(3);
        for (auto shape:options.gemms) {
            checkedTensorSize<double>(shape.m,shape.k,shape.batch);
            checkedTensorSize<double>(shape.k,shape.n,shape.batch);
            checkedTensorSize<double>(shape.m,shape.n,shape.batch);
            benchmarkGemm<float>(options,shape,stream); benchmarkGemm<double>(options,shape,stream);
        }
        for (auto shape:options.rows) {
            checkedTensorSize<double>(shape.rows,shape.cols,1);
            benchmarkRows<float>(options,shape,stream); benchmarkRows<double>(options,shape,stream);
        }
        gpuErrChk(cudaDeviceSynchronize());
        std::cerr << "All benchmark correctness checks passed.\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
