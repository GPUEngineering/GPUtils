# GPU benchmarks

Build this standalone project from the repository root. It does not modify the
library's algorithms or require GoogleTest. Choose an architecture supported by
your GPU and CUDA toolkit (the current remote Blackwell GPU uses `120`).

```bash
cmake -S benchmark -B benchmark/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build benchmark/build -j4
benchmark/build/gputils_benchmark > benchmark/build/results.csv \
  2> benchmark/build/metadata.txt
```

Defaults: 30 warm-up iterations and 200 measured samples per variant, for both
`float` and `double`. The suite covers four GEMM and four extraction shapes.
Use representative application dimensions when available:

```bash
benchmark/build/gputils_benchmark --samples 1000 --warmup 100 \
  --gemm 64 32 48 128 --rows 512 256 64 > benchmark/build/custom.csv
```

`--gemm M N K BATCH` selects `(M x K) * (K x N)` and replaces the default GEMM
cases; `--rows ROWS COLS SELECTED` replaces the default extraction cases. Both
families always run. Extraction uses the middle contiguous range of rows from a
single column-major matrix. All options require positive integers.

## Comparisons

- GEMM `preallocated`: current `DTensor::addAB` versus strided batched cuBLAS,
  with existing tensors. This isolates execution; both share the already-built
  tensor pointer arrays. For a batch of one the current implementation uses
  ordinary GEMM.
- GEMM `allocate`: construction of A, B and C, device-to-device initialization
  of A and B, GEMM, completion, and destruction. The pointer variant uses actual
  DTensors; the strided variant uses raw device buffers without matrix-pointer
  arrays. This estimates the benefit of a future pointer-free storage path;
  changing `addAB` alone will not remove the current DTensor pointer allocations.
- Rows `preallocated`: original per-row cuBLAS copy loop versus one asynchronous
  2D device copy versus one extraction kernel, using an existing destination.
- Rows `allocate`: actual `getRows` is reported separately, alongside the old
  per-row loop and alternatives allocating an identically shaped DTensor.
  Completion and destruction are included. Source upload is not timed.

All candidates are checked against expected values before timing. GEMM checks
nonzero inputs and non-default alpha/beta, then times alpha=1.25, beta=0 so the
output does not accumulate across repetitions. Row checks compare every output
element to the source range. Check failures cause a nonzero exit.

## Interpreting output

CSV dimensions are M,N,K for GEMM and ROWS,COLS,SELECTED for extraction. Times
are in microseconds; percentiles use nearest ranks. Host timings measure one
completed operation using `steady_clock`, including the final stream wait.
Allocation-inclusive timings also include allocation, initialization and frees.
CUDA-event timings are collected in a separate pass for preallocated cases;
they include device work and any gaps between host submissions. Event timings
are not emitted for allocation cases, where host work is central to the result.

This measures isolated, synchronized latency, not pipelined throughput. The
stream is the session's cuBLAS stream. Library defaults for math mode are kept.
GPU name, compute capability, driver/runtime versions, build type and sample
counts are written to stderr. Run on an otherwise idle GPU, retain the metadata,
record the source revision, compiler version, GPU clocks/power/temperature and
persistence state, and repeat the run to check stability. Persistence avoids
driver teardown; it does not lock clocks or eliminate thermal and scheduling
variation. p99 and maximum are observations, not real-time bounds. CUDA event
resolution and synchronization overhead can dominate very short operations.

The default shapes are exploratory; use application workloads before deciding
which production implementation to adopt. No performance threshold is enforced.
