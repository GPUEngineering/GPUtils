<!-- ---------------------
      v0.3.0
     --------------------- -->
## [v0.1.0] - 20-04-2024

### Added

- Implementation of `DTensor<T>`, which is our basic entity for data storage and maniputation (supports basic linear algebra using `cublas` and `cusolver`); implementation of `=+`, `-=`, `*=` (for scalars and other tensors), `+`, `-`, `*` (scalars and tensors), printing (using `std::cout <<`), computation of norms (Frobenius and sum of absolute values of all elements); device vectors and matrices are tensors  
- Singular value decomposition using `cublas`
- Least-squares on tensors
- Computation of nullspace matrices (on tensor objects)
- Cholesky factorisation 