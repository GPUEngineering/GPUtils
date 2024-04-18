# ----
add_library(device INTERFACE)
target_sources(device
        PRIVATE
        tensor.cuh
)
target_link_libraries(device
        INTERFACE
        device_compiler_flags
        cublas
        cusolver
        cudadevrt)
# ----
