# Example project
This is an example project that can serve as a template to get your project up and running.

### Prerequisites
- A machine with CUDA-capable GPU device.
- Download [NVIDIA CUDA Toolkit 12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive). 
- Install and test on machine using [these instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). 

## Run your first project...
...by following these steps:
1. Copy-pasta this `example` folder to your machine.
2. Make sure toolkit location in line 9 of `CMakeLists.txt` is correct for your installation.
3. Load CMake project from `CMakeLists.txt`.
4. Run executable `example_main`.
5. (Optional) Edit lines 3-5 of `CMakeLists.txt` and reload CMake project.

> Note: `example_main` runs `main.cu` with data stored in `src/data`.

Happy number crunching!
