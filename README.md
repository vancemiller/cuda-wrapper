# cuda wrapper
This repository contains code for generating a shared library that intercepts calls to the NVIDIA CUDA runtime library.

## Quickstart:
Include this project as a submodule of your project. The project exports a CMake rule that generates a cuda wrapper library.
Your project's CMakeLists.txt should minimally contain:
```
cmake_minimum_required(VERSION 3.16)
project(my_project)

add_subdirectory(cuda-wrapper)
generate_cuda_wrapper(my_wrapper_name)
```

Now generate the stubs for your wrapper library by running the build:

```
mkdir build
cmake ..
make
```

This generated some empty stub files in a directory called stubs. This directory should be in the same location as the CMakeLists.txt that called generate_cuda_wrapper.

Edit the stubs, then build again to generate the final library.

## Components:
1. `make_stubs.py` generates CUDA function stubs, for you to fill in with custom functionality
2. `wrap_generate.py` combines stubs into a shim library

## Notes:
Compile your CUDA program to dynamically link the CUDA library, not statically. This is achieved with the `--cudart shared` flag to `nvcc`.
To use the shim library: `LD_PRELOAD=/path/to/cuda_wrapper.so ./cuda_program`
