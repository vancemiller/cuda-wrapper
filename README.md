# cuda wrapper
This repository contains code for generating a shared library that intercepts calls to the NVIVIA CUDA runtime library.

## Quickstart:
    mkdir build
    cmake ..
    make
    # edit stubs
    make

## Components:
1. 'make\_stubs.py' generates CUDA function stubs, for you to fill in with custom functionality
2. 'wrap\_generate.py' combines stubs into a shim library

## Notes:
Compile your CUDA program to dynamically link the CUDA library, not statically. This is achieved with the `--cudart shared` flag to `nvcc`.
Use the shim library: `LD_PRELOAD=/path/to/cuda_wrapper.so ./cuda_program`
