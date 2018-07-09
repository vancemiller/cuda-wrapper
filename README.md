# cuda wrapper
This repository contains code for generating a shared library that intercepts calls to the NVIVIA CUDA runtime library.

## How to use:
1. Generate cuda function stubs through the `GPUSync/Wrapper/make_stubs.py` program. See program comments for instructions.
2. Find cuda calls that you would like to modify and edit the stub files with your code.
3. Generate the library via `GPUSync/Wrapper/wrap_generate.py`.
4. Compile the library with the Makefile
5. Compile your CUDA program to dynamically link the CUDA library, not statically. This is achieved with the `--cudart shared` flag to `nvcc`.
6. Use the interception library: `LD_PRELOAD=/path/to/libcudart_wrapper.so ./cuda_program`
