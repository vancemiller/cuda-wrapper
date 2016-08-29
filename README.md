# gpu-sync Lite

This repository contains code for generating a shared library that intercepts calls to the NVIVIA Cuda 6 runtime library and redirects them to your own locking code.

## How to use:
1. Generate cuda function stubs through the `GPUSync/Wrapper/make_stubs.py` program. See program comments for instructions.
2. Find cuda calls that you would like to trigger locking and insert locking code (described in `Wrapper/GPU_Locks.h`). We recommend:
 1. `cudaLaunch()`: acquire lock, run kernel, release lock.
 2. all variants of `cudaMemcpy()`: acquire lock, do memcpy, release lock.
 3. all async variants of i and ii: acquire lock, do operation.
 4. `cudaStreamSynchronize`, `cudaDeviceSynchronize`: release last acquired lock.
3. Generate the library via `GPUSync/Wrapper/wrap_generate.py`.
4. Compile the library:
 1. Use the makefile in `GPUSync/Locks/Kernel_Locks/` if you would like to use a kernel module.
 2. Use the makefile in `GPUSync/Locks/POSIX_Locks/` if you would like to use a runtime implementation.
5. Use the locks:
 1. Compile your CUDA 6 program to link the cuda library at runtime, not statically. This is achieved with the `--cudart shared` flag to `nvcc`.
 2. Use the interception library: `LD_PRELOAD=/path/to/libcudart_wrapper.so ./cuda_program`
