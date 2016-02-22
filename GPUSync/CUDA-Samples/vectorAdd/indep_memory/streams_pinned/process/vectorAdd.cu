/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */
/*
 * Modified to iterate the vector addition multiple times to use as
 * benchmark/stress test for GPU locking using Cuda call-wrapping
 * functions.  The program's performance is dominated by memory copies 
 * between Host and Device using the copy engine (CE), while computation
 * on the execution engine (EE) is signigicantly less time consuming than
 * the copying.
 *
 * This version uses a user allocated stream and asynchronous memory
 * copy operations (cudaMemcpyAsync()).  Cuda kernel invocations on the
 * stream are also asynchronous.  cudaStreamSynchronize() is used to 
 * synchronize with both the copy and kernel executions.  Host pinned
 * memory was also added to better work with the extensive copy operations.
 *
 * Modified by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill
 * 2015
 */

// control number of iterations by count or elapsed time
#define MAX_LOOPS 10000  // iteration count
#define TIME_LENGTH 30  // elapsed time (seconds)

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(int argc, char *argv[])
{
    int i;
    int count = 0;
    pid_t my_pid;
    time_t start_time, now, elapsed;

    int sync_level = 2; //default -- process blocking

    my_pid = getpid();

    /*
     * The only parameter is an integer that indicates the desired level of
     * synchronization used by the GPU driver (values defined below).  The
     * specified level is used in cudaSetDeviceFlags() to set the level
     * prior to initialization.
     */
    if (argc == 2)
       sync_level = atoi(argv[1]);
            // level 0 - spin polling (busy waiting) for GPU to finish
            // level 1 - yield each time through the polling loop to let another thread run
            // level 2 - block process waiting for GPU to finish
    switch (sync_level)
      {
       case 0:
          cudaSetDeviceFlags(cudaDeviceScheduleSpin);
          fprintf(stderr, "PID %d started > Synch Level is Spin\n", my_pid);
          break;
       case 1:
          cudaSetDeviceFlags(cudaDeviceScheduleYield);
          fprintf(stderr, "PID %d started > Synch Level is Yield\n", my_pid);
          break;
       default:
          cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
          fprintf(stderr, "PID %d started > Synch Level is Block\n", my_pid);
      }

#ifdef SET_PRIORITY
/*
 * WARNING: this code has not been tested.
 */
// set initial priority to specified value (must be odd > 2)
// assume the initial priority is set before CUDA initialied
    int rc;
    struct sched_param my_param;
    int my_prio = 0;

    if (argc == 2) {
       my_prio = atoi(argv[1]);
       if ((my_prio < 3) ||
           ((my_prio % 2) == 0))
          my_prio = 0;
    }
    if (my_prio == 0) {
       fprintf(stderr, "PID %d running SCHED_OTHER\n", my_pid);
       my_param.sched_priority = 0;
       rc = sched_setscheduler(0, SCHED_OTHER, &my_param);
    }
    else {
       fprintf(stderr, "PID %d running SCHED_FIFO priority %d\n",
               my_pid, my_prio);
       my_param.sched_priority = my_prio;
       rc = sched_setscheduler(0, SCHED_FIFO, &my_param);
    }
    if (rc != 0) {
       fprintf(stderr, "PID %d Set Scheduler FAILED, running default, error %d\n", my_pid, errno);
    }
#endif

    // Follow convention and initialize CUDA/GPU
    // used here to invoke initialization of GPU locking
    cudaFree(0);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // create a user defined stream
    cudaStream_t my_stream;
    cudaStreamCreate(&my_stream);

    int numElements = 4000000;
    size_t size = numElements * sizeof(float); // 16,000,000 bytes

    float *h_A, *h_B, *h_C;

    // Host allocations in pinned memory
    // Allocate the host input vector A
    err = cudaMallocHost((void **)&h_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate host vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the host input vector B
    err = cudaMallocHost((void **)&h_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate host vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the host output vector C
    err = cudaMallocHost((void **)&h_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate host vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef RESET_PRIORITY
/*
 * WARNING: this code has not been tested.
 */
  // reset main priority to make callback have greater priority
    if (my_prio > 0) {
       my_param.sched_priority = my_prio - 1;
       sched_setscheduler(0, SCHED_FIFO, &my_param);
    }
#endif

     fprintf(stderr, "PID %d Iterating Vector Add CUDA Kernel for %d seconds, %d max loops\n", my_pid, TIME_LENGTH, MAX_LOOPS);
     now = start_time = time(NULL);

 for (i = 0; 
            ((now - TIME_LENGTH) < start_time) &&
              i < MAX_LOOPS; i++) {

    // copy the A and B vectors from Host to Device memory
    // these calls are asynchronous so only the lock of CE can be handled in the wrapper
    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, my_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, my_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    // lock of EE is handled in wrapper for cudaLaunch()
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, d_B, d_C, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // synchronize with the stream after kernel execution
    // the wrapper for this function releases any lock held (EE here)
    cudaStreamSynchronize(my_stream);

    // copy the result vector from Device to Host memory
    // this call is asynchronous so only the lock of CE can be handled in the wrapper
    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, my_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);
    now = time(NULL);

 } // ends for loop
   elapsed = now - start_time;
   count = i;

    // Verify that the result vector is correct
    // This verification is applied only to the 
    // last result computed
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fprintf(stderr, "PID %d Test PASSED\n", my_pid);
    fprintf(stderr, "PID %d completed %d, duration %ld seconds\n", my_pid, count, elapsed);
    fprintf(stdout, "%d,", count);

    // Free device global memory for inputs A and B and result C
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory that was pinned
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    // clean up the user allocated stream
    cudaStreamSynchronize(my_stream);
    cudaStreamDestroy(my_stream);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
}

