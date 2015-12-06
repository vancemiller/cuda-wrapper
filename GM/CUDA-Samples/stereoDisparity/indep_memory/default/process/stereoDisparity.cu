/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A CUDA program that demonstrates how to compute a stereo disparity map using
 *   SIMD SAD (Sum of Absolute Difference) intrinsics
 */

/*
 * Modified to iterate the stereo disparity multiple times to use as
 * benchmark/stress test for GPU locking using Cuda call-wrapping
 * functions.  The program's performance is dominated by 
 * the computation on the execution engine (EE) while memory copies 
 * between Host and Device using the copy engine (CE) are significantly
 * less time consuming.
 *
 * This version uses the default stream and synchronous memory copy
 * operations (cudaMemcpy()).  Cuda kernel invocations are always
 * asynchronous so cudaDeviceSynchronize() is used to synchronize
 * with kernel execution.  Host pinned memory is not used because
 * the copy operations are not a significant element of performance.
 *
 * The program depends on two input files containing the image 
 * representations for the left and right stereo images 
 * (stereo.im0.640x533.ppm and stereo.im1.640x533.ppm)
 * which must be in the directory with the executable.
 *
 * Modified by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill
 * 2015
 */
// control number of iterations by count or elapsed time
#define MAX_LOOPS 10000 // iteration count
#define TIME_LENGTH 30  // elapsed time (seconds)

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

// includes, kernels
// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>
// The kernel code
#include "stereoDisparity_kernel.cuh"

// includes, project
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! CUDA Sample for calculating depth maps
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int i;
    int count = 0;

    int sync_level = 2; //default -- process blocking

    pid_t my_pid;
    time_t start_time, now, elapsed;

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
          printf("PID %d started > Synch Level is Spin\n", my_pid);
          break;
       case 1:
          cudaSetDeviceFlags(cudaDeviceScheduleYield);
          printf("PID %d started > Synch Level is Yield\n", my_pid);
          break;
       default:
          cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
          printf("PID %d started > Synch Level is Block\n", my_pid);
      }

    // follow convention and initialize CUDA/GPU
    // used here to invoke initialization of GPU locking
    cudaFree(0);

    // use device 0, the only one on a TK1
    cudaSetDevice(0);
  
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    // Search paramters
    int minDisp = -16;
    int maxDisp = 0;

    // Load image data
    // functions allocate memory for the images on host side
    // initialize pointers to NULL to request lib call to allocate as needed
    // PPM images are loaded into 4 byte/pixel memory (RGBX)
    unsigned char *h_img0 = NULL;
    unsigned char *h_img1 = NULL;
    unsigned int w, h;
    char *fname0 = sdkFindFilePath("stereo.im0.640x533.ppm", argv[0]);
    char *fname1 = sdkFindFilePath("stereo.im1.640x533.ppm", argv[0]);

    if (!sdkLoadPPM4ub(fname0, &h_img0, &w, &h))
    {
        fprintf(stderr, "PID %d Failed to load <%s>\n", my_pid, fname0);
        exit(-1);
    }

    if (!sdkLoadPPM4ub(fname1, &h_img1, &w, &h))
    {
        fprintf(stderr, "PID %d Failed to load <%s>\n", my_pid, fname1);
        exit(-1);
    }

    // set up parameters used in rest of program
    dim3 numThreads = dim3(blockSize_x, blockSize_y, 1);
    dim3 numBlocks = dim3(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));
    unsigned int numData = w*h;
    unsigned int memSize = sizeof(int) * numData;

    //allocate memory for the result on host side
    unsigned int *h_odata = (unsigned int *)malloc(memSize);

    // allocate device memory for inputs and the result
    unsigned int *d_odata, *d_img0, *d_img1;
    checkCudaErrors(cudaMalloc((void **) &d_odata, memSize));
    checkCudaErrors(cudaMalloc((void **) &d_img0, memSize));
    checkCudaErrors(cudaMalloc((void **) &d_img1, memSize));

    // more setup for using the GPU
    size_t offset = 0;
    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

    tex2Dleft.addressMode[0] = cudaAddressModeClamp;
    tex2Dleft.addressMode[1] = cudaAddressModeClamp;
    tex2Dleft.filterMode     = cudaFilterModePoint;
    tex2Dleft.normalized     = false;
    tex2Dright.addressMode[0] = cudaAddressModeClamp;
    tex2Dright.addressMode[1] = cudaAddressModeClamp;
    tex2Dright.filterMode     = cudaFilterModePoint;
    tex2Dright.normalized     = false;
    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dleft,  d_img0, ca_desc0, w, h, w*4));
    assert(offset == 0);

    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dright, d_img1, ca_desc1, w, h, w*4));
    assert(offset == 0);
    // all setup and initialization complete, start iterations


    printf("PID %d Iterating stereoDisparity CUDA Kernel for %d seconds, %d max loops\n", my_pid, TIME_LENGTH, MAX_LOOPS);
    now = start_time = time(NULL);

 for (i = 0; 
            ((now - TIME_LENGTH) < start_time) &&
              i < MAX_LOOPS; i++) {

    //initalize the memory for output data to zeros
    for (unsigned int i = 0; i < numData; i++)
        h_odata[i] = 0;

    // copy host memory with images to device
    // copy host memory that was set to zero to initialize device output
    // these calls are synchronous so lock/unlock of CE can be handled in wrappers
    checkCudaErrors(cudaMemcpy(d_img0,  h_img0, memSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_img1,  h_img1, memSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_odata, memSize, cudaMemcpyHostToDevice));

    // First run the warmup kernel (which we'll use to get the GPU in the correct max power state)
    // lock of EE is handled in wrapper for cudaLaunch()
    stereoDisparityKernel<<<numBlocks, numThreads>>>(d_img0, d_img1, d_odata, w, h, minDisp/2, maxDisp);

    // synchronize the default stream
    // used here so wrapper function can release EE lock
    cudaDeviceSynchronize();

    // copy host memory that was set to zero to initialize device output
    checkCudaErrors(cudaMemcpy(d_odata, h_odata, memSize, cudaMemcpyHostToDevice));

    // launch the stereoDisparity kernel
    // lock of EE is handled in wrapper for cudaLaunch()
    stereoDisparityKernel<<<numBlocks, numThreads>>>(d_img0, d_img1, d_odata, w, h, minDisp, maxDisp);

    // synchronize the default stream
    // used here so wrapper function can release EE lock
    cudaDeviceSynchronize();

    // Check to make sure the kernel didn't fail
    getLastCudaError("Kernel execution failed");

    //Copy result from device to host for verification
    // these calls are synchronous so lock/unlock of CE can be handled in wrappers
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, memSize, cudaMemcpyDeviceToHost));

    now = time(NULL);

} // ends for loop
    elapsed = now - start_time;
    count = i;

    // calculate checksum of resultant GPU image
    // This verification is applied only to the 
    // last result computed
    unsigned int checkSum = 0;

    for (unsigned int i=0 ; i<w *h ; i++)
    {
        checkSum += h_odata[i];
    }

    if (checkSum == 4293895789) //valid checksum only for these two images
       printf("PID %d Test PASSED\n", my_pid);
    else {
       fprintf(stderr, "PID %d verification failed, GPU Checksum = %u, ", my_pid, checkSum);
       exit(-1);
    }

    printf("PID %d completed %d, duration %ld seconds\n", my_pid, count, elapsed);

#ifdef WRITE_DISPARITY
    // write out the resulting disparity image.
    // creates file in directory containing executable
    unsigned char *dispOut = (unsigned char *)malloc(numData);
    int mult = 20;

    char fnameOut[50] = "";
    sprintf(fnameOut,"PID_%d_", my_pid);
    strcat(fnameOut, "output_GPU.pgm");

    for (unsigned int i=0; i<numData; i++)
    {
        dispOut[i] = (int)h_odata[i]*mult;
    }

    printf("GPU image: <%s>\n", fnameOut);
    sdkSavePGM(fnameOut, dispOut, w, h);
    if (dispOut != NULL) free(dispOut);

#endif

    // cleanup device memory
    checkCudaErrors(cudaFree(d_odata));
    checkCudaErrors(cudaFree(d_img0));
    checkCudaErrors(cudaFree(d_img1));

    // cleanup host memory
    if (h_odata != NULL) free(h_odata);

    if (h_img0 != NULL) free(h_img0);

    if (h_img1 != NULL) free(h_img1);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}
