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
 * The program's performance is dominated by the computation on the
 * execution engine (EE) while memory copies between Host and Device
 * using the copy engine (CE) are significantly less time consuming.
 *
 * This version uses a user allocated stream and asynchronous memory
 * copy operations (cudaMemcpyAsync()).  Cuda kernel invocations on the
 * stream are also asynchronous.  cudaStreamSynchronize() is used to 
 * synchronize with both the copy and kernel executions.  Host pinned
 * memory is not used because the copy operations are not a significant 
 * element of performance.
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
// the kernel code
#include "stereoDisparity_kernel.cuh"

// includes, project
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing

int iDivUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

unsigned int numData;
dim3 numThreads;
dim3 numBlocks;
unsigned int *h_odata;
unsigned int *d_odata, *d_img0, *d_img1;
unsigned int memSize;
cudaStream_t my_stream;
unsigned char *h_img0;
unsigned char *h_img1;
int minDisp;
int maxDisp;
unsigned int w, h;


void stereoDisparity() {
  //initalize the memory for output data to zeros
  for (unsigned int i = 0; i < numData; i++)
    h_odata[i] = 0;

  // copy host memory with images to device

  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(d_img0,  h_img0, memSize, cudaMemcpyHostToDevice, my_stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(my_stream);

  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(d_img1,  h_img1, memSize, cudaMemcpyHostToDevice, my_stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(my_stream);

  // copy host memory that was set to zero to initialize device output
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(d_odata, h_odata, memSize, cudaMemcpyHostToDevice, my_stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(my_stream);

  // First run the warmup kernel (which we'll use to get the GPU in the correct max power state)
  // lock of EE is handled in wrapper for cudaLaunch()
  stereoDisparityKernel<<<numBlocks, numThreads, 0, my_stream>>>(d_img0, d_img1, d_odata, w, h, minDisp/2, maxDisp);

  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(my_stream);

  // copy host memory that was set to zero to initialize device output
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(d_odata, h_odata, memSize, cudaMemcpyHostToDevice, my_stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(my_stream);

  // launch the stereoDisparity kernel
  // lock of EE is handled in wrapper for cudaLaunch()
  stereoDisparityKernel<<<numBlocks, numThreads, 0, my_stream>>>(d_img0, d_img1, d_odata, w, h, minDisp, maxDisp);

  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(my_stream);

  // Check to make sure the kernel didn't fail
  getLastCudaError("Kernel execution failed");

  //Copy result from device to host for verification
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(h_odata, d_odata, memSize, cudaMemcpyDeviceToHost, my_stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(my_stream);

#ifdef PRINT_CHECKSUM
  // calculate sum of resultant GPU image
  // This verification is applied only to the
  // last result computed
  unsigned int checkSum = 0;
  for (unsigned int i=0 ; i <w *h ; i++) {
    checkSum += h_odata[i];
  }
  if (checkSum == 4293895789) //valid checksum only for these two images
    printf("Test PASSED\n");
  else {
    fprintf(stderr, "Verification failed, GPU Checksum = %u, ", checkSum);
    exit(-1);
  }
#endif

#ifdef WRITE_DISPARITY
  // write out the resulting disparity image.
  // creates file in directory containing executable
  unsigned char *dispOut = (unsigned char *)malloc(numData);
  int mult = 20;

  char fnameOut[50] = "";
  strcat(fnameOut, "output_GPU.pgm");

  for (unsigned int i=0; i<numData; i++)
  {
    dispOut[i] = (int)h_odata[i]*mult;
  }

  printf("GPU image: <%s>\n", fnameOut);
  sdkSavePGM(fnameOut, dispOut, w, h);
  if (dispOut != NULL) free(dispOut);
#endif
  // prepare to clean up 
  // wrapper will release any lock held
  cudaDeviceSynchronize();

  // cleanup device memory
  checkCudaErrors(cudaFree(d_odata));
  checkCudaErrors(cudaFree(d_img0));
  checkCudaErrors(cudaFree(d_img1));

  // cleanup host memory
  if (h_odata != NULL) free(h_odata);

  if (h_img0 != NULL) free(h_img0);

  if (h_img1 != NULL) free(h_img1);

  // finish clean up with deleting the user-created stream
  cudaStreamDestroy(my_stream);

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();
}

int main(int argc, char **argv)
{
  int sync_level = 2; //default -- process blocking

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
      break;
    case 1:
      cudaSetDeviceFlags(cudaDeviceScheduleYield);
      break;
    default:
      cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  }

  // follow convention and initialize CUDA/GPU
  // used here to invoke initialization of GPU locking
  cudaFree(0);

  // use device 0, the only one on a TK1
  cudaSetDevice(0);

  // create a user-defined stream
  cudaStreamCreate(&my_stream);

  // Search paramters
  minDisp = -16;
  maxDisp = 0;

  // Load image data
  // functions allocate memory for the images on host side
  // initialize pointers to NULL to request lib call to allocate as needed
  // PPM images are loaded into 4 byte/pixel memory (RGBX)
  h_img0 = NULL;
  h_img1 = NULL;
  char *fname0 = sdkFindFilePath("stereo.im0.640x533.ppm", argv[0]);
  char *fname1 = sdkFindFilePath("stereo.im1.640x533.ppm", argv[0]);

  if (!sdkLoadPPM4ub(fname0, &h_img0, &w, &h))
  {
    fprintf(stderr, "Failed to load <%s>\n", fname0);
    exit(-1);
  }

  if (!sdkLoadPPM4ub(fname1, &h_img1, &w, &h))
  {
    fprintf(stderr, "Failed to load <%s>\n", fname1);
    exit(-1);
  }

  // set up parameters used in the rest of program
  numThreads = dim3(blockSize_x, blockSize_y, 1);
  numBlocks = dim3(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));
  numData = w*h;
  memSize = sizeof(int) * numData;

  //allocate memory for the result on host side
  h_odata = (unsigned int *)malloc(memSize);

  // allocate device memory for inputs and result
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
  stereoDisparity();
}
