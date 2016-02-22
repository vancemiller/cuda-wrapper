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
 * A configurable number of user threads are created, each of which
 * independently performs iterations of stereo disparity and 
 * contend for GPU CE and EE resources.  POSIX pthreads are used. 
 *  
 * The threading structure was originally written by Glenn Elliott
 * for CUDA 6.0 (now written for CUDA 6.5)
 *
 * Modified by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill
 * 2015
 */
// control number of iterations by count or elapsed time
#define MAX_LOOPS 10000
#define TIME_LENGTH 30

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <pthread.h>

// includes, kernels
#include <cuda.h>
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

// A few global variables that will be shared by all threads

// Used to synchronize start of thread loops after initialization 
pthread_barrier_t worker_barrier;
// parameter defaults (see main() for more details)
bool verbose = false;  // no verbose output
int sync_level = 2; //default -- process blocking

//use syscall to use kernel function to get thread ID (TID)
//glic has no wrapper for this!
inline pid_t gettid(void)
{
  return syscall(__NR_gettid);
}

//to pass parameters to threads (see main() for more details)
struct worker_args
{
  time_t runtime;
  int id;

  int cpu;
  int sched_policy;
  int sched_priority;
};

//to return iteration counts from threads
struct results
{
  pid_t id;
  long ncompleted;
};

/*
 * Set up the scheduling policy and priority for calling process or thread
 * Also includes currently disabled code to set CPU affinity
 * for the thread (disabled becuase it causes an infinite loop
 * on one of the cores when used on the TK1 with Linux SCHED_FIFO).
 *
 */
void setsched(int policy, int priority, int cpu = -1)
{
  
  int ret;

  char pbuf[80];

  struct sched_param params; // parameter for kernel call

  memset(&params, 0, sizeof(params));

#ifdef FOOBAR
// This CPU affinity stuff DOES NOT work on TK1 with SCHED_FIFO
  int ncpus;
  cpu_set_t *cpu_set;
  size_t sz;

  if (cpu >= 0)
     {
      // in caller to specified CPU
      ncpus = sysconf(_SC_NPROCESSORS_ONLN);
      if (ncpus <= cpu)
  	 {
  	  fprintf(stderr, "Bad CPU affinity value %d. (valid: [0,%d])\n", cpu, ncpus);
  	  exit(-1);
  	 }

       cpu_set = CPU_ALLOC(ncpus);
       sz = CPU_ALLOC_SIZE(ncpus);
       CPU_ZERO_S(sz, cpu_set);
       CPU_SET_S(cpu, sz, cpu_set);
       ret = sched_setaffinity(gettid(), sz, cpu_set);
       if (ret != 0)
  	  {
  	   perror("Failed to set CPU affinity");
  	   exit(-1);
  	  }
  	CPU_FREE(cpu_set);
       }
#endif

  if (SCHED_OTHER == policy) // Default Linux "fair" scheduler
     {
      // set SCHED_OTHER policy and interpret priority as a nice value
      if (priority < -20 || priority > 19)
  	 {
  	  fprintf(stderr, "Bad SCHED_OTHER priority %d. (valid: [-20,19])\n", priority);
  	  exit(-1);
  	 }
      sched_setscheduler(0, policy, &params);
      ret = setpriority(PRIO_PROCESS, gettid(), priority);
      if (ret != 0)
  	 {
  	  sprintf(pbuf, "Failed to set NICE priority %d", priority);
  	  perror(pbuf);
  	  exit(-1);
  	 }
      }
  else if (SCHED_FIFO == policy)  // Linux Real-Time scheduling class
  	  {
           // set SCHED_FIFO policy and priority
  	   if (priority > 99 || priority < 1)
  	     {
  	      fprintf(stderr, "Bad SCHED_FIFO priority %d. (valid: [1,99])\n", priority);
  	      exit(-1);
  	     }
  	   params.sched_priority = priority;
  	   ret = sched_setscheduler(0, policy, &params);
  	   if (ret != 0)
  	     {
  	      perror("Failed to set SCHED_FIFO");
  	      exit(-1);
             }
  	  }
  	  else
  	    {
  	     fprintf(stderr, "Unsupported sched policy: %d\n", policy);
  	     exit(-1);
            }
}

/*
 * Entry point for worker pthreads
 *
 * Each thread is independently executing the embedded loop
 * as fast as it can. Note that each thread wiil
 * have its own copy of the input and output vectors and
 * allocate its own device memory for the vectors.
 *
 */
void* work(void* _args)
{
  // get thread parameters
  struct worker_args args = *(struct worker_args*)_args;
  long count = 0, total_count = 0;
  time_t start_time, now, elapsed;
  int i;

  pid_t my_tid;

  // allocate struct to return results
  struct results* r = (struct results*)malloc(sizeof(*r));

  free(_args);

  my_tid = gettid(); //each thread has a unique thread ID (TID)

  // Output thread ID
  switch (sync_level)
    {
     case 0:
        printf("TID %d started > Synch Level is Spin\n", my_tid);
        break;
     case 1:
        printf("TID %d started > Synch Level is Yield\n", my_tid);
        break;
     default:
        printf("TID %d started > Synch Level is Block\n", my_tid);
    }

  // do this before any CUDA calls because the GPU driver 
  // signaling thread created to work with this thread
  // will inherit this thread's priority 
  setsched(args.sched_policy, args.sched_priority, args.cpu);


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
    char *fname0 = sdkFindFilePath("stereo.im0.640x533.ppm", "./stereoDisparity");
    char *fname1 = sdkFindFilePath("stereo.im1.640x533.ppm", "./stereoDisparity");

    if (!sdkLoadPPM4ub(fname0, &h_img0, &w, &h))
    {
        fprintf(stderr, "TID %d Failed to load <%s>\n", my_tid, fname0);
        exit(-1);
    }

    if (!sdkLoadPPM4ub(fname1, &h_img1, &w, &h))
    {
        fprintf(stderr, "TID %d Failed to load <%s>\n", my_tid, fname1);
        exit(-1);
    }

    // all threads use device 0, the only on on TK1
    cudaSetDevice(0);

    // create a user-defined stream
    cudaStream_t my_stream;
    cudaStreamCreate(&my_stream);

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

  // Wait for all worker threads to be ready
  // This ensures that all threads have completed the
  // initialization steps before starting to iterate
  // All should begin contending for CE and EE at
  // approximately the same time

  pthread_barrier_wait(&worker_barrier);

  // all setup and initialization complete, start iterations
  printf("TID %d Iterating stereoDisparity CUDA Kernel for %d seconds, %d max loops\n", my_tid, TIME_LENGTH, MAX_LOOPS);
  start_time = now = time(NULL);

  for (i = 0;
            ((now - TIME_LENGTH) < start_time) &&
              i < MAX_LOOPS; i++) {

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

      count++;
      total_count++;

      now = time(NULL);
   } // ends for loop
   elapsed = now - start_time;

    // be sure all locks have been released
    // prepares for cleaning up 
    cudaDeviceSynchronize();

    // calculate sum of resultant GPU image
    // This verification is applied only to the 
    // last result computed

    unsigned int checkSum = 0;

    for (unsigned int i=0 ; i<w *h ; i++)
    {
        checkSum += h_odata[i];
    }
    if (checkSum == 4293895789) //valid checksum only for these two images
       printf("TID %d Test PASSED\n", my_tid);
    else {
       fprintf(stderr, "TID %d verification failed, GPU Checksum = %u, ", my_tid, checkSum);
       exit(-1);
    }

    printf("TID %d completed %ld, duration %ld seconds\n", my_tid, count, elapsed);

#ifdef WRITE_DISPARITY
    // write out the resulting disparity image.
    // creates file in directory containing executable
    unsigned char *dispOut = (unsigned char *)malloc(numData);
    int mult = 20;
    char fnameOut[50] = "";
    sprintf(fnameOut,"TID_%d_", my_tid);
    strcat(fnameOut, "output_GPU.pgm");

    for (unsigned int i=0; i<numData; i++)
    {
        dispOut[i] = (int)h_odata[i]*mult;
    }

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

  // Wait for all threads to complete
  // so all iteration counts are complete
  pthread_barrier_wait(&worker_barrier);

  r->id = my_tid;
  r->ncompleted = total_count;

  pthread_exit(r); // return iteration count from this thread
}

#define OPTSTR "b:n:s:rvf"
int main(int argc, char* argv[])
{
  // set default parameter values
  int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
  bool realtime = false;  // use Linux SCHED_OTHER scheduling class
  bool flat = false;  // use different priority per thread
  time_t runtime = TIME_LENGTH; // time for iterating
  int nthreads = 4;  // 4 threads (== cores on TK1)
  int baseprio = 19;  // nice value for SCHED_OTHER

  pthread_t* workers;
  struct worker_args *wargs;
  struct results *r;
  long throughput = 0;

  int opt;
  while ((opt = getopt(argc, argv, OPTSTR)) != -1)
     {
      switch(opt)
	{
	 case 'b':  // priority for threads. by default, each thread is assigned baseprio+i priority
	    baseprio = atoi(optarg); //base is nice value for SCHED_OTHER, or the real-time priority
                                      //(1-99) where 1 is lowest for SCHED_FIFO
	 break;

         case 's':  // set to control GPU synchronization with host process/thread
            sync_level = atoi(optarg);
            // level 0 - spin polling (busy waiting) for GPU to finish
            // level 1 - yield each time through the polling loop to let another thread run
            // level 2 - block process waiting for GPU to finish
	 break;

	 case 'r':  // use SCHED_FIFO policy and interpret priority as a SCHED_FIFO priority.
	            // otherwise, use SCHED_OTHER and interpret priority as a nice value.
	    realtime = true;
	 break;

	 case 'f':  // make all threads have same priority
	    flat = true;
	 break;

	 case 'n':  // number of CPU worker threads
	    nthreads = atoi(optarg);
	 break;

	 case 'v':
	    verbose = true;
	 break;
	}
     }

  // require a runtime as the last parameter (positional)
  if (argc - optind < 1)
     {
      fprintf(stderr, "Missing runtime argument (seconds).\n");
      exit(-1);
     }
  runtime = atoi(argv[optind + 0]);

  printf("Test: %d threads, %s, for %ld seconds.\n", nthreads, (realtime)? "SCHED_FIFO" : "SCHED_OTHER", runtime);

  // Set main priority before the CUDA runtime has a chance to spawn any signaling threads.
  // This way, those CUDA threads will adopt the policy and priority we set here.
  // Note that this prioritization is set up only for SCHED_FIFO
  if (realtime)
     {
      // set the main process scheduling class and priority
      setsched(SCHED_FIFO, baseprio+nthreads+1); // +1 over highest worker thread
                                                 // so ALL signals will have highest priority
     }

  // set the device flags for CPU and GPU program synchronization
  // before CUDA runtime is initialized
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

  // force initialization of GPU and CUDA runtime
  cuInit(0);
  // use conventional NoOp call to initialize locking
  cudaFree(0);

  pthread_barrier_init(&worker_barrier, 0, nthreads);

  // set parameters for threads and create them
  workers = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  for (int i = 0; i < nthreads; i++)
     {
      wargs = (struct worker_args*)malloc(sizeof(*wargs));
      wargs->runtime = runtime; // time for iteratons
      wargs->id = i;
      wargs->cpu = i % ncpus; // distribute threads among CPUs
      if (realtime)
	 {
	  wargs->sched_policy = SCHED_FIFO;
	  // assign an increasing priority (unless 'flat' is true)
	  wargs->sched_priority = (!flat) ? baseprio + i + 1 : baseprio;
	 }
      else
	 {
	  wargs->sched_policy = SCHED_OTHER;
	  // assign a decreasing priority, becoming less nice (unless 'flat' is true)
          // multiplier of 4 because Linux treats 4 prioities as an equivalence class
	  wargs->sched_priority = (!flat) ? baseprio - i*4 : baseprio;
	 }
      pthread_create(&workers[i], 0, work, wargs);
     }

  // Wait for threads to complete and print out statistics. Worker
  // threads collectively wait on a barrier before exiting, so all
  // threads will be done with GPU work by the time the first call
  // to pthread_join() returns.
  for (int i = 0; i < nthreads; i++)
     {
      pthread_join(workers[i], (void**)&r);

      throughput += r->ncompleted;

      free(r);
     }

  fprintf(stdout, "total 'stereo images': %lu\n", throughput);
  free(workers);

  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();

  return 0;
}

