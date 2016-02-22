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
 * A configurable number of user threads are created, each of which
 * independently performs iterations of the vector addition and 
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
#define MAX_LOOPS 10000  // iteration count
#define TIME_LENGTH 30   // elapsed time (seconds)
#define CUDA_CORES 192

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <cuda.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// A few global variables that will be shared by all threads

// Used to synchronize start of thread loops after initialization 
pthread_barrier_t worker_barrier;

// parameter defaults (see main() for more details)
bool verbose = false;
int sync_level = 2; //default -- process block

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
  int id;
  long ncompleted;
};

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

  struct sched_param params;  //parameter for kernel call

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

  if (SCHED_OTHER == policy)  // Default Linux "fair" scheduler
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
  struct worker_args args = *(struct worker_args*)_args;

  long count = 0, total_count = 0;
  time_t start_time, now, elapsed; 
  int i;
  // allocate struct to return results
  struct results* r = (struct results*)malloc(sizeof(*r));
  pid_t my_tid;

  // set parameters for the vectorAdd GPU kernel
  int numElements = CUDA_CORES / 2;
  size_t size = numElements * sizeof(float);  //16,000,000 bytes

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;

  int blocksPerGrid;

  my_tid = gettid();  //each thread has a unique thread ID (TID)
  free(_args);

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

  // all threads use device 0
  cudaSetDevice(0);

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // create a user defined stream
  cudaStream_t my_stream;
  cudaStreamCreate(&my_stream);

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
  d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Wait for all worker threads to be ready
  // This ensures that all threads have completed the
  // initialization steps before starting to iterate
  // All should begin contending for CE and EE at
  // approximately the same time

  pthread_barrier_wait(&worker_barrier);

  printf("TID %d Iterating Vector Add CUDA Kernel for %d seconds, %d max loops\n", my_tid, TIME_LENGTH, MAX_LOOPS);
  start_time = now = time(NULL);

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
      blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

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
      
      count++;
      total_count++;

      now = time(NULL);
     } // ends for loop
  elapsed = now - start_time;

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
  printf("TID %d Test PASSED\n", my_tid);
  printf("TID %d completed %ld, duration %ld seconds\n", my_tid, total_count, elapsed);

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

  // Wait for all threads to complete
  // so all iteration counts are complete
  pthread_barrier_wait(&worker_barrier);

  r->id = args.id;
  r->ncompleted = total_count; // return iteration count from this thread

  pthread_exit(r);
}

#define OPTSTR "b:n:s:rvf"
int main(int argc, char* argv[])
{
  // set default parameter values
  int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
  bool realtime = false; // use Linux SCHED_OTHER scheduling class
  bool flat = false;  // use different priority per thread
  time_t runtime = TIME_LENGTH;  // time for iterating
  int nthreads = 4;  // 4 threads (== cores on TK1)
  int baseprio = 19; // nice value for SCHED_OTHER

  pthread_t* workers;
  struct worker_args *wargs;
  struct results *r;
  
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

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

  // force initialization of cuda runtime
  cuInit(0);
  cudaFree(0); // used here to invoke initialization of GPU locking

  pthread_barrier_init(&worker_barrier, 0, nthreads);

  // set parameters for threads and create them
  workers = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
  for (int i = 0; i < nthreads; i++)
     {
      wargs = (struct worker_args*)malloc(sizeof(*wargs));
      wargs->runtime = runtime; // time for iterations
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


  fprintf(stdout, "total 'frames': %lu\n", throughput);
  free(workers);

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
