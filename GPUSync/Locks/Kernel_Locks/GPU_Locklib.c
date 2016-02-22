/*
 * This program implements the API to acquire and release GPU locks. 
 * It is implemented as a library that is compiled and linked with the
 * cuda wrappers dynamic load module (see the Makefile).  It can also be 
 * compiled and linked with a Cuda program for using the lock calls directly
 * in a program instead of using the wrappers.  Each process (including those
 * that create threads) effectively has a copy of this program.
 *
 * The library uses the kernel loadable module, GPU_Locks.c, that contains the 
 * mutex implementation of GPU CE (copy engine) and EE (execution engine) Locks. 
 * Currently there are only two locks/mutexes supporting a single
 * copy and execute engine pair as found in the NVIDIA Jetson
 * TK1 which has an integrated GPU.  Adding support for a second
 * copy engine as would be typical for a discrete GPU should be trivial.
 *
 * The interface to the kernel module is implemented as a "virtual" file
 * in the debugfs file system (usually) mounted at /sys/kernel/debug.
 * Basically a system-call-like API is provided by writing a string
 * identifying the requested operation into the "virtual" file that has
 * a name that is shared by the library and kernel module.  The write
 * operation on this file is actually implemented in the kernel module
 * which is passed a buffer pointer containing the request string.  If
 * an error condition is found by the kernel module, the return value 
 * is < 0 and the calling process is terminated (see the do_syscall()
 * function. 
 *
 * IMPORTANT: Assumes that processes will create threads only with the POSIX 
 * API call pthread_create() and not use a system call like clone() directly.
 *
 * Note that all calls have a void return.  If a call returns to the caller, it
 * can assume that the call completed correctly.  If any error occurs, the process
 * is terminated.
 *
 * Written by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill.
 * 2015.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "GPU_Locks_kernel.h"  // define shared file name

int fp;
int Initialized = 0; //Only initialize once -- set to 1 the first time
static void do_syscall(char* name);

/* 
 * Function to open the debugfs file shared with the kernel module.
 * This function should be called only once in each process.  When using 
 * the Cuda call wrappers, this call is embedded in the interception of a 
 * cudaFree(0) call which is commonly used in Cuda programs to initialize
 * the GPU and Cuda runtime before any other Cuda calls.  It is required
 * to be used this way (and called only once) by convention for use with
 * the Cuda call wrappers.  Any threads created by the calling process 
 * should be created after this initialization and will share the file
 * handle returned by open().
 */

void GPU_LockInit(void)
{
  char the_file[256];

  if (Initialized != 0) { // It is an error to call this more than once in a process

    // If threads share memory, they share the state of the locklib, so it may be unavoidable to call this more than once
    return;
    // fprintf(stderr, "GPU_LockInit: Initialization Re-entered\n");
    // exit(-1);
  }

  the_file[0] = '\0';
  strcat(the_file, syscall_location);
  strcat(the_file, dir_name);
  strcat(the_file, "/");
  strcat(the_file, file_name);

  if ((fp = open(the_file, O_RDWR)) == -1) {
    fprintf(stderr, "error opening %s\n", the_file);
    exit(-1);
  }
  Initialized = 1;  // mark to detect multiple calls
  fprintf(stderr, "GPU - CUDA calls Intercepted\n");
  return;
}

void GPU_UnLock(void)
{
  do_syscall("GPU_UnLock");
}

void EE_Lock(void)
{
  do_syscall("EE_Lock");
}

void EE_UnLock(void)
{
  do_syscall("EE_UnLock");
}

void CE_Lock(void)
{
  do_syscall("CE_Lock");
}

void CE_UnLock(void)
{
  do_syscall("CE_UnLock");
}

static void do_syscall(char* name)
{
  int rc;
  char call_buf[MAX_CALL];

  strcpy(call_buf, name);

  rc = write(fp, call_buf, strlen(call_buf) + 1);
  if (rc < 0) {
    pid_t tid = getpid();
    fprintf(stderr, "GPU_Locks: PID %d Call %s Failed: %i\n", tid, name, rc);
    fflush(stderr);
    exit(-1);
  }
  return;
}
