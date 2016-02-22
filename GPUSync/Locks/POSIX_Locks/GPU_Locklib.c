/*
 * This program implements the call interface to acquire and release GPU locks. 
 * It is implemented as a library that is compiled and linked with the
 * cuda wrappers dynamic load module (see the Makefile).  It can also be 
 * compiled and linked with a Cuda program for using the lock calls directly
 * in a program instead of using the wrappers.  Each process (including those
 * that create threads) effectively has a copy of this program.
 *
 * The library depends on a shared memory segment (allocated and initialized by
 * the GPU_Locks.c program) that contains the global GPU locks implemented by
 * shared POSIX mutexes.  The shared segment contains the mutex implementation of 
 * GPU CE (copy engine) and EE (execution engine) Locks. 
 * Currently there are only two locks/mutexes supporting a single
 * copy and execute engine pair as found in the NVIDIA Jetson
 * TK1 which has an integrated GPU.  Adding support for a second
 * copy engine as would be typical for a discrete GPU should be trivial.
 *
 * State is maintained for the locks to enforce the rule that a process may only
 * hold one of the locks at a time (no deadlocks allowed).  The rule that a lock
 * can be held by only one process is enforced by the POSIX mutex implementation.
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
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <errno.h>

#include <pthread.h>

/*
 * The shared space is allocated as a named file typically in
 * /dev/shm 
 */
const char *name = "/shm-GPU_Locks";	// file name
const int SIZE = 4096;		// file size

int Initialized = 0;  //Only initialize once -- set to 1 the first time

/*
 * This structure is the sole occupant of the shared memory space and
 * contains the two mutexes that implement the GPU locks.
 */
typedef struct {
  pthread_mutex_t   EE_mutex;
  pthread_mutex_t   CE_mutex;
  int       the_data;
} buffer_t;

/*
 * pointer to the structure in the mappped file space.
 * The buffer_t structure is the first and only data allocated in
 * the shared space.
 */
buffer_t *buffer;

// pointers to the individual mutexes in the shared space
pthread_mutex_t    *EE_mptr; //Mutex Pointer                                                        
pthread_mutex_t    *CE_mptr; //Mutex Pointer                                                        

/*
 * This structure contains the lock state for a process and the threads that it
 * creates.  It is shared among all threads created by a process.  Each thread has
 * an entry in the table identified by its thread ID (see function gettid below).
 */
#define MAX_TASK 100 // WARNING - limits the number of threads created by a process
static struct task {
  pid_t GPU_task;   // identifies a thread
  pthread_mutex_t *LockHeld;  // address of mutex held by task or NULL
} task_tab[MAX_TASK];

// local mutex for critical section searching and updating the table.
// see get_task() function
pthread_mutex_t task_tab_lock = PTHREAD_MUTEX_INITIALIZER;

// used to get thread TID
inline pid_t gettid(void)
{
  return syscall(__NR_gettid);
}

/*
 * This function is used to search the task_tab array for the entry for the
 * current executing thread.  If the task ID of the current task is in the
 * table, its array index is returned.  If an array element with the value
 * zero is encountered, the task is not in the table so it is allocated
 * at that array index and the index returned.  Note that tasks are allocated
 * an array element in the order they make their first locking call and are
 * never deleted.
 *
 * Note that the array index returned is used by only one thread in this process
 * (and each process has its own table instance).  Thus the calling thread can
 * safely use the returned task_tab entry for itself so a lock is not needed in other 
 * XX_Lock() or XX_UnLock() functions that use this index to access the thread's
 * table entry.
 */
int get_task(void)
{
  int i;
  int task_id;
  pid_t cur_task;

  // lock the table for search and possible update
  pthread_mutex_lock(&task_tab_lock);

  // get the TID of the currently executing task
  cur_task = gettid();
  for (i = 0; i < MAX_TASK; i++) {
      if (task_tab[i].GPU_task == cur_task)
          break;
      if (task_tab[i].GPU_task == 0) {
          task_tab[i].GPU_task = cur_task;
          break;
      }
  }
  task_id = i;
  // unlock the table
  pthread_mutex_unlock(&task_tab_lock);

  if (i == MAX_TASK) {
     fprintf(stderr, "GPU_Locks: GPU tasks exceed %d\n", MAX_TASK);
     exit(-1);           
  }
  return(task_id); // returns array index for the thread
}

/*
 * Function to map the shared memory initialized by GPU_Locks.c into
 * the address space of this process so it can access the shared mutexes
 * that implement the GPU resource locks.  It also initializes the array
 * task_tab that will hold lock state for each thread.  This function
 * should be called only once in each process.  When using the Cuda call
 * wrappers, this call is embedded in the interception of a 
 * cudaFree(0) call which is commonly used in Cuda programs to initialize
 * the GPU and Cuda runtime before any other Cuda calls.  It is required
 * to be used this way (and called only once) by convention for use with
 * the Cuda call wrappers.  Any threads created by the calling process
 * should be created after this initialization and will share the buffer
 * pointer to the shared memory segment containing the mutexes.
 *
 */
void GPU_LockInit(void)
{
  int i;
  pid_t my_pid;
  int shm_fd;		// file descriptor, from shm_open()

  my_pid = getpid();

  // It is an error to call this more than once in a process
  if (Initialized != 0) {
      fprintf(stderr, "GPU_LockInit: Initialization Re-entered\n");
      exit(1);
  }

  /* open the shared memory segment as if it was a file */
  shm_fd = shm_open(name, O_RDWR, 0666);
  /* if shared memory exists, assume it has been initialized by GPU_Locks.c */
  if (shm_fd != -1) {
     /* map the shared memory segment to the address space of the process and
      * set the pointer to the structure with mutexes
      */
      buffer = (buffer_t*)mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
      if (buffer == MAP_FAILED) {
          fprintf(stderr, "GPU_LockInit: Map failed: %s\n", strerror(errno));
         // close and shm_unlink?
         exit(1);
      }

     // record the addresses of the shared mutexes
     EE_mptr = &(buffer->EE_mutex);
     CE_mptr = &(buffer->CE_mutex);

     // lock the tabke for initialization 
     pthread_mutex_lock(&task_tab_lock);
     for (i = 0; i < MAX_TASK; i++) {
          task_tab[i].GPU_task = 0;
          task_tab[i].LockHeld = NULL;
     }
     Initialized = 1; // mark to detect multiple calls

     // unlock the table  
     pthread_mutex_unlock(&task_tab_lock);

     fprintf(stderr, "GPU - CUDA calls Intercepted, PID %d\n", my_pid);
  }
  else {
    printf("PID %d Failed -- No GPU locks found\n", my_pid);
    exit (-1);
  }
}

/*
 * A generic unlock function that releases the lock currently held by the
 * calling thread.  It is used in the wrapper for cudaStreamSynchronize().  
 * This call should be used by convention in Cuda programs using the streams 
 * model following calls to asynchronous copy functions and kernel launches.  
 * It will release either a CE lock acquired by the wrapper of cudaMemcpyAsync() 
 * or a EE lock acquired by the wrapper of cudaLaunch().  It is also used in the 
 * wrapper for cudaDeviceSynchronize() which should be used by convention in cuda 
 * programs using the default stream (stream 0) following kernel launches 
 * (kernels are always asynchronous).
 * 
 * If the thread does not currently hold a lock, the call becomes a NOOP.
 */
void GPU_UnLock(void)
{
  int rc;
  int task_id;
  pid_t my_pid;

  // get the task_tab array index for the current thread
  task_id = get_task();

  my_pid = gettid();

  if (task_tab[task_id].LockHeld != NULL) {
     // unlock the mutex representing the held lock
     rc = pthread_mutex_unlock(task_tab[task_id].LockHeld);
     if (rc != 0) {
        fprintf(stderr, "PID %d Failed - ?? mutex_unlock\n", my_pid);
        exit (-1);
     }
     task_tab[task_id].LockHeld = NULL; // thread no longer holds a lock
  }
  return;
}

/*
 * This function acquires the mutex representing the execution engine (EE) lock.
 * If the lock is already held by another process/thread, the requesting thread
 * will block here until that lock is released.  This call is used in the wrapper
 * for cudaLaunch.
 *
 * If the thread currently holds the EE lock, the call becomes a NOOP.  If the 
 * thread already holds the CE lock, the process exits with a failure status.
 */
void EE_Lock(void) 
{
  int rc;
  int task_id;
  pid_t my_pid;

  // get the task_tab array index for the current thread
  task_id = get_task();

  my_pid = gettid();

  if (task_tab[task_id].LockHeld == EE_mptr)
     return;
  if (task_tab[task_id].LockHeld != NULL) {
      fprintf(stderr, "PID %d Failed - EE lock request while holding lock\n", my_pid);
      exit (-1);
  }

  // acquire the lock or block until it is free
  rc = pthread_mutex_lock(EE_mptr);
  if (rc != 0) {
      fprintf(stderr, "PID %d Failed - EE mutex_lock\n", my_pid);
      exit (-1);
  }
  task_tab[task_id].LockHeld = EE_mptr; // thread now holds the EE lock
  return;
}

/* This function releases the mutex representing the execution engine (EE) lock.
 * If the calling thread does not hold the EE lock, the process exits with a
 * failure status.
 *
 * This call is not currently used in the wrappers.
 */
void EE_UnLock(void)
{
  int rc;
  int task_id;
  pid_t my_pid;


  // get the task_tab array index for the current thread
  task_id = get_task();

  my_pid = gettid();

  if (task_tab[task_id].LockHeld != EE_mptr) {
     fprintf(stderr, "PID %d Failed - EE unlock for lock not held\n", my_pid);
     exit (-1);
  }

  // release the mutex representing the EE lock
  rc = pthread_mutex_unlock(EE_mptr);
  if (rc != 0) {
      fprintf(stderr, "PID %d Failed - EE mutex_unlock\n", my_pid);
      exit (-1);
  }
  task_tab[task_id].LockHeld = NULL; // thread no longer holds a lock
  return;
}

/*
 * This function acquires the mutex representing the copy engine (CE) lock.
 * If the lock is already held by another process/thread, the requesting thread
 * will block here until that lock is released.  This call is used in the wrappers
 * for cudaMemcpyAsync() and cudaMemcpy().
 *
 * If the thread currently holds the CE lock, the call becomes a NOOP.  If the 
 * thread already holds the EE lock, the process exits with a failure status.
 */

void CE_Lock(void)
{
  int rc;
  int task_id;
  pid_t my_pid;

  // get the task_tab array index for the current thread
  task_id = get_task();

  my_pid = gettid();

  if (task_tab[task_id].LockHeld == CE_mptr)
     return;
  if (task_tab[task_id].LockHeld != NULL) {
      fprintf(stderr, "PID %d Failed - CE lock request while holding lock\n", my_pid);
      exit (-1);
  }

  // acquire the lock or block until it is free
  rc = pthread_mutex_lock(CE_mptr);
  if (rc != 0) {
      fprintf(stderr, "PID %d Failed - CE mutex_lock\n", my_pid);
      exit (-1);
  }
  task_tab[task_id].LockHeld = CE_mptr; // thread now holds the CE lock
  return;
}

/* This function releases the mutex representing the copy engine (CE) lock.
 * If the calling thread does not hold the CE lock, the process exits with a
 * failure status. This call is used in the wrapper for cudaMemcpy().
 */
void CE_UnLock(void)
{
  int rc;
  int task_id;
  pid_t my_pid;

  // get the task_tab array index for the current thread
  task_id = get_task();

  my_pid = gettid();

  if (task_tab[task_id].LockHeld != CE_mptr) {
      fprintf(stderr, "PID %d Failed - CE unlock for lock not held\n", my_pid);
      exit (-1);
  }

  // release the mutex representing the CE lock
  rc = pthread_mutex_unlock(CE_mptr);
  if (rc != 0) {
      fprintf(stderr, "PID %d Failed - CE mutex_unlock\n", my_pid);
      exit (-1);
  }
  task_tab[task_id].LockHeld = NULL; // thread no longer holds a lock
  return;
}
