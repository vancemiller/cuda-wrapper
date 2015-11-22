/*
 * Create and initialize the shared GPU resource locks.
 * This module implements a shared memory segment that contains
 * the mutex implementation of GPU CE (copy engine) and EE 
 * (execution engine) Locks.  The sole purpose is to create the
 * shared memory segment and initialize the mutex attributes.
 * Currently there are only two locks/mutexes supporting a single
 * copy and execute engine pair as found in the NVIDIA Jetson
 * TK1 which has an integrated GPU.  Adding support for a second
 * copy engine as would be typical for a discrete GPU should be
 * trivial.
 *
 * The program is intended to be run as a non-terminating 
 * background job so that the shared memory segment lasts as long
 * as there are GPU-using processes active (think of it as a 
 * user space daemon).  The intended invocation at the command
 * line is:
 * % GPU_Locks &
 *
 * WARNING: It is possible for a process holding one of the shared
 * GPU locks to fail before releasing the lock.  There is no 
 * automatic detection and cleanup in such cases.  It is then necessary
 * to kill the GPU_Locks process (send SIGTERM signal so it will
 * delete the shared segment containing the locks) and then restart
 * GPU_Locks to reinitialize.  It is also important to check that
 * the file /dev/shm/shm-GPU_Locks was deleted and delete it manually
 * if it was not deleted by sending SIGTERM (this can happen if one or
 * more GPU programs is still running and has mapped the shared memory
 * segment).
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
#include <errno.h>
#include <signal.h>

// necessary for the initialization of attributes
#define __USE_UNIX98
#include <pthread.h>


/*
 * The shared space is allocated as a named file typically in
 * /dev/shm 
 */
const char *name = "/shm-GPU_Locks";	// file name
const int SIZE = 4096;		// file size

int shm_fd;		// file descriptor, from shm_open()

pid_t my_pid;

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
 * pointer (not shared) to the structure in the mappped file space.
 * The buffer_t structure is the first and only data allocated in
 * the shared space.
 */

buffer_t *buffer;

// pointers (not shared) to the mutexes
pthread_mutex_t    *EE_mptr; //Mutex Pointer                                                        
pthread_mutex_t    *CE_mptr; //Mutex Pointer                                                        

pthread_mutexattr_t matr; //Mutex Attribute                                                      

/*
 * Handle the SIGTERM signal (sent by default by the kill
 * command with no signal specified). Delete the shared
 * memory segment containing the mutexes and terminate.
 */
void term_handler(int signum) 
{
  printf("Deleting GPU Lock shared memory\n");
  shm_unlink(name);
  exit(-1);
}


int main(void)
{
 /* Unused variables. Remove?
  char *shm_base;	// base address, from mmap()
  char *ptr;		// shm_base is fixed, ptr is movable

  pthread_t th;
  int i;*/

  int rtn = 0;
 
  // create the signal handler to delete shared memory
  signal(SIGTERM, term_handler);

  my_pid = getpid();

  /* create the shared memory segment as if it was a file */
  shm_fd = shm_open(name, O_CREAT | O_RDWR | O_EXCL, 0666);
  /* if shared memory exists, do not create again*/
  if (shm_fd == -1) {
     printf("PID %d found existing locks\n", my_pid);
     exit (-1);
  }
  else {
     /* configure the size of the shared memory segment */
     ftruncate(shm_fd, SIZE);

     /* map the shared memory segment to the address space of the process and
      * set the pointer to the structure with mutexes
      */
     buffer = (buffer_t*)mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
     if (buffer == MAP_FAILED) {
        printf("prod: Map failed: %s\n", strerror(errno));
       // close and shm_unlink?
       exit(1);
     }

     EE_mptr = &(buffer->EE_mutex);
     CE_mptr = &(buffer->CE_mutex);

     printf("PID %d does initialization\n", my_pid);

     // Setup Mutex and initializers                                                                                
    if ((rtn = pthread_mutexattr_init(&matr)))
       {
        fprintf(stderr,"pthreas_mutexattr_init: %s",strerror(rtn)),exit(1);
       }

    // set the attribute that makes the mutex shared among processes
    if ((rtn = pthread_mutexattr_setpshared(&matr,PTHREAD_PROCESS_SHARED)))
       {
        fprintf(stderr,"pthread_mutexattr_setpshared %s",strerror(rtn)),exit(1);
       }

    // set the atrribute that enables priority inheritance in case of priority inversions
    if ((rtn = pthread_mutexattr_setprotocol(&matr,PTHREAD_PRIO_INHERIT)))
       {
        fprintf(stderr,"pthread_mutexattr_setprotocol %s",strerror(rtn)),exit(1);
       }

    // initialize both mutexes with the same attributes
    if ((rtn = pthread_mutex_init(EE_mptr, &matr)))
       {
        fprintf(stderr,"pthread_mutex_init EE %s",strerror(rtn)), exit(1);
       }
    if ((rtn = pthread_mutex_init(CE_mptr, &matr)))
       {
        fprintf(stderr,"pthread_mutex_init CE %s",strerror(rtn)), exit(1);
       }
  }

  // idle forever until killed */                                                            
  while (1) {
    sleep(60);
  }
  return 0;
}


