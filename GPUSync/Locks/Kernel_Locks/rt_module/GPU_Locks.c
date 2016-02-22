/*
 * This kernel loadable module implements the GPU locks and locking API
 * as an in-kernel extension.  It uses the kernel rt_mutex support to implement 
 * GPU CE (copy engine) and EE (execution engine) Locks.  rt_mutex protocols
 * include priority inheritance when priority inversions are detected.
 * Currently there are only two locks/mutexes supporting a single
 * copy and execute engine pair as found in the NVIDIA Jetson
 * TK1 which has an integrated GPU.  Adding support for a second
 * copy engine as would be typical for a discrete GPU should be trivial.
 *
 * State is maintained for the locks to enforce the rule that a process may only
 * hold one of the locks at a time (no deadlocks allowed).  The rule that a lock
 * can be held by only one process is enforced by the kernel rt_mutex implementation.
 *
 * The interface to the kernel module is implemented as a "virtual" file
 * in the debugfs file system (usually) mounted at /sys/kernel/debug.
 * Basically a system-call-like API is provided by writing a string
 * identifying the requested operation into the "virtual" file that has
 * a name that is shared by user programs and this kernel module.  The write
 * operation on this file is actually implemented in this module
 * which is passed a buffer pointer containing the request string.  If
 * an error condition is found by the kernel module, the return value 
 * is < 0.
 *
 * The Makfile for this module generates the loadable module GPU_Locks.ko.  
 * It can be inserted into the kernel with the insmod command run as root, i.e.
 *    sudo insmod GPU_Locks.ko
 *
 * WARNING: It is possible for a process holding one of the shared
 * GPU locks to fail before releasing the lock.  There is no 
 * automatic detection and cleanup in such cases.  It is then necessary
 * to remove the module from the kernel with the rmmod command and insert 
 * it again using the insmod command.
 *
 * Written by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill.
 * 2015.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/debugfs.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/mutex.h>
#include <linux/wait.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/sched.h>
#include <linux/rtmutex.h>
#include <linux/list.h>
#include <linux/types.h>
#include <linux/ktime.h>

#include "../GPU_Locks_kernel.h" // define shared file name

int file_value;
struct dentry *dir, *file;

// local lock for critical section searching and updating the table
DEFINE_SPINLOCK(task_tab_lock);

#define MAX_TASK 100  // WARNING: limits the number of unique thread PIDs
                      // that use GPU locks each time the module is loaded
/*
 * This structure contains the lock state for a process and the threads that it
 * creates.  It is shared among all threads created by a process.  Each thread has
 * an entry in the table identified by its thread ID (see function gettid below).
 */
static struct task {
  pid_t GPU_task;  // identifies a thread by its PID
  struct rt_mutex *LockHeld;  // address of mutex held by thread or NULL
} task_tab[MAX_TASK];

// The lock mutexes and pointers to them
struct rt_mutex EE_rtm;
struct rt_mutex CE_rtm;
struct rt_mutex *EE_rtmptr = &EE_rtm;
struct rt_mutex *CE_rtmptr = &CE_rtm;

u64 start; //WARNING elapsed time (when) maximum is about one hour

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

int GPU_UnLock(int task_id, pid_t cur_pid)
{
  u32 when;

  if (task_tab[task_id].LockHeld != NULL) {
     when = (u32)(((u64)ktime_to_us(ktime_get())) - start);
     if (task_tab[task_id].LockHeld == EE_rtmptr)
        printk(KERN_DEBUG "GPU_Locks: PID %d EE 0 %u\n", cur_pid, when);
     else 
        printk(KERN_DEBUG "GPU_Locks: PID %d CE 0 %u\n", cur_pid, when);

     // unlock the mutex representing the held lock
     rt_mutex_unlock(task_tab[task_id].LockHeld);
     task_tab[task_id].LockHeld = NULL;
  }
  return 0;
}

/*
 * This function acquires the mutex representing the execution engine (EE) lock.
 * If the lock is already held by another process/thread, the requesting thread
 * will block here until that lock is released.  This call is used in the wrapper
 * for cudaLaunch.
 *
 * If the thread currently holds the EE lock, the call becomes a NOOP.  If the
 * thread already holds the CE lock, an error status is returned for the write().
 */

int EE_Lock(int task_id, pid_t cur_pid)
{
  u32 when;

  if (task_tab[task_id].LockHeld == EE_rtmptr)
     return 0;
  if (task_tab[task_id].LockHeld != NULL)
     return -EPERM;

  // acquire the lock or block thread until it is free
  rt_mutex_lock(EE_rtmptr);
  when = (u32)(((u64)ktime_to_us(ktime_get())) - start);
  printk(KERN_DEBUG "GPU_Locks: PID %d EE 1 %u\n", cur_pid, when);

  task_tab[task_id].LockHeld = EE_rtmptr; // thread now holds EE lock
  return 0;
}

/* This function releases the mutex representing the execution engine (EE) lock.
 * If the calling thread does not hold the EE lock, an error status is returned 
 * for the write().
 *
 * This call is not currently used in the wrappers.
 */

int EE_UnLock(int task_id, pid_t cur_pid)
{
  u32 when;

  if (task_tab[task_id].LockHeld != EE_rtmptr)
    return -EPERM;
  when = (u32)(((u64)ktime_to_us(ktime_get())) - start);
  printk(KERN_DEBUG "GPU_Locks: PID %d EE 0 %u\n", cur_pid, when);

  // unlock the mutex representing the held lock
  rt_mutex_unlock(EE_rtmptr);
  task_tab[task_id].LockHeld = NULL; // thread no longer holds a lock
  return 0;
}

/*
 * This function acquires the mutex representing the copy engine (CE) lock.
 * If the lock is already held by another process/thread, the requesting thread
 * will block here until that lock is released.  This call is used in the wrappers
 * for cudaMemcpyAsync() and cudaMemcpy().
 *
 * If the thread currently holds the CE lock, the call becomes a NOOP.  If the
 * thread already holds the EE lock, an error status is returned for the write(). 
 */

int CE_Lock(int task_id, pid_t cur_pid)
{
  u32 when;

  if (task_tab[task_id].LockHeld == CE_rtmptr)
     return 0;
  if (task_tab[task_id].LockHeld != NULL)
     return -EAGAIN;

  // acquire the lock or block thread until it is free
  rt_mutex_lock(CE_rtmptr);
  when = (u32)(((u64)ktime_to_us(ktime_get())) - start);
  printk(KERN_DEBUG "GPU_Locks: PID %d CE 1 %u\n", cur_pid, when);

  task_tab[task_id].LockHeld = CE_rtmptr; // thread now holds the CE lock
  return 0;
}

/* This function releases the mutex representing the copy engine (CE) lock.
 * If the calling thread does not hold the CE lock, an error status is returned
 * for the write(). This call is used in the wrapper for cudaMemcpy().
 */

int CE_UnLock(int task_id, pid_t cur_pid)
{
  u32 when;

  if (task_tab[task_id].LockHeld != CE_rtmptr)
    return -EPERM;
  when = (u32)(((u64)ktime_to_us(ktime_get())) - start);
  printk(KERN_DEBUG "GPU_Locks: PID %d CE 0 %u\n", cur_pid, when);

  // unlock the mutex representing the held lock
  rt_mutex_unlock(CE_rtmptr);
  task_tab[task_id].LockHeld = NULL; // thread no longer holds a lock
  return 0;
}

/*
 * This function is called by the kernel file system when a write() system call is
 * executed in user space using the file handle of the shared file in debugfs.  The
 * user space buffer address, write count and file pointer are passed on the call.
 * This module copies the data from the buffer, checks it for a valid API request 
 * string, and performs the operation.  Any error detected will result in a returned
 * error code (< 0) or 0 for no error.
 */
 
static ssize_t GPU_lock_call(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    int rc;
    int i;
    struct task_struct *cur_task;
    pid_t cur_pid = 0;
    struct pid *pid_p;

    char callbuf[MAX_CALL];

    if (count >= MAX_CALL)
        return -EINVAL;

    // use kernel function that copies data across user/kernel protection boundary    
    rc = copy_from_user(callbuf, buf, count);
    callbuf[MAX_CALL - 1] = '\0';
    *ppos = 0;  // reset file pointer to initial byte each time

    // lock the state table for search and update
    spin_lock(&task_tab_lock);

    // obtain the PID for the calling thread
    // this uses kernel functions to find the PID for a task_struct
    cur_task = current;  // current is always a pointer to the task_stuct
    pid_p = get_task_pid(cur_task, PIDTYPE_PID);
    if (pid_p != NULL)
      cur_pid = pid_vnr(pid_p);
    else {
          printk(KERN_DEBUG "GPU_Locks: Invalid PID\n");
          spin_unlock(&task_tab_lock);
          return -EINVAL;
    }
/*
 * Search the task_tab array for the entry for the
 * current executing thread.  If the PID of the current thread is in the
 * table, its array index is returned.  If an array element with the value
 * zero is encountered, the thread is not in the table so it is allocated
 * at that array index and the index returned.  Note that threads are allocated
 * an array element in the order they make their first locking call and are
 * never deleted.
 */
    for (i = 0; i < MAX_TASK; i++) {
        if (task_tab[i].GPU_task == cur_pid)
	   break;
        if (task_tab[i].GPU_task == 0) {
           task_tab[i].GPU_task = cur_pid;
           break;
	}
    }
    spin_unlock(&task_tab_lock);

    // Thread not in table and no empty slots remain
    if (i == MAX_TASK) {
        printk(KERN_DEBUG "GPU_Locks: GPU tasks exceed %d\n", MAX_TASK);
        return -EINVAL;           
    }

/*
 * Determine the request made by the calling thread and perform the
 * operation.  If no valid request found, return error status for write().
 */
    if (0 == strcmp("GPU_UnLock", callbuf)) {
      return (GPU_UnLock(i, cur_pid));
    } else if (0 == strcmp("EE_Lock", callbuf)) {
      return (EE_Lock(i, cur_pid));
    } else if (0 == strcmp("EE_UnLock", callbuf)) {
      return (EE_UnLock(i, cur_pid));
    } else if (0 == strcmp("CE_Lock", callbuf)) {
      return (CE_Lock(i, cur_pid));
    } else if (0 == strcmp("CE_UnLock", callbuf)) {
      return (CE_UnLock(i, cur_pid));
    } else {
        printk(KERN_DEBUG "GPU_Locks: invalid callbuf: %s\n", callbuf);
        return -EINVAL;           
    }
    return 0;
}

/*
 * This function is called by the kernel file system when a read() system call is
 * executed in user space using the file handle of the shared file in debugfs.  The
 * call is essentially a NOOP since a read() call is not needed.
 */
static ssize_t GPU_lock_return(struct file *file, char __user *userbuf,
                                size_t count, loff_t *ppos)
{
    *ppos = 0; /* reset the offset to zero */
    return 0; /* read() calls return the number of bytes read */
}

// defines mapping between file system operations and functions in 
// this module that implement them
static const struct file_operations my_fops = {
    .write = GPU_lock_call,
    .read = GPU_lock_return,
};

/* This function is called from the insmod command when the module is
 * inserted in the kernel.  It performs all necessary initialization
 * including creating the debugfs file used for emulating a system
 * call interface between user space and the kernel module.  It also
 * initializes the mutexes and the table holding thread lock state.
 */
static int __init GPU_lock_module_init(void)
{
  int i;

    dir = debugfs_create_dir(dir_name, NULL);
    if (dir == NULL)
    {
        printk(KERN_DEBUG "GPU_Locks: error creating %s directory\n", dir_name);
        return -ENODEV;
    }
    
    /* create the in-memory file used for communication;
     * make the permission read+write by "world"
     */
    file = debugfs_create_file(file_name, 0666, dir, &file_value, &my_fops);
    if (file == NULL) {
        printk(KERN_DEBUG "GPU_Locks: error creating %s file\n", file_name);
        return -ENODEV;
    }

    printk(KERN_DEBUG "GPU_Locks: created new debugfs directory and file\n");

    rt_mutex_init(EE_rtmptr);
    rt_mutex_init(CE_rtmptr);

    for (i = 0; i < MAX_TASK; i++) {
         task_tab[i].GPU_task = 0;
         task_tab[i].LockHeld = NULL;
    }

    start = (u64)ktime_to_us(ktime_get());
    return 0;
}

/* 
 * This function is called from the rmmod command when the module is removed
 * from the kernel.  It removes the debugfs file created during initialization.
 */
 
static void __exit GPU_lock_module_exit(void)
{
    debugfs_remove(file);
    debugfs_remove(dir);
    printk(KERN_DEBUG "GPU_Locks: removed debugfs directory and file\n");

}

// required for kernel modules to define initialization and cleanup functions
module_init(GPU_lock_module_init);
module_exit(GPU_lock_module_exit);
MODULE_LICENSE("GPL");
