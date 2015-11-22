/*
 * Definition of the file name for the "virtual" file used as
 * the interface between user space (GPU_Locklib.c) and kernel
 * space (GPU_Locks.c) for lock/unlock calls.  Both must share
 * a common name for the "virtual" file that is in the debugfs
 * file system, usually mounted by default at
 *   /sys/kernel/debug
 * The debug directory must have r+x privileges for all users.
 *
 */

#define MAX_CALL 100
char syscall_location[] = "/sys/kernel/debug/";
char dir_name[] = "GPU_Lock";
char file_name[] = "call";
