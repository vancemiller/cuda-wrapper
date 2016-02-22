/*
 * This include file provides function prototypes for the
 * GPU lock functions implemented by GPU_Locklib.c and used
 * in the Cuda wrapper functions.
 */
void GPU_LockInit(void);
void GPU_UnLock(void);
void EE_Lock(void);
void EE_UnLock(void);
void CE_Lock(void);
void CE_UnLock(void);
