#define _GNU_SOURCE

#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include "cuda_runtime_api.h"

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceReset)(void) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceReset(void) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceReset();
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceSynchronize)(void) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceSynchronize();
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceSetLimit)(enum cudaLimit limit, size_t value) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceSetLimit(limit, value);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetLimit)(size_t *pValue, enum cudaLimit limit) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetLimit(pValue, limit);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetCacheConfig)(enum cudaFuncCache *pCacheConfig) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetCacheConfig(pCacheConfig);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetStreamPriorityRange)(int *leastPriority, int *greatestPriority) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceSetCacheConfig)(enum cudaFuncCache cacheConfig) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceSetCacheConfig(cacheConfig);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetSharedMemConfig)(enum cudaSharedMemConfig *pConfig) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetSharedMemConfig(pConfig);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceSetSharedMemConfig)(enum cudaSharedMemConfig config) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceSetSharedMemConfig(config);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetByPCIBusId)(int *device, const char *pciBusId) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetByPCIBusId(device, pciBusId);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetPCIBusId)(char *pciBusId, int len, int device) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetPCIBusId(pciBusId, len, device);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaIpcGetEventHandle)(cudaIpcEventHandle_t *handle, cudaEvent_t event) = NULL;

__host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaIpcGetEventHandle(handle, event);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaIpcOpenEventHandle)(cudaEvent_t *event, cudaIpcEventHandle_t handle) = NULL;

__host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaIpcOpenEventHandle(event, handle);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaIpcGetMemHandle)(cudaIpcMemHandle_t *handle, void *devPtr) = NULL;

__host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaIpcGetMemHandle(handle, devPtr);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaIpcOpenMemHandle)(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaIpcOpenMemHandle(devPtr, handle, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaIpcCloseMemHandle)(void *devPtr) = NULL;

__host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaIpcCloseMemHandle(devPtr);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaThreadExit)(void) = NULL;

__host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaThreadExit();
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaThreadSynchronize)(void) = NULL;

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaThreadSynchronize();
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaThreadSetLimit)(enum cudaLimit limit, size_t value) = NULL;

__host__ cudaError_t CUDARTAPI cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaThreadSetLimit(limit, value);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaThreadGetLimit)(size_t *pValue, enum cudaLimit limit) = NULL;

__host__ cudaError_t CUDARTAPI cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaThreadGetLimit(pValue, limit);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaThreadGetCacheConfig)(enum cudaFuncCache *pCacheConfig) = NULL;

__host__ cudaError_t CUDARTAPI cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaThreadGetCacheConfig(pCacheConfig);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaThreadSetCacheConfig)(enum cudaFuncCache cacheConfig) = NULL;

__host__ cudaError_t CUDARTAPI cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaThreadSetCacheConfig(cacheConfig);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaGetLastError)(void) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetLastError();
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaPeekAtLastError)(void) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaPeekAtLastError();
  return ret;

}

static __host__ __cudart_builtin__ const char* CUDARTAPI (*orig_cudaGetErrorName)(cudaError_t error) = NULL;

__host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
  const char* ret;
  // Write your own code here
  ret = orig_cudaGetErrorName(error);
  return ret;

}

static __host__ __cudart_builtin__ const char* CUDARTAPI (*orig_cudaGetErrorString)(cudaError_t error) = NULL;

__host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
  const char* ret;
  // Write your own code here
  ret = orig_cudaGetErrorString(error);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaGetDeviceCount)(int *count) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetDeviceCount(count);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaGetDeviceProperties)(struct cudaDeviceProp *prop, int device) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetDeviceProperties(prop, device);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetAttribute)(int *value, enum cudaDeviceAttr attr, int device) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetAttribute(value, attr, device);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaDeviceGetP2PAttribute)(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaChooseDevice)(int *device, const struct cudaDeviceProp *prop) = NULL;

__host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaChooseDevice(device, prop);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaSetDevice)(int device) = NULL;

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaSetDevice(device);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaGetDevice)(int *device) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetDevice(device);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaSetValidDevices)(int *device_arr, int len) = NULL;

__host__ cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaSetValidDevices(device_arr, len);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaSetDeviceFlags)( unsigned int flags ) = NULL;

__host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaSetDeviceFlags(flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetDeviceFlags)( unsigned int *flags ) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetDeviceFlags( unsigned int *flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetDeviceFlags(flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaStreamCreate)(cudaStream_t *pStream) = NULL;

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamCreate(pStream);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamCreateWithFlags)(cudaStream_t *pStream, unsigned int flags) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamCreateWithFlags(pStream, flags);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamCreateWithPriority)(cudaStream_t *pStream, unsigned int flags, int priority) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamCreateWithPriority(pStream, flags, priority);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamGetPriority)(cudaStream_t hStream, int *priority) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamGetPriority(hStream, priority);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamGetFlags)(cudaStream_t hStream, unsigned int *flags) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamGetFlags(hStream, flags);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamDestroy)(cudaStream_t stream) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamDestroy(stream);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamWaitEvent)(cudaStream_t stream, cudaEvent_t event, unsigned int flags) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamWaitEvent(stream, event, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaStreamAddCallback)(cudaStream_t stream,        cudaStreamCallback_t callback, void *userData, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,        cudaStreamCallback_t callback, void *userData, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamAddCallback(stream, callback, userData, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaStreamSynchronize)(cudaStream_t stream) = NULL;

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamSynchronize(stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaStreamQuery)(cudaStream_t stream) = NULL;

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamQuery(stream);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaStreamAttachMemAsync)(cudaStream_t stream, void *devPtr, size_t length , unsigned int flags ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length , unsigned int flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaStreamAttachMemAsync(stream, devPtr, length, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaEventCreate)(cudaEvent_t *event) = NULL;

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventCreate(event);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaEventCreateWithFlags)(cudaEvent_t *event, unsigned int flags) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventCreateWithFlags(event, flags);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaEventRecord)(cudaEvent_t event, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventRecord(event, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaEventQuery)(cudaEvent_t event) = NULL;

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventQuery(event);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaEventSynchronize)(cudaEvent_t event) = NULL;

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventSynchronize(event);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaEventDestroy)(cudaEvent_t event) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventDestroy(event);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaEventElapsedTime)(float *ms, cudaEvent_t start, cudaEvent_t end) = NULL;

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaEventElapsedTime(ms, start, end);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaLaunchKernel)(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) = NULL;

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaLaunchCooperativeKernel)(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) = NULL;

__host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaLaunchCooperativeKernelMultiDevice)(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  ) = NULL;

__host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaFuncSetCacheConfig)(const void *func, enum cudaFuncCache cacheConfig) = NULL;

__host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFuncSetCacheConfig(func, cacheConfig);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaFuncSetSharedMemConfig)(const void *func, enum cudaSharedMemConfig config) = NULL;

__host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFuncSetSharedMemConfig(func, config);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaFuncGetAttributes)(struct cudaFuncAttributes *attr, const void *func) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFuncGetAttributes(attr, func);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaFuncSetAttribute)(const void *func, enum cudaFuncAttribute attr, int value) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFuncSetAttribute(func, attr, value);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaSetDoubleForDevice)(double *d) = NULL;

__host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaSetDoubleForDevice(d);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaSetDoubleForHost)(double *d) = NULL;

__host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaSetDoubleForHost(d);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaOccupancyMaxActiveBlocksPerMultiprocessor)(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaConfigureCall)(dim3 gridDim, dim3 blockDim, size_t sharedMem , cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem , cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaSetupArgument)(const void *arg, size_t size, size_t offset) = NULL;

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaSetupArgument(arg, size, offset);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaLaunch)(const void *func) = NULL;

__host__ cudaError_t CUDARTAPI cudaLaunch(const void *func) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaLaunch(func);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMallocManaged)(void **devPtr, size_t size, unsigned int flags ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMallocManaged(devPtr, size, flags);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMalloc)(void **devPtr, size_t size) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMalloc(devPtr, size);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMallocHost)(void **ptr, size_t size) = NULL;

__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMallocHost(ptr, size);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMallocPitch)(void **devPtr, size_t *pitch, size_t width, size_t height) = NULL;

__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMallocPitch(devPtr, pitch, width, height);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMallocArray)(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height , unsigned int flags ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height , unsigned int flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMallocArray(array, desc, width, height, flags);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaFree)(void *devPtr) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFree(devPtr);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaFreeHost)(void *ptr) = NULL;

__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFreeHost(ptr);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaFreeArray)(cudaArray_t array) = NULL;

__host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFreeArray(array);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaFreeMipmappedArray)(cudaMipmappedArray_t mipmappedArray) = NULL;

__host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaFreeMipmappedArray(mipmappedArray);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaHostAlloc)(void **pHost, size_t size, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaHostAlloc(pHost, size, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaHostRegister)(void *ptr, size_t size, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaHostRegister(ptr, size, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaHostUnregister)(void *ptr) = NULL;

__host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaHostUnregister(ptr);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaHostGetDevicePointer)(void **pDevice, void *pHost, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaHostGetDevicePointer(pDevice, pHost, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaHostGetFlags)(unsigned int *pFlags, void *pHost) = NULL;

__host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaHostGetFlags(pFlags, pHost);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMalloc3D)(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) = NULL;

__host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMalloc3D(pitchedDevPtr, extent);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMalloc3DArray)(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMalloc3DArray(array, desc, extent, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMallocMipmappedArray)(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetMipmappedArrayLevel)(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy3D)(const struct cudaMemcpy3DParms *p) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy3D(p);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy3DPeer)(const struct cudaMemcpy3DPeerParms *p) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy3DPeer(p);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy3DAsync(p, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy3DPeerAsync)(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy3DPeerAsync(p, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemGetInfo)(size_t *free, size_t *total) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemGetInfo(free, total);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaArrayGetInfo)(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array) = NULL;

__host__ cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaArrayGetInfo(desc, extent, flags, array);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy(dst, src, count, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyPeer)(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyToArray)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyFromArray)(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2D)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2DToArray)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2DFromArray)(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2DArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyToSymbol)(const void *symbol, const void *src, size_t count, size_t offset , enum cudaMemcpyKind kind ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset , enum cudaMemcpyKind kind ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyToSymbol(symbol, src, count, offset, kind);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyFromSymbol)(void *dst, const void *symbol, size_t count, size_t offset , enum cudaMemcpyKind kind ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset , enum cudaMemcpyKind kind ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMemcpyAsync)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyAsync(dst, src, count, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyPeerAsync)(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyFromArrayAsync)(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2DAsync)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2DToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpy2DFromArrayAsync)(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyToSymbolAsync)(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemcpyFromSymbolAsync)(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemset)(void *devPtr, int value, size_t count) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemset(devPtr, value, count);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemset2D)(void *devPtr, size_t pitch, int value, size_t width, size_t height) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemset2D(devPtr, pitch, value, width, height);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemset3D)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemset3D(pitchedDevPtr, value, extent);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMemsetAsync)(void *devPtr, int value, size_t count, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemsetAsync(devPtr, value, count, stream);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMemset2DAsync)(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaMemset3DAsync)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream ) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetSymbolAddress)(void **devPtr, const void *symbol) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetSymbolAddress(devPtr, symbol);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetSymbolSize)(size_t *size, const void *symbol) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetSymbolSize(size, symbol);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemPrefetchAsync)(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemAdvise)(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemAdvise(devPtr, count, advice, device);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemRangeGetAttribute)(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaMemRangeGetAttributes)(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count) = NULL;

__host__ cudaError_t CUDARTAPI cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaPointerGetAttributes)(struct cudaPointerAttributes *attributes, const void *ptr) = NULL;

__host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaPointerGetAttributes(attributes, ptr);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceCanAccessPeer)(int *canAccessPeer, int device, int peerDevice) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceEnablePeerAccess)(int peerDevice, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceEnablePeerAccess(peerDevice, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDeviceDisablePeerAccess)(int peerDevice) = NULL;

__host__ cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDeviceDisablePeerAccess(peerDevice);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsUnregisterResource)(cudaGraphicsResource_t resource) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsUnregisterResource(resource);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource, unsigned int flags) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsResourceSetMapFlags(resource, flags);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsMapResources)(int count, cudaGraphicsResource_t *resources, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsMapResources(count, resources, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsUnmapResources)(int count, cudaGraphicsResource_t *resources, cudaStream_t stream ) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsUnmapResources(count, resources, stream);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsResourceGetMappedPointer)(void **devPtr, size_t *size, cudaGraphicsResource_t resource) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsSubResourceGetMappedArray)(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGraphicsResourceGetMappedMipmappedArray)(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) = NULL;

__host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetChannelDesc)(struct cudaChannelFormatDesc *desc, cudaArray_const_t array) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetChannelDesc(desc, array);
  return ret;

}

static __host__ struct cudaChannelFormatDesc CUDARTAPI (*orig_cudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind f) = NULL;

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  struct cudaChannelFormatDesc ret;
  // Write your own code here
  ret = orig_cudaCreateChannelDesc(x, y, z, w, f);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaBindTexture)(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size ) = NULL;

__host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size ) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaBindTexture(offset, texref, devPtr, desc, size);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaBindTexture2D)(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch) = NULL;

__host__ cudaError_t CUDARTAPI cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaBindTextureToArray)(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) = NULL;

__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaBindTextureToArray(texref, array, desc);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaBindTextureToMipmappedArray)(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc) = NULL;

__host__ cudaError_t CUDARTAPI cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaBindTextureToMipmappedArray(texref, mipmappedArray, desc);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaUnbindTexture)(const struct textureReference *texref) = NULL;

__host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaUnbindTexture(texref);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetTextureAlignmentOffset)(size_t *offset, const struct textureReference *texref) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetTextureAlignmentOffset(offset, texref);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetTextureReference)(const struct textureReference **texref, const void *symbol) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const void *symbol) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetTextureReference(texref, symbol);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaBindSurfaceToArray)(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) = NULL;

__host__ cudaError_t CUDARTAPI cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaBindSurfaceToArray(surfref, array, desc);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetSurfaceReference)(const struct surfaceReference **surfref, const void *symbol) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetSurfaceReference(surfref, symbol);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaCreateTextureObject)(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc) = NULL;

__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDestroyTextureObject)(cudaTextureObject_t texObject) = NULL;

__host__ cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDestroyTextureObject(texObject);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetTextureObjectResourceDesc)(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetTextureObjectResourceDesc(pResDesc, texObject);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetTextureObjectTextureDesc)(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetTextureObjectResourceViewDesc)(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaCreateSurfaceObject)(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc) = NULL;

__host__ cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaCreateSurfaceObject(pSurfObject, pResDesc);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDestroySurfaceObject)(cudaSurfaceObject_t surfObject) = NULL;

__host__ cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDestroySurfaceObject(surfObject);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetSurfaceObjectResourceDesc)(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaDriverGetVersion)(int *driverVersion) = NULL;

__host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaDriverGetVersion(driverVersion);
  return ret;

}

static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*orig_cudaRuntimeGetVersion)(int *runtimeVersion) = NULL;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaRuntimeGetVersion(runtimeVersion);
  return ret;

}

static __host__ cudaError_t CUDARTAPI (*orig_cudaGetExportTable)(const void **ppExportTable, const cudaUUID_t *pExportTableId) = NULL;

__host__ cudaError_t CUDARTAPI cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId) {
  cudaError_t ret;
  // Write your own code here
  ret = orig_cudaGetExportTable(ppExportTable, pExportTableId);
  return ret;

}
__attribute__((constructor)) static void init() {
  char *dl_error;
  // clear dl error
  dlerror();
  if (orig_cudaDeviceReset == NULL) {
    orig_cudaDeviceReset = dlsym(RTLD_NEXT, "cudaDeviceReset");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceSynchronize == NULL) {
    orig_cudaDeviceSynchronize = dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceSetLimit == NULL) {
    orig_cudaDeviceSetLimit = dlsym(RTLD_NEXT, "cudaDeviceSetLimit");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetLimit == NULL) {
    orig_cudaDeviceGetLimit = dlsym(RTLD_NEXT, "cudaDeviceGetLimit");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetCacheConfig == NULL) {
    orig_cudaDeviceGetCacheConfig = dlsym(RTLD_NEXT, "cudaDeviceGetCacheConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetStreamPriorityRange == NULL) {
    orig_cudaDeviceGetStreamPriorityRange = dlsym(RTLD_NEXT, "cudaDeviceGetStreamPriorityRange");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceSetCacheConfig == NULL) {
    orig_cudaDeviceSetCacheConfig = dlsym(RTLD_NEXT, "cudaDeviceSetCacheConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetSharedMemConfig == NULL) {
    orig_cudaDeviceGetSharedMemConfig = dlsym(RTLD_NEXT, "cudaDeviceGetSharedMemConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceSetSharedMemConfig == NULL) {
    orig_cudaDeviceSetSharedMemConfig = dlsym(RTLD_NEXT, "cudaDeviceSetSharedMemConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetByPCIBusId == NULL) {
    orig_cudaDeviceGetByPCIBusId = dlsym(RTLD_NEXT, "cudaDeviceGetByPCIBusId");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetPCIBusId == NULL) {
    orig_cudaDeviceGetPCIBusId = dlsym(RTLD_NEXT, "cudaDeviceGetPCIBusId");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaIpcGetEventHandle == NULL) {
    orig_cudaIpcGetEventHandle = dlsym(RTLD_NEXT, "cudaIpcGetEventHandle");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaIpcOpenEventHandle == NULL) {
    orig_cudaIpcOpenEventHandle = dlsym(RTLD_NEXT, "cudaIpcOpenEventHandle");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaIpcGetMemHandle == NULL) {
    orig_cudaIpcGetMemHandle = dlsym(RTLD_NEXT, "cudaIpcGetMemHandle");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaIpcOpenMemHandle == NULL) {
    orig_cudaIpcOpenMemHandle = dlsym(RTLD_NEXT, "cudaIpcOpenMemHandle");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaIpcCloseMemHandle == NULL) {
    orig_cudaIpcCloseMemHandle = dlsym(RTLD_NEXT, "cudaIpcCloseMemHandle");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaThreadExit == NULL) {
    orig_cudaThreadExit = dlsym(RTLD_NEXT, "cudaThreadExit");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaThreadSynchronize == NULL) {
    orig_cudaThreadSynchronize = dlsym(RTLD_NEXT, "cudaThreadSynchronize");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaThreadSetLimit == NULL) {
    orig_cudaThreadSetLimit = dlsym(RTLD_NEXT, "cudaThreadSetLimit");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaThreadGetLimit == NULL) {
    orig_cudaThreadGetLimit = dlsym(RTLD_NEXT, "cudaThreadGetLimit");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaThreadGetCacheConfig == NULL) {
    orig_cudaThreadGetCacheConfig = dlsym(RTLD_NEXT, "cudaThreadGetCacheConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaThreadSetCacheConfig == NULL) {
    orig_cudaThreadSetCacheConfig = dlsym(RTLD_NEXT, "cudaThreadSetCacheConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetLastError == NULL) {
    orig_cudaGetLastError = dlsym(RTLD_NEXT, "cudaGetLastError");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaPeekAtLastError == NULL) {
    orig_cudaPeekAtLastError = dlsym(RTLD_NEXT, "cudaPeekAtLastError");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetErrorName == NULL) {
    orig_cudaGetErrorName = dlsym(RTLD_NEXT, "cudaGetErrorName");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetErrorString == NULL) {
    orig_cudaGetErrorString = dlsym(RTLD_NEXT, "cudaGetErrorString");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetDeviceCount == NULL) {
    orig_cudaGetDeviceCount = dlsym(RTLD_NEXT, "cudaGetDeviceCount");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetDeviceProperties == NULL) {
    orig_cudaGetDeviceProperties = dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetAttribute == NULL) {
    orig_cudaDeviceGetAttribute = dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceGetP2PAttribute == NULL) {
    orig_cudaDeviceGetP2PAttribute = dlsym(RTLD_NEXT, "cudaDeviceGetP2PAttribute");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaChooseDevice == NULL) {
    orig_cudaChooseDevice = dlsym(RTLD_NEXT, "cudaChooseDevice");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaSetDevice == NULL) {
    orig_cudaSetDevice = dlsym(RTLD_NEXT, "cudaSetDevice");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetDevice == NULL) {
    orig_cudaGetDevice = dlsym(RTLD_NEXT, "cudaGetDevice");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaSetValidDevices == NULL) {
    orig_cudaSetValidDevices = dlsym(RTLD_NEXT, "cudaSetValidDevices");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaSetDeviceFlags == NULL) {
    orig_cudaSetDeviceFlags = dlsym(RTLD_NEXT, "cudaSetDeviceFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetDeviceFlags == NULL) {
    orig_cudaGetDeviceFlags = dlsym(RTLD_NEXT, "cudaGetDeviceFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamCreate == NULL) {
    orig_cudaStreamCreate = dlsym(RTLD_NEXT, "cudaStreamCreate");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamCreateWithFlags == NULL) {
    orig_cudaStreamCreateWithFlags = dlsym(RTLD_NEXT, "cudaStreamCreateWithFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamCreateWithPriority == NULL) {
    orig_cudaStreamCreateWithPriority = dlsym(RTLD_NEXT, "cudaStreamCreateWithPriority");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamGetPriority == NULL) {
    orig_cudaStreamGetPriority = dlsym(RTLD_NEXT, "cudaStreamGetPriority");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamGetFlags == NULL) {
    orig_cudaStreamGetFlags = dlsym(RTLD_NEXT, "cudaStreamGetFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamDestroy == NULL) {
    orig_cudaStreamDestroy = dlsym(RTLD_NEXT, "cudaStreamDestroy");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamWaitEvent == NULL) {
    orig_cudaStreamWaitEvent = dlsym(RTLD_NEXT, "cudaStreamWaitEvent");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamAddCallback == NULL) {
    orig_cudaStreamAddCallback = dlsym(RTLD_NEXT, "cudaStreamAddCallback");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamSynchronize == NULL) {
    orig_cudaStreamSynchronize = dlsym(RTLD_NEXT, "cudaStreamSynchronize");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamQuery == NULL) {
    orig_cudaStreamQuery = dlsym(RTLD_NEXT, "cudaStreamQuery");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaStreamAttachMemAsync == NULL) {
    orig_cudaStreamAttachMemAsync = dlsym(RTLD_NEXT, "cudaStreamAttachMemAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventCreate == NULL) {
    orig_cudaEventCreate = dlsym(RTLD_NEXT, "cudaEventCreate");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventCreateWithFlags == NULL) {
    orig_cudaEventCreateWithFlags = dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventRecord == NULL) {
    orig_cudaEventRecord = dlsym(RTLD_NEXT, "cudaEventRecord");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventQuery == NULL) {
    orig_cudaEventQuery = dlsym(RTLD_NEXT, "cudaEventQuery");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventSynchronize == NULL) {
    orig_cudaEventSynchronize = dlsym(RTLD_NEXT, "cudaEventSynchronize");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventDestroy == NULL) {
    orig_cudaEventDestroy = dlsym(RTLD_NEXT, "cudaEventDestroy");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaEventElapsedTime == NULL) {
    orig_cudaEventElapsedTime = dlsym(RTLD_NEXT, "cudaEventElapsedTime");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaLaunchKernel == NULL) {
    orig_cudaLaunchKernel = dlsym(RTLD_NEXT, "cudaLaunchKernel");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaLaunchCooperativeKernel == NULL) {
    orig_cudaLaunchCooperativeKernel = dlsym(RTLD_NEXT, "cudaLaunchCooperativeKernel");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaLaunchCooperativeKernelMultiDevice == NULL) {
    orig_cudaLaunchCooperativeKernelMultiDevice = dlsym(RTLD_NEXT, "cudaLaunchCooperativeKernelMultiDevice");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFuncSetCacheConfig == NULL) {
    orig_cudaFuncSetCacheConfig = dlsym(RTLD_NEXT, "cudaFuncSetCacheConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFuncSetSharedMemConfig == NULL) {
    orig_cudaFuncSetSharedMemConfig = dlsym(RTLD_NEXT, "cudaFuncSetSharedMemConfig");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFuncGetAttributes == NULL) {
    orig_cudaFuncGetAttributes = dlsym(RTLD_NEXT, "cudaFuncGetAttributes");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFuncSetAttribute == NULL) {
    orig_cudaFuncSetAttribute = dlsym(RTLD_NEXT, "cudaFuncSetAttribute");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaSetDoubleForDevice == NULL) {
    orig_cudaSetDoubleForDevice = dlsym(RTLD_NEXT, "cudaSetDoubleForDevice");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaSetDoubleForHost == NULL) {
    orig_cudaSetDoubleForHost = dlsym(RTLD_NEXT, "cudaSetDoubleForHost");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaOccupancyMaxActiveBlocksPerMultiprocessor == NULL) {
    orig_cudaOccupancyMaxActiveBlocksPerMultiprocessor = dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags == NULL) {
    orig_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaConfigureCall == NULL) {
    orig_cudaConfigureCall = dlsym(RTLD_NEXT, "cudaConfigureCall");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaSetupArgument == NULL) {
    orig_cudaSetupArgument = dlsym(RTLD_NEXT, "cudaSetupArgument");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaLaunch == NULL) {
    orig_cudaLaunch = dlsym(RTLD_NEXT, "cudaLaunch");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMallocManaged == NULL) {
    orig_cudaMallocManaged = dlsym(RTLD_NEXT, "cudaMallocManaged");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMalloc == NULL) {
    orig_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMallocHost == NULL) {
    orig_cudaMallocHost = dlsym(RTLD_NEXT, "cudaMallocHost");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMallocPitch == NULL) {
    orig_cudaMallocPitch = dlsym(RTLD_NEXT, "cudaMallocPitch");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMallocArray == NULL) {
    orig_cudaMallocArray = dlsym(RTLD_NEXT, "cudaMallocArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFree == NULL) {
    orig_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFreeHost == NULL) {
    orig_cudaFreeHost = dlsym(RTLD_NEXT, "cudaFreeHost");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFreeArray == NULL) {
    orig_cudaFreeArray = dlsym(RTLD_NEXT, "cudaFreeArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaFreeMipmappedArray == NULL) {
    orig_cudaFreeMipmappedArray = dlsym(RTLD_NEXT, "cudaFreeMipmappedArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaHostAlloc == NULL) {
    orig_cudaHostAlloc = dlsym(RTLD_NEXT, "cudaHostAlloc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaHostRegister == NULL) {
    orig_cudaHostRegister = dlsym(RTLD_NEXT, "cudaHostRegister");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaHostUnregister == NULL) {
    orig_cudaHostUnregister = dlsym(RTLD_NEXT, "cudaHostUnregister");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaHostGetDevicePointer == NULL) {
    orig_cudaHostGetDevicePointer = dlsym(RTLD_NEXT, "cudaHostGetDevicePointer");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaHostGetFlags == NULL) {
    orig_cudaHostGetFlags = dlsym(RTLD_NEXT, "cudaHostGetFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMalloc3D == NULL) {
    orig_cudaMalloc3D = dlsym(RTLD_NEXT, "cudaMalloc3D");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMalloc3DArray == NULL) {
    orig_cudaMalloc3DArray = dlsym(RTLD_NEXT, "cudaMalloc3DArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMallocMipmappedArray == NULL) {
    orig_cudaMallocMipmappedArray = dlsym(RTLD_NEXT, "cudaMallocMipmappedArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetMipmappedArrayLevel == NULL) {
    orig_cudaGetMipmappedArrayLevel = dlsym(RTLD_NEXT, "cudaGetMipmappedArrayLevel");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy3D == NULL) {
    orig_cudaMemcpy3D = dlsym(RTLD_NEXT, "cudaMemcpy3D");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy3DPeer == NULL) {
    orig_cudaMemcpy3DPeer = dlsym(RTLD_NEXT, "cudaMemcpy3DPeer");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy3DAsync == NULL) {
    orig_cudaMemcpy3DAsync = dlsym(RTLD_NEXT, "cudaMemcpy3DAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy3DPeerAsync == NULL) {
    orig_cudaMemcpy3DPeerAsync = dlsym(RTLD_NEXT, "cudaMemcpy3DPeerAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemGetInfo == NULL) {
    orig_cudaMemGetInfo = dlsym(RTLD_NEXT, "cudaMemGetInfo");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaArrayGetInfo == NULL) {
    orig_cudaArrayGetInfo = dlsym(RTLD_NEXT, "cudaArrayGetInfo");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy == NULL) {
    orig_cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyPeer == NULL) {
    orig_cudaMemcpyPeer = dlsym(RTLD_NEXT, "cudaMemcpyPeer");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyToArray == NULL) {
    orig_cudaMemcpyToArray = dlsym(RTLD_NEXT, "cudaMemcpyToArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyFromArray == NULL) {
    orig_cudaMemcpyFromArray = dlsym(RTLD_NEXT, "cudaMemcpyFromArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyArrayToArray == NULL) {
    orig_cudaMemcpyArrayToArray = dlsym(RTLD_NEXT, "cudaMemcpyArrayToArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2D == NULL) {
    orig_cudaMemcpy2D = dlsym(RTLD_NEXT, "cudaMemcpy2D");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2DToArray == NULL) {
    orig_cudaMemcpy2DToArray = dlsym(RTLD_NEXT, "cudaMemcpy2DToArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2DFromArray == NULL) {
    orig_cudaMemcpy2DFromArray = dlsym(RTLD_NEXT, "cudaMemcpy2DFromArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2DArrayToArray == NULL) {
    orig_cudaMemcpy2DArrayToArray = dlsym(RTLD_NEXT, "cudaMemcpy2DArrayToArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyToSymbol == NULL) {
    orig_cudaMemcpyToSymbol = dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyFromSymbol == NULL) {
    orig_cudaMemcpyFromSymbol = dlsym(RTLD_NEXT, "cudaMemcpyFromSymbol");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyAsync == NULL) {
    orig_cudaMemcpyAsync = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyPeerAsync == NULL) {
    orig_cudaMemcpyPeerAsync = dlsym(RTLD_NEXT, "cudaMemcpyPeerAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyToArrayAsync == NULL) {
    orig_cudaMemcpyToArrayAsync = dlsym(RTLD_NEXT, "cudaMemcpyToArrayAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyFromArrayAsync == NULL) {
    orig_cudaMemcpyFromArrayAsync = dlsym(RTLD_NEXT, "cudaMemcpyFromArrayAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2DAsync == NULL) {
    orig_cudaMemcpy2DAsync = dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2DToArrayAsync == NULL) {
    orig_cudaMemcpy2DToArrayAsync = dlsym(RTLD_NEXT, "cudaMemcpy2DToArrayAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpy2DFromArrayAsync == NULL) {
    orig_cudaMemcpy2DFromArrayAsync = dlsym(RTLD_NEXT, "cudaMemcpy2DFromArrayAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyToSymbolAsync == NULL) {
    orig_cudaMemcpyToSymbolAsync = dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemcpyFromSymbolAsync == NULL) {
    orig_cudaMemcpyFromSymbolAsync = dlsym(RTLD_NEXT, "cudaMemcpyFromSymbolAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemset == NULL) {
    orig_cudaMemset = dlsym(RTLD_NEXT, "cudaMemset");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemset2D == NULL) {
    orig_cudaMemset2D = dlsym(RTLD_NEXT, "cudaMemset2D");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemset3D == NULL) {
    orig_cudaMemset3D = dlsym(RTLD_NEXT, "cudaMemset3D");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemsetAsync == NULL) {
    orig_cudaMemsetAsync = dlsym(RTLD_NEXT, "cudaMemsetAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemset2DAsync == NULL) {
    orig_cudaMemset2DAsync = dlsym(RTLD_NEXT, "cudaMemset2DAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemset3DAsync == NULL) {
    orig_cudaMemset3DAsync = dlsym(RTLD_NEXT, "cudaMemset3DAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetSymbolAddress == NULL) {
    orig_cudaGetSymbolAddress = dlsym(RTLD_NEXT, "cudaGetSymbolAddress");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetSymbolSize == NULL) {
    orig_cudaGetSymbolSize = dlsym(RTLD_NEXT, "cudaGetSymbolSize");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemPrefetchAsync == NULL) {
    orig_cudaMemPrefetchAsync = dlsym(RTLD_NEXT, "cudaMemPrefetchAsync");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemAdvise == NULL) {
    orig_cudaMemAdvise = dlsym(RTLD_NEXT, "cudaMemAdvise");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemRangeGetAttribute == NULL) {
    orig_cudaMemRangeGetAttribute = dlsym(RTLD_NEXT, "cudaMemRangeGetAttribute");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaMemRangeGetAttributes == NULL) {
    orig_cudaMemRangeGetAttributes = dlsym(RTLD_NEXT, "cudaMemRangeGetAttributes");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaPointerGetAttributes == NULL) {
    orig_cudaPointerGetAttributes = dlsym(RTLD_NEXT, "cudaPointerGetAttributes");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceCanAccessPeer == NULL) {
    orig_cudaDeviceCanAccessPeer = dlsym(RTLD_NEXT, "cudaDeviceCanAccessPeer");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceEnablePeerAccess == NULL) {
    orig_cudaDeviceEnablePeerAccess = dlsym(RTLD_NEXT, "cudaDeviceEnablePeerAccess");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDeviceDisablePeerAccess == NULL) {
    orig_cudaDeviceDisablePeerAccess = dlsym(RTLD_NEXT, "cudaDeviceDisablePeerAccess");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsUnregisterResource == NULL) {
    orig_cudaGraphicsUnregisterResource = dlsym(RTLD_NEXT, "cudaGraphicsUnregisterResource");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsResourceSetMapFlags == NULL) {
    orig_cudaGraphicsResourceSetMapFlags = dlsym(RTLD_NEXT, "cudaGraphicsResourceSetMapFlags");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsMapResources == NULL) {
    orig_cudaGraphicsMapResources = dlsym(RTLD_NEXT, "cudaGraphicsMapResources");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsUnmapResources == NULL) {
    orig_cudaGraphicsUnmapResources = dlsym(RTLD_NEXT, "cudaGraphicsUnmapResources");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsResourceGetMappedPointer == NULL) {
    orig_cudaGraphicsResourceGetMappedPointer = dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedPointer");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsSubResourceGetMappedArray == NULL) {
    orig_cudaGraphicsSubResourceGetMappedArray = dlsym(RTLD_NEXT, "cudaGraphicsSubResourceGetMappedArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGraphicsResourceGetMappedMipmappedArray == NULL) {
    orig_cudaGraphicsResourceGetMappedMipmappedArray = dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedMipmappedArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetChannelDesc == NULL) {
    orig_cudaGetChannelDesc = dlsym(RTLD_NEXT, "cudaGetChannelDesc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaCreateChannelDesc == NULL) {
    orig_cudaCreateChannelDesc = dlsym(RTLD_NEXT, "cudaCreateChannelDesc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaBindTexture == NULL) {
    orig_cudaBindTexture = dlsym(RTLD_NEXT, "cudaBindTexture");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaBindTexture2D == NULL) {
    orig_cudaBindTexture2D = dlsym(RTLD_NEXT, "cudaBindTexture2D");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaBindTextureToArray == NULL) {
    orig_cudaBindTextureToArray = dlsym(RTLD_NEXT, "cudaBindTextureToArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaBindTextureToMipmappedArray == NULL) {
    orig_cudaBindTextureToMipmappedArray = dlsym(RTLD_NEXT, "cudaBindTextureToMipmappedArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaUnbindTexture == NULL) {
    orig_cudaUnbindTexture = dlsym(RTLD_NEXT, "cudaUnbindTexture");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetTextureAlignmentOffset == NULL) {
    orig_cudaGetTextureAlignmentOffset = dlsym(RTLD_NEXT, "cudaGetTextureAlignmentOffset");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetTextureReference == NULL) {
    orig_cudaGetTextureReference = dlsym(RTLD_NEXT, "cudaGetTextureReference");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaBindSurfaceToArray == NULL) {
    orig_cudaBindSurfaceToArray = dlsym(RTLD_NEXT, "cudaBindSurfaceToArray");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetSurfaceReference == NULL) {
    orig_cudaGetSurfaceReference = dlsym(RTLD_NEXT, "cudaGetSurfaceReference");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaCreateTextureObject == NULL) {
    orig_cudaCreateTextureObject = dlsym(RTLD_NEXT, "cudaCreateTextureObject");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDestroyTextureObject == NULL) {
    orig_cudaDestroyTextureObject = dlsym(RTLD_NEXT, "cudaDestroyTextureObject");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetTextureObjectResourceDesc == NULL) {
    orig_cudaGetTextureObjectResourceDesc = dlsym(RTLD_NEXT, "cudaGetTextureObjectResourceDesc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetTextureObjectTextureDesc == NULL) {
    orig_cudaGetTextureObjectTextureDesc = dlsym(RTLD_NEXT, "cudaGetTextureObjectTextureDesc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetTextureObjectResourceViewDesc == NULL) {
    orig_cudaGetTextureObjectResourceViewDesc = dlsym(RTLD_NEXT, "cudaGetTextureObjectResourceViewDesc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaCreateSurfaceObject == NULL) {
    orig_cudaCreateSurfaceObject = dlsym(RTLD_NEXT, "cudaCreateSurfaceObject");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDestroySurfaceObject == NULL) {
    orig_cudaDestroySurfaceObject = dlsym(RTLD_NEXT, "cudaDestroySurfaceObject");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetSurfaceObjectResourceDesc == NULL) {
    orig_cudaGetSurfaceObjectResourceDesc = dlsym(RTLD_NEXT, "cudaGetSurfaceObjectResourceDesc");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaDriverGetVersion == NULL) {
    orig_cudaDriverGetVersion = dlsym(RTLD_NEXT, "cudaDriverGetVersion");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaRuntimeGetVersion == NULL) {
    orig_cudaRuntimeGetVersion = dlsym(RTLD_NEXT, "cudaRuntimeGetVersion");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }


  // clear dl error
  dlerror();
  if (orig_cudaGetExportTable == NULL) {
    orig_cudaGetExportTable = dlsym(RTLD_NEXT, "cudaGetExportTable");
  }
  if ((dl_error = dlerror()) != NULL)
  {
    fprintf(stderr, ">>>>>>> %s\n", dl_error);
  }

}
