/**
 * The interface between the runtime (cudart.cpp) and the CUDA driver API (cuda_driver.cpp), which share one library and
 * one set of objects: a driver API stream or event is the runtime's, the primary context of a device is the runtime's
 * device, and a CUfunction is a kernel handle the runtime launches like a cudaKernel_t.
 *
 * The two live in separate files because cuda.h and the LibreCUDA driver's header define the same names.
 */
#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/**
 * A driver API context. Every context of a device works on the runtime's state of the device (its primary context):
 * contexts created with cuCtxCreate are separate handles of the same state.
 */
struct CUctx_st {
    int device;
    bool primary;
    unsigned int flags;
    // primary contexts: cuDevicePrimaryCtxRetain calls not yet released
    std::atomic<int> retains{0};
};

namespace librecuda_bridge {

// the number of devices (0 if the runtime failed to initialize)
int deviceCount();

// the primary context of the device; nullptr for invalid ordinals
CUctx_st *primaryContext(int device);

// whether the device's primary context has been created
bool primaryContextActive(int device);

// creates the device's primary context (initializes the runtime's state of the device)
cudaError_t activatePrimaryContext(int device);

// the context stack of the calling thread. The runtime binds the primary context of its current device to threads
// that have none; making a context current makes its device the runtime's current device.
CUctx_st *currentContext();
void setCurrentContext(CUctx_st *ctx);
void pushContext(CUctx_st *ctx);
CUctx_st *popContext();
// removes a destroyed context from the calling thread's stack
void forgetContext(CUctx_st *ctx);

// the runtime's last error (cudaGetLastError) of the calling thread; driver API calls leave it unchanged
cudaError_t lastError();
void setLastError(cudaError_t error);

// modules: an image (cubin, fat binary or PTX) loaded on the current device. Functions are cudaKernel_t handles.
cudaError_t moduleLoad(const void *image, void **pModule);
cudaError_t moduleUnload(void *module);
cudaError_t moduleGetFunction(void *module, const char *name, void **pFunction);
cudaError_t moduleGetGlobal(void *module, const char *name, void **pDevicePtr, size_t *pSize);
cudaError_t moduleFunctions(void *module, std::vector<void *> &functions);
bool isFunction(const void *function);
void *functionModule(const void *function);
const char *functionName(const void *function);
// offset and size of each kernel parameter (on the current device)
cudaError_t functionParams(const void *function, std::vector<std::pair<size_t, size_t>> &params);

// the cubin of an image (cubin, fat binary or PTX) for the current device
cudaError_t compileImage(const void *image, std::vector<uint8_t> &cubin);

// cuDevicePrimaryCtxReset: like cudaDeviceReset on the device
cudaError_t resetPrimaryContext(int device);

// fills count elements of 1, 2 or 4 bytes
cudaError_t memsetElements(void *dst, unsigned int value, size_t elementSize, size_t count, cudaStream_t stream,
                           bool synchronous);

// the allocation containing ptr
cudaError_t addressRange(const void *ptr, void **pBase, size_t *pSize);

// virtual memory management (cuMem*): allocations of a device as handles of LibreCUDA's driver. Sizes and addresses
// are multiples of vmmGranularity().
size_t vmmGranularity();
cudaError_t vmmCreate(int device, size_t size, uint64_t *pHandle);
cudaError_t vmmRelease(uint64_t handle);
cudaError_t vmmReserve(size_t size, size_t alignment, uint64_t *pAddress);
cudaError_t vmmMap(uint64_t address, size_t size, size_t offset, uint64_t handle);
cudaError_t vmmUnmap(uint64_t address, size_t size);
cudaError_t vmmSetAccess(uint64_t address, size_t size, int device);
cudaError_t vmmRetain(const void *address, uint64_t *pHandle);
cudaError_t vmmProperties(uint64_t handle, int *pDevice, size_t *pSize);
cudaError_t vmmExport(uint64_t handle, int *pFd);
// (creates the contexts of all devices: the allocation's GPU needs one)
cudaError_t vmmImport(int fd, uint64_t *pHandle);

// cuGetProcAddress (in the driver API file), for cudaGetDriverEntryPoint
int driverEntryPoint(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags, int *symbolStatus);

} // namespace librecuda_bridge
