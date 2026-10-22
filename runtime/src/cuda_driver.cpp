/**
 * LibreCUDA's CUDA driver API (libcuda), implemented on the runtime of this library (cudart.cpp) so that both work on
 * the same objects:
 *  - a device's primary context is the runtime's device; making a context current makes its device the runtime's
 *    current device, and the runtime makes the primary context of its device current (as CUDA does)
 *  - CUstream and CUevent are the runtime's cudaStream_t and cudaEvent_t, CUgraph / CUgraphExec its graphs, CUmemoryPool
 *    its memory pools
 *  - modules are loaded like the runtime's fat binaries; their functions are kernel handles the runtime launches, so
 *    driver launches take part in stream capture, graphs and the runtime's stream semantics
 *
 * Contexts created with cuCtxCreate are further handles of the device's primary context. Functions that need features
 * LibreCUDA lacks (virtual memory management, IPC, tensor maps, green contexts, ...) fail with CUDA_ERROR_NOT_SUPPORTED;
 * cuGetExportTable (whose tables are not public) finds nothing.
 *
 * Only the public cuda.h / cuda_runtime_api.h are used.
 */
#include <cuda.h>
#include <cudaProfiler.h>
#include <cuda_runtime_api.h>

#include "driver_bridge.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace bridge = librecuda_bridge;

// deprecated functions are part of the API too
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

extern "C" CUresult CUDAAPI cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);

namespace {

CUresult toResult(cudaError_t error) {
    // the runtime's error codes are the driver's, except for the runtime-only codes below 100
    const int value = static_cast<int>(error);
    if (value > 8 && value < 100) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return static_cast<CUresult>(value);
}

/**
 * Runs runtime calls for a driver API function: the runtime's last error (cudaGetLastError) stays as it was
 */
template<typename F>
CUresult run(F &&calls) {
    const cudaError_t saved = bridge::lastError();
    const cudaError_t error = calls();
    bridge::setLastError(saved);
    return toResult(error);
}

#define DRV_CHECK(expr) do { CUresult drv_check_result = (expr); if (drv_check_result != CUDA_SUCCESS) { return drv_check_result; } } while (0)
#define RT_TRY(expr) do { cudaError_t rt_try_error = (expr); if (rt_try_error != cudaSuccess) { return rt_try_error; } } while (0)

// the calling thread's current context (its device is the runtime's current device)
CUresult currentContext(CUctx_st **pCtx = nullptr) {
    CUctx_st *ctx = bridge::currentContext();
    if (ctx == nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (pCtx != nullptr) {
        *pCtx = ctx;
    }
    return CUDA_SUCCESS;
}

bool validDevice(CUdevice device) {
    return device >= 0 && device < bridge::deviceCount();
}

// contexts created with cuCtxCreate (primary contexts belong to the runtime)
std::mutex g_contextMutex;
std::vector<CUctx_st *> g_createdContexts{};

bool validContext(CUcontext ctx) {
    if (ctx == nullptr) {
        return false;
    }
    if (ctx->primary) {
        return bridge::primaryContext(ctx->device) == ctx;
    }
    std::lock_guard context_lock(g_contextMutex);
    return std::find(g_createdContexts.begin(), g_createdContexts.end(), ctx) != g_createdContexts.end();
}

void *ptr(CUdeviceptr address) {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(address));
}

const void *kernel(CUfunction function) {
    return reinterpret_cast<const void *>(function);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// initialization, versions and errors

CUresult CUDAAPI cuInit(unsigned int flags) {
    if (flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return bridge::deviceCount() > 0 ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
    return run([&] { return cudaDriverGetVersion(driverVersion); });
}

namespace {
struct ErrorName {
    int value;
    const char *name;
};

const ErrorName ERROR_NAMES[] = {
#include "driver_error_names.inc"
};
} // namespace

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
    if (pStr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (const ErrorName &entry: ERROR_NAMES) {
        if (entry.value == error) {
            *pStr = entry.name;
            return CUDA_SUCCESS;
        }
    }
    *pStr = nullptr;
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
    if (pStr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const char *name{};
    if (cuGetErrorName(error, &name) != CUDA_SUCCESS) {
        *pStr = nullptr;
        return CUDA_ERROR_INVALID_VALUE;
    }
    // the runtime's description of the same code
    *pStr = cudaGetErrorString(static_cast<cudaError_t>(error));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable, const CUuuid *) {
    // the driver's private tables are not public
    if (ppExportTable != nullptr) {
        *ppExportTable = nullptr;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuProfilerStart(void) {
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuProfilerStop(void) {
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// devices

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(ordinal)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
    if (count == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *count = bridge::deviceCount();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (name == nullptr || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] {
        cudaDeviceProp prop{};
        RT_TRY(cudaGetDeviceProperties(&prop, dev));
        snprintf(name, len, "%s", prop.name);
        return cudaSuccess;
    });
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (uuid == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] {
        cudaDeviceProp prop{};
        RT_TRY(cudaGetDeviceProperties(&prop, dev));
        static_assert(sizeof(prop.uuid) == sizeof(*uuid));
        memcpy(uuid, &prop.uuid, sizeof(*uuid));
        return cudaSuccess;
    });
}

CUresult CUDAAPI cuDeviceGetLuid(char *, unsigned int *, CUdevice) {
    // LUIDs exist on Windows only
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] {
        cudaDeviceProp prop{};
        RT_TRY(cudaGetDeviceProperties(&prop, dev));
        *bytes = prop.totalGlobalMem;
        return cudaSuccess;
    });
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (pi == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // the runtime's attribute numbering is the driver's
    return run([&] { return cudaDeviceGetAttribute(pi, static_cast<cudaDeviceAttr>(attrib), dev); });
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    if (major == nullptr || minor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    return cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    if (prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] {
        cudaDeviceProp p{};
        RT_TRY(cudaGetDeviceProperties(&p, dev));
        prop->maxThreadsPerBlock = p.maxThreadsPerBlock;
        std::copy(std::begin(p.maxThreadsDim), std::end(p.maxThreadsDim), prop->maxThreadsDim);
        std::copy(std::begin(p.maxGridSize), std::end(p.maxGridSize), prop->maxGridSize);
        prop->sharedMemPerBlock = static_cast<int>(p.sharedMemPerBlock);
        prop->totalConstantMemory = static_cast<int>(p.totalConstMem);
        prop->SIMDWidth = p.warpSize;
        prop->memPitch = static_cast<int>(std::min<size_t>(p.memPitch, INT32_MAX));
        prop->regsPerBlock = p.regsPerBlock;
        int clock = 0;
        RT_TRY(cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, dev));
        prop->clockRate = clock;
        prop->textureAlign = static_cast<int>(p.textureAlignment);
        return cudaSuccess;
    });
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    if (pciBusId == nullptr || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] { return cudaDeviceGetPCIBusId(pciBusId, len, dev); });
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    return run([&] { return cudaDeviceGetByPCIBusId(dev, pciBusId); });
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    if (!validDevice(dev) || !validDevice(peerDev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] { return cudaDeviceCanAccessPeer(canAccessPeer, dev, peerDev); });
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice,
                                         CUdevice dstDevice) {
    if (!validDevice(srcDevice) || !validDevice(dstDevice)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] {
        return cudaDeviceGetP2PAttribute(value, static_cast<cudaDeviceP2PAttr>(attrib), srcDevice, dstDevice);
    });
}

CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] { return cudaDeviceGetDefaultMemPool(pool_out, dev); });
}

CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    return cuDeviceGetDefaultMemPool(pool, dev);
}

CUresult CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return run([&] { return cudaDeviceSetMemPool(dev, pool); });
}

// ---------------------------------------------------------------------------------------------------------------------
// primary contexts

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    DRV_CHECK(run([&] { return bridge::activatePrimaryContext(dev); }));
    CUctx_st *ctx = bridge::primaryContext(dev);
    ctx->retains++;
    *pctx = ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // the runtime keeps using the context: releasing it doesn't destroy it
    CUctx_st *ctx = bridge::primaryContext(dev);
    int retains = ctx->retains.load();
    while (retains > 0 && !ctx->retains.compare_exchange_weak(retains, retains - 1)) {
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // scheduling flags (spin, yield, blocking sync) and CU_CTX_LMEM_RESIZE_TO_MAX / CU_CTX_SYNC_MEMOPS are accepted;
    // the runtime decides how it waits
    bridge::primaryContext(dev)->flags = flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    if (flags == nullptr || active == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *flags = bridge::primaryContext(dev)->flags;
    *active = bridge::primaryContextActive(dev) ? 1 : 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    bridge::primaryContext(dev)->retains = 0;
    return run([&] { return bridge::resetPrimaryContext(dev); });
}

// ---------------------------------------------------------------------------------------------------------------------
// contexts

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validDevice(dev)) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // execution affinity and CIG contexts are not supported
    if (ctxCreateParams != nullptr && (ctxCreateParams->numExecAffinityParams > 0 || ctxCreateParams->cigParams)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    DRV_CHECK(run([&] { return bridge::activatePrimaryContext(dev); }));
    auto *ctx = new CUctx_st{};
    ctx->device = dev;
    ctx->primary = false;
    ctx->flags = flags;
    {
        std::lock_guard context_lock(g_contextMutex);
        g_createdContexts.push_back(ctx);
    }
    bridge::pushContext(ctx);
    *pctx = ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
    if (!validContext(ctx) || ctx->primary) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    {
        std::lock_guard context_lock(g_contextMutex);
        g_createdContexts.erase(std::find(g_createdContexts.begin(), g_createdContexts.end(), ctx));
    }
    bridge::forgetContext(ctx);
    delete ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
    if (!validContext(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    bridge::pushContext(ctx);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
    CUctx_st *ctx = bridge::popContext();
    if (ctx == nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (pctx != nullptr) {
        *pctx = ctx;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    if (ctx != nullptr && !validContext(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    bridge::setCurrentContext(ctx);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = bridge::currentContext();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUctx_st *ctx{};
    DRV_CHECK(currentContext(&ctx));
    *device = ctx->device;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice_v2(CUdevice *device, CUcontext ctx) {
    if (ctx == nullptr) {
        return cuCtxGetDevice(device);
    }
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!validContext(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *device = ctx->device;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
    if (flags == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUctx_st *ctx{};
    DRV_CHECK(currentContext(&ctx));
    *flags = ctx->flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetFlags(unsigned int flags) {
    CUctx_st *ctx{};
    DRV_CHECK(currentContext(&ctx));
    ctx->flags = flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetId(CUcontext ctx, unsigned long long *ctxId) {
    if (ctxId == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (ctx == nullptr) {
        DRV_CHECK(currentContext(&ctx));
    } else if (!validContext(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *ctxId = reinterpret_cast<uintptr_t>(ctx);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    if (version == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (ctx == nullptr) {
        DRV_CHECK(currentContext(&ctx));
    } else if (!validContext(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    // contexts created by this API version
    *version = 3020;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void) {
    DRV_CHECK(currentContext());
    return run([] { return cudaDeviceSynchronize(); });
}

CUresult CUDAAPI cuCtxSynchronize_v2(CUcontext ctx) {
    if (ctx == nullptr) {
        return cuCtxSynchronize();
    }
    if (!validContext(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    DRV_CHECK(cuCtxPushCurrent(ctx));
    const CUresult result = cuCtxSynchronize();
    cuCtxPopCurrent(nullptr);
    return result;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceSetLimit(static_cast<cudaLimit>(limit), value); });
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceGetLimit(pvalue, static_cast<cudaLimit>(limit)); });
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceGetCacheConfig(reinterpret_cast<cudaFuncCache *>(pconfig)); });
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(config)); });
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority); });
}

CUresult CUDAAPI cuCtxResetPersistingL2Cache(void) {
    DRV_CHECK(currentContext());
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) {
    if (!validContext(peerContext)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceEnablePeerAccess(peerContext->device, flags); });
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
    if (!validContext(peerContext)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    DRV_CHECK(currentContext());
    return run([&] { return cudaDeviceDisablePeerAccess(peerContext->device); });
}

// ---------------------------------------------------------------------------------------------------------------------
// memory

CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemGetInfo(free, total); });
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    return run([&] { return cudaMalloc(reinterpret_cast<void **>(dptr), bytesize); });
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
    if (dptr == nullptr || pPitch == nullptr || WidthInBytes == 0 || Height == 0 ||
        (ElementSizeBytes != 4 && ElementSizeBytes != 8 && ElementSizeBytes != 16)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    return run([&] { return cudaMallocPitch(reinterpret_cast<void **>(dptr), pPitch, WidthInBytes, Height); });
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaFree(ptr(dptr)); });
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    DRV_CHECK(currentContext());
    void *base{};
    size_t size{};
    if (bridge::addressRange(ptr(dptr), &base, &size) != cudaSuccess) {
        return CUDA_ERROR_NOT_FOUND;
    }
    if (pbase != nullptr) {
        *pbase = reinterpret_cast<CUdeviceptr>(base);
    }
    if (psize != nullptr) {
        *psize = size;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMallocHost(pp, bytesize); });
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaHostAlloc(pp, bytesize, Flags); });
}

CUresult CUDAAPI cuMemFreeHost(void *p) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaFreeHost(p); });
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaHostGetDevicePointer(reinterpret_cast<void **>(pdptr), p, Flags); });
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaHostGetFlags(pFlags, p); });
}

CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaHostRegister(p, bytesize, Flags); });
}

CUresult CUDAAPI cuMemHostUnregister(void *p) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaHostUnregister(p); });
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    return run([&] { return cudaMallocManaged(reinterpret_cast<void **>(dptr), bytesize, flags); });
}

CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMallocAsync(reinterpret_cast<void **>(dptr), bytesize, hStream); });
}

CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaFreeAsync(ptr(dptr), hStream); });
}

CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    return run([&] { return cudaMemPoolSetAttribute(pool, static_cast<cudaMemPoolAttr>(attr), value); });
}

CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    return run([&] { return cudaMemPoolGetAttribute(pool, static_cast<cudaMemPoolAttr>(attr), value); });
}

CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
    static_assert(sizeof(CUmemAccessDesc) == sizeof(cudaMemAccessDesc));
    return run([&] { return cudaMemPoolSetAccess(pool, reinterpret_cast<const cudaMemAccessDesc *>(map), count); });
}

CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location) {
    return run([&] {
        return cudaMemPoolGetAccess(reinterpret_cast<cudaMemAccessFlags *>(flags), memPool,
                                    reinterpret_cast<cudaMemLocation *>(location));
    });
}

CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    return run([&] { return cudaMemPoolTrimTo(pool, minBytesToKeep); });
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags,
                                    CUstream hStream) {
    DRV_CHECK(currentContext());
    cudaMemLocation loc{};
    loc.type = static_cast<cudaMemLocationType>(location.type);
    loc.id = location.id;
    return run([&] { return cudaMemPrefetchAsync(ptr(devPtr), count, loc, flags, hStream); });
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) {
    DRV_CHECK(currentContext());
    cudaMemLocation loc{};
    loc.type = static_cast<cudaMemLocationType>(location.type);
    loc.id = location.id;
    return run([&] { return cudaMemAdvise(ptr(devPtr), count, static_cast<cudaMemoryAdvise>(advice), loc); });
}

CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaMemRangeGetAttribute(data, dataSize, static_cast<cudaMemRangeAttribute>(attribute), ptr(devPtr),
                                        count);
    });
}

// virtual memory management: not supported (CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED is 0)

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr, size_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr, size_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemGetAllocationGranularity(size_t *, const CUmemAllocationProp *,
                                               CUmemAllocationGranularity_flags) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *, void *) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *, CUmemGenericAllocationHandle) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemExportToShareableHandle(void *, CUmemGenericAllocationHandle, CUmemAllocationHandleType,
                                              unsigned long long) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

// interprocess communication: not supported

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *, CUdeviceptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *, CUipcMemHandle, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *, CUevent) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *, CUipcEventHandle) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------------------------------------------------
// pointer attributes

namespace {

struct PointerInfo {
    bool known{};
    cudaPointerAttributes attributes{};
    void *base{};
    size_t size{};
};

PointerInfo pointerInfo(CUdeviceptr address) {
    PointerInfo info{};
    const cudaError_t saved = bridge::lastError();
    if (cudaPointerGetAttributes(&info.attributes, ptr(address)) == cudaSuccess &&
        info.attributes.type != cudaMemoryTypeUnregistered) {
        info.known = bridge::addressRange(ptr(address), &info.base, &info.size) == cudaSuccess;
    }
    bridge::setLastError(saved);
    return info;
}

// one attribute of a pointer of LibreCUDA's memory (info.known)
CUresult pointerAttribute(void *data, CUpointer_attribute attribute, const PointerInfo &info, CUdeviceptr address) {
    const cudaPointerAttributes &attr = info.attributes;
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_CONTEXT:
            *static_cast<CUcontext *>(data) = bridge::primaryContext(attr.device);
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
            *static_cast<unsigned int *>(data) = attr.type == cudaMemoryTypeHost ? CU_MEMORYTYPE_HOST
                                                                                 : CU_MEMORYTYPE_DEVICE;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
            if (attr.devicePointer == nullptr) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            *static_cast<CUdeviceptr *>(data) = reinterpret_cast<CUdeviceptr>(attr.devicePointer);
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:
            if (attr.hostPointer == nullptr) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            *static_cast<void **>(data) = attr.hostPointer;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
        case CU_POINTER_ATTRIBUTE_MAPPED:
            *static_cast<int *>(data) = 1;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_BUFFER_ID:
            *static_cast<unsigned long long *>(data) = reinterpret_cast<uintptr_t>(info.base);
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:
            *static_cast<int *>(data) = attr.type == cudaMemoryTypeManaged ? 1 : 0;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
            *static_cast<int *>(data) = attr.device;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
        case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
            *static_cast<int *>(data) = 0;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
            *static_cast<CUdeviceptr *>(data) = reinterpret_cast<CUdeviceptr>(info.base);
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
            *static_cast<size_t *>(data) = info.size;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
            *static_cast<unsigned long long *>(data) = CU_MEM_HANDLE_TYPE_NONE;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
            *static_cast<unsigned int *>(data) = CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE;
            return CUDA_SUCCESS;
        case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
            *static_cast<CUmemoryPool *>(data) = nullptr;
            return CUDA_SUCCESS;
        default:
            (void) address;
            return CUDA_ERROR_INVALID_VALUE;
    }
}

} // namespace

CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const PointerInfo info = pointerInfo(ptr);
    if (!info.known) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return pointerAttribute(data, attribute, info, ptr);
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data,
                                        CUdeviceptr ptr) {
    if (attributes == nullptr || data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const PointerInfo info = pointerInfo(ptr);
    for (unsigned int i = 0; i < numAttributes; i++) {
        if (data[i] == nullptr) {
            continue;
        }
        if (!info.known) {
            // memory CUDA doesn't know: memory type 0, no context, ...
            switch (attributes[i]) {
                case CU_POINTER_ATTRIBUTE_CONTEXT:
                case CU_POINTER_ATTRIBUTE_HOST_POINTER:
                case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
                    *static_cast<void **>(data[i]) = nullptr;
                    break;
                case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
                case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
                case CU_POINTER_ATTRIBUTE_BUFFER_ID:
                case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
                    *static_cast<unsigned long long *>(data[i]) = 0;
                    break;
                case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
                    *static_cast<size_t *>(data[i]) = 0;
                    break;
                case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
                    *static_cast<int *>(data[i]) = -2;
                    break;
                default:
                    *static_cast<unsigned int *>(data[i]) = 0;
                    break;
            }
            continue;
        }
        DRV_CHECK(pointerAttribute(data[i], attributes[i], info, ptr));
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuPointerSetAttribute(const void *, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (!pointerInfo(ptr).known) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // memory operations on LibreCUDA's memory are always synchronous with respect to it
    return attribute == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// ---------------------------------------------------------------------------------------------------------------------
// copies and fills

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpy(ptr(dst), ptr(src), ByteCount, cudaMemcpyDefault); });
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpyAsync(ptr(dst), ptr(src), ByteCount, cudaMemcpyDefault, hStream); });
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpy(ptr(dstDevice), srcHost, ByteCount, cudaMemcpyHostToDevice); });
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpy(dstHost, ptr(srcDevice), ByteCount, cudaMemcpyDeviceToHost); });
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpy(ptr(dstDevice), ptr(srcDevice), ByteCount, cudaMemcpyDeviceToDevice); });
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpyAsync(ptr(dstDevice), srcHost, ByteCount, cudaMemcpyHostToDevice, hStream); });
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaMemcpyAsync(dstHost, ptr(srcDevice), ByteCount, cudaMemcpyDeviceToHost, hStream); });
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaMemcpyAsync(ptr(dstDevice), ptr(srcDevice), ByteCount, cudaMemcpyDeviceToDevice, hStream);
    });
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
    if (!validContext(dstContext) || !validContext(srcContext)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return run([&] {
        return cudaMemcpyPeer(ptr(dstDevice), dstContext->device, ptr(srcDevice), srcContext->device, ByteCount);
    });
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                   CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    if (!validContext(dstContext) || !validContext(srcContext)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return run([&] {
        return cudaMemcpyPeerAsync(ptr(dstDevice), dstContext->device, ptr(srcDevice), srcContext->device, ByteCount,
                                   hStream);
    });
}

namespace {

// the address of a copy side of linear memory; nullptr for CUDA arrays (not supported by the driver API here)
const void *copyAddress(CUmemorytype type, const void *host, CUdeviceptr device) {
    switch (type) {
        case CU_MEMORYTYPE_HOST:
            return host;
        case CU_MEMORYTYPE_DEVICE:
        case CU_MEMORYTYPE_UNIFIED:
            return ptr(device);
        default:
            return nullptr;
    }
}

CUresult memcpy2D(const CUDA_MEMCPY2D *p, CUstream stream, bool async) {
    if (p == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    const auto *src = static_cast<const uint8_t *>(copyAddress(p->srcMemoryType, p->srcHost, p->srcDevice));
    auto *dst = static_cast<uint8_t *>(const_cast<void *>(copyAddress(p->dstMemoryType, p->dstHost, p->dstDevice)));
    if (src == nullptr || dst == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    src += p->srcY * p->srcPitch + p->srcXInBytes;
    dst += p->dstY * p->dstPitch + p->dstXInBytes;
    return run([&] {
        if (async) {
            return cudaMemcpy2DAsync(dst, p->dstPitch, src, p->srcPitch, p->WidthInBytes, p->Height,
                                     cudaMemcpyDefault, stream);
        }
        return cudaMemcpy2D(dst, p->dstPitch, src, p->srcPitch, p->WidthInBytes, p->Height, cudaMemcpyDefault);
    });
}

CUresult memcpy3D(const CUDA_MEMCPY3D *p, CUstream stream, bool async) {
    if (p == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    const void *src = copyAddress(p->srcMemoryType, p->srcHost, p->srcDevice);
    const void *dst = copyAddress(p->dstMemoryType, p->dstHost, p->dstDevice);
    if (src == nullptr || dst == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    cudaMemcpy3DParms params{};
    params.srcPtr = cudaPitchedPtr{const_cast<void *>(src), p->srcPitch, p->WidthInBytes,
                                   p->srcHeight != 0 ? p->srcHeight : p->Height};
    params.srcPos = cudaPos{p->srcXInBytes, p->srcY, p->srcZ};
    params.dstPtr = cudaPitchedPtr{const_cast<void *>(dst), p->dstPitch, p->WidthInBytes,
                                   p->dstHeight != 0 ? p->dstHeight : p->Height};
    params.dstPos = cudaPos{p->dstXInBytes, p->dstY, p->dstZ};
    params.extent = cudaExtent{p->WidthInBytes, p->Height, p->Depth};
    params.kind = cudaMemcpyDefault;
    return run([&] { return async ? cudaMemcpy3DAsync(&params, stream) : cudaMemcpy3D(&params); });
}

} // namespace

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
    return memcpy2D(pCopy, nullptr, false);
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
    return memcpy2D(pCopy, nullptr, false);
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    return memcpy2D(pCopy, hStream, true);
}

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
    return memcpy3D(pCopy, nullptr, false);
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    return memcpy3D(pCopy, hStream, true);
}

namespace {

// like cudaMemset(Async), fills of device memory are asynchronous with respect to the host
CUresult fill(CUdeviceptr dst, unsigned int value, size_t elementSize, size_t count, CUstream stream) {
    DRV_CHECK(currentContext());
    return run([&] { return bridge::memsetElements(ptr(dst), value, elementSize, count, stream, false); });
}

CUresult fill2D(CUdeviceptr dst, size_t pitch, unsigned int value, size_t elementSize, size_t width, size_t height,
                CUstream stream) {
    for (size_t row = 0; row < height; row++) {
        DRV_CHECK(fill(dst + row * pitch, value, elementSize, width, stream));
    }
    return CUDA_SUCCESS;
}

} // namespace

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    return fill(dstDevice, uc, 1, N, nullptr);
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    return fill(dstDevice, us, 2, N, nullptr);
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    return fill(dstDevice, ui, 4, N, nullptr);
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    return fill(dstDevice, uc, 1, N, hStream);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
    return fill(dstDevice, us, 2, N, hStream);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    return fill(dstDevice, ui, 4, N, hStream);
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
    return fill2D(dstDevice, dstPitch, uc, 1, Width, Height, nullptr);
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
                               size_t Height) {
    return fill2D(dstDevice, dstPitch, us, 2, Width, Height, nullptr);
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
    return fill2D(dstDevice, dstPitch, ui, 4, Width, Height, nullptr);
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
    return fill2D(dstDevice, dstPitch, uc, 1, Width, Height, hStream);
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
    return fill2D(dstDevice, dstPitch, us, 2, Width, Height, hStream);
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
    return fill2D(dstDevice, dstPitch, ui, 4, Width, Height, hStream);
}

// ---------------------------------------------------------------------------------------------------------------------
// streams

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamCreateWithFlags(phStream, Flags); });
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamCreateWithPriority(phStream, flags, priority); });
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
    return run([&] { return cudaStreamDestroy(hStream); });
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamSynchronize(hStream); });
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamQuery(hStream); });
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamWaitEvent(hStream, hEvent, Flags); });
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamGetFlags(hStream, flags); });
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamGetPriority(hStream, priority); });
}

CUresult CUDAAPI cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamGetId(hStream, streamId); });
}

CUresult CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *device) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamGetDevice(hStream, device); });
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUctx_st *current{};
    DRV_CHECK(currentContext(&current));
    // the default streams belong to the current context; other streams to their device's primary context
    if (hStream == nullptr || hStream == CU_STREAM_LEGACY || hStream == CU_STREAM_PER_THREAD) {
        *pctx = current;
        return CUDA_SUCCESS;
    }
    int device = 0;
    DRV_CHECK(run([&] { return cudaStreamGetDevice(hStream, &device); }));
    *pctx = current->device == device ? current : bridge::primaryContext(device);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetCtx_v2(CUstream hStream, CUcontext *pCtx, CUgreenCtx *pGreenCtx) {
    if (pGreenCtx != nullptr) {
        *pGreenCtx = nullptr;
    }
    return cuStreamGetCtx(hStream, pCtx);
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
    DRV_CHECK(currentContext());
    // the callback types differ only in the status's enumeration (whose values are the same)
    return run([&] {
        return cudaStreamAddCallback(hStream, reinterpret_cast<cudaStreamCallback_t>(callback), userData, flags);
    });
}

CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaLaunchHostFunc(hStream, fn, userData); });
}

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamAttachMemAsync(hStream, ptr(dptr), length, flags); });
}

// stream memory operations: not supported (CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS... are 0)

CUresult CUDAAPI cuStreamWaitValue32(CUstream, CUdeviceptr, cuuint32_t, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamWaitValue64(CUstream, CUdeviceptr, cuuint64_t, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream, CUdeviceptr, cuuint32_t, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuStreamWriteValue64(CUstream, CUdeviceptr, cuuint64_t, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

// stream capture and graphs (the runtime's)

CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamBeginCapture(hStream, static_cast<cudaStreamCaptureMode>(mode)); });
}

CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaStreamEndCapture(hStream, phGraph); });
}

CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaStreamIsCapturing(hStream, reinterpret_cast<cudaStreamCaptureStatus *>(captureStatus));
    });
}

CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
    return run([&] { return cudaThreadExchangeStreamCaptureMode(reinterpret_cast<cudaStreamCaptureMode *>(mode)); });
}

CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaGraphInstantiateWithFlags(phGraphExec, hGraph, flags); });
}

CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaGraphLaunch(hGraphExec, hStream); });
}

CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaGraphUpload(hGraphExec, hStream); });
}

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
    return run([&] { return cudaGraphExecDestroy(hGraphExec); });
}

CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph) {
    return run([&] { return cudaGraphDestroy(hGraph); });
}

CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
    return run([&] { return cudaGraphCreate(phGraph, flags); });
}

// ---------------------------------------------------------------------------------------------------------------------
// events

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaEventCreateWithFlags(phEvent, Flags); });
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaEventRecord(hEvent, hStream); });
}

CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
    DRV_CHECK(currentContext());
    return run([&] { return cudaEventRecordWithFlags(hEvent, hStream, flags); });
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
    return run([&] { return cudaEventQuery(hEvent); });
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
    return run([&] { return cudaEventSynchronize(hEvent); });
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
    return run([&] { return cudaEventDestroy(hEvent); });
}

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    return run([&] { return cudaEventElapsedTime(pMilliseconds, hStart, hEnd); });
}

// ---------------------------------------------------------------------------------------------------------------------
// modules

namespace {

// JIT options with outputs: empty logs, no compile time
void answerJitOptions(unsigned int numOptions, const CUjit_option *options, void **optionValues) {
    if (options == nullptr || optionValues == nullptr) {
        return;
    }
    for (unsigned int i = 0; i < numOptions; i++) {
        switch (options[i]) {
            case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
            case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
                optionValues[i] = nullptr;
                break;
            case CU_JIT_INFO_LOG_BUFFER:
            case CU_JIT_ERROR_LOG_BUFFER:
                if (optionValues[i] != nullptr) {
                    *static_cast<char *>(optionValues[i]) = 0;
                }
                break;
            case CU_JIT_WALL_TIME: {
                float zero = 0;
                memcpy(&optionValues[i], &zero, sizeof(zero));
                break;
            }
            default:
                break;
        }
    }
}

bool readFile(const char *path, std::vector<uint8_t> &data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    // PTX text needs its terminating zero
    data.push_back(0);
    return true;
}

} // namespace

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
    DRV_CHECK(currentContext());
    return run([&] { return bridge::moduleLoad(image, reinterpret_cast<void **>(module)); });
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                                    CUjit_option *options, void **optionValues) {
    answerJitOptions(numOptions, options, optionValues);
    return cuModuleLoadData(module, image);
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    return cuModuleLoadData(module, fatCubin);
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    if (fname == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::vector<uint8_t> data;
    if (!readFile(fname, data)) {
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    return cuModuleLoadData(module, data.data());
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    return run([&] { return bridge::moduleUnload(hmod); });
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    DRV_CHECK(currentContext());
    return run([&] { return bridge::moduleGetFunction(hmod, name, reinterpret_cast<void **>(hfunc)); });
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    DRV_CHECK(currentContext());
    return run([&] { return bridge::moduleGetGlobal(hmod, name, reinterpret_cast<void **>(dptr), bytes); });
}

CUresult CUDAAPI cuModuleGetFunctionCount(unsigned int *count, CUmodule mod) {
    if (count == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    std::vector<void *> functions;
    DRV_CHECK(run([&] { return bridge::moduleFunctions(mod, functions); }));
    *count = static_cast<unsigned int>(functions.size());
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleEnumerateFunctions(CUfunction *functions, unsigned int numFunctions, CUmodule mod) {
    if (functions == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    std::vector<void *> all;
    DRV_CHECK(run([&] { return bridge::moduleFunctions(mod, all); }));
    for (unsigned int i = 0; i < numFunctions && i < all.size(); i++) {
        functions[i] = static_cast<CUfunction>(all[i]);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
    if (mode == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // modules are loaded when created; functions are prepared when first used
    *mode = CU_MODULE_EAGER_LOADING;
    return CUDA_SUCCESS;
}

// linking: one input (cubin, fat binary or PTX), compiled for the current device

struct CUlinkState_st {
    std::vector<std::vector<uint8_t>> inputs{};
    std::vector<uint8_t> cubin{};
};

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues,
                              CUlinkState *stateOut) {
    if (stateOut == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    answerJitOptions(numOptions, options, optionValues);
    *stateOut = new CUlinkState_st{};
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *,
                               unsigned int numOptions, CUjit_option *options, void **optionValues) {
    if (state == nullptr || data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    answerJitOptions(numOptions, options, optionValues);
    if (type == CU_JIT_INPUT_LIBRARY || type == CU_JIT_INPUT_OBJECT || type == CU_JIT_INPUT_NVVM) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const auto *bytes = static_cast<const uint8_t *>(data);
    std::vector<uint8_t> input(bytes, bytes + size);
    if (type == CU_JIT_INPUT_PTX && (input.empty() || input.back() != 0)) {
        input.push_back(0);
    }
    state->inputs.push_back(std::move(input));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
    std::vector<uint8_t> data;
    if (path == nullptr || !readFile(path, data)) {
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    return cuLinkAddData(state, type, data.data(), data.size(), path, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    if (state == nullptr || cubinOut == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // linking several inputs needs a device linker
    if (state->inputs.size() != 1) {
        return state->inputs.empty() ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_NOT_SUPPORTED;
    }
    DRV_CHECK(currentContext());
    DRV_CHECK(run([&] { return bridge::compileImage(state->inputs[0].data(), state->cubin); }));
    *cubinOut = state->cubin.data();
    if (sizeOut != nullptr) {
        *sizeOut = state->cubin.size();
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
    delete state;
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// functions and launches

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    if (pi == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (hfunc == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    DRV_CHECK(currentContext());
    cudaFuncAttributes attr{};
    DRV_CHECK(run([&] { return cudaFuncGetAttributes(&attr, kernel(hfunc)); }));
    switch (attrib) {
        case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = attr.maxThreadsPerBlock;
            break;
        case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
            *pi = static_cast<int>(attr.sharedSizeBytes);
            break;
        case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
            *pi = static_cast<int>(attr.constSizeBytes);
            break;
        case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
            *pi = static_cast<int>(attr.localSizeBytes);
            break;
        case CU_FUNC_ATTRIBUTE_NUM_REGS:
            *pi = attr.numRegs;
            break;
        case CU_FUNC_ATTRIBUTE_PTX_VERSION:
            *pi = attr.ptxVersion;
            break;
        case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
            *pi = attr.binaryVersion;
            break;
        case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
            *pi = attr.cacheModeCA;
            break;
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
            *pi = attr.maxDynamicSharedSizeBytes;
            break;
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
            *pi = attr.preferredShmemCarveout;
            break;
        case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET:
            *pi = attr.clusterDimMustBeSet;
            break;
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH:
            *pi = attr.requiredClusterWidth;
            break;
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
            *pi = attr.requiredClusterHeight;
            break;
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH:
            *pi = attr.requiredClusterDepth;
            break;
        case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED:
            *pi = attr.nonPortableClusterSizeAllowed;
            break;
        case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
            *pi = attr.clusterSchedulingPolicyPreference;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    if (hfunc == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    DRV_CHECK(currentContext());
    // the settable attributes have the runtime's numbers
    return run([&] { return cudaFuncSetAttribute(kernel(hfunc), static_cast<cudaFuncAttribute>(attrib), value); });
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    if (hfunc == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    DRV_CHECK(currentContext());
    return run([&] { return cudaFuncSetCacheConfig(kernel(hfunc), static_cast<cudaFuncCache>(config)); });
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig) {
    return hfunc == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetName(const char **name, CUfunction hfunc) {
    if (name == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = bridge::functionName(kernel(hfunc));
    return *name != nullptr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
    if (hmod == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *hmod = static_cast<CUmodule>(bridge::functionModule(kernel(hfunc)));
    return *hmod != nullptr ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuFuncLoad(CUfunction function) {
    return bridge::isFunction(kernel(function)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuFuncIsLoaded(CUfunctionLoadingState *state, CUfunction function) {
    if (state == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *state = CU_FUNCTION_LOADING_STATE_LOADED;
    return bridge::isFunction(kernel(function)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t *paramOffset, size_t *paramSize) {
    DRV_CHECK(currentContext());
    std::vector<std::pair<size_t, size_t>> params;
    DRV_CHECK(run([&] { return bridge::functionParams(kernel(func), params); }));
    if (paramIndex >= params.size()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (paramOffset != nullptr) {
        *paramOffset = params[paramIndex].first;
    }
    if (paramSize != nullptr) {
        *paramSize = params[paramIndex].second;
    }
    return CUDA_SUCCESS;
}

namespace {

/**
 * The kernel parameters of a launch: kernelParams, or pointers into the buffer of the extra options
 * (CU_LAUNCH_PARAM_BUFFER_POINTER / CU_LAUNCH_PARAM_BUFFER_SIZE)
 */
CUresult launchParams(CUfunction f, void **kernelParams, void **extra, std::vector<void *> &storage,
                      void ***pArgs) {
    if (extra == nullptr) {
        *pArgs = kernelParams;
        return CUDA_SUCCESS;
    }
    if (kernelParams != nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint8_t *buffer = nullptr;
    size_t size = 0;
    for (size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
        if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            buffer = static_cast<uint8_t *>(extra[i + 1]);
        } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            size = *static_cast<size_t *>(extra[i + 1]);
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    std::vector<std::pair<size_t, size_t>> params;
    DRV_CHECK(run([&] { return bridge::functionParams(kernel(f), params); }));
    storage.clear();
    for (const auto &[offset, param_size]: params) {
        if (buffer == nullptr || offset + param_size > size) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        storage.push_back(buffer + offset);
    }
    *pArgs = storage.data();
    return CUDA_SUCCESS;
}

} // namespace

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    if (f == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    DRV_CHECK(currentContext());
    std::vector<void *> storage;
    void **args{};
    DRV_CHECK(launchParams(f, kernelParams, extra, storage, &args));
    return run([&] {
        return cudaLaunchKernel(kernel(f), dim3(gridDimX, gridDimY, gridDimZ), dim3(blockDimX, blockDimY, blockDimZ),
                                args, sharedMemBytes, hStream);
    });
}

CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                           unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                           unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                                           void **kernelParams) {
    if (f == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaLaunchCooperativeKernel(kernel(f), dim3(gridDimX, gridDimY, gridDimZ),
                                           dim3(blockDimX, blockDimY, blockDimZ), kernelParams, sharedMemBytes,
                                           hStream);
    });
}

namespace {

cudaLaunchConfig_t runtimeConfig(const CUlaunchConfig *config) {
    static_assert(sizeof(CUlaunchAttribute) == sizeof(cudaLaunchAttribute));
    cudaLaunchConfig_t out{};
    out.gridDim = dim3(config->gridDimX, config->gridDimY, config->gridDimZ);
    out.blockDim = dim3(config->blockDimX, config->blockDimY, config->blockDimZ);
    out.dynamicSmemBytes = config->sharedMemBytes;
    out.stream = config->hStream;
    // the attributes (and their identifiers) are the runtime's
    out.attrs = reinterpret_cast<cudaLaunchAttribute *>(config->attrs);
    out.numAttrs = config->numAttrs;
    return out;
}

} // namespace

CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra) {
    if (config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (f == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    DRV_CHECK(currentContext());
    std::vector<void *> storage;
    void **args{};
    DRV_CHECK(launchParams(f, kernelParams, extra, storage, &args));
    const cudaLaunchConfig_t runtime_config = runtimeConfig(config);
    return run([&] { return cudaLaunchKernelExC(&runtime_config, kernel(f), args); });
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize,
                                                             size_t dynamicSMemSize) {
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, kernel(func), blockSize, dynamicSMemSize);
    });
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize,
                                                                      size_t dynamicSMemSize, unsigned int flags) {
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, kernel(func), blockSize,
                                                                      dynamicSMemSize, flags);
    });
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func,
                                                           CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                           size_t dynamicSMemSize, int blockSizeLimit,
                                                           unsigned int flags) {
    if (minGridSize == nullptr || blockSize == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUctx_st *ctx{};
    DRV_CHECK(currentContext(&ctx));
    int max_threads = 0, sm_count = 0, warp_size = 32;
    DRV_CHECK(cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func));
    DRV_CHECK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ctx->device));
    DRV_CHECK(cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, ctx->device));
    const int limit = blockSizeLimit > 0 ? std::min(blockSizeLimit, max_threads) : max_threads;
    // the block size (a multiple of the warp size, largest first) with the most resident threads per SM
    int best_size = 0, best_threads = -1, best_blocks = 0;
    for (int size = (limit / warp_size) * warp_size; size > 0; size -= warp_size) {
        const size_t smem = blockSizeToDynamicSMemSize != nullptr ? blockSizeToDynamicSMemSize(size)
                                                                  : dynamicSMemSize;
        int blocks = 0;
        DRV_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&blocks, func, size, smem, flags));
        if (blocks * size > best_threads) {
            best_threads = blocks * size;
            best_size = size;
            best_blocks = blocks;
        }
    }
    *blockSize = best_size;
    *minGridSize = best_blocks * sm_count;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
                                                  CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                  size_t dynamicSMemSize, int blockSizeLimit) {
    return cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                                                     dynamicSMemSize, blockSizeLimit, CU_OCCUPANCY_DEFAULT);
}

CUresult CUDAAPI cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks,
                                                         int blockSize) {
    DRV_CHECK(currentContext());
    return run([&] {
        return cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, kernel(func), numBlocks, blockSize);
    });
}

CUresult CUDAAPI cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func, const CUlaunchConfig *config) {
    if (config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    const cudaLaunchConfig_t runtime_config = runtimeConfig(config);
    return run([&] { return cudaOccupancyMaxActiveClusters(numClusters, kernel(func), &runtime_config); });
}

CUresult CUDAAPI cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func, const CUlaunchConfig *config) {
    if (config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    DRV_CHECK(currentContext());
    const cudaLaunchConfig_t runtime_config = runtimeConfig(config);
    return run([&] { return cudaOccupancyMaxPotentialClusterSize(clusterSize, kernel(func), &runtime_config); });
}

// tensor maps: not supported (their descriptor format is not public)

CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *, const cuuint64_t *,
                                        const cuuint64_t *, const cuuint32_t *, const cuuint32_t *,
                                        CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
                                        CUtensorMapFloatOOBfill) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------------------------------------------------
// entry points

namespace {

// name as passed to cuGetProcAddress (without version suffix) -> the newest version of the function. The macros of
// cuda.h map most names to their newest version (cuMemAlloc -> cuMemAlloc_v2); # keeps the name as written.
#define ENTRY(name) {#name, reinterpret_cast<void *>(&name)}
const std::map<std::string, void *> &entryPoints() {
    static const std::map<std::string, void *> entries{
        ENTRY(cuInit), ENTRY(cuDriverGetVersion), ENTRY(cuGetErrorName), ENTRY(cuGetErrorString),
        ENTRY(cuGetProcAddress), ENTRY(cuGetExportTable), ENTRY(cuProfilerStart), ENTRY(cuProfilerStop),
        // devices
        ENTRY(cuDeviceGet), ENTRY(cuDeviceGetCount), ENTRY(cuDeviceGetName), ENTRY(cuDeviceGetUuid),
        ENTRY(cuDeviceGetLuid), ENTRY(cuDeviceTotalMem), ENTRY(cuDeviceGetAttribute),
        ENTRY(cuDeviceComputeCapability), ENTRY(cuDeviceGetProperties), ENTRY(cuDeviceGetPCIBusId),
        ENTRY(cuDeviceGetByPCIBusId), ENTRY(cuDeviceCanAccessPeer), ENTRY(cuDeviceGetP2PAttribute),
        ENTRY(cuDeviceGetDefaultMemPool), ENTRY(cuDeviceGetMemPool), ENTRY(cuDeviceSetMemPool),
        // contexts
        ENTRY(cuDevicePrimaryCtxRetain), ENTRY(cuDevicePrimaryCtxRelease), ENTRY(cuDevicePrimaryCtxSetFlags),
        ENTRY(cuDevicePrimaryCtxGetState), ENTRY(cuDevicePrimaryCtxReset), ENTRY(cuCtxCreate), ENTRY(cuCtxDestroy),
        ENTRY(cuCtxPushCurrent), ENTRY(cuCtxPopCurrent), ENTRY(cuCtxSetCurrent), ENTRY(cuCtxGetCurrent),
        ENTRY(cuCtxGetDevice), ENTRY(cuCtxGetDevice_v2), ENTRY(cuCtxGetFlags), ENTRY(cuCtxSetFlags),
        ENTRY(cuCtxGetId), ENTRY(cuCtxGetApiVersion), ENTRY(cuCtxSynchronize), ENTRY(cuCtxSynchronize_v2),
        ENTRY(cuCtxSetLimit), ENTRY(cuCtxGetLimit), ENTRY(cuCtxGetCacheConfig), ENTRY(cuCtxSetCacheConfig),
        ENTRY(cuCtxGetStreamPriorityRange), ENTRY(cuCtxResetPersistingL2Cache), ENTRY(cuCtxEnablePeerAccess),
        ENTRY(cuCtxDisablePeerAccess),
        // memory
        ENTRY(cuMemGetInfo), ENTRY(cuMemAlloc), ENTRY(cuMemAllocPitch), ENTRY(cuMemFree),
        ENTRY(cuMemGetAddressRange), ENTRY(cuMemAllocHost), ENTRY(cuMemHostAlloc), ENTRY(cuMemFreeHost),
        ENTRY(cuMemHostGetDevicePointer), ENTRY(cuMemHostGetFlags), ENTRY(cuMemHostRegister),
        ENTRY(cuMemHostUnregister), ENTRY(cuMemAllocManaged), ENTRY(cuMemAllocAsync), ENTRY(cuMemFreeAsync),
        ENTRY(cuMemPoolSetAttribute), ENTRY(cuMemPoolGetAttribute), ENTRY(cuMemPoolSetAccess),
        ENTRY(cuMemPoolGetAccess), ENTRY(cuMemPoolTrimTo), ENTRY(cuMemPrefetchAsync), ENTRY(cuMemAdvise),
        ENTRY(cuMemRangeGetAttribute), ENTRY(cuMemAddressReserve), ENTRY(cuMemAddressFree), ENTRY(cuMemCreate),
        ENTRY(cuMemRelease), ENTRY(cuMemMap), ENTRY(cuMemUnmap), ENTRY(cuMemSetAccess),
        ENTRY(cuMemGetAllocationGranularity), ENTRY(cuMemRetainAllocationHandle),
        ENTRY(cuMemGetAllocationPropertiesFromHandle), ENTRY(cuMemExportToShareableHandle),
        ENTRY(cuMemImportFromShareableHandle), ENTRY(cuIpcGetMemHandle), ENTRY(cuIpcOpenMemHandle),
        ENTRY(cuIpcCloseMemHandle), ENTRY(cuIpcGetEventHandle), ENTRY(cuIpcOpenEventHandle),
        ENTRY(cuPointerGetAttribute), ENTRY(cuPointerGetAttributes), ENTRY(cuPointerSetAttribute),
        // copies and fills
        ENTRY(cuMemcpy), ENTRY(cuMemcpyAsync), ENTRY(cuMemcpyHtoD), ENTRY(cuMemcpyDtoH), ENTRY(cuMemcpyDtoD),
        ENTRY(cuMemcpyHtoDAsync), ENTRY(cuMemcpyDtoHAsync), ENTRY(cuMemcpyDtoDAsync), ENTRY(cuMemcpyPeer),
        ENTRY(cuMemcpyPeerAsync), ENTRY(cuMemcpy2D), ENTRY(cuMemcpy2DUnaligned), ENTRY(cuMemcpy2DAsync),
        ENTRY(cuMemcpy3D), ENTRY(cuMemcpy3DAsync), ENTRY(cuMemsetD8), ENTRY(cuMemsetD16), ENTRY(cuMemsetD32),
        ENTRY(cuMemsetD8Async), ENTRY(cuMemsetD16Async), ENTRY(cuMemsetD32Async), ENTRY(cuMemsetD2D8),
        ENTRY(cuMemsetD2D16), ENTRY(cuMemsetD2D32), ENTRY(cuMemsetD2D8Async), ENTRY(cuMemsetD2D16Async),
        ENTRY(cuMemsetD2D32Async),
        // streams and graphs
        ENTRY(cuStreamCreate), ENTRY(cuStreamCreateWithPriority), ENTRY(cuStreamDestroy),
        ENTRY(cuStreamSynchronize), ENTRY(cuStreamQuery), ENTRY(cuStreamWaitEvent), ENTRY(cuStreamGetFlags),
        ENTRY(cuStreamGetPriority), ENTRY(cuStreamGetId), ENTRY(cuStreamGetDevice), ENTRY(cuStreamGetCtx),
        ENTRY(cuStreamGetCtx_v2), ENTRY(cuStreamAddCallback), ENTRY(cuLaunchHostFunc),
        ENTRY(cuStreamAttachMemAsync), ENTRY(cuStreamWaitValue32), ENTRY(cuStreamWaitValue64),
        ENTRY(cuStreamWriteValue32), ENTRY(cuStreamWriteValue64), ENTRY(cuStreamBeginCapture),
        ENTRY(cuStreamEndCapture), ENTRY(cuStreamIsCapturing), ENTRY(cuThreadExchangeStreamCaptureMode),
        ENTRY(cuGraphInstantiateWithFlags), ENTRY(cuGraphInstantiate), ENTRY(cuGraphLaunch), ENTRY(cuGraphUpload),
        ENTRY(cuGraphExecDestroy), ENTRY(cuGraphDestroy), ENTRY(cuGraphCreate),
        // events
        ENTRY(cuEventCreate), ENTRY(cuEventRecord), ENTRY(cuEventRecordWithFlags), ENTRY(cuEventQuery),
        ENTRY(cuEventSynchronize), ENTRY(cuEventDestroy), ENTRY(cuEventElapsedTime),
        // modules
        ENTRY(cuModuleLoadData), ENTRY(cuModuleLoadDataEx), ENTRY(cuModuleLoadFatBinary), ENTRY(cuModuleLoad),
        ENTRY(cuModuleUnload), ENTRY(cuModuleGetFunction), ENTRY(cuModuleGetGlobal), ENTRY(cuModuleGetFunctionCount),
        ENTRY(cuModuleEnumerateFunctions), ENTRY(cuModuleGetLoadingMode), ENTRY(cuLinkCreate), ENTRY(cuLinkAddData),
        ENTRY(cuLinkAddFile), ENTRY(cuLinkComplete), ENTRY(cuLinkDestroy),
        // functions and launches
        ENTRY(cuFuncGetAttribute), ENTRY(cuFuncSetAttribute), ENTRY(cuFuncSetCacheConfig),
        ENTRY(cuFuncSetSharedMemConfig), ENTRY(cuFuncGetName), ENTRY(cuFuncGetModule), ENTRY(cuFuncLoad),
        ENTRY(cuFuncIsLoaded), ENTRY(cuFuncGetParamInfo), ENTRY(cuLaunchKernel), ENTRY(cuLaunchCooperativeKernel),
        ENTRY(cuLaunchKernelEx), ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessor),
        ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags), ENTRY(cuOccupancyMaxPotentialBlockSize),
        ENTRY(cuOccupancyMaxPotentialBlockSizeWithFlags), ENTRY(cuOccupancyAvailableDynamicSMemPerBlock),
        ENTRY(cuOccupancyMaxActiveClusters), ENTRY(cuOccupancyMaxPotentialClusterSize),
        ENTRY(cuTensorMapEncodeTiled),
    };
    return entries;
}
#undef ENTRY

} // namespace

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                                  CUdriverProcAddressQueryResult *symbolStatus) {
    if (symbol == nullptr || pfn == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    (void) flags;
    // the newest version of each function is provided (callers built for older versions get it too); the per-thread
    // default stream variants are the same functions (the per-thread default stream is the legacy default stream)
    (void) cudaVersion;
    const auto &entries = entryPoints();
    auto it = entries.find(symbol);
    if (it == entries.end()) {
        *pfn = nullptr;
        if (symbolStatus != nullptr) {
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        }
        return CUDA_ERROR_NOT_FOUND;
    }
    *pfn = it->second;
    if (symbolStatus != nullptr) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
}

namespace librecuda_bridge {

int driverEntryPoint(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags, int *symbolStatus) {
    CUdriverProcAddressQueryResult status{};
    const CUresult result = cuGetProcAddress(symbol, pfn, cudaVersion, flags, &status);
    *symbolStatus = status;
    return result == CUDA_ERROR_NOT_FOUND ? 0 : result;
}

} // namespace librecuda_bridge
