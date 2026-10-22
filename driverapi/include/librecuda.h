#ifndef LIBRECUDA_H_
#define LIBRECUDA_H_

#include <cstdint>
#include "librecuda_status.h"

#define LIBRECUDA_EXPORT extern "C"

struct LibreCUdevice_;
struct LibreCUcontext_;
struct LibreCUmodule_;
struct LibreCUFunction_;
struct LibreCUstream_;
struct LibreCUEvent_;

typedef LibreCUdevice_ *LibreCUdevice;
typedef LibreCUcontext_ *LibreCUcontext;
typedef LibreCUmodule_ *LibreCUmodule;
typedef LibreCUFunction_ *LibreCUFunction;
typedef LibreCUstream_ *LibreCUstream;
typedef LibreCUEvent_ *LibreCUEvent;

#define CU_CTX_SCHED_SPIN 0x01
#define CU_CTX_SCHED_YIELD 0x02
#define CU_CTX_SCHED_BLOCKING_SYNC 0x04

// values follow the numbering of the public CUDA driver API headers
enum LibreCuDeviceAttribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,

    // LibreCUDA specific: number of GPCs (graphics processing clusters); a thread block cluster runs within one GPC
    LIBRECUDA_DEVICE_ATTRIBUTE_GPC_COUNT = 0x10000,
};
enum LibreCuFunctionAttribute {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
};

/**
 * Maximum number of blocks in a thread block cluster (the portable cluster size)
 */
#define LIBRECUDA_MAX_CLUSTER_SIZE 8

LIBRECUDA_EXPORT libreCudaStatus_t libreCuInit(int flags);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuDeviceGetCount(int *pDeviceCount);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuDeviceGet(LibreCUdevice *device, int deviceOrdinal);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuGetErrorString(libreCudaStatus_t status, const char **pStatusName);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxCreate_v2(LibreCUcontext *pCtx, int flags, LibreCUdevice device);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxDestroy(LibreCUcontext ctx);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxSetCurrent(LibreCUcontext ctx);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxGetCurrent(LibreCUcontext *pCtxOut);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemAlloc(void **pDevicePointer, size_t bytesize, bool mapToCpu = false);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemCpy(void *dst, void *src, size_t byteCount, LibreCUstream stream, bool async = false);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemFree(void *devicePointer);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuStreamCreate(LibreCUstream *pStreamOut, uint32_t flags);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuStreamDestroy(LibreCUstream stream);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuDeviceGetName(char *pDeviceName, int length, LibreCUdevice device);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuDeviceGetAttribute(int *pValOut, LibreCuDeviceAttribute attribute, LibreCUdevice device);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuFuncSetAttribute(LibreCUFunction function, LibreCuFunctionAttribute attribute, int value);

LIBRECUDA_EXPORT libreCudaStatus_t
libreCuModuleLoadData(LibreCUmodule *pModuleOut, const void *image, size_t imageSize);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuModuleGetFunctionCount(uint32_t *count, LibreCUmodule mod);

LIBRECUDA_EXPORT libreCudaStatus_t
libreCuModuleEnumerateFunctions(LibreCUFunction *functionsOut, uint32_t numFunctions, LibreCUmodule mod);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuFuncGetName(const char **pNameOut, LibreCUFunction func);

/**
 * Number of kernel parameters, i.e. the number of entries libreCuLaunchKernel expects in kernelParams
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuFuncGetParamCount(size_t *pCountOut, LibreCUFunction func);

/**
 * Whether launches of the function wait until it and the kernels it launches from the device completed (dynamic
 * parallelism; launches of such kernels are synchronous)
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuFuncGetLaunchesKernels(bool *pOut, LibreCUFunction func);

LIBRECUDA_EXPORT libreCudaStatus_t
libreCuModuleGetFunction(LibreCUFunction *pFunc, LibreCUmodule module, const char *name);

/**
 * Texture object handle (cudaTextureObject_t): index of the texture's header and sampler in the context's pools;
 * 0 means none
 */
typedef uint64_t LibreCUtexObject;

enum LibreCuTexFormat {
    LIBRECUDA_TEX_FORMAT_FLOAT,
    LIBRECUDA_TEX_FORMAT_SINT,
    LIBRECUDA_TEX_FORMAT_UINT,
    // 8 / 16 bit integers read as floats in [0, 1] / [-1, 1] (cudaReadModeNormalizedFloat)
    LIBRECUDA_TEX_FORMAT_UNORM,
    LIBRECUDA_TEX_FORMAT_SNORM,
};

/**
 * Required alignment of the memory of a linear texture, and of the memory and the pitch of a pitched texture
 */
#define LIBRECUDA_TEXTURE_LINEAR_ALIGNMENT 256
#define LIBRECUDA_TEXTURE_PITCH_ALIGNMENT 32

enum LibreCuTexAddressMode {
    LIBRECUDA_TEX_ADDRESS_WRAP,
    LIBRECUDA_TEX_ADDRESS_CLAMP,
    LIBRECUDA_TEX_ADDRESS_MIRROR,
    LIBRECUDA_TEX_ADDRESS_BORDER,
};

/**
 * How a texture is sampled (cudaTextureDesc): address mode per axis, point or linear filtering, coordinates in
 * texels or normalized to [0, 1), border color for LIBRECUDA_TEX_ADDRESS_BORDER
 */
struct LibreCuTexSampler {
    LibreCuTexAddressMode addressMode[3] = {LIBRECUDA_TEX_ADDRESS_CLAMP, LIBRECUDA_TEX_ADDRESS_CLAMP,
                                            LIBRECUDA_TEX_ADDRESS_CLAMP};
    bool linearFilter = false;
    bool normalizedCoords = false;
    float borderColor[4] = {0, 0, 0, 0};
};

/**
 * Creates a texture object for linear device memory (tex1Dfetch): sizeBytes / element size elements of channels
 * (1, 2 or 4) channels of bitsPerChannel (8, 16 or 32) bits each.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuTexObjectCreateLinear(LibreCUtexObject *pTexObject, void *devPtr,
                                                                size_t sizeBytes, LibreCuTexFormat format,
                                                                uint32_t channels, uint32_t bitsPerChannel);

/**
 * Creates a texture object for a 2D texture in pitched device memory (tex2D): width x height elements, rows
 * pitchBytes apart (a multiple of LIBRECUDA_TEXTURE_PITCH_ALIGNMENT). Blackwell only.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuTexObjectCreatePitch2D(LibreCUtexObject *pTexObject, void *devPtr,
                                                                 uint32_t width, uint32_t height, size_t pitchBytes,
                                                                 LibreCuTexFormat format, uint32_t channels,
                                                                 uint32_t bitsPerChannel,
                                                                 const LibreCuTexSampler *sampler);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuTexObjectDestroy(LibreCUtexObject texObject);

/**
 * A CUDA array: texels in the GPU's block linear layout, for textures (Blackwell only). Arrays are 1D (height and
 * depth 0), 2D (depth 0) or 3D; layered arrays are arrays of 1D (height 0) or 2D textures, depth = number of layers.
 */
struct LibreCUarray_;
typedef LibreCUarray_ *LibreCUarray;

LIBRECUDA_EXPORT libreCudaStatus_t libreCuArrayCreate(LibreCUarray *pArray, uint32_t width, uint32_t height,
                                                      uint32_t depth, bool layered, LibreCuTexFormat format,
                                                      uint32_t channels, uint32_t bitsPerChannel);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuArrayDestroy(LibreCUarray array);

/**
 * One side of libreCuMemcpy3D: an array with the region's origin in texels (x, y, z; z is the layer of layered
 * arrays), or GPU accessible memory (device memory or pinned host memory) with the region's start, the bytes between
 * rows and between slices
 */
struct LibreCuMemcpy3DSide {
    LibreCUarray array;
    uint32_t x, y, z;
    void *ptr;
    size_t pitch;
    size_t slicePitch;
};

/**
 * Copies widthBytes x height x depth (bytes, rows, slices) between arrays and GPU accessible memory with the copy engine
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemcpy3D(const LibreCuMemcpy3DSide *dst, const LibreCuMemcpy3DSide *src,
                                                   size_t widthBytes, size_t height, size_t depth,
                                                   LibreCUstream stream, bool async);

/**
 * Creates a texture object for an array (tex1D / tex2D / tex3D, tex1DLayered / tex2DLayered). format says how the
 * texels are read (e.g. 8 bit integers as LIBRECUDA_TEX_FORMAT_UNORM for normalized floats); its element size must be
 * the array's.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuTexObjectCreateArray(LibreCUtexObject *pTexObject, LibreCUarray array,
                                                               LibreCuTexFormat format,
                                                               const LibreCuTexSampler *sampler);

/**
 * Launches a kernel with thread block clusters of clusterDimX x clusterDimY x clusterDimZ blocks (Blackwell). The grid
 * dimensions are in blocks and must be multiples of the cluster dimensions; a cluster has at most
 * LIBRECUDA_MAX_CLUSTER_SIZE blocks. Kernels compiled with __cluster_dims__ use their compiled cluster size with
 * libreCuLaunchKernel; launching them with a different size here fails.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuLaunchKernelCluster(LibreCUFunction function,
                                                              uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                                              uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                                              uint32_t clusterDimX, uint32_t clusterDimY,
                                                              uint32_t clusterDimZ,
                                                              uint32_t sharedMemBytes, LibreCUstream stream,
                                                              void **kernelParams, size_t numParams);

/**
 * Launches a kernel that may synchronize all of its blocks (cooperative_groups::this_grid().sync()). All blocks must
 * be resident at once: the caller must keep the grid within the occupancy limit (blocks per SM x SM count).
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuLaunchCooperativeKernel(LibreCUFunction function,
                                                                  uint32_t gridDimX, uint32_t gridDimY,
                                                                  uint32_t gridDimZ,
                                                                  uint32_t blockDimX, uint32_t blockDimY,
                                                                  uint32_t blockDimZ,
                                                                  uint32_t sharedMemBytes, LibreCUstream stream,
                                                                  void **kernelParams, size_t numParams);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuLaunchKernel(LibreCUFunction function,
                                                       uint32_t gridDimX, uint32_t gridDimY,
                                                       uint32_t gridDimZ,
                                                       uint32_t blockDimX, uint32_t blockDimY,
                                                       uint32_t blockDimZ,
                                                       uint32_t sharedMemBytes, LibreCUstream stream,
                                                       void **kernelParams, size_t numParams,
                                                       void **extra,
                                                       bool async=false);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuEventCreate(LibreCUEvent *pEventOut, uint32_t flags);
LIBRECUDA_EXPORT libreCudaStatus_t libreCuEventRecord(LibreCUEvent event, LibreCUstream stream);
LIBRECUDA_EXPORT libreCudaStatus_t libreCuEventSynchronize(LibreCUEvent event);
LIBRECUDA_EXPORT libreCudaStatus_t libreCuEventElapsedTime(float *pMillisecondsOut, LibreCUEvent start, LibreCUEvent end);
LIBRECUDA_EXPORT libreCudaStatus_t libreCuEventDestroy(LibreCUEvent event);

/**
 * Submits the built up command buffer to the gpu.
 * Operations performed on streams fall into two types: "compute" (eg. launch kernel) and "dma".
 * "dma" operations are eg. host to device / device to host memcpy operations.
 * NOTE: Device to device copies are implemented as a compute kernel, so they DO NOT COUNT AS DMA!
 * If only operations of a single type are performed, commence will not block and return near instantly.
 * To correctly interleave compute and dma operations, not all operations can be dispatched immediately and cpu-side
 * waiting is required to dispatch operations in order.
 * The more context switches between "compute" and "dma" mode you encounter, the more often cpu-side synchronization can
 * slow down execution of the timeline.
 * libreCuStreamCommence will block for the same amount of time as if libreCuStreamAwait was called, however
 * this behavior should not be relied upon. Always call libreCuStreamAwait for safety.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuStreamCommence(LibreCUstream stream);
LIBRECUDA_EXPORT libreCudaStatus_t libreCuStreamAwait(LibreCUstream stream);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuModuleUnload(LibreCUmodule module);

/**
 * Looks up a __device__ or __constant__ variable of the module.
 * Either output pointer may be null.
 */
LIBRECUDA_EXPORT libreCudaStatus_t
libreCuModuleGetGlobal(void **pDevicePtrOut, size_t *pBytesOut, LibreCUmodule module, const char *name);

LIBRECUDA_EXPORT libreCudaStatus_t
libreCuFuncGetAttribute(int *pValueOut, LibreCuFunctionAttribute attribute, LibreCUFunction function);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuDeviceTotalMem(size_t *pBytesOut, LibreCUdevice device);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuDeviceGetUuid(uint8_t uuidOut[16], LibreCUdevice device);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemGetInfo(size_t *pFreeOut, size_t *pTotalOut);

/**
 * Allocates page-locked host memory that is mapped into the GPU address space at the same virtual address.
 * Such memory can be used as source/destination of DMA copies without staging and can be accessed by kernels.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemAllocHost(void **pHostPointer, size_t bytesize);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuMemFreeHost(void *hostPointer);

/**
 * @return true if ptr points into memory returned by libreCuMemAllocHost (for any offset into the allocation)
 */
LIBRECUDA_EXPORT bool libreCuMemIsPinnedHost(const void *ptr);

/**
 * @return true if ptr points into device memory allocated by LibreCUDA
 */
LIBRECUDA_EXPORT bool libreCuMemIsDevice(const void *ptr);

/**
 * Fills numElements elements of the given size (1, 2 or 4 bytes) at dst with value, using the copy engine.
 * dst must be GPU accessible (device memory or pinned host memory).
 */
LIBRECUDA_EXPORT libreCudaStatus_t
libreCuMemsetAsync(void *dst, uint32_t value, size_t elementSize, size_t numElements, LibreCUstream stream);

/**
 * Makes all future work submitted to stream wait (on the GPU) until event has completed.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuStreamWaitEvent(LibreCUstream stream, LibreCUEvent event);

/**
 * @return LIBRECUDA_SUCCESS if all work submitted to the stream has completed, LIBRECUDA_ERROR_NOT_READY otherwise
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuStreamQuery(LibreCUstream stream);

/**
 * @return LIBRECUDA_SUCCESS if the event has completed (or was never recorded), LIBRECUDA_ERROR_NOT_READY otherwise
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuEventQuery(LibreCUEvent event);

/**
 * Sets the per thread stack size used for kernels whose stack size cannot be determined statically.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxSetStackLimit(size_t bytes);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxGetStackLimit(size_t *pBytesOut);

/**
 * Size of the buffer for device side printf output (default 1 MiB, like cudaLimitPrintfFifoSize). Can only be set
 * before the first module calling printf is loaded in the context.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxSetPrintfFifoSize(size_t size);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxGetPrintfFifoSize(size_t *pSize);

/**
 * Size of the heap for device side malloc (default 8 MiB, like cudaLimitMallocHeapSize). Can only be set before the
 * first module calling malloc / free is loaded in the context.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxSetMallocHeapSize(size_t size);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxGetMallocHeapSize(size_t *pSize);

/**
 * Number of kernel launches device code can have pending (default 2048, like cudaLimitDevRuntimePendingLaunchCount).
 * Can only be set before a module launching kernels from the device is loaded.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxSetDeviceLaunchQueueSize(uint32_t launches);

LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxGetDeviceLaunchQueueSize(uint32_t *pLaunches);

/**
 * Prints the device side printf output of the current context's completed kernels. Stream and event
 * synchronization do this automatically.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxFlushPrintf();

/**
 * Waits for the current context's work and checks whether a kernel hit an SM exception (e.g. a misaligned address).
 * Such errors are otherwise detected in the background and may be reported one synchronization late. This check
 * takes a few hundred microseconds.
 */
LIBRECUDA_EXPORT libreCudaStatus_t libreCuCtxCheckErrors();

#endif //LIBRECUDA_H_