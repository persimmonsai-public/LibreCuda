// NVCC_FLAGS: -arch=$ARCH -rdc=true
// Device runtime queries, compared with the host's answers: cudaDeviceGetAttribute, cudaDeviceGetLimit,
// cudaDeviceGetCacheConfig, cudaGetErrorName / cudaGetErrorString, cudaFuncGetAttributes and
// cudaOccupancyMaxActiveBlocksPerMultiprocessor(WithFlags) in device code; and device side cudaMemcpy3DAsync to and from
// CUDA arrays (a LibreCUDA extension: NVIDIA's device runtime refuses them). Checked against NVIDIA's runtime too.
#include "check.h"
#include <cstring>

constexpr int ATTRIBUTE_COUNT = 8;
__constant__ cudaDeviceAttr ATTRIBUTES[ATTRIBUTE_COUNT] = {
    cudaDevAttrMaxThreadsPerBlock, cudaDevAttrMultiProcessorCount, cudaDevAttrComputeCapabilityMajor,
    cudaDevAttrComputeCapabilityMinor, cudaDevAttrMaxSharedMemoryPerBlock, cudaDevAttrMaxRegistersPerMultiprocessor,
    cudaDevAttrWarpSize, cudaDevAttrMaxBlocksPerMultiprocessor};

struct Answers {
    int attributes[ATTRIBUTE_COUNT];
    int attributeStatus;
    size_t limits[3];
    int cacheConfig;
    char errorNames[3][48];
    char errorStrings[3][64];
    cudaFuncAttributes functions[2];
    int functionStatus;
    int occupancy[4];
    int occupancyStatus;
};

__global__ void heavy(float *p) {
    __shared__ float tile[1024];
    tile[threadIdx.x] = p[threadIdx.x];
    __syncthreads();
    p[threadIdx.x] = tile[(threadIdx.x + 1) % blockDim.x] * 2.0f;
}

__global__ void light(int *p) {
    p[threadIdx.x] = threadIdx.x;
}

__device__ void copyString(char *dst, const char *src, int size) {
    int i = 0;
    for (; src != nullptr && src[i] != 0 && i < size - 1; i++) {
        dst[i] = src[i];
    }
    dst[i] = 0;
}

__global__ void query(Answers *a, int device) {
    a->attributeStatus = 0;
    for (int i = 0; i < ATTRIBUTE_COUNT; i++) {
        a->attributeStatus |= cudaDeviceGetAttribute(&a->attributes[i], ATTRIBUTES[i], device);
    }
    cudaDeviceGetLimit(&a->limits[0], cudaLimitStackSize);
    cudaDeviceGetLimit(&a->limits[1], cudaLimitMallocHeapSize);
    cudaDeviceGetLimit(&a->limits[2], cudaLimitDevRuntimePendingLaunchCount);
    cudaFuncCache cache;
    cudaDeviceGetCacheConfig(&cache);
    a->cacheConfig = cache;
    const cudaError_t errors[3] = {cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure};
    for (int i = 0; i < 3; i++) {
        copyString(a->errorNames[i], cudaGetErrorName(errors[i]), sizeof(a->errorNames[i]));
        copyString(a->errorStrings[i], cudaGetErrorString(errors[i]), sizeof(a->errorStrings[i]));
    }
    a->functionStatus = cudaFuncGetAttributes(&a->functions[0], heavy) | cudaFuncGetAttributes(&a->functions[1], light);
    a->occupancyStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a->occupancy[0], heavy, 256, 0) |
                         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a->occupancy[1], heavy, 1024, 16384) |
                         cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a->occupancy[2], light, 64, 0) |
                         cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&a->occupancy[3], light, 128, 0,
                                                                                cudaOccupancyDefault);
}

// device side copies: linear -> array -> linear
__global__ void arrayCopies(cudaArray_t array, int *src, int *dst, int width, int height, int *status) {
    // (make_cudaPitchedPtr / make_cudaExtent are host functions)
    cudaMemcpy3DParms to{};
    to.srcPtr.ptr = src;
    to.srcPtr.pitch = width * sizeof(int);
    to.srcPtr.xsize = width;
    to.srcPtr.ysize = height;
    to.dstArray = array;
    to.extent.width = width;
    to.extent.height = height;
    to.extent.depth = 1;
    to.kind = cudaMemcpyDeviceToDevice;
    status[0] = cudaMemcpy3DAsync(&to, 0);
    cudaMemcpy3DParms from{};
    from.srcArray = array;
    from.dstPtr.ptr = dst;
    from.dstPtr.pitch = width * sizeof(int);
    from.dstPtr.xsize = width;
    from.dstPtr.ysize = height;
    from.extent = to.extent;
    from.kind = cudaMemcpyDeviceToDevice;
    status[1] = cudaMemcpy3DAsync(&from, 0);
}

static bool sameAttributes(const cudaFuncAttributes &a, const cudaFuncAttributes &b) {
    return a.sharedSizeBytes == b.sharedSizeBytes && a.constSizeBytes == b.constSizeBytes &&
           a.localSizeBytes == b.localSizeBytes && a.maxThreadsPerBlock == b.maxThreadsPerBlock &&
           a.numRegs == b.numRegs && a.ptxVersion == b.ptxVersion && a.binaryVersion == b.binaryVersion;
    // (device side, maxDynamicSharedSizeBytes and preferredShmemCarveout are 0 with NVIDIA's runtime)
}

int main() {
    int device = 0;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    Answers *a;
    CHECK(cudaMallocManaged(&a, sizeof(Answers)));
    memset(a, 0, sizeof(Answers));
    query<<<1, 1>>>(a, device);
    CHECK(cudaDeviceSynchronize());

    const cudaDeviceAttr attributes[ATTRIBUTE_COUNT] = {
        cudaDevAttrMaxThreadsPerBlock, cudaDevAttrMultiProcessorCount, cudaDevAttrComputeCapabilityMajor,
        cudaDevAttrComputeCapabilityMinor, cudaDevAttrMaxSharedMemoryPerBlock, cudaDevAttrMaxRegistersPerMultiprocessor,
        cudaDevAttrWarpSize, cudaDevAttrMaxBlocksPerMultiprocessor};
    EXPECT(a->attributeStatus == 0, "device cudaDeviceGetAttribute status %d", a->attributeStatus);
    for (int i = 0; i < ATTRIBUTE_COUNT; i++) {
        int host = 0;
        CHECK(cudaDeviceGetAttribute(&host, attributes[i], device));
        EXPECT(a->attributes[i] == host, "attribute %d: device %d, host %d", int(attributes[i]), a->attributes[i],
               host);
    }
    const cudaLimit limits[3] = {cudaLimitStackSize, cudaLimitMallocHeapSize, cudaLimitDevRuntimePendingLaunchCount};
    for (int i = 0; i < 3; i++) {
        size_t host = 0;
        CHECK(cudaDeviceGetLimit(&host, limits[i]));
        EXPECT(a->limits[i] == host, "limit %d: device %zu, host %zu", int(limits[i]), a->limits[i], host);
    }
    cudaFuncCache cache;
    CHECK(cudaDeviceGetCacheConfig(&cache));
    EXPECT(a->cacheConfig == cache, "cache config: device %d, host %d", a->cacheConfig, int(cache));
    const cudaError_t errors[3] = {cudaSuccess, cudaErrorInvalidValue, cudaErrorLaunchFailure};
    for (int i = 0; i < 3; i++) {
        EXPECT(strcmp(a->errorNames[i], cudaGetErrorName(errors[i])) == 0, "error name %d: %s / %s", int(errors[i]),
               a->errorNames[i], cudaGetErrorName(errors[i]));
        EXPECT(strncmp(a->errorStrings[i], cudaGetErrorString(errors[i]), sizeof(a->errorStrings[i]) - 1) == 0,
               "error string %d: %s", int(errors[i]), a->errorStrings[i]);
    }
    EXPECT(a->functionStatus == 0, "device cudaFuncGetAttributes status %d", a->functionStatus);
    cudaFuncAttributes heavy_host, light_host;
    CHECK(cudaFuncGetAttributes(&heavy_host, heavy));
    CHECK(cudaFuncGetAttributes(&light_host, light));
    EXPECT(sameAttributes(a->functions[0], heavy_host) && sameAttributes(a->functions[1], light_host),
           "function attributes: device shared %zu regs %d, host shared %zu regs %d", a->functions[0].sharedSizeBytes,
           a->functions[0].numRegs, heavy_host.sharedSizeBytes, heavy_host.numRegs);
    EXPECT(a->occupancyStatus == 0, "device occupancy status %d", a->occupancyStatus);
    int occupancy[4];
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[0], heavy, 256, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[1], heavy, 1024, 16384));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy[2], light, 64, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancy[3], light, 128, 0, cudaOccupancyDefault));
    for (int i = 0; i < 4; i++) {
        EXPECT(a->occupancy[i] == occupancy[i], "occupancy %d: device %d, host %d", i, a->occupancy[i], occupancy[i]);
    }
    PASS("device runtime queries");

    // device side copies to and from a CUDA array
    const int width = 64, height = 16;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaArray_t array;
    CHECK(cudaMallocArray(&array, &desc, width, height));
    int *src, *dst, *status;
    CHECK(cudaMallocManaged(&src, width * height * sizeof(int)));
    CHECK(cudaMallocManaged(&dst, width * height * sizeof(int)));
    CHECK(cudaMallocManaged(&status, 2 * sizeof(int)));
    for (int i = 0; i < width * height; i++) {
        src[i] = 7 * i + 1;
        dst[i] = 0;
    }
    status[0] = status[1] = -1;
    arrayCopies<<<1, 1>>>(array, src, dst, width, height, status);
    CHECK(cudaDeviceSynchronize());
    // NVIDIA's device runtime refuses copies involving arrays (cudaErrorInvalidValue); LibreCUDA runs them
    const bool supported = status[0] == 0 && status[1] == 0;
    EXPECT(supported || (status[0] == cudaErrorInvalidValue && status[1] == cudaErrorInvalidValue),
           "device cudaMemcpy3DAsync status %d %d", status[0], status[1]);
    for (int i = 0; supported && i < width * height; i++) {
        EXPECT(dst[i] == 7 * i + 1, "array round trip [%d] = %d", i, dst[i]);
    }
    CHECK(cudaFreeArray(array));
    CHECK(cudaFree(status));
    CHECK(cudaFree(dst));
    CHECK(cudaFree(src));
    CHECK(cudaFree(a));
    CHECK(cudaGetLastError());
    if (supported) {
        PASS("device side copies with CUDA arrays");
    } else {
        printf("PASS device side copies with CUDA arrays (refused by this runtime)\n");
    }
    return 0;
}
