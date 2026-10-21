#include <cstdint>

#define MEMCPY_BLOCK_SIZE 256UL

extern "C" __global__ void memcpyKernelHighBW(uint32_t *dst, const uint32_t *src) {
    size_t idx = ((MEMCPY_BLOCK_SIZE * (blockIdx.y * gridDim.x + blockIdx.x)) << 2) + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[idx] = src[idx];
        idx += MEMCPY_BLOCK_SIZE;
    }
}

extern "C" __global__ void memcpyKernelLowLatency(uint32_t *dst, const uint32_t *src, size_t n) {
    size_t tid = MEMCPY_BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

extern "C" __global__ void memcpyKernelTrailing(uint8_t *dst, const uint8_t *src, size_t n) {
    size_t tid = MEMCPY_BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}
// Grid-stride copy / fill kernels used for device memory operations (cudaMemcpy D2D, cudaMemset).
// The driver picks the widest variant the alignment of the pointers allows and handles unaligned head/tail bytes
// with the byte variants.

extern "C" __global__ void memcpyVec16(uint4 *__restrict__ dst, const uint4 *__restrict__ src, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    // two independent loads in flight per iteration
    for (; i + stride < n; i += 2 * stride) {
        uint4 a = src[i];
        uint4 b = src[i + stride];
        dst[i] = a;
        dst[i + stride] = b;
    }
    if (i < n) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void memcpyVec4(uint32_t *__restrict__ dst, const uint32_t *__restrict__ src, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void memcpyBytes(uint8_t *__restrict__ dst, const uint8_t *__restrict__ src, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void memsetVec16(uint4 *__restrict__ dst, uint32_t value, size_t n) {
    const uint4 v = make_uint4(value, value, value, value);
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = v;
    }
}

extern "C" __global__ void memsetBytes(uint8_t *__restrict__ dst, uint32_t value, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = static_cast<uint8_t>(value);
    }
}
