/**
 * Kernels of LibreCUDA's NCCL device transport: a chunk of a collective combined from the rank's own data (scaled for
 * pre-multiplied sums) and a chunk received from a peer, written to up to two places (the rank's output and the slot
 * of the next peer), finally divided by the rank count for averages. 16 byte accesses where all pointers allow.
 */
#include "nccl_device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

template<typename T>
struct Compute {
    using type = T;
};
template<>
struct Compute<__half> {
    using type = float;
};
template<>
struct Compute<__nv_bfloat16> {
    using type = float;
};

template<typename T>
__device__ inline typename Compute<T>::type toCompute(T v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return v;
    }
}

template<typename T>
__device__ inline T fromCompute(typename Compute<T>::type v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return T(v);
    }
}

template<typename C>
__device__ inline C apply(C a, C b, int op) {
    switch (op) {
        case NCCL_DEVICE_PROD: return a * b;
        case NCCL_DEVICE_MAX: return a > b ? a : b;
        case NCCL_DEVICE_MIN: return a < b ? a : b;
        default: return a + b;
    }
}

template<typename T>
__device__ inline T combineOne(const NcclCombine &c, const T *a, const T *b, size_t i) {
    using C = typename Compute<T>::type;
    C v;
    if (a != nullptr) {
        v = toCompute(a[i]);
        if (c.scaled) v = C(double(v) * c.scale);
        if (b != nullptr) v = apply(v, toCompute(b[i]), c.op);
    } else {
        v = toCompute(b[i]);
    }
    if (c.divide > 1) v = C(v / C(c.divide));
    return fromCompute<T>(v);
}

// per thread: VEC elements (16 bytes) at a time while all pointers are 16 byte aligned, the rest one by one
template<typename T>
__global__ void combineKernel(NcclCombine c) {
    const T *a = static_cast<const T *>(c.a), *b = static_cast<const T *>(c.b);
    T *o1 = static_cast<T *>(c.out1), *o2 = static_cast<T *>(c.out2);
    constexpr int VEC = 16 / sizeof(T);
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t first = 0;
    auto aligned = [](const void *p) { return p == nullptr || reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    if (VEC > 1 && aligned(a) && aligned(b) && aligned(o1) && aligned(o2)) {
        const size_t vectors = c.count / VEC;
        for (size_t v = blockIdx.x * size_t(blockDim.x) + threadIdx.x; v < vectors; v += stride) {
            T in_a[VEC], in_b[VEC], out[VEC];
            if (a != nullptr) *reinterpret_cast<uint4 *>(in_a) = reinterpret_cast<const uint4 *>(a)[v];
            if (b != nullptr) *reinterpret_cast<uint4 *>(in_b) = reinterpret_cast<const uint4 *>(b)[v];
#pragma unroll
            for (int k = 0; k < VEC; k++) {
                out[k] = combineOne(c, a != nullptr ? in_a : nullptr, b != nullptr ? in_b : nullptr, k);
            }
            if (o1 != nullptr) reinterpret_cast<uint4 *>(o1)[v] = *reinterpret_cast<uint4 *>(out);
            if (o2 != nullptr) reinterpret_cast<uint4 *>(o2)[v] = *reinterpret_cast<uint4 *>(out);
        }
        first = vectors * VEC;
    }
    for (size_t i = first + blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < c.count; i += stride) {
        const T v = combineOne(c, a, b, i);
        if (o1 != nullptr) o1[i] = v;
        if (o2 != nullptr) o2[i] = v;
    }
    // the writes (into another GPU's memory, too) are performed before the stream announces the chunk: the release
    // that follows the kernel only waits for it to complete
    __threadfence_system();
}

template<typename T>
cudaError_t launch(const NcclCombine &c, cudaStream_t stream) {
    constexpr int VEC = 16 / sizeof(T);
    const size_t work = (c.count + VEC - 1) / VEC;
    const size_t cap = c.blocks > 0 ? size_t(c.blocks) : 128;
    const unsigned blocks = unsigned(std::max<size_t>(1, std::min<size_t>((work + 255) / 256, cap)));
    combineKernel<T><<<blocks, 256, 0, stream>>>(c);
    return cudaGetLastError();
}

} // namespace

cudaError_t ncclDeviceCombine(const NcclCombine &c, cudaStream_t stream) {
    if (c.count == 0) return cudaSuccess;
    switch (c.type) {
        case ncclInt8: return launch<int8_t>(c, stream);
        case ncclUint8: return launch<uint8_t>(c, stream);
        case ncclInt32: return launch<int32_t>(c, stream);
        case ncclUint32: return launch<uint32_t>(c, stream);
        case ncclInt64: return launch<int64_t>(c, stream);
        case ncclUint64: return launch<uint64_t>(c, stream);
        case ncclFloat16: return launch<__half>(c, stream);
        case ncclBfloat16: return launch<__nv_bfloat16>(c, stream);
        case ncclFloat32: return launch<float>(c, stream);
        case ncclFloat64: return launch<double>(c, stream);
        default: return cudaErrorInvalidValue;
    }
}
