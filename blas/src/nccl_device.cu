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

__device__ inline uint64_t loadAcquire(const uint64_t *p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// spins until *p >= value, polling gently (system scope loads in a tight loop slow down the GPU's other traffic)
__device__ inline void waitAtLeast(const uint64_t *p, uint64_t value) {
    while (loadAcquire(p) < value) {
        __nanosleep(200);
    }
}

__device__ inline void storeRelease(uint64_t *p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
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
    if (c.wait[0] != nullptr || c.wait[1] != nullptr) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 2; i++) {
                if (c.wait[i] != nullptr) waitAtLeast(c.wait[i], c.waitValue[i]);
            }
        }
        __syncthreads();
    }
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
    if (c.finished != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0 && atomicAdd(c.finished, 1u) == gridDim.x - 1) {
            *c.finished = 0;
            // (every block passed its wait; the peers set the flags again only after the signals)
            for (int i = 0; i < 2 && c.toggle; i++) {
                if (c.wait[i] != nullptr) *const_cast<uint64_t *>(c.wait[i]) = 0;
            }
            for (int i = 0; i < 2; i++) {
                if (c.signal[i] != nullptr) storeRelease(c.signal[i], c.signalValue[i]);
            }
        }
    }
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

// the last block to arrive (counted in *counter, which it resets) gets true, after every block's writes
__device__ bool lastBlock(unsigned *counter) {
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
        if (last) *counter = 0;
    }
    __syncthreads();
    return last;
}

template<typename T>
__global__ void exchangeKernel(NcclExchange x) {
    // the peer consumed what this slot held
    if (threadIdx.x == 0 && x.freed != nullptr) waitAtLeast(x.freed, x.freedValue);
    __syncthreads();
    const size_t stride = size_t(gridDim.x) * blockDim.x, first = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    const size_t bytes = x.count * sizeof(T);
    if (reinterpret_cast<uintptr_t>(x.mine) % 16 == 0 && reinterpret_cast<uintptr_t>(x.peerInbox) % 16 == 0) {
        for (size_t i = first; i < bytes / 16; i += stride) {
            reinterpret_cast<uint4 *>(x.peerInbox)[i] = reinterpret_cast<const uint4 *>(x.mine)[i];
        }
        for (size_t i = bytes / 16 * 16 + first; i < bytes; i += stride) {
            static_cast<char *>(x.peerInbox)[i] = static_cast<const char *>(x.mine)[i];
        }
    } else {
        for (size_t i = first; i < bytes; i += stride) {
            static_cast<char *>(x.peerInbox)[i] = static_cast<const char *>(x.mine)[i];
        }
    }
    if (lastBlock(x.counters) && threadIdx.x == 0) {
        // (every block waited for the slot; the peer frees it again only after this chunk)
        if (x.toggle && x.freed != nullptr) *const_cast<uint64_t *>(x.freed) = 0;
        storeRelease(x.peerReady, x.sendSeq);
        storeRelease(x.sent, x.sendSeq);
    }
    if (threadIdx.x == 0) {
        waitAtLeast(x.ready, x.recvSeq);
        waitAtLeast(x.sent, x.sendSeq);
    }
    __syncthreads();
    NcclCombine c{};
    c.op = x.op;
    c.count = x.count;
    c.divide = x.divide;
    const T *a = static_cast<const T *>(x.mine), *b = static_cast<const T *>(x.inbox);
    T *out = static_cast<T *>(x.out);
    for (size_t i = first; i < x.count; i += stride) out[i] = combineOne(c, a, b, i);
    if (lastBlock(x.counters + 1) && threadIdx.x == 0) {
        if (x.toggle) {
            *const_cast<uint64_t *>(x.ready) = 0;
            *x.sent = 0;
        }
        storeRelease(x.peerFreed, x.recvSeq);
    }
}

template<typename T>
cudaError_t launchExchange(const NcclExchange &x, cudaStream_t stream) {
    const size_t work = (x.count * sizeof(T) + 15) / 16;
    const unsigned blocks = unsigned(std::max<size_t>(1, std::min<size_t>((work + 255) / 256, 16)));
    exchangeKernel<T><<<blocks, 256, 0, stream>>>(x);
    return cudaGetLastError();
}

} // namespace

cudaError_t ncclDeviceExchange(const NcclExchange &x, cudaStream_t stream) {
    switch (x.type) {
        case ncclInt8: return launchExchange<int8_t>(x, stream);
        case ncclUint8: return launchExchange<uint8_t>(x, stream);
        case ncclInt32: return launchExchange<int32_t>(x, stream);
        case ncclUint32: return launchExchange<uint32_t>(x, stream);
        case ncclInt64: return launchExchange<int64_t>(x, stream);
        case ncclUint64: return launchExchange<uint64_t>(x, stream);
        case ncclFloat16: return launchExchange<__half>(x, stream);
        case ncclBfloat16: return launchExchange<__nv_bfloat16>(x, stream);
        case ncclFloat32: return launchExchange<float>(x, stream);
        case ncclFloat64: return launchExchange<double>(x, stream);
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t ncclDeviceCombine(const NcclCombine &c, cudaStream_t stream) {
    if (c.count == 0 && c.wait[0] == nullptr && c.wait[1] == nullptr && c.finished == nullptr) return cudaSuccess;
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
