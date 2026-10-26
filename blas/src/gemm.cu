/**
 * The GEMM engine of LibreCUDA's cuBLAS / cuBLASLt: CUTLASS kernels for the common cases (16 bit operands on tensor
 * cores, float and double), a generic kernel for everything else (complex numbers, conjugation, unaligned or strided
 * operands, scalars of other types), and the cuBLASLt epilogues (bias, ReLU, GELU): in the CUTLASS kernels' epilogue
 * where they can, else as a pass over the result.
 */
#include "blas_internal.h"
#include "cutlass_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#include <algorithm>
#include <type_traits>
#include <cstring>
#include <map>
#include <tuple>
#include <mutex>
#include <unordered_map>

namespace librecuda_blas {
namespace {

template<typename R>
struct Cplx {
    R re, im;
};

// ---- arithmetic of the accumulator types (float, double, Cplx<float>, Cplx<double>; int for 8 bit integers, wrapping
// around like int32 accumulation)

template<typename A>
__host__ __device__ A zero() { return A{}; }

__host__ __device__ inline float mul(float a, float b) { return a * b; }
__host__ __device__ inline int mul(int a, int b) { return int(unsigned(a) * unsigned(b)); }
__host__ __device__ inline double mul(double a, double b) { return a * b; }
template<typename R>
__host__ __device__ Cplx<R> mul(Cplx<R> a, Cplx<R> b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

__host__ __device__ inline float add(float a, float b) { return a + b; }
__host__ __device__ inline int add(int a, int b) { return int(unsigned(a) + unsigned(b)); }
__host__ __device__ inline double add(double a, double b) { return a + b; }
template<typename R>
__host__ __device__ Cplx<R> add(Cplx<R> a, Cplx<R> b) { return {a.re + b.re, a.im + b.im}; }

__host__ __device__ inline float conjugate(float a) { return a; }
__host__ __device__ inline int conjugate(int a) { return a; }
__host__ __device__ inline double conjugate(double a) { return a; }
template<typename R>
__host__ __device__ Cplx<R> conjugate(Cplx<R> a) { return {a.re, -a.im}; }

__host__ __device__ inline bool isZero(float a) { return a == 0; }
__host__ __device__ inline bool isZero(int a) { return a == 0; }
__host__ __device__ inline bool isZero(double a) { return a == 0; }
template<typename R>
__host__ __device__ bool isZero(Cplx<R> a) { return a.re == 0 && a.im == 0; }

// ---- conversions between element types and accumulator types

template<typename A>
__host__ __device__ A fromComplex(double re, double im);
template<> __host__ __device__ inline float fromComplex<float>(double re, double) { return float(re); }
template<> __host__ __device__ inline double fromComplex<double>(double re, double) { return re; }
template<> __host__ __device__ inline int fromComplex<int>(double re, double) { return int(re); }
template<> __host__ __device__ inline Cplx<float> fromComplex<Cplx<float>>(double re, double im) {
    return {float(re), float(im)};
}
template<> __host__ __device__ inline Cplx<double> fromComplex<Cplx<double>>(double re, double im) { return {re, im}; }

// an element of the given type (scalars of any type cuBLAS allows)
template<typename A>
__host__ __device__ A loadAny(const void *ptr, cudaDataType type) {
    switch (type) {
        case CUDA_R_16F:
            return fromComplex<A>(__half2float(*static_cast<const __half *>(ptr)), 0);
        case CUDA_R_16BF:
            return fromComplex<A>(__bfloat162float(*static_cast<const __nv_bfloat16 *>(ptr)), 0);
        case CUDA_R_32F:
            return fromComplex<A>(*static_cast<const float *>(ptr), 0);
        case CUDA_R_64F:
            return fromComplex<A>(*static_cast<const double *>(ptr), 0);
        case CUDA_R_32I:
            return fromComplex<A>(*static_cast<const int *>(ptr), 0);
        case CUDA_R_8I:
            return fromComplex<A>(*static_cast<const int8_t *>(ptr), 0);
        case CUDA_C_32F: {
            const auto *c = static_cast<const float *>(ptr);
            return fromComplex<A>(c[0], c[1]);
        }
        case CUDA_C_64F: {
            const auto *c = static_cast<const double *>(ptr);
            return fromComplex<A>(c[0], c[1]);
        }
        default:
            return A{};
    }
}

template<typename T, typename A>
__device__ A load(const T *ptr) {
    if constexpr (std::is_same_v<T, cuComplex>) {
        return fromComplex<A>(ptr->x, ptr->y);
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return fromComplex<A>(ptr->x, ptr->y);
    } else if constexpr (std::is_same_v<T, __half>) {
        return fromComplex<A>(__half2float(*ptr), 0);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return fromComplex<A>(__bfloat162float(*ptr), 0);
    } else {
        return fromComplex<A>(*ptr, 0);
    }
}

template<typename T, typename A>
__device__ void store(T *ptr, A value) {
    if constexpr (std::is_same_v<A, Cplx<float>> || std::is_same_v<A, Cplx<double>>) {
        if constexpr (std::is_same_v<T, cuComplex>) {
            *ptr = make_cuComplex(float(value.re), float(value.im));
        } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            *ptr = make_cuDoubleComplex(value.re, value.im);
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        *ptr = __float2half(float(value));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        *ptr = __float2bfloat16(float(value));
    } else {
        *ptr = T(value);
    }
}

// a scalar: a value, or (on the device) read from ptr when the kernel starts
template<typename A>
struct KernelScalar {
    A value;
    const void *ptr;
    cudaDataType type;

    __device__ A get() const { return ptr != nullptr ? loadAny<A>(ptr, type) : value; }
};

template<typename A>
KernelScalar<A> kernelScalar(const Scalar &s) {
    if (s.onDevice) {
        return {A{}, s.ptr, s.type};
    }
    return {loadAny<A>(s.ptr, s.type), nullptr, s.type};
}

// ---- the generic kernel: 16 x 16 output tiles, operands staged through shared memory

constexpr int TILE = 16;

template<typename T>
__device__ const T *batchPointer(const Operand &o, long long batch) {
    if (o.array != nullptr) {
        return static_cast<const T *>(o.array[batch]);
    }
    return static_cast<const T *>(o.ptr) + batch * o.stride;
}

// element (row, col) of op(X) where X is stored column-major (rows of op(X) are X's columns when transposed)
template<typename T, typename A>
__device__ A opElement(const T *base, const Operand &o, cublasOperation_t op, long long row, long long col) {
    if (op == CUBLAS_OP_N) {
        return load<T, A>(base + row * o.inc + col * o.ld);
    }
    const A value = load<T, A>(base + col * o.inc + row * o.ld);
    return op == CUBLAS_OP_C ? conjugate(value) : value;
}

template<typename TA, typename TB, typename TD, typename A>
__global__ void gemmGeneric(GemmParams p, KernelScalar<A> alpha_arg, KernelScalar<A> beta_arg, int colOffset) {
    __shared__ A tile_a[TILE][TILE + 1];
    __shared__ A tile_b[TILE][TILE + 1];
    const A alpha = alpha_arg.get(), beta = beta_arg.get();
    const int tx = threadIdx.x, ty = threadIdx.y;
    const long long row = static_cast<long long>(blockIdx.x) * TILE + tx;
    const long long col = static_cast<long long>(blockIdx.y) * TILE + ty + colOffset;
    const long long tile_col = static_cast<long long>(blockIdx.y) * TILE + colOffset;
    for (long long batch = blockIdx.z; batch < p.batch; batch += gridDim.z) {
        const TA *a = batchPointer<TA>(p.a, batch);
        const TB *b = batchPointer<TB>(p.b, batch);
        A acc = zero<A>();
        for (int k0 = 0; k0 < p.k; k0 += TILE) {
            // thread (tx, ty) stages op(A)(row, k0 + ty) and op(B)(k0 + tx, tile_col + ty)
            tile_a[ty][tx] = row < p.m && k0 + ty < p.k ? opElement<TA, A>(a, p.a, p.transa, row, k0 + ty) : zero<A>();
            tile_b[ty][tx] = k0 + tx < p.k && tile_col + ty < p.n
                             ? opElement<TB, A>(b, p.b, p.transb, k0 + tx, tile_col + ty) : zero<A>();
            __syncthreads();
            for (int l = 0; l < TILE; l++) {
                acc = add(acc, mul(tile_a[l][tx], tile_b[ty][l]));
            }
            __syncthreads();
        }
        if (row < p.m && col < p.n) {
            A result = mul(alpha, acc);
            // with beta 0, C isn't read (it may hold anything)
            if (!isZero(beta)) {
                const TD *c = batchPointer<TD>(p.c, batch);
                result = add(result, mul(beta, load<TD, A>(c + row * p.c.inc + col * p.c.ld)));
            }
            TD *d = const_cast<TD *>(batchPointer<TD>(p.d, batch));
            store(d + row * p.d.inc + col * p.d.ld, result);
        }
    }
}

template<typename TA, typename TB, typename TD, typename A>
cublasStatus_t launchGeneric(const GemmParams &p) {
    const KernelScalar<A> alpha = kernelScalar<A>(p.alpha), beta = kernelScalar<A>(p.beta);
    const int row_tiles = (p.m + TILE - 1) / TILE;
    // the grid's y dimension holds at most 65535 column tiles per launch
    for (long long first = 0; first < p.n; first += 65535LL * TILE) {
        const int col_tiles = static_cast<int>((std::min<long long>(p.n - first, 65535LL * TILE) + TILE - 1) / TILE);
        dim3 grid(row_tiles, col_tiles, std::min(p.batch, 65535));
        gemmGeneric<TA, TB, TD, A><<<grid, dim3(TILE, TILE), 0, p.stream>>>(p, alpha, beta, static_cast<int>(first));
        if (cudaGetLastError() != cudaSuccess) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t generic(const GemmParams &p) {
    const cudaDataType ta = p.a.type, tb = p.b.type, td = p.d.type;
    if (ta != tb) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    switch (ta) {
        case CUDA_R_16F:
            if (td == CUDA_R_16F) return launchGeneric<__half, __half, __half, float>(p);
            if (td == CUDA_R_32F) return launchGeneric<__half, __half, float, float>(p);
            break;
        case CUDA_R_16BF:
            if (td == CUDA_R_16BF) return launchGeneric<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float>(p);
            if (td == CUDA_R_32F) return launchGeneric<__nv_bfloat16, __nv_bfloat16, float, float>(p);
            break;
        case CUDA_R_32F:
            if (td == CUDA_R_32F) return launchGeneric<float, float, float, float>(p);
            break;
        case CUDA_R_64F:
            if (td == CUDA_R_64F) return launchGeneric<double, double, double, double>(p);
            break;
        case CUDA_R_8I:
            if (td == CUDA_R_32I) return launchGeneric<int8_t, int8_t, int32_t, int>(p);
            break;
        case CUDA_C_32F:
            if (td == CUDA_C_32F) return launchGeneric<cuComplex, cuComplex, cuComplex, Cplx<float>>(p);
            break;
        case CUDA_C_64F:
            if (td == CUDA_C_64F) {
                return launchGeneric<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, Cplx<double>>(p);
            }
            break;
        default:
            break;
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// ---- CUTLASS kernels

size_t elementSize(cudaDataType type) {
    switch (type) {
        case CUDA_R_16F:
        case CUDA_R_16BF:
            return 2;
        case CUDA_R_32F:
            return 4;
        case CUDA_R_64F:
        case CUDA_C_32F:
            return 8;
        case CUDA_C_64F:
            return 16;
        default:
            return 1;
    }
}

// the scalars as the CUTLASS kernel's compute type; false if they're device scalars of another type
template<typename Compute>
bool cutlassScalars(const GemmParams &p, CutlassScalars<Compute> &s, cudaDataType computeType) {
    if (p.alpha.onDevice || p.beta.onDevice) {
        if (!p.alpha.onDevice || !p.beta.onDevice || p.alpha.type != computeType || p.beta.type != computeType) {
            return false;
        }
        s = {Compute{}, Compute{}, static_cast<const Compute *>(p.alpha.ptr), static_cast<const Compute *>(p.beta.ptr)};
        return true;
    }
    s = {loadAny<Compute>(p.alpha.ptr, p.alpha.type), loadAny<Compute>(p.beta.ptr, p.beta.type), nullptr, nullptr};
    return true;
}

bool operandsAligned(const GemmParams &p, int alignment, int outputAlignment) {
    const bool c_ok = (p.c.ptr == nullptr && p.c.array == nullptr) ||
                      aligned(p.c, elementSize(p.c.type), outputAlignment);
    return aligned(p.a, elementSize(p.a.type), alignment) && aligned(p.b, elementSize(p.b.type), alignment) &&
           aligned(p.d, elementSize(p.d.type), outputAlignment) && c_ok;
}

int multiprocessors() {
    static std::mutex mutex;
    static std::unordered_map<int, int> counts;
    int device = 0;
    cudaGetDevice(&device);
    std::lock_guard lock(mutex);
    int &count = counts[device];
    if (count == 0 && cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError();
        count = 1;
    }
    return std::max(count, 1);
}

// the L2 cache size of the current device in bytes
long long l2Bytes() {
    static std::mutex mutex;
    static std::unordered_map<int, int> sizes;
    int device = 0;
    cudaGetDevice(&device);
    std::lock_guard lock(mutex);
    int &size = sizes[device];
    if (size == 0 && cudaDeviceGetAttribute(&size, cudaDevAttrL2CacheSize, device) != cudaSuccess) {
        cudaGetLastError();
        size = 1;
    }
    return std::max(size, 1);
}

// 64 x 64 tiles (when k fills their deeper pipeline) unless 128 x 128 tiles fill the GPU 4 times or more and the k x n
// operand outgrows L2: the CTAs running at once cover the matrix's rows for a few columns, so each column of 64 x 64
// tiles reads that operand again, from memory once it no longer fits (RTX 5060, 24 MB of L2: 2560 x 4096 x 4096 bf16
// in 3.33 ms against 2.16 with 128 x 128 tiles; at k = 2560, 20 MB, the small tiles are still faster). Otherwise the
// smaller tiles keep more SMs busy and run within about 1% of cuBLAS (llama.cpp's 512 token prompts: 4096 x 512 x 2560
// in 277 us against 313 with 128 x 128 tiles).
bool smallTiles(const GemmParams &p) {
    if (p.k < 256) {
        return false;
    }
    const long long tiles = ((p.m + 127LL) / 128) * ((p.n + 127LL) / 128) * p.batch;
    const long long operand_bytes = static_cast<long long>(elementSize(p.b.type)) * p.n * p.k;
    return tiles < 4LL * multiprocessors() || operand_bytes <= l2Bytes();
}

// ---- matrix times a few vectors (decoding: one token per sequence): 16 bit A transposed, so that each output's weights
// are contiguous (a linear layer's x W^T), B's columns contiguous, at most 4 of them (from 8 on the tensor core kernels
// are faster). Memory bound: each warp computes GEMV_ROWS outputs per column with 16 byte loads of A (streamed past L1)
// and B, several in flight per lane, and sums across its lanes; small blocks of few rows spread even the smaller layers
// over all SMs. The tensor core kernels' tiles waste most of their width on so few columns: a Qwen3-4B decode step's
// GEMVs (tensor parallelism 2) read their weights at 401 GB/s on an RTX 5060 here, 331 GB/s with the tensor core
// kernels (cuBLAS: 378).

constexpr int GEMV_ROWS = 2, GEMV_WARPS = 4, GEMV_MAX_COLUMNS = 4;

template<typename T>
__device__ inline void accumulate8(float &acc, uint4 a, uint4 b) {
    const auto *ha = reinterpret_cast<const typename std::conditional<std::is_same<T, __half>::value, __half2,
                                                                      __nv_bfloat162>::type *>(&a);
    const auto *hb = reinterpret_cast<decltype(ha)>(&b);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 fa, fb;
        if constexpr (std::is_same<T, __half>::value) {
            fa = __half22float2(ha[i]);
            fb = __half22float2(hb[i]);
        } else {
            fa = __bfloat1622float2(ha[i]);
            fb = __bfloat1622float2(hb[i]);
        }
        acc = fmaf(fa.x, fb.x, acc);
        acc = fmaf(fa.y, fb.y, acc);
    }
}

template<typename TD>
__device__ inline float loadOut(const TD *p) {
    if constexpr (std::is_same<TD, float>::value) {
        return *p;
    } else if constexpr (std::is_same<TD, __half>::value) {
        return __half2float(*p);
    } else {
        return __bfloat162float(*p);
    }
}

template<typename TD>
__device__ inline TD storeOut(float v) {
    if constexpr (std::is_same<TD, float>::value) {
        return v;
    } else if constexpr (std::is_same<TD, __half>::value) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

template<typename T, typename TD>
__global__ void __launch_bounds__(GEMV_WARPS * 32) gemvKernel(int m, int n, int k, const T *A, long long lda,
                                                              long long strideA, const T *B, long long ldb,
                                                              long long strideB, const TD *C, long long ldc,
                                                              long long strideC, TD *D, long long ldd, long long strideD,
                                                              CutlassScalars<float> s) {
    const int lane = threadIdx.x % 32;
    const long long row0 = (blockIdx.x * (long long) GEMV_WARPS + threadIdx.x / 32) * GEMV_ROWS;
    if (row0 >= m) {
        return;
    }
    const long long batch = blockIdx.y;
    A += batch * strideA;
    B += batch * strideB;
    D += batch * strideD;
    if (C != nullptr) C += batch * strideC;
    const int vectors = k / 8;
    const uint4 *a[GEMV_ROWS];
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS; r++) {
        // (rows past the end read the last row; their sums aren't written)
        a[r] = reinterpret_cast<const uint4 *>(A + min(row0 + r, (long long) m - 1) * lda);
    }
    float acc[GEMV_ROWS][GEMV_MAX_COLUMNS] = {};
    #pragma unroll 2
    for (int v = lane; v < vectors; v += 32) {
        uint4 av[GEMV_ROWS];
        #pragma unroll
        for (int r = 0; r < GEMV_ROWS; r++) {
            av[r] = __ldcs(a[r] + v);
        }
        #pragma unroll
        for (int j = 0; j < GEMV_MAX_COLUMNS; j++) {
            if (j < n) {
                const uint4 bv = __ldg(reinterpret_cast<const uint4 *>(B + j * ldb) + v);
                #pragma unroll
                for (int r = 0; r < GEMV_ROWS; r++) {
                    accumulate8<T>(acc[r][j], av[r], bv);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS; r++) {
        #pragma unroll
        for (int j = 0; j < GEMV_MAX_COLUMNS; j++) {
            if (j < n) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], offset);
                }
            }
        }
    }
    if (lane != 0) {
        return;
    }
    const float alpha = s.alphaPtr != nullptr ? *s.alphaPtr : s.alpha;
    const float beta = s.betaPtr != nullptr ? *s.betaPtr : s.beta;
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS; r++) {
        const long long i = row0 + r;
        if (i >= m) {
            break;
        }
        for (int j = 0; j < n; j++) {
            float v = alpha * acc[r][j];
            if (beta != 0.0f && C != nullptr) {
                v += beta * loadOut(C + i + j * ldc);
            }
            D[i + j * ldd] = storeOut<TD>(v);
        }
    }
}

template<typename T, typename TD>
cublasStatus_t launchGemv(const GemmParams &p, const CutlassScalars<float> &s) {
    const long long rowsPerBlock = (long long) GEMV_WARPS * GEMV_ROWS;
    const dim3 grid(unsigned((p.m + rowsPerBlock - 1) / rowsPerBlock), unsigned(p.batch));
    gemvKernel<T, TD><<<grid, GEMV_WARPS * 32, 0, p.stream>>>(
            p.m, p.n, p.k, static_cast<const T *>(p.a.ptr), p.a.ld, p.a.stride, static_cast<const T *>(p.b.ptr), p.b.ld,
            p.b.stride, static_cast<const TD *>(p.c.ptr), p.c.ld, p.c.stride, static_cast<TD *>(const_cast<void *>(p.d.ptr)), p.d.ld,
            p.d.stride, s);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t runGemv(const GemmParams &p) {
    const bool sixteen = p.a.type == CUDA_R_16F || p.a.type == CUDA_R_16BF;
    if (!sixteen || p.b.type != p.a.type || (p.d.type != p.a.type && p.d.type != CUDA_R_32F) || p.n > GEMV_MAX_COLUMNS ||
        p.k % 8 != 0 || p.transa == CUBLAS_OP_N || p.transb != CUBLAS_OP_N || p.a.array != nullptr ||
        p.b.array != nullptr || p.d.array != nullptr || p.c.array != nullptr || p.batch > 65535) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if ((p.c.ptr != nullptr && p.c.type != p.d.type) || p.a.inc != 1 || p.b.inc != 1 || p.d.inc != 1 ||
        !aligned(p.a, 2, 8) || !aligned(p.b, 2, 8)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    CutlassScalars<float> s{};
    if (!cutlassScalars(p, s, CUDA_R_32F)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (p.a.type == CUDA_R_16F) {
        return p.d.type == CUDA_R_32F ? launchGemv<__half, float>(p, s) : launchGemv<__half, __half>(p, s);
    }
    return p.d.type == CUDA_R_32F ? launchGemv<__nv_bfloat16, float>(p, s)
                                  : launchGemv<__nv_bfloat16, __nv_bfloat16>(p, s);
}

bool isFp8(cudaDataType type) {
    return type == CUDA_R_8F_E4M3 || type == CUDA_R_8F_E5M2;
}

void *scratch(cudaStream_t stream, size_t bytes, int slot);

// dst (rows x cols, column-major) = src, or the transpose of src (cols x rows); bytes, batch entries
__global__ void copyBytesKernel(uint8_t *dst, long long dstLd, long long dstStride, const uint8_t *src, long long srcLd,
                                long long srcStride, long long rows, long long cols, bool transpose, int batch) {
    const long long total = rows * cols;
    for (long long b = blockIdx.y; b < batch; b += gridDim.y) {
        for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
             i += static_cast<long long>(gridDim.x) * blockDim.x) {
            const long long row = i % rows, col = i / rows;
            dst[b * dstStride + col * dstLd + row] = transpose ? src[b * srcStride + row * srcLd + col]
                                                               : src[b * srcStride + col * srcLd + row];
        }
    }
}

/**
 * An 8 bit operand as the kernels take it: k rows (A: op T, B: op N), 16 byte aligned. Otherwise (another
 * transposition, misaligned) copied to scratch memory (at offset in the stream's buffer) in that layout. cols: m or n.
 */
cublasStatus_t fp8Operand(const GemmParams &p, const Operand &o, bool transposed, long long cols, char *buffer,
                          Operand &result) {
    result = o;
    if (!transposed && aligned(o, 1, 16)) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (o.array != nullptr || o.inc != 1) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const long long ld = (p.k + 15LL) / 16 * 16;
    // an operand shared by the batch entries (stride 0) is copied once
    const int batch = o.stride == 0 ? 1 : p.batch;
    result = Operand{buffer, nullptr, o.type, ld, o.stride == 0 ? 0 : ld * cols, 1};
    const long long total = p.k * cols;
    const dim3 grid(static_cast<unsigned int>(std::min<long long>((total + 255) / 256, 8192)), std::min(batch, 65535));
    copyBytesKernel<<<grid, 256, 0, p.stream>>>(reinterpret_cast<uint8_t *>(buffer), ld, result.stride,
                                                static_cast<const uint8_t *>(o.ptr), o.ld, o.stride, p.k, cols,
                                                transposed, batch);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

/**
 * 8 bit float operands (cuBLASLt): only the CUTLASS kernels, which apply cuBLASLt's scales and compute the absolute
 * maximum in their epilogue, and add the bias there as C with leading dimension 0 (fused) when C itself isn't read.
 * Other epilogues run as a separate pass when no scale of D or amax has to follow them.
 */
cublasStatus_t runFp8(const GemmParams &p, bool *fused) {
    if (!isFp8(p.a.type) || !isFp8(p.b.type) || p.k == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    CutlassScalars<float> s{};
    if (!cutlassScalars(p, s, CUDA_R_32F)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool fp8_output = isFp8(p.d.type);
    Fp8Epilogue e{s.alpha, s.beta, s.alphaPtr, s.betaPtr, p.scaleA, p.scaleB, p.scaleC, p.scaleD, p.amax};
    GemmParams q = p;
    if (s.betaPtr == nullptr && s.beta == 0.0f) {
        // C isn't read: callers pass any pointer and type for it (PyTorch: A's data, the bias' type)
        q.c = Operand{nullptr, nullptr, p.d.type, p.d.ld, p.d.stride, 1};
    } else if (p.c.type != p.d.type) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    bool fuse = false;
    if (p.bias != nullptr || p.activation != Activation::NONE) {
        fuse = p.activation == Activation::NONE && fused != nullptr && p.biasType == p.d.type && q.c.ptr == nullptr &&
               reinterpret_cast<uintptr_t>(p.bias) % 16 == 0 &&
               p.biasStride % fp8OutputAlignment(elementSize(p.d.type)) == 0;
        if (fuse) {
            q.c = Operand{p.bias, nullptr, p.d.type, 0, p.biasStride, 1};
            e.beta = 1.0f;
            e.betaPtr = nullptr;
            e.scaleC = nullptr;
        } else if (fp8_output || e.scaleD != nullptr || e.amax != nullptr || fused == nullptr) {
            // the separate pass would come after D's scale and amax
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    if (!operandsAligned(q, 1, fp8OutputAlignment(elementSize(p.d.type)))) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    // A as op T and B as op N (A op N / B op T are the transposes of what the kernels take)
    const long long ld = (p.k + 15LL) / 16 * 16;
    const size_t a_bytes = (size_t(ld) * p.m * p.batch + 255) / 256 * 256, b_bytes = size_t(ld) * p.n * p.batch;
    const bool copy_a = p.transa == CUBLAS_OP_N || !aligned(p.a, 1, 16);
    const bool copy_b = p.transb != CUBLAS_OP_N || !aligned(p.b, 1, 16);
    char *buffer = nullptr;
    if (copy_a || copy_b) {
        buffer = static_cast<char *>(scratch(p.stream, a_bytes + b_bytes, 2));
        if (buffer == nullptr) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    cublasStatus_t status = fp8Operand(p, p.a, p.transa == CUBLAS_OP_N, p.m, buffer, q.a);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = fp8Operand(p, p.b, p.transb != CUBLAS_OP_N, p.n, buffer + a_bytes, q.b);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    q.transa = CUBLAS_OP_T;
    q.transb = CUBLAS_OP_N;
    if (p.amax != nullptr && cudaMemsetAsync(p.amax, 0, sizeof(float), p.stream) != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    if (fuse) {
        *fused = true;
    }
    // 64 x 64 tiles only when 128 x 128 ones wouldn't fill the GPU twice: the large tiles, at twice the bf16 rate, run
    // faster even where B fits in L2 (4096^3 at 141 against 110 TFLOPS on an RTX 5060)
    const bool small = q.k >= 256 && ((q.m + 127LL) / 128) * ((q.n + 127LL) / 128) * q.batch < 2LL * multiprocessors();
    if (p.a.type == CUDA_R_8F_E4M3) {
        return p.b.type == CUDA_R_8F_E4M3 ? cutlassE4M3E4M3(q, e, small) : cutlassE4M3E5M2(q, e, small);
    }
    return p.b.type == CUDA_R_8F_E4M3 ? cutlassE5M2E4M3(q, e, small) : cutlassE5M2E5M2(q, e, small);
}

/**
 * The CUTLASS kernels; fused: whether they also applied the cuBLASLt epilogue (16 bit operands and output, wide
 * accesses; the bias as C with leading dimension 0, when the GEMM itself doesn't read C)
 */
cublasStatus_t runFast(const GemmParams &p, bool *fused = nullptr) {
    if (isFp8(p.a.type) || isFp8(p.b.type)) {
        return runFp8(p, fused);
    }
    // CUTLASS handles transposition (as a row-major layout) but not conjugation; real operands have nothing to conjugate
    if (p.a.type != p.b.type || p.k == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    GemmParams q = p;
    if (q.transa == CUBLAS_OP_C) q.transa = CUBLAS_OP_T;
    if (q.transb == CUBLAS_OP_C) q.transb = CUBLAS_OP_T;
    // array batches with an operand without array (or the reverse) aren't supported
    if ((p.a.array != nullptr) != (p.b.array != nullptr) || (p.a.array != nullptr) != (p.d.array != nullptr)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    switch (p.a.type) {
        case CUDA_R_16F:
        case CUDA_R_16BF: {
            const bool output_float = p.d.type == CUDA_R_32F;
            if (p.d.type != p.a.type && !output_float) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            CutlassScalars<float> s{};
            if (!cutlassScalars(p, s, CUDA_R_32F)) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            const int out_wide = output_float ? 4 : 8, out_narrow = output_float ? 1 : 2;
            const bool wide = operandsAligned(p, 8, out_wide);
            if (!wide && !operandsAligned(p, 2, out_narrow)) {
                /**
                 * Elements one at a time run at about half the speed of aligned tiles: large problems are faster
                 * copied into aligned ones (repacked), except with few rows or columns, where the copy of the other,
                 * large operand dominates (copying A for an 8 column problem took 3 times as long as computing it
                 * unaligned). Small problems (Swin's 49 x 49 window attention, batches of thousands) are as fast
                 * unaligned as on NVIDIA's cuBLAS, and 10 times faster than on the generic kernel.
                 */
                if (!operandsAligned(p, 1, 1) ||
                    (double(p.m) * p.n * p.k >= double(1 << 21) && p.m >= 64 && p.n >= 64)) {
                    return CUBLAS_STATUS_NOT_SUPPORTED;
                }
                return p.a.type == CUDA_R_16F ? cutlassHalfUnaligned(q, s, output_float)
                                              : cutlassBFloat16Unaligned(q, s, output_float);
            }
            Activation activation = Activation::NONE;
            if (fused != nullptr && wide && !output_float && (p.bias != nullptr || p.activation != Activation::NONE)) {
                const bool bias_fits = p.bias == nullptr ||
                                       (p.biasType == p.d.type && s.betaPtr == nullptr && s.beta == 0.0f &&
                                        p.a.array == nullptr && reinterpret_cast<uintptr_t>(p.bias) % 16 == 0 &&
                                        p.biasStride % 8 == 0);
                if (bias_fits) {
                    if (p.bias != nullptr) {
                        q.c = Operand{p.bias, nullptr, p.d.type, 0, p.biasStride, 1};
                        s.beta = 1.0f;
                    }
                    activation = p.activation;
                    *fused = true;
                }
            }
            const bool small = smallTiles(p);
            return p.a.type == CUDA_R_16F ? cutlassHalf(q, s, output_float, wide, small, activation)
                                          : cutlassBFloat16(q, s, output_float, wide, small, activation);
        }
        case CUDA_R_32F: {
            CutlassScalars<float> s{};
            if (p.d.type != CUDA_R_32F || !cutlassScalars(p, s, CUDA_R_32F) || !operandsAligned(p, 1, 1)) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            if (p.allowTF32) {
                // 64 x 64 tiles only when 128 x 128 ones wouldn't fill the GPU once (not tuned beyond that for TF32)
                const long long tiles = ((p.m + 127LL) / 128) * ((p.n + 127LL) / 128) * p.batch;
                return cutlassTF32(q, s, operandsAligned(p, 4, 4), tiles < multiprocessors());
            }
            return cutlassFloat(q, s);
        }
        case CUDA_R_64F: {
            CutlassScalars<double> s{};
            if (p.d.type != CUDA_R_64F || !cutlassScalars(p, s, CUDA_R_64F) || !operandsAligned(p, 1, 1)) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            return cutlassDouble(q, s);
        }
        default:
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
}

// ---- 16 bit operands the tensor core kernels can't take as they are (odd sizes, leading dimensions, offsets): copied
// to aligned scratch memory with the sizes rounded up to multiples of 8 (zero padded), multiplied there, and the result
// copied back. The copies cost O(mk + kn + mn), little next to the O(mnk) the generic kernel would spend.

// scratch memory per device and stream (work on one stream is ordered, so a stream can reuse its buffer)
void *scratch(cudaStream_t stream, size_t bytes, int slot = 0) {
    struct Buffer {
        void *ptr;
        size_t size;
    };
    static std::mutex mutex;
    static std::map<std::tuple<int, cudaStream_t, int>, Buffer> buffers;
    int device = 0;
    cudaGetDevice(&device);
    std::lock_guard lock(mutex);
    Buffer &buffer = buffers[{device, stream, slot}];
    if (buffer.size < bytes) {
        if (buffer.ptr != nullptr) {
            // the stream's earlier GEMMs may still use it
            cudaStreamSynchronize(stream);
            cudaFree(buffer.ptr);
        }
        const size_t size = std::max(bytes, buffer.size * 2);
        buffer = {nullptr, 0};
        if (cudaMalloc(&buffer.ptr, size) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        buffer.size = size;
    }
    return buffer.ptr;
}

// dst (dstRows x dstCols) = src (rows x cols), zero where src has no element
template<typename T>
__global__ void copyMatrixKernel(T *dst, long long dstLd, long long dstStride, long long dstRows, long long dstCols,
                                 const T *src, long long srcLd, long long srcStride, long long rows, long long cols,
                                 int batch) {
    const long long total = dstRows * dstCols;
    for (long long b = blockIdx.y; b < batch; b += gridDim.y) {
        for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
             i += static_cast<long long>(gridDim.x) * blockDim.x) {
            const long long row = i % dstRows, col = i / dstRows;
            dst[b * dstStride + col * dstLd + row] = row < rows && col < cols ? src[b * srcStride + col * srcLd + row]
                                                                              : T{};
        }
    }
}

// copies a (batch of) column-major matrices between leading dimensions / batch strides, padding with zeros
cublasStatus_t copyMatrix(void *dst, long long dstLd, long long dstStride, long long dstRows, long long dstCols,
                          const void *src, long long srcLd, long long srcStride, long long rows, long long cols,
                          int batch, size_t element, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(std::min<long long>((dstRows * dstCols + 255) / 256, 8192)),
                    std::min(batch, 65535));
    if (element == 2) {
        copyMatrixKernel<<<grid, 256, 0, stream>>>(static_cast<uint16_t *>(dst), dstLd, dstStride, dstRows, dstCols,
                                                   static_cast<const uint16_t *>(src), srcLd, srcStride, rows, cols,
                                                   batch);
    } else {
        copyMatrixKernel<<<grid, 256, 0, stream>>>(static_cast<uint32_t *>(dst), dstLd, dstStride, dstRows, dstCols,
                                                   static_cast<const uint32_t *>(src), srcLd, srcStride, rows, cols,
                                                   batch);
    }
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t repacked(const GemmParams &p) {
    const bool sixteen_bit = p.a.type == CUDA_R_16F || p.a.type == CUDA_R_16BF;
    if (!sixteen_bit || p.a.type != p.b.type || p.k == 0 || p.a.array != nullptr || p.b.array != nullptr ||
        p.d.array != nullptr || p.a.inc != 1 || p.b.inc != 1 || p.d.inc != 1 || (p.c.ptr != nullptr && p.c.inc != 1) ||
        (p.d.type != p.a.type && p.d.type != CUDA_R_32F)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    // small problems: the generic kernel is fast enough
    if (double(p.m) * p.n * p.k < double(1 << 21)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    CutlassScalars<float> scalars{};
    if (!cutlassScalars(p, scalars, CUDA_R_32F)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    // C is read unless beta is (known to be) 0
    const bool read_c = p.beta.onDevice || !isZero(loadAny<float>(p.beta.ptr, p.beta.type));
    if (read_c && p.c.ptr == nullptr) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const size_t in_size = elementSize(p.a.type), out_size = elementSize(p.d.type);
    /**
     * Only the output's rows misaligned (A and B along k, aligned: a linear layer's x W^T with a number of outputs not
     * a multiple of 8, as Whisper's vocabulary projection of 51865): the aligned rows run on A and B as they are
     * into an aligned D, the last few with the generic kernel, and only D is copied. Copying A (the weights, 80 MB
     * for Whisper's projection) for every decoding step took 3.5 times as long as the multiplication.
     */
    if (p.transa != CUBLAS_OP_N && p.transb == CUBLAS_OP_N && p.k % 8 == 0 && p.m >= 8 && aligned(p.a, in_size, 8) &&
        aligned(p.b, in_size, 8)) {
        const long long m8 = p.m / 8 * 8;
        const size_t d_bytes = (size_t(m8) * p.n * p.batch * out_size + 255) / 256 * 256;
        char *base = static_cast<char *>(scratch(p.stream, d_bytes));
        if (base == nullptr) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        GemmParams q = p;
        q.m = static_cast<int>(m8);
        q.d = Operand{base, nullptr, p.d.type, m8, m8 * p.n, 1};
        q.c = read_c ? q.d : Operand{nullptr, nullptr, p.d.type, m8, m8 * p.n, 1};
        q.bias = nullptr;
        q.activation = Activation::NONE;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        if (read_c) {
            status = copyMatrix(base, m8, m8 * p.n, m8, p.n, p.c.ptr, p.c.ld, p.c.stride, m8, p.n, p.batch, out_size,
                                p.stream);
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = runFast(q);
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = copyMatrix(const_cast<void *>(p.d.ptr), p.d.ld, p.d.stride, m8, p.n, base, m8, m8 * p.n, m8, p.n,
                                p.batch, out_size, p.stream);
        }
        if (status == CUBLAS_STATUS_SUCCESS && m8 < p.m) {
            // the last rows: rows m8.. of op(A) are A's columns m8..
            GemmParams r = p;
            r.m = static_cast<int>(p.m - m8);
            r.a.ptr = static_cast<const char *>(p.a.ptr) + m8 * p.a.ld * in_size;
            r.d.ptr = static_cast<const char *>(p.d.ptr) + m8 * out_size;
            if (r.c.ptr != nullptr) {
                r.c.ptr = static_cast<const char *>(p.c.ptr) + m8 * elementSize(p.c.type);
            }
            r.bias = nullptr;
            r.activation = Activation::NONE;
            status = generic(r);
        }
        return status;
    }
    auto round = [](long long value) { return (value + 7) / 8 * 8; };
    // the padded problem and the stored dimensions of A and B in it (the original ones: a_rows x a_cols, ...)
    const long long m = round(p.m), n = round(p.n), k = round(p.k);
    const long long a_rows = p.transa == CUBLAS_OP_N ? p.m : p.k, a_cols = p.transa == CUBLAS_OP_N ? p.k : p.m;
    const long long b_rows = p.transb == CUBLAS_OP_N ? p.k : p.n, b_cols = p.transb == CUBLAS_OP_N ? p.n : p.k;
    const long long a_ld = p.transa == CUBLAS_OP_N ? m : k, a_width = p.transa == CUBLAS_OP_N ? k : m;
    const long long b_ld = p.transb == CUBLAS_OP_N ? k : n, b_width = p.transb == CUBLAS_OP_N ? n : k;
    auto bytes = [&](long long ld, long long cols, size_t element) {
        return (size_t(ld) * cols * p.batch * element + 255) / 256 * 256;
    };
    const size_t a_bytes = bytes(a_ld, a_width, in_size), b_bytes = bytes(b_ld, b_width, in_size);
    char *base = static_cast<char *>(scratch(p.stream, a_bytes + b_bytes + bytes(m, n, out_size)));
    if (base == nullptr) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    GemmParams q = p;
    q.m = static_cast<int>(m);
    q.n = static_cast<int>(n);
    q.k = static_cast<int>(k);
    q.a = Operand{base, nullptr, p.a.type, a_ld, a_ld * a_width, 1};
    q.b = Operand{base + a_bytes, nullptr, p.b.type, b_ld, b_ld * b_width, 1};
    q.d = Operand{base + a_bytes + b_bytes, nullptr, p.d.type, m, m * n, 1};
    q.c = read_c ? q.d : Operand{nullptr, nullptr, p.d.type, m, m * n, 1};
    q.bias = nullptr;
    q.activation = Activation::NONE;
    cublasStatus_t status = copyMatrix(const_cast<void *>(q.a.ptr), q.a.ld, q.a.stride, a_ld, a_width, p.a.ptr,
                                       p.a.ld, p.a.stride, a_rows, a_cols, p.batch, in_size, p.stream);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = copyMatrix(const_cast<void *>(q.b.ptr), q.b.ld, q.b.stride, b_ld, b_width, p.b.ptr, p.b.ld,
                            p.b.stride, b_rows, b_cols, p.batch, in_size, p.stream);
    }
    if (status == CUBLAS_STATUS_SUCCESS && read_c) {
        status = copyMatrix(const_cast<void *>(q.d.ptr), q.d.ld, q.d.stride, m, n, p.c.ptr, p.c.ld, p.c.stride, p.m,
                            p.n, p.batch, out_size, p.stream);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = runFast(q);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = copyMatrix(const_cast<void *>(p.d.ptr), p.d.ld, p.d.stride, p.m, p.n, q.d.ptr, q.d.ld, q.d.stride,
                            p.m, p.n, p.batch, out_size, p.stream);
    }
    return status;
}

// ---- cuBLASLt epilogues: D = activation(D + bias)

// (bias element sizes on the device)
__device__ inline size_t biasElementSize(cudaDataType type) {
    return type == CUDA_R_16F || type == CUDA_R_16BF ? 2 : type == CUDA_R_32F ? 4 : 8;
}

__device__ inline float activate(float x, Activation activation) {
    switch (activation) {
        case Activation::RELU:
            return x > 0 ? x : 0;
        case Activation::GELU:
            // the tanh approximation, as cuBLASLt
            return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

template<typename TD>
__global__ void epilogueKernel(GemmParams p) {
    const long long total = static_cast<long long>(p.m) * p.n;
    for (long long batch = blockIdx.y; batch < p.batch; batch += gridDim.y) {
        TD *d = const_cast<TD *>(batchPointer<TD>(p.d, batch));
        for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
             i += static_cast<long long>(gridDim.x) * blockDim.x) {
            const long long row = i % p.m, col = i / p.m;
            TD *element = d + row * p.d.inc + col * p.d.ld;
            float value = load<TD, float>(element);
            if (p.bias != nullptr) {
                value += loadAny<float>(static_cast<const char *>(p.bias) +
                                        (batch * p.biasStride + row) * biasElementSize(p.biasType), p.biasType);
            }
            store(element, activate(value, p.activation));
        }
    }
}

cublasStatus_t epilogue(const GemmParams &p) {
    if (p.bias == nullptr && p.activation == Activation::NONE) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const long long total = static_cast<long long>(p.m) * p.n;
    const dim3 grid(static_cast<unsigned int>(std::min<long long>((total + 255) / 256, 4096)),
                    std::min(p.batch, 65535));
    switch (p.d.type) {
        case CUDA_R_16F:
            epilogueKernel<__half><<<grid, 256, 0, p.stream>>>(p);
            break;
        case CUDA_R_16BF:
            epilogueKernel<__nv_bfloat16><<<grid, 256, 0, p.stream>>>(p);
            break;
        case CUDA_R_32F:
            epilogueKernel<float><<<grid, 256, 0, p.stream>>>(p);
            break;
        default:
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

// ---- dot products

template<typename T, typename A, typename TR>
__global__ void dotKernel(int n, const T *x, long long incx, const T *y, long long incy, TR *result, bool conj) {
    __shared__ A partial[256];
    A acc = zero<A>();
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        A xv = load<T, A>(x + i * incx);
        if (conj) {
            xv = conjugate(xv);
        }
        acc = add(acc, mul(xv, load<T, A>(y + i * incy)));
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] = add(partial[threadIdx.x], partial[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store(result, partial[0]);
    }
}

// a small device buffer per device and thread for results cuBLAS returns to the host
void *hostResultBuffer() {
    thread_local std::unordered_map<int, void *> buffers;
    int device = 0;
    cudaGetDevice(&device);
    void *&buffer = buffers[device];
    if (buffer == nullptr && cudaMalloc(&buffer, 64) != cudaSuccess) {
        buffer = nullptr;
    }
    return buffer;
}

template<typename T, typename A, typename TR>
cublasStatus_t launchDot(int n, const void *x, long long incx, const void *y, long long incy, void *result,
                         bool conj, cudaStream_t stream) {
    // negative increments run backwards from the last element
    const T *xs = static_cast<const T *>(x) + (incx < 0 ? (1LL - n) * incx : 0);
    const T *ys = static_cast<const T *>(y) + (incy < 0 ? (1LL - n) * incy : 0);
    dotKernel<T, A, TR><<<1, 256, 0, stream>>>(n, xs, incx, ys, incy, static_cast<TR *>(result), conj);
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

} // namespace
} // namespace librecuda_blas

using namespace librecuda_blas;

namespace librecuda_blas {
namespace {

cublasStatus_t dispatchGemm(const GemmParams &p) {
    bool fused = false;
    cublasStatus_t status = runGemv(p);
    if (p.a.type == CUDA_R_8I && p.b.type == CUDA_R_8I) {
        CutlassScalars<int32_t> s{};
        const long long tiles = ((p.m + 127) / 128) * ((p.n + 127) / 128) * static_cast<long long>(p.batch);
        status = cutlassScalars(p, s, CUDA_R_32I) ? cutlassInt8(p, s, tiles < 2 * multiprocessors())
                                                  : CUBLAS_STATUS_NOT_SUPPORTED;
        return status == CUBLAS_STATUS_NOT_SUPPORTED ? generic(p) : status;
    }
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = runFast(p, &fused);
    }
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        fused = false;
        status = repacked(p);
    }
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = generic(p);
    }
    if (status != CUBLAS_STATUS_SUCCESS || fused) {
        return status;
    }
    return epilogue(p);
}

// D = alpha * (sum of the slices' products) + beta * C (slices: float m x n, one after another)
template<typename TD>
__global__ void reduceSlices(GemmParams p, const float *partials, int slices, KernelScalar<float> alpha_arg,
                             KernelScalar<float> beta_arg) {
    const float alpha = alpha_arg.get(), beta = beta_arg.get();
    const long long total = static_cast<long long>(p.m) * p.n;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float sum = 0;
        for (int s = 0; s < slices; s++) sum += partials[s * total + i];
        const long long row = i % p.m, col = i / p.m;
        float value = alpha * sum;
        if (beta != 0.0f) value += beta * load<TD, float>(static_cast<const TD *>(p.c.ptr) + row + col * p.c.ld);
        store(const_cast<TD *>(static_cast<const TD *>(p.d.ptr)) + row + col * p.d.ld, value);
    }
}

/**
 * GEMMs of few output tiles and a long inner dimension (a weight gradient: 128 x 128 outputs over 100000 rows of a
 * graph's nodes): the inner dimension split into slices, run as one batched GEMM into float partial products (the
 * operands offset along k), then summed. One thread block walked all of k for a tile: 7 ms where cuBLAS's split
 * kernels take 0.3.
 */
cublasStatus_t splitK(const GemmParams &p) {
    auto real16or32 = [](cudaDataType t) { return t == CUDA_R_16F || t == CUDA_R_16BF || t == CUDA_R_32F; };
    if (p.batch != 1 || p.a.array != nullptr || p.b.array != nullptr || p.d.array != nullptr ||
        p.c.array != nullptr || p.a.type != p.b.type || !real16or32(p.a.type) || !real16or32(p.d.type) ||
        p.a.inc != 1 || p.b.inc != 1 || p.d.inc != 1 || (p.c.ptr != nullptr && (p.c.inc != 1 || p.c.type != p.d.type)) ||
        p.transa == CUBLAS_OP_C || p.transb == CUBLAS_OP_C || p.n <= GEMV_MAX_COLUMNS) {
        // (a few columns: the matrix times vectors kernel reads A once, at full bandwidth)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const long long tiles = ((p.m + 127) / 128) * static_cast<long long>((p.n + 127) / 128);
    const int sms = multiprocessors();
    if (2 * tiles >= sms) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const long long slices = std::min<long long>({p.k / 2048, (2 * sms + tiles - 1) / tiles, 32});
    if (slices < 2) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    // slices of chunk (a multiple of 8, keeping the operands' alignment), the last one the rest
    const long long chunk = (p.k / slices) / 8 * 8, last = p.k - chunk * (slices - 1);
    const long long elements = static_cast<long long>(p.m) * p.n;
    auto *partials = static_cast<float *>(scratch(p.stream, size_t(elements) * slices * sizeof(float), 1));
    if (partials == nullptr) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    static const float one = 1, zero = 0;
    const size_t in_size = elementSize(p.a.type);
    const long long a_step = p.transa == CUBLAS_OP_N ? chunk * p.a.ld : chunk;
    const long long b_step = p.transb == CUBLAS_OP_N ? chunk : chunk * p.b.ld;
    GemmParams q = p;
    q.k = static_cast<int>(chunk);
    q.batch = static_cast<int>(slices - 1);
    q.a.stride = a_step;
    q.b.stride = b_step;
    q.d = Operand{partials, nullptr, CUDA_R_32F, p.m, elements, 1};
    q.c = Operand{nullptr, nullptr, CUDA_R_32F, p.m, elements, 1};
    q.alpha = Scalar{&one, CUDA_R_32F, false};
    q.beta = Scalar{&zero, CUDA_R_32F, false};
    q.bias = nullptr;
    q.activation = Activation::NONE;
    cublasStatus_t status = dispatchGemm(q);
    if (status == CUBLAS_STATUS_SUCCESS) {
        GemmParams r = q;
        r.k = static_cast<int>(last);
        r.batch = 1;
        r.a.ptr = static_cast<const char *>(p.a.ptr) + (slices - 1) * a_step * in_size;
        r.b.ptr = static_cast<const char *>(p.b.ptr) + (slices - 1) * b_step * in_size;
        r.d.ptr = partials + (slices - 1) * elements;
        status = dispatchGemm(r);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    const KernelScalar<float> alpha = kernelScalar<float>(p.alpha), beta = kernelScalar<float>(p.beta);
    const unsigned blocks = unsigned(std::min<long long>((elements + 255) / 256, 4096));
    switch (p.d.type) {
        case CUDA_R_16F:
            reduceSlices<__half><<<blocks, 256, 0, p.stream>>>(p, partials, int(slices), alpha, beta);
            break;
        case CUDA_R_16BF:
            reduceSlices<__nv_bfloat16><<<blocks, 256, 0, p.stream>>>(p, partials, int(slices), alpha, beta);
            break;
        default:
            reduceSlices<float><<<blocks, 256, 0, p.stream>>>(p, partials, int(slices), alpha, beta);
            break;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return epilogue(p);
}

} // namespace
} // namespace librecuda_blas

cublasStatus_t librecuda_blas_gemm(const GemmParams *params) {
    const GemmParams &p = *params;
    if (p.m < 0 || p.n < 0 || p.k < 0 || p.batch < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (p.m == 0 || p.n == 0 || p.batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const cublasStatus_t status = splitK(p);
    return status == CUBLAS_STATUS_NOT_SUPPORTED ? dispatchGemm(p) : status;
}

cublasStatus_t librecuda_blas_dot(int n, const void *x, cudaDataType xType, long long incx, const void *y,
                                  cudaDataType yType, long long incy, void *result, cudaDataType resultType,
                                  bool resultOnDevice, bool conjugate, cudaStream_t stream) {
    if (xType != yType || xType != resultType) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    void *out = resultOnDevice ? result : hostResultBuffer();
    if (out == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    cublasStatus_t status;
    if (n <= 0) {
        // the result is 0
        status = cudaMemsetAsync(out, 0, elementSize(resultType), stream) == cudaSuccess
                 ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
    } else {
        switch (xType) {
            case CUDA_R_16F:
                status = launchDot<__half, float, __half>(n, x, incx, y, incy, out, false, stream);
                break;
            case CUDA_R_16BF:
                status = launchDot<__nv_bfloat16, float, __nv_bfloat16>(n, x, incx, y, incy, out, false, stream);
                break;
            case CUDA_R_32F:
                status = launchDot<float, float, float>(n, x, incx, y, incy, out, false, stream);
                break;
            case CUDA_R_64F:
                status = launchDot<double, double, double>(n, x, incx, y, incy, out, false, stream);
                break;
            case CUDA_C_32F:
                status = launchDot<cuComplex, Cplx<float>, cuComplex>(n, x, incx, y, incy, out, conjugate, stream);
                break;
            case CUDA_C_64F:
                status = launchDot<cuDoubleComplex, Cplx<double>, cuDoubleComplex>(n, x, incx, y, incy, out, conjugate,
                                                                                   stream);
                break;
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    if (status != CUBLAS_STATUS_SUCCESS || resultOnDevice) {
        return status;
    }
    // host results: the call returns when the result is there
    if (cudaMemcpyAsync(result, out, elementSize(resultType), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// (slot 1: the GEMM engine's repacking uses slot 0 while the routines' buffers are in use)
void *librecuda_blas_scratch(cudaStream_t stream, size_t bytes) {
    return scratch(stream, bytes, 1);
}
