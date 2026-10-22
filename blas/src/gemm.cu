/**
 * The GEMM engine of LibreCUDA's cuBLAS / cuBLASLt: CUTLASS kernels for the common cases (16 bit operands on tensor
 * cores, float and double), a generic kernel for everything else (complex numbers, conjugation, unaligned or strided
 * operands, scalars of other types), and the cuBLASLt epilogues (bias, ReLU, GELU) as a pass over the result.
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

// ---- arithmetic of the accumulator types (float, double, Cplx<float>, Cplx<double>)

template<typename A>
__host__ __device__ A zero() { return A{}; }

__host__ __device__ inline float mul(float a, float b) { return a * b; }
__host__ __device__ inline double mul(double a, double b) { return a * b; }
template<typename R>
__host__ __device__ Cplx<R> mul(Cplx<R> a, Cplx<R> b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

__host__ __device__ inline float add(float a, float b) { return a + b; }
__host__ __device__ inline double add(double a, double b) { return a + b; }
template<typename R>
__host__ __device__ Cplx<R> add(Cplx<R> a, Cplx<R> b) { return {a.re + b.re, a.im + b.im}; }

__host__ __device__ inline float conjugate(float a) { return a; }
__host__ __device__ inline double conjugate(double a) { return a; }
template<typename R>
__host__ __device__ Cplx<R> conjugate(Cplx<R> a) { return {a.re, -a.im}; }

__host__ __device__ inline bool isZero(float a) { return a == 0; }
__host__ __device__ inline bool isZero(double a) { return a == 0; }
template<typename R>
__host__ __device__ bool isZero(Cplx<R> a) { return a.re == 0 && a.im == 0; }

// ---- conversions between element types and accumulator types

template<typename A>
__host__ __device__ A fromComplex(double re, double im);
template<> __host__ __device__ inline float fromComplex<float>(double re, double) { return float(re); }
template<> __host__ __device__ inline double fromComplex<double>(double re, double) { return re; }
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

cublasStatus_t runFast(const GemmParams &p) {
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
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            return p.a.type == CUDA_R_16F ? cutlassHalf(q, s, output_float, wide)
                                          : cutlassBFloat16(q, s, output_float, wide);
        }
        case CUDA_R_32F: {
            CutlassScalars<float> s{};
            if (p.d.type != CUDA_R_32F || !cutlassScalars(p, s, CUDA_R_32F) || !operandsAligned(p, 1, 1)) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            if (p.allowTF32) {
                return cutlassTF32(q, s, operandsAligned(p, 4, 4));
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

cublasStatus_t librecuda_blas_gemm(const GemmParams *params) {
    const GemmParams &p = *params;
    if (p.m < 0 || p.n < 0 || p.k < 0 || p.batch < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (p.m == 0 || p.n == 0 || p.batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasStatus_t status = runFast(p);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = repacked(p);
    }
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = generic(p);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return epilogue(p);
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
