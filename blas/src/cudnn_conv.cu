/**
 * Convolutions of LibreCUDA's cuDNN as im2col and GEMMs on the GEMM engine (tensor cores for half / bfloat16 data).
 *
 * Per chunk of images and per group g (Cg = C / groups input channels, Kg = K / groups filters, Kd = Cg * R*S*T the
 * filter volume, PQ the output positions):
 *  - forward:          out (Kg x PQ*n)  = W_g (Kg x Kd) * cols_g (Kd x PQ*n), then y = alpha out + beta y
 *  - backward data:    cols_g (Kd x PQ*n) = W_g^T * dy_g (Kg x PQ*n), then dx = alpha col2im(cols) + beta dx (gathered)
 *  - backward filter:  dw_g (Kg x Kd) += dy_g * cols_g^T over the chunks, then dw = alpha dw + beta dw
 * cols holds the input windows (im2col) of the chunk, [g][n][PQ][Kd]; dy and out are [n][PQ][K]. Filters are packed
 * as [K][Kd] (flipped for CUDNN_CONVOLUTION). Accumulation is float (double for double data).
 */
#include "blas_internal.h"
#include "dnn_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace librecuda_dnn {

size_t dataTypeSize(cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_HALF:
        case CUDNN_DATA_BFLOAT16:
            return 2;
        case CUDNN_DATA_FLOAT:
        case CUDNN_DATA_INT32:
            return 4;
        case CUDNN_DATA_DOUBLE:
        case CUDNN_DATA_INT64:
            return 8;
        case CUDNN_DATA_INT8:
        case CUDNN_DATA_UINT8:
        case CUDNN_DATA_BOOLEAN:
            return 1;
        default:
            return 0;
    }
}

namespace {

// the shape of a problem with 3 spatial dimensions (missing ones have size 1)
// largest depthwise filter (taps) the direct kernels take (weight gradients keep a sum per tap in registers)
constexpr int64_t DEPTHWISE_MAX_TAPS = 49;

struct Shape {
    int64_t n, c, k, groups, cg, kg;
    int64_t in[3], filter[3], out[3];
    int64_t xs[5], ws[5], ys[5];  // strides: N / K, C, spatial (3)
    int64_t kd, pq;
    int64_t pad[3], stride[3], dilation[3];
    bool flip;
    bool tf32;  // float data multiplied as TF32
};

Shape shapeOf(const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g) {
    Shape s{};
    s.n = x.dims[0];
    s.c = x.dims[1];
    s.k = w.dims[0];
    s.cg = w.dims[1];
    s.groups = s.c / s.cg;
    s.kg = s.k / s.groups;
    for (int i = 0; i < 2; i++) {
        s.xs[i] = x.strides[i];
        s.ws[i] = w.strides[i];
        s.ys[i] = y.strides[i];
    }
    for (int d = 0; d < 3; d++) {
        const bool used = d < g.spatial;
        // (the spatial dimensions end with the last: pad in front)
        const int src = 2 + d - (3 - g.spatial);
        const bool real = d >= 3 - g.spatial;
        (void) used;
        s.in[d] = real ? x.dims[src] : 1;
        s.filter[d] = real ? w.dims[src] : 1;
        s.out[d] = real ? y.dims[src] : 1;
        s.xs[2 + d] = real ? x.strides[src] : 0;
        s.ws[2 + d] = real ? w.strides[src] : 0;
        s.ys[2 + d] = real ? y.strides[src] : 0;
        const int gd = d - (3 - g.spatial);
        s.pad[d] = real ? g.padBefore[gd] : 0;
        s.stride[d] = real ? g.stride[gd] : 1;
        s.dilation[d] = real ? g.dilation[gd] : 1;
    }
    s.kd = s.cg * s.filter[0] * s.filter[1] * s.filter[2];
    s.pq = s.out[0] * s.out[1] * s.out[2];
    s.flip = g.flip;
    s.tf32 = g.allowTF32 && x.type == CUDNN_DATA_FLOAT;
    return s;
}

template<typename T>
__device__ inline float toFloat(T v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return float(v);
    }
}

template<typename T, typename A>
__device__ inline T fromAcc(A v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(float(v));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(float(v));
    } else {
        return T(v);
    }
}

template<typename T, typename A>
__device__ inline A toAcc(T v) {
    if constexpr (std::is_same_v<T, double>) {
        return A(v);
    } else {
        return A(toFloat(v));
    }
}

#define GRID_LOOP(i, total) \
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < (total); \
         i += static_cast<long long>(gridDim.x) * blockDim.x)

// cols[g][n][pq][kd] = x[n0 + n][g * cg + c][window position]
template<typename T>
__global__ void im2col(Shape s, const T *x, int64_t n0, int64_t count, T *cols) {
    const long long total = s.groups * count * s.pq * s.kd;
    GRID_LOOP(e, total) {
        long long rest = e;
        const int64_t kd = rest % s.kd;
        rest /= s.kd;
        const int64_t pq = rest % s.pq;
        rest /= s.pq;
        const int64_t n = rest % count;
        const int64_t g = rest / count;
        int64_t f = kd;
        const int64_t r2 = f % s.filter[2];
        f /= s.filter[2];
        const int64_t r1 = f % s.filter[1];
        f /= s.filter[1];
        const int64_t r0 = f % s.filter[0];
        const int64_t cg = f / s.filter[0];
        int64_t o = pq;
        const int64_t p2 = o % s.out[2];
        o /= s.out[2];
        const int64_t p1 = o % s.out[1];
        const int64_t p0 = o / s.out[1];
        const int64_t i0 = p0 * s.stride[0] - s.pad[0] + r0 * s.dilation[0];
        const int64_t i1 = p1 * s.stride[1] - s.pad[1] + r1 * s.dilation[1];
        const int64_t i2 = p2 * s.stride[2] - s.pad[2] + r2 * s.dilation[2];
        T value{};
        if (i0 >= 0 && i0 < s.in[0] && i1 >= 0 && i1 < s.in[1] && i2 >= 0 && i2 < s.in[2]) {
            value = x[(n0 + n) * s.xs[0] + (g * s.cg + cg) * s.xs[1] + i0 * s.xs[2] + i1 * s.xs[3] + i2 * s.xs[4]];
        }
        cols[e] = value;
    }
}

// packed filters [k][kd] (flipped for CUDNN_CONVOLUTION)
template<typename T>
__global__ void packFilter(Shape s, const T *w, T *packed) {
    GRID_LOOP(e, s.k * s.kd) {
        const int64_t k = e / s.kd;
        int64_t f = e % s.kd;
        int64_t r2 = f % s.filter[2];
        f /= s.filter[2];
        int64_t r1 = f % s.filter[1];
        f /= s.filter[1];
        int64_t r0 = f % s.filter[0];
        const int64_t cg = f / s.filter[0];
        if (s.flip) {
            r0 = s.filter[0] - 1 - r0;
            r1 = s.filter[1] - 1 - r1;
            r2 = s.filter[2] - 1 - r2;
        }
        packed[e] = w[k * s.ws[0] + cg * s.ws[1] + r0 * s.ws[2] + r1 * s.ws[3] + r2 * s.ws[4]];
    }
}

// packed[n][pq][k] = y[n0 + n][k][pq]
template<typename T>
__global__ void packOutput(Shape s, const T *y, int64_t n0, int64_t count, T *packed) {
    GRID_LOOP(e, count * s.pq * s.k) {
        const int64_t k = e % s.k;
        const int64_t pq = (e / s.k) % s.pq;
        const int64_t n = e / (s.k * s.pq);
        int64_t o = pq;
        const int64_t p2 = o % s.out[2];
        o /= s.out[2];
        const int64_t p1 = o % s.out[1];
        const int64_t p0 = o / s.out[1];
        packed[e] = y[(n0 + n) * s.ys[0] + k * s.ys[1] + p0 * s.ys[2] + p1 * s.ys[3] + p2 * s.ys[4]];
    }
}

// y[n0 + n][k][pq] = alpha out[n][pq][k] + beta y
template<typename T, typename A>
__global__ void storeOutput(Shape s, const A *out, int64_t n0, int64_t count, T *y, A alpha, A beta) {
    GRID_LOOP(e, count * s.pq * s.k) {
        const int64_t k = e % s.k;
        const int64_t pq = (e / s.k) % s.pq;
        const int64_t n = e / (s.k * s.pq);
        int64_t o = pq;
        const int64_t p2 = o % s.out[2];
        o /= s.out[2];
        const int64_t p1 = o % s.out[1];
        const int64_t p0 = o / s.out[1];
        T &target = y[(n0 + n) * s.ys[0] + k * s.ys[1] + p0 * s.ys[2] + p1 * s.ys[3] + p2 * s.ys[4]];
        A value = alpha * out[e];
        if (beta != A(0)) {
            value += beta * toAcc<T, A>(target);
        }
        target = fromAcc<T, A>(value);
    }
}

// dx[n0 + n][c][position] = alpha sum of the window entries of cols[g][n][pq][kd] covering it + beta dx
template<typename T, typename A>
__global__ void col2im(Shape s, const A *cols, int64_t n0, int64_t count, T *dx, A alpha, A beta) {
    const long long total = count * s.c * s.in[0] * s.in[1] * s.in[2];
    GRID_LOOP(e, total) {
        long long rest = e;
        const int64_t i2 = rest % s.in[2];
        rest /= s.in[2];
        const int64_t i1 = rest % s.in[1];
        rest /= s.in[1];
        const int64_t i0 = rest % s.in[0];
        rest /= s.in[0];
        const int64_t c = rest % s.c;
        const int64_t n = rest / s.c;
        const int64_t g = c / s.cg, cg = c % s.cg;
        A sum = 0;
        for (int64_t r0 = 0; r0 < s.filter[0]; r0++) {
            const int64_t t0 = i0 + s.pad[0] - r0 * s.dilation[0];
            if (t0 < 0 || t0 % s.stride[0] != 0 || t0 / s.stride[0] >= s.out[0]) continue;
            for (int64_t r1 = 0; r1 < s.filter[1]; r1++) {
                const int64_t t1 = i1 + s.pad[1] - r1 * s.dilation[1];
                if (t1 < 0 || t1 % s.stride[1] != 0 || t1 / s.stride[1] >= s.out[1]) continue;
                for (int64_t r2 = 0; r2 < s.filter[2]; r2++) {
                    const int64_t t2 = i2 + s.pad[2] - r2 * s.dilation[2];
                    if (t2 < 0 || t2 % s.stride[2] != 0 || t2 / s.stride[2] >= s.out[2]) continue;
                    const int64_t pq = (t0 / s.stride[0] * s.out[1] + t1 / s.stride[1]) * s.out[2] + t2 / s.stride[2];
                    const int64_t kd = ((cg * s.filter[0] + r0) * s.filter[1] + r1) * s.filter[2] + r2;
                    sum += cols[((g * count + n) * s.pq + pq) * s.kd + kd];
                }
            }
        }
        T &target = dx[(n0 + n) * s.xs[0] + c * s.xs[1] + i0 * s.xs[2] + i1 * s.xs[3] + i2 * s.xs[4]];
        A value = alpha * sum;
        if (beta != A(0)) {
            value += beta * toAcc<T, A>(target);
        }
        target = fromAcc<T, A>(value);
    }
}

// dw[k][cg][r] = alpha acc[kd * K + k] + beta dw (flipped back for CUDNN_CONVOLUTION)
template<typename T, typename A>
__global__ void storeFilter(Shape s, const A *acc, T *w, A alpha, A beta) {
    GRID_LOOP(e, s.k * s.kd) {
        const int64_t k = e % s.k;
        const int64_t kd = e / s.k;
        int64_t f = kd;
        int64_t r2 = f % s.filter[2];
        f /= s.filter[2];
        int64_t r1 = f % s.filter[1];
        f /= s.filter[1];
        int64_t r0 = f % s.filter[0];
        const int64_t cg = f / s.filter[0];
        if (s.flip) {
            r0 = s.filter[0] - 1 - r0;
            r1 = s.filter[1] - 1 - r1;
            r2 = s.filter[2] - 1 - r2;
        }
        T &target = w[k * s.ws[0] + cg * s.ws[1] + r0 * s.ws[2] + r1 * s.ws[3] + r2 * s.ws[4]];
        A value = alpha * acc[e];
        if (beta != A(0)) {
            value += beta * toAcc<T, A>(target);
        }
        target = fromAcc<T, A>(value);
    }
}

unsigned int gridFor(long long count) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>((count + 255) / 256, 1 << 20)));
}

cudaDataType blasType(cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_HALF: return CUDA_R_16F;
        case CUDNN_DATA_BFLOAT16: return CUDA_R_16BF;
        case CUDNN_DATA_DOUBLE: return CUDA_R_64F;
        default: return CUDA_R_32F;
    }
}

/**
 * D (m x n, ld ldd, accumulator type) = op(A) B (+ D if accumulate) with column-major operands of the data type
 */
cudnnStatus_t gemm(cudaStream_t stream, cudnnDataType_t type, bool doubles, bool tf32, cublasOperation_t ta,
                   cublasOperation_t tb, int64_t m, int64_t n, int64_t k, const void *a, int64_t lda, const void *b,
                   int64_t ldb, void *d, int64_t ldd, bool accumulate) {
    using namespace librecuda_blas;
    const double one_d = 1, beta_d = accumulate ? 1 : 0;
    const float one_f = 1, beta_f = accumulate ? 1 : 0;
    GemmParams p{};
    p.transa = ta;
    p.transb = tb;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = 1;
    p.a = Operand{a, nullptr, blasType(type), lda, 0, 1};
    p.b = Operand{b, nullptr, blasType(type), ldb, 0, 1};
    const cudaDataType acc = doubles ? CUDA_R_64F : CUDA_R_32F;
    p.c = Operand{d, nullptr, acc, ldd, 0, 1};
    p.d = p.c;
    p.compute = doubles ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    p.alpha = Scalar{doubles ? static_cast<const void *>(&one_d) : &one_f, acc, false};
    p.beta = Scalar{doubles ? static_cast<const void *>(&beta_d) : &beta_f, acc, false};
    p.allowTF32 = tf32;
    p.stream = stream;
    return librecuda_blas_gemm(&p) == CUBLAS_STATUS_SUCCESS ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

/**
 * The forward pass straight into y: y_n^T (PQ x K, positions contiguous) = cols_n^T W^T for each image of the chunk, one
 * batched GEMM (W shared). For float and double data, one group, alpha 1, beta 0 and y with contiguous positions
 * (NCHW, NCW): storing the packed products into y was a pass a quarter as long as the GEMM (torchaudio's resampling,
 * a 1 x 334 filter bank of 441 outputs).
 */
cudnnStatus_t gemmIntoOutput(cudaStream_t stream, cudnnDataType_t type, bool doubles, bool tf32, const Shape &s,
                             const void *cols, const void *packedW, void *y, int64_t count) {
    using namespace librecuda_blas;
    const double one_d = 1, zero_d = 0;
    const float one_f = 1, zero_f = 0;
    GemmParams p{};
    p.transa = CUBLAS_OP_T;
    p.transb = CUBLAS_OP_N;
    p.m = static_cast<int>(s.pq);
    p.n = static_cast<int>(s.k);
    p.k = static_cast<int>(s.kd);
    p.batch = static_cast<int>(count);
    p.a = Operand{cols, nullptr, blasType(type), s.kd, s.pq * s.kd, 1};
    p.b = Operand{packedW, nullptr, blasType(type), s.kd, 0, 1};
    p.c = Operand{nullptr, nullptr, blasType(type), s.ys[1], s.ys[0], 1};
    p.d = Operand{y, nullptr, blasType(type), s.ys[1], s.ys[0], 1};
    p.compute = doubles ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    p.alpha = Scalar{doubles ? static_cast<const void *>(&one_d) : &one_f, doubles ? CUDA_R_64F : CUDA_R_32F, false};
    p.beta = Scalar{doubles ? static_cast<const void *>(&zero_d) : &zero_f, doubles ? CUDA_R_64F : CUDA_R_32F, false};
    p.allowTF32 = tf32;
    p.stream = stream;
    return librecuda_blas_gemm(&p) == CUBLAS_STATUS_SUCCESS ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// images per chunk: the column buffer of a chunk stays below a budget (at least one image)
int64_t chunkImages(const Shape &s, size_t element) {
    const size_t per_image = static_cast<size_t>(s.groups * s.kd * s.pq) * std::max<size_t>(element, 4) +
                             static_cast<size_t>(s.k * s.pq) * 8;
    const size_t budget = size_t(256) << 20;
    return std::max<int64_t>(1, std::min<int64_t>(s.n, static_cast<int64_t>(budget / std::max<size_t>(per_image, 1))));
}

struct Layout {
    size_t filter, cols, outputs, acc, total;
};

// workspace: packed filters, columns of a chunk, packed outputs (or gradients) of a chunk, the filter gradient
Layout layoutOf(ConvKind kind, const Shape &s, cudnnDataType_t type, bool doubles) {
    const size_t element = dataTypeSize(type), acc = doubles ? 8 : 4;
    const int64_t chunk = chunkImages(s, element);
    auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    Layout l{};
    l.filter = align(static_cast<size_t>(s.k * s.kd) * element);
    // backward data accumulates the columns
    l.cols = align(static_cast<size_t>(s.groups * chunk * s.pq * s.kd) *
                   (kind == ConvKind::BACKWARD_DATA ? acc : element));
    l.outputs = align(static_cast<size_t>(chunk * s.pq * s.k) * (kind == ConvKind::FORWARD ? acc : element));
    l.acc = kind == ConvKind::BACKWARD_FILTER ? align(static_cast<size_t>(s.k * s.kd) * acc) : 0;
    l.total = l.filter + l.cols + l.outputs + l.acc;
    return l;
}

template<typename T, typename A>
cudnnStatus_t run(ConvKind kind, cudaStream_t stream, const Shape &s, cudnnDataType_t type, void *x, void *w, void *y,
                  A alpha, A beta, char *workspace) {
    constexpr bool doubles = std::is_same_v<A, double>;
    const Layout l = layoutOf(kind, s, type, doubles);
    const int64_t chunk = chunkImages(s, sizeof(T));
    T *packed_w = reinterpret_cast<T *>(workspace);
    char *cols = workspace + l.filter;
    char *outputs = cols + l.cols;
    A *acc = reinterpret_cast<A *>(outputs + l.outputs);
    if (kind != ConvKind::BACKWARD_FILTER) {
        packFilter<<<gridFor(s.k * s.kd), 256, 0, stream>>>(s, static_cast<const T *>(w), packed_w);
    }
    // (positions of y contiguous: each spatial dimension of more than one position with the stride of those after it,
    // and the channel stride PQ)
    bool contiguous = s.k == 1 || s.ys[1] == s.pq;
    for (int64_t d = 2, expected = 1; d >= 0; d--) {
        if (s.out[d] > 1 && s.ys[2 + d] != expected) contiguous = false;
        expected *= s.out[d];
    }
    const bool direct = kind == ConvKind::FORWARD && std::is_same_v<T, A> && s.groups == 1 && alpha == A(1) &&
                        beta == A(0) && contiguous;
    for (int64_t n0 = 0; n0 < s.n; n0 += chunk) {
        const int64_t count = std::min(chunk, s.n - n0);
        const int64_t columns = s.pq * count;
        if (kind != ConvKind::BACKWARD_DATA) {
            im2col<<<gridFor(s.groups * columns * s.kd), 256, 0, stream>>>(s, static_cast<const T *>(x), n0, count,
                                                                           reinterpret_cast<T *>(cols));
        }
        if (kind != ConvKind::FORWARD) {
            packOutput<<<gridFor(columns * s.k), 256, 0, stream>>>(s, static_cast<const T *>(y), n0, count,
                                                                  reinterpret_cast<T *>(outputs));
        }
        if (cudaGetLastError() != cudaSuccess) {
            return CUDNN_STATUS_EXECUTION_FAILED;
        }
        if (direct) {
            const cudnnStatus_t status = gemmIntoOutput(stream, type, doubles, s.tf32, s, cols, packed_w,
                                                        static_cast<T *>(y) + n0 * s.ys[0], count);
            if (status != CUDNN_STATUS_SUCCESS) {
                return status;
            }
            continue;
        }
        for (int64_t g = 0; g < s.groups; g++) {
            cudnnStatus_t status;
            const T *wg = packed_w + g * s.kg * s.kd;
            if (kind == ConvKind::FORWARD) {
                // out (rows g*Kg.., ld K) = W_g (Kg x Kd) cols_g
                status = gemm(stream, type, doubles, s.tf32, CUBLAS_OP_T, CUBLAS_OP_N, s.kg, columns, s.kd, wg, s.kd,
                              reinterpret_cast<T *>(cols) + g * columns * s.kd, s.kd,
                              reinterpret_cast<A *>(outputs) + g * s.kg, s.k, false);
            } else if (kind == ConvKind::BACKWARD_DATA) {
                // cols_g (Kd x PQ*n) = W_g^T dy_g
                status = gemm(stream, type, doubles, s.tf32, CUBLAS_OP_N, CUBLAS_OP_N, s.kd, columns, s.kg, wg, s.kd,
                              reinterpret_cast<T *>(outputs) + g * s.kg, s.k,
                              reinterpret_cast<A *>(cols) + g * columns * s.kd, s.kd, false);
            } else {
                // dw_g (Kg x Kd, rows g*Kg.. of acc, ld K) += dy_g cols_g^T
                status = gemm(stream, type, doubles, s.tf32, CUBLAS_OP_N, CUBLAS_OP_T, s.kg, s.kd, columns,
                              reinterpret_cast<T *>(outputs) + g * s.kg, s.k,
                              reinterpret_cast<T *>(cols) + g * columns * s.kd, s.kd, acc + g * s.kg, s.k, n0 > 0);
            }
            if (status != CUDNN_STATUS_SUCCESS) {
                return status;
            }
        }
        if (kind == ConvKind::FORWARD) {
            storeOutput<<<gridFor(columns * s.k), 256, 0, stream>>>(s, reinterpret_cast<const A *>(outputs), n0, count,
                                                                   static_cast<T *>(y), alpha, beta);
        } else if (kind == ConvKind::BACKWARD_DATA) {
            col2im<<<gridFor(count * s.c * s.in[0] * s.in[1] * s.in[2]), 256, 0, stream>>>(
                s, reinterpret_cast<const A *>(cols), n0, count, static_cast<T *>(x), alpha, beta);
        }
    }
    if (kind == ConvKind::BACKWARD_FILTER) {
        storeFilter<<<gridFor(s.k * s.kd), 256, 0, stream>>>(s, acc, static_cast<T *>(w), alpha, beta);
    }
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

} // namespace

cudnnStatus_t convCheck(ConvKind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g) {
    if (x.nbDims < 3 || x.nbDims > 5 || w.nbDims != x.nbDims || y.nbDims != x.nbDims || g.spatial != x.nbDims - 2) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (x.type != w.type || x.type != y.type) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    if (x.type != CUDNN_DATA_FLOAT && x.type != CUDNN_DATA_HALF && x.type != CUDNN_DATA_BFLOAT16 &&
        x.type != CUDNN_DATA_DOUBLE) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    if (w.dims[1] <= 0 || x.dims[1] % w.dims[1] != 0 || w.dims[0] % (x.dims[1] / w.dims[1]) != 0 ||
        y.dims[0] != x.dims[0] || y.dims[1] != w.dims[0]) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    for (int d = 0; d < g.spatial; d++) {
        if (g.stride[d] <= 0 || g.dilation[d] <= 0 ||
            y.dims[2 + d] != convOutputSize(x.dims[2 + d], w.dims[2 + d], g.padBefore[d], g.padAfter[d], g.stride[d],
                                            g.dilation[d])) {
            return CUDNN_STATUS_BAD_PARAM;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

bool tf32Permitted() {
    static const bool permitted = [] {
        const char *v = getenv("NVIDIA_TF32_OVERRIDE");
        return v == nullptr || strcmp(v, "0") != 0;
    }();
    return permitted;
}

namespace {

// LIBRECUDA_CUDNN_IM2COL=1: im2col for all convolutions (for comparisons)
bool implicitAllowed() {
    static const bool allowed = [] {
        const char *v = getenv("LIBRECUDA_CUDNN_IM2COL");
        return v == nullptr || v[0] == '\0' || v[0] == '0';
    }();
    return allowed;
}

// ---- 1x1 convolutions (stride 1, no padding): GEMMs per image on the tensors themselves

/**
 * The layout of a pointwise problem: NCHW-like (the positions of a channel are contiguous) or NHWC-like (the channels
 * of a position are contiguous, positions follow each other), the same for x and y; any image strides.
 */
struct Pointwise {
    bool channelsLast;
    int64_t positions;
    int64_t xld, yld;  // between channels (NCHW) / positions (NHWC)
    int64_t wld;       // filters: elements between filters (the channels of a filter are contiguous)
};

// whether the spatial dimensions are contiguous with this step (size 1 dimensions don't matter)
bool spatialContiguous(const Shape &s, const int64_t *strides, int64_t step) {
    int64_t expected = step;
    for (int d = 2; d >= 0; d--) {
        if (s.in[d] != 1 && strides[2 + d] != expected) return false;
        expected *= s.in[d];
    }
    return true;
}

bool pointwiseOf(const Shape &s, Pointwise &pw) {
    if (s.groups != 1 || s.ws[1] != 1) return false;
    for (int d = 0; d < 3; d++) {
        if (s.filter[d] != 1 || s.stride[d] != 1 || s.pad[d] != 0 || s.out[d] != s.in[d]) return false;
    }
    pw.positions = s.pq;
    pw.wld = s.k > 1 ? s.ws[0] : s.c;
    if (spatialContiguous(s, s.xs, 1) && spatialContiguous(s, s.ys, 1) && (s.c == 1 || s.xs[1] >= s.pq) &&
        (s.k == 1 || s.ys[1] >= s.pq)) {
        pw.channelsLast = false;
        pw.xld = s.c > 1 ? s.xs[1] : s.pq;
        pw.yld = s.k > 1 ? s.ys[1] : s.pq;
        return true;
    }
    if ((s.c == 1 || s.xs[1] == 1) && (s.k == 1 || s.ys[1] == 1) && spatialContiguous(s, s.xs, s.c) &&
        spatialContiguous(s, s.ys, s.k)) {
        pw.channelsLast = true;
        pw.xld = s.c;
        pw.yld = s.k;
        return true;
    }
    return false;
}

struct GemmOperand {
    const void *ptr;
    cublasOperation_t op;
    int64_t ld, stride;
};

/**
 * batched D = alpha op(A) op(B) + beta D, column-major; A and B of the data type (float as TF32 if tf32), D of type
 * dType; alpha / beta in the accumulation type
 */
cudnnStatus_t gemmBatched(cudaStream_t stream, cudnnDataType_t type, bool doubles, bool tf32, int64_t m, int64_t n,
                          int64_t k, GemmOperand a, GemmOperand b, void *d, cudaDataType dType, int64_t ldd,
                          int64_t strideD, int64_t batch, double alpha, double beta) {
    using namespace librecuda_blas;
    if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX || batch > INT32_MAX) return CUDNN_STATUS_NOT_SUPPORTED;
    const float alpha_f = float(alpha), beta_f = float(beta);
    const cudaDataType acc = doubles ? CUDA_R_64F : CUDA_R_32F;
    GemmParams p{};
    p.transa = a.op;
    p.transb = b.op;
    p.m = int(m);
    p.n = int(n);
    p.k = int(k);
    p.batch = int(batch);
    p.a = Operand{a.ptr, nullptr, blasType(type), a.ld, a.stride, 1};
    p.b = Operand{b.ptr, nullptr, blasType(type), b.ld, b.stride, 1};
    p.c = Operand{d, nullptr, dType, ldd, strideD, 1};
    p.d = p.c;
    p.compute = doubles ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    p.alpha = Scalar{doubles ? static_cast<const void *>(&alpha) : &alpha_f, acc, false};
    p.beta = Scalar{doubles ? static_cast<const void *>(&beta) : &beta_f, acc, false};
    p.allowTF32 = tf32;
    p.stream = stream;
    return librecuda_blas_gemm(&p) == CUBLAS_STATUS_SUCCESS ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// weight gradients: images per chunk of per-image partial products (at least one)
int64_t pointwiseChunk(const Shape &s, size_t acc) {
    const size_t per_image = size_t(s.c * s.k) * acc;
    return std::max<int64_t>(1, std::min<int64_t>(s.n, int64_t((size_t(128) << 20) / per_image)));
}

size_t pointwiseWorkspace(ConvKind kind, const Shape &s, bool doubles) {
    if (kind != ConvKind::BACKWARD_FILTER) return 0;
    const size_t acc = doubles ? 8 : 4;
    return (size_t(s.c * s.k) * acc + 255) / 256 * 256 + size_t(pointwiseChunk(s, acc) * s.c * s.k) * acc;
}

// sum[e] (= or +=) the partials [count][size]
template<typename A>
__global__ void sumPartials(const A *partials, int64_t count, int64_t size, A *sum, bool accumulate) {
    GRID_LOOP(e, size) {
        A v = accumulate ? sum[e] : A(0);
        for (int64_t i = 0; i < count; i++) v += partials[i * size + e];
        sum[e] = v;
    }
}

// dw[k][c] = alpha sum[k * C + c] + beta dw
template<typename T, typename A>
__global__ void storePointwiseFilter(Shape s, const A *sum, T *w, A alpha, A beta) {
    GRID_LOOP(e, s.k * s.c) {
        T &target = w[(e / s.c) * s.ws[0] + (e % s.c) * s.ws[1]];
        A value = alpha * sum[e];
        if (beta != A(0)) value += beta * toAcc<T, A>(target);
        target = fromAcc<T, A>(value);
    }
}

template<typename T, typename A>
cudnnStatus_t runPointwise(ConvKind kind, cudaStream_t stream, const Shape &s, const Pointwise &pw,
                           cudnnDataType_t type, void *x, void *w, void *y, double alpha, double beta,
                           char *workspace) {
    constexpr bool doubles = std::is_same_v<A, double>;
    const cudaDataType t = blasType(type);
    const int64_t P = pw.positions;
    /**
     * The images folded into the positions where x and y hold each position's channels contiguously, the positions and
     * then the images one after another (packed channels-last tensors; any layout of one position, as the squeeze and
     * excitation layers' 1 x 1 inputs): one GEMM over N * P positions. Per image, those layers' GEMMs had one row
     * (forward, data) or an inner dimension of one (filter) and ran on the generic kernel, 64 at a time.
     */
    const bool merged = (s.c == 1 || s.xs[1] == 1) && (s.k == 1 || s.ys[1] == 1) && spatialContiguous(s, s.xs, s.c) &&
                        spatialContiguous(s, s.ys, s.k) && (s.n == 1 || (s.xs[0] == P * s.c && s.ys[0] == P * s.k));
    if (merged) {
        const int64_t rows = s.n * P;
        switch (kind) {
            case ConvKind::FORWARD:
                return gemmBatched(stream, type, doubles, s.tf32, s.k, rows, s.c, {w, CUBLAS_OP_T, pw.wld, 0},
                                   {x, CUBLAS_OP_N, s.c, 0}, y, t, s.k, 0, 1, alpha, beta);
            case ConvKind::BACKWARD_DATA:
                return gemmBatched(stream, type, doubles, s.tf32, s.c, rows, s.k, {w, CUBLAS_OP_N, pw.wld, 0},
                                   {y, CUBLAS_OP_N, s.k, 0}, x, t, s.c, 0, 1, alpha, beta);
            default: {
                // dW (C x K) = X (C x rows) dY^T, then w = alpha dW + beta w
                A *sum = reinterpret_cast<A *>(workspace);
                const cudnnStatus_t status =
                    gemmBatched(stream, type, doubles, s.tf32, s.c, s.k, rows, {x, CUBLAS_OP_N, s.c, 0},
                                {y, CUBLAS_OP_T, s.k, 0}, sum, doubles ? CUDA_R_64F : CUDA_R_32F, s.c, 0, 1, 1, 0);
                if (status != CUDNN_STATUS_SUCCESS) return status;
                storePointwiseFilter<T, A><<<gridFor(s.c * s.k), 256, 0, stream>>>(s, sum, static_cast<T *>(w),
                                                                                     A(alpha), A(beta));
                return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
            }
        }
    }
    if (!pw.channelsLast) {
        // per image: x_n (P x C, ld xld), y_n (P x K, ld yld), filters (C x K, ld wld)
        switch (kind) {
            case ConvKind::FORWARD:
                return gemmBatched(stream, type, doubles, s.tf32, P, s.k, s.c, {x, CUBLAS_OP_N, pw.xld, s.xs[0]},
                                   {w, CUBLAS_OP_N, pw.wld, 0}, y, t, pw.yld, s.ys[0], s.n, alpha, beta);
            case ConvKind::BACKWARD_DATA:
                return gemmBatched(stream, type, doubles, s.tf32, P, s.c, s.k, {y, CUBLAS_OP_N, pw.yld, s.ys[0]},
                                   {w, CUBLAS_OP_T, pw.wld, 0}, x, t, pw.xld, s.xs[0], s.n, alpha, beta);
            default:
                break;
        }
    } else {
        // per image: x_n (C x P, ld C), y_n (K x P, ld K)
        switch (kind) {
            case ConvKind::FORWARD:
                return gemmBatched(stream, type, doubles, s.tf32, s.k, P, s.c, {w, CUBLAS_OP_T, pw.wld, 0},
                                   {x, CUBLAS_OP_N, pw.xld, s.xs[0]}, y, t, pw.yld, s.ys[0], s.n, alpha, beta);
            case ConvKind::BACKWARD_DATA:
                return gemmBatched(stream, type, doubles, s.tf32, s.c, P, s.k, {w, CUBLAS_OP_N, pw.wld, 0},
                                   {y, CUBLAS_OP_N, pw.yld, s.ys[0]}, x, t, pw.xld, s.xs[0], s.n, alpha, beta);
            default:
                break;
        }
    }
    // weight gradients: per image partials (C x K) = X_n^T dY_n, summed over the images
    const size_t accSize = sizeof(A);
    A *sum = reinterpret_cast<A *>(workspace);
    A *partials = reinterpret_cast<A *>(workspace + (size_t(s.c * s.k) * accSize + 255) / 256 * 256);
    const int64_t chunk = pointwiseChunk(s, accSize);
    const cudaDataType accType = doubles ? CUDA_R_64F : CUDA_R_32F;
    const size_t element = dataTypeSize(type);
    for (int64_t n0 = 0; n0 < s.n; n0 += chunk) {
        const int64_t count = std::min(chunk, s.n - n0);
        const char *xn = static_cast<const char *>(x) + size_t(n0 * s.xs[0]) * element;
        const char *yn = static_cast<const char *>(y) + size_t(n0 * s.ys[0]) * element;
        const cudnnStatus_t status =
            !pw.channelsLast
                ? gemmBatched(stream, type, doubles, s.tf32, s.c, s.k, P, {xn, CUBLAS_OP_T, pw.xld, s.xs[0]},
                              {yn, CUBLAS_OP_N, pw.yld, s.ys[0]}, partials, accType, s.c, s.c * s.k, count, 1, 0)
                : gemmBatched(stream, type, doubles, s.tf32, s.c, s.k, P, {xn, CUBLAS_OP_N, pw.xld, s.xs[0]},
                              {yn, CUBLAS_OP_T, pw.yld, s.ys[0]}, partials, accType, s.c, s.c * s.k, count, 1, 0);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        sumPartials<A><<<gridFor(s.c * s.k), 256, 0, stream>>>(partials, count, s.c * s.k, sum, n0 > 0);
    }
    storePointwiseFilter<T, A><<<gridFor(s.c * s.k), 256, 0, stream>>>(s, sum, static_cast<T *>(w), A(alpha),
                                                                         A(beta));
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// ---- depthwise convolutions (one input channel per group, any number of filters per channel: MobileNet's and
// EfficientNet's): direct kernels. As im2col and a GEMM per group (hundreds of tiny GEMMs per layer) MobileNetV3's
// training step took 50 times as long as with cuDNN.


/**
 * A depthwise problem in 32 bit integers (sizes and strides below 2^31; elements of a tensor below 2^32): the kernels
 * took most of their time in 64 bit index arithmetic (divisions are calls of ~70 instructions)
 */
struct Depthwise {
    int n, c, k, kg;
    int in0, in1, in2, out0, out1, out2, f0, f1, f2;
    int pad0, pad1, pad2, stride0, stride1, stride2, dil0, dil1, dil2;
    int xs0, xs1, xs2, xs3, xs4, ys0, ys1, ys2, ys3, ys4, ws0, ws2, ws3, ws4;
    bool flip, unitStride;
};

bool depthwiseFits(const Shape &s) {
    const int64_t limit = INT32_MAX;
    for (int64_t v: {s.n, s.c, s.k, s.xs[0], s.xs[1], s.ys[0], s.ys[1], s.ws[0]}) {
        if (std::abs(v) > limit) return false;
    }
    return s.n * std::max(s.c, s.k) * std::max(s.pq, s.in[0] * s.in[1] * s.in[2]) < (1ll << 32);
}

bool depthwiseOf(const Shape &s) {
    return s.cg == 1 && s.groups == s.c && s.groups > 1 &&
           s.filter[0] * s.filter[1] * s.filter[2] <= DEPTHWISE_MAX_TAPS && depthwiseFits(s);
}

Depthwise depthwiseOfShape(const Shape &s) {
    Depthwise d{};
    d.n = int(s.n), d.c = int(s.c), d.k = int(s.k), d.kg = int(s.kg);
    d.in0 = int(s.in[0]), d.in1 = int(s.in[1]), d.in2 = int(s.in[2]);
    d.out0 = int(s.out[0]), d.out1 = int(s.out[1]), d.out2 = int(s.out[2]);
    d.f0 = int(s.filter[0]), d.f1 = int(s.filter[1]), d.f2 = int(s.filter[2]);
    d.pad0 = int(s.pad[0]), d.pad1 = int(s.pad[1]), d.pad2 = int(s.pad[2]);
    d.stride0 = int(s.stride[0]), d.stride1 = int(s.stride[1]), d.stride2 = int(s.stride[2]);
    d.dil0 = int(s.dilation[0]), d.dil1 = int(s.dilation[1]), d.dil2 = int(s.dilation[2]);
    d.xs0 = int(s.xs[0]), d.xs1 = int(s.xs[1]), d.xs2 = int(s.xs[2]), d.xs3 = int(s.xs[3]), d.xs4 = int(s.xs[4]);
    d.ys0 = int(s.ys[0]), d.ys1 = int(s.ys[1]), d.ys2 = int(s.ys[2]), d.ys3 = int(s.ys[3]), d.ys4 = int(s.ys[4]);
    d.ws0 = int(s.ws[0]), d.ws2 = int(s.ws[2]), d.ws3 = int(s.ws[3]), d.ws4 = int(s.ws[4]);
    d.flip = s.flip;
    d.unitStride = s.stride[0] == 1 && s.stride[1] == 1 && s.stride[2] == 1;
    return d;
}

// an element's (image, channel, position) from its index, channels fastest for channels-last tensors (memory order)
__device__ inline void decompose(unsigned e, unsigned channels, unsigned positions, bool channelsLast, int &n,
                                 int &ch, int &pos) {
    if (channelsLast) {
        ch = int(e % channels);
        const unsigned rest = e / channels;
        pos = int(rest % positions);
        n = int(rest / positions);
    } else {
        pos = int(e % positions);
        const unsigned rest = e / positions;
        ch = int(rest % channels);
        n = int(rest / channels);
    }
}

// a filter tap's offset in w (flipped for CUDNN_CONVOLUTION)
__device__ inline long long tapOffset(const Depthwise &d, int f0, int f1, int f2) {
    if (d.flip) {
        f0 = d.f0 - 1 - f0;
        f1 = d.f1 - 1 - f1;
        f2 = d.f2 - 1 - f2;
    }
    return f0 * d.ws2 + f1 * d.ws3 + f2 * d.ws4;
}

template<typename T, typename A>
__global__ void depthwiseForward(Depthwise d, const T *x, const T *w, T *y, A alpha, A beta, bool channelsLast) {
    const unsigned positions = unsigned(d.out0) * d.out1 * d.out2, total = unsigned(d.n) * d.k * positions;
    for (unsigned e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        int n, k, pos;
        decompose(e, d.k, positions, channelsLast, n, k, pos);
        const int o2 = pos % d.out2, rest = pos / d.out2, o1 = rest % d.out1, o0 = rest / d.out1;
        const T *xn = x + (long long) n * d.xs0 + (long long) (k / d.kg) * d.xs1;
        const T *wk = w + (long long) k * d.ws0;
        A acc = 0;
        for (int f0 = 0; f0 < d.f0; f0++) {
            const int i0 = o0 * d.stride0 - d.pad0 + f0 * d.dil0;
            if (i0 < 0 || i0 >= d.in0) continue;
            for (int f1 = 0; f1 < d.f1; f1++) {
                const int i1 = o1 * d.stride1 - d.pad1 + f1 * d.dil1;
                if (i1 < 0 || i1 >= d.in1) continue;
#pragma unroll 3
                for (int f2 = 0; f2 < d.f2; f2++) {
                    const int i2 = o2 * d.stride2 - d.pad2 + f2 * d.dil2;
                    if (i2 < 0 || i2 >= d.in2) continue;
                    acc += toAcc<T, A>(xn[(long long) i0 * d.xs2 + (long long) i1 * d.xs3 + (long long) i2 * d.xs4]) *
                           toAcc<T, A>(wk[tapOffset(d, f0, f1, f2)]);
                }
            }
        }
        T &out = y[(long long) n * d.ys0 + (long long) k * d.ys1 + (long long) o0 * d.ys2 + (long long) o1 * d.ys3 +
                   (long long) o2 * d.ys4];
        A value = alpha * acc;
        if (beta != A(0)) value += beta * toAcc<T, A>(out);
        out = fromAcc<T, A>(value);
    }
}

// the output position t / stride along a dimension (false where the tap doesn't reach an output)
__device__ inline bool outputOf(int t, int stride, int out, bool unit, int &o) {
    if (t < 0) return false;
    if (unit) {
        o = t;
    } else {
        if (t % stride != 0) return false;
        o = t / stride;
    }
    return o < out;
}

template<typename T, typename A>
__global__ void depthwiseBackwardData(Depthwise d, T *dx, const T *w, const T *dy, A alpha, A beta,
                                      bool channelsLast) {
    const unsigned positions = unsigned(d.in0) * d.in1 * d.in2, total = unsigned(d.n) * d.c * positions;
    for (unsigned e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        int n, c, pos;
        decompose(e, d.c, positions, channelsLast, n, c, pos);
        const int i2 = pos % d.in2, rest = pos / d.in2, i1 = rest % d.in1, i0 = rest / d.in1;
        A acc = 0;
        for (int m = 0; m < d.kg; m++) {
            const int k = c * d.kg + m;
            const T *dyk = dy + (long long) n * d.ys0 + (long long) k * d.ys1;
            const T *wk = w + (long long) k * d.ws0;
            for (int f0 = 0; f0 < d.f0; f0++) {
                int o0;
                if (!outputOf(i0 + d.pad0 - f0 * d.dil0, d.stride0, d.out0, d.unitStride, o0)) continue;
                for (int f1 = 0; f1 < d.f1; f1++) {
                    int o1;
                    if (!outputOf(i1 + d.pad1 - f1 * d.dil1, d.stride1, d.out1, d.unitStride, o1)) continue;
#pragma unroll 3
                    for (int f2 = 0; f2 < d.f2; f2++) {
                        int o2;
                        if (!outputOf(i2 + d.pad2 - f2 * d.dil2, d.stride2, d.out2, d.unitStride, o2)) continue;
                        acc += toAcc<T, A>(dyk[(long long) o0 * d.ys2 + (long long) o1 * d.ys3 + (long long) o2 * d.ys4]) *
                               toAcc<T, A>(wk[tapOffset(d, f0, f1, f2)]);
                    }
                }
            }
        }
        T &out = dx[(long long) n * d.xs0 + (long long) c * d.xs1 + (long long) i0 * d.xs2 + (long long) i1 * d.xs3 +
                    (long long) i2 * d.xs4];
        A value = alpha * acc;
        if (beta != A(0)) value += beta * toAcc<T, A>(out);
        out = fromAcc<T, A>(value);
    }
}

/**
 * Weight gradients: acc[k][tap] += sum over images and output positions of dy * x. Threads sum their share of the
 * positions for every tap in registers and add the sums to acc (float atomics). Channels-last tensors: blocks of 32
 * filters x 8 rows of positions (coalesced along the channels); others: a block per filter and slice of the positions.
 */
template<typename T, typename A>
__device__ inline void depthwiseTaps(const Depthwise &d, const T *x, const T *dy, int k, int n, int pos, A *sums) {
    const int o2 = pos % d.out2, rest = pos / d.out2, o1 = rest % d.out1, o0 = rest / d.out1;
    const A g = toAcc<T, A>(dy[(long long) n * d.ys0 + (long long) k * d.ys1 + (long long) o0 * d.ys2 +
                               (long long) o1 * d.ys3 + (long long) o2 * d.ys4]);
    const T *xn = x + (long long) n * d.xs0 + (long long) (k / d.kg) * d.xs1;
    int tap = 0;
    for (int f0 = 0; f0 < d.f0; f0++) {
        const int i0 = o0 * d.stride0 - d.pad0 + f0 * d.dil0;
        for (int f1 = 0; f1 < d.f1; f1++) {
            const int i1 = o1 * d.stride1 - d.pad1 + f1 * d.dil1;
            for (int f2 = 0; f2 < d.f2; f2++, tap++) {
                const int i2 = o2 * d.stride2 - d.pad2 + f2 * d.dil2;
                if (i0 < 0 || i0 >= d.in0 || i1 < 0 || i1 >= d.in1 || i2 < 0 || i2 >= d.in2) continue;
                sums[tap] += g * toAcc<T, A>(xn[(long long) i0 * d.xs2 + (long long) i1 * d.xs3 + (long long) i2 * d.xs4]);
            }
        }
    }
}

template<typename T, typename A>
__global__ void depthwiseFilterLast(Depthwise d, const T *x, const T *dy, A *acc, int taps) {
    const int k = blockIdx.x * 32 + threadIdx.x;
    if (k >= d.k) return;
    A sums[DEPTHWISE_MAX_TAPS] = {};
    const unsigned positions = unsigned(d.out0) * d.out1 * d.out2, rows = unsigned(d.n) * positions;
    for (unsigned r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += gridDim.y * blockDim.y) {
        depthwiseTaps<T, A>(d, x, dy, k, int(r / positions), int(r % positions), sums);
    }
    for (int t = 0; t < taps; t++) atomicAdd(&acc[k * taps + t], sums[t]);
}

template<typename T, typename A>
__global__ void depthwiseFilterPlanes(Depthwise d, const T *x, const T *dy, A *acc, int taps) {
    const int k = blockIdx.x;
    A sums[DEPTHWISE_MAX_TAPS] = {};
    const unsigned positions = unsigned(d.out0) * d.out1 * d.out2, rows = unsigned(d.n) * positions;
    for (unsigned r = blockIdx.y * blockDim.x + threadIdx.x; r < rows; r += gridDim.y * blockDim.x) {
        depthwiseTaps<T, A>(d, x, dy, k, int(r / positions), int(r % positions), sums);
    }
    // (warp sums first: a block's 256 threads would add to the same few addresses)
    for (int t = 0; t < taps; t++) {
        A v = sums[t];
        for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffffu, v, offset);
        if (threadIdx.x % 32 == 0) atomicAdd(&acc[k * taps + t], v);
    }
}

// ---- channels-last depthwise convolutions with one filter per channel: V adjacent channels per thread (16 bytes:
// 8 half / bfloat16, 4 float values), the filters transposed to [tap][channel] so their loads are vectors too. With a
// channel per thread, a warp moved 64 bytes per load and the kernels took 3 to 10 times as long as cuDNN's.

template<typename T, int V>
struct alignas(sizeof(T) * V) Lanes {
    T v[V];
};

// wt[tap * K + k] = w[k][tap] (flipped for CUDNN_CONVOLUTION)
template<typename T>
__global__ void depthwiseTransposeFilters(Depthwise d, const T *w, T *wt) {
    const int taps = d.f0 * d.f1 * d.f2;
    GRID_LOOP(e, (long long) d.k * taps) {
        const int tap = int(e / d.k), k = int(e % d.k);
        const int f2 = tap % d.f2, f1 = (tap / d.f2) % d.f1, f0 = tap / (d.f1 * d.f2);
        wt[e] = w[(long long) k * d.ws0 + tapOffset(d, f0, f1, f2)];
    }
}

template<typename T, typename A, int V>
__global__ void depthwiseForwardLanes(Depthwise d, const T *x, const T *wt, T *y, A alpha, A beta) {
    using L = Lanes<T, V>;
    const unsigned groups = unsigned(d.k / V), positions = unsigned(d.out0) * d.out1 * d.out2;
    const unsigned total = unsigned(d.n) * positions * groups;
    for (unsigned e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        const int k0 = int(e % groups) * V;
        const unsigned rest = e / groups;
        const int pos = int(rest % positions), n = int(rest / positions);
        const int o2 = pos % d.out2, r = pos / d.out2, o1 = r % d.out1, o0 = r / d.out1;
        const T *xn = x + (long long) n * d.xs0 + k0;
        A acc[V] = {};
        int tap = 0;
        for (int f0 = 0; f0 < d.f0; f0++) {
            const int i0 = o0 * d.stride0 - d.pad0 + f0 * d.dil0;
            for (int f1 = 0; f1 < d.f1; f1++) {
                const int i1 = o1 * d.stride1 - d.pad1 + f1 * d.dil1;
                for (int f2 = 0; f2 < d.f2; f2++, tap++) {
                    const int i2 = o2 * d.stride2 - d.pad2 + f2 * d.dil2;
                    if (i0 < 0 || i0 >= d.in0 || i1 < 0 || i1 >= d.in1 || i2 < 0 || i2 >= d.in2) continue;
                    const L xv = *reinterpret_cast<const L *>(xn + (long long) i0 * d.xs2 + (long long) i1 * d.xs3 +
                                                              (long long) i2 * d.xs4);
                    const L wv = *reinterpret_cast<const L *>(wt + (long long) tap * d.k + k0);
#pragma unroll
                    for (int i = 0; i < V; i++) acc[i] += toAcc<T, A>(xv.v[i]) * toAcc<T, A>(wv.v[i]);
                }
            }
        }
        T *out = y + (long long) n * d.ys0 + k0 + (long long) o0 * d.ys2 + (long long) o1 * d.ys3 +
                 (long long) o2 * d.ys4;
        L prior{}, result;
        if (beta != A(0)) prior = *reinterpret_cast<const L *>(out);
#pragma unroll
        for (int i = 0; i < V; i++) {
            A value = alpha * acc[i];
            if (beta != A(0)) value += beta * toAcc<T, A>(prior.v[i]);
            result.v[i] = fromAcc<T, A>(value);
        }
        *reinterpret_cast<L *>(out) = result;
    }
}

template<typename T, typename A, int V>
__global__ void depthwiseBackwardDataLanes(Depthwise d, T *dx, const T *wt, const T *dy, A alpha, A beta) {
    using L = Lanes<T, V>;
    const unsigned groups = unsigned(d.c / V), positions = unsigned(d.in0) * d.in1 * d.in2;
    const unsigned total = unsigned(d.n) * positions * groups;
    for (unsigned e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += gridDim.x * blockDim.x) {
        const int c0 = int(e % groups) * V;
        const unsigned rest = e / groups;
        const int pos = int(rest % positions), n = int(rest / positions);
        const int i2 = pos % d.in2, r = pos / d.in2, i1 = r % d.in1, i0 = r / d.in1;
        const T *dyn = dy + (long long) n * d.ys0 + c0;
        A acc[V] = {};
        int tap = 0;
        for (int f0 = 0; f0 < d.f0; f0++) {
            int o0;
            const bool in0 = outputOf(i0 + d.pad0 - f0 * d.dil0, d.stride0, d.out0, d.unitStride, o0);
            for (int f1 = 0; f1 < d.f1; f1++) {
                int o1;
                const bool in1 = in0 && outputOf(i1 + d.pad1 - f1 * d.dil1, d.stride1, d.out1, d.unitStride, o1);
                for (int f2 = 0; f2 < d.f2; f2++, tap++) {
                    int o2;
                    if (!in1 || !outputOf(i2 + d.pad2 - f2 * d.dil2, d.stride2, d.out2, d.unitStride, o2)) continue;
                    const L gv = *reinterpret_cast<const L *>(dyn + (long long) o0 * d.ys2 + (long long) o1 * d.ys3 +
                                                              (long long) o2 * d.ys4);
                    const L wv = *reinterpret_cast<const L *>(wt + (long long) tap * d.k + c0);
#pragma unroll
                    for (int i = 0; i < V; i++) acc[i] += toAcc<T, A>(gv.v[i]) * toAcc<T, A>(wv.v[i]);
                }
            }
        }
        T *out = dx + (long long) n * d.xs0 + c0 + (long long) i0 * d.xs2 + (long long) i1 * d.xs3 +
                 (long long) i2 * d.xs4;
        L prior{}, result;
        if (beta != A(0)) prior = *reinterpret_cast<const L *>(out);
#pragma unroll
        for (int i = 0; i < V; i++) {
            A value = alpha * acc[i];
            if (beta != A(0)) value += beta * toAcc<T, A>(prior.v[i]);
            result.v[i] = fromAcc<T, A>(value);
        }
        *reinterpret_cast<L *>(out) = result;
    }
}

/**
 * weight gradients: V channels x TAPS taps of sums per thread; blocks of channel groups (as many as the filters have,
 * up to 32) x rows of positions (256 threads), the rows' sums added up in shared memory before the atomics (each
 * thread's 72 atomics on a few hundred addresses, and 30 of 32 threads idle for 16 channels, took 10 times as long)
 */
template<typename T, typename A, int V, int TAPS>
__global__ void depthwiseFilterLanes(Depthwise d, const T *x, const T *dy, A *acc) {
    using L = Lanes<T, V>;
    __shared__ A shared[256 * V];
    const int k0 = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    const bool active = k0 < d.k;
    A sums[TAPS][V] = {};
    const unsigned positions = unsigned(d.out0) * d.out1 * d.out2, rows = unsigned(d.n) * positions;
    for (unsigned row = blockIdx.y * blockDim.y + threadIdx.y; active && row < rows; row += gridDim.y * blockDim.y) {
        const int pos = int(row % positions), n = int(row / positions);
        const int o2 = pos % d.out2, r = pos / d.out2, o1 = r % d.out1, o0 = r / d.out1;
        const L gv = *reinterpret_cast<const L *>(dy + (long long) n * d.ys0 + k0 + (long long) o0 * d.ys2 +
                                                  (long long) o1 * d.ys3 + (long long) o2 * d.ys4);
        const T *xn = x + (long long) n * d.xs0 + k0;
        int tap = 0;
#pragma unroll
        for (int t = 0; t < TAPS; t++) {
            const int f2 = t % d.f2, f1 = (t / d.f2) % d.f1, f0 = t / (d.f1 * d.f2);
            const int i0 = o0 * d.stride0 - d.pad0 + f0 * d.dil0;
            const int i1 = o1 * d.stride1 - d.pad1 + f1 * d.dil1;
            const int i2 = o2 * d.stride2 - d.pad2 + f2 * d.dil2;
            if (t >= d.f0 * d.f1 * d.f2 || i0 < 0 || i0 >= d.in0 || i1 < 0 || i1 >= d.in1 || i2 < 0 || i2 >= d.in2) {
                continue;
            }
            const L xv = *reinterpret_cast<const L *>(xn + (long long) i0 * d.xs2 + (long long) i1 * d.xs3 +
                                                      (long long) i2 * d.xs4);
#pragma unroll
            for (int i = 0; i < V; i++) sums[t][i] += toAcc<T, A>(gv.v[i]) * toAcc<T, A>(xv.v[i]);
        }
        (void) tap;
    }
    const int taps = d.f0 * d.f1 * d.f2;
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < TAPS; t++) {
        if (t >= taps) break;
#pragma unroll
        for (int i = 0; i < V; i++) shared[thread * V + i] = sums[t][i];
        __syncthreads();
        if (threadIdx.y == 0 && active) {
            for (int r = 1; r < int(blockDim.y); r++) {
#pragma unroll
                for (int i = 0; i < V; i++) sums[t][i] += shared[(r * blockDim.x + threadIdx.x) * V + i];
            }
#pragma unroll
            for (int i = 0; i < V; i++) atomicAdd(&acc[(k0 + i) * taps + t], sums[t][i]);
        }
        __syncthreads();
    }
}

// dw[k][tap] = alpha acc + beta dw
template<typename T, typename A>
__global__ void depthwiseStoreFilter(Depthwise d, const A *acc, T *w, A alpha, A beta, int taps) {
    GRID_LOOP(e, (long long) d.k * taps) {
        const int k = int(e / taps), tap = int(e % taps);
        const int f2 = tap % d.f2, f1 = (tap / d.f2) % d.f1, f0 = tap / (d.f1 * d.f2);
        T &out = w[(long long) k * d.ws0 + tapOffset(d, f0, f1, f2)];
        A value = alpha * acc[e];
        if (beta != A(0)) value += beta * toAcc<T, A>(out);
        out = fromAcc<T, A>(value);
    }
}

// backward filter: the float (double) sums; else the filters transposed (at most 8 bytes per value)
size_t depthwiseWorkspace(const Shape &s) {
    return (size_t(s.k) * s.filter[0] * s.filter[1] * s.filter[2] * 8 + 255) / 256 * 256;
}

// channels per thread of the channels-last kernels: 16 bytes of channels where the tensors allow (one filter per
// channel, channels-last x and y, aligned), else 1
template<typename T>
int depthwiseLanes(const Shape &s, const void *x, const void *y) {
    const int v = int(16 / sizeof(T));
    if (s.kg != 1 || s.xs[1] != 1 || s.ys[1] != 1 || s.c % v != 0) return 1;
    for (int64_t stride: {s.xs[0], s.xs[2], s.xs[3], s.xs[4], s.ys[0], s.ys[2], s.ys[3], s.ys[4]}) {
        if (stride % v != 0) return 1;
    }
    if (reinterpret_cast<uintptr_t>(x) % 16 != 0 || reinterpret_cast<uintptr_t>(y) % 16 != 0) return 1;
    return v;
}

template<typename T, typename A>
cudnnStatus_t runDepthwise(ConvKind kind, cudaStream_t stream, const Shape &s, void *x, void *w, void *y, A alpha,
                           A beta, char *workspace) {
    const Depthwise d = depthwiseOfShape(s);
    // (channels-last: the channels of a position adjacent)
    const bool last = kind == ConvKind::BACKWARD_DATA ? s.xs[1] == 1 : s.ys[1] == 1;
    const int64_t elements = kind == ConvKind::BACKWARD_DATA ? s.n * s.c * s.in[0] * s.in[1] * s.in[2]
                                                             : s.n * s.k * s.pq;
    const unsigned blocks = unsigned(std::max<int64_t>(1, std::min<int64_t>((elements + 255) / 256, 1 << 20)));
    constexpr int V = int(16 / sizeof(T));
    const int lanes = depthwiseLanes<T>(s, x, y);
    const int taps = int(s.filter[0] * s.filter[1] * s.filter[2]);
    if (lanes == V && kind != ConvKind::BACKWARD_FILTER) {
        T *wt = reinterpret_cast<T *>(workspace);
        depthwiseTransposeFilters<T><<<gridFor(s.k * taps), 256, 0, stream>>>(d, static_cast<const T *>(w), wt);
        const unsigned vblocks = unsigned(std::max<int64_t>(1, std::min<int64_t>((elements / V + 255) / 256, 1 << 20)));
        if (kind == ConvKind::FORWARD) {
            depthwiseForwardLanes<T, A, V><<<vblocks, 256, 0, stream>>>(d, static_cast<const T *>(x), wt,
                                                                        static_cast<T *>(y), alpha, beta);
        } else {
            depthwiseBackwardDataLanes<T, A, V><<<vblocks, 256, 0, stream>>>(d, static_cast<T *>(x), wt,
                                                                             static_cast<const T *>(y), alpha, beta);
        }
        return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
    }
    switch (kind) {
        case ConvKind::FORWARD:
            depthwiseForward<T, A><<<blocks, 256, 0, stream>>>(d, static_cast<const T *>(x), static_cast<const T *>(w),
                                                               static_cast<T *>(y), alpha, beta, last);
            break;
        case ConvKind::BACKWARD_DATA:
            depthwiseBackwardData<T, A><<<blocks, 256, 0, stream>>>(d, static_cast<T *>(x), static_cast<const T *>(w),
                                                                    static_cast<const T *>(y), alpha, beta, last);
            break;
        case ConvKind::BACKWARD_FILTER: {
            A *acc = reinterpret_cast<A *>(workspace);
            if (cudaMemsetAsync(acc, 0, size_t(s.k) * taps * sizeof(A), stream) != cudaSuccess) {
                return CUDNN_STATUS_EXECUTION_FAILED;
            }
            int device = 0, sms = 1;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
            const int64_t rows = s.n * s.pq;
            if (lanes == V && taps <= 25) {
                // (V channels per thread for 3 x 3 filters, V / 2 for up to 5 x 5)
                const int v = taps <= 9 ? V : V / 2;
                const int64_t groups = s.k / v;
                const unsigned width = unsigned(std::min<int64_t>(groups, 32)), height = 256 / width;
                const int64_t tiles = (groups + width - 1) / width;
                const unsigned slices = unsigned(std::clamp<int64_t>(4 * sms / tiles, 1, (rows + height - 1) / height));
                const dim3 grid(unsigned(tiles), slices), block(width, height);
                if (taps <= 9) {
                    depthwiseFilterLanes<T, A, V, 9><<<grid, block, 0, stream>>>(d, static_cast<const T *>(x),
                                                                                static_cast<const T *>(y), acc);
                } else {
                    depthwiseFilterLanes<T, A, V / 2, 25><<<grid, block, 0, stream>>>(d, static_cast<const T *>(x),
                                                                                     static_cast<const T *>(y), acc);
                }
            } else if (last) {
                const int64_t tiles = (s.k + 31) / 32;
                const unsigned slices = unsigned(std::clamp<int64_t>(8 * sms / tiles, 1, (rows + 63) / 64));
                depthwiseFilterLast<T, A><<<dim3(unsigned(tiles), slices), dim3(32, 8), 0, stream>>>(
                    d, static_cast<const T *>(x), static_cast<const T *>(y), acc, taps);
            } else {
                const unsigned slices = unsigned(std::clamp<int64_t>(8 * sms / s.k, 1, (rows + 255) / 256));
                depthwiseFilterPlanes<T, A><<<dim3(unsigned(s.k), slices), 256, 0, stream>>>(
                    d, static_cast<const T *>(x), static_cast<const T *>(y), acc, taps);
            }
            depthwiseStoreFilter<T, A><<<gridFor(s.k * taps), 256, 0, stream>>>(d, acc, static_cast<T *>(w), alpha,
                                                                                beta, taps);
            break;
        }
    }
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

} // namespace

size_t convWorkspaceSize(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g) {
    if (convCheck(kind, x, w, y, g) != CUDNN_STATUS_SUCCESS) {
        return 0;
    }
    const Shape s = shapeOf(x, w, y, g);
    if (implicitAllowed() && depthwiseOf(s)) {
        return depthwiseWorkspace(s);
    }
    Pointwise pw;
    if (implicitAllowed() && pointwiseOf(s, pw)) {
        return pointwiseWorkspace(kind, s, x.type == CUDNN_DATA_DOUBLE);
    }
    if (implicitAllowed()) {
        const size_t implicit = implicitConvWorkspace(kind, x, w, y, g);
        if (implicit != 0) {
            return implicit;
        }
    }
    return layoutOf(kind, s, x.type, x.type == CUDNN_DATA_DOUBLE).total;
}

cudnnStatus_t convRun(ConvKind kind, cudaStream_t stream, const Tensor &x, void *xData, const Tensor &w, void *wData,
                      const Tensor &y, void *yData, const ConvGeometry &g, double alpha, double beta, void *workspace,
                      size_t workspaceSize) {
    const cudnnStatus_t check = convCheck(kind, x, w, y, g);
    if (check != CUDNN_STATUS_SUCCESS) {
        return check;
    }
    const Shape s = shapeOf(x, w, y, g);
    if (s.n == 0 || s.k == 0 || s.c == 0 || s.pq == 0) {
        return CUDNN_STATUS_SUCCESS;
    }
    const size_t needed = convWorkspaceSize(kind, x, w, y, g);
    if (workspaceSize < needed || (workspace == nullptr && needed != 0)) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (implicitAllowed() && depthwiseOf(s)) {
        char *ws = static_cast<char *>(workspace);
        switch (x.type) {
            case CUDNN_DATA_HALF:
                return runDepthwise<__half, float>(kind, stream, s, xData, wData, yData, float(alpha), float(beta), ws);
            case CUDNN_DATA_BFLOAT16:
                return runDepthwise<__nv_bfloat16, float>(kind, stream, s, xData, wData, yData, float(alpha),
                                                          float(beta), ws);
            case CUDNN_DATA_DOUBLE:
                return runDepthwise<double, double>(kind, stream, s, xData, wData, yData, alpha, beta, ws);
            default:
                return runDepthwise<float, float>(kind, stream, s, xData, wData, yData, float(alpha), float(beta), ws);
        }
    }
    Pointwise pw;
    if (implicitAllowed() && pointwiseOf(s, pw)) {
        char *ws = static_cast<char *>(workspace);
        switch (x.type) {
            case CUDNN_DATA_HALF:
                return runPointwise<__half, float>(kind, stream, s, pw, x.type, xData, wData, yData, alpha, beta, ws);
            case CUDNN_DATA_BFLOAT16:
                return runPointwise<__nv_bfloat16, float>(kind, stream, s, pw, x.type, xData, wData, yData, alpha,
                                                          beta, ws);
            case CUDNN_DATA_DOUBLE:
                return runPointwise<double, double>(kind, stream, s, pw, x.type, xData, wData, yData, alpha, beta, ws);
            default:
                return runPointwise<float, float>(kind, stream, s, pw, x.type, xData, wData, yData, alpha, beta, ws);
        }
    }
    if (implicitAllowed() && implicitConvWorkspace(kind, x, w, y, g) != 0) {
        return implicitConvRun(kind, stream, x, xData, w, wData, y, yData, g, alpha, beta, workspace);
    }
    char *ws = static_cast<char *>(workspace);
    switch (x.type) {
        case CUDNN_DATA_HALF:
            return run<__half, float>(kind, stream, s, x.type, xData, wData, yData, float(alpha), float(beta), ws);
        case CUDNN_DATA_BFLOAT16:
            return run<__nv_bfloat16, float>(kind, stream, s, x.type, xData, wData, yData, float(alpha), float(beta),
                                             ws);
        case CUDNN_DATA_DOUBLE:
            return run<double, double>(kind, stream, s, x.type, xData, wData, yData, alpha, beta, ws);
        default:
            return run<float, float>(kind, stream, s, x.type, xData, wData, yData, float(alpha), float(beta), ws);
    }
}

} // namespace librecuda_dnn
