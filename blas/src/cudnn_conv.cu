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
struct Shape {
    int64_t n, c, k, groups, cg, kg;
    int64_t in[3], filter[3], out[3];
    int64_t xs[5], ws[5], ys[5];  // strides: N / K, C, spatial (3)
    int64_t kd, pq;
    int64_t pad[3], stride[3], dilation[3];
    bool flip;
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
cudnnStatus_t gemm(cudaStream_t stream, cudnnDataType_t type, bool doubles, cublasOperation_t ta, cublasOperation_t tb,
                   int64_t m, int64_t n, int64_t k, const void *a, int64_t lda, const void *b, int64_t ldb, void *d,
                   int64_t ldd, bool accumulate) {
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
        for (int64_t g = 0; g < s.groups; g++) {
            cudnnStatus_t status;
            const T *wg = packed_w + g * s.kg * s.kd;
            if (kind == ConvKind::FORWARD) {
                // out (rows g*Kg.., ld K) = W_g (Kg x Kd) cols_g
                status = gemm(stream, type, doubles, CUBLAS_OP_T, CUBLAS_OP_N, s.kg, columns, s.kd, wg, s.kd,
                              reinterpret_cast<T *>(cols) + g * columns * s.kd, s.kd,
                              reinterpret_cast<A *>(outputs) + g * s.kg, s.k, false);
            } else if (kind == ConvKind::BACKWARD_DATA) {
                // cols_g (Kd x PQ*n) = W_g^T dy_g
                status = gemm(stream, type, doubles, CUBLAS_OP_N, CUBLAS_OP_N, s.kd, columns, s.kg, wg, s.kd,
                              reinterpret_cast<T *>(outputs) + g * s.kg, s.k,
                              reinterpret_cast<A *>(cols) + g * columns * s.kd, s.kd, false);
            } else {
                // dw_g (Kg x Kd, rows g*Kg.. of acc, ld K) += dy_g cols_g^T
                status = gemm(stream, type, doubles, CUBLAS_OP_N, CUBLAS_OP_T, s.kg, s.kd, columns,
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

size_t convWorkspaceSize(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g) {
    if (convCheck(kind, x, w, y, g) != CUDNN_STATUS_SUCCESS) {
        return 0;
    }
    return layoutOf(kind, shapeOf(x, w, y, g), x.type, x.type == CUDNN_DATA_DOUBLE).total;
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
    if (workspace == nullptr || workspaceSize < convWorkspaceSize(kind, x, w, y, g)) {
        return CUDNN_STATUS_BAD_PARAM;
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
