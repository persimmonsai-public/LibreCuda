/**
 * LibreCUDA's cuBLAS beyond GEMM: level 1 vector operations, level 2 and 3 routines (built on the GEMM engine where
 * they can be: triangular solves recursively with GEMMs, the triangular / symmetric products through GEMMs on expanded
 * copies in scratch memory), and the batched LAPACK style functions (one thread block per matrix, lapack_device.cuh).
 */
#include "blas_helpers.cuh"
#include "cublas_handle.h"
#include "lapack_device.cuh"

#include <cublas_v2.h>

#include <cmath>

using namespace librecuda_blas;

namespace {

// ---------------------------------------------------------------------------------------------------------------------
// scalars

// a scalar argument: its value (host pointer mode) or its address on the device
template<typename S>
struct Arg {
    S value;
    const S *ptr;

    __device__ S get() const { return ptr != nullptr ? *ptr : value; }
};

template<typename S>
Arg<S> arg(cublasHandle_t handle, const S *p) {
    if (handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE) {
        return {S{}, p};
    }
    return {*p, nullptr};
}

template<typename S>
Scalar engineScalar(cublasHandle_t handle, const S *p) {
    return Scalar{p, dataType<S>(), handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE};
}

template<typename T>
const T *constant(T value) {
    // (the engine reads host scalars when the call is made)
    thread_local T storage[4];
    thread_local int next = 0;
    T *slot = &storage[next++ % 4];
    *slot = value;
    return slot;
}

template<typename T>
Scalar hostScalar(const T *value) {
    return Scalar{value, dataType<T>(), false};
}

// element i of a vector of n elements with increment inc (negative: the last element first)
__host__ __device__ inline long long element(long long i, long long n, long long inc) {
    return inc >= 0 ? i * inc : (n - 1 - i) * -inc;
}

// ---------------------------------------------------------------------------------------------------------------------
// level 1

template<typename T, typename S>
__global__ void axpyKernel(int n, Arg<S> alpha, const T *x, int incx, T *y, int incy) {
    const T a = make<T>(re(alpha.get()), im(alpha.get()));
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        T &yi = y[element(i, n, incy)];
        yi = add(yi, mul(a, x[element(i, n, incx)]));
    }
}

template<typename T, typename S>
__global__ void scalKernel(int n, Arg<S> alpha, T *x, int incx) {
    const S a = alpha.get();
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        T &xi = x[i * incx];
        if constexpr (std::is_same_v<S, T>) {
            xi = mul(a, xi);
        } else {
            xi = scale(xi, a);
        }
    }
}

template<typename T>
__global__ void copyKernel(int n, const T *x, int incx, T *y, int incy, bool swap) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        T &xi = const_cast<T &>(x[element(i, n, incx)]);
        T &yi = y[element(i, n, incy)];
        if (swap) {
            const T tmp = xi;
            xi = yi;
            yi = tmp;
        } else {
            yi = xi;
        }
    }
}

// x' = c x + s y, y' = c y - conj(s) x
template<typename T, typename S>
__global__ void rotKernel(int n, T *x, int incx, T *y, int incy, Arg<Real<T>> c, Arg<S> s) {
    const Real<T> cv = c.get();
    const T sv = make<T>(re(s.get()), im(s.get()));
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        T &xi = x[element(i, n, incx)];
        T &yi = y[element(i, n, incy)];
        const T xv = xi, yv = yi;
        xi = add(scale(xv, cv), mul(sv, yv));
        yi = sub(scale(yv, cv), mul(conjugate(sv), xv));
    }
}

enum class Reduction {
    NRM2, ASUM, IAMAX, IAMIN
};

template<typename T, typename R>
__global__ void reduceKernel(int n, const T *x, int incx, Reduction op, R *result) {
    __shared__ BlockShared s;
    if (op == Reduction::IAMAX || op == Reduction::IAMIN) {
        // the first index of the largest (smallest) |re| + |im|, 1-based
        double best = 0;
        int best_index = -1;
        for (int i = threadIdx.x; i < n; i += THREADS) {
            const double v = cabs1(x[static_cast<long long>(i) * incx]);
            const double key = op == Reduction::IAMAX ? v : -v;
            if (best_index < 0 || key > best) {
                best = key;
                best_index = i;
            }
        }
        const int index = blockArgMax(best, best_index, s);
        if (threadIdx.x == 0) {
            *result = R(index + 1);
        }
        return;
    }
    if (op == Reduction::ASUM) {
        double sum = 0;
        for (int i = threadIdx.x; i < n; i += THREADS) {
            sum += cabs1(x[static_cast<long long>(i) * incx]);
        }
        sum = blockSumReal(sum, s);
        if (threadIdx.x == 0) {
            *result = R(sum);
        }
        return;
    }
    // the Euclidean norm, scaled by the largest magnitude (no overflow / underflow of the squares)
    double largest = 0;
    for (int i = threadIdx.x; i < n; i += THREADS) {
        largest = fmax(largest, double(absval(x[static_cast<long long>(i) * incx])));
    }
    s.re[threadIdx.x] = largest;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s.re[threadIdx.x] = fmax(s.re[threadIdx.x], s.re[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    largest = s.re[0];
    __syncthreads();
    double sum = 0;
    if (largest > 0) {
        for (int i = threadIdx.x; i < n; i += THREADS) {
            const T v = x[static_cast<long long>(i) * incx];
            sum += norm2(make<T>(Real<T>(re(v) / largest), Real<T>(im(v) / largest)));
        }
    }
    sum = blockSumReal(sum, s);
    if (threadIdx.x == 0) {
        *result = R(largest * sqrt(sum));
    }
}

template<typename T, typename R>
cublasStatus_t reduce(cublasHandle_t handle, int n, const T *x, int incx, Reduction op, R *result) {
    CHECK_HANDLE(handle);
    if (result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool device = handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE;
    if (n <= 0 || incx <= 0) {
        if (device) {
            return cudaMemsetAsync(result, 0, sizeof(R), handle->stream) == cudaSuccess
                   ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
        }
        *result = R(0);
        return CUBLAS_STATUS_SUCCESS;
    }
    R *out = device ? result : static_cast<R *>(librecuda_blas_scratch(handle->stream, sizeof(R)));
    if (out == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    reduceKernel<<<1, THREADS, 0, handle->stream>>>(n, x, incx, op, out);
    if (launched() != CUBLAS_STATUS_SUCCESS) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    if (!device && (cudaMemcpyAsync(result, out, sizeof(R), cudaMemcpyDeviceToHost, handle->stream) != cudaSuccess ||
                    cudaStreamSynchronize(handle->stream) != cudaSuccess)) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

template<typename T, typename S>
cublasStatus_t axpy(cublasHandle_t handle, int n, const S *alpha, const T *x, int incx, T *y, int incy) {
    CHECK_HANDLE(handle);
    if (alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    axpyKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, arg(handle, alpha), x, incx, y, incy);
    return launched();
}

template<typename T, typename S>
cublasStatus_t scal(cublasHandle_t handle, int n, const S *alpha, T *x, int incx) {
    CHECK_HANDLE(handle);
    if (alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    scalKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, arg(handle, alpha), x, incx);
    return launched();
}

template<typename T>
cublasStatus_t copy(cublasHandle_t handle, int n, const T *x, int incx, T *y, int incy, bool swap) {
    CHECK_HANDLE(handle);
    if (n <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    copyKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, x, incx, y, incy, swap);
    return launched();
}

template<typename T, typename S>
cublasStatus_t rot(cublasHandle_t handle, int n, T *x, int incx, T *y, int incy, const Real<T> *c, const S *s) {
    CHECK_HANDLE(handle);
    if (c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    rotKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, x, incx, y, incy, arg(handle, c), arg(handle, s));
    return launched();
}

// Givens rotation (reference BLAS rotg)
template<typename T>
__host__ __device__ void rotgCompute(T *a, T *b, Real<T> *c, T *s) {
    using R = Real<T>;
    if constexpr (IS_COMPLEX<T>) {
        const R abs_a = absval(*a), abs_b = absval(*b);
        if (abs_a == 0) {
            *c = 0;
            *s = make<T>(1);
            *a = *b;
            return;
        }
        const R scl = abs_a + abs_b;
        const R norm = scl * sqrt(norm2(scale(*a, 1 / scl)) + norm2(scale(*b, 1 / scl)));
        const T alpha = scale(*a, 1 / abs_a);
        *c = abs_a / norm;
        *s = scale(mul(alpha, conjugate(*b)), 1 / norm);
        *a = scale(alpha, norm);
    } else {
        const R av = *a, bv = *b;
        const R roe = fabs(av) > fabs(bv) ? av : bv;
        const R scl = fabs(av) + fabs(bv);
        if (scl == 0) {
            *c = 1;
            *s = 0;
            *a = 0;
            *b = 0;
            return;
        }
        R r = scl * sqrt((av / scl) * (av / scl) + (bv / scl) * (bv / scl));
        r = roe < 0 ? -r : r;
        *c = av / r;
        *s = bv / r;
        R z = 1;
        if (fabs(av) > fabs(bv)) {
            z = *s;
        } else if (*c != 0) {
            z = 1 / *c;
        }
        *a = r;
        *b = z;
    }
}

template<typename T>
__global__ void rotgKernel(T *a, T *b, Real<T> *c, T *s) {
    rotgCompute(a, b, c, s);
}

template<typename T>
cublasStatus_t rotg(cublasHandle_t handle, T *a, T *b, Real<T> *c, T *s) {
    CHECK_HANDLE(handle);
    if (handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE) {
        rotgKernel<<<1, 1, 0, handle->stream>>>(a, b, c, s);
        return launched();
    }
    rotgCompute(a, b, c, s);
    return CUBLAS_STATUS_SUCCESS;
}

// modified Givens rotation (reference BLAS rotm)
template<typename T>
__global__ void rotmKernel(int n, T *x, int incx, T *y, int incy, const T *param_ptr, T p0, T p1, T p2, T p3, T p4) {
    T flag = p0, h11 = p1, h21 = p2, h12 = p3, h22 = p4;
    if (param_ptr != nullptr) {
        flag = param_ptr[0];
        h11 = param_ptr[1];
        h21 = param_ptr[2];
        h12 = param_ptr[3];
        h22 = param_ptr[4];
    }
    if (flag == -2) {
        return;
    }
    if (flag == 0) {
        h11 = h22 = 1;
    } else if (flag == 1) {
        h12 = 1;
        h21 = -1;
    }
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        T &xi = x[element(i, n, incx)];
        T &yi = y[element(i, n, incy)];
        const T w = xi, z = yi;
        xi = w * h11 + z * h12;
        yi = w * h21 + z * h22;
    }
}

template<typename T>
cublasStatus_t rotm(cublasHandle_t handle, int n, T *x, int incx, T *y, int incy, const T *param) {
    CHECK_HANDLE(handle);
    if (param == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE) {
        rotmKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, x, incx, y, incy, param, T(0), T(0), T(0), T(0), T(0));
    } else {
        rotmKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, x, incx, y, incy, static_cast<const T *>(nullptr),
                                                             param[0], param[1], param[2], param[3], param[4]);
    }
    return launched();
}

// ---------------------------------------------------------------------------------------------------------------------
// matrices in scratch memory

enum class Expand {
    // the other triangle mirrored (symmetric / Hermitian) or zero (triangular)
    SYMMETRIC, HERMITIAN, TRIANGULAR
};

template<typename T>
__global__ void expandKernel(int n, const T *a, long long lda, bool lower, Expand kind, bool unit, T *full) {
    const long long total = static_cast<long long>(n) * n;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % n), j = int(e / n);
        const bool stored = lower ? i >= j : i <= j;
        T value;
        if (i == j) {
            value = unit && kind == Expand::TRIANGULAR ? make<T>(1)
                    : kind == Expand::HERMITIAN ? make<T>(re(a[i + j * lda])) : a[i + j * lda];
        } else if (stored) {
            value = a[i + j * lda];
        } else if (kind == Expand::TRIANGULAR) {
            value = make<T>(0);
        } else {
            value = a[j + i * lda];
            if (kind == Expand::HERMITIAN) {
                value = conjugate(value);
            }
        }
        full[e] = value;
    }
}

template<typename T>
T *expand(cudaStream_t stream, int n, const T *a, long long lda, bool lower, Expand kind, bool unit, T *full) {
    expandKernel<<<blocksFor(static_cast<long long>(n) * n), 256, 0, stream>>>(n, a, lda, lower, kind, unit, full);
    return full;
}

// C = alpha W1 + alphaSecond W2 + beta C on the triangle of C (n x n); W2 may be nullptr; Hermitian: real diagonal
template<typename T, typename S, typename B>
__global__ void triangleCombine(int n, bool lower, Arg<S> alpha, const T *w1, Arg<S> alpha2, bool conjAlpha2,
                                const T *w2, Arg<B> beta, T *c, long long ldc, bool hermitian) {
    const S a = alpha.get();
    S a2 = w2 != nullptr ? alpha2.get() : S{};
    if (conjAlpha2) {
        a2 = conjugate(a2);
    }
    const B bv = beta.get();
    const long long total = static_cast<long long>(n) * n;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % n), j = int(e / n);
        if (lower ? i < j : i > j) {
            continue;
        }
        auto times = [](S s, T v) {
            if constexpr (std::is_same_v<S, T>) {
                return mul(s, v);
            } else {
                return scale(v, s);
            }
        };
        T value = times(a, w1[e]);
        if (w2 != nullptr) {
            value = add(value, times(a2, w2[e]));
        }
        T &ce = c[i + j * ldc];
        if (!isZero(make<T>(re(bv), im(bv)))) {
            T scaled;
            if constexpr (std::is_same_v<B, T>) {
                scaled = mul(bv, ce);
            } else {
                scaled = scale(ce, bv);
            }
            value = add(value, scaled);
        }
        if (hermitian && i == j) {
            value = make<T>(re(value));
        }
        ce = value;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// triangular solves

template<typename T, typename S>
__global__ void scaleMatrixKernel(int m, int n, Arg<S> alpha, T *b, long long ldb) {
    const S a = alpha.get();
    const long long total = static_cast<long long>(m) * n;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        T &v = b[e % m + (e / m) * ldb];
        // alpha 0: B is set to 0 without reading it
        v = isZero(a) ? make<T>(0) : mul(a, v);
    }
}

constexpr int TRSM_BASE = 64;
constexpr int TRSM_SLICE = 64;

template<typename T>
__global__ void trsmLeftKernel(bool lower, cublasOperation_t trans, bool unit, int n, int nrhs, const T *a,
                               long long lda, T *b, long long ldb) {
    const int first = blockIdx.x * TRSM_SLICE;
    trsmLeftBlock(lower, trans, unit, n, min(TRSM_SLICE, nrhs - first), a, lda, b + first * ldb, ldb);
}

template<typename T>
__global__ void trsmRightKernel(bool lower, cublasOperation_t trans, bool unit, int m, int n, const T *a,
                                long long lda, T *b, long long ldb) {
    const int first = blockIdx.x * TRSM_SLICE;
    trsmRightBlock(lower, trans, unit, min(TRSM_SLICE, m - first), n, a, lda, b + first, ldb);
}

/**
 * Solves op(A) X = B (left) or X op(A) = B in place: GEMMs between halves, one block kernels for small triangles
 */
template<typename T>
cublasStatus_t trsmSolve(cudaStream_t stream, bool left, bool lower, cublasOperation_t trans, bool unit, int m, int n,
                         const T *a, long long lda, T *b, long long ldb) {
    const int size = left ? m : n;
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (size <= TRSM_BASE) {
        if (left) {
            trsmLeftKernel<<<(n + TRSM_SLICE - 1) / TRSM_SLICE, THREADS, 0, stream>>>(lower, trans, unit, m, n, a,
                                                                                      lda, b, ldb);
        } else {
            trsmRightKernel<<<(m + TRSM_SLICE - 1) / TRSM_SLICE, THREADS, 0, stream>>>(lower, trans, unit, m, n, a,
                                                                                       lda, b, ldb);
        }
        return launched();
    }
    const int s1 = size / 2 / 32 * 32 > 0 ? size / 2 / 32 * 32 : size / 2, s2 = size - s1;
    const T one = make<T>(1), minus = make<T>(-1);
    const T *a11 = a, *a22 = a + s1 + s1 * lda;
    // the off-diagonal blocks of op(A): op(A)21 (s2 x s1) and op(A)12 (s1 x s2) with their operation
    const bool n_op = trans == CUBLAS_OP_N;
    const T *op21 = n_op ? a + s1 : a + s1 * lda;
    const T *op12 = n_op ? a + s1 * lda : a + s1;
    cublasStatus_t status;
    if (left) {
        T *b1 = b, *b2 = b + s1;
        const bool eff_lower = lower == n_op;
        if (eff_lower) {
            if ((status = trsmSolve(stream, true, lower, trans, unit, s1, n, a11, lda, b1, ldb))) return status;
            if ((status = gemm(stream, trans, CUBLAS_OP_N, s2, n, s1, minus, op21, lda, b1, ldb, one, b2, ldb)))
                return status;
            return trsmSolve(stream, true, lower, trans, unit, s2, n, a22, lda, b2, ldb);
        }
        if ((status = trsmSolve(stream, true, lower, trans, unit, s2, n, a22, lda, b2, ldb))) return status;
        if ((status = gemm(stream, trans, CUBLAS_OP_N, s1, n, s2, minus, op12, lda, b2, ldb, one, b1, ldb)))
            return status;
        return trsmSolve(stream, true, lower, trans, unit, s1, n, a11, lda, b1, ldb);
    }
    T *b1 = b, *b2 = b + s1 * ldb;
    const bool eff_upper = lower != n_op;
    if (eff_upper) {
        if ((status = trsmSolve(stream, false, lower, trans, unit, m, s1, a11, lda, b1, ldb))) return status;
        if ((status = gemm(stream, CUBLAS_OP_N, trans, m, s2, s1, minus, b1, ldb, op12, lda, one, b2, ldb)))
            return status;
        return trsmSolve(stream, false, lower, trans, unit, m, s2, a22, lda, b2, ldb);
    }
    if ((status = trsmSolve(stream, false, lower, trans, unit, m, s2, a22, lda, b2, ldb))) return status;
    if ((status = gemm(stream, CUBLAS_OP_N, trans, m, s1, s2, minus, b2, ldb, op21, lda, one, b1, ldb)))
        return status;
    return trsmSolve(stream, false, lower, trans, unit, m, s1, a11, lda, b1, ldb);
}

template<typename T>
cublasStatus_t trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const T *alpha, const T *a, int lda, T *b, int ldb) {
    CHECK_HANDLE(handle);
    const int k = side == CUBLAS_SIDE_LEFT ? m : n;
    if (m < 0 || n < 0 || lda < std::max(1, k) || ldb < std::max(1, m) || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const bool scaled = handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE || !(re(*alpha) == 1 && im(*alpha) == 0);
    if (scaled) {
        scaleMatrixKernel<<<blocksFor(static_cast<long long>(m) * n), 256, 0, handle->stream>>>(
            m, n, arg(handle, alpha), b, ldb);
        if (launched() != CUBLAS_STATUS_SUCCESS) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return trsmSolve(handle->stream, side == CUBLAS_SIDE_LEFT, uplo == CUBLAS_FILL_MODE_LOWER, trans,
                     diag == CUBLAS_DIAG_UNIT, m, n, a, lda, b, ldb);
}

// ---------------------------------------------------------------------------------------------------------------------
// level 3 through GEMMs

template<typename T>
cublasStatus_t trmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const T *alpha, const T *a, int lda, const T *b, int ldb,
                    T *c, int ldc) {
    CHECK_HANDLE(handle);
    const bool left = side == CUBLAS_SIDE_LEFT;
    const int k = left ? m : n;
    if (m < 0 || n < 0 || lda < std::max(1, k) || ldb < std::max(1, m) || ldc < std::max(1, m) || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    T *full = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * k * k));
    if (full == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    expand(handle->stream, k, a, lda, uplo == CUBLAS_FILL_MODE_LOWER, Expand::TRIANGULAR, diag == CUBLAS_DIAG_UNIT,
           full);
    const T zero = make<T>(0);
    const Scalar alpha_s = engineScalar(handle, alpha), zero_s = hostScalar(&zero);
    if (left) {
        return gemm(handle->stream, trans, CUBLAS_OP_N, m, n, m, alpha_s, full, k, b, ldb, zero_s, c, ldc);
    }
    return gemm(handle->stream, CUBLAS_OP_N, trans, m, n, n, alpha_s, b, ldb, full, k, zero_s, c, ldc);
}

template<typename T>
cublasStatus_t symm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const T *alpha,
                    const T *a, int lda, const T *b, int ldb, const T *beta, T *c, int ldc, bool hermitian) {
    CHECK_HANDLE(handle);
    const bool left = side == CUBLAS_SIDE_LEFT;
    const int k = left ? m : n;
    if (m < 0 || n < 0 || lda < std::max(1, k) || ldb < std::max(1, m) || ldc < std::max(1, m) || alpha == nullptr ||
        beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    T *full = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * k * k));
    if (full == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    expand(handle->stream, k, a, lda, uplo == CUBLAS_FILL_MODE_LOWER,
           hermitian ? Expand::HERMITIAN : Expand::SYMMETRIC, false, full);
    const Scalar alpha_s = engineScalar(handle, alpha), beta_s = engineScalar(handle, beta);
    if (left) {
        return gemm(handle->stream, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, alpha_s, full, k, b, ldb, beta_s, c, ldc);
    }
    return gemm(handle->stream, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha_s, b, ldb, full, k, beta_s, c, ldc);
}

/**
 * syrk / herk (b == nullptr) and syr2k / her2k on the triangle of C: products into scratch memory, then combined
 */
template<typename T, typename S, typename B>
cublasStatus_t rankK(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
                     const S *alpha, const T *a, int lda, const T *b, int ldb, const B *beta, T *c, int ldc,
                     bool hermitian) {
    CHECK_HANDLE(handle);
    const bool no_trans = trans == CUBLAS_OP_N;
    const int rows = no_trans ? n : k;
    if (n < 0 || k < 0 || lda < std::max(1, rows) || (b != nullptr && ldb < std::max(1, rows)) ||
        ldc < std::max(1, n) || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const size_t square = static_cast<size_t>(n) * n;
    T *w = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * square * (b != nullptr ? 2 : 1)));
    if (w == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    // W1 = A op(X)^T, X = B (2k) or A; W2 = B op(A)^T
    const cublasOperation_t t = hermitian ? CUBLAS_OP_C : CUBLAS_OP_T;
    const cublasOperation_t first = no_trans ? CUBLAS_OP_N : t, second = no_trans ? t : CUBLAS_OP_N;
    const T one = make<T>(1), zero = make<T>(0);
    const T *x = b != nullptr ? b : a;
    const int ldx = b != nullptr ? ldb : lda;
    cublasStatus_t status = gemm(handle->stream, first, second, n, n, k, one, a, lda, x, ldx, zero, w, n);
    if (status == CUBLAS_STATUS_SUCCESS && b != nullptr) {
        status = gemm(handle->stream, first, second, n, n, k, one, b, ldb, a, lda, zero, w + square, n);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    triangleCombine<<<blocksFor(square), 256, 0, handle->stream>>>(
        n, uplo == CUBLAS_FILL_MODE_LOWER, arg(handle, alpha), w, arg(handle, alpha), hermitian,
        b != nullptr ? w + square : nullptr, arg(handle, beta), c, ldc, hermitian);
    return launched();
}

// ---------------------------------------------------------------------------------------------------------------------
// level 2

template<typename T>
cublasStatus_t engineGemv(cudaStream_t stream, cublasOperation_t trans, int m, int n, Scalar alpha, const T *a,
                          long long lda, const T *x, int incx, Scalar beta, T *y, int incy) {
    const int rows = trans == CUBLAS_OP_N ? m : n, inner = trans == CUBLAS_OP_N ? n : m;
    if (rows == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const T *xs = x + (incx < 0 ? (1LL - inner) * incx : 0);
    T *ys = y + (incy < 0 ? (1LL - rows) * incy : 0);
    GemmParams p{};
    p.transa = trans;
    p.transb = CUBLAS_OP_N;
    p.m = rows;
    p.n = 1;
    p.k = inner;
    p.batch = 1;
    p.a = Operand{a, nullptr, dataType<T>(), lda, 0, 1};
    p.b = Operand{xs, nullptr, dataType<T>(), incx == 1 ? std::max(1, inner) : 1, 0, incx};
    p.c = Operand{ys, nullptr, dataType<T>(), incy == 1 ? rows : 1, 0, incy};
    p.d = p.c;
    p.compute = computeType<T>();
    p.alpha = alpha;
    p.beta = beta;
    p.stream = stream;
    return librecuda_blas_gemm(&p);
}

// A += alpha x op(y)^T (ger / geru: transpose, gerc: conjugate transpose)
template<typename T>
cublasStatus_t ger(cublasHandle_t handle, int m, int n, const T *alpha, const T *x, int incx, const T *y, int incy,
                   T *a, int lda, bool conj) {
    CHECK_HANDLE(handle);
    if (m < 0 || n < 0 || incx == 0 || incy == 0 || lda < std::max(1, m) || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const T one = make<T>(1);
    GemmParams p{};
    p.transa = CUBLAS_OP_N;
    p.transb = conj ? CUBLAS_OP_C : CUBLAS_OP_T;
    p.m = m;
    p.n = n;
    p.k = 1;
    p.batch = 1;
    p.a = Operand{x + (incx < 0 ? (1LL - m) * incx : 0), nullptr, dataType<T>(), m, 0, incx};
    p.b = Operand{y + (incy < 0 ? (1LL - n) * incy : 0), nullptr, dataType<T>(), 1, 0, incy};
    p.c = Operand{a, nullptr, dataType<T>(), lda, 0, 1};
    p.d = p.c;
    p.compute = computeType<T>();
    p.alpha = engineScalar(handle, alpha);
    p.beta = hostScalar(&one);
    p.stream = handle->stream;
    return librecuda_blas_gemm(&p);
}

template<typename T>
cublasStatus_t symv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const T *alpha, const T *a, int lda,
                    const T *x, int incx, const T *beta, T *y, int incy, bool hermitian) {
    CHECK_HANDLE(handle);
    if (n < 0 || lda < std::max(1, n) || incx == 0 || incy == 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    T *full = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * n * n));
    if (full == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    expand(handle->stream, n, a, lda, uplo == CUBLAS_FILL_MODE_LOWER,
           hermitian ? Expand::HERMITIAN : Expand::SYMMETRIC, false, full);
    return engineGemv(handle->stream, CUBLAS_OP_N, n, n, engineScalar(handle, alpha), full, n, x, incx,
                      engineScalar(handle, beta), y, incy);
}

// A += alpha x y^T + alpha y x^T (syr2), alpha x y^H + conj(alpha) y x^H (her2), alpha x x^T / x x^H (y nullptr)
template<typename T, typename S>
__global__ void rankUpdateKernel(int n, bool lower, Arg<S> alpha, const T *x, int incx, const T *y, int incy, T *a,
                                 long long lda, bool hermitian) {
    const S av = alpha.get();
    const long long total = static_cast<long long>(n) * n;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % n), j = int(e / n);
        if (lower ? i < j : i > j) {
            continue;
        }
        const T xi = x[element(i, n, incx)], xj = x[element(j, n, incx)];
        T value;
        if (y == nullptr) {
            value = mul(xi, hermitian ? conjugate(xj) : xj);
            if constexpr (std::is_same_v<S, T>) {
                value = mul(av, value);
            } else {
                value = scale(value, av);
            }
        } else {
            const T yi = y[element(i, n, incy)], yj = y[element(j, n, incy)];
            const T a1 = make<T>(re(av), im(av));
            const T a2 = hermitian ? conjugate(a1) : a1;
            value = add(mul(a1, mul(xi, hermitian ? conjugate(yj) : yj)),
                        mul(a2, mul(yi, hermitian ? conjugate(xj) : xj)));
        }
        T &ae = a[i + j * lda];
        ae = add(ae, value);
        if (hermitian && i == j) {
            ae = make<T>(re(ae));
        }
    }
}

template<typename T, typename S>
cublasStatus_t rankUpdate(cublasHandle_t handle, cublasFillMode_t uplo, int n, const S *alpha, const T *x, int incx,
                          const T *y, int incy, T *a, int lda, bool hermitian) {
    CHECK_HANDLE(handle);
    if (n < 0 || incx == 0 || (y != nullptr && incy == 0) || lda < std::max(1, n) || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    rankUpdateKernel<<<blocksFor(static_cast<long long>(n) * n), 256, 0, handle->stream>>>(
        n, uplo == CUBLAS_FILL_MODE_LOWER, arg(handle, alpha), x, incx, y, incy, a, lda, hermitian);
    return launched();
}

template<typename T>
cublasStatus_t trmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
                    int n, const T *a, int lda, T *x, int incx) {
    CHECK_HANDLE(handle);
    if (n < 0 || lda < std::max(1, n) || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    T *full = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * (size_t(n) * n + n)));
    if (full == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    T *copy_x = full + size_t(n) * n;
    expand(handle->stream, n, a, lda, uplo == CUBLAS_FILL_MODE_LOWER, Expand::TRIANGULAR, diag == CUBLAS_DIAG_UNIT,
           full);
    copyKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, x, incx, copy_x, 1, false);
    const T one = make<T>(1), zero = make<T>(0);
    return engineGemv(handle->stream, trans, n, n, hostScalar(&one), full, n, copy_x, 1, hostScalar(&zero), x, incx);
}

template<typename T>
cublasStatus_t trsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
                    int n, const T *a, int lda, T *x, int incx) {
    CHECK_HANDLE(handle);
    if (n < 0 || lda < std::max(1, n) || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    T *v = x;
    if (incx != 1) {
        v = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * n));
        if (v == nullptr) {
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        copyKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, x, incx, v, 1, false);
    }
    const cublasStatus_t status = trsmSolve(handle->stream, true, uplo == CUBLAS_FILL_MODE_LOWER, trans,
                                            diag == CUBLAS_DIAG_UNIT, n, 1, a, lda, v, n);
    if (status == CUBLAS_STATUS_SUCCESS && incx != 1) {
        copyKernel<<<blocksFor(n), 256, 0, handle->stream>>>(n, v, 1, x, incx, false);
        return launched();
    }
    return status;
}

// ---------------------------------------------------------------------------------------------------------------------
// batched LAPACK style functions (one block per matrix)

template<typename T>
__global__ void trsmBatchedKernel(bool left, bool lower, cublasOperation_t trans, bool unit, int m, int n,
                                  Arg<T> alpha, const T *const *a, int lda, T *const *b, int ldb) {
    T *bm = b[blockIdx.x];
    const T av = alpha.get();
    for (long long e = threadIdx.x; e < static_cast<long long>(m) * n; e += THREADS) {
        T &v = bm[e % m + (e / m) * ldb];
        v = isZero(av) ? make<T>(0) : mul(av, v);
    }
    __syncthreads();
    if (left) {
        trsmLeftBlock(lower, trans, unit, m, n, a[blockIdx.x], lda, bm, ldb);
    } else {
        trsmRightBlock(lower, trans, unit, m, n, a[blockIdx.x], lda, bm, ldb);
    }
}

template<typename T>
__global__ void getrfBatchedKernel(int n, T *const *a, int lda, int *pivots, int *info) {
    __shared__ BlockShared s;
    if (threadIdx.x == 0) {
        info[blockIdx.x] = 0;
    }
    __syncthreads();
    luBlock(a[blockIdx.x], n, n, lda, pivots != nullptr ? pivots + static_cast<long long>(blockIdx.x) * n : nullptr, 0,
            info + blockIdx.x, 0, s);
}

template<typename T>
__global__ void getrsBatchedKernel(cublasOperation_t trans, int n, int nrhs, const T *const *a, int lda,
                                   const int *pivots, T *const *b, int ldb) {
    getrsBlock(trans, n, nrhs, a[blockIdx.x], lda, pivots + static_cast<long long>(blockIdx.x) * n, b[blockIdx.x],
               ldb);
}

template<typename T>
__global__ void geqrfBatchedKernel(int m, int n, T *const *a, int lda, T *const *tau) {
    __shared__ BlockShared s;
    qrBlock(m, n, a[blockIdx.x], lda, tau[blockIdx.x], s);
}

// least squares min ||A X - C|| (m >= n): QR of A, C = Q^H C, R X = C; info: the first zero diagonal element of R
template<typename T>
__global__ void gelsBatchedKernel(int m, int n, int nrhs, T *const *a, int lda, T *const *c, int ldc, T *tau,
                                  int *info) {
    __shared__ BlockShared s;
    T *am = a[blockIdx.x];
    T *taum = tau + static_cast<long long>(blockIdx.x) * n;
    qrBlock(m, n, am, lda, taum, s);
    applyQBlock(true, true, m, nrhs, n, am, lda, taum, c[blockIdx.x], ldc);
    __syncthreads();
    if (threadIdx.x == 0 && info != nullptr) {
        int value = 0;
        for (int i = 0; i < n && value == 0; i++) {
            if (isZero(at(am, lda, i, i))) {
                value = i + 1;
            }
        }
        info[blockIdx.x] = value;
    }
    trsmLeftBlock(false, CUBLAS_OP_N, false, n, nrhs, am, lda, c[blockIdx.x], ldc);
}

template<typename T>
cublasStatus_t trsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const T *alpha,
                           const T *const a[], int lda, T *const b[], int ldb, int batch) {
    CHECK_HANDLE(handle);
    const int k = side == CUBLAS_SIDE_LEFT ? m : n;
    if (m < 0 || n < 0 || batch < 0 || lda < std::max(1, k) || ldb < std::max(1, m) || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0 || batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    trsmBatchedKernel<<<batch, THREADS, 0, handle->stream>>>(side == CUBLAS_SIDE_LEFT,
                                                             uplo == CUBLAS_FILL_MODE_LOWER, trans,
                                                             diag == CUBLAS_DIAG_UNIT, m, n, arg(handle, alpha), a,
                                                             lda, b, ldb);
    return launched();
}

template<typename T>
cublasStatus_t getrfBatched(cublasHandle_t handle, int n, T *const a[], int lda, int *pivots, int *info, int batch) {
    CHECK_HANDLE(handle);
    if (n < 0 || lda < std::max(1, n) || batch < 0 || info == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    getrfBatchedKernel<<<batch, THREADS, 0, handle->stream>>>(n, a, lda, pivots, info);
    return launched();
}

template<typename T>
cublasStatus_t getrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const T *const a[],
                            int lda, const int *pivots, T *const b[], int ldb, int *info, int batch) {
    CHECK_HANDLE(handle);
    if (info == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // (info reports the first invalid argument, as a negative number)
    *info = n < 0 ? -2 : nrhs < 0 ? -3 : lda < std::max(1, n) ? -5 : pivots == nullptr ? -6 : ldb < std::max(1, n)
                                                                                              ? -8 : batch < 0 ? -10
                                                                                                               : 0;
    if (*info != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || nrhs == 0 || batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    getrsBatchedKernel<<<batch, THREADS, 0, handle->stream>>>(trans, n, nrhs, a, lda, pivots, b, ldb);
    return launched();
}

template<typename T>
cublasStatus_t geqrfBatched(cublasHandle_t handle, int m, int n, T *const a[], int lda, T *const tau[], int *info,
                            int batch) {
    CHECK_HANDLE(handle);
    if (info == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *info = m < 0 ? -1 : n < 0 ? -2 : lda < std::max(1, m) ? -4 : batch < 0 ? -7 : 0;
    if (*info != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0 || batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    geqrfBatchedKernel<<<batch, THREADS, 0, handle->stream>>>(m, n, a, lda, tau);
    return launched();
}

template<typename T>
cublasStatus_t gelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, T *const a[],
                           int lda, T *const c[], int ldc, int *info, int *devInfo, int batch) {
    CHECK_HANDLE(handle);
    if (info == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // (like cuBLAS: only the non-transposed overdetermined case)
    *info = trans != CUBLAS_OP_N ? -1 : m < n ? -2 : n < 0 ? -3 : nrhs < 0 ? -4 : lda < std::max(1, m) ? -6
                                                                                  : ldc < std::max(1, m) ? -8
                                                                                                         : batch < 0
                                                                                                           ? -11 : 0;
    if (*info != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || batch == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    T *tau = static_cast<T *>(librecuda_blas_scratch(handle->stream, sizeof(T) * size_t(n) * batch));
    if (tau == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    gelsBatchedKernel<<<batch, THREADS, 0, handle->stream>>>(m, n, nrhs, a, lda, c, ldc, tau, devInfo);
    return launched();
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// the API

#define FOR_REAL(X) X(S, float) X(D, double)
#define FOR_COMPLEX(X) X(C, cuComplex) X(Z, cuDoubleComplex)
#define FOR_ALL(X) FOR_REAL(X) FOR_COMPLEX(X)

// level 1
#define AXPY(P, T) cublasStatus_t cublas##P##axpy_v2(cublasHandle_t h, int n, const T *alpha, const T *x, int incx,   \
                                                     T *y, int incy) { return axpy(h, n, alpha, x, incx, y, incy); }
FOR_ALL(AXPY)
#define SCAL(P, T) cublasStatus_t cublas##P##scal_v2(cublasHandle_t h, int n, const T *alpha, T *x, int incx) {        \
    return scal(h, n, alpha, x, incx); }
FOR_ALL(SCAL)
cublasStatus_t cublasCsscal_v2(cublasHandle_t h, int n, const float *alpha, cuComplex *x, int incx) {
    return scal(h, n, alpha, x, incx);
}
cublasStatus_t cublasZdscal_v2(cublasHandle_t h, int n, const double *alpha, cuDoubleComplex *x, int incx) {
    return scal(h, n, alpha, x, incx);
}
#define COPY(P, T) cublasStatus_t cublas##P##copy_v2(cublasHandle_t h, int n, const T *x, int incx, T *y, int incy) {   \
    return copy(h, n, x, incx, y, incy, false); }                                                                       \
cublasStatus_t cublas##P##swap_v2(cublasHandle_t h, int n, T *x, int incx, T *y, int incy) {                            \
    return copy(h, n, static_cast<const T *>(x), incx, y, incy, true); }
FOR_ALL(COPY)
#define REDUCE(name, T, R, op) cublasStatus_t name(cublasHandle_t h, int n, const T *x, int incx, R *result) {          \
    return reduce(h, n, x, incx, op, result); }
REDUCE(cublasSnrm2_v2, float, float, Reduction::NRM2)
REDUCE(cublasDnrm2_v2, double, double, Reduction::NRM2)
REDUCE(cublasScnrm2_v2, cuComplex, float, Reduction::NRM2)
REDUCE(cublasDznrm2_v2, cuDoubleComplex, double, Reduction::NRM2)
REDUCE(cublasSasum_v2, float, float, Reduction::ASUM)
REDUCE(cublasDasum_v2, double, double, Reduction::ASUM)
REDUCE(cublasScasum_v2, cuComplex, float, Reduction::ASUM)
REDUCE(cublasDzasum_v2, cuDoubleComplex, double, Reduction::ASUM)
REDUCE(cublasIsamax_v2, float, int, Reduction::IAMAX)
REDUCE(cublasIdamax_v2, double, int, Reduction::IAMAX)
REDUCE(cublasIcamax_v2, cuComplex, int, Reduction::IAMAX)
REDUCE(cublasIzamax_v2, cuDoubleComplex, int, Reduction::IAMAX)
REDUCE(cublasIsamin_v2, float, int, Reduction::IAMIN)
REDUCE(cublasIdamin_v2, double, int, Reduction::IAMIN)
REDUCE(cublasIcamin_v2, cuComplex, int, Reduction::IAMIN)
REDUCE(cublasIzamin_v2, cuDoubleComplex, int, Reduction::IAMIN)
#define ROT(P, T, S) cublasStatus_t cublas##P##rot_v2(cublasHandle_t h, int n, T *x, int incx, T *y, int incy,          \
                                                      const Real<T> *c, const S *s) {                                   \
    return rot(h, n, x, incx, y, incy, c, s); }
ROT(S, float, float)
ROT(D, double, double)
ROT(C, cuComplex, cuComplex)
ROT(Z, cuDoubleComplex, cuDoubleComplex)
ROT(Cs, cuComplex, float)
ROT(Zd, cuDoubleComplex, double)
#define ROTG(P, T) cublasStatus_t cublas##P##rotg_v2(cublasHandle_t h, T *a, T *b, Real<T> *c, T *s) {                 \
    return rotg(h, a, b, c, s); }
FOR_ALL(ROTG)
#define ROTM(P, T) cublasStatus_t cublas##P##rotm_v2(cublasHandle_t h, int n, T *x, int incx, T *y, int incy,           \
                                                     const T *param) { return rotm(h, n, x, incx, y, incy, param); }
FOR_REAL(ROTM)

// level 2
#define GER(name, T, conj) cublasStatus_t name(cublasHandle_t h, int m, int n, const T *alpha, const T *x, int incx,    \
                                               const T *y, int incy, T *A, int lda) {                                   \
    return ger(h, m, n, alpha, x, incx, y, incy, A, lda, conj); }
GER(cublasSger_v2, float, false)
GER(cublasDger_v2, double, false)
GER(cublasCgeru_v2, cuComplex, false)
GER(cublasZgeru_v2, cuDoubleComplex, false)
GER(cublasCgerc_v2, cuComplex, true)
GER(cublasZgerc_v2, cuDoubleComplex, true)
#define SYMV(name, T, herm) cublasStatus_t name(cublasHandle_t h, cublasFillMode_t uplo, int n, const T *alpha,        \
                                                const T *A, int lda, const T *x, int incx, const T *beta, T *y,         \
                                                int incy) {                                                             \
    return symv(h, uplo, n, alpha, A, lda, x, incx, beta, y, incy, herm); }
SYMV(cublasSsymv_v2, float, false)
SYMV(cublasDsymv_v2, double, false)
SYMV(cublasCsymv_v2, cuComplex, false)
SYMV(cublasZsymv_v2, cuDoubleComplex, false)
SYMV(cublasChemv_v2, cuComplex, true)
SYMV(cublasZhemv_v2, cuDoubleComplex, true)
#define SYR(name, T, S, herm) cublasStatus_t name(cublasHandle_t h, cublasFillMode_t uplo, int n, const S *alpha,      \
                                                  const T *x, int incx, T *A, int lda) {                                \
    return rankUpdate(h, uplo, n, alpha, x, incx, static_cast<const T *>(nullptr), 0, A, lda, herm); }
SYR(cublasSsyr_v2, float, float, false)
SYR(cublasDsyr_v2, double, double, false)
SYR(cublasCsyr_v2, cuComplex, cuComplex, false)
SYR(cublasZsyr_v2, cuDoubleComplex, cuDoubleComplex, false)
SYR(cublasCher_v2, cuComplex, float, true)
SYR(cublasZher_v2, cuDoubleComplex, double, true)
#define SYR2(name, T, herm) cublasStatus_t name(cublasHandle_t h, cublasFillMode_t uplo, int n, const T *alpha,        \
                                                const T *x, int incx, const T *y, int incy, T *A, int lda) {            \
    return rankUpdate(h, uplo, n, alpha, x, incx, y, incy, A, lda, herm); }
SYR2(cublasSsyr2_v2, float, false)
SYR2(cublasDsyr2_v2, double, false)
SYR2(cublasCsyr2_v2, cuComplex, false)
SYR2(cublasZsyr2_v2, cuDoubleComplex, false)
SYR2(cublasCher2_v2, cuComplex, true)
SYR2(cublasZher2_v2, cuDoubleComplex, true)
#define TRV(P, T) cublasStatus_t cublas##P##trmv_v2(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans,  \
                                                    cublasDiagType_t diag, int n, const T *A, int lda, T *x,            \
                                                    int incx) { return trmv(h, uplo, trans, diag, n, A, lda, x, incx); } \
cublasStatus_t cublas##P##trsv_v2(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans,                     \
                                  cublasDiagType_t diag, int n, const T *A, int lda, T *x, int incx) {                  \
    return trsv(h, uplo, trans, diag, n, A, lda, x, incx); }
FOR_ALL(TRV)

// level 3
#define TRSM(P, T) cublasStatus_t cublas##P##trsm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,   \
                                                     cublasOperation_t trans, cublasDiagType_t diag, int m, int n,      \
                                                     const T *alpha, const T *A, int lda, T *B, int ldb) {              \
    return trsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }                                             \
cublasStatus_t cublas##P##trmm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,                       \
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const T *alpha,         \
                                  const T *A, int lda, const T *B, int ldb, T *C, int ldc) {                            \
    return trmm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc); }                                     \
cublasStatus_t cublas##P##trsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,                   \
                                      cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const T *alpha,     \
                                      const T *const A[], int lda, T *const B[], int ldb, int batchCount) {             \
    return trsmBatched(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount); }                          \
cublasStatus_t cublas##P##getrfBatched(cublasHandle_t h, int n, T *const A[], int lda, int *P, int *info,               \
                                       int batchSize) { return getrfBatched(h, n, A, lda, P, info, batchSize); }        \
cublasStatus_t cublas##P##getrsBatched(cublasHandle_t h, cublasOperation_t trans, int n, int nrhs,                      \
                                       const T *const Aarray[], int lda, const int *devIpiv, T *const Barray[],         \
                                       int ldb, int *info, int batchSize) {                                             \
    return getrsBatched(h, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize); }                       \
cublasStatus_t cublas##P##geqrfBatched(cublasHandle_t h, int m, int n, T *const Aarray[], int lda,                      \
                                       T *const TauArray[], int *info, int batchSize) {                                 \
    return geqrfBatched(h, m, n, Aarray, lda, TauArray, info, batchSize); }                                             \
cublasStatus_t cublas##P##gelsBatched(cublasHandle_t h, cublasOperation_t trans, int m, int n, int nrhs,                \
                                      T *const Aarray[], int lda, T *const Carray[], int ldc, int *info,                \
                                      int *devInfoArray, int batchSize) {                                               \
    return gelsBatched(h, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize); }                \
cublasStatus_t cublas##P##symm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,         \
                                  const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C,        \
                                  int ldc) { return symm(h, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc,     \
                                                         false); }                                                      \
cublasStatus_t cublas##P##syrk_v2(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,       \
                                  const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc) {                  \
    return rankK(h, uplo, trans, n, k, alpha, A, lda, static_cast<const T *>(nullptr), 0, beta, C, ldc, false); }       \
cublasStatus_t cublas##P##syr2k_v2(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,      \
                                   const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C,       \
                                   int ldc) { return rankK(h, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc,   \
                                                           false); }
FOR_ALL(TRSM)

#define HERM(P, T) cublasStatus_t cublas##P##hemm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,   \
                                                     int m, int n, const T *alpha, const T *A, int lda, const T *B,     \
                                                     int ldb, const T *beta, T *C, int ldc) {                           \
    return symm(h, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, true); }                                      \
cublasStatus_t cublas##P##herk_v2(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,       \
                                  const Real<T> *alpha, const T *A, int lda, const Real<T> *beta, T *C, int ldc) {      \
    return rankK(h, uplo, trans, n, k, alpha, A, lda, static_cast<const T *>(nullptr), 0, beta, C, ldc, true); }        \
cublasStatus_t cublas##P##her2k_v2(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,      \
                                   const T *alpha, const T *A, int lda, const T *B, int ldb, const Real<T> *beta,       \
                                   T *C, int ldc) { return rankK(h, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C,  \
                                                                 ldc, true); }
FOR_COMPLEX(HERM)
