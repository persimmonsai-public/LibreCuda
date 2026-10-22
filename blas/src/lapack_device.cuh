/**
 * LAPACK style algorithms on the GPU for LibreCUDA's cuBLAS (batched factorizations) and cuSOLVER.
 *
 * The *Block functions are run by one thread block (THREADS threads) on one matrix: unblocked LU with partial pivoting,
 * Cholesky, Householder QR, forming and applying Q, triangular solves and the triangular factor of a block reflector.
 * They follow LAPACK's conventions (pivots 1-based, reflectors H = I - tau v v^H with v(0) = 1 implicit, info = first
 * failing column, 1-based). Large problems combine them with GEMMs (cusolver.cu).
 *
 * The Jacobi kernels compute eigen decompositions of Hermitian matrices (two-sided) and singular value decompositions
 * (one-sided, Hestenes) with one rotation per thread block, in rounds of disjoint pairs (round robin ordering).
 */
#pragma once

#include "scalar.cuh"

#include <cublas_api.h>

namespace librecuda_blas {

constexpr int THREADS = 256;

struct BlockShared {
    double re[THREADS];
    double im[THREADS];
    int index[THREADS];
    int flag;
};

template<typename T>
__device__ inline T &at(T *a, long long ld, long long i, long long j) {
    return a[i + j * ld];
}

template<typename T>
__device__ inline T at(const T *a, long long ld, long long i, long long j) {
    return a[i + j * ld];
}

// |re| + |im| (LAPACK's pivot measure)
template<typename T>
__device__ inline double cabs1(T x) {
    return fabs(double(re(x))) + fabs(double(im(x)));
}

__device__ inline double blockSumReal(double value, BlockShared &s) {
    s.re[threadIdx.x] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s.re[threadIdx.x] += s.re[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const double result = s.re[0];
    __syncthreads();
    return result;
}

template<typename T>
__device__ inline T blockSum(T value, BlockShared &s) {
    s.re[threadIdx.x] = re(value);
    s.im[threadIdx.x] = im(value);
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s.re[threadIdx.x] += s.re[threadIdx.x + stride];
            s.im[threadIdx.x] += s.im[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const T result = make<T>(Real<T>(s.re[0]), Real<T>(s.im[0]));
    __syncthreads();
    return result;
}

// the index of the largest value (the smallest index among equal ones); -1 if no thread has one
__device__ inline int blockArgMax(double value, int index, BlockShared &s) {
    s.re[threadIdx.x] = value;
    s.index[threadIdx.x] = index;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const double other = s.re[threadIdx.x + stride];
            const int other_index = s.index[threadIdx.x + stride];
            if (other_index >= 0 && (s.index[threadIdx.x] < 0 || other > s.re[threadIdx.x] ||
                                     (other == s.re[threadIdx.x] && other_index < s.index[threadIdx.x]))) {
                s.re[threadIdx.x] = other;
                s.index[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    const int result = s.index[0];
    __syncthreads();
    return result;
}

template<typename T>
__device__ inline T warpSum(T value) {
    Real<T> r = re(value), i = im(value);
    for (int offset = 16; offset > 0; offset /= 2) {
        r += __shfl_down_sync(0xffffffffu, r, offset);
        if constexpr (IS_COMPLEX<T>) {
            i += __shfl_down_sync(0xffffffffu, i, offset);
        }
    }
    r = __shfl_sync(0xffffffffu, r, 0);
    if constexpr (IS_COMPLEX<T>) {
        i = __shfl_sync(0xffffffffu, i, 0);
    }
    return make<T>(r, i);
}

// sets *info (if 0) to value: the first failure is reported
__device__ inline void reportInfo(int *info, int value) {
    if (info != nullptr && *info == 0) {
        *info = value;
    }
}

/**
 * LU factorization with partial pivoting (pivots: nullptr for none) of the m x n matrix A: A = P L U. Pivot rows are
 * reported as row + 1 + pivotOffset, singular pivots as column + 1 + infoOffset.
 */
template<typename T>
__device__ void luBlock(T *a, int m, int n, long long lda, int *ipiv, int pivotOffset, int *info, int infoOffset,
                        BlockShared &s) {
    const int tid = threadIdx.x;
    const int kmax = min(m, n);
    for (int j = 0; j < kmax; j++) {
        if (ipiv != nullptr) {
            double best = -1;
            int best_index = -1;
            for (int i = j + tid; i < m; i += THREADS) {
                const double value = cabs1(at(a, lda, i, j));
                if (value > best) {
                    best = value;
                    best_index = i;
                }
            }
            const int p = blockArgMax(best, best_index, s);
            if (tid == 0) {
                ipiv[j] = p + 1 + pivotOffset;
            }
            if (p != j) {
                for (int c = tid; c < n; c += THREADS) {
                    const T tmp = at(a, lda, j, c);
                    at(a, lda, j, c) = at(a, lda, p, c);
                    at(a, lda, p, c) = tmp;
                }
            }
            __syncthreads();
        }
        const T pivot = at(a, lda, j, j);
        if (isZero(pivot)) {
            if (tid == 0) {
                reportInfo(info, j + 1 + infoOffset);
            }
        } else {
            for (int i = j + 1 + tid; i < m; i += THREADS) {
                at(a, lda, i, j) = div(at(a, lda, i, j), pivot);
            }
        }
        __syncthreads();
        const long long rows = m - j - 1, cols = n - j - 1;
        for (long long e = tid; e < rows * cols; e += THREADS) {
            const long long i = j + 1 + e % rows, c = j + 1 + e / rows;
            at(a, lda, i, c) = sub(at(a, lda, i, c), mul(at(a, lda, i, j), at(a, lda, j, c)));
        }
        __syncthreads();
    }
}

// element (i, k) of op(A)
template<typename T>
__device__ inline T opElement(const T *a, long long lda, cublasOperation_t trans, int i, int k) {
    if (trans == CUBLAS_OP_N) {
        return at(a, lda, i, k);
    }
    const T value = at(a, lda, k, i);
    return trans == CUBLAS_OP_C ? conjugate(value) : value;
}

/**
 * Solves op(A) X = B in place (B: n x nrhs), A triangular n x n
 */
template<typename T>
__device__ void trsmLeftBlock(bool lower, cublasOperation_t trans, bool unit, int n, int nrhs, const T *a,
                              long long lda, T *b, long long ldb) {
    const int tid = threadIdx.x;
    const bool eff_lower = lower == (trans == CUBLAS_OP_N);
    for (int step = 0; step < n; step++) {
        const int k = eff_lower ? step : n - 1 - step;
        if (!unit) {
            const T d = opElement(a, lda, trans, k, k);
            for (int c = tid; c < nrhs; c += THREADS) {
                at(b, ldb, k, c) = div(at(b, ldb, k, c), d);
            }
            __syncthreads();
        }
        const int lo = eff_lower ? k + 1 : 0, hi = eff_lower ? n : k;
        const long long rows = hi - lo;
        for (long long e = tid; e < rows * nrhs; e += THREADS) {
            const int i = lo + int(e % rows), c = int(e / rows);
            at(b, ldb, i, c) = sub(at(b, ldb, i, c), mul(opElement(a, lda, trans, i, k), at(b, ldb, k, c)));
        }
        __syncthreads();
    }
}

/**
 * Solves X op(A) = B in place (B: m x n), A triangular n x n
 */
template<typename T>
__device__ void trsmRightBlock(bool lower, cublasOperation_t trans, bool unit, int m, int n, const T *a,
                               long long lda, T *b, long long ldb) {
    const int tid = threadIdx.x;
    const bool eff_upper = lower != (trans == CUBLAS_OP_N);
    for (int step = 0; step < n; step++) {
        const int k = eff_upper ? step : n - 1 - step;
        if (!unit) {
            const T d = opElement(a, lda, trans, k, k);
            for (int r = tid; r < m; r += THREADS) {
                at(b, ldb, r, k) = div(at(b, ldb, r, k), d);
            }
            __syncthreads();
        }
        const int lo = eff_upper ? k + 1 : 0, hi = eff_upper ? n : k;
        const long long cols = hi - lo;
        for (long long e = tid; e < cols * m; e += THREADS) {
            const int r = int(e % m), j = lo + int(e / m);
            at(b, ldb, r, j) = sub(at(b, ldb, r, j), mul(at(b, ldb, r, k), opElement(a, lda, trans, k, j)));
        }
        __syncthreads();
    }
}

// applies the row interchanges ipiv[0..n) (1-based) to the columns of B, forward or backward
template<typename T>
__device__ void laswpBlock(int n, int nrhs, T *b, long long ldb, const int *ipiv, bool forward) {
    for (int step = 0; step < n; step++) {
        const int i = forward ? step : n - 1 - step;
        const int p = ipiv[i] - 1;
        if (p != i) {
            for (int c = threadIdx.x; c < nrhs; c += THREADS) {
                const T tmp = at(b, ldb, i, c);
                at(b, ldb, i, c) = at(b, ldb, p, c);
                at(b, ldb, p, c) = tmp;
            }
        }
        __syncthreads();
    }
}

/**
 * Solves op(A) X = B with the LU factorization of A (luBlock)
 */
template<typename T>
__device__ void getrsBlock(cublasOperation_t trans, int n, int nrhs, const T *a, long long lda, const int *ipiv, T *b,
                           long long ldb) {
    if (trans == CUBLAS_OP_N) {
        laswpBlock(n, nrhs, b, ldb, ipiv, true);
        trsmLeftBlock(true, CUBLAS_OP_N, true, n, nrhs, a, lda, b, ldb);
        trsmLeftBlock(false, CUBLAS_OP_N, false, n, nrhs, a, lda, b, ldb);
    } else {
        // A^T = U^T L^T P^T
        trsmLeftBlock(false, trans, false, n, nrhs, a, lda, b, ldb);
        trsmLeftBlock(true, trans, true, n, nrhs, a, lda, b, ldb);
        laswpBlock(n, nrhs, b, ldb, ipiv, false);
    }
}

/**
 * Cholesky factorization of the Hermitian positive definite n x n matrix A: A = L L^H (lower) or U^H U (upper), in
 * the given triangle (the other one isn't touched). Stops at the first non-positive pivot (column + 1 + infoOffset).
 * Returns false if it failed.
 */
template<typename T>
__device__ bool choleskyBlock(bool lower, int n, T *a, long long lda, int *info, int infoOffset, BlockShared &s) {
    const int tid = threadIdx.x;
    for (int j = 0; j < n; j++) {
        if (tid == 0) {
            const Real<T> d = re(at(a, lda, j, j));
            s.flag = !(d > 0);
            if (s.flag) {
                reportInfo(info, j + 1 + infoOffset);
            } else {
                at(a, lda, j, j) = make<T>(sqrt(d));
            }
        }
        __syncthreads();
        if (s.flag) {
            return false;
        }
        const Real<T> inv = Real<T>(1) / re(at(a, lda, j, j));
        for (int i = j + 1 + tid; i < n; i += THREADS) {
            if (lower) {
                at(a, lda, i, j) = scale(at(a, lda, i, j), inv);
            } else {
                at(a, lda, j, i) = scale(at(a, lda, j, i), inv);
            }
        }
        __syncthreads();
        const long long rows = n - j - 1;
        for (long long e = tid; e < rows * rows; e += THREADS) {
            const int i = j + 1 + int(e % rows), c = j + 1 + int(e / rows);
            if (lower && c <= i) {
                at(a, lda, i, c) = sub(at(a, lda, i, c), mul(at(a, lda, i, j), conjugate(at(a, lda, c, j))));
            } else if (!lower && i <= c) {
                at(a, lda, i, c) = sub(at(a, lda, i, c), mul(conjugate(at(a, lda, j, i)), at(a, lda, j, c)));
            }
        }
        __syncthreads();
        // (the diagonal of a Hermitian matrix is real)
        if (tid == 0 && j + 1 < n) {
            at(a, lda, j + 1, j + 1) = make<T>(re(at(a, lda, j + 1, j + 1)));
        }
        __syncthreads();
    }
    return true;
}

/**
 * Solves A X = B with the Cholesky factorization of A (choleskyBlock)
 */
template<typename T>
__device__ void potrsBlock(bool lower, int n, int nrhs, const T *a, long long lda, T *b, long long ldb) {
    if (lower) {
        trsmLeftBlock(true, CUBLAS_OP_N, false, n, nrhs, a, lda, b, ldb);
        trsmLeftBlock(true, CUBLAS_OP_C, false, n, nrhs, a, lda, b, ldb);
    } else {
        trsmLeftBlock(false, CUBLAS_OP_C, false, n, nrhs, a, lda, b, ldb);
        trsmLeftBlock(false, CUBLAS_OP_N, false, n, nrhs, a, lda, b, ldb);
    }
}

/**
 * Householder reflector of column j of A (rows j..m): A(j, j) = beta, A(j+1.., j) = v, returns tau
 */
template<typename T>
__device__ T householderBlock(int m, T *a, long long lda, int j, BlockShared &s) {
    double part = 0;
    for (int i = j + 1 + threadIdx.x; i < m; i += THREADS) {
        part += norm2(at(a, lda, i, j));
    }
    const double xnorm2 = blockSumReal(part, s);
    const T alpha = at(a, lda, j, j);
    if (xnorm2 == 0 && im(alpha) == 0) {
        return make<T>(0);
    }
    const Real<T> beta = -copysign(Real<T>(sqrt(double(norm2(alpha)) + xnorm2)), re(alpha));
    const T tau = make<T>((beta - re(alpha)) / beta, -im(alpha) / beta);
    const T factor = div(make<T>(1), sub(alpha, make<T>(beta)));
    for (int i = j + 1 + threadIdx.x; i < m; i += THREADS) {
        at(a, lda, i, j) = mul(at(a, lda, i, j), factor);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        at(a, lda, j, j) = make<T>(beta);
    }
    __syncthreads();
    return tau;
}

/**
 * C(row0.., cols) = (I - t v v^H) C for the reflector v stored in column vcol of V from row row0 (v(row0) = 1)
 */
template<typename T>
__device__ void applyReflectorLeft(int m, const T *v, long long ldv, int vcol, int row0, T t, T *c, long long ldc,
                                   int col0, int col1) {
    if (isZero(t)) {
        return;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int col = col0 + warp; col < col1; col += THREADS / 32) {
        T w = lane == 0 ? at(c, ldc, row0, col) : make<T>(0);
        for (int i = row0 + 1 + lane; i < m; i += 32) {
            w = add(w, mul(conjugate(at(v, ldv, i, vcol)), at(c, ldc, i, col)));
        }
        w = warpSum(w);
        const T f = mul(t, w);
        if (lane == 0) {
            at(c, ldc, row0, col) = sub(at(c, ldc, row0, col), f);
        }
        for (int i = row0 + 1 + lane; i < m; i += 32) {
            at(c, ldc, i, col) = sub(at(c, ldc, i, col), mul(at(v, ldv, i, vcol), f));
        }
    }
    __syncthreads();
}

/**
 * C(rows, col0..) = C (I - t v v^H) for the reflector v (length n - col0, v(col0) = 1) stored in column vcol of V
 */
template<typename T>
__device__ void applyReflectorRight(int n, const T *v, long long ldv, int vcol, int col0, T t, T *c, long long ldc,
                                    int row0, int row1) {
    if (isZero(t)) {
        return;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int row = row0 + warp; row < row1; row += THREADS / 32) {
        T w = lane == 0 ? at(c, ldc, row, col0) : make<T>(0);
        for (int j = col0 + 1 + lane; j < n; j += 32) {
            w = add(w, mul(at(c, ldc, row, j), at(v, ldv, j, vcol)));
        }
        w = warpSum(w);
        const T f = mul(w, t);
        if (lane == 0) {
            at(c, ldc, row, col0) = sub(at(c, ldc, row, col0), f);
        }
        for (int j = col0 + 1 + lane; j < n; j += 32) {
            at(c, ldc, row, j) = sub(at(c, ldc, row, j), mul(f, conjugate(at(v, ldv, j, vcol))));
        }
    }
    __syncthreads();
}

/**
 * QR factorization of the m x n matrix A: R in the upper triangle, the reflectors below with their factors in tau
 */
template<typename T>
__device__ void qrBlock(int m, int n, T *a, long long lda, T *tau, BlockShared &s) {
    const int kmax = min(m, n);
    for (int j = 0; j < kmax; j++) {
        const T t = householderBlock(m, a, lda, j, s);
        if (threadIdx.x == 0) {
            tau[j] = t;
        }
        // H^H = I - conj(tau) v v^H on the remaining columns
        applyReflectorLeft(m, a, lda, j, j, conjugate(t), a, lda, j + 1, n);
    }
}

/**
 * C = op(Q) C (left: C is m x n, Q m x m) or C op(Q) (right: Q n x n), Q = H(0) ... H(k-1) from the reflectors of A
 */
template<typename T>
__device__ void applyQBlock(bool left, bool conjTrans, int m, int n, int k, const T *a, long long lda, const T *tau,
                            T *c, long long ldc) {
    const bool forward = left == conjTrans;
    for (int step = 0; step < k; step++) {
        const int i = forward ? step : k - 1 - step;
        const T t = conjTrans ? conjugate(tau[i]) : tau[i];
        if (left) {
            applyReflectorLeft(m, a, lda, i, i, t, c, ldc, 0, n);
        } else {
            applyReflectorRight(n, a, lda, i, i, t, c, ldc, 0, m);
        }
    }
}

/**
 * Forms the first n columns of Q = H(0) ... H(k-1) (m x n, m >= n >= k) in place of the reflectors (LAPACK ung2r)
 */
template<typename T>
__device__ void orgqrBlock(int m, int n, int k, T *a, long long lda, const T *tau) {
    const int tid = threadIdx.x;
    for (long long e = tid; e < static_cast<long long>(m) * (n - k); e += THREADS) {
        const int i = int(e % m), j = k + int(e / m);
        at(a, lda, i, j) = make<T>(i == j ? 1 : 0);
    }
    __syncthreads();
    for (int i = k - 1; i >= 0; i--) {
        if (i < n - 1) {
            applyReflectorLeft(m, a, lda, i, i, tau[i], a, lda, i + 1, n);
        }
        for (int r = tid; r < m; r += THREADS) {
            if (r > i) {
                at(a, lda, r, i) = neg(mul(tau[i], at(a, lda, r, i)));
            } else if (r == i) {
                at(a, lda, r, i) = sub(make<T>(1), tau[i]);
            } else {
                at(a, lda, r, i) = make<T>(0);
            }
        }
        __syncthreads();
    }
}

/**
 * The upper triangular factor T (k x k, zero below the diagonal) of the block reflector H = I - V T V^H of the
 * reflectors in the columns of V (rows from 0, v_i(i) = 1, m rows) (LAPACK larft, forward, columnwise)
 */
template<typename T>
__device__ void larftBlock(int m, int k, const T *v, long long ldv, const T *tau, T *t, long long ldt) {
    __shared__ T x[64];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int e = threadIdx.x; e < k * k; e += THREADS) {
        at(t, ldt, e % k, e / k) = make<T>(0);
    }
    __syncthreads();
    for (int i = 0; i < k; i++) {
        // x(j) = -tau_i v_j^H v_i for j < i
        for (int j = warp; j < i; j += THREADS / 32) {
            T w = lane == 0 ? conjugate(at(v, ldv, i, j)) : make<T>(0);
            for (int r = i + 1 + lane; r < m; r += 32) {
                w = add(w, mul(conjugate(at(v, ldv, r, j)), at(v, ldv, r, i)));
            }
            w = warpSum(w);
            if (lane == 0) {
                x[j] = neg(mul(tau[i], w));
            }
        }
        __syncthreads();
        // T(0:i, i) = T(0:i, 0:i) x
        for (int j = threadIdx.x; j < i; j += THREADS) {
            T sum = make<T>(0);
            for (int l = j; l < i; l++) {
                sum = add(sum, mul(at(t, ldt, j, l), x[l]));
            }
            at(t, ldt, j, i) = sum;
        }
        if (threadIdx.x == 0) {
            at(t, ldt, i, i) = tau[i];
        }
        __syncthreads();
    }
}

// ---- Jacobi rotations

// the round robin pairing: pair k of round r over count (even) indices
__device__ inline void roundRobinPair(int count, int round, int k, int &p, int &q) {
    if (k == 0) {
        p = count - 1;
        q = round;
    } else {
        p = (round + k) % (count - 1);
        q = (round - k + count - 1) % (count - 1);
    }
    if (p > q) {
        const int tmp = p;
        p = q;
        q = tmp;
    }
}

/**
 * The rotation J = [[c, s], [-s conj(e), c conj(e)]] (e = b / |b|) with J^H [[a, b], [conj(b), d]] J diagonal
 */
template<typename T>
__device__ inline void jacobiRotation(double a, double d, T b, double &c, double &s, T &e) {
    const double r = hypot(double(re(b)), double(im(b)));
    e = make<T>(Real<T>(re(b) / r), Real<T>(im(b) / r));
    const double zeta = (d - a) / (2 * r);
    const double t = (zeta >= 0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1 + zeta * zeta));
    c = 1 / sqrt(1 + t * t);
    s = t * c;
}

// off-diagonal elements below this are left alone (their rotation would divide by a denormal)
template<typename T>
__device__ inline double rotationFloor() {
    return std::is_same_v<Real<T>, float> ? 1e-30 : 1e-290;
}

struct Rotation {
    double c, s, er, ei;
    int active;
};

/**
 * One round of a Hermitian Jacobi sweep, rows: W <- J^H W for each pair (p, q). Block (pair, matrix). The rotations
 * are stored for the column round. count: active rotations (atomically incremented).
 */
template<typename T>
__global__ void jacobiEigenRows(int n, T *w, long long ldw, long long strideW, int round, double tol,
                                Rotation *rotations, int *count) {
    const int pairs = (n + 1) / 2;
    const int pair = blockIdx.x;
    T *wb = w + blockIdx.y * strideW;
    Rotation &rot = rotations[blockIdx.y * pairs + pair];
    int p, q;
    roundRobinPair(pairs * 2, round, pair, p, q);
    __shared__ Rotation r;
    if (threadIdx.x == 0) {
        r.active = 0;
        if (q < n) {
            const double app = re(at(wb, ldw, p, p)), aqq = re(at(wb, ldw, q, q));
            const T apq = at(wb, ldw, p, q);
            const double off = double(absval(apq));
            if (off > tol * sqrt(fabs(app * aqq)) && off > rotationFloor<T>()) {
                T e;
                jacobiRotation(app, aqq, apq, r.c, r.s, e);
                r.er = re(e);
                r.ei = im(e);
                r.active = 1;
                atomicAdd(count, 1);
            }
        }
        rot = r;
    }
    __syncthreads();
    if (!r.active) {
        return;
    }
    const T e = make<T>(Real<T>(r.er), Real<T>(r.ei));
    const Real<T> c = r.c, s = r.s;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        const T wp = at(wb, ldw, p, col), wq = at(wb, ldw, q, col);
        // rows p, q of J^H W
        at(wb, ldw, p, col) = sub(scale(wp, c), mul(scale(e, s), wq));
        at(wb, ldw, q, col) = add(scale(wp, s), mul(scale(e, c), wq));
    }
}

/**
 * The columns of the round: W <- W J, V <- V J (V: nullptr when no vectors are wanted); the rotated element is zero
 */
template<typename T>
__global__ void jacobiEigenColumns(int n, T *w, long long ldw, long long strideW, T *v, long long ldv,
                                   long long strideV, int round, const Rotation *rotations) {
    const int pairs = (n + 1) / 2;
    const Rotation r = rotations[blockIdx.y * pairs + blockIdx.x];
    if (!r.active) {
        return;
    }
    int p, q;
    roundRobinPair(pairs * 2, round, blockIdx.x, p, q);
    T *wb = w + blockIdx.y * strideW;
    const T ec = conjugate(make<T>(Real<T>(r.er), Real<T>(r.ei)));
    const Real<T> c = r.c, s = r.s;
    for (int row = threadIdx.x; row < n; row += blockDim.x) {
        const T wp = at(wb, ldw, row, p), wq = at(wb, ldw, row, q);
        at(wb, ldw, row, p) = sub(scale(wp, c), mul(scale(ec, s), wq));
        at(wb, ldw, row, q) = add(scale(wp, s), mul(scale(ec, c), wq));
        if (v != nullptr) {
            T *vb = v + blockIdx.y * strideV;
            const T vp = at(vb, ldv, row, p), vq = at(vb, ldv, row, q);
            at(vb, ldv, row, p) = sub(scale(vp, c), mul(scale(ec, s), vq));
            at(vb, ldv, row, q) = add(scale(vp, s), mul(scale(ec, c), vq));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        at(wb, ldw, p, q) = make<T>(0);
        at(wb, ldw, q, p) = make<T>(0);
        at(wb, ldw, p, p) = make<T>(re(at(wb, ldw, p, p)));
        at(wb, ldw, q, q) = make<T>(re(at(wb, ldw, q, q)));
    }
}

/**
 * One round of a one-sided Jacobi sweep: orthogonalizes columns p and q of U (m rows): U <- U J, V <- V J
 */
template<typename T>
__global__ void jacobiSvdRound(int m, int n, T *u, long long ldu, long long strideU, T *v, long long ldv,
                               long long strideV, int round, double tol, int *count) {
    __shared__ BlockShared s;
    __shared__ Rotation r;
    const int pairs = (n + 1) / 2;
    int p, q;
    roundRobinPair(pairs * 2, round, blockIdx.x, p, q);
    if (q >= n) {
        return;
    }
    T *ub = u + blockIdx.y * strideU;
    double alpha = 0, beta = 0;
    T gamma = make<T>(0);
    for (int i = threadIdx.x; i < m; i += THREADS) {
        const T up = at(ub, ldu, i, p), uq = at(ub, ldu, i, q);
        alpha += norm2(up);
        beta += norm2(uq);
        gamma = add(gamma, mul(conjugate(up), uq));
    }
    alpha = blockSumReal(alpha, s);
    beta = blockSumReal(beta, s);
    gamma = blockSum(gamma, s);
    if (threadIdx.x == 0) {
        const double off = double(absval(gamma));
        r.active = off > tol * sqrt(alpha * beta) && off > rotationFloor<T>();
        if (r.active) {
            T e;
            jacobiRotation(alpha, beta, gamma, r.c, r.s, e);
            r.er = re(e);
            r.ei = im(e);
            atomicAdd(count, 1);
        }
    }
    __syncthreads();
    if (!r.active) {
        return;
    }
    const T ec = conjugate(make<T>(Real<T>(r.er), Real<T>(r.ei)));
    const Real<T> c = r.c, sn = r.s;
    for (int i = threadIdx.x; i < m; i += THREADS) {
        const T up = at(ub, ldu, i, p), uq = at(ub, ldu, i, q);
        at(ub, ldu, i, p) = sub(scale(up, c), mul(scale(ec, sn), uq));
        at(ub, ldu, i, q) = add(scale(up, sn), mul(scale(ec, c), uq));
    }
    if (v != nullptr) {
        T *vb = v + blockIdx.y * strideV;
        for (int i = threadIdx.x; i < n; i += THREADS) {
            const T vp = at(vb, ldv, i, p), vq = at(vb, ldv, i, q);
            at(vb, ldv, i, p) = sub(scale(vp, c), mul(scale(ec, sn), vq));
            at(vb, ldv, i, q) = add(scale(vp, sn), mul(scale(ec, c), vq));
        }
    }
}

/**
 * Orthonormal completion: replaces the columns of Q (rows x cols) flagged in need[] by unit vectors orthogonal to all
 * other columns (Gram-Schmidt of unit vectors, twice), one matrix per block
 */
template<typename T>
__device__ void completeBasisBlock(int rows, int cols, T *q, long long ldq, const int *need, BlockShared &s) {
    int candidate = 0;
    for (int j = 0; j < cols; j++) {
        if (!need[j]) {
            continue;
        }
        for (; candidate < rows; candidate++) {
            for (int i = threadIdx.x; i < rows; i += THREADS) {
                at(q, ldq, i, j) = make<T>(i == candidate ? 1 : 0);
            }
            __syncthreads();
            for (int pass = 0; pass < 2; pass++) {
                for (int k = 0; k < cols; k++) {
                    // the columns still to be completed (j and later ones flagged) don't count yet
                    if (k == j || (need[k] && k > j)) {
                        continue;
                    }
                    T dot = make<T>(0);
                    for (int i = threadIdx.x; i < rows; i += THREADS) {
                        dot = add(dot, mul(conjugate(at(q, ldq, i, k)), at(q, ldq, i, j)));
                    }
                    dot = blockSum(dot, s);
                    for (int i = threadIdx.x; i < rows; i += THREADS) {
                        at(q, ldq, i, j) = sub(at(q, ldq, i, j), mul(dot, at(q, ldq, i, k)));
                    }
                    __syncthreads();
                }
            }
            double part = 0;
            for (int i = threadIdx.x; i < rows; i += THREADS) {
                part += norm2(at(q, ldq, i, j));
            }
            // some unit vector keeps at least 1 / sqrt(rows) of its length in the complement
            const double norm = sqrt(blockSumReal(part, s));
            if (norm > 0.5 / sqrt(double(rows))) {
                for (int i = threadIdx.x; i < rows; i += THREADS) {
                    at(q, ldq, i, j) = scale(at(q, ldq, i, j), Real<T>(1 / norm));
                }
                __syncthreads();
                candidate++;
                break;
            }
        }
    }
}

} // namespace librecuda_blas
