/**
 * LibreCUDA's cuSOLVER (libcusolver.so.12): the dense functions PyTorch uses.
 *
 *  - LU (getrf / getrs), Cholesky (potrf / potrs, batched) and QR (geqrf, orgqr / ungqr, ormqr / unmqr): one thread
 *    block per matrix for small problems (lapack_device.cuh); large ones are blocked, with panel kernels, GEMMs and
 *    triangular solves of LibreCUDA's cuBLAS
 *  - Hermitian eigen decompositions (syevd / heevd, syevj / heevj, batched, Xsyevd, XsyevBatched): two-sided Jacobi
 *  - singular value decompositions (gesvd, gesvdj, gesvdjBatched, gesvdaStridedBatched): one-sided Jacobi
 *
 * The Jacobi sweeps run as rounds of disjoint rotations (one thread block each); the host checks for convergence after
 * every sweep
 *  - non-symmetric eigenvalues and right eigenvectors (Xgeev): on the host, like cuSOLVER's hybrid algorithm (Householder
 *    reduction to Hessenberg form, shifted complex QR iterations, back substitution), in double precision
 *
 * Not supported: the LDL^T factorization (sytrf, Xsytrs). Functions of cusolverDn.h not implemented here return
 * CUSOLVER_STATUS_NOT_SUPPORTED (stubs).
 */
#include "blas_helpers.cuh"
#include "lapack_device.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cfloat>
#include <cstring>
#include <complex>
#include <limits>
#include <vector>

using namespace librecuda_blas;

struct cusolverDnContext {
    uint32_t magic;
    cudaStream_t stream;
    cublasHandle_t blas;
};

struct syevjInfo {
    double tolerance;
    int maxSweeps;
    int sortEig;
    double residual;
    int sweeps;
};

struct gesvdjInfo {
    double tolerance;
    int maxSweeps;
    int sortEig;
    double residual;
    int sweeps;
};

struct cusolverDnParams {
    int unused;
};

namespace {

constexpr uint32_t SOLVER_MAGIC = 0x4c43534c;

bool valid(cusolverDnHandle_t handle) {
    return handle != nullptr && handle->magic == SOLVER_MAGIC;
}

#define CHECK_SOLVER(handle) do { if (!valid(handle)) { return CUSOLVER_STATUS_NOT_INITIALIZED; } } while (0)

cusolverStatus_t fromBlas(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return CUSOLVER_STATUS_SUCCESS;
        case CUBLAS_STATUS_ALLOC_FAILED: return CUSOLVER_STATUS_ALLOC_FAILED;
        case CUBLAS_STATUS_INVALID_VALUE: return CUSOLVER_STATUS_INVALID_VALUE;
        case CUBLAS_STATUS_NOT_SUPPORTED: return CUSOLVER_STATUS_NOT_SUPPORTED;
        default: return CUSOLVER_STATUS_EXECUTION_FAILED;
    }
}

#define SOLVER_TRY(expr) do { cusolverStatus_t solver_try_status = (expr); if (solver_try_status != CUSOLVER_STATUS_SUCCESS) { return solver_try_status; } } while (0)
#define BLAS_TRY(expr) SOLVER_TRY(fromBlas(expr))

cusolverStatus_t launchedSolver() {
    return cudaGetLastError() == cudaSuccess ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

cusolverStatus_t clearInfo(cusolverDnHandle_t handle, int *info, int count = 1) {
    if (info == nullptr) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    return cudaMemsetAsync(info, 0, sizeof(int) * count, handle->stream) == cudaSuccess
           ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// the typed triangular solves of cuBLAS (on the solver's stream)
cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const float *alpha, const float *a, int lda, float *b,
                    int ldb) {
    return cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const double *alpha, const double *a, int lda, double *b,
                    int ldb) {
    return cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *a, int lda,
                    cuComplex *b, int ldb) {
    return cublasCtrsm(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *a,
                    int lda, cuDoubleComplex *b, int ldb) {
    return cublasZtrsm(h, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
}

template<typename T>
cublasStatus_t trsmOne(cusolverDnHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                       cublasDiagType_t diag, int m, int n, const T *a, int lda, T *b, int ldb) {
    const T one = make<T>(1);
    cublasSetPointerMode(h->blas, CUBLAS_POINTER_MODE_HOST);
    return trsm(h->blas, side, uplo, trans, diag, m, n, &one, a, lda, b, ldb);
}

// ---------------------------------------------------------------------------------------------------------------------
// kernels

template<typename T>
__global__ void luKernel(int m, int n, T *a, int lda, int *ipiv, int pivotOffset, int *info, int infoOffset) {
    __shared__ BlockShared s;
    luBlock(a, m, n, lda, ipiv, pivotOffset, info, infoOffset, s);
}

// applies the interchanges ipiv[first..last) (1-based absolute rows) to columns [col0, col1) of A
template<typename T>
__global__ void laswpKernel(T *a, long long lda, int col0, int col1, const int *ipiv, int first, int last,
                            bool forward) {
    for (int col = col0 + blockIdx.x * blockDim.x + threadIdx.x; col < col1; col += gridDim.x * blockDim.x) {
        for (int step = 0; step < last - first; step++) {
            const int i = forward ? first + step : last - 1 - step;
            const int p = ipiv[i] - 1;
            if (p != i) {
                const T tmp = at(a, lda, i, col);
                at(a, lda, i, col) = at(a, lda, p, col);
                at(a, lda, p, col) = tmp;
            }
        }
    }
}

template<typename T>
__global__ void choleskyKernel(bool lower, int n, T *const *a, T *single, int lda, int *info, bool infoPerMatrix,
                               int infoOffset) {
    __shared__ BlockShared s;
    T *matrix = a != nullptr ? a[blockIdx.x] : single;
    choleskyBlock(lower, n, matrix, lda, info + (infoPerMatrix ? blockIdx.x : 0), infoOffset, s);
}

template<typename T>
__global__ void potrsBatchedKernel(bool lower, int n, int nrhs, T *const *a, int lda, T *const *b, int ldb) {
    potrsBlock(lower, n, nrhs, a[blockIdx.x], lda, b[blockIdx.x], ldb);
}

// A11 (n x n) -= W on its lower (upper) triangle
template<typename T>
__global__ void triangleSubtract(int n, bool lower, T *a, long long lda, const T *w, long long ldw) {
    const long long total = static_cast<long long>(n) * n;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % n), j = int(e / n);
        if (lower ? i >= j : i <= j) {
            at(a, lda, i, j) = sub(at(a, lda, i, j), at(w, ldw, i, j));
        }
    }
}

template<typename T>
__global__ void qrKernel(int m, int n, T *a, int lda, T *tau) {
    __shared__ BlockShared s;
    qrBlock(m, n, a, lda, tau, s);
}

template<typename T>
__global__ void applyQKernel(bool left, bool conjTrans, int m, int n, int k, const T *a, int lda, const T *tau, T *c,
                             int ldc) {
    applyQBlock(left, conjTrans, m, n, k, a, lda, tau, c, ldc);
}

template<typename T>
__global__ void orgqrKernel(int m, int n, int k, T *a, int lda, const T *tau) {
    orgqrBlock(m, n, k, a, lda, tau);
}

template<typename T>
__global__ void larftKernel(int m, int k, const T *v, int ldv, const T *tau, T *t, int ldt) {
    larftBlock(m, k, v, ldv, tau, t, ldt);
}

// V (rows x k) = the reflectors of A's columns with explicit unit diagonal and zeros above
template<typename T>
__global__ void explicitReflectors(int rows, int k, const T *a, long long lda, T *v) {
    const long long total = static_cast<long long>(rows) * k;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % rows), j = int(e / rows);
        v[e] = i > j ? at(a, lda, i, j) : make<T>(i == j ? 1 : 0);
    }
}

// columns [col0, n) of A (m rows) set to identity columns
template<typename T>
__global__ void identityColumns(int m, int col0, int n, T *a, long long lda) {
    const long long total = static_cast<long long>(m) * (n - col0);
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % m), j = col0 + int(e / m);
        at(a, lda, i, j) = make<T>(i == j ? 1 : 0);
    }
}

// rows [0, rows) of columns [col0, col1) of A set to zero
template<typename T>
__global__ void zeroBlock(int rows, int col0, int col1, T *a, long long lda) {
    const long long total = static_cast<long long>(rows) * (col1 - col0);
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        at(a, lda, int(e % rows), col0 + int(e / rows)) = make<T>(0);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// LU

constexpr int LU_BLOCK = 64;

template<typename T>
cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, T *a, int lda, int *ipiv, int *info) {
    CHECK_SOLVER(h);
    if (m < 0 || n < 0 || lda < std::max(1, m)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    const int kmin = std::min(m, n);
    if (kmin == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (kmin <= 4 * LU_BLOCK) {
        luKernel<<<1, THREADS, 0, h->stream>>>(m, n, a, lda, ipiv, 0, info, 0);
        return launchedSolver();
    }
    const T one = make<T>(1), minus = make<T>(-1);
    for (int j = 0; j < kmin; j += LU_BLOCK) {
        const int jb = std::min(LU_BLOCK, kmin - j);
        luKernel<<<1, THREADS, 0, h->stream>>>(m - j, jb, a + j + static_cast<long long>(j) * lda, lda,
                                               ipiv != nullptr ? ipiv + j : nullptr, j, info, j);
        SOLVER_TRY(launchedSolver());
        if (ipiv != nullptr) {
            if (j > 0) {
                laswpKernel<<<blocksFor(j), 256, 0, h->stream>>>(a, lda, 0, j, ipiv, j, j + jb, true);
            }
            if (j + jb < n) {
                laswpKernel<<<blocksFor(n - j - jb), 256, 0, h->stream>>>(a, lda, j + jb, n, ipiv, j, j + jb, true);
            }
            SOLVER_TRY(launchedSolver());
        }
        if (j + jb < n) {
            T *a12 = a + j + static_cast<long long>(j + jb) * lda;
            BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, jb,
                             n - j - jb, a + j + static_cast<long long>(j) * lda, lda, a12, lda));
            BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, CUBLAS_OP_N, m - j - jb, n - j - jb, jb, minus,
                          a + j + jb + static_cast<long long>(j) * lda, lda, a12, lda, one,
                          a12 + jb, lda));
        }
    }
    return CUSOLVER_STATUS_SUCCESS;
}

template<typename T>
cusolverStatus_t getrs(cusolverDnHandle_t h, cublasOperation_t trans, int n, int nrhs, const T *a, int lda,
                       const int *ipiv, T *b, int ldb, int *info) {
    CHECK_SOLVER(h);
    if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || ipiv == nullptr) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (n == 0 || nrhs == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (trans == CUBLAS_OP_N) {
        laswpKernel<<<blocksFor(nrhs), 256, 0, h->stream>>>(b, ldb, 0, nrhs, ipiv, 0, n, true);
        SOLVER_TRY(launchedSolver());
        BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, nrhs, a, lda,
                         b, ldb));
        BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, a,
                         lda, b, ldb));
        return CUSOLVER_STATUS_SUCCESS;
    }
    BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, trans, CUBLAS_DIAG_NON_UNIT, n, nrhs, a, lda, b,
                     ldb));
    BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_UNIT, n, nrhs, a, lda, b, ldb));
    laswpKernel<<<blocksFor(nrhs), 256, 0, h->stream>>>(b, ldb, 0, nrhs, ipiv, 0, n, false);
    return launchedSolver();
}

// ---------------------------------------------------------------------------------------------------------------------
// Cholesky

constexpr int CHOLESKY_BLOCK = 128;

int potrfWork(int n) {
    return n > 2 * CHOLESKY_BLOCK ? CHOLESKY_BLOCK * CHOLESKY_BLOCK : 1;
}

template<typename T>
cusolverStatus_t potrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T *a, int lda, T *work, int lwork,
                       int *info) {
    CHECK_SOLVER(h);
    if (n < 0 || lda < std::max(1, n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (n == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    const bool lower = uplo == CUBLAS_FILL_MODE_LOWER;
    if (n <= 2 * CHOLESKY_BLOCK) {
        choleskyKernel<<<1, THREADS, 0, h->stream>>>(lower, n, static_cast<T *const *>(nullptr), a, lda, info, false,
                                                     0);
        return launchedSolver();
    }
    if (work == nullptr || lwork < potrfWork(n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    const T one = make<T>(1), zero = make<T>(0), minus = make<T>(-1);
    auto block = [&](int i, int j) { return a + i + static_cast<long long>(j) * lda; };
    for (int j = 0; j < n; j += CHOLESKY_BLOCK) {
        const int jb = std::min(CHOLESKY_BLOCK, n - j);
        const int rest = n - j - jb;
        if (j > 0) {
            // A11 -= A10 A10^H (lower) / A01^H A01 (upper), on the triangle only
            if (lower) {
                BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, CUBLAS_OP_C, jb, jb, j, one, block(j, 0), lda, block(j, 0), lda,
                              zero, work, jb));
            } else {
                BLAS_TRY(gemm(h->stream, CUBLAS_OP_C, CUBLAS_OP_N, jb, jb, j, one, block(0, j), lda, block(0, j), lda,
                              zero, work, jb));
            }
            triangleSubtract<<<blocksFor(static_cast<long long>(jb) * jb), 256, 0, h->stream>>>(jb, lower,
                                                                                               block(j, j), lda,
                                                                                               work, jb);
            SOLVER_TRY(launchedSolver());
        }
        choleskyKernel<<<1, THREADS, 0, h->stream>>>(lower, jb, static_cast<T *const *>(nullptr), block(j, j), lda,
                                                     info, false, j);
        SOLVER_TRY(launchedSolver());
        if (rest == 0) {
            continue;
        }
        if (lower) {
            if (j > 0) {
                BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, CUBLAS_OP_C, rest, jb, j, minus, block(j + jb, 0), lda,
                              block(j, 0), lda, one, block(j + jb, j), lda));
            }
            BLAS_TRY(trsmOne(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, rest, jb,
                             block(j, j), lda, block(j + jb, j), lda));
        } else {
            if (j > 0) {
                BLAS_TRY(gemm(h->stream, CUBLAS_OP_C, CUBLAS_OP_N, jb, rest, j, minus, block(0, j), lda,
                              block(0, j + jb), lda, one, block(j, j + jb), lda));
            }
            BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, jb, rest,
                             block(j, j), lda, block(j, j + jb), lda));
        }
    }
    return CUSOLVER_STATUS_SUCCESS;
}

template<typename T>
cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs, const T *a, int lda, T *b,
                       int ldb, int *info) {
    CHECK_SOLVER(h);
    if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (n == 0 || nrhs == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (uplo == CUBLAS_FILL_MODE_LOWER) {
        BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, a, lda, b, ldb));
        BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, n, nrhs, a, lda, b, ldb));
    } else {
        BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, n, nrhs, a, lda, b, ldb));
        BLAS_TRY(trsmOne(h, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, a, lda, b, ldb));
    }
    return CUSOLVER_STATUS_SUCCESS;
}

template<typename T>
cusolverStatus_t potrfBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T *a[], int lda, int *info,
                              int batch) {
    CHECK_SOLVER(h);
    if (n < 0 || lda < std::max(1, n) || batch < 0) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info, batch));
    if (n == 0 || batch == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    choleskyKernel<<<batch, THREADS, 0, h->stream>>>(uplo == CUBLAS_FILL_MODE_LOWER, n, a, static_cast<T *>(nullptr),
                                                     lda, info, true, 0);
    return launchedSolver();
}

template<typename T>
cusolverStatus_t potrsBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs, T *a[], int lda, T *b[],
                              int ldb, int *info, int batch) {
    CHECK_SOLVER(h);
    if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || batch < 0) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (n == 0 || nrhs == 0 || batch == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    potrsBatchedKernel<<<batch, THREADS, 0, h->stream>>>(uplo == CUBLAS_FILL_MODE_LOWER, n, nrhs, a, lda, b, ldb);
    return launchedSolver();
}

// ---------------------------------------------------------------------------------------------------------------------
// QR

constexpr int QR_BLOCK = 32;

// workspace (elements) of the blocked QR routines: reflectors, triangular factor and two products
long long qrWork(int rows, int cols) {
    return static_cast<long long>(rows) * QR_BLOCK + QR_BLOCK * QR_BLOCK + 2LL * QR_BLOCK * std::max(rows, cols) + 1;
}

/**
 * C = op(H) C (left) or C op(H) (right) for the block reflector H = I - V T V^H of the reflectors in A's columns
 * col..col+b (rows from col): op(H) = H^H if conjTrans. Work: qrWork.
 */
template<typename T>
cusolverStatus_t applyBlockReflector(cusolverDnHandle_t h, bool left, bool conjTrans, int mq, int col, int b,
                                     const T *a, int lda, const T *tau, T *c, int ldc, int crows, int ccols, T *work) {
    const int vrows = mq - col;
    T *v = work, *t = v + static_cast<long long>(vrows) * QR_BLOCK, *w = t + QR_BLOCK * QR_BLOCK;
    T *w2 = w + static_cast<long long>(QR_BLOCK) * std::max(crows, ccols);
    explicitReflectors<<<blocksFor(static_cast<long long>(vrows) * b), 256, 0, h->stream>>>(
        vrows, b, a + col + static_cast<long long>(col) * lda, lda, v);
    larftKernel<<<1, THREADS, 0, h->stream>>>(vrows, b, v, vrows, tau + col, t, QR_BLOCK);
    SOLVER_TRY(launchedSolver());
    const T one = make<T>(1), zero = make<T>(0), minus = make<T>(-1);
    const cublasOperation_t t_op = conjTrans ? CUBLAS_OP_C : CUBLAS_OP_N;
    if (left) {
        // C (rows col.., ccols) -= V op(T) (V^H C)
        T *cc = c + col;
        BLAS_TRY(gemm(h->stream, CUBLAS_OP_C, CUBLAS_OP_N, b, ccols, vrows, one, v, vrows, cc, ldc, zero, w, b));
        BLAS_TRY(gemm(h->stream, t_op, CUBLAS_OP_N, b, ccols, b, one, t, QR_BLOCK, w, b, zero, w2, b));
        BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, CUBLAS_OP_N, vrows, ccols, b, minus, v, vrows, w2, b, one, cc, ldc));
    } else {
        // C (crows, columns col..) -= ((C V) op(T)) V^H
        T *cc = c + static_cast<long long>(col) * ldc;
        BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, CUBLAS_OP_N, crows, b, vrows, one, cc, ldc, v, vrows, zero, w, crows));
        BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, t_op, crows, b, b, one, w, crows, t, QR_BLOCK, zero, w2, crows));
        BLAS_TRY(gemm(h->stream, CUBLAS_OP_N, CUBLAS_OP_C, crows, vrows, b, minus, w2, crows, v, vrows, one, cc,
                      ldc));
    }
    return CUSOLVER_STATUS_SUCCESS;
}

bool smallQr(int m, int n) {
    return n <= QR_BLOCK || static_cast<long long>(m) * n <= 256 * 256;
}

template<typename T>
cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, T *a, int lda, T *tau, T *work, int lwork, int *info) {
    CHECK_SOLVER(h);
    if (m < 0 || n < 0 || lda < std::max(1, m)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (m == 0 || n == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (smallQr(m, n)) {
        qrKernel<<<1, THREADS, 0, h->stream>>>(m, n, a, lda, tau);
        return launchedSolver();
    }
    if (work == nullptr || lwork < qrWork(m, n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    const int kmin = std::min(m, n);
    for (int j = 0; j < kmin; j += QR_BLOCK) {
        const int jb = std::min(QR_BLOCK, kmin - j);
        qrKernel<<<1, THREADS, 0, h->stream>>>(m - j, jb, a + j + static_cast<long long>(j) * lda, lda, tau + j);
        SOLVER_TRY(launchedSolver());
        if (j + jb < n) {
            SOLVER_TRY(applyBlockReflector(h, true, true, m, j, jb, a, lda, tau, a + static_cast<long long>(j + jb) * lda,
                                           lda, m, n - j - jb, work));
        }
    }
    return CUSOLVER_STATUS_SUCCESS;
}

template<typename T>
cusolverStatus_t ormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                       const T *a, int lda, const T *tau, T *c, int ldc, T *work, int lwork, int *info) {
    CHECK_SOLVER(h);
    const bool left = side == CUBLAS_SIDE_LEFT;
    const int mq = left ? m : n;
    if (m < 0 || n < 0 || k < 0 || k > mq || lda < std::max(1, mq) || ldc < std::max(1, m)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (m == 0 || n == 0 || k == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    const bool conj_trans = trans != CUBLAS_OP_N;
    if (k <= QR_BLOCK || static_cast<long long>(mq) * k <= 256 * 256) {
        applyQKernel<<<1, THREADS, 0, h->stream>>>(left, conj_trans, m, n, k, a, lda, tau, c, ldc);
        return launchedSolver();
    }
    if (work == nullptr || lwork < qrWork(mq, std::max(m, n))) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    const bool forward = left == conj_trans;
    const int blocks = (k + QR_BLOCK - 1) / QR_BLOCK;
    for (int step = 0; step < blocks; step++) {
        const int col = (forward ? step : blocks - 1 - step) * QR_BLOCK;
        const int b = std::min(QR_BLOCK, k - col);
        SOLVER_TRY(applyBlockReflector(h, left, conj_trans, mq, col, b, a, lda, tau, c, ldc, m, n, work));
    }
    return CUSOLVER_STATUS_SUCCESS;
}

template<typename T>
cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k, T *a, int lda, const T *tau, T *work, int lwork,
                       int *info) {
    CHECK_SOLVER(h);
    if (m < 0 || n < 0 || n > m || k < 0 || k > n || lda < std::max(1, m)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info));
    if (n == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (k <= QR_BLOCK || static_cast<long long>(m) * n <= 256 * 256) {
        orgqrKernel<<<1, THREADS, 0, h->stream>>>(m, n, k, a, lda, tau);
        return launchedSolver();
    }
    if (work == nullptr || lwork < qrWork(m, n)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    if (k < n) {
        identityColumns<<<blocksFor(static_cast<long long>(m) * (n - k)), 256, 0, h->stream>>>(m, k, n, a, lda);
    }
    const int blocks = (k + QR_BLOCK - 1) / QR_BLOCK;
    for (int block = blocks - 1; block >= 0; block--) {
        const int col = block * QR_BLOCK, b = std::min(QR_BLOCK, k - col);
        if (col + b < n) {
            // the block reflector on the columns already formed (rows col..)
            SOLVER_TRY(applyBlockReflector(h, true, false, m, col, b, a, lda, tau,
                                           a + static_cast<long long>(col + b) * lda, lda, m, n - col - b, work));
        }
        // the block's own columns, then zeros above
        orgqrKernel<<<1, THREADS, 0, h->stream>>>(m - col, b, b, a + col + static_cast<long long>(col) * lda, lda,
                                                  tau + col);
        if (col > 0) {
            zeroBlock<<<blocksFor(static_cast<long long>(col) * b), 256, 0, h->stream>>>(col, col, col + b, a, lda);
        }
        SOLVER_TRY(launchedSolver());
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// Jacobi eigen decomposition and SVD

template<typename T>
double machineEpsilon() {
    return std::is_same_v<Real<T>, double> ? DBL_EPSILON : FLT_EPSILON;
}

// W (n x n per matrix) = the Hermitian matrix A from its uplo triangle; V = I
template<typename T>
__global__ void eigenSetup(int n, const T *a, long long lda, long long strideA, bool lower, T *w, T *v) {
    const long long total = static_cast<long long>(n) * n;
    const T *am = a + blockIdx.y * strideA;
    T *wm = w + blockIdx.y * total;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % n), j = int(e / n);
        const bool stored = lower ? i >= j : i <= j;
        T value = stored ? at(am, lda, i, j) : conjugate(at(am, lda, j, i));
        if (i == j) {
            value = make<T>(re(value));
        }
        wm[e] = value;
        if (v != nullptr) {
            v[blockIdx.y * total + e] = make<T>(i == j ? 1 : 0);
        }
    }
}

// eigenvalues ascending (or as they are) into W, the eigenvectors in the same order into A
template<typename T>
__global__ void eigenFinish(int n, const T *w, const T *v, T *a, long long lda, long long strideA, Real<T> *values,
                            long long strideValues, bool sort) {
    const long long total = static_cast<long long>(n) * n;
    const T *wm = w + blockIdx.x * total;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const Real<T> value = re(wm[j + static_cast<long long>(j) * n]);
        int rank = j;
        if (sort) {
            rank = 0;
            for (int k = 0; k < n; k++) {
                const Real<T> other = re(wm[k + static_cast<long long>(k) * n]);
                rank += other < value || (other == value && k < j);
            }
        }
        values[blockIdx.x * strideValues + rank] = value;
        if (v != nullptr) {
            T *am = a + blockIdx.x * strideA;
            const T *vm = v + blockIdx.x * total;
            for (int i = 0; i < n; i++) {
                at(am, lda, i, rank) = vm[i + static_cast<long long>(j) * n];
            }
        }
    }
}

// workspace (elements of T) per matrix of the eigen decomposition
template<typename T>
long long eigenWork(int n) {
    const long long rotations = (static_cast<long long>((n + 1) / 2) * sizeof(Rotation) + sizeof(T) - 1) / sizeof(T);
    return 2LL * n * n + rotations + 1;
}

/**
 * Jacobi eigen decomposition of batch Hermitian matrices (A's uplo triangle): eigenvalues into values, eigenvectors
 * (jobz) into A. info (per matrix if infoPerMatrix): 0, or n + 1 if a sweep limit was reached.
 */
template<typename T>
cusolverStatus_t eigen(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, T *a, long long lda,
                       long long strideA, Real<T> *values, long long strideValues, int batch, T *work, long long lwork,
                       int *info, bool infoPerMatrix, double tolerance, int maxSweeps, bool sort, int *sweepsOut,
                       double *residualOut) {
    CHECK_SOLVER(h);
    if (n < 0 || lda < std::max(1, n) || batch < 0 || values == nullptr) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info, infoPerMatrix ? batch : 1));
    if (n == 0 || batch == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (work == nullptr || lwork < eigenWork<T>(n) * batch) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    const long long square = static_cast<long long>(n) * n;
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;
    T *w = work, *v = vectors ? work + square * batch : nullptr;
    auto *rotations = reinterpret_cast<Rotation *>(work + 2 * square * batch);
    int *count = static_cast<int *>(librecuda_blas_scratch(h->stream, sizeof(int)));
    if (count == nullptr) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    }
    eigenSetup<<<dim3(blocksFor(square), batch), 256, 0, h->stream>>>(n, a, lda, strideA,
                                                                      uplo == CUBLAS_FILL_MODE_LOWER, w, v);
    SOLVER_TRY(launchedSolver());
    const int pairs = (n + 1) / 2, rounds = 2 * pairs - 1;
    const double tol = std::max(tolerance > 0 ? tolerance : 0.0, machineEpsilon<T>());
    int sweep = 0;
    bool converged = n == 1;
    for (; sweep < maxSweeps && !converged; sweep++) {
        if (cudaMemsetAsync(count, 0, sizeof(int), h->stream) != cudaSuccess) {
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        }
        for (int round = 0; round < rounds; round++) {
            jacobiEigenRows<<<dim3(pairs, batch), THREADS, 0, h->stream>>>(n, w, n, square, round, tol, rotations,
                                                                           count);
            jacobiEigenColumns<<<dim3(pairs, batch), THREADS, 0, h->stream>>>(n, w, n, square, v, n, square, round,
                                                                              rotations);
        }
        SOLVER_TRY(launchedSolver());
        int rotated = 0;
        if (cudaMemcpyAsync(&rotated, count, sizeof(int), cudaMemcpyDeviceToHost, h->stream) != cudaSuccess ||
            cudaStreamSynchronize(h->stream) != cudaSuccess) {
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        }
        converged = rotated == 0;
    }
    eigenFinish<<<batch, THREADS, 0, h->stream>>>(n, w, v, a, lda, strideA, values, strideValues, sort);
    SOLVER_TRY(launchedSolver());
    if (!converged && info != nullptr) {
        // like cuSOLVER's Jacobi: n + 1 when the sweep limit was reached
        std::vector<int> failed(infoPerMatrix ? batch : 1, n + 1);
        if (cudaMemcpyAsync(info, failed.data(), failed.size() * sizeof(int), cudaMemcpyHostToDevice, h->stream) !=
            cudaSuccess || cudaStreamSynchronize(h->stream) != cudaSuccess) {
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        }
    }
    if (sweepsOut != nullptr) {
        *sweepsOut = sweep;
    }
    if (residualOut != nullptr) {
        *residualOut = 0;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// X (rows x rows per matrix, first cols columns used) = A (m >= n) or A^H; V = I (cols x cols)
template<typename T>
__global__ void svdSetup(int m, int n, const T *a, long long lda, long long strideA, bool transpose, int rows,
                         int cols, T *x, T *v) {
    const long long xsize = static_cast<long long>(rows) * rows, vsize = static_cast<long long>(cols) * cols;
    const T *am = a + blockIdx.y * strideA;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < xsize;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = int(e % rows), j = int(e / rows);
        T value = make<T>(0);
        if (j < cols) {
            value = transpose ? conjugate(at(am, lda, j, i)) : at(am, lda, i, j);
        }
        x[blockIdx.y * xsize + e] = value;
        if (e < vsize) {
            v[blockIdx.y * vsize + e] = make<T>(e % cols == e / cols ? 1 : 0);
        }
    }
    (void) m;
    (void) n;
}

template<typename T>
__device__ inline double machineEpsilonDevice();

template<>
__device__ inline double machineEpsilonDevice<float>() { return FLT_EPSILON; }
template<>
__device__ inline double machineEpsilonDevice<double>() { return DBL_EPSILON; }
template<>
__device__ inline double machineEpsilonDevice<cuComplex>() { return FLT_EPSILON; }
template<>
__device__ inline double machineEpsilonDevice<cuDoubleComplex>() { return DBL_EPSILON; }

/**
 * After the sweeps: singular values (descending) into s, X's columns normalized and ordered into Y (rows x rows, first
 * cols columns; the others, and those of zero singular values, completed to an orthonormal basis when fullBasis), V's
 * columns ordered into W (cols x cols)
 */
template<typename T>
__global__ void svdFinish(int rows, int cols, const T *x, const T *v, T *y, T *w, Real<T> *s, long long strideS,
                          int *need, bool fullBasis, bool sort) {
    __shared__ BlockShared shared;
    __shared__ double norms[1024];
    __shared__ double sorted[1024];
    __shared__ int ranks[1024];
    const long long xsize = static_cast<long long>(rows) * rows, vsize = static_cast<long long>(cols) * cols;
    const T *xm = x + blockIdx.x * xsize;
    T *ym = y + blockIdx.x * xsize;
    int *needm = need + static_cast<long long>(blockIdx.x) * rows;
    for (int j = 0; j < cols; j++) {
        double part = 0;
        for (int i = threadIdx.x; i < rows; i += blockDim.x) {
            part += norm2(xm[i + static_cast<long long>(j) * rows]);
        }
        const double norm = sqrt(blockSumReal(part, shared));
        if (threadIdx.x == 0) {
            norms[j] = norm;
        }
    }
    __syncthreads();
    // ranks: descending singular values (ties by position)
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        int rank = j;
        if (sort) {
            rank = 0;
            for (int k = 0; k < cols; k++) {
                rank += norms[k] > norms[j] || (norms[k] == norms[j] && k < j);
            }
        }
        ranks[j] = rank;
        sorted[rank] = norms[j];
        s[blockIdx.x * strideS + rank] = Real<T>(norms[j]);
    }
    __syncthreads();
    double largest = 0;
    for (int j = 0; j < cols; j++) {
        largest = fmax(largest, norms[j]);
    }
    // singular values below rounding noise have no direction of their own
    const double small = largest * machineEpsilonDevice<T>() * rows;
    for (long long e = threadIdx.x; e < static_cast<long long>(rows) * cols; e += blockDim.x) {
        const int i = int(e % rows), j = int(e / rows);
        const double norm = norms[j];
        ym[i + static_cast<long long>(ranks[j]) * rows] =
            norm > small && norm > 0 ? scale(xm[i + static_cast<long long>(j) * rows], Real<T>(1 / norm))
                                     : make<T>(0);
    }
    if (w != nullptr) {
        const T *vm = v + blockIdx.x * vsize;
        T *wm = w + blockIdx.x * vsize;
        for (long long e = threadIdx.x; e < vsize; e += blockDim.x) {
            const int i = int(e % cols), j = int(e / cols);
            wm[i + static_cast<long long>(ranks[j]) * cols] = vm[e];
        }
    }
    // the columns to complete: those of zero singular values, and the ones beyond cols for a full basis
    bool any = false;
    for (int j = 0; j < rows; j++) {
        const bool needed = j < cols ? !(sorted[j] > small && sorted[j] > 0) : fullBasis;
        any = any || needed;
        if (threadIdx.x == 0) {
            needm[j] = needed;
        }
    }
    __syncthreads();
    if (any) {
        completeBasisBlock(rows, fullBasis ? rows : cols, ym, rows, needm, shared);
    }
}

/**
 * Stores the decomposition: U (m x ucols) and V (n x vcols) or V^H (vrows x n) from Y (rows x rows) and W (cols x cols)
 */
template<typename T>
__global__ void svdStore(int m, int n, bool transposed, int rows, int cols, const T *y, const T *w, T *u, long long ldu,
                         long long strideU, int ucols, T *v, long long ldv, long long strideV, int vcount,
                         bool vHermitian) {
    const long long ysize = static_cast<long long>(rows) * rows, wsize = static_cast<long long>(cols) * cols;
    const T *ym = y + blockIdx.y * ysize, *wm = w + blockIdx.y * wsize;
    // A = Y S W^H (m >= n) or A^H = Y S W^H, so A = W S Y^H
    const T *um = transposed ? wm : ym;
    const int uld = transposed ? cols : rows;
    const T *vm = transposed ? ym : wm;
    const int vld = transposed ? rows : cols;
    const long long utotal = u != nullptr ? static_cast<long long>(m) * ucols : 0;
    const long long vtotal = v != nullptr ? static_cast<long long>(n) * vcount : 0;
    for (long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; e < utotal + vtotal;
         e += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (e < utotal) {
            const int i = int(e % m), j = int(e / m);
            u[blockIdx.y * strideU + i + j * ldu] = um[i + static_cast<long long>(j) * uld];
        } else {
            const long long f = e - utotal;
            const int i = int(f % n), j = int(f / n);
            const T value = vm[i + static_cast<long long>(j) * vld];
            if (vHermitian) {
                // V^H: row j, column i
                v[blockIdx.y * strideV + j + static_cast<long long>(i) * ldv] = conjugate(value);
            } else {
                v[blockIdx.y * strideV + i + static_cast<long long>(j) * ldv] = value;
            }
        }
    }
}

// workspace (elements of T) per matrix of the SVD
template<typename T>
long long svdWork(int m, int n) {
    const long long rows = std::max(m, n), cols = std::min(m, n);
    return 2 * rows * rows + 2 * cols * cols + (rows * sizeof(int) + sizeof(T) - 1) / sizeof(T) + 1;
}

/**
 * Jacobi SVD of batch m x n matrices: s (min(m, n) per matrix, descending), U (m x ucols, ucols 0: none) and V
 * (n x vcount) or V^H (vcount x n, vHermitian). info: 0, or min(m, n) + 1 if a sweep limit was reached.
 */
template<typename T>
cusolverStatus_t svd(cusolverDnHandle_t h, int m, int n, const T *a, long long lda, long long strideA, Real<T> *s,
                     long long strideS, T *u, long long ldu, long long strideU, int ucols, T *v, long long ldv,
                     long long strideV, int vcount, bool vHermitian, int batch, T *work, long long lwork, int *info,
                     bool infoPerMatrix, double tolerance, int maxSweeps, bool sort, int *sweepsOut) {
    CHECK_SOLVER(h);
    if (m < 0 || n < 0 || lda < std::max(1, m) || batch < 0 || s == nullptr) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    SOLVER_TRY(clearInfo(h, info, infoPerMatrix ? batch : 1));
    if (m == 0 || n == 0 || batch == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    if (work == nullptr || lwork < svdWork<T>(m, n) * batch) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    const bool transposed = m < n;
    const int rows = std::max(m, n), cols = std::min(m, n);
    if (cols > 1024) {
        return CUSOLVER_STATUS_NOT_SUPPORTED;
    }
    const long long xsize = static_cast<long long>(rows) * rows, vsize = static_cast<long long>(cols) * cols;
    T *x = work, *vw = x + xsize * batch, *y = vw + vsize * batch, *w = y + xsize * batch;
    int *need = reinterpret_cast<int *>(w + vsize * batch);
    int *count = static_cast<int *>(librecuda_blas_scratch(h->stream, sizeof(int)));
    if (count == nullptr) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    }
    svdSetup<<<dim3(blocksFor(xsize), batch), 256, 0, h->stream>>>(m, n, a, lda, strideA, transposed, rows, cols, x,
                                                                   vw);
    SOLVER_TRY(launchedSolver());
    const int pairs = (cols + 1) / 2, rounds = 2 * pairs - 1;
    // the off-diagonal product left after a rotation is rounding noise of about rows * eps
    const double tol = std::max(tolerance > 0 ? tolerance : 0.0, machineEpsilon<T>() * std::sqrt(double(rows)) * 4);
    int sweep = 0;
    bool converged = cols == 1;
    for (; sweep < maxSweeps && !converged; sweep++) {
        if (cudaMemsetAsync(count, 0, sizeof(int), h->stream) != cudaSuccess) {
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        }
        for (int round = 0; round < rounds; round++) {
            jacobiSvdRound<<<dim3(pairs, batch), THREADS, 0, h->stream>>>(rows, cols, x, rows, xsize, vw, cols, vsize,
                                                                          round, tol, count);
        }
        SOLVER_TRY(launchedSolver());
        int rotated = 0;
        if (cudaMemcpyAsync(&rotated, count, sizeof(int), cudaMemcpyDeviceToHost, h->stream) != cudaSuccess ||
            cudaStreamSynchronize(h->stream) != cudaSuccess) {
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        }
        converged = rotated == 0;
    }
    // a full basis of Y is needed when U (or V for m < n) asks for more columns than cols
    const bool full = transposed ? vcount > cols : ucols > cols;
    svdFinish<<<batch, THREADS, 0, h->stream>>>(rows, cols, x, vw, y, w, s, strideS, need, full, sort);
    SOLVER_TRY(launchedSolver());
    if (u != nullptr || v != nullptr) {
        svdStore<<<dim3(blocksFor(static_cast<long long>(rows) * rows * 2), batch), 256, 0, h->stream>>>(
            m, n, transposed, rows, cols, y, w, u, ldu, strideU, ucols, v, ldv, strideV, vcount, vHermitian);
        SOLVER_TRY(launchedSolver());
    }
    if (!converged && info != nullptr) {
        std::vector<int> failed(infoPerMatrix ? batch : 1, cols + 1);
        if (cudaMemcpyAsync(info, failed.data(), failed.size() * sizeof(int), cudaMemcpyHostToDevice, h->stream) !=
            cudaSuccess || cudaStreamSynchronize(h->stream) != cudaSuccess) {
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        }
    }
    if (sweepsOut != nullptr) {
        *sweepsOut = sweep;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

int clampWork(long long elements) {
    return static_cast<int>(std::min<long long>(elements, std::numeric_limits<int>::max()));
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// handles and parameters

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
    if (handle == nullptr) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    cublasHandle_t blas{};
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        return CUSOLVER_STATUS_NOT_INITIALIZED;
    }
    *handle = new cusolverDnContext{SOLVER_MAGIC, nullptr, blas};
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    CHECK_SOLVER(handle);
    cublasDestroy(handle->blas);
    handle->magic = 0;
    delete handle;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) {
    CHECK_SOLVER(handle);
    handle->stream = streamId;
    cublasSetStream(handle->blas, streamId);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) {
    CHECK_SOLVER(handle);
    if (streamId == nullptr) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    *streamId = handle->stream;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t *params) {
    if (params == nullptr) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    *params = new cusolverDnParams{};
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
    delete params;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetAdvOptions(cusolverDnParams_t params, cusolverDnFunction_t, cusolverAlgMode_t) {
    return params != nullptr ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INVALID_VALUE;
}

#define JACOBI_INFO(Name, name, Type)                                                                                        \
cusolverStatus_t cusolverDnCreate##Name##Info(Type *info) {                                                            \
    if (info == nullptr) {                                                                                             \
        return CUSOLVER_STATUS_INVALID_VALUE;                                                                          \
    }                                                                                                                  \
    *info = new std::remove_pointer_t<Type>{0, 100, 1, 0, 0};                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDnDestroy##Name##Info(Type info) {                                                            \
    delete info;                                                                                                       \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDnX##name##SetTolerance(Type info, double tolerance) {                                        \
    if (info == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                         \
    info->tolerance = tolerance;                                                                                       \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDnX##name##SetMaxSweeps(Type info, int max_sweeps) {                                          \
    if (info == nullptr || max_sweeps <= 0) return CUSOLVER_STATUS_INVALID_VALUE;                                      \
    info->maxSweeps = max_sweeps;                                                                                      \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDnX##name##GetSweeps(cusolverDnHandle_t, Type info, int *executed_sweeps) {                   \
    if (info == nullptr || executed_sweeps == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                           \
    *executed_sweeps = info->sweeps;                                                                                   \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDnX##name##GetResidual(cusolverDnHandle_t, Type info, double *residual) {                     \
    if (info == nullptr || residual == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                  \
    *residual = info->residual;                                                                                        \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}

JACOBI_INFO(Syevj, syevj, syevjInfo_t)
JACOBI_INFO(Gesvdj, gesvdj, gesvdjInfo_t)

cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) {
    if (info == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;
    info->sortEig = sort_eig;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd) {
    if (info == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;
    info->sortEig = sort_svd;
    return CUSOLVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// the typed API

namespace {

// Jacobi parameters (nullptr: defaults)
template<typename Info>
double tolOf(Info info) { return info != nullptr ? info->tolerance : 0; }
template<typename Info>
int sweepsOf(Info info) { return info != nullptr ? info->maxSweeps : 100; }
template<typename Info>
bool sortOf(Info info) { return info == nullptr || info->sortEig != 0; }

// columns of U for gesvd's job ('A': all m, 'S': min(m, n), 'N': none)
int jobColumns(signed char job, int full, int thin) {
    return job == 'A' ? full : job == 'S' ? thin : job == 'N' ? 0 : -1;
}

} // namespace

#define SOLVER_TYPED(P, T)                                                                                             \
cusolverStatus_t cusolverDn##P##getrf_bufferSize(cusolverDnHandle_t h, int, int, T *, int, int *lwork) {               \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = 1;                                                                                                        \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##getrf(cusolverDnHandle_t h, int m, int n, T *A, int lda, T *, int *devIpiv,            \
                                      int *devInfo) { return getrf(h, m, n, A, lda, devIpiv, devInfo); }               \
cusolverStatus_t cusolverDn##P##getrs(cusolverDnHandle_t h, cublasOperation_t trans, int n, int nrhs, const T *A,      \
                                      int lda, const int *devIpiv, T *B, int ldb, int *devInfo) {                      \
    return getrs(h, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo); }                                               \
cusolverStatus_t cusolverDn##P##potrf_bufferSize(cusolverDnHandle_t h, cublasFillMode_t, int n, T *, int,              \
                                                 int *lwork) {                                                         \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = potrfWork(n);                                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##potrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T *A, int lda, T *work,      \
                                      int lwork, int *devInfo) { return potrf(h, uplo, n, A, lda, work, lwork,         \
                                                                              devInfo); }                              \
cusolverStatus_t cusolverDn##P##potrs(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs, const T *A,        \
                                      int lda, T *B, int ldb, int *devInfo) {                                          \
    return potrs(h, uplo, n, nrhs, A, lda, B, ldb, devInfo); }                                                         \
cusolverStatus_t cusolverDn##P##potrfBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T *Aarray[], int lda, \
                                             int *infoArray, int batchSize) {                                          \
    return potrfBatched(h, uplo, n, Aarray, lda, infoArray, batchSize); }                                              \
cusolverStatus_t cusolverDn##P##potrsBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs, T *A[],     \
                                             int lda, T *B[], int ldb, int *d_info, int batchSize) {                   \
    return potrsBatched(h, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize); }                                        \
cusolverStatus_t cusolverDn##P##geqrf_bufferSize(cusolverDnHandle_t h, int m, int n, T *, int, int *lwork) {           \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(qrWork(m, n));                                                                                  \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##geqrf(cusolverDnHandle_t h, int m, int n, T *A, int lda, T *TAU, T *Workspace,         \
                                      int Lwork, int *devInfo) { return geqrf(h, m, n, A, lda, TAU, Workspace, Lwork,  \
                                                                              devInfo); }                              \
cusolverStatus_t cusolverDn##P##syevd_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t, cublasFillMode_t, int n,     \
                                                 const T *, int, const Real<T> *, int *lwork) {                        \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(eigenWork<T>(n));                                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,      \
                                      T *A, int lda, Real<T> *W, T *work, int lwork, int *info) {                      \
    return eigen(h, jobz, uplo, n, A, lda, 0, W, 0, 1, work, lwork, info, false, 0, 100, true, nullptr, nullptr); }    \
cusolverStatus_t cusolverDn##P##syevj_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t, cublasFillMode_t, int n,     \
                                                 const T *, int, const Real<T> *, int *lwork, syevjInfo_t) {           \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(eigenWork<T>(n));                                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##syevj(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,      \
                                      T *A, int lda, Real<T> *W, T *work, int lwork, int *info, syevjInfo_t params) {  \
    return eigen(h, jobz, uplo, n, A, lda, 0, W, 0, 1, work, lwork, info, false, tolOf(params), sweepsOf(params),      \
                 sortOf(params), params != nullptr ? &params->sweeps : nullptr,                                        \
                 params != nullptr ? &params->residual : nullptr); }                                                   \
cusolverStatus_t cusolverDn##P##syevjBatched_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t, cublasFillMode_t,     \
                                                        int n, const T *, int, const Real<T> *, int *lwork,            \
                                                        syevjInfo_t, int batchSize) {                                  \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(eigenWork<T>(n) * batchSize);                                                                   \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##syevjBatched(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,      \
                                             int n, T *A, int lda, Real<T> *W, T *work, int lwork, int *info,          \
                                             syevjInfo_t params, int batchSize) {                                      \
    return eigen(h, jobz, uplo, n, A, lda, static_cast<long long>(lda) * n, W, n, batchSize, work, lwork, info, true,  \
                 tolOf(params), sweepsOf(params), sortOf(params), params != nullptr ? &params->sweeps : nullptr,       \
                 params != nullptr ? &params->residual : nullptr); }                                                   \
cusolverStatus_t cusolverDn##P##gesvd_bufferSize(cusolverDnHandle_t h, int m, int n, int *lwork) {                     \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(svdWork<T>(m, n));                                                                              \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##gesvd(cusolverDnHandle_t h, signed char jobu, signed char jobvt, int m, int n, T *A,    \
                                      int lda, Real<T> *S, T *U, int ldu, T *VT, int ldvt, T *work, int lwork,         \
                                      Real<T> *, int *info) {                                                          \
    const int thin = std::min(m, n), ucols = jobColumns(jobu, m, thin), vrows = jobColumns(jobvt, n, thin);            \
    if (ucols < 0 || vrows < 0) return CUSOLVER_STATUS_INVALID_VALUE;                                                  \
    return svd(h, m, n, A, lda, 0, S, 0, ucols > 0 ? U : nullptr, ldu, 0, ucols, vrows > 0 ? VT : nullptr, ldvt, 0,    \
               vrows, true, 1, work, lwork, info, false, 0, 100, true, nullptr); }                                     \
cusolverStatus_t cusolverDn##P##gesvdj_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t, int, int m, int n,          \
                                                  const T *, int, const Real<T> *, const T *, int, const T *, int,     \
                                                  int *lwork, gesvdjInfo_t) {                                          \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(svdWork<T>(m, n));                                                                              \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##gesvdj(cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n, T *A,     \
                                       int lda, Real<T> *S, T *U, int ldu, T *V, int ldv, T *work, int lwork,          \
                                       int *info, gesvdjInfo_t params) {                                               \
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;                                                             \
    const int thin = std::min(m, n);                                                                                   \
    return svd(h, m, n, A, lda, 0, S, 0, vectors ? U : nullptr, ldu, 0, vectors ? (econ ? thin : m) : 0,               \
               vectors ? V : nullptr, ldv, 0, vectors ? (econ ? thin : n) : 0, false, 1, work, lwork, info, false,     \
               tolOf(params), sweepsOf(params), sortOf(params), params != nullptr ? &params->sweeps : nullptr); }      \
cusolverStatus_t cusolverDn##P##gesvdjBatched_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t, int m, int n,        \
                                                         const T *, int, const Real<T> *, const T *, int, const T *,   \
                                                         int, int *lwork, gesvdjInfo_t, int batchSize) {               \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(svdWork<T>(m, n) * batchSize);                                                                  \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##gesvdjBatched(cusolverDnHandle_t h, cusolverEigMode_t jobz, int m, int n, T *A,        \
                                              int lda, Real<T> *S, T *U, int ldu, T *V, int ldv, T *work, int lwork,   \
                                              int *info, gesvdjInfo_t params, int batchSize) {                         \
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;                                                             \
    return svd(h, m, n, A, lda, static_cast<long long>(lda) * n, S, std::min(m, n), vectors ? U : nullptr, ldu,        \
               static_cast<long long>(ldu) * m, vectors ? m : 0, vectors ? V : nullptr, ldv,                           \
               static_cast<long long>(ldv) * n, vectors ? n : 0, false, batchSize, work, lwork, info, true,            \
               tolOf(params), sweepsOf(params), sortOf(params), params != nullptr ? &params->sweeps : nullptr); }      \
cusolverStatus_t cusolverDn##P##gesvdaStridedBatched_bufferSize(                                                       \
        cusolverDnHandle_t h, cusolverEigMode_t, int, int m, int n, const T *, int, long long, const Real<T> *,        \
        long long, const T *, int, long long, const T *, int, long long, int *lwork, int batchSize) {                  \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(svdWork<T>(m, n) * batchSize);                                                                  \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##gesvdaStridedBatched(                                                                  \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int rank, int m, int n, const T *d_A, int lda,                   \
        long long strideA, Real<T> *d_S, long long strideS, T *d_U, int ldu, long long strideU, T *d_V, int ldv,       \
        long long strideV, T *d_work, int lwork, int *d_info, double *h_R_nrmF, int batchSize) {                       \
    if (rank <= 0 || rank > std::min(m, n)) return CUSOLVER_STATUS_INVALID_VALUE;                                      \
    const bool vectors = jobz == CUSOLVER_EIG_MODE_VECTOR;                                                             \
    /* (the singular values beyond rank are computed too: the work array holds min(m, n) per matrix) */                \
    Real<T> *s_full = d_S;                                                                                             \
    if (rank < std::min(m, n) || strideS < std::min(m, n)) {                                                           \
        s_full = static_cast<Real<T> *>(librecuda_blas_scratch(h->stream, sizeof(Real<T>) * std::min(m, n) *           \
                                                               static_cast<size_t>(batchSize)));                       \
        if (s_full == nullptr) return CUSOLVER_STATUS_ALLOC_FAILED;                                                    \
    }                                                                                                                  \
    const long long s_stride = s_full == d_S ? strideS : std::min(m, n);                                               \
    cusolverStatus_t status = svd(h, m, n, d_A, lda, strideA, s_full, s_stride, vectors ? d_U : nullptr, ldu, strideU, \
                                  vectors ? rank : 0, vectors ? d_V : nullptr, ldv, strideV, vectors ? rank : 0,       \
                                  false, batchSize, d_work, lwork, d_info, true, 0, 100, true, nullptr);               \
    if (status == CUSOLVER_STATUS_SUCCESS && s_full != d_S &&                                                          \
        cudaMemcpy2DAsync(d_S, strideS * sizeof(Real<T>), s_full, s_stride * sizeof(Real<T>), rank * sizeof(Real<T>),  \
                          batchSize, cudaMemcpyDeviceToDevice, h->stream) != cudaSuccess) {                            \
        status = CUSOLVER_STATUS_EXECUTION_FAILED;                                                                     \
    }                                                                                                                  \
    if (h_R_nrmF != nullptr) {                                                                                         \
        for (int i = 0; i < batchSize; i++) h_R_nrmF[i] = 0;                                                           \
    }                                                                                                                  \
    return status; }

SOLVER_TYPED(S, float)
SOLVER_TYPED(D, double)
SOLVER_TYPED(C, cuComplex)
SOLVER_TYPED(Z, cuDoubleComplex)

// heevd / heevj are the complex syevd / syevj
#define HERMITIAN_ALIAS(P, T)                                                                                          \
cusolverStatus_t cusolverDn##P##heevd_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,  \
                                                 int n, const T *A, int lda, const Real<T> *W, int *lwork) {           \
    return cusolverDn##P##syevd_bufferSize(h, jobz, uplo, n, A, lda, W, lwork); }                                      \
cusolverStatus_t cusolverDn##P##heevd(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,      \
                                      T *A, int lda, Real<T> *W, T *work, int lwork, int *info) {                      \
    return cusolverDn##P##syevd(h, jobz, uplo, n, A, lda, W, work, lwork, info); }                                     \
cusolverStatus_t cusolverDn##P##heevj_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,  \
                                                 int n, const T *A, int lda, const Real<T> *W, int *lwork,             \
                                                 syevjInfo_t params) {                                                 \
    return cusolverDn##P##syevj_bufferSize(h, jobz, uplo, n, A, lda, W, lwork, params); }                              \
cusolverStatus_t cusolverDn##P##heevj(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,      \
                                      T *A, int lda, Real<T> *W, T *work, int lwork, int *info, syevjInfo_t params) {  \
    return cusolverDn##P##syevj(h, jobz, uplo, n, A, lda, W, work, lwork, info, params); }                             \
cusolverStatus_t cusolverDn##P##heevjBatched_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t jobz,                  \
                                                        cublasFillMode_t uplo, int n, const T *A, int lda,             \
                                                        const Real<T> *W, int *lwork, syevjInfo_t params,              \
                                                        int batchSize) {                                               \
    return cusolverDn##P##syevjBatched_bufferSize(h, jobz, uplo, n, A, lda, W, lwork, params, batchSize); }            \
cusolverStatus_t cusolverDn##P##heevjBatched(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,      \
                                             int n, T *A, int lda, Real<T> *W, T *work, int lwork, int *info,          \
                                             syevjInfo_t params, int batchSize) {                                      \
    return cusolverDn##P##syevjBatched(h, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize); }

HERMITIAN_ALIAS(C, cuComplex)
HERMITIAN_ALIAS(Z, cuDoubleComplex)

// orgqr / ormqr (real) and ungqr / unmqr (complex)
#define Q_FUNCTIONS(P, T, gen, mult)                                                                                   \
cusolverStatus_t cusolverDn##P##gen##_bufferSize(cusolverDnHandle_t h, int m, int n, int, const T *, int, const T *,   \
                                                 int *lwork) {                                                         \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(qrWork(m, n));                                                                                  \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##gen(cusolverDnHandle_t h, int m, int n, int k, T *A, int lda, const T *tau, T *work,   \
                                    int lwork, int *info) { return orgqr(h, m, n, k, A, lda, tau, work, lwork, info); }\
cusolverStatus_t cusolverDn##P##mult##_bufferSize(cusolverDnHandle_t h, cublasSideMode_t, cublasOperation_t, int m,    \
                                                  int n, int, const T *, int, const T *, const T *, int, int *lwork) { \
    CHECK_SOLVER(h);                                                                                                   \
    if (lwork == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;                                                        \
    *lwork = clampWork(qrWork(std::max(m, n), std::max(m, n)));                                                        \
    return CUSOLVER_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
cusolverStatus_t cusolverDn##P##mult(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m,      \
                                     int n, int k, const T *A, int lda, const T *tau, T *C, int ldc, T *work,          \
                                     int lwork, int *devInfo) {                                                        \
    return ormqr(h, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo); }

Q_FUNCTIONS(S, float, orgqr, ormqr)
Q_FUNCTIONS(D, double, orgqr, ormqr)
Q_FUNCTIONS(C, cuComplex, ungqr, unmqr)
Q_FUNCTIONS(Z, cuDoubleComplex, ungqr, unmqr)

// ---------------------------------------------------------------------------------------------------------------------
// the 64 bit generic API

namespace {

// calls fn with a null pointer of the element type of a data type
template<typename F>
cusolverStatus_t dispatch(cudaDataType type, F fn) {
    switch (type) {
        case CUDA_R_32F: return fn(static_cast<float *>(nullptr));
        case CUDA_R_64F: return fn(static_cast<double *>(nullptr));
        case CUDA_C_32F: return fn(static_cast<cuComplex *>(nullptr));
        case CUDA_C_64F: return fn(static_cast<cuDoubleComplex *>(nullptr));
        default: return CUSOLVER_STATUS_NOT_SUPPORTED;
    }
}

bool fitsInt(int64_t value) {
    return value >= 0 && value <= std::numeric_limits<int>::max();
}

} // namespace

cusolverStatus_t cusolverDnXpotrf_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, cublasFillMode_t, int64_t n,
                                             cudaDataType dataTypeA, const void *, int64_t, cudaDataType,
                                             size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    CHECK_SOLVER(h);
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = sizeof(T) * potrfWork(int(n));
        if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
        return CUSOLVER_STATUS_SUCCESS;
    });
}

cusolverStatus_t cusolverDnXpotrf(cusolverDnHandle_t h, cusolverDnParams_t, cublasFillMode_t uplo, int64_t n,
                                  cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType, void *bufferOnDevice,
                                  size_t workspaceInBytesOnDevice, void *, size_t, int *info) {
    CHECK_SOLVER(h);
    if (!fitsInt(n) || !fitsInt(lda)) return CUSOLVER_STATUS_INVALID_VALUE;
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        return potrf(h, uplo, int(n), static_cast<T *>(A), int(lda), static_cast<T *>(bufferOnDevice),
                     clampWork(workspaceInBytesOnDevice / sizeof(T)), info);
    });
}

cusolverStatus_t cusolverDnXpotrs(cusolverDnHandle_t h, cusolverDnParams_t, cublasFillMode_t uplo, int64_t n,
                                  int64_t nrhs, cudaDataType dataTypeA, const void *A, int64_t lda,
                                  cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
    CHECK_SOLVER(h);
    if (dataTypeA != dataTypeB || !fitsInt(n) || !fitsInt(nrhs) || !fitsInt(lda) || !fitsInt(ldb)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        return potrs(h, uplo, int(n), int(nrhs), static_cast<const T *>(A), int(lda), static_cast<T *>(B), int(ldb),
                     info);
    });
}

cusolverStatus_t cusolverDnXgeqrf_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, int64_t m, int64_t n,
                                             cudaDataType dataTypeA, const void *, int64_t, cudaDataType, const void *,
                                             cudaDataType, size_t *workspaceInBytesOnDevice,
                                             size_t *workspaceInBytesOnHost) {
    CHECK_SOLVER(h);
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = sizeof(T) * qrWork(int(m), int(n));
        if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
        return CUSOLVER_STATUS_SUCCESS;
    });
}

cusolverStatus_t cusolverDnXgeqrf(cusolverDnHandle_t h, cusolverDnParams_t, int64_t m, int64_t n,
                                  cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeTau, void *tau,
                                  cudaDataType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *, size_t,
                                  int *info) {
    CHECK_SOLVER(h);
    if (dataTypeA != dataTypeTau || !fitsInt(m) || !fitsInt(n) || !fitsInt(lda)) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        return geqrf(h, int(m), int(n), static_cast<T *>(A), int(lda), static_cast<T *>(tau),
                     static_cast<T *>(bufferOnDevice), clampWork(workspaceInBytesOnDevice / sizeof(T)), info);
    });
}

cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t,
                                             cublasFillMode_t, int64_t n, cudaDataType dataTypeA, const void *,
                                             int64_t, cudaDataType, const void *, cudaDataType,
                                             size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    CHECK_SOLVER(h);
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = sizeof(T) * eigenWork<T>(int(n));
        if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
        return CUSOLVER_STATUS_SUCCESS;
    });
}

cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda,
                                  cudaDataType, void *W, cudaDataType, void *bufferOnDevice,
                                  size_t workspaceInBytesOnDevice, void *, size_t, int *info) {
    CHECK_SOLVER(h);
    if (!fitsInt(n) || !fitsInt(lda)) return CUSOLVER_STATUS_INVALID_VALUE;
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        return eigen(h, jobz, uplo, int(n), static_cast<T *>(A), lda, 0, static_cast<Real<T> *>(W), 0, 1,
                     static_cast<T *>(bufferOnDevice), workspaceInBytesOnDevice / sizeof(T), info, false, 0, 100, true,
                     nullptr, nullptr);
    });
}

cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t,
                                                   cublasFillMode_t, int64_t n, cudaDataType dataTypeA, const void *,
                                                   int64_t, cudaDataType, const void *, cudaDataType,
                                                   size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost,
                                                   int64_t batchSize) {
    CHECK_SOLVER(h);
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = sizeof(T) * eigenWork<T>(int(n)) * batchSize;
        if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
        return CUSOLVER_STATUS_SUCCESS;
    });
}

cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A,
                                        int64_t lda, cudaDataType, void *W, cudaDataType, void *bufferOnDevice,
                                        size_t workspaceInBytesOnDevice, void *, size_t, int *info,
                                        int64_t batchSize) {
    CHECK_SOLVER(h);
    if (!fitsInt(n) || !fitsInt(lda) || !fitsInt(batchSize)) return CUSOLVER_STATUS_INVALID_VALUE;
    return dispatch(dataTypeA, [&](auto *p) {
        using T = std::remove_pointer_t<decltype(p)>;
        return eigen(h, jobz, uplo, int(n), static_cast<T *>(A), lda, lda * n, static_cast<Real<T> *>(W), n,
                     int(batchSize), static_cast<T *>(bufferOnDevice), workspaceInBytesOnDevice / sizeof(T), info, true,
                     0, 100, true, nullptr, nullptr);
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// non-symmetric eigenvalues (host)

namespace {

using cplx = std::complex<double>;

/**
 * Eigenvalues w and (vectors) right eigenvectors V (columns, norm 1, largest component real) of the n x n matrix A
 * (column-major, overwritten). Returns 0, or the number of eigenvalues that didn't converge.
 */
int geevHost(int n, std::vector<cplx> &a, std::vector<cplx> &w, std::vector<cplx> &v, bool vectors) {
    auto A = [&](int i, int j) -> cplx & { return a[i + size_t(j) * n]; };
    std::vector<cplx> z;
    if (vectors) {
        z.assign(size_t(n) * n, 0);
        for (int i = 0; i < n; i++) z[i + size_t(i) * n] = 1;
    }
    auto Z = [&](int i, int j) -> cplx & { return z[i + size_t(j) * n]; };
    double norm = 0;
    for (const cplx &x: a) norm = std::max(norm, std::abs(x));
    // Householder reduction to upper Hessenberg form: A = Q H Q^H, Z = Q
    std::vector<cplx> u(n);
    for (int k = 0; k + 2 < n; k++) {
        double xnorm = 0;
        for (int i = k + 1; i < n; i++) xnorm += std::norm(A(i, k));
        xnorm = std::sqrt(xnorm);
        if (xnorm == 0) continue;
        const cplx alpha = A(k + 1, k);
        const cplx phase = std::abs(alpha) > 0 ? alpha / std::abs(alpha) : cplx(1);
        // u = x + phase |x| e1, H = I - 2 u u^H / u^H u
        for (int i = k + 1; i < n; i++) u[i] = A(i, k);
        u[k + 1] += phase * xnorm;
        double unorm = 0;
        for (int i = k + 1; i < n; i++) unorm += std::norm(u[i]);
        if (unorm == 0) continue;
        // A = H A H
        for (int j = 0; j < n; j++) {
            cplx dot = 0;
            for (int i = k + 1; i < n; i++) dot += std::conj(u[i]) * A(i, j);
            dot *= 2 / unorm;
            for (int i = k + 1; i < n; i++) A(i, j) -= u[i] * dot;
        }
        for (int i = 0; i < n; i++) {
            cplx dot = 0;
            for (int j = k + 1; j < n; j++) dot += A(i, j) * u[j];
            dot *= 2 / unorm;
            for (int j = k + 1; j < n; j++) A(i, j) -= dot * std::conj(u[j]);
        }
        if (vectors) {
            for (int i = 0; i < n; i++) {
                cplx dot = 0;
                for (int j = k + 1; j < n; j++) dot += Z(i, j) * u[j];
                dot *= 2 / unorm;
                for (int j = k + 1; j < n; j++) Z(i, j) -= dot * std::conj(u[j]);
            }
        }
        for (int i = k + 2; i < n; i++) A(i, k) = 0;
    }
    // shifted QR iterations to the Schur form T = Z^H A Z (upper triangular)
    const double eps = DBL_EPSILON;
    int hi = n - 1, iterations = 0;
    while (hi > 0) {
        int l = hi;
        while (l > 0 && std::abs(A(l, l - 1)) > eps * (std::abs(A(l, l)) + std::abs(A(l - 1, l - 1)))) {
            if (std::abs(A(l, l - 1)) <= eps * norm * 1e-3) break;
            l--;
        }
        if (l > 0) A(l, l - 1) = 0;
        if (l == hi) {
            hi--;
            iterations = 0;
            continue;
        }
        if (++iterations > 60 * n) {
            return hi + 1;
        }
        // Wilkinson shift (exceptional shifts every 10 iterations)
        cplx shift;
        if (iterations % 10 == 0) {
            shift = A(hi, hi) + std::abs(A(hi, hi - 1));
        } else {
            const cplx p = A(hi - 1, hi - 1), q = A(hi - 1, hi), r = A(hi, hi - 1), d = A(hi, hi);
            const cplx half = (p + d) / 2.0, disc = std::sqrt(half * half - (p * d - q * r));
            const cplx mu1 = half + disc, mu2 = half - disc;
            shift = std::abs(mu1 - d) < std::abs(mu2 - d) ? mu1 : mu2;
        }
        // implicit single shift step on rows / columns l..hi: chase the bulge with Givens rotations
        cplx x = A(l, l) - shift, y = A(l + 1, l);
        for (int k = l; k < hi; k++) {
            const double r = std::hypot(std::abs(x), std::abs(y));
            double c = 1;
            cplx sn = 0;
            if (r > 0) {
                if (std::abs(x) == 0) {
                    c = 0;
                    sn = std::conj(y) / std::abs(y);
                } else {
                    c = std::abs(x) / r;
                    sn = (x / std::abs(x)) * std::conj(y) / r;
                }
            }
            // rows k, k+1 (all columns from the bulge on), then columns k, k+1 (rows up to k + 2)
            for (int j = std::max(l, k - 1); j < n; j++) {
                const cplx t1 = A(k, j), t2 = A(k + 1, j);
                A(k, j) = c * t1 + sn * t2;
                A(k + 1, j) = -std::conj(sn) * t1 + c * t2;
            }
            for (int i = 0; i <= std::min(k + 2, hi); i++) {
                const cplx t1 = A(i, k), t2 = A(i, k + 1);
                A(i, k) = c * t1 + std::conj(sn) * t2;
                A(i, k + 1) = -sn * t1 + c * t2;
            }
            if (vectors) {
                for (int i = 0; i < n; i++) {
                    const cplx t1 = Z(i, k), t2 = Z(i, k + 1);
                    Z(i, k) = c * t1 + std::conj(sn) * t2;
                    Z(i, k + 1) = -sn * t1 + c * t2;
                }
            }
            if (k + 1 < hi) {
                x = A(k + 1, k);
                y = A(k + 2, k);
                if (k > l) {
                    A(k + 1, k - 1) = 0;
                }
            }
        }
        if (hi - l >= 2) {
            A(hi, hi - 2) = 0;
        }
    }
    w.resize(n);
    for (int i = 0; i < n; i++) w[i] = A(i, i);
    if (!vectors) {
        return 0;
    }
    // eigenvectors of T by back substitution, transformed by Z
    v.assign(size_t(n) * n, 0);
    std::vector<cplx> y(n);
    const double small = eps * std::max(norm, 1e-300);
    for (int k = 0; k < n; k++) {
        std::fill(y.begin(), y.end(), 0);
        y[k] = 1;
        for (int j = k - 1; j >= 0; j--) {
            cplx sum = 0;
            for (int l = j + 1; l <= k; l++) sum += A(j, l) * y[l];
            cplx d = A(j, j) - w[k];
            if (std::abs(d) < small) d = small;
            y[j] = -sum / d;
        }
        double vnorm = 0;
        cplx largest = 0;
        for (int i = 0; i < n; i++) {
            cplx value = 0;
            for (int l = 0; l <= k; l++) value += Z(i, l) * y[l];
            v[i + size_t(k) * n] = value;
            vnorm += std::norm(value);
            if (std::abs(value) > std::abs(largest)) largest = value;
        }
        // norm 1, largest component real
        const cplx factor = std::conj(largest) / (std::abs(largest) * std::sqrt(vnorm));
        for (int i = 0; i < n; i++) v[i + size_t(k) * n] *= factor;
    }
    return 0;
}

template<typename T>
void readHost(const void *src, std::vector<cplx> &out, size_t count) {
    std::vector<T> h(count);
    std::memcpy(h.data(), src, count * sizeof(T));
    out.resize(count);
    for (size_t i = 0; i < count; i++) {
        if constexpr (IS_COMPLEX<T>) {
            out[i] = cplx(h[i].x, h[i].y);
        } else {
            out[i] = double(h[i]);
        }
    }
}

template<typename T>
std::vector<T> toType(const std::vector<cplx> &in) {
    std::vector<T> out(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        out[i] = make<T>(Real<T>(in[i].real()), Real<T>(in[i].imag()));
    }
    return out;
}

size_t sizeOf(cudaDataType type) {
    switch (type) {
        case CUDA_R_32F: return 4;
        case CUDA_R_64F: return 8;
        case CUDA_C_32F: return 8;
        case CUDA_C_64F: return 16;
        default: return 0;
    }
}

bool isComplexType(cudaDataType type) {
    return type == CUDA_C_32F || type == CUDA_C_64F;
}

} // namespace

cusolverStatus_t cusolverDnXgeev_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t,
                                            cusolverEigMode_t, int64_t, cudaDataType dataTypeA, const void *,
                                            int64_t, cudaDataType, const void *, cudaDataType, const void *, int64_t,
                                            cudaDataType, const void *, int64_t, cudaDataType,
                                            size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    CHECK_SOLVER(h);
    if (sizeOf(dataTypeA) == 0) return CUSOLVER_STATUS_NOT_SUPPORTED;
    // (the computation keeps its own host memory)
    if (workspaceInBytesOnDevice) *workspaceInBytesOnDevice = 0;
    if (workspaceInBytesOnHost) *workspaceInBytesOnHost = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgeev(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t jobvl,
                                 cusolverEigMode_t jobvr, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda,
                                 cudaDataType dataTypeW, void *W, cudaDataType dataTypeVL, void *, int64_t ldvl,
                                 cudaDataType dataTypeVR, void *VR, int64_t ldvr, cudaDataType, void *, size_t, void *,
                                 size_t, int *info) {
    CHECK_SOLVER(h);
    const bool vectors = jobvr == CUSOLVER_EIG_MODE_VECTOR;
    // (like cuSOLVER: right eigenvectors only)
    if (jobvl != CUSOLVER_EIG_MODE_NOVECTOR || n < 0 || !fitsInt(n) || lda < std::max<int64_t>(1, n) ||
        (vectors && ldvr < n) || ldvl < 1) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    const bool real_a = dataTypeA == CUDA_R_32F || dataTypeA == CUDA_R_64F;
    const bool double_a = dataTypeA == CUDA_R_64F || dataTypeA == CUDA_C_64F;
    if (sizeOf(dataTypeA) == 0 || (real_a ? dataTypeVR != dataTypeA : dataTypeVR != dataTypeA) ||
        (dataTypeW != dataTypeA && !(real_a && isComplexType(dataTypeW) &&
                                     (dataTypeW == CUDA_C_64F) == double_a))) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }
    (void) dataTypeVL;
    SOLVER_TRY(clearInfo(h, info));
    if (n == 0) {
        return CUSOLVER_STATUS_SUCCESS;
    }
    const size_t element = sizeOf(dataTypeA);
    // A to the host (after the stream's preceding work)
    std::vector<char> raw(static_cast<size_t>(n) * n * element);
    if (cudaMemcpy2DAsync(raw.data(), n * element, A, lda * element, n * element, n, cudaMemcpyDeviceToHost,
                          h->stream) != cudaSuccess || cudaStreamSynchronize(h->stream) != cudaSuccess) {
        return CUSOLVER_STATUS_EXECUTION_FAILED;
    }
    std::vector<cplx> a, w, v;
    switch (dataTypeA) {
        case CUDA_R_32F: readHost<float>(raw.data(), a, raw.size() / element); break;
        case CUDA_R_64F: readHost<double>(raw.data(), a, raw.size() / element); break;
        case CUDA_C_32F: readHost<cuComplex>(raw.data(), a, raw.size() / element); break;
        default: readHost<cuDoubleComplex>(raw.data(), a, raw.size() / element); break;
    }
    double norm = 0;
    for (const cplx &x: a) norm = std::max(norm, std::abs(x));
    const int failed = geevHost(int(n), a, w, v, vectors);
    std::vector<cplx> w_out = w, v_out;
    if (real_a) {
        // eigenvalues of a real matrix: real ones (rounding noise removed) and conjugate pairs, the one with positive
        // imaginary part first; eigenvectors of a pair packed as VR(:, j) +- i VR(:, j + 1)
        const double tiny = DBL_EPSILON * std::max(norm, 1e-300) * n * 64;
        std::vector<int> order;
        std::vector<bool> used(n, false);
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            if (std::abs(w[i].imag()) <= tiny) {
                used[i] = true;
                order.push_back(i);
                continue;
            }
            // the conjugate partner: the closest unused eigenvalue to conj(w[i])
            int partner = -1;
            for (int j = 0; j < n; j++) {
                if (j != i && !used[j] && (partner < 0 || std::abs(w[j] - std::conj(w[i])) <
                                                          std::abs(w[partner] - std::conj(w[i])))) {
                    partner = j;
                }
            }
            used[i] = true;
            if (partner < 0) {
                order.push_back(i);
                continue;
            }
            used[partner] = true;
            const int first = w[i].imag() > 0 ? i : partner, second = first == i ? partner : i;
            order.push_back(first);
            order.push_back(second);
        }
        w_out.assign(n, 0);
        v_out.assign(vectors ? size_t(n) * n : 0, 0);
        for (int k = 0; k < n; k++) {
            const int i = order[k];
            cplx value = w[i];
            const bool pair_first = std::abs(value.imag()) > tiny && k + 1 < n &&
                                    std::abs(w[order[k + 1]] - std::conj(value)) <= std::abs(value) * 1e-6 + tiny;
            if (std::abs(value.imag()) <= tiny) {
                value = value.real();
            }
            w_out[k] = value;
            if (pair_first) {
                w_out[k + 1] = std::conj(value);
            }
            if (vectors) {
                for (int r = 0; r < n; r++) {
                    const cplx e = v[r + size_t(i) * n];
                    if (pair_first) {
                        v_out[r + size_t(k) * n] = e.real();
                        v_out[r + size_t(k + 1) * n] = e.imag();
                    } else if (std::abs(value.imag()) <= tiny) {
                        v_out[r + size_t(k) * n] = e.real();
                    }
                }
            }
            if (pair_first) {
                k++;
            }
        }
    } else {
        v_out = v;
    }
    // results to the device (synchronous: the host buffers go away)
    auto upload = [&](void *dst, const std::vector<cplx> &values, cudaDataType type, int64_t rows, int64_t cols,
                      int64_t ld) {
        std::vector<char> bytes;
        const size_t size = sizeOf(type);
        auto convert = [&](auto tag) {
            using T = decltype(tag);
            const std::vector<T> typed = toType<T>(values);
            bytes.assign(reinterpret_cast<const char *>(typed.data()),
                         reinterpret_cast<const char *>(typed.data() + typed.size()));
        };
        switch (type) {
            case CUDA_R_32F: convert(float{}); break;
            case CUDA_R_64F: convert(double{}); break;
            case CUDA_C_32F: convert(cuComplex{}); break;
            default: convert(cuDoubleComplex{}); break;
        }
        return cudaMemcpy2DAsync(dst, ld * size, bytes.data(), rows * size, rows * size, cols, cudaMemcpyHostToDevice,
                                 h->stream) == cudaSuccess && cudaStreamSynchronize(h->stream) == cudaSuccess;
    };
    bool ok;
    if (real_a && !isComplexType(dataTypeW)) {
        // real W: the real parts, then the imaginary parts
        std::vector<cplx> parts(2 * size_t(n));
        for (int i = 0; i < n; i++) {
            parts[i] = w_out[i].real();
            parts[n + i] = w_out[i].imag();
        }
        ok = upload(W, parts, dataTypeW, 2 * n, 1, 2 * n);
    } else {
        ok = upload(W, w_out, dataTypeW, n, 1, n);
    }
    if (ok && vectors && failed == 0) {
        ok = upload(VR, v_out, dataTypeVR, n, n, ldvr);
    }
    if (ok && failed != 0 && info != nullptr) {
        ok = cudaMemcpyAsync(info, &failed, sizeof(int), cudaMemcpyHostToDevice, h->stream) == cudaSuccess &&
             cudaStreamSynchronize(h->stream) == cudaSuccess;
    }
    return ok ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}
