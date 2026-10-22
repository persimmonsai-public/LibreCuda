/**
 * LibreCUDA's cuBLAS (libcublas.so.13): handles, the GEMM family (typed, Ex, batched and strided batched), GEMV and dot
 * products on the GEMM engine of libcublasLt.so.13, and the vector / matrix copy helpers. Functions not implemented here
 * return CUBLAS_STATUS_NOT_SUPPORTED (stubs.cpp).
 */
#include "blas_internal.h"
#include "cublas_handle.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cuComplex.h>
#include <cuda_fp16.h>

using namespace librecuda_blas;

namespace {

bool validOp(cublasOperation_t op) {
    return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

bool isComplex(cudaDataType type) {
    return type == CUDA_C_32F || type == CUDA_C_64F || type == CUDA_C_16F || type == CUDA_C_16BF;
}

// the type of alpha and beta for a compute type
cudaDataType scaleTypeOf(cublasComputeType_t compute, cudaDataType outputType) {
    switch (compute) {
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            return CUDA_R_16F;
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            return isComplex(outputType) ? CUDA_C_64F : CUDA_R_64F;
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            return CUDA_R_32I;
        default:
            return isComplex(outputType) ? CUDA_C_32F : CUDA_R_32F;
    }
}

bool tf32Allowed(cublasHandle_t handle, cublasComputeType_t compute) {
    if (compute == CUBLAS_COMPUTE_32F_FAST_TF32 || compute == CUBLAS_COMPUTE_32F_FAST_16F ||
        compute == CUBLAS_COMPUTE_32F_FAST_16BF) {
        return true;
    }
    const int mode = handle->mathMode & 0xf;
    return compute == CUBLAS_COMPUTE_32F && mode == CUBLAS_TF32_TENSOR_OP_MATH;
}

/**
 * C = alpha op(A) op(B) + beta C: strided (arrays null) or with arrays of per-batch pointers
 */
cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                    const void *alpha, const void *A, const void *const *Aarray, cudaDataType Atype, int lda,
                    long long strideA, const void *B, const void *const *Barray, cudaDataType Btype, int ldb,
                    long long strideB, const void *beta, void *C, void *const *Carray, cudaDataType Ctype, int ldc,
                    long long strideC, int batchCount, cublasComputeType_t compute, cudaDataType scaleType) {
    CHECK_HANDLE(handle);
    if (!validOp(transa) || !validOp(transb) || m < 0 || n < 0 || k < 0 || batchCount < 0 ||
        lda < std::max(1, transa == CUBLAS_OP_N ? m : k) || ldb < std::max(1, transb == CUBLAS_OP_N ? k : n) ||
        ldc < std::max(1, m)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool device_scalars = handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE;
    GemmParams p{};
    p.transa = transa;
    p.transb = transb;
    p.m = m;
    p.n = n;
    p.k = k;
    p.batch = batchCount;
    p.a = Operand{A, Aarray, Atype, lda, strideA, 1};
    p.b = Operand{B, Barray, Btype, ldb, strideB, 1};
    p.c = Operand{C, Carray, Ctype, ldc, strideC, 1};
    p.d = p.c;
    p.compute = compute;
    p.alpha = Scalar{alpha, scaleType, device_scalars};
    p.beta = Scalar{beta, scaleType, device_scalars};
    p.allowTF32 = tf32Allowed(handle, compute);
    p.activation = Activation::NONE;
    p.stream = handle->stream;
    return librecuda_blas_gemm(&p);
}

cublasStatus_t gemmStrided(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                           int k, const void *alpha, const void *A, cudaDataType Atype, int lda, long long strideA,
                           const void *B, cudaDataType Btype, int ldb, long long strideB, const void *beta, void *C,
                           cudaDataType Ctype, int ldc, long long strideC, int batchCount,
                           cublasComputeType_t compute) {
    return gemm(handle, transa, transb, m, n, k, alpha, A, nullptr, Atype, lda, strideA, B, nullptr, Btype, ldb,
                strideB, beta, C, nullptr, Ctype, ldc, strideC, batchCount, compute, scaleTypeOf(compute, Ctype));
}

cublasStatus_t gemmArray(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                         int k, const void *alpha, const void *const Aarray[], cudaDataType Atype, int lda,
                         const void *const Barray[], cudaDataType Btype, int ldb, const void *beta,
                         void *const Carray[], cudaDataType Ctype, int ldc, int batchCount,
                         cublasComputeType_t compute) {
    if (batchCount > 0 && (Aarray == nullptr || Barray == nullptr || Carray == nullptr)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return gemm(handle, transa, transb, m, n, k, alpha, nullptr, Aarray, Atype, lda, 0, nullptr, Barray, Btype, ldb, 0,
                beta, nullptr, Carray, Ctype, ldc, 0, batchCount, compute, scaleTypeOf(compute, Ctype));
}

// y = alpha op(A) x + beta y, as a GEMM with one column (A is m x n)
cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const void *alpha, const void *A,
                    int lda, const void *x, int incx, const void *beta, void *y, int incy, cudaDataType type,
                    cublasComputeType_t compute) {
    CHECK_HANDLE(handle);
    if (!validOp(trans) || m < 0 || n < 0 || lda < std::max(1, m) || incx == 0 || incy == 0 || alpha == nullptr ||
        beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const int rows = trans == CUBLAS_OP_N ? m : n, inner = trans == CUBLAS_OP_N ? n : m;
    if (rows == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    size_t element = type == CUDA_R_32F ? 4 : type == CUDA_R_64F || type == CUDA_C_32F ? 8 : 16;
    // negative increments run backwards from the last element
    const auto *xs = static_cast<const char *>(x) + (incx < 0 ? (1LL - inner) * incx * (long long) element : 0);
    auto *ys = static_cast<char *>(y) + (incy < 0 ? (1LL - rows) * incy * (long long) element : 0);
    const bool device_scalars = handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE;
    GemmParams p{};
    p.transa = trans;
    p.transb = CUBLAS_OP_N;
    p.m = rows;
    p.n = 1;
    p.k = inner;
    p.batch = 1;
    p.a = Operand{A, nullptr, type, lda, 0, 1};
    // contiguous vectors are matrices with one column (which the fast kernels take)
    p.b = Operand{xs, nullptr, type, incx == 1 ? std::max(1, inner) : 1, 0, incx};
    p.c = Operand{ys, nullptr, type, incy == 1 ? rows : 1, 0, incy};
    p.d = p.c;
    p.compute = compute;
    const cudaDataType scale_type = scaleTypeOf(compute, type);
    p.alpha = Scalar{alpha, scale_type, device_scalars};
    p.beta = Scalar{beta, scale_type, device_scalars};
    p.allowTF32 = tf32Allowed(handle, compute);
    p.stream = handle->stream;
    return librecuda_blas_gemm(&p);
}

cublasStatus_t dot(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result,
                   cudaDataType type, bool conjugate) {
    CHECK_HANDLE(handle);
    if (result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return librecuda_blas_dot(n, x, type, incx, y, type, incy, result, type,
                              handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE, conjugate, handle->stream);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// handles and modes

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        cudaGetLastError();
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    *handle = new cublasContext{HANDLE_MAGIC, device, nullptr, nullptr, 0, CUBLAS_POINTER_MODE_HOST,
                                CUBLAS_DEFAULT_MATH, CUBLAS_ATOMICS_NOT_ALLOWED, 0};
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    CHECK_HANDLE(handle);
    handle->magic = 0;
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    CHECK_HANDLE(handle);
    if (version == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *version = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
    return cublasLtGetProperty(type, value);
}

size_t cublasGetCudartVersion(void) {
    return CUDART_VERSION;
}

const char *cublasGetStatusName(cublasStatus_t status) {
    return cublasLtGetStatusName(status);
}

const char *cublasGetStatusString(cublasStatus_t status) {
    return cublasLtGetStatusString(status);
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    CHECK_HANDLE(handle);
    handle->stream = streamId;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId) {
    CHECK_HANDLE(handle);
    if (streamId == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *streamId = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    CHECK_HANDLE(handle);
    // (the kernels need no workspace)
    handle->workspace = workspace;
    handle->workspaceSize = workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
    CHECK_HANDLE(handle);
    if (mode == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = handle->pointerMode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
    CHECK_HANDLE(handle);
    if (mode != CUBLAS_POINTER_MODE_HOST && mode != CUBLAS_POINTER_MODE_DEVICE) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    handle->pointerMode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    CHECK_HANDLE(handle);
    if (mode == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = handle->mathMode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    CHECK_HANDLE(handle);
    handle->mathMode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
    CHECK_HANDLE(handle);
    if (mode == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = handle->atomicsMode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    CHECK_HANDLE(handle);
    handle->atomicsMode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget) {
    CHECK_HANDLE(handle);
    if (smCountTarget == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *smCountTarget = handle->smCountTarget;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) {
    CHECK_HANDLE(handle);
    handle->smCountTarget = smCountTarget;
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// copies between host and device

namespace {

cublasStatus_t copy2D(void *dst, long long dpitch, const void *src, long long spitch, long long width, long long height,
                      cudaStream_t stream, bool async) {
    if (width == 0 || height == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const cudaError_t error = async ? cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault,
                                                        stream)
                                    : cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault);
    if (error != cudaSuccess) {
        cudaGetLastError();
        return CUBLAS_STATUS_MAPPING_ERROR;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t copyVector(int n, int elemSize, const void *x, int incx, void *y, int incy, cudaStream_t stream,
                          bool async) {
    if (n < 0 || elemSize <= 0 || incx <= 0 || incy <= 0 || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return copy2D(y, (long long) incy * elemSize, x, (long long) incx * elemSize, elemSize, n, stream, async);
}

cublasStatus_t copyMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb,
                          cudaStream_t stream, bool async) {
    if (rows < 0 || cols < 0 || elemSize <= 0 || lda < std::max(1, rows) || ldb < std::max(1, rows) || A == nullptr ||
        B == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return copy2D(B, (long long) ldb * elemSize, A, (long long) lda * elemSize, (long long) rows * elemSize, cols,
                  stream, async);
}

} // namespace

cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy) {
    return copyVector(n, elemSize, x, incx, devicePtr, incy, nullptr, false);
}

cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
    return copyVector(n, elemSize, x, incx, y, incy, nullptr, false);
}

cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy,
                                    cudaStream_t stream) {
    return copyVector(n, elemSize, hostPtr, incx, devicePtr, incy, stream, true);
}

cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy,
                                    cudaStream_t stream) {
    return copyVector(n, elemSize, devicePtr, incx, hostPtr, incy, stream, true);
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
    return copyMatrix(rows, cols, elemSize, A, lda, B, ldb, nullptr, false);
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
    return copyMatrix(rows, cols, elemSize, A, lda, B, ldb, nullptr, false);
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream) {
    return copyMatrix(rows, cols, elemSize, A, lda, B, ldb, stream, true);
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream) {
    return copyMatrix(rows, cols, elemSize, A, lda, B, ldb, stream, true);
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM

#define TYPED_GEMM(name, T, type, compute)                                                                             \
cublasStatus_t name(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,    \
                    const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc) {          \
    return gemmStrided(handle, transa, transb, m, n, k, alpha, A, type, lda, 0, B, type, ldb, 0, beta, C, type, ldc,   \
                       0, 1, compute);                                                                                 \
}

TYPED_GEMM(cublasSgemm_v2, float, CUDA_R_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMM(cublasDgemm_v2, double, CUDA_R_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMM(cublasCgemm_v2, cuComplex, CUDA_C_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMM(cublasZgemm_v2, cuDoubleComplex, CUDA_C_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMM(cublasHgemm, __half, CUDA_R_16F, CUBLAS_COMPUTE_16F)

#define TYPED_GEMM_STRIDED(name, T, type, compute)                                                                     \
cublasStatus_t name(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,    \
                    const T *alpha, const T *A, int lda, long long strideA, const T *B, int ldb, long long strideB,   \
                    const T *beta, T *C, int ldc, long long strideC, int batchCount) {                                 \
    return gemmStrided(handle, transa, transb, m, n, k, alpha, A, type, lda, strideA, B, type, ldb, strideB, beta, C,  \
                       type, ldc, strideC, batchCount, compute);                                                       \
}

TYPED_GEMM_STRIDED(cublasSgemmStridedBatched, float, CUDA_R_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMM_STRIDED(cublasDgemmStridedBatched, double, CUDA_R_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMM_STRIDED(cublasCgemmStridedBatched, cuComplex, CUDA_C_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMM_STRIDED(cublasZgemmStridedBatched, cuDoubleComplex, CUDA_C_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMM_STRIDED(cublasHgemmStridedBatched, __half, CUDA_R_16F, CUBLAS_COMPUTE_16F)

#define TYPED_GEMM_BATCHED(name, T, type, compute)                                                                     \
cublasStatus_t name(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,    \
                    const T *alpha, const T *const Aarray[], int lda, const T *const Barray[], int ldb, const T *beta, \
                    T *const Carray[], int ldc, int batchCount) {                                                      \
    return gemmArray(handle, transa, transb, m, n, k, alpha, reinterpret_cast<const void *const *>(Aarray), type, lda, \
                     reinterpret_cast<const void *const *>(Barray), type, ldb, beta,                                   \
                     reinterpret_cast<void *const *>(Carray), type, ldc, batchCount, compute);                         \
}

TYPED_GEMM_BATCHED(cublasSgemmBatched, float, CUDA_R_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMM_BATCHED(cublasDgemmBatched, double, CUDA_R_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMM_BATCHED(cublasCgemmBatched, cuComplex, CUDA_C_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMM_BATCHED(cublasZgemmBatched, cuDoubleComplex, CUDA_C_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMM_BATCHED(cublasHgemmBatched, __half, CUDA_R_16F, CUBLAS_COMPUTE_16F)

cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                             int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B,
                             cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc) {
    return gemm(handle, transa, transb, m, n, k, alpha, A, nullptr, Atype, lda, 0, B, nullptr, Btype, ldb, 0, beta, C,
                nullptr, Ctype, ldc, 0, 1, CUBLAS_COMPUTE_32F, CUDA_R_32F);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B,
                            cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t) {
    return gemmStrided(handle, transa, transb, m, n, k, alpha, A, Atype, lda, 0, B, Btype, ldb, 0, beta, C, Ctype, ldc,
                       0, 1, computeType);
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
                                          int lda, long long strideA, const void *B, cudaDataType Btype, int ldb,
                                          long long strideB, const void *beta, void *C, cudaDataType Ctype, int ldc,
                                          long long strideC, int batchCount, cublasComputeType_t computeType,
                                          cublasGemmAlgo_t) {
    return gemmStrided(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C,
                       Ctype, ldc, strideC, batchCount, computeType);
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                   int n, int k, const void *alpha, const void *const Aarray[], cudaDataType Atype,
                                   int lda, const void *const Barray[], cudaDataType Btype, int ldb, const void *beta,
                                   void *const Carray[], cudaDataType Ctype, int ldc, int batchCount,
                                   cublasComputeType_t computeType, cublasGemmAlgo_t) {
    return gemmArray(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray,
                     Ctype, ldc, batchCount, computeType);
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMV and dot products

#define TYPED_GEMV(name, T, type, compute)                                                                             \
cublasStatus_t name(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const T *alpha, const T *A, int lda, \
                    const T *x, int incx, const T *beta, T *y, int incy) {                                             \
    return gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, type, compute);                            \
}

TYPED_GEMV(cublasSgemv_v2, float, CUDA_R_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMV(cublasDgemv_v2, double, CUDA_R_64F, CUBLAS_COMPUTE_64F)
TYPED_GEMV(cublasCgemv_v2, cuComplex, CUDA_C_32F, CUBLAS_COMPUTE_32F)
TYPED_GEMV(cublasZgemv_v2, cuDoubleComplex, CUDA_C_64F, CUBLAS_COMPUTE_64F)

#define TYPED_DOT(name, T, type, conjugate)                                                                            \
cublasStatus_t name(cublasHandle_t handle, int n, const T *x, int incx, const T *y, int incy, T *result) {             \
    return dot(handle, n, x, incx, y, incy, result, type, conjugate);                                                  \
}

TYPED_DOT(cublasSdot_v2, float, CUDA_R_32F, false)
TYPED_DOT(cublasDdot_v2, double, CUDA_R_64F, false)
TYPED_DOT(cublasCdotu_v2, cuComplex, CUDA_C_32F, false)
TYPED_DOT(cublasCdotc_v2, cuComplex, CUDA_C_32F, true)
TYPED_DOT(cublasZdotu_v2, cuDoubleComplex, CUDA_C_64F, false)
TYPED_DOT(cublasZdotc_v2, cuDoubleComplex, CUDA_C_64F, true)

cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y,
                           cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType) {
    CHECK_HANDLE(handle);
    if (result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return librecuda_blas_dot(n, x, xType, incx, y, yType, incy, result, resultType,
                              handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE, false, handle->stream);
}

cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y,
                            cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType) {
    CHECK_HANDLE(handle);
    if (result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    return librecuda_blas_dot(n, x, xType, incx, y, yType, incy, result, resultType,
                              handle->pointerMode == CUBLAS_POINTER_MODE_DEVICE, true, handle->stream);
}
