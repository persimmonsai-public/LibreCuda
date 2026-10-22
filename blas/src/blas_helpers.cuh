/**
 * Helpers of the BLAS / LAPACK routines built on the GEMM engine (blas_routines.cu, cusolver.cu)
 */
#pragma once

#include "blas_internal.h"
#include "scalar.cuh"

#include <algorithm>

namespace librecuda_blas {

template<typename T>
constexpr cudaDataType dataType() {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
        return CUDA_R_64F;
    } else if constexpr (std::is_same_v<T, cuComplex>) {
        return CUDA_C_32F;
    } else {
        return CUDA_C_64F;
    }
}

template<typename T>
constexpr cublasComputeType_t computeType() {
    return std::is_same_v<Real<T>, double> ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
}

/**
 * C = alpha op(A) op(B) + beta C with host scalars (exact float arithmetic, no TF32)
 */
template<typename T>
cublasStatus_t gemm(cudaStream_t stream, cublasOperation_t ta, cublasOperation_t tb, long long m, long long n,
                    long long k, T alpha, const T *a, long long lda, const T *b, long long ldb, T beta, T *c,
                    long long ldc) {
    if (m <= 0 || n <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    GemmParams p{};
    p.transa = ta;
    p.transb = tb;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = 1;
    p.a = Operand{a, nullptr, dataType<T>(), lda, 0, 1};
    p.b = Operand{b, nullptr, dataType<T>(), ldb, 0, 1};
    p.c = Operand{c, nullptr, dataType<T>(), ldc, 0, 1};
    p.d = p.c;
    p.compute = computeType<T>();
    p.alpha = Scalar{&alpha, dataType<T>(), false};
    p.beta = Scalar{&beta, dataType<T>(), false};
    p.allowTF32 = false;
    p.activation = Activation::NONE;
    p.stream = stream;
    return librecuda_blas_gemm(&p);
}

/**
 * C = alpha op(A) op(B) + beta C with scalars as given (host or device pointers of type T)
 */
template<typename T>
cublasStatus_t gemm(cudaStream_t stream, cublasOperation_t ta, cublasOperation_t tb, long long m, long long n,
                    long long k, Scalar alpha, const T *a, long long lda, const T *b, long long ldb, Scalar beta, T *c,
                    long long ldc) {
    if (m <= 0 || n <= 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    GemmParams p{};
    p.transa = ta;
    p.transb = tb;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = 1;
    p.a = Operand{a, nullptr, dataType<T>(), lda, 0, 1};
    p.b = Operand{b, nullptr, dataType<T>(), ldb, 0, 1};
    p.c = Operand{c, nullptr, dataType<T>(), ldc, 0, 1};
    p.d = p.c;
    p.compute = computeType<T>();
    p.alpha = alpha;
    p.beta = beta;
    p.allowTF32 = false;
    p.activation = Activation::NONE;
    p.stream = stream;
    return librecuda_blas_gemm(&p);
}

inline cublasStatus_t launched() {
    return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}

inline unsigned int blocksFor(long long count, int threads = 256) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>((count + threads - 1) / threads,
                                                                                 65535)));
}

} // namespace librecuda_blas
