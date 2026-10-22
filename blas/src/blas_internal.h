/**
 * LibreCUDA's cuBLAS / cuBLASLt: the GEMM engine (in libcublasLt.so.13, used by both libraries) and the helpers the two
 * share. Only the public cublas_api.h / cublasLt.h headers are used for types; the kernels are CUTLASS's and our own.
 */
#pragma once

#include <cublasLt.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <library_types.h>

#include <cstddef>
#include <cstdint>

// exported from libcublasLt.so.13 for libcublas.so.13 (symbol version LIBRECUDA_BLAS_PRIVATE)
#define LIBRECUDA_BLAS_EXPORT extern "C" __attribute__((visibility("default")))

namespace librecuda_blas {

// a scalar (alpha / beta) as passed to cuBLAS: on the host or on the device, of the operation's scale type
struct Scalar {
    const void *ptr;
    cudaDataType type;
    bool onDevice;
};

// a matrix operand: column-major, ld elements between columns, stride elements between batch entries, or an array of
// per-batch pointers (on the device). inc: elements between rows (1 for matrices; vectors of gemv may have others,
// which only the generic kernel handles)
struct Operand {
    const void *ptr;
    const void *const *array;
    cudaDataType type;
    long long ld;
    long long stride;
    long long inc;
};

enum class Activation : int {
    NONE, RELU, GELU
};

/**
 * D = alpha * op(A) op(B) + beta * C (then + bias, activation): column-major, m x n x k, batch entries
 */
struct GemmParams {
    cublasOperation_t transa, transb;
    int m, n, k;
    int batch;
    Operand a, b, c, d;
    cublasComputeType_t compute;
    Scalar alpha, beta;
    // TF32 tensor cores may be used for float operands
    bool allowTF32;
    // cuBLASLt epilogue: bias (m elements per batch entry, added to every column) and activation
    const void *bias;
    cudaDataType biasType;
    long long biasStride;
    Activation activation;
    cudaStream_t stream;
};

} // namespace librecuda_blas

// the GEMM engine
LIBRECUDA_BLAS_EXPORT cublasStatus_t librecuda_blas_gemm(const librecuda_blas::GemmParams *params);

// result (host or device memory, of the given type) = sum over i of conj?(x[i]) * y[i]
LIBRECUDA_BLAS_EXPORT cublasStatus_t librecuda_blas_dot(int n, const void *x, cudaDataType xType, long long incx,
                                                        const void *y, cudaDataType yType, long long incy,
                                                        void *result, cudaDataType resultType, bool resultOnDevice,
                                                        bool conjugate, cudaStream_t stream);
