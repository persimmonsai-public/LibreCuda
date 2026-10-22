/**
 * The cuBLAS handle of LibreCUDA's cuBLAS (shared by cublas.cpp and blas_routines.cu)
 */
#pragma once

#include <cublas_api.h>
#include <cuda_runtime_api.h>

#include <cstdint>

struct cublasContext {
    uint32_t magic;
    int device;
    cudaStream_t stream;
    void *workspace;
    size_t workspaceSize;
    cublasPointerMode_t pointerMode;
    cublasMath_t mathMode;
    cublasAtomicsMode_t atomicsMode;
    int smCountTarget;
};

namespace librecuda_blas {

constexpr uint32_t HANDLE_MAGIC = 0x4c434248;

inline bool valid(cublasHandle_t handle) {
    return handle != nullptr && handle->magic == HANDLE_MAGIC;
}

} // namespace librecuda_blas

#define CHECK_HANDLE(handle) do { if (!librecuda_blas::valid(handle)) { return CUBLAS_STATUS_NOT_INITIALIZED; } } while (0)
