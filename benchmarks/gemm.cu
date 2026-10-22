// GEMM throughput through cuBLAS (cublasGemmEx): LibreCUDA's cuBLAS with LD_LIBRARY_PATH=build/lib, NVIDIA's without.
//   nvcc -O3 -arch=sm_120 -cudart=shared benchmarks/gemm.cu -lcublas -o gemm
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static void bench(cublasHandle_t handle, const char *name, cudaDataType type, cudaDataType out,
                  cublasComputeType_t compute, int m, int n, int k, int batch) {
    const size_t elem = type == CUDA_R_64F ? 8 : type == CUDA_R_32F ? 4 : 2;
    const size_t out_elem = out == CUDA_R_64F ? 8 : out == CUDA_R_32F ? 4 : 2;
    void *a, *b, *c;
    cudaMalloc(&a, elem * m * k * batch);
    cudaMalloc(&b, elem * k * n * batch);
    cudaMalloc(&c, out_elem * m * n * batch);
    cudaMemset(a, 0, elem * m * k * batch);
    cudaMemset(b, 0, elem * k * n * batch);
    double alpha_d = 1, beta_d = 0;
    float alpha_f = 1, beta_f = 0;
    const void *alpha = compute == CUBLAS_COMPUTE_64F ? (const void *) &alpha_d : &alpha_f;
    const void *beta = compute == CUBLAS_COMPUTE_64F ? (const void *) &beta_d : &beta_f;
    auto run = [&] {
        return cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, a, type, m,
                                          (long long) m * k, b, type, n, (long long) k * n, beta, c, out, m,
                                          (long long) m * n, batch, compute, CUBLAS_GEMM_DEFAULT);
    };
    if (run() != CUBLAS_STATUS_SUCCESS) {
        printf("%-34s failed\n", name);
        return;
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) run();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        best = ms / 10 < best ? ms / 10 : best;
    }
    printf("%-34s %8.3f ms  %7.2f TFLOPS\n", name, best, 2.0 * m * n * k * batch / (best * 1e9));
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    bench(handle, "half 4096^3", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 4096, 4096, 4096, 1);
    bench(handle, "bfloat16 4096^3", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 4096, 4096, 4096, 1);
    bench(handle, "half -> float 4096^3", CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 4096, 4096, 4096, 1);
    bench(handle, "float 4096^3", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 4096, 4096, 4096, 1);
    bench(handle, "float (TF32) 4096^3", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4096, 4096, 4096, 1);
    bench(handle, "double 1024^3", CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F, 1024, 1024, 1024, 1);
    bench(handle, "half 512^3", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 512, 512, 512, 1);
    bench(handle, "half batched 64 x 128x128x64", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 128, 128, 64, 64);
    bench(handle, "float 1000x1000x999 (unaligned)", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 1000, 1000, 999, 1);
    bench(handle, "half 1001x1001x1001 (unaligned)", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 1001, 1001, 1001, 1);
    bench(handle, "half 1002x1002x1002 (4 byte aligned)", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 1002, 1002, 1002, 1);
    bench(handle, "half 50257x512x768 (LM head)", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 50257, 512, 768, 1);
    cublasDestroy(handle);
    return 0;
}
