// Basic runtime flow: cudaMalloc, cudaMemcpy (pageable), <<<>>> launches with a grid > 65535 blocks, error queries
#include "check.h"
#include <vector>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void saxpy(float alpha, const float *x, float *y, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = alpha * x[i] + y[i];
}

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    EXPECT(count > 0, "no devices");
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, cc %d.%d, %d SMs, %.1f GiB\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1073741824.0);

    const int n = 50'000'017; // not a multiple of the block size, > 65535 blocks
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = 1.0f - i; }
    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, n * sizeof(float)));
    CHECK(cudaMalloc(&db, n * sizeof(float)));
    CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    vectorAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CHECK(cudaGetLastError());
    saxpy<<<prop.multiProcessorCount * 8, 1024>>>(2.0f, da, dc, (size_t) n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        float expect = 2.0f * a[i] + (a[i] + b[i]);
        EXPECT(c[i] == expect, "c[%d] = %f, expected %f", i, c[i], expect);
    }

    // invalid launch configurations are reported
    vectorAdd<<<1, 2048>>>(da, db, dc, n);
    EXPECT(cudaGetLastError() != cudaSuccess, "launch with 2048 threads per block should fail");
    EXPECT(cudaGetLastError() == cudaSuccess, "error should be cleared");

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
    PASS("vector_add");
    return 0;
}
