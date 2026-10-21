// Dynamic shared memory up to the opt-in maximum (> 48 KiB) via cudaFuncSetAttribute, mixed static + dynamic
// shared memory, 2D/3D block shapes, occupancy API
#include "check.h"
#include <vector>

__global__ void reverseLarge(const int *in, int *out, int n) {
    extern __shared__ int buf[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = in[blockIdx.x * n + i];
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[blockIdx.x * n + i] = buf[n - 1 - i];
}

__global__ void mixedShared(int *out) {
    __shared__ int staticBuf[1024];
    extern __shared__ int dynBuf[];
    int t = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int nt = blockDim.x * blockDim.y * blockDim.z;
    for (int i = t; i < 1024; i += nt) staticBuf[i] = i;
    for (int i = t; i < 2048; i += nt) dynBuf[i] = 2 * i;
    __syncthreads();
    int idx = blockIdx.x * nt + t;
    out[idx] = staticBuf[(t * 7) % 1024] + dynBuf[(t * 13) % 2048];
}

int main() {
    int dev = 0, optin = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    printf("max opt-in shared memory per block: %d bytes\n", optin);
    EXPECT(optin > 48 * 1024, "opt-in shared memory %d", optin);

    const int n = optin / sizeof(int);
    const int blocks = 60;
    std::vector<int> h(n * blocks);
    for (size_t i = 0; i < h.size(); i++) h[i] = (int) i * 3 + 1;
    int *din, *dout;
    CHECK(cudaMalloc(&din, h.size() * 4));
    CHECK(cudaMalloc(&dout, h.size() * 4));
    CHECK(cudaMemcpy(din, h.data(), h.size() * 4, cudaMemcpyHostToDevice));

    // without opt-in a launch with > 48 KiB of dynamic shared memory fails
    reverseLarge<<<blocks, 512, n * sizeof(int)>>>(din, dout, n);
    cudaError_t err = cudaGetLastError();
    EXPECT(err != cudaSuccess, "launch with %zu bytes of dynamic shared memory should require opt-in", n * sizeof(int));

    CHECK(cudaFuncSetAttribute(reverseLarge, cudaFuncAttributeMaxDynamicSharedMemorySize, n * sizeof(int)));
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, reverseLarge));
    EXPECT(attr.maxDynamicSharedSizeBytes == (int) (n * sizeof(int)), "max dynamic smem %d", attr.maxDynamicSharedSizeBytes);
    reverseLarge<<<blocks, 512, n * sizeof(int)>>>(din, dout, n);
    CHECK(cudaGetLastError());
    std::vector<int> out(h.size());
    CHECK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    for (int b = 0; b < blocks; b++)
        for (int i = 0; i < n; i++)
            EXPECT(out[b * n + i] == h[b * n + n - 1 - i], "block %d element %d", b, i);

    dim3 block(8, 4, 4);
    mixedShared<<<blocks, block, 2048 * sizeof(int)>>>(dout);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(out.data(), dout, blocks * 128 * 4, cudaMemcpyDeviceToHost));
    for (int b = 0; b < blocks; b++)
        for (int t = 0; t < 128; t++)
            EXPECT(out[b * 128 + t] == (t * 7) % 1024 + 2 * ((t * 13) % 2048), "mixed %d %d = %d", b, t, out[b * 128 + t]);

    int num_blocks = 0, min_grid = 0, best_block = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, mixedShared, 128, 2048 * sizeof(int)));
    CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, mixedShared, 2048 * sizeof(int)));
    printf("occupancy: %d blocks/SM of 128 threads; suggested block size %d, min grid %d\n", num_blocks, best_block,
           min_grid);
    EXPECT(num_blocks >= 1 && best_block >= 32, "occupancy");
    PASS("dynamic_shared");
    return 0;
}
