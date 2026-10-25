// Managed memory, 64 MiB (cudaMallocManaged): CPU writes (first touch, again), a CPU read, the first kernel after the CPU
// wrote (the pages migrate to the GPU), the kernel again, and a CPU read after the GPU wrote (the pages migrate back).
// Simple CPU loops over the buffer, a kernel adding 1 to every element. Plain CUDA runtime code: build with nvcc and run
// on any runtime (LibreCUDA or NVIDIA's).
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s (%s:%d)\n", #call, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double seconds(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

__global__ void addOne(int *p, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) p[i] += 1;
}

int main() {
    const size_t bytes = size_t(64) << 20, n = bytes / sizeof(int);
    int *p;
    CHECK(cudaMallocManaged(&p, bytes));
    auto t0 = clk::now();
    for (size_t i = 0; i < n; i++) p[i] = int(i);
    auto t1 = clk::now();
    for (size_t i = 0; i < n; i++) p[i] = int(i);
    auto t2 = clk::now();
    volatile long long sum = 0;
    for (size_t i = 0; i < n; i++) sum += p[i];
    auto t3 = clk::now();
    printf("CPU write, first touch / again:   %.2f / %.2f GB/s\n", bytes / seconds(t0, t1) / 1e9, bytes / seconds(t1, t2) / 1e9);
    printf("CPU read:                          %.2f GB/s\n", bytes / seconds(t2, t3) / 1e9);
    t0 = clk::now();
    addOne<<<240, 256>>>(p, n);
    CHECK(cudaDeviceSynchronize());
    t1 = clk::now();
    addOne<<<240, 256>>>(p, n);
    CHECK(cudaDeviceSynchronize());
    t2 = clk::now();
    printf("first kernel after CPU writes:     %.2f ms\n", seconds(t0, t1) * 1e3);
    printf("kernel again:                      %.2f ms\n", seconds(t1, t2) * 1e3);
    t0 = clk::now();
    long long check = 0;
    for (size_t i = 0; i < n; i++) check += p[i] - int(i);
    t1 = clk::now();
    printf("CPU read after the GPU wrote:      %.2f GB/s (%s)\n", bytes / seconds(t0, t1) / 1e9,
           check == 2 * (long long) n ? "correct" : "WRONG");
    CHECK(cudaFree(p));
    return 0;
}
