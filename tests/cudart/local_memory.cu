// Local memory: large per-thread arrays (spilled to local memory), recursion with a configured stack limit,
// dynamically indexed private arrays
#include "check.h"
#include <vector>

__device__ int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

__device__ long long sumRecursive(const int *a, int lo, int hi) {
    if (hi - lo <= 4) {
        long long s = 0;
        for (int i = lo; i < hi; i++) s += a[i];
        return s;
    }
    int mid = (lo + hi) / 2;
    return sumRecursive(a, lo, mid) + sumRecursive(a, mid, hi);
}

__global__ void privateArrays(int *out, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int buf[512]; // forces local memory
    for (int i = 0; i < 512; i++) buf[i] = (i * 31 + tid + seed) & 1023;
    // dynamic indexing
    int acc = 0;
    for (int i = 0; i < 512; i++) acc += buf[(i * 17 + tid) & 511] * (i & 7);
    out[tid] = acc;
}

__global__ void recursion(int *fibOut, long long *sumOut, const int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fibOut[tid] = fib(10 + tid % 12);
    sumOut[tid] = sumRecursive(data, 0, n - (tid % 100));
}

int main() {
    CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    size_t limit = 0;
    CHECK(cudaDeviceGetLimit(&limit, cudaLimitStackSize));
    EXPECT(limit >= 4096, "stack limit %zu", limit);

    const int threads = 256, blocks = 120;
    const int total = threads * blocks;
    int *d_out;
    CHECK(cudaMalloc(&d_out, total * sizeof(int)));
    privateArrays<<<blocks, threads>>>(d_out, 7);
    CHECK(cudaGetLastError());
    std::vector<int> out(total);
    CHECK(cudaMemcpy(out.data(), d_out, total * sizeof(int), cudaMemcpyDeviceToHost));
    for (int tid = 0; tid < total; tid++) {
        int buf[512];
        for (int i = 0; i < 512; i++) buf[i] = (i * 31 + tid + 7) & 1023;
        int acc = 0;
        for (int i = 0; i < 512; i++) acc += buf[(i * 17 + tid) & 511] * (i & 7);
        EXPECT(out[tid] == acc, "privateArrays[%d] = %d, expected %d", tid, out[tid], acc);
    }

    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, privateArrays));
    printf("privateArrays: %zu bytes local memory, %d registers\n", attr.localSizeBytes, attr.numRegs);
    EXPECT(attr.localSizeBytes >= 2048, "local size %zu", attr.localSizeBytes);

    const int n = 1000;
    std::vector<int> data(n);
    for (int i = 0; i < n; i++) data[i] = i * 3 - 7;
    int *d_data, *d_fib; long long *d_sum;
    CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CHECK(cudaMalloc(&d_fib, total * sizeof(int)));
    CHECK(cudaMalloc(&d_sum, total * sizeof(long long)));
    CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    recursion<<<blocks, threads>>>(d_fib, d_sum, d_data, n);
    CHECK(cudaGetLastError());
    std::vector<int> fibs(total);
    std::vector<long long> sums(total);
    CHECK(cudaMemcpy(fibs.data(), d_fib, total * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sums.data(), d_sum, total * sizeof(long long), cudaMemcpyDeviceToHost));
    auto hfib = [](int k) { int a = 0, b = 1; for (int i = 0; i < k; i++) { int t = a + b; a = b; b = t; } return a; };
    for (int tid = 0; tid < total; tid++) {
        EXPECT(fibs[tid] == hfib(10 + tid % 12), "fib[%d] = %d", tid, fibs[tid]);
        long long s = 0;
        for (int i = 0; i < n - tid % 100; i++) s += data[i];
        EXPECT(sums[tid] == s, "sum[%d] = %lld, expected %lld", tid, sums[tid], s);
    }
    PASS("local_memory");
    return 0;
}
