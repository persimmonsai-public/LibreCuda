// Microbenchmarks for the CUDA runtime: launch overhead, transfers, device memory bandwidth, allocation, SGEMM.
// Plain CUDA runtime code: build with nvcc and run on any runtime (LibreCUDA or NVIDIA's).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s (%s:%d)\n", #call, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double seconds(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

__global__ void emptyKernel() {}
__global__ void emptyKernelArgs(const float *a, float *b, int n, float x, double y) {}

__global__ void triad(float *__restrict__ a, const float *__restrict__ b, const float *__restrict__ c, float s, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        a[i] = b[i] + s * c[i];
}

__global__ void copyKernel(float4 *__restrict__ dst, const float4 *__restrict__ src, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// 128x128 block tile, 8x8 per thread, 256 threads
__global__ void __launch_bounds__(256) sgemm(const float *__restrict__ A, const float *__restrict__ B,
                                             float *__restrict__ C, int M, int N, int K) {
    __shared__ float As[8][128];
    __shared__ float Bs[8][128];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const int row0 = blockIdx.y * 128, col0 = blockIdx.x * 128;
    float acc[8][8] = {};
    for (int k0 = 0; k0 < K; k0 += 8) {
        for (int i = threadIdx.x; i < 8 * 128; i += 256) {
            int r = i / 8, k = i % 8;
            As[k][r] = A[(size_t) (row0 + r) * K + k0 + k];
            int kk = i / 128, c = i % 128;
            Bs[kk][c] = B[(size_t) (k0 + kk) * N + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float a[8], b[8];
#pragma unroll
            for (int i = 0; i < 8; i++) { a[i] = As[k][ty * 8 + i]; b[i] = Bs[k][tx * 8 + i]; }
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) C[(size_t) (row0 + ty * 8 + i) * N + col0 + tx * 8 + j] = acc[i][j];
}

template<typename F>
static double bestOf(int reps, F f) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = clk::now();
        f();
        best = std::min(best, seconds(t0, clk::now()));
    }
    return best;
}

int main(int argc, char **argv) {
    // "bench launch" runs only the launch overhead section
    const bool launch_only = argc > 1 && strcmp(argv[1], "launch") == 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);
    CHECK(cudaFree(nullptr));

    // warm up clocks
    {
        float *w;
        CHECK(cudaMalloc(&w, 64 << 20));
        for (int i = 0; i < 200; i++) triad<<<prop.multiProcessorCount * 8, 256>>>(w, w, w, 1.0f, (64 << 20) / 4);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(w));
    }

    printf("== launch overhead\n");
    {
        const int n = 20000;
        emptyKernel<<<1, 1>>>();
        CHECK(cudaDeviceSynchronize());
        double t = bestOf(3, [&] { for (int i = 0; i < n; i++) emptyKernel<<<1, 1>>>(); CHECK(cudaDeviceSynchronize()); });
        printf("  async launches (empty kernel):          %7.2f us/launch\n", t / n * 1e6);
        float *p = nullptr;
        t = bestOf(3, [&] { for (int i = 0; i < n; i++) emptyKernelArgs<<<1, 1>>>(p, p, i, 1.0f, 2.0); CHECK(cudaDeviceSynchronize()); });
        printf("  async launches (5 args):                %7.2f us/launch\n", t / n * 1e6);
        const int m = 2000;
        t = bestOf(3, [&] { for (int i = 0; i < m; i++) { emptyKernel<<<1, 1>>>(); CHECK(cudaDeviceSynchronize()); } });
        printf("  launch + synchronize round trip:        %7.2f us\n", t / m * 1e6);
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < n; i++) emptyKernel<<<1, 1>>>();
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("  GPU time per empty kernel (events):     %7.2f us\n", ms * 1e3 / n);
    }
    if (launch_only) {
        return 0;
    }

    printf("\n== host <-> device transfers (GB/s)\n");
    {
        const size_t max_bytes = 256 << 20;
        void *pinned, *dev;
        std::vector<char> pageable(max_bytes, 1);
        CHECK(cudaMallocHost(&pinned, max_bytes));
        memset(pinned, 1, max_bytes);
        CHECK(cudaMalloc(&dev, max_bytes));
        printf("  %10s %10s %10s %10s %10s\n", "size", "H2D pin", "D2H pin", "H2D page", "D2H page");
        for (size_t bytes: {(size_t) 4 << 10, (size_t) 64 << 10, (size_t) 1 << 20, (size_t) 16 << 20, max_bytes}) {
            int reps = bytes < (1 << 20) ? 200 : 10;
            auto bw = [&](void *dst, const void *src) {
                double t = bestOf(3, [&] { for (int r = 0; r < reps; r++) CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault)); });
                return bytes * (double) reps / t / 1e9;
            };
            double h2d = bw(dev, pinned), d2h = bw(pinned, dev);
            double h2dp = bw(dev, pageable.data()), d2hp = bw(pageable.data(), dev);
            printf("  %8zuKiB %10.2f %10.2f %10.2f %10.2f\n", bytes >> 10, h2d, d2h, h2dp, d2hp);
        }
        // small copy latency
        double t = bestOf(3, [&] { for (int r = 0; r < 1000; r++) CHECK(cudaMemcpy(dev, pinned, 4, cudaMemcpyDefault)); });
        printf("  4 byte cudaMemcpy H2D latency (pinned): %7.2f us\n", t / 1000 * 1e6);
        t = bestOf(3, [&] { for (int r = 0; r < 1000; r++) CHECK(cudaMemcpy(pageable.data(), dev, 4, cudaMemcpyDefault)); });
        printf("  4 byte cudaMemcpy D2H latency (pageable): %5.2f us\n", t / 1000 * 1e6);

        // bidirectional: H2D and D2H on two streams at the same time
        cudaStream_t s0, s1;
        CHECK(cudaStreamCreate(&s0)); CHECK(cudaStreamCreate(&s1));
        void *pinned2, *dev2;
        CHECK(cudaMallocHost(&pinned2, max_bytes)); CHECK(cudaMalloc(&dev2, max_bytes));
        t = bestOf(3, [&] {
            CHECK(cudaMemcpyAsync(dev, pinned, max_bytes, cudaMemcpyHostToDevice, s0));
            CHECK(cudaMemcpyAsync(pinned2, dev2, max_bytes, cudaMemcpyDeviceToHost, s1));
            CHECK(cudaDeviceSynchronize());
        });
        printf("  bidirectional (H2D || D2H, 2 streams): %8.2f GB/s total\n", 2.0 * max_bytes / t / 1e9);
        CHECK(cudaFreeHost(pinned)); CHECK(cudaFreeHost(pinned2)); CHECK(cudaFree(dev)); CHECK(cudaFree(dev2));
        CHECK(cudaStreamDestroy(s0)); CHECK(cudaStreamDestroy(s1));
    }

    printf("\n== device memory (GB/s)\n");
    {
        const size_t n = 64 << 20; // floats
        float *a, *b, *c;
        CHECK(cudaMalloc(&a, n * 4)); CHECK(cudaMalloc(&b, n * 4)); CHECK(cudaMalloc(&c, n * 4));
        CHECK(cudaMemset(b, 0, n * 4)); CHECK(cudaMemset(c, 0, n * 4));
        double t = bestOf(10, [&] { triad<<<prop.multiProcessorCount * 16, 256>>>(a, b, c, 2.0f, n); CHECK(cudaDeviceSynchronize()); });
        printf("  triad kernel (3 x 256 MiB):             %7.1f\n", 3.0 * n * 4 / t / 1e9);
        t = bestOf(10, [&] { copyKernel<<<prop.multiProcessorCount * 16, 256>>>((float4 *) a, (const float4 *) b, n / 4); CHECK(cudaDeviceSynchronize()); });
        printf("  copy kernel (read + write):             %7.1f\n", 2.0 * n * 4 / t / 1e9);
        // (a device to device cudaMemcpy may return before the copy completed)
        t = bestOf(10, [&] { CHECK(cudaMemcpy(a, b, n * 4, cudaMemcpyDeviceToDevice)); CHECK(cudaDeviceSynchronize()); });
        printf("  cudaMemcpy D2D (read + write):          %7.1f\n", 2.0 * n * 4 / t / 1e9);
        t = bestOf(10, [&] { CHECK(cudaMemset(a, 7, n * 4)); CHECK(cudaDeviceSynchronize()); });
        printf("  cudaMemset (write):                     %7.1f\n", 1.0 * n * 4 / t / 1e9);
        CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
    }

    printf("\n== allocation\n");
    {
        for (size_t bytes: {(size_t) 4 << 10, (size_t) 1 << 20, (size_t) 64 << 20}) {
            const int n = 100;
            std::vector<void *> ptrs(n);
            auto t0 = clk::now();
            for (auto &p: ptrs) CHECK(cudaMalloc(&p, bytes));
            auto t1 = clk::now();
            for (auto p: ptrs) CHECK(cudaFree(p));
            auto t2 = clk::now();
            printf("  cudaMalloc/cudaFree %6zu KiB:         %7.1f / %7.1f us\n", bytes >> 10, seconds(t0, t1) / n * 1e6,
                   seconds(t1, t2) / n * 1e6);
        }
    }

    printf("\n== compute\n");
    {
        const int M = 4096, N = 4096, K = 4096;
        float *A, *B, *C;
        CHECK(cudaMalloc(&A, (size_t) M * K * 4)); CHECK(cudaMalloc(&B, (size_t) K * N * 4)); CHECK(cudaMalloc(&C, (size_t) M * N * 4));
        CHECK(cudaMemset(A, 0, (size_t) M * K * 4)); CHECK(cudaMemset(B, 0, (size_t) K * N * 4));
        dim3 grid(N / 128, M / 128);
        sgemm<<<grid, 256>>>(A, B, C, M, N, K);
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        float best = 1e30f;
        for (int r = 0; r < 5; r++) {
            CHECK(cudaEventRecord(e0));
            sgemm<<<grid, 256>>>(A, B, C, M, N, K);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        printf("  SGEMM 4096^3 (simple 8x8 register tiling): %6.2f ms, %7.1f GFLOP/s\n", best, 2.0 * M * N * K / (best * 1e6));
    }
    return 0;
}
