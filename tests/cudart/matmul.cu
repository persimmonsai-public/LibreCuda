// Tiled SGEMM with static shared memory, 2D grids/blocks, events for timing; compared to a CPU reference
#include "check.h"
#include <vector>
#include <random>

constexpr int TILE = 32;

__global__ void matmulTiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x, bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[(size_t) row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[(size_t) bk * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t) row * N + col] = acc;
}

// register blocked variant: each thread computes a 4x4 micro tile with 64x64 block tiles
__global__ void __launch_bounds__(256) matmulRegBlocked(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[16][64];
    __shared__ float Bs[16][64];
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    int rowBase = blockIdx.y * 64, colBase = blockIdx.x * 64;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 16 * 64; i += 256) {
            int r = i / 16, k = i % 16;
            As[k][r] = (rowBase + r < M && k0 + k < K) ? A[(size_t) (rowBase + r) * K + k0 + k] : 0.f;
            int kk = i / 64, c = i % 64;
            Bs[kk][c] = (k0 + kk < K && colBase + c < N) ? B[(size_t) (k0 + kk) * N + colBase + c] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; k++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = As[k][ty * 4 + i]; b[i] = Bs[k][tx * 4 + i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = rowBase + ty * 4 + i, c = colBase + tx * 4 + j;
            if (r < M && c < N) C[(size_t) r * N + c] = acc[i][j];
        }
}

int main() {
    const int M = 1000, N = 1100, K = 1200;
    std::vector<float> A((size_t) M * K), B((size_t) K * N), C((size_t) M * N), ref((size_t) M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x: A) x = dist(rng);
    for (auto &x: B) x = dist(rng);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double) A[(size_t) i * K + k] * B[(size_t) k * N + j];
            ref[(size_t) i * N + j] = (float) acc;
        }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, A.size() * 4));
    CHECK(cudaMalloc(&dB, B.size() * 4));
    CHECK(cudaMalloc(&dC, C.size() * 4));
    CHECK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int variant = 0; variant < 2; variant++) {
        CHECK(cudaMemset(dC, 0, C.size() * 4));
        auto run = [&] {
            if (variant == 0) {
                dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
                matmulTiled<<<grid, block>>>(dA, dB, dC, M, N, K);
            } else {
                dim3 grid((N + 63) / 64, (M + 63) / 64);
                matmulRegBlocked<<<grid, 256>>>(dA, dB, dC, M, N, K);
            }
        };
        run();
        CHECK(cudaGetLastError());
        const int iters = 20;
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) run();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < C.size(); i++)
            EXPECT(nearlyEqual(C[i], ref[i], 1e-3, 1e-3), "variant %d C[%zu] = %f, expected %f", variant, i, C[i], ref[i]);
        printf("%s: %.3f ms/iter, %.1f GFLOP/s\n", variant == 0 ? "tiled" : "register blocked", ms / iters,
               2.0 * M * N * K * iters / (ms * 1e6));
    }
    PASS("matmul");
    return 0;
}
