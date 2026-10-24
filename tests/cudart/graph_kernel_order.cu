// Kernels of a graph run one after another and see each other's writes: a kernel that sets a flag late is followed by
// one that records it (a kernel started early would see the flag cleared), chains of kernels that read what the
// previous one wrote in other blocks, replayed on two streams at once (the kernels of one replay run after those of the
// other)
#include "check.h"
#include <vector>

__global__ void clearFlag(int *flag) { *flag = 0; }
__global__ void setFlagLate(int *flag) {
    const long long t0 = clock64();
    while (clock64() - t0 < 200000) {}
    *(volatile int *) flag = 1;
}
__global__ void recordFlag(const int *flag, int *seen, int k) { seen[k] = *(volatile const int *) flag; }

// adds 1 to a value another block wrote, through the read-only (L1) path; the first block finishes late
__global__ void step(const float *__restrict__ src, float *__restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0) {
        const long long t0 = clock64();
        while (clock64() - t0 < 20000) {}
    }
    // (32 bit arithmetic: a 64 bit remainder is a function call, which keeps the kernel out of command lists)
    if (i < n) dst[i] = __ldg(&src[(unsigned(i) * 7919u + 13u) % unsigned(n)]) + 1.f;
}

int main() {
    cudaStream_t s, t;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&t, cudaStreamNonBlocking));

    {
        constexpr int K = 100;
        int *flag, *seen;
        CHECK(cudaMalloc(&flag, sizeof(int)));
        CHECK(cudaMalloc(&seen, K * sizeof(int)));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < K; k++) {
            clearFlag<<<1, 1, 0, s>>>(flag);
            setFlagLate<<<1, 1, 0, s>>>(flag);
            recordFlag<<<1, 1, 0, s>>>(flag, seen, k);
        }
        CHECK(cudaStreamEndCapture(s, &graph));
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CHECK(cudaMemsetAsync(seen, 0, K * sizeof(int), s));
        CHECK(cudaGraphLaunch(exec, s));
        CHECK(cudaStreamSynchronize(s));
        std::vector<int> h(K);
        CHECK(cudaMemcpy(h.data(), seen, K * sizeof(int), cudaMemcpyDeviceToHost));
        for (int k = 0; k < K; k++) {
            EXPECT(h[k] == 1, "kernel %d started before the preceding one completed", k);
        }
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaFree(flag));
        CHECK(cudaFree(seen));
    }

    for (int n: {1024, 1 << 20}) {
        constexpr int K = 500; // even: the result is in x
        const int threads = 128, blocks = (n + threads - 1) / threads;
        float *x, *y;
        CHECK(cudaMalloc(&x, n * sizeof(float)));
        CHECK(cudaMalloc(&y, n * sizeof(float)));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK(cudaMemset(x, 0, n * sizeof(float)));
        CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < K; k++) {
            step<<<blocks, threads, 0, s>>>(k % 2 ? y : x, k % 2 ? x : y, n);
        }
        CHECK(cudaStreamEndCapture(s, &graph));
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        std::vector<float> h(n);
        for (int r = 0; r < 4; r++) {
            // each replay continues from the result of the previous one
            CHECK(cudaGraphLaunch(exec, r % 2 ? t : s));
            CHECK(cudaGraphLaunch(exec, r % 2 ? s : t));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));
            const int expected = 2 * (r + 1) * K;
            for (int i = 0; i < n; i++) {
                EXPECT(h[i] == float(expected), "n %d, replay %d: element %d is %.0f, expected %d", n, r, i, h[i],
                       expected);
            }
        }
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaFree(x));
        CHECK(cudaFree(y));
    }
    PASS("graph kernel order");
    return 0;
}
