// CUDA graph replay: a chain of 100 empty kernels, 4 independent branches of 25 empty kernels, and 4 branches of 25
// kernels of about 20k dependent FMAs (one block each). Time per cudaGraphLaunch, launched back to back (best of 5
// rounds of 200 launches). Plain CUDA runtime code: build with nvcc and run on any runtime (LibreCUDA or NVIDIA's).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s (%s:%d)\n", #call, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void empty() {}

__global__ void heavy(float *out, int iterations) {
    float x = threadIdx.x;
    for (int i = 0; i < iterations; i++) x = x * 1.0000001f + 0.5f;
    if (x == -1.0f) out[threadIdx.x] = x;  // (keeps the loop)
}

// a graph of `branches` independent chains of `length` kernels, captured from forked streams
static cudaGraphExec_t build(int branches, int length, bool heavyKernels, float *out) {
    cudaStream_t origin, side[8];
    CHECK(cudaStreamCreateWithFlags(&origin, cudaStreamNonBlocking));
    for (int b = 0; b < branches; b++) CHECK(cudaStreamCreateWithFlags(&side[b], cudaStreamNonBlocking));
    cudaEvent_t fork, join[8];
    CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    for (int b = 0; b < branches; b++) CHECK(cudaEventCreateWithFlags(&join[b], cudaEventDisableTiming));
    CHECK(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
    CHECK(cudaEventRecord(fork, origin));
    for (int b = 0; b < branches; b++) {
        cudaStream_t s = b == 0 ? origin : side[b];
        if (b > 0) CHECK(cudaStreamWaitEvent(s, fork, 0));
        for (int i = 0; i < length; i++) {
            if (heavyKernels) {
                heavy<<<1, 32, 0, s>>>(out, 20000);
            } else {
                empty<<<1, 1, 0, s>>>();
            }
        }
        if (b > 0) {
            CHECK(cudaEventRecord(join[b], s));
            CHECK(cudaStreamWaitEvent(origin, join[b], 0));
        }
    }
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(origin, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphDestroy(graph));
    return exec;
}

static double timeGraph(cudaGraphExec_t exec, cudaStream_t s) {
    double best = 1e30;
    for (int round = 0; round < 6; round++) {
        const int n = 200;
        CHECK(cudaStreamSynchronize(s));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) CHECK(cudaGraphLaunch(exec, s));
        CHECK(cudaStreamSynchronize(s));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        if (round > 0) best = std::min(best, us);  // (round 0 warms up)
    }
    return best;
}

int main() {
    float *out;
    CHECK(cudaMalloc(&out, 1024));
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    printf("chain of 100 empty kernels:       %7.1f us per launch\n", timeGraph(build(1, 100, false, out), s));
    printf("4 branches x 25 empty kernels:    %7.1f us per launch\n", timeGraph(build(4, 25, false, out), s));
    printf("4 branches x 25 heavier kernels:  %7.1f us per launch\n", timeGraph(build(4, 25, true, out), s));
    return 0;
}
