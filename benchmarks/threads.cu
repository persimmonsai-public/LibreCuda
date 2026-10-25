// Host threads submitting at the same time on one device, 1 / 2 / 4 threads with a stream each: launch + synchronize
// round trips, async launches (then one synchronize), synchronizations of an idle stream, event record + query, and
// 4 byte device to host copies + synchronize. Operations per ms of all threads together (each thread does N of them,
// after one untimed). Plain CUDA runtime code: build with nvcc (-cudart shared) and run on any runtime (LibreCUDA or
// NVIDIA's). An argument selects the tests whose name contains it.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(call) do { cudaError_t error_ = (call); if (error_ != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s (%s:%d)\n", #call, cudaGetErrorString(error_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int N = 20000;

__global__ void empty() {}

static int *g_device;

struct Test {
    const char *name;
    void (*body)(cudaStream_t, cudaEvent_t, int);
};

static const Test TESTS[] = {
    {"launch + sync", [](cudaStream_t s, cudaEvent_t, int n) {
        for (int i = 0; i < n; i++) {
            empty<<<1, 1, 0, s>>>();
            CHECK(cudaStreamSynchronize(s));
        }
    }},
    {"async launches", [](cudaStream_t s, cudaEvent_t, int n) {
        for (int i = 0; i < n; i++) {
            empty<<<1, 1, 0, s>>>();
        }
        CHECK(cudaStreamSynchronize(s));
    }},
    {"sync idle stream", [](cudaStream_t s, cudaEvent_t, int n) {
        for (int i = 0; i < n; i++) {
            CHECK(cudaStreamSynchronize(s));
        }
    }},
    {"event record + query", [](cudaStream_t s, cudaEvent_t e, int n) {
        for (int i = 0; i < n; i++) {
            CHECK(cudaEventRecord(e, s));
            cudaEventQuery(e);
        }
        CHECK(cudaStreamSynchronize(s));
    }},
    {"4 B copy D2H + sync", [](cudaStream_t s, cudaEvent_t, int n) {
        int host;
        for (int i = 0; i < n; i++) {
            CHECK(cudaMemcpyAsync(&host, g_device, 4, cudaMemcpyDeviceToHost, s));
            CHECK(cudaStreamSynchronize(s));
        }
    }},
};

static double rate(const Test &test, int threads) {
    std::vector<cudaStream_t> streams(threads);
    std::vector<cudaEvent_t> events(threads);
    for (int i = 0; i < threads; i++) {
        CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
        test.body(streams[i], events[i], 1);  // (first use)
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; i++) {
        pool.emplace_back([&, i] {
            CHECK(cudaSetDevice(0));
            test.body(streams[i], events[i], N);
        });
    }
    for (auto &t: pool) {
        t.join();
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int i = 0; i < threads; i++) {
        CHECK(cudaEventDestroy(events[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }
    return threads * N / ms;
}

int main(int argc, char **argv) {
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&g_device, 64));
    for (const Test &test: TESTS) {
        if (argc > 1 && strstr(test.name, argv[1]) == nullptr) {
            continue;
        }
        printf("%-22s", test.name);
        for (int threads: {1, 2, 4}) {
            printf("  %d: %6.0f/ms", threads, rate(test, threads));
            fflush(stdout);
        }
        printf("\n");
    }
    CHECK(cudaFree(g_device));
    return 0;
}
