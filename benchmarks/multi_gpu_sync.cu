// Synchronization across devices and host threads: two-device launch + synchronize round trips, cross-device event
// ping-pong (GPU waits for the other GPU), host waits for another device's event, peer copy latency and bandwidth
// (with and without peer access) and launch + synchronize round trips from several host threads.
// Plain CUDA runtime code: build with nvcc and run on any runtime (LibreCUDA or NVIDIA's). Needs 2 devices.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s (%s:%d)\n", #call, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double seconds(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

__global__ void emptyKernel() {}

// advances a token shared by both devices from expected to expected + 1 (the ping-pong checks the order)
__global__ void pass(volatile int *token, int expected, int *errors) {
    if (*token != expected) {
        atomicAdd(errors, 1);
    }
    *token = expected + 1;
}

// waits for both devices (cudaMemcpyPeer may return before the copy completed)
static void synchronizeBoth() {
    int current;
    CHECK(cudaGetDevice(&current));
    for (int d = 0; d < 2; d++) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaSetDevice(current));
}

static void row(const char *name, double value, const char *unit) {
    printf("%-58s %10.2f %s\n", name, value, unit);
}

int main(int argc, char **argv) {
    const int iterations = argc > 1 ? atoi(argv[1]) : 2000;
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        fprintf(stderr, "needs 2 devices, found %d\n", count);
        return 1;
    }
    cudaStream_t stream[2];
    cudaEvent_t done[2];
    for (int d = 0; d < 2; d++) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
        CHECK(cudaEventCreateWithFlags(&done[d], cudaEventDisableTiming));
        // first uses set up lazily created state
        for (int i = 0; i < 20; i++) {
            emptyKernel<<<1, 1>>>();
            emptyKernel<<<1, 1, 0, stream[d]>>>();
            CHECK(cudaDeviceSynchronize());
        }
    }

    // 1. one device: launch + synchronize (reference)
    {
        CHECK(cudaSetDevice(0));
        auto t0 = clk::now();
        for (int i = 0; i < iterations; i++) {
            emptyKernel<<<1, 1, 0, stream[0]>>>();
            CHECK(cudaStreamSynchronize(stream[0]));
        }
        row("one device: launch + cudaStreamSynchronize", seconds(t0, clk::now()) / iterations * 1e6, "us");
    }

    // 2. both devices: launch on each, then synchronize each
    {
        auto t0 = clk::now();
        for (int i = 0; i < iterations; i++) {
            for (int d = 0; d < 2; d++) {
                CHECK(cudaSetDevice(d));
                emptyKernel<<<1, 1>>>();
            }
            for (int d = 0; d < 2; d++) {
                CHECK(cudaSetDevice(d));
                CHECK(cudaDeviceSynchronize());
            }
        }
        row("two devices: launch on both + cudaDeviceSynchronize both", seconds(t0, clk::now()) / iterations * 1e6, "us");
    }

    // 3. GPU to GPU ping-pong through events: device 0 runs, device 1 waits for it on the GPU, runs, device 0 waits...
    int *token, *errors;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMallocHost(&token, sizeof(int)));
    CHECK(cudaMallocHost(&errors, sizeof(int)));
    {
        *token = 0;
        *errors = 0;
        auto t0 = clk::now();
        for (int i = 0; i < iterations; i++) {
            CHECK(cudaSetDevice(0));
            if (i > 0) {
                CHECK(cudaStreamWaitEvent(stream[0], done[1], 0));
            }
            pass<<<1, 1, 0, stream[0]>>>(token, 2 * i, errors);
            CHECK(cudaEventRecord(done[0], stream[0]));
            CHECK(cudaSetDevice(1));
            CHECK(cudaStreamWaitEvent(stream[1], done[0], 0));
            pass<<<1, 1, 0, stream[1]>>>(token, 2 * i + 1, errors);
            CHECK(cudaEventRecord(done[1], stream[1]));
        }
        CHECK(cudaStreamSynchronize(stream[1]));
        const double t = seconds(t0, clk::now());
        if (*token != 2 * iterations || *errors != 0) {
            fprintf(stderr, "ping-pong out of order: token %d (expected %d), %d errors\n", *token, 2 * iterations, *errors);
            return 1;
        }
        row("GPU ping-pong via cudaStreamWaitEvent, per hop", t / (2 * iterations) * 1e6, "us");
    }

    // 4. host waits for another device's event (device 1 current)
    {
        auto t0 = clk::now();
        for (int i = 0; i < iterations; i++) {
            CHECK(cudaSetDevice(0));
            emptyKernel<<<1, 1, 0, stream[0]>>>();
            CHECK(cudaEventRecord(done[0], stream[0]));
            CHECK(cudaSetDevice(1));
            CHECK(cudaEventSynchronize(done[0]));
        }
        row("launch on device 0 + cudaEventSynchronize from device 1", seconds(t0, clk::now()) / iterations * 1e6, "us");
    }

    // 5. peer copies, without and with peer access
    const size_t big = 64 << 20;
    void *buf[2];
    for (int d = 0; d < 2; d++) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaMalloc(&buf[d], big));
        CHECK(cudaMemset(buf[d], d, big));
    }
    CHECK(cudaSetDevice(0));
    for (int peer = 0; peer < 2; peer++) {
        if (peer) {
            int can = 0;
            CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
            if (!can) {
                printf("no peer access between devices 0 and 1\n");
                break;
            }
            CHECK(cudaSetDevice(1));
            CHECK(cudaDeviceEnablePeerAccess(0, 0));
            CHECK(cudaSetDevice(0));
            CHECK(cudaDeviceEnablePeerAccess(1, 0));
        }
        const char *mode = peer ? "peer access" : "no peer access";
        char name[128];
        const int small_iterations = iterations / 4;
        auto t0 = clk::now();
        for (int i = 0; i < small_iterations; i++) {
            CHECK(cudaMemcpyPeer(buf[1], 1, buf[0], 0, 4));
        }
        synchronizeBoth();
        snprintf(name, sizeof(name), "cudaMemcpyPeer 4 B, %s", mode);
        row(name, seconds(t0, clk::now()) / small_iterations * 1e6, "us");
        for (int dir = 0; dir < 2; dir++) {
            CHECK(cudaMemcpyPeer(buf[1 - dir], 1 - dir, buf[dir], dir, big));
            synchronizeBoth();
            const int reps = 5;
            t0 = clk::now();
            for (int i = 0; i < reps; i++) {
                CHECK(cudaMemcpyPeer(buf[1 - dir], 1 - dir, buf[dir], dir, big));
            }
            synchronizeBoth();
            snprintf(name, sizeof(name), "cudaMemcpyPeer 64 MiB %d -> %d, %s", dir, 1 - dir, mode);
            row(name, reps * double(big) / seconds(t0, clk::now()) / 1e9, "GB/s");
        }
        // async peer copies on the stream of device 0
        synchronizeBoth();
        t0 = clk::now();
        for (int i = 0; i < small_iterations; i++) {
            CHECK(cudaMemcpyPeerAsync(buf[1], 1, buf[0], 0, 4, stream[0]));
            CHECK(cudaStreamSynchronize(stream[0]));
        }
        snprintf(name, sizeof(name), "cudaMemcpyPeerAsync 4 B + sync, %s", mode);
        row(name, seconds(t0, clk::now()) / small_iterations * 1e6, "us");
    }

    // 6. host threads, each launching and synchronizing its own stream
    synchronizeBoth();
    auto threads = [&](int n, bool sameDevice) {
        std::vector<std::thread> pool;
        std::vector<double> perThread(n);
        std::vector<clk::time_point> finished(n);
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        clk::time_point t0;
        for (int t = 0; t < n; t++) {
            pool.emplace_back([&, t] {
                const int device = sameDevice ? 0 : t % 2;
                CHECK(cudaSetDevice(device));
                cudaStream_t s;
                CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
                for (int i = 0; i < 20; i++) {
                    emptyKernel<<<1, 1, 0, s>>>();
                    CHECK(cudaStreamSynchronize(s));
                }
                // all threads start together once set up
                ready++;
                while (!go) {
                }
                auto start = clk::now();
                for (int i = 0; i < iterations; i++) {
                    emptyKernel<<<1, 1, 0, s>>>();
                    CHECK(cudaStreamSynchronize(s));
                }
                finished[t] = clk::now();
                perThread[t] = seconds(start, finished[t]) / iterations * 1e6;
                CHECK(cudaStreamDestroy(s));
            });
        }
        while (ready < n) {
        }
        t0 = clk::now();
        go = true;
        for (auto &th: pool) {
            th.join();
        }
        const double total = seconds(t0, *std::max_element(finished.begin(), finished.end()));
        double mean = 0;
        for (double v: perThread) {
            mean += v;
        }
        char name[128];
        snprintf(name, sizeof(name), "%d thread(s), %s: launch + sync per thread", n,
                 sameDevice ? "one device" : "one device each");
        row(name, mean / n, "us");
        snprintf(name, sizeof(name), "%d thread(s), %s: total round trips", n, sameDevice ? "one device" : "one device each");
        row(name, n * iterations / total / 1e3, "per ms");
    };
    threads(1, true);
    threads(2, true);
    threads(4, true);
    threads(2, false);
    threads(4, false);

    CHECK(cudaFreeHost(token));
    CHECK(cudaFreeHost(errors));
    CHECK(cudaGetLastError());
    return 0;
}
