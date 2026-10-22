// Several host threads using the runtime at once: per-thread streams with kernels, copies and synchronization;
// a thread waiting for the GPU must not keep other threads from submitting work (a kernel waits for a flag another
// thread copies in); synchronizing a stream while another thread launches into it; concurrent cudaMalloc / cudaFree;
// device synchronization while streams are created and destroyed.
#include "check.h"
#include <atomic>
#include <thread>
#include <vector>

__global__ void addValue(int *p, int n, int value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] += value;
    }
}

// spins until *flag is set (or about a second passed: then reports failure instead of hanging the GPU)
__global__ void waitForFlag(volatile int *flag, int *result) {
    const long long start = clock64();
    while (*flag == 0) {
        if (clock64() - start > 3000000000ll) {
            *result = -1;
            return;
        }
    }
    *result = 1;
}

__global__ void increment(int *counter) {
    atomicAdd(counter, 1);
}

int main() {
    // warm up: creates the context
    CHECK(cudaFree(nullptr));

    // independent work in several threads
    {
        const int threads = 6, n = 1 << 16, rounds = 100;
        std::vector<std::thread> workers;
        std::atomic<int> failures{0};
        for (int t = 0; t < threads; t++) {
            workers.emplace_back([&, t] {
                cudaStream_t stream;
                CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                int *device, *host;
                CHECK(cudaMalloc(&device, n * sizeof(int)));
                CHECK(cudaMallocHost(&host, n * sizeof(int)));
                for (int i = 0; i < n; i++) {
                    host[i] = t * 1000;
                }
                CHECK(cudaMemcpyAsync(device, host, n * sizeof(int), cudaMemcpyHostToDevice, stream));
                for (int r = 0; r < rounds; r++) {
                    addValue<<<32, 256, 0, stream>>>(device, n, 1);
                    if (r % 10 == 0) {
                        CHECK(cudaStreamSynchronize(stream));
                    }
                }
                CHECK(cudaMemcpyAsync(host, device, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
                CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < n; i++) {
                    if (host[i] != t * 1000 + rounds) {
                        fprintf(stderr, "thread %d: [%d] = %d\n", t, i, host[i]);
                        failures++;
                        break;
                    }
                }
                // pageable copies too (staged)
                std::vector<int> pageable(n);
                CHECK(cudaMemcpy(pageable.data(), device, n * sizeof(int), cudaMemcpyDeviceToHost));
                if (pageable[n - 1] != t * 1000 + rounds) {
                    failures++;
                }
                CHECK(cudaFreeHost(host));
                CHECK(cudaFree(device));
                CHECK(cudaStreamDestroy(stream));
            });
        }
        for (auto &w: workers) {
            w.join();
        }
        EXPECT(failures == 0, "%d threads saw wrong data", failures.load());
    }

    // a thread waiting for the GPU while another thread provides what the GPU waits for
    {
        int *flag, *result;
        CHECK(cudaMalloc(&flag, sizeof(int)));
        CHECK(cudaMalloc(&result, sizeof(int)));
        CHECK(cudaMemset(flag, 0, sizeof(int)));
        CHECK(cudaMemset(result, 0, sizeof(int)));
        CHECK(cudaDeviceSynchronize());
        cudaStream_t waiting, setting;
        CHECK(cudaStreamCreateWithFlags(&waiting, cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&setting, cudaStreamNonBlocking));
        waitForFlag<<<1, 1, 0, waiting>>>(flag, result);
        std::atomic<bool> waiter_started{false};
        std::thread waiter([&] {
            waiter_started = true;
            CHECK(cudaStreamSynchronize(waiting));
        });
        while (!waiter_started) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // with the waiter holding a global lock this would only run after the kernel timed out
        static int one = 1;
        CHECK(cudaMemcpyAsync(flag, &one, sizeof(int), cudaMemcpyHostToDevice, setting));
        CHECK(cudaStreamSynchronize(setting));
        waiter.join();
        int host_result = 0;
        CHECK(cudaMemcpy(&host_result, result, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT(host_result == 1, "the kernel didn't see the flag (result %d)", host_result);
        CHECK(cudaStreamDestroy(waiting));
        CHECK(cudaStreamDestroy(setting));
        CHECK(cudaFree(flag));
        CHECK(cudaFree(result));
    }

    // one thread synchronizes a stream while others launch into it
    {
        int *counter;
        CHECK(cudaMalloc(&counter, sizeof(int)));
        CHECK(cudaMemset(counter, 0, sizeof(int)));
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        const int launchers = 3, launches = 300;
        std::atomic<int> done{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < launchers; t++) {
            threads.emplace_back([&] {
                for (int i = 0; i < launches; i++) {
                    increment<<<1, 32, 0, stream>>>(counter);
                }
                CHECK(cudaGetLastError());
                done++;
            });
        }
        threads.emplace_back([&] {
            while (done < launchers) {
                CHECK(cudaStreamSynchronize(stream));
            }
        });
        for (auto &t: threads) {
            t.join();
        }
        CHECK(cudaStreamSynchronize(stream));
        int host_counter = 0;
        CHECK(cudaMemcpy(&host_counter, counter, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT(host_counter == launchers * launches * 32, "counter %d, expected %d", host_counter,
               launchers * launches * 32);
        CHECK(cudaStreamDestroy(stream));
        CHECK(cudaFree(counter));
    }

    // allocations from several threads
    {
        const int threads = 8, rounds = 200;
        std::atomic<int> failures{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < threads; t++) {
            workers.emplace_back([&, t] {
                for (int r = 0; r < rounds; r++) {
                    const size_t size = size_t(256) << ((t + r) % 12);
                    unsigned char *p;
                    CHECK(cudaMalloc(&p, size));
                    CHECK(cudaMemset(p, t + 1, size));
                    unsigned char last = 0;
                    CHECK(cudaMemcpy(&last, p + size - 1, 1, cudaMemcpyDeviceToHost));
                    if (last != t + 1) {
                        failures++;
                    }
                    CHECK(cudaFree(p));
                }
            });
        }
        for (auto &w: workers) {
            w.join();
        }
        EXPECT(failures == 0, "%d allocations had wrong contents", failures.load());
    }

    // device synchronization while other threads create, use and destroy streams
    {
        std::atomic<bool> stop{false};
        int *scratch;
        CHECK(cudaMalloc(&scratch, 1024 * sizeof(int)));
        std::thread syncer([&] {
            while (!stop) {
                CHECK(cudaDeviceSynchronize());
            }
        });
        std::vector<std::thread> churners;
        for (int t = 0; t < 3; t++) {
            churners.emplace_back([&] {
                for (int i = 0; i < 50; i++) {
                    cudaStream_t s;
                    CHECK(cudaStreamCreate(&s));
                    addValue<<<4, 256, 0, s>>>(scratch, 1024, 1);
                    cudaEvent_t e;
                    CHECK(cudaEventCreate(&e));
                    CHECK(cudaEventRecord(e, s));
                    CHECK(cudaEventSynchronize(e));
                    CHECK(cudaEventDestroy(e));
                    CHECK(cudaStreamDestroy(s));
                }
            });
        }
        for (auto &t: churners) {
            t.join();
        }
        stop = true;
        syncer.join();
        CHECK(cudaFree(scratch));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("multithreading");
    return 0;
}
