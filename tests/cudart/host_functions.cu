// Host functions and stream callbacks run asynchronously: cudaLaunchHostFunc returns while the stream is still busy,
// the function sees the results of the stream's preceding work, following work waits for the function (and sees what
// it wrote), functions of a stream run in order (many of them, on two streams at once), stream callbacks get their
// arguments and status, and synchronizing a stream waits for its host functions. Checked against NVIDIA's runtime too.
#include "check.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

__global__ void spin(long long cycles, int *flag, int value) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
    *flag = value;
}

__global__ void addFromHost(int *out, const volatile int *host) {
    *out = *host + 1;
}

struct Check {
    int *device;   // written by the kernel before the host function
    int seen;      // what the host function saw
    int *host;     // written by the host function for the kernel after it
};

static void CUDART_CB observe(void *data) {
    auto *c = static_cast<Check *>(data);
    c->seen = *c->device;
    *c->host = c->seen * 10;
}

struct Order {
    std::atomic<int> next{0};
    std::atomic<int> errors{0};
};

struct Step {
    Order *order;
    int index;
};

static void CUDART_CB inOrder(void *data) {
    auto *step = static_cast<Step *>(data);
    if (step->order->next.fetch_add(1) != step->index) {
        step->order->errors++;
    }
}

static std::atomic<int> g_callbacks{0};
static cudaStream_t g_callbackStream = nullptr;

static void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *data) {
    if (stream == g_callbackStream && status == cudaSuccess && *static_cast<int *>(data) == 42) {
        g_callbacks++;
    }
}

int main() {
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    int *managed, *pinned;
    CHECK(cudaMallocManaged(&managed, 16 * sizeof(int)));
    CHECK(cudaMallocHost(&pinned, 16 * sizeof(int)));
    for (int i = 0; i < 16; i++) {
        managed[i] = pinned[i] = 0;
    }
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // the call doesn't wait for the 100 ms kernel; the function sees its result, the next kernel the function's
    Check check{managed, -1, pinned};
    const long long cycles = static_cast<long long>(clock_khz) * 100;
    spin<<<1, 1, 0, stream>>>(cycles, managed, 7);
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(cudaLaunchHostFunc(stream, observe, &check));
    const double call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    addFromHost<<<1, 1, 0, stream>>>(managed + 1, pinned);
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(call_ms < 50, "cudaLaunchHostFunc waited for the stream: %.1f ms", call_ms);
    EXPECT(check.seen == 7, "the host function saw %d", check.seen);
    EXPECT(managed[1] == 71, "the kernel after the host function saw %d", managed[1] - 1);
    PASS("asynchronous host function");

    // many host functions on two streams: in order per stream
    cudaStream_t second;
    CHECK(cudaStreamCreate(&second));
    Order orders[2];
    std::vector<Step> steps;
    steps.reserve(400);
    for (int i = 0; i < 200; i++) {
        for (int s = 0; s < 2; s++) {
            steps.push_back(Step{&orders[s], i});
            if (i % 10 == 0) {
                spin<<<1, 1, 0, s == 0 ? stream : second>>>(clock_khz / 10, managed + 2 + s, i);
            }
            CHECK(cudaLaunchHostFunc(s == 0 ? stream : second, inOrder, &steps.back()));
        }
    }
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamSynchronize(second));
    for (int s = 0; s < 2; s++) {
        EXPECT(orders[s].next == 200 && orders[s].errors == 0, "stream %d: %d functions, %d out of order", s,
               orders[s].next.load(), orders[s].errors.load());
    }
    PASS("host functions in order");

    // stream callbacks
    int value = 42;
    g_callbackStream = stream;
    for (int i = 0; i < 5; i++) {
        spin<<<1, 1, 0, stream>>>(clock_khz / 10, managed + 4, i);
        CHECK(cudaStreamAddCallback(stream, callback, &value, 0));
    }
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(g_callbacks == 5, "callbacks: %d", g_callbacks.load());
    PASS("stream callbacks");

    // the legacy default stream and device synchronization
    check = Check{managed, -1, pinned};
    spin<<<1, 1>>>(cycles / 10, managed, 9);
    CHECK(cudaLaunchHostFunc(0, observe, &check));
    CHECK(cudaDeviceSynchronize());
    EXPECT(check.seen == 9 && pinned[0] == 90, "legacy default stream: %d %d", check.seen, pinned[0]);
    PASS("host function on the legacy default stream");

    CHECK(cudaStreamDestroy(second));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFreeHost(pinned));
    CHECK(cudaFree(managed));
    CHECK(cudaGetLastError());
    return 0;
}
