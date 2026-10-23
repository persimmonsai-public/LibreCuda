// Interprocess communication: a parent process exports device memory (a large allocation and a small one) and an
// interprocess event to a child process (forked before either uses CUDA). Once the child initialized CUDA, the parent
// starts a kernel that writes the memory after about a second and records the event; the child finds the event not
// reached yet, waits for it on the GPU, reads the memory, writes into it and opens the same memory twice. The parent
// sees the child's writes. Checked against NVIDIA's runtime too.
#include "check.h"
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

constexpr int LARGE = 1 << 20, SMALL = 1000;

struct Handles {
    cudaIpcMemHandle_t large, small;
    cudaIpcEventHandle_t event;
};

// spins for about the given number of clock cycles (every block), then fills p[i] = base + i
__global__ void fillAfter(int *p, int n, int base, long long cycles) {
    if (threadIdx.x == 0) {
        const long long start = clock64();
        while (clock64() - start < cycles) {
        }
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] = base + i;
    }
}

// counts the elements of p that aren't base + i, then negates them
__global__ void checkAndNegate(int *p, int n, int base, int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (p[i] != base + i) atomicAdd(errors, 1);
        p[i] = -p[i];
    }
}

static void writeAll(int fd, const void *data, size_t size) {
    EXPECT(write(fd, data, size) == ssize_t(size), "pipe write");
}

static void readAll(int fd, void *data, size_t size) {
    EXPECT(read(fd, data, size) == ssize_t(size), "pipe read");
}

static int child(int in, int out) {
    // (initialized before the parent's kernel starts)
    CHECK(cudaFree(nullptr));
    const int initialized = 1;
    writeAll(out, &initialized, sizeof(initialized));
    Handles h{};
    readAll(in, &h, sizeof(h));
    int *large = nullptr, *small = nullptr, *again = nullptr;
    CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&large), h.large, cudaIpcMemLazyEnablePeerAccess));
    CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&small), h.small, cudaIpcMemLazyEnablePeerAccess));
    cudaEvent_t event;
    CHECK(cudaIpcOpenEventHandle(&event, h.event));
    // the parent's kernel is still running: the event isn't reached yet
    EXPECT(cudaEventQuery(event) == cudaErrorNotReady, "event reached early");
    cudaGetLastError();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaStreamWaitEvent(stream, event, 0));
    int *errors;
    CHECK(cudaMalloc(&errors, 2 * sizeof(int)));
    CHECK(cudaMemsetAsync(errors, 0, 2 * sizeof(int), stream));
    checkAndNegate<<<64, 256, 0, stream>>>(large, LARGE, 7, errors);
    checkAndNegate<<<4, 256, 0, stream>>>(small, SMALL, 1000, errors + 1);
    int counts[2];
    CHECK(cudaMemcpyAsync(counts, errors, sizeof(counts), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventSynchronize(event));
    CHECK(cudaEventQuery(event));
    EXPECT(counts[0] == 0 && counts[1] == 0, "child read %d / %d wrong elements", counts[0], counts[1]);
    // opening the memory again gives the same address
    CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&again), h.large, cudaIpcMemLazyEnablePeerAccess));
    EXPECT(again == large, "reopened at %p, first at %p", static_cast<void *>(again), static_cast<void *>(large));
    CHECK(cudaIpcCloseMemHandle(again));
    int first = 0;
    CHECK(cudaMemcpy(&first, large + 5, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT(first == -12, "after one close: %d", first);
    CHECK(cudaIpcCloseMemHandle(large));
    CHECK(cudaIpcCloseMemHandle(small));
    CHECK(cudaEventDestroy(event));
    const int done = 1;
    writeAll(out, &done, sizeof(done));
    return 0;
}

int main() {
    int toChild[2], toParent[2];
    EXPECT(pipe(toChild) == 0 && pipe(toParent) == 0, "pipe");
    const pid_t pid = fork();
    EXPECT(pid >= 0, "fork");
    if (pid == 0) {
        close(toChild[1]);
        close(toParent[0]);
        exit(child(toChild[0], toParent[1]));
    }
    // (a child that fails closes its end: the parent's reads fail instead of waiting forever)
    close(toChild[0]);
    close(toParent[1]);
    int *large, *small;
    CHECK(cudaMalloc(&large, LARGE * sizeof(int)));
    CHECK(cudaMalloc(&small, SMALL * sizeof(int)));
    CHECK(cudaMemset(large, 0, LARGE * sizeof(int)));
    CHECK(cudaMemset(small, 0, SMALL * sizeof(int)));
    cudaEvent_t event;
    CHECK(cudaEventCreateWithFlags(&event, cudaEventInterprocess | cudaEventDisableTiming));
    Handles h{};
    CHECK(cudaIpcGetMemHandle(&h.large, large));
    CHECK(cudaIpcGetMemHandle(&h.small, small));
    CHECK(cudaIpcGetEventHandle(&h.event, event));
    // events not created as interprocess have no handle
    cudaEvent_t local;
    CHECK(cudaEventCreate(&local));
    cudaIpcEventHandle_t unused;
    EXPECT(cudaIpcGetEventHandle(&unused, local) != cudaSuccess, "handle of a local event");
    cudaGetLastError();
    int initialized = 0, done = 0;
    readAll(toParent[0], &initialized, sizeof(initialized));
    fillAfter<<<64, 256>>>(large, LARGE, 7, 2500000000LL);
    fillAfter<<<4, 256>>>(small, SMALL, 1000, 0);
    CHECK(cudaEventRecord(event, 0));
    writeAll(toChild[1], &h, sizeof(h));
    readAll(toParent[0], &done, sizeof(done));
    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child failed (status %d)", status);
    std::vector<int> l(LARGE), s(SMALL);
    CHECK(cudaMemcpy(l.data(), large, LARGE * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(s.data(), small, SMALL * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < LARGE; i++) EXPECT(l[i] == -(7 + i), "large[%d] = %d", i, l[i]);
    for (int i = 0; i < SMALL; i++) EXPECT(s[i] == -(1000 + i), "small[%d] = %d", i, s[i]);
    CHECK(cudaEventDestroy(event));
    CHECK(cudaEventDestroy(local));
    CHECK(cudaFree(large));
    CHECK(cudaFree(small));
    PASS("memory and event shared with a child process");
    return 0;
}
