// A kernel that hits an SM exception must make the context report the matching error. LibreCUDA learns about these
// exceptions from RM some time after the kernel completed, so the error may surface a few synchronizations late;
// correct kernels must never report one. Each kind of exception runs in its own process (a context is unusable after
// one); the parent process doesn't use CUDA.
#include "check.h"
#include <chrono>
#include <cstring>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

__device__ __noinline__ int recurse(int n) {
    volatile int frame[64];
    for (int i = 0; i < 64; i++) {
        frame[i] = n + i;
    }
    if (n == 0) {
        return frame[3];
    }
    // the result is used after the call, so the recursion can't become a loop
    const int r = recurse(n - 1);
    return r + frame[r & 63];
}

__global__ void fill(int *p, int n, int value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] = value + i;
    }
}

// A misaligned PC (warp error 0x5, cudaErrorInvalidPc) was seen with a probe calling through a code pointer off by 4
// bytes, but ptxas usually represents function pointers as jump table offsets, so it can't be provoked reliably here.
enum Fault {
    MISALIGNED_ADDRESS,
    INVALID_ADDRESS_SPACE,
    TRAP,
    STACK_OVERFLOW,
};

__global__ void fault(int kind, int *p) {
    if (threadIdx.x != 3) {
        return;
    }
    switch (kind) {
        case MISALIGNED_ADDRESS:
            *reinterpret_cast<volatile int *>(reinterpret_cast<char *>(p) + 1) = 5;
            break;
        case INVALID_ADDRESS_SPACE: {
            // a global memory load from a shared memory address
            __shared__ int value;
            value = 1;
            int loaded;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(loaded) : "l"(&value));
            p[0] = loaded;
            break;
        }
        case TRAP:
            __trap();
            break;
        case STACK_OVERFLOW:
            p[0] = recurse(100000);
            break;
    }
}

// Runs in a child process: provokes the fault and waits for the error. Returns the process exit code.
static int runFault(int kind, cudaError_t expected, bool check_correct_kernels) {
    const int n = 1 << 20;
    int *buf;
    CHECK(cudaMalloc(&buf, n * sizeof(int)));
    if (check_correct_kernels) {
        // correct kernels for a while: no error may be reported
        const auto start = std::chrono::steady_clock::now();
        int rounds = 0;
        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200)) {
            fill<<<64, 256>>>(buf, n, rounds++);
            CHECK(cudaDeviceSynchronize());
        }
    }
    fault<<<1, 32>>>(kind, buf);
    cudaError_t err = cudaSuccess;
    int syncs = 0;
    const auto fault_time = std::chrono::steady_clock::now();
    while (err == cudaSuccess && std::chrono::steady_clock::now() - fault_time < std::chrono::seconds(2)) {
        err = cudaDeviceSynchronize();
        syncs++;
        if (err == cudaSuccess) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fault_time).count();
    printf("fault %d reported as %s after %d synchronization(s), %.1f ms\n", kind, cudaGetErrorName(err), syncs, ms);
    if (err != expected) {
        fprintf(stderr, "fault %d: expected %s, got %s\n", kind, cudaGetErrorName(expected), cudaGetErrorName(err));
        return 1;
    }
    if (cudaDeviceSynchronize() != expected) {
        fprintf(stderr, "fault %d: error is not sticky\n", kind);
        return 1;
    }
    return 0;
}

int main() {
    const struct {
        Fault kind;
        cudaError_t expected;
    } cases[] = {
        {MISALIGNED_ADDRESS, cudaErrorMisalignedAddress},
        {INVALID_ADDRESS_SPACE, cudaErrorInvalidAddressSpace},
        {TRAP, cudaErrorIllegalInstruction},
        {STACK_OVERFLOW, cudaErrorIllegalAddress},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        fflush(stdout);
        const pid_t child = fork();
        EXPECT(child >= 0, "fork failed");
        if (child == 0) {
            const int code = runFault(cases[i].kind, cases[i].expected, i == 0);
            fflush(stdout);
            _exit(code);
        }
        int status = 0;
        waitpid(child, &status, 0);
        EXPECT(WIFEXITED(status) && WEXITSTATUS(status) == 0, "fault %d not reported correctly", int(cases[i].kind));
    }
    PASS("sm_exception");
    return 0;
}
