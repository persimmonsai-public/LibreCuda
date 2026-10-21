// Device side assert: passing asserts cost nothing, a failing one prints CUDA's message (on stderr) and makes the
// context report cudaErrorAssert. LibreCUDA learns about the failure from RM some time after the kernel, so the error
// may surface a few synchronizations late.
#include "check.h"
#include <cassert>
#include <chrono>
#include <string>
#include <thread>
#include <unistd.h>

__global__ void checkAll(const int *values, int expected) {
    assert(values[blockIdx.x * blockDim.x + threadIdx.x] == expected);
}

__global__ void checkOne(const int *values) {
    if (threadIdx.x == 0) {
        printf("printf before the assert\n");
    }
    assert(values[threadIdx.x] == 1);
}

static std::string readAll(FILE *file) {
    std::string text;
    rewind(file);
    char buffer[4096];
    size_t n;
    while ((n = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        text.append(buffer, n);
    }
    return text;
}

int main() {
    const int n = 4096;
    int *values;
    CHECK(cudaMalloc(&values, n * sizeof(int)));
    CHECK(cudaMemset(values, 0, n * sizeof(int)));

    // passing asserts
    for (int i = 0; i < 10; i++) {
        checkAll<<<n / 256, 256>>>(values, 0);
        CHECK(cudaDeviceSynchronize());
    }

    // one failing assert: thread 5 of block 0 (stderr and stdout captured)
    CHECK(cudaMemset(values, 0, n * sizeof(int)));
    int ones[32];
    for (int &v: ones) {
        v = 1;
    }
    ones[5] = 0;
    CHECK(cudaMemcpy(values, ones, sizeof(ones), cudaMemcpyHostToDevice));
    fflush(stdout);
    fflush(stderr);
    const int saved_out = dup(STDOUT_FILENO), saved_err = dup(STDERR_FILENO);
    FILE *out_file = tmpfile(), *err_file = tmpfile();
    dup2(fileno(out_file), STDOUT_FILENO);
    dup2(fileno(err_file), STDERR_FILENO);

    checkOne<<<1, 32>>>(values);
    cudaError_t err = cudaSuccess;
    int syncs = 0;
    const auto start = std::chrono::steady_clock::now();
    while (err == cudaSuccess && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
        err = cudaDeviceSynchronize();
        syncs++;
        if (err == cudaSuccess) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    fflush(stdout);
    fflush(stderr);
    dup2(saved_out, STDOUT_FILENO);
    dup2(saved_err, STDERR_FILENO);
    const std::string out = readAll(out_file), errors = readAll(err_file);
    printf("failing assert reported as %s after %d synchronization(s)\n", cudaGetErrorName(err), syncs);
    EXPECT(err == cudaErrorAssert, "expected cudaErrorAssert, got %s", cudaGetErrorName(err));
    EXPECT(out.find("printf before the assert\n") != std::string::npos, "printf output missing: %s", out.c_str());
    const std::string expected_message =
        "block: [0,0,0], thread: [5,0,0] Assertion `values[threadIdx.x] == 1` failed.\n";
    EXPECT(errors.find("device_assert.cu:") != std::string::npos &&
           errors.find(expected_message) != std::string::npos,
           "assert message missing, stderr was: %s", errors.c_str());
    EXPECT(cudaDeviceSynchronize() == cudaErrorAssert, "error is not sticky");
    PASS("device_assert");
    return 0;
}
