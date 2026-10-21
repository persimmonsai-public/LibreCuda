// Device side printf: conversions, strings, many threads, output at synchronization points. The test captures its
// own stdout to check the output.
#include "check.h"
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

__global__ void conversions(const char *str, const char *null_str) {
    printf("int %d neg %i unsigned %u hex %x HEX %X oct %o char %c\n", 42, -7, 3000000000u, 0xbeef, 0xbeef, 8, 'z');
    printf("ll %lld ull %llu l %ld size %zu\n", -1234567890123LL, 18446744073709551615ULL, 99L, (size_t) 77);
    printf("float %f %.3f %e %g double %.10f\n", 1.5f, 3.14159f, 12345.678, 0.0001, 2.0 / 3.0);
    printf("str [%s] [%10s] [%-6s] [%.3s] null [%s]\n", str, "right", "left", "truncate", null_str);
    printf("width [%5d] [%-5d] [%05d] star [%*d] [%.*f] percent %% short %hd char %hhd\n", 7, 7, 7, 6, 9, 2, 2.71828,
           (short) -3, (char) 65);
}

__global__ void manyThreads() {
    printf("thread %d block %d\n", threadIdx.x, blockIdx.x);
}

__device__ int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

__global__ void withCalls(int n) {
    printf("factorial(%d) = %d\n", n, factorial(n));
}

__global__ void tagged(int tag) {
    printf("tag %d\n", tag);
}

struct Capture {
    int saved_fd{};
    FILE *file{};

    void begin() {
        fflush(stdout);
        saved_fd = dup(STDOUT_FILENO);
        file = tmpfile();
        dup2(fileno(file), STDOUT_FILENO);
    }

    std::string end() {
        fflush(stdout);
        dup2(saved_fd, STDOUT_FILENO);
        close(saved_fd);
        std::string text;
        rewind(file);
        char buffer[4096];
        size_t n;
        while ((n = fread(buffer, 1, sizeof(buffer), file)) > 0) {
            text.append(buffer, n);
        }
        fclose(file);
        return text;
    }
};

static size_t countOccurrences(const std::string &text, const std::string &needle) {
    size_t count = 0;
    for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1)) {
        count++;
    }
    return count;
}

static void expectLine(const std::string &text, const std::string &line) {
    EXPECT(text.find(line + "\n") != std::string::npos, "missing line \"%s\" in output:\n%s", line.c_str(),
           text.c_str());
}

int main() {
    size_t fifo = 0;
    CHECK(cudaDeviceGetLimit(&fifo, cudaLimitPrintfFifoSize));
    EXPECT(fifo >= (1 << 20), "printf fifo size %zu", fifo);

    char *str;
    CHECK(cudaMalloc(&str, 16));
    CHECK(cudaMemcpy(str, "device", 7, cudaMemcpyHostToDevice));

    Capture capture;
    capture.begin();
    conversions<<<1, 1>>>(str, nullptr);
    CHECK(cudaDeviceSynchronize());
    std::string out = capture.end();
    char expected[256];
    expectLine(out, "int 42 neg -7 unsigned 3000000000 hex beef HEX BEEF oct 10 char z");
    expectLine(out, "ll -1234567890123 ull 18446744073709551615 l 99 size 77");
    snprintf(expected, sizeof(expected), "float %f %.3f %e %g double %.10f", 1.5, 3.14159f, 12345.678, 0.0001,
             2.0 / 3.0);
    expectLine(out, expected);
    expectLine(out, "str [device] [     right] [left  ] [tru] null [(null)]");
    expectLine(out, "width [    7] [7    ] [00007] star [     9] [2.72] percent % short -3 char 65");

    // many threads: every line exactly once
    const int blocks = 16, threads = 256;
    capture.begin();
    manyThreads<<<blocks, threads>>>();
    CHECK(cudaDeviceSynchronize());
    out = capture.end();
    EXPECT(countOccurrences(out, "\n") == size_t(blocks * threads), "%zu lines instead of %d",
           countOccurrences(out, "\n"), blocks * threads);
    for (int b = 0; b < blocks; b += 5) {
        for (int t = 0; t < threads; t += 37) {
            snprintf(expected, sizeof(expected), "thread %d block %d", t, b);
            EXPECT(countOccurrences(out, std::string(expected) + "\n") == 1, "line \"%s\" missing or repeated",
                   expected);
        }
    }

    // printf in a kernel with calls and recursion (shares the stack with them)
    capture.begin();
    withCalls<<<1, 1>>>(10);
    CHECK(cudaDeviceSynchronize());
    out = capture.end();
    expectLine(out, "factorial(10) = 3628800");

    // output appears at stream and event synchronization, in launch order within a stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t event;
    CHECK(cudaEventCreate(&event));
    capture.begin();
    for (int i = 0; i < 3; i++) {
        tagged<<<1, 1, 0, stream>>>(i);
    }
    CHECK(cudaStreamSynchronize(stream));
    tagged<<<1, 1, 0, stream>>>(3);
    CHECK(cudaEventRecord(event, stream));
    CHECK(cudaEventSynchronize(event));
    out = capture.end();
    EXPECT(out == "tag 0\ntag 1\ntag 2\ntag 3\n", "stream output:\n%s", out.c_str());
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaEventDestroy(event));

    // the buffer is reused: many rounds of output beyond its size
    capture.begin();
    for (int round = 0; round < 40; round++) {
        manyThreads<<<blocks, threads>>>();
        CHECK(cudaDeviceSynchronize());
    }
    out = capture.end();
    EXPECT(countOccurrences(out, "\n") == size_t(40 * blocks * threads), "%zu lines after 40 rounds",
           countOccurrences(out, "\n"));

    CHECK(cudaFree(str));
    PASS("device_printf");
    return 0;
}
