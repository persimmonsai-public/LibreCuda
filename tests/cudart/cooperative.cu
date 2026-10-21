// Cooperative launches: cooperative_groups::this_grid().sync() across all blocks of a grid, launched with
// cudaLaunchCooperativeKernel and with cudaLaunchKernelEx's cooperative attribute; grids that cannot be resident at
// once are rejected; kernels from regular launches see an invalid grid group.
#include "check.h"
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

// Each round, every block reads the value its right neighbour block wrote in the previous round. Without a working
// grid barrier, blocks read stale or future values.
__global__ void rotate(int *buf, int rounds) {
    cg::grid_group grid = cg::this_grid();
    const int n = gridDim.x;
    int *cur = buf, *next = buf + n;
    for (int r = 0; r < rounds; r++) {
        if (threadIdx.x == 0) {
            next[blockIdx.x] = cur[(blockIdx.x + 1) % n] + 1;
        }
        grid.sync();
        int *t = cur;
        cur = next;
        next = t;
    }
}

__global__ void gridValid(int *out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out = cg::this_grid().is_valid() ? 1 : 0;
    }
}

static void checkRotate(const std::vector<int> &host, int n, int rounds, const char *what) {
    // after r rounds block b holds initial[(b + r) % n] + r, with initial[b] = b * 1000
    const int *result = host.data() + (rounds % 2 == 0 ? 0 : n);
    for (int b = 0; b < n; b++) {
        const int expected = ((b + rounds) % n) * 1000 + rounds;
        EXPECT(result[b] == expected, "%s: block %d holds %d, expected %d", what, b, result[b], expected);
    }
}

int main() {
    int attr = 0;
    CHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeLaunch, 0));
    EXPECT(attr == 1, "cooperative launch attribute is %d", attr);

    int sm_count = 0, blocks_per_sm = 0;
    const int threads = 128;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, rotate, threads, 0));
    const int n = sm_count * blocks_per_sm;
    const int rounds = 1001;
    printf("grid of %d blocks (%d SMs x %d), %d grid syncs\n", n, sm_count, blocks_per_sm, rounds);

    std::vector<int> init(2 * n, -1), host(2 * n);
    for (int b = 0; b < n; b++) {
        init[b] = b * 1000;
    }
    int *buf;
    CHECK(cudaMalloc(&buf, 2 * n * sizeof(int)));

    // full grid, cudaLaunchCooperativeKernel
    {
        CHECK(cudaMemcpy(buf, init.data(), init.size() * sizeof(int), cudaMemcpyHostToDevice));
        int r = rounds;
        void *args[] = {&buf, &r};
        CHECK(cudaLaunchCooperativeKernel((void *) rotate, dim3(n), dim3(threads), args, 0, nullptr));
        CHECK(cudaMemcpy(host.data(), buf, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
        checkRotate(host, n, rounds, "cudaLaunchCooperativeKernel");
    }

    // two cooperative kernels on different streams at the same time, each with half the grid: they must not share a
    // barrier
    {
        const int half = n / 2;
        int *bufs[2];
        cudaStream_t streams[2];
        for (int i = 0; i < 2; i++) {
            CHECK(cudaMalloc(&bufs[i], 2 * half * sizeof(int)));
            std::vector<int> h(2 * half, -1);
            for (int b = 0; b < half; b++) {
                h[b] = b * 1000;
            }
            CHECK(cudaMemcpy(bufs[i], h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }
        for (int i = 0; i < 2; i++) {
            int r = rounds;
            void *args[] = {&bufs[i], &r};
            CHECK(cudaLaunchCooperativeKernel((void *) rotate, dim3(half), dim3(threads), args, 0, streams[i]));
        }
        CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 2; i++) {
            std::vector<int> h(2 * half);
            CHECK(cudaMemcpy(h.data(), bufs[i], h.size() * sizeof(int), cudaMemcpyDeviceToHost));
            checkRotate(h, half, rounds, i == 0 ? "concurrent kernel 0" : "concurrent kernel 1");
            CHECK(cudaStreamDestroy(streams[i]));
            CHECK(cudaFree(bufs[i]));
        }
    }

    // cudaLaunchKernelEx with the cooperative attribute
    {
        CHECK(cudaMemcpy(buf, init.data(), init.size() * sizeof(int), cudaMemcpyHostToDevice));
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeCooperative;
        attribute.val.cooperative = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(n);
        config.blockDim = dim3(threads);
        config.attrs = &attribute;
        config.numAttrs = 1;
        CHECK(cudaLaunchKernelEx(&config, rotate, buf, rounds - 1));
        CHECK(cudaMemcpy(host.data(), buf, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
        checkRotate(host, n, rounds - 1, "cudaLaunchKernelEx");
    }

    // too many blocks to be resident at once
    {
        int r = 1;
        void *args[] = {&buf, &r};
        const cudaError_t err = cudaLaunchCooperativeKernel((void *) rotate, dim3(n + 1), dim3(threads), args, 0,
                                                            nullptr);
        EXPECT(err == cudaErrorCooperativeLaunchTooLarge, "oversized grid returned %s", cudaGetErrorName(err));
        cudaGetLastError();
    }

    // a regular launch has no grid workspace
    {
        int *valid;
        CHECK(cudaMalloc(&valid, sizeof(int)));
        gridValid<<<2, 32>>>(valid);
        int v = -1;
        CHECK(cudaMemcpy(&v, valid, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT(v == 0, "this_grid().is_valid() is %d in a regular launch", v);
        CHECK(cudaFree(valid));
    }

    CHECK(cudaFree(buf));
    PASS("cooperative");
    return 0;
}
