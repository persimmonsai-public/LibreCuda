// ncclAllReduce of floats between two ranks (one process each, forked; two GPUs), 256 bytes to 256 MB: time per
// all-reduce back to back on one stream and algorithm bandwidth, each size checked (out of place and in place).
// Plain CUDA runtime code against NCCL's API: run it with LibreCUDA's NCCL (on LibreCUDA's or NVIDIA's runtime) or
// with NVIDIA's NCCL (which needs CUDA_MODULE_LOADING=EAGER):
//
//    nvcc -O2 -std=c++17 -arch=sm_120 -cudart=shared -I<nccl include> benchmarks/nccl_allreduce.cu -L<build>/lib \
//        -lnccl -o nccl_allreduce
//    CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=<build>/lib ./nccl_allreduce
#include <nccl.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "rank %d: %s:%d %s\n", g_rank, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { fprintf(stderr, "rank %d: %s:%d %s: %s\n", g_rank, __FILE__, __LINE__, #x, ncclGetErrorString(r_)); exit(1); } } while (0)

static int g_rank = -1;
const int RANKS = 2;
const size_t SIZES[] = {256, 4 << 10, 64 << 10, 256 << 10, 1 << 20, 16 << 20, 256 << 20};

static void runRank(int rank, ncclUniqueId id) {
    g_rank = rank;
    CK(cudaSetDevice(rank));
    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, RANKS, id, rank));
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const size_t maxBytes = SIZES[sizeof(SIZES) / sizeof(SIZES[0]) - 1];
    float *in, *out;
    CK(cudaMalloc(&in, maxBytes));
    CK(cudaMalloc(&out, maxBytes));

    for (size_t bytes: SIZES) {
        const size_t count = bytes / sizeof(float);
        // correctness: out of place, then in place
        std::vector<float> host(count), got(count);
        for (int inPlace = 0; inPlace < 2; inPlace++) {
            for (size_t i = 0; i < count; i++) host[i] = float((rank + 1) * (inPlace + 3)) + float(i % 31);
            CK(cudaMemcpyAsync(in, host.data(), bytes, cudaMemcpyHostToDevice, stream));
            float *result = inPlace ? in : out;
            NK(ncclAllReduce(in, result, count, ncclFloat, ncclSum, comm, stream));
            CK(cudaMemcpyAsync(got.data(), result, bytes, cudaMemcpyDeviceToHost, stream));
            CK(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < count; i++) {
                float want = 0;
                for (int p = 0; p < RANKS; p++) want += float((p + 1) * (inPlace + 3)) + float(i % 31);
                if (got[i] != want) {
                    fprintf(stderr, "rank %d: %zu bytes%s [%zu] = %g, expected %g\n", rank, bytes,
                            inPlace ? " in place" : "", i, got[i], want);
                    exit(1);
                }
            }
        }
        // time: back to back after a warm-up and a small all-reduce lining the ranks up
        const int reps = bytes <= (size_t(1) << 20) ? 5000 : bytes <= (size_t(16) << 20) ? 200 : 20;
        auto all = [&] { NK(ncclAllReduce(in, out, count, ncclFloat, ncclSum, comm, stream)); };
        for (int i = 0; i < (reps < 100 ? reps : 100); i++) all();
        NK(ncclAllReduce(out, out, 1, ncclFloat, ncclSum, comm, stream));
        CK(cudaStreamSynchronize(stream));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) all();
        CK(cudaStreamSynchronize(stream));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        if (rank == 0) {
            printf("%10zu B: %10.2f us (%.2f GB/s)\n", bytes, us, bytes / us / 1e3);
            fflush(stdout);
        }
    }
    CK(cudaFree(in));
    CK(cudaFree(out));
    NK(ncclCommDestroy(comm));
}

int main() {
    // (no CUDA call before the fork: the children initialize CUDA)
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) return 1;
    const pid_t child = fork();
    if (child == 0) {
        runRank(1, id);
        return 0;
    }
    runRank(0, id);
    int status = 0;
    waitpid(child, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}
