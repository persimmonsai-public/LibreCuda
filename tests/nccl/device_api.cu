// NCCL 2.28's device API between two ranks (one process each, forked): symmetric memory windows (ncclMemAlloc,
// ncclCommWindowRegister), a device communicator with LSA barriers (ncclDevCommCreate), and kernels using NCCL's own
// device headers:
//  - exchange: every rank writes a pattern into every peer's window (ncclGetPeerPointer), an LSA barrier, then each
//    rank checks what the peers wrote into its window;
//  - all-reduce: a kernel sums every rank's data, read through LSA pointers between two barriers, repeated with new
//    inputs (the barriers' epochs advance);
//  - latency: the 4 KB device API all-reduce against ncclAllReduce of 4 KB.
// Run by tests/nccl/run.sh against LibreCUDA's NCCL (on LibreCUDA's or NVIDIA's runtime) and NVIDIA's NCCL.
#include <nccl.h>
#include <nccl_device.h>
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
const size_t ELEMENTS = 1024;  // 4 KB of floats per rank

// window layout (floats): [0, RANKS * ELEMENTS) exchange slots (slot p: written by rank p), then the all-reduce input
__global__ void exchange(ncclDevComm comm, ncclWindow_t win, int round) {
    for (int peer = 0; peer < comm.nRanks; peer++) {
        float *slot = (float *) ncclGetPeerPointer(win, (size_t(comm.rank) * ELEMENTS) * sizeof(float), peer);
        for (size_t i = threadIdx.x; i < ELEMENTS; i += blockDim.x) slot[i] = float(round * 1000 + comm.rank * 100 + i % 97);
    }
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), comm, ncclTeamTagLsa(), 0};
    bar.sync(ncclCoopCta(), cuda::memory_order_seq_cst);
}

// out = sum over ranks of their input (the window's second part), read through LSA pointers
__global__ void allReduce(ncclDevComm comm, ncclWindow_t win, float *out, size_t count) {
    const size_t input = size_t(RANKS) * ELEMENTS * sizeof(float);
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x};
    // every rank's input is in place
    bar.sync(ncclCoopCta(), cuda::memory_order_seq_cst);
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        float sum = 0;
        for (int peer = 0; peer < comm.lsaSize; peer++) sum += ((const float *) ncclGetLsaPointer(win, input, peer))[i];
        out[i] = sum;
    }
    // nobody overwrites its input while others may still read it
    bar.sync(ncclCoopCta(), cuda::memory_order_seq_cst);
}

static void runRank(int rank, ncclUniqueId id) {
    g_rank = rank;
    CK(cudaSetDevice(rank));
    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, RANKS, id, rank));
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const size_t windowBytes = (size_t(RANKS) + 1) * ELEMENTS * sizeof(float);
    float *buffer = nullptr;
    NK(ncclMemAlloc((void **) &buffer, windowBytes));
    ncclWindow_t win;
    NK(ncclCommWindowRegister(comm, buffer, windowBytes, &win, NCCL_WIN_COLL_SYMMETRIC));
    ncclDevCommRequirements reqs{};
    reqs.lsaBarrierCount = 4;
    ncclDevComm dev;
    NK(ncclDevCommCreate(comm, &reqs, &dev));
    if (dev.rank != rank || dev.nRanks != RANKS || dev.lsaSize != RANKS) {
        fprintf(stderr, "rank %d: device communicator rank %d of %d, LSA size %d\n", rank, dev.rank, dev.nRanks, dev.lsaSize);
        exit(1);
    }

    // exchange, three rounds
    std::vector<float> host(size_t(RANKS) * ELEMENTS);
    for (int round = 1; round <= 3; round++) {
        exchange<<<1, 256, 0, stream>>>(dev, win, round);
        CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(host.data(), buffer, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int p = 0; p < RANKS; p++) {
            for (size_t i = 0; i < ELEMENTS; i++) {
                const float want = float(round * 1000 + p * 100 + i % 97);
                if (host[p * ELEMENTS + i] != want) {
                    fprintf(stderr, "rank %d: round %d slot %d [%zu] = %g, expected %g\n", rank, round, p, i,
                            host[p * ELEMENTS + i], want);
                    exit(1);
                }
            }
        }
    }
    printf("rank %d: exchange through peer pointers and LSA barriers ok\n", rank);

    // all-reduce, ten rounds with new inputs
    float *input = buffer + size_t(RANKS) * ELEMENTS, *out;
    CK(cudaMalloc(&out, ELEMENTS * sizeof(float)));
    std::vector<float> in(ELEMENTS), result(ELEMENTS);
    for (int round = 0; round < 10; round++) {
        for (size_t i = 0; i < ELEMENTS; i++) in[i] = float((rank + 1) * (round + 1)) + float(i % 13);
        CK(cudaMemcpyAsync(input, in.data(), ELEMENTS * sizeof(float), cudaMemcpyHostToDevice, stream));
        allReduce<<<4, 256, 0, stream>>>(dev, win, out, ELEMENTS);
        CK(cudaMemcpyAsync(result.data(), out, ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < ELEMENTS; i++) {
            float want = 0;
            for (int p = 0; p < RANKS; p++) want += float((p + 1) * (round + 1)) + float(i % 13);
            if (result[i] != want) {
                fprintf(stderr, "rank %d: all-reduce round %d [%zu] = %g, expected %g\n", rank, round, i, result[i], want);
                exit(1);
            }
        }
    }
    printf("rank %d: all-reduce through LSA pointers ok\n", rank);

    // latency: back-to-back 4 KB all-reduces, the device API kernel and ncclAllReduce
    auto time = [&](auto fn) {
        for (int i = 0; i < 200; i++) fn();
        CK(cudaStreamSynchronize(stream));
        const int n = 5000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) fn();
        CK(cudaStreamSynchronize(stream));
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
    };
    const double kernel = time([&] { allReduce<<<1, 256, 0, stream>>>(dev, win, out, ELEMENTS); });
    const double nccl = time([&] { NK(ncclAllReduce(input, out, ELEMENTS, ncclFloat, ncclSum, comm, stream)); });
    if (rank == 0) printf("4 KB all-reduce: device API kernel %.2f us, ncclAllReduce %.2f us\n", kernel, nccl);

    NK(ncclDevCommDestroy(comm, &dev));
    NK(ncclCommWindowDeregister(comm, win));
    NK(ncclMemFree(buffer));
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
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "rank 1 failed\n");
        return 1;
    }
    printf("PASS NCCL device API\n");
    return 0;
}
