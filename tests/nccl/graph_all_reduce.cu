// All-reduces captured into CUDA graphs between two ranks (one process each, forked), out of place as vLLM's pynccl
// issues them: several graphs, each capturing all-reduces of a few sizes (one small exchange, several chunks), replayed
// in rounds with new inputs and checked. Out of place, a chunk's sum doesn't depend on the chunk's upload, and a replay
// that runs them on one stream (LIBRECUDA_GRAPH_LANES=1, or streams sharing a compute channel) must still not put the
// sum first: both ranks would then wait for each other's chunk. In place ones in between.
// Run by tests/nccl/run.sh against LibreCUDA's NCCL (on LibreCUDA's or NVIDIA's runtime) and NVIDIA's NCCL.
#include <nccl.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "rank %d: %s:%d %s\n", g_rank, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { fprintf(stderr, "rank %d: %s:%d %s: %s\n", g_rank, __FILE__, __LINE__, #x, ncclGetErrorString(r_)); exit(1); } } while (0)

static int g_rank = -1;
const int RANKS = 2;
const int GRAPHS = 6;
const int ROUNDS = 3;
// floats: a small exchange (4 KB), several chunks (2.6 MB, as vLLM's 512 tokens x 2560 bf16 x 2), many chunks (12 MB)
const size_t SIZES[] = {1000, 655360, 3000001};
const int NSIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// a rank's input element i of a graph in a round
static float value(int rank, int round, int graph, size_t i) {
    return float((rank + 1) * (i % 101) + round * 7 + graph);
}

static void runRank(int rank, ncclUniqueId id) {
    g_rank = rank;
    CK(cudaSetDevice(rank));
    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, RANKS, id, rank));
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // streams up front, as frameworks have them before they replay graphs: LibreCUDA creates a compute channel when a
    // stream is first assigned to it, which can't happen while a kernel of the context waits for the peer (the graphs'
    // own streams, created at their first launch, then share the channels)
    std::vector<cudaStream_t> streams(16);
    for (cudaStream_t &s: streams) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // per graph and size: the input, the output (out of place) and an in-place buffer
    std::vector<float *> in(GRAPHS * NSIZES), out(GRAPHS * NSIZES), inplace(GRAPHS * NSIZES);
    for (int k = 0; k < GRAPHS * NSIZES; k++) {
        const size_t bytes = SIZES[k % NSIZES] * sizeof(float);
        CK(cudaMalloc(&in[k], bytes));
        CK(cudaMalloc(&out[k], bytes));
        CK(cudaMalloc(&inplace[k], bytes));
        CK(cudaMemset(in[k], 0, bytes));
    }
    auto body = [&](int g) {
        for (int n = 0; n < NSIZES; n++) {
            const int k = g * NSIZES + n;
            NK(ncclAllReduce(in[k], out[k], SIZES[n], ncclFloat, ncclSum, comm, stream));
            CK(cudaMemcpyAsync(inplace[k], in[k], SIZES[n] * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            NK(ncclAllReduce(inplace[k], inplace[k], SIZES[n], ncclFloat, ncclSum, comm, stream));
        }
    };
    std::vector<cudaGraphExec_t> execs(GRAPHS);
    for (int g = 0; g < GRAPHS; g++) {
        body(g);  // (warm-up outside the capture, as frameworks do)
        CK(cudaStreamSynchronize(stream));
        cudaGraph_t graph;
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        body(g);
        CK(cudaStreamEndCapture(stream, &graph));
        CK(cudaGraphInstantiate(&execs[g], graph, 0));
        CK(cudaGraphDestroy(graph));
    }
    std::vector<float> host;
    for (int round = 0; round < ROUNDS; round++) {
        for (int g = 0; g < GRAPHS; g++) {
            for (int n = 0; n < NSIZES; n++) {
                host.resize(SIZES[n]);
                for (size_t i = 0; i < SIZES[n]; i++) host[i] = value(rank, round, g, i);
                CK(cudaMemcpy(in[g * NSIZES + n], host.data(), SIZES[n] * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
        for (int g = 0; g < GRAPHS; g++) CK(cudaGraphLaunch(execs[g], stream));
        // an uncaptured all-reduce between the rounds
        NK(ncclAllReduce(in[0], out[0], SIZES[0], ncclFloat, ncclSum, comm, stream));
        CK(cudaStreamSynchronize(stream));
        for (int g = 0; g < GRAPHS; g++) {
            for (int n = 0; n < NSIZES; n++) {
                for (float *buffer: {out[g * NSIZES + n], inplace[g * NSIZES + n]}) {
                    host.resize(SIZES[n]);
                    CK(cudaMemcpy(host.data(), buffer, SIZES[n] * sizeof(float), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < SIZES[n]; i++) {
                        const float want = value(0, round, g, i) + value(1, round, g, i);
                        if (std::fabs(host[i] - want) > 0) {
                            fprintf(stderr, "rank %d: round %d graph %d size %zu %s element %zu: %g, want %g\n", rank,
                                    round, g, SIZES[n], buffer == out[g * NSIZES + n] ? "out of place" : "in place", i,
                                    host[i], want);
                            exit(1);
                        }
                    }
                }
            }
        }
    }
    for (cudaGraphExec_t exec: execs) CK(cudaGraphExecDestroy(exec));
    for (cudaStream_t s: streams) CK(cudaStreamDestroy(s));
    for (int k = 0; k < GRAPHS * NSIZES; k++) {
        CK(cudaFree(in[k]));
        CK(cudaFree(out[k]));
        CK(cudaFree(inplace[k]));
    }
    NK(ncclCommDestroy(comm));
}

int main() {
    // (no CUDA call before the fork: the children initialize CUDA)
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) return 1;
    const pid_t child = fork();
    if (child == 0) {
        // (no rank left spinning on its GPU when the other one fails)
        prctl(PR_SET_PDEATHSIG, SIGKILL);
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
    printf("PASS all-reduces in CUDA graphs\n");
    return 0;
}
