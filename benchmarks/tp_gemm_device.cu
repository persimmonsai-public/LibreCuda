// Tensor-parallel GEMM on two GPUs with NCCL 2.28's device API: a row-parallel layer Y = X @ W with K split between
// the ranks (one process each, forked); each rank multiplies its half (bf16, cuBLAS) straight into a symmetric memory
// window and the partial results are summed by kernels reading and writing the peers' windows through LSA pointers,
// against cuBLAS + ncclAllReduce. Sequential (GEMM, then the all-reduce) and chunked overlap (the all-reduce of a row
// block runs on a high-priority stream while the next one is multiplied). Device API all-reduces:
//  - rsag: reduce-scatter + all-gather in one kernel, in place: each rank sums its share of the elements over all
//    ranks and writes the sums into every rank's window (reads and writes half the data over the link);
//  - pull: every rank reads all peers' partial results and sums them into a separate output (reads all of it).
// The results are checked against ncclAllReduce's.
//
// Plain CUDA runtime code, built against NCCL's headers (2.28 or later) and run with LibreCUDA's NCCL on LibreCUDA's
// or NVIDIA's runtime, or with NVIDIA's NCCL (which needs CUDA_MODULE_LOADING=EAGER), as tests/nccl/run.sh does:
//
//    nvcc -O2 -std=c++17 -arch=sm_120 -cudart=shared -I<nccl include> benchmarks/tp_gemm_device.cu \
//        -L<build>/lib -lnccl -lcublas -o tp_gemm_device
//    CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=<build>/lib ./tp_gemm_device
#include <nccl.h>
#include <nccl_device.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "rank %d: %s:%d %s\n", g_rank, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { fprintf(stderr, "rank %d: %s:%d %s: %s\n", g_rank, __FILE__, __LINE__, #x, ncclGetErrorString(r_)); exit(1); } } while (0)
#define BK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "rank %d: %s:%d %s: %d\n", g_rank, __FILE__, __LINE__, #x, int(s_)); exit(1); } } while (0)

static int g_rank = -1;
static int g_blocks = 8;
static double g_share0 = 0.5;  // rsag: rank 0's share of the elements (TP_SHARE0)
const int RANKS = 2;
const int MAX_BLOCKS = 32, THREADS = 512;  // per all-reduce kernel (TP_BLOCKS, default 8); one LSA barrier per block
const int UNROLL = 4;                  // 16-byte vectors in flight per thread and rank

struct Shape { int M, K, N; };
const Shape SHAPES[] = {{512, 8192, 4096}, {2048, 8192, 4096}, {8192, 8192, 4096}};

// sums 8 bf16 values of each source (in rank order: the same sums on every rank)
__device__ uint4 sum8(const uint4 *v, int n) {
    float s[8] = {};
    for (int r = 0; r < n; r++) {
        const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v[r]);
        for (int j = 0; j < 4; j++) {
            const float2 f = __bfloat1622float2(h[j]);
            s[2 * j] += f.x;
            s[2 * j + 1] += f.y;
        }
    }
    uint4 out;
    __nv_bfloat162 *o = reinterpret_cast<__nv_bfloat162 *>(&out);
    for (int j = 0; j < 4; j++) o[j] = __floats2bfloat162_rn(s[2 * j], s[2 * j + 1]);
    return out;
}

// All-reduce of `vectors` 16-byte vectors at `offset` in every rank's window. Block b of every rank works on the same
// elements, so the per-block barriers order everything: the first one waits for every rank's input (each rank's
// GEMM finished before its kernel started), the second one for every rank's reads and writes, after which the window
// (rsag) or the input (pull) may be overwritten.
template<bool PULL>
__global__ void __launch_bounds__(THREADS) allReduce(ncclDevComm comm, ncclWindow_t win, size_t offset,
                                                     size_t vectors, size_t split, uint4 *out) {
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x};
    bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);

    const int n = comm.lsaSize;
    const uint4 *src[RANKS];
    uint4 *dst[RANKS];
    for (int r = 0; r < n; r++) {
        src[r] = static_cast<const uint4 *>(ncclGetLsaPointer(win, offset, r));
        dst[r] = static_cast<uint4 *>(ncclGetLsaPointer(win, offset, r));
    }
    // rsag (two ranks): rank 0 sums the vectors before `split`, rank 1 the rest; pull: all of them
    size_t begin = 0, end = vectors;
    if (!PULL) {
        begin = comm.lsaRank == 0 ? 0 : split;
        end = comm.lsaRank == 0 ? split : vectors;
    }
    const size_t stride = size_t(gridDim.x) * THREADS;
    for (size_t base = begin + blockIdx.x * size_t(THREADS) + threadIdx.x; base < end; base += stride * UNROLL) {
        uint4 v[UNROLL][RANKS];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const size_t i = base + u * stride;
            if (i < end) {
                for (int r = 0; r < n; r++) v[u][r] = src[r][i];
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            const size_t i = base + u * stride;
            if (i < end) {
                const uint4 s = sum8(v[u], n);
                if (PULL) {
                    out[i] = s;
                } else {
                    for (int r = 0; r < n; r++) dst[r][i] = s;
                }
            }
        }
    }
    bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

enum Method { NCCL, RSAG, PULL };
const char *const METHOD_NAMES[] = {"ncclAllReduce", "device API rsag", "device API pull"};

struct Rank {
    ncclComm_t comm;
    ncclDevComm dev;
    ncclWindow_t win;
    cublasHandle_t blas;
    cudaStream_t compute, comm_stream;
    std::vector<cudaEvent_t> events;
    __nv_bfloat16 *x, *w, *y, *out;  // y: the window (GEMM output, all-reduced in place except by pull)
};

// the row-parallel layer: GEMM of `chunks` row blocks, each all-reduced as soon as it is ready (chunks > 1: on the
// communication stream)
static void layer(Rank &k, Method method, const Shape &s, int chunks) {
    const int rows = s.M / chunks, kHalf = s.K / RANKS;
    const float one = 1, zero = 0;
    for (int c = 0; c < chunks; c++) {
        __nv_bfloat16 *y = k.y + size_t(c) * rows * s.N;
        // row-major Y[rows, N] = X[rows, K/2] W[K/2, N]: column-major Y^T = W^T X^T
        BK(cublasGemmEx(k.blas, CUBLAS_OP_N, CUBLAS_OP_N, s.N, rows, kHalf, &one, k.w, CUDA_R_16BF, s.N,
                        k.x + size_t(c) * rows * kHalf, CUDA_R_16BF, kHalf, &zero, y, CUDA_R_16BF, s.N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        cudaStream_t stream = k.compute;
        if (chunks > 1) {
            CK(cudaEventRecord(k.events[c], k.compute));
            CK(cudaStreamWaitEvent(k.comm_stream, k.events[c], 0));
            stream = k.comm_stream;
        }
        const size_t elements = size_t(rows) * s.N, offset = size_t(c) * elements;
        switch (method) {
            case NCCL: NK(ncclAllReduce(y, y, elements, ncclBfloat16, ncclSum, k.comm, stream)); break;
            case RSAG:
                allReduce<false><<<g_blocks, THREADS, 0, stream>>>(k.dev, k.win, offset * 2, elements / 8,
                                                                    size_t(elements / 8 * g_share0), nullptr);
                break;
            case PULL:
                allReduce<true><<<g_blocks, THREADS, 0, stream>>>(k.dev, k.win, offset * 2, elements / 8, 0,
                                                               reinterpret_cast<uint4 *>(k.out + offset));
                break;
        }
    }
    if (chunks > 1) {
        // the next layer waits for the communication
        CK(cudaEventRecord(k.events[chunks], k.comm_stream));
        CK(cudaStreamWaitEvent(k.compute, k.events[chunks], 0));
    }
}

static void barrier(Rank &k) {
    // (both ranks' streams idle, then a small all-reduce)
    CK(cudaDeviceSynchronize());
    NK(ncclAllReduce(k.out, k.out, 1, ncclFloat, ncclSum, k.comm, k.compute));
    CK(cudaStreamSynchronize(k.compute));
}

template<typename F>
static double timeMs(Rank &k, int reps, F &&fn) {
    fn();
    fn();
    barrier(k);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) fn();
    CK(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

static void fill(std::vector<__nv_bfloat16> &v, uint32_t seed, float scale) {
    for (auto &e : v) {
        seed = seed * 1664525u + 1013904223u;
        e = __float2bfloat16((float(seed >> 8) / float(1 << 24) - 0.5f) * scale);
    }
}

static std::vector<__nv_bfloat16> download(const __nv_bfloat16 *p, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    CK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
    return h;
}

static void runRank(int rank, ncclUniqueId id) {
    g_rank = rank;
    CK(cudaSetDevice(rank));
    Rank k{};
    NK(ncclCommInitRank(&k.comm, RANKS, id, rank));
    BK(cublasCreate(&k.blas));
    int lowest = 0, highest = 0;
    CK(cudaDeviceGetStreamPriorityRange(&lowest, &highest));
    CK(cudaStreamCreateWithPriority(&k.compute, cudaStreamNonBlocking, lowest));
    CK(cudaStreamCreateWithPriority(&k.comm_stream, cudaStreamNonBlocking, highest));
    BK(cublasSetStream(k.blas, k.compute));
    k.events.resize(17);
    for (auto &e : k.events) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

    size_t maxOut = 0, maxX = 0, maxW = 0;
    for (const Shape &s : SHAPES) {
        maxOut = std::max(maxOut, size_t(s.M) * s.N);
        maxX = std::max(maxX, size_t(s.M) * s.K / RANKS);
        maxW = std::max(maxW, size_t(s.K) / RANKS * s.N);
    }
    NK(ncclMemAlloc((void **) &k.y, maxOut * 2));
    NK(ncclCommWindowRegister(k.comm, k.y, maxOut * 2, &k.win, NCCL_WIN_COLL_SYMMETRIC));
    ncclDevCommRequirements reqs{};
    reqs.lsaBarrierCount = MAX_BLOCKS;
    NK(ncclDevCommCreate(k.comm, &reqs, &k.dev));
    CK(cudaMalloc(&k.x, maxX * 2));
    CK(cudaMalloc(&k.w, maxW * 2));
    CK(cudaMalloc(&k.out, maxOut * 2));

    for (const Shape &s : SHAPES) {
        const size_t xn = size_t(s.M) * s.K / RANKS, wn = size_t(s.K) / RANKS * s.N, yn = size_t(s.M) * s.N;
        std::vector<__nv_bfloat16> hx(xn), hw(wn);
        fill(hx, 1 + rank, 2.0f / sqrtf(sqrtf(float(s.K))));
        fill(hw, 101 + rank, 2.0f / sqrtf(sqrtf(float(s.K))));
        CK(cudaMemcpy(k.x, hx.data(), xn * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(k.w, hw.data(), wn * 2, cudaMemcpyHostToDevice));
        const int reps = s.M <= 2048 ? 20 : 8;

        // reference: GEMM + ncclAllReduce
        layer(k, NCCL, s, 1);
        CK(cudaDeviceSynchronize());
        const std::vector<__nv_bfloat16> ref = download(k.y, yn);
        const double gemm = timeMs(k, reps, [&] {
            const float one = 1, zero = 0;
            BK(cublasGemmEx(k.blas, CUBLAS_OP_N, CUBLAS_OP_N, s.N, s.M, s.K / RANKS, &one, k.w, CUDA_R_16BF, s.N,
                            k.x, CUDA_R_16BF, s.K / RANKS, &zero, k.y, CUDA_R_16BF, s.N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
        });
        char line[512];
        int at = snprintf(line, sizeof(line), "M=%5d K=%d N=%d (%3zu MB out): GEMM %6.2f ms", s.M, s.K, s.N,
                          yn * 2 >> 20, gemm);
        for (Method m : {NCCL, RSAG, PULL}) {
            at += snprintf(line + at, sizeof(line) - at, " | %s", METHOD_NAMES[m]);
            for (int chunks : {1, 4}) {
                // correctness first, then the time
                layer(k, m, s, chunks);
                CK(cudaDeviceSynchronize());
                const std::vector<__nv_bfloat16> got = download(m == PULL ? k.out : k.y, yn);
                double err = 0;
                for (size_t i = 0; i < yn; i++) {
                    err = std::max(err, double(fabsf(__bfloat162float(got[i]) - __bfloat162float(ref[i]))));
                }
                if (err > 0.0625) {
                    fprintf(stderr, "rank %d: %s, %d chunks, M=%d: max difference %g from ncclAllReduce\n", rank,
                            METHOD_NAMES[m], chunks, s.M, err);
                    exit(1);
                }
                const double t = timeMs(k, reps, [&] { layer(k, m, s, chunks); });
                at += snprintf(line + at, sizeof(line) - at, " %s %6.2f", chunks == 1 ? "seq" : "ov4", t);
            }
        }
        if (rank == 0) printf("%s ms\n", line);
        fflush(stdout);
    }

    NK(ncclDevCommDestroy(k.comm, &k.dev));
    NK(ncclCommWindowDeregister(k.comm, k.win));
    NK(ncclMemFree(k.y));
    CK(cudaFree(k.x));
    CK(cudaFree(k.w));
    CK(cudaFree(k.out));
    BK(cublasDestroy(k.blas));
    NK(ncclCommDestroy(k.comm));
}

int main() {
    // (no CUDA call before the fork: the children initialize CUDA)
    if (const char *b = getenv("TP_BLOCKS")) g_blocks = std::min(MAX_BLOCKS, std::max(1, atoi(b)));
    if (const char *f = getenv("TP_SHARE0")) g_share0 = std::min(1.0, std::max(0.0, atof(f)));
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
