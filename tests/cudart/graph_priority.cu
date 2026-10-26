// Kernel node priorities in graphs: kernels captured from a high priority stream keep its priority
// (cudaLaunchAttributePriority), and in a graph instantiated with cudaGraphInstantiateFlagUseNodePriority that runs
// on a normal priority stream they still run ahead of a long grid of the same graph, as GROMACS's PME transforms run
// ahead of its non-bonded kernel. Checked against NVIDIA's runtime too.
#include "check.h"

#include <vector>

__device__ unsigned long long globalTimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// each block spins for `ns`; the last block to finish records the time
__global__ void longGrid(unsigned long long ns, unsigned long long *end) {
    const unsigned long long start = globalTimer();
    while (globalTimer() - start < ns) {
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(end, globalTimer());
}

__global__ void stamp(unsigned long long *at) {
    *at = globalTimer();
}

__global__ void spin(unsigned long long ns) {
    const unsigned long long start = globalTimer();
    while (globalTimer() - start < ns) {
    }
}

int main() {
    int least = 0, greatest = 0;
    CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    cudaStream_t launch, normal, high;
    CHECK(cudaStreamCreateWithPriority(&launch, cudaStreamNonBlocking, least));
    CHECK(cudaStreamCreateWithPriority(&normal, cudaStreamNonBlocking, least));
    CHECK(cudaStreamCreateWithPriority(&high, cudaStreamNonBlocking, greatest));

    int device = 0, sms = 0, perSm = 0;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, longGrid, 128, 0));
    const int blocks = 40 * sms * perSm;
    unsigned long long *times;
    CHECK(cudaMalloc(&times, 3 * sizeof(unsigned long long)));

    // the launch stream forks into a normal priority branch (a stamp, then about 40 ms of 1 ms blocks) and a high
    // priority one (a 5 ms kernel, so that the long grid has started, then a stamp)
    cudaEvent_t fork, joinNormal, joinHigh;
    CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&joinNormal, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&joinHigh, cudaEventDisableTiming));
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(launch, cudaStreamCaptureModeGlobal));
    CHECK(cudaEventRecord(fork, launch));
    CHECK(cudaStreamWaitEvent(normal, fork, 0));
    CHECK(cudaStreamWaitEvent(high, fork, 0));
    stamp<<<1, 1, 0, normal>>>(times + 0);
    longGrid<<<blocks, 128, 0, normal>>>(1000000ULL, times + 1);
    spin<<<1, 1, 0, high>>>(5000000ULL);
    stamp<<<1, 1, 0, high>>>(times + 2);
    CHECK(cudaEventRecord(joinNormal, normal));
    CHECK(cudaEventRecord(joinHigh, high));
    CHECK(cudaStreamWaitEvent(launch, joinNormal, 0));
    CHECK(cudaStreamWaitEvent(launch, joinHigh, 0));
    CHECK(cudaStreamEndCapture(launch, &graph));

    // the captured kernel nodes carry their streams' priorities
    size_t count = 0;
    CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    int highNodes = 0, kernelNodes = 0;
    for (cudaGraphNode_t node: nodes) {
        cudaGraphNodeType type;
        CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) continue;
        kernelNodes++;
        cudaKernelNodeAttrValue value{};
        CHECK(cudaGraphKernelNodeGetAttribute(node, cudaLaunchAttributePriority, &value));
        highNodes += value.priority == greatest;
    }
    EXPECT(kernelNodes == 4 && highNodes == 2, "%d of %d kernel nodes have the high priority", highNodes, kernelNodes);

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, cudaGraphInstantiateFlagUseNodePriority));
    for (int run = 0; run < 2; run++) {
        CHECK(cudaMemset(times, 0, 3 * sizeof(unsigned long long)));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGraphLaunch(exec, launch));
        CHECK(cudaStreamSynchronize(launch));
        unsigned long long h[3];
        CHECK(cudaMemcpy(h, times, sizeof(h), cudaMemcpyDeviceToHost));
        const double longMs = (h[1] - h[0]) / 1e6, probeMs = (h[2] - h[0]) / 1e6;
        printf("launch %d: the high priority branch ran after %.1f ms of a %.1f ms grid\n", run, probeMs, longMs);
        EXPECT(longMs > 30, "the long grid took only %.1f ms", longMs);
        EXPECT(probeMs < longMs / 2, "the high priority node waited for the grid (%.1f of %.1f ms)", probeMs, longMs);
    }

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(fork));
    CHECK(cudaEventDestroy(joinNormal));
    CHECK(cudaEventDestroy(joinHigh));
    CHECK(cudaFree(times));
    CHECK(cudaStreamDestroy(launch));
    CHECK(cudaStreamDestroy(normal));
    CHECK(cudaStreamDestroy(high));
    PASS("graph_priority");
    return 0;
}
