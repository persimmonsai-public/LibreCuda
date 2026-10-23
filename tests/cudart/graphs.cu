// CUDA graphs and stream capture: captured kernels, copies, fills and host functions replayed several times; fork /
// join across streams with events; capture errors (unjoined forks, synchronization, unsafe calls, implicit legacy
// stream synchronization) and the relaxed mode; graphs built node by node with all node types, queried, updated per
// node and with cudaGraphExecUpdate; child graphs. Checked against NVIDIA's runtime too.
#include "check.h"
#include <atomic>
#include <cstdio>
#include <vector>

__global__ void addValue(int *p, int n, int value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] += value;
    }
}

__global__ void scale(int *p, int n, int factor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] *= factor;
    }
}

__global__ void sum2(const int *a, const int *b, int *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

static std::atomic<int> g_hostCalls{0};

static void CUDART_CB hostIncrement(void *) {
    g_hostCalls++;
}

static std::vector<int> download(const int *device, int n) {
    std::vector<int> host(n);
    CHECK(cudaMemcpy(host.data(), device, n * sizeof(int), cudaMemcpyDeviceToHost));
    return host;
}

static size_t nodeCount(cudaGraph_t graph) {
    size_t n = 0;
    CHECK(cudaGraphGetNodes(graph, nullptr, &n));
    return n;
}

static size_t edgeCount(cudaGraph_t graph) {
    size_t n = 0;
    CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &n));
    return n;
}

int main() {
    const int n = 4096;
    int *a, *b, *c;
    CHECK(cudaMalloc(&a, n * sizeof(int)));
    CHECK(cudaMalloc(&b, n * sizeof(int)));
    CHECK(cudaMalloc(&c, n * sizeof(int)));
    cudaStream_t s0, s1;
    CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

    // a captured chain: memset, kernel, copy, kernel, host function; replayed several times
    {
        CHECK(cudaMemset(a, 0, n * sizeof(int)));
        CHECK(cudaMemset(b, 0, n * sizeof(int)));
        CHECK(cudaDeviceSynchronize());
        g_hostCalls = 0;
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        cudaStreamCaptureStatus status;
        CHECK(cudaStreamIsCapturing(s0, &status));
        EXPECT(status == cudaStreamCaptureStatusActive, "capture status %d", int(status));
        CHECK(cudaMemsetAsync(b, 0, n * sizeof(int), s0));
        addValue<<<16, 256, 0, s0>>>(a, n, 3);
        CHECK(cudaMemcpyAsync(b, a, n * sizeof(int), cudaMemcpyDeviceToDevice, s0));
        scale<<<16, 256, 0, s0>>>(b, n, 2);
        CHECK(cudaLaunchHostFunc(s0, hostIncrement, nullptr));
        unsigned long long id = 0;
        size_t dependencies = 0;
        CHECK(cudaStreamGetCaptureInfo(s0, &status, &id, nullptr, nullptr, nullptr, &dependencies));
        EXPECT(status == cudaStreamCaptureStatusActive && id != 0 && dependencies == 1, "capture info %d %llu %zu",
               int(status), id, dependencies);
        cudaGraph_t graph;
        CHECK(cudaStreamEndCapture(s0, &graph));
        CHECK(cudaStreamIsCapturing(s0, &status));
        EXPECT(status == cudaStreamCaptureStatusNone, "still capturing");
        // nothing ran during the capture
        EXPECT(download(a, n)[5] == 0 && g_hostCalls == 0, "captured work ran during the capture");
        EXPECT(nodeCount(graph) == 5 && edgeCount(graph) == 4, "captured chain: %zu nodes, %zu edges",
               nodeCount(graph), edgeCount(graph));
        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        for (int launch = 1; launch <= 5; launch++) {
            CHECK(cudaGraphLaunch(exec, s0));
            CHECK(cudaStreamSynchronize(s0));
            const std::vector<int> ha = download(a, n), hb = download(b, n);
            EXPECT(ha[n - 1] == 3 * launch && hb[n - 1] == 6 * launch, "launch %d: a = %d, b = %d", launch,
                   ha[n - 1], hb[n - 1]);
        }
        EXPECT(g_hostCalls == 5, "host function ran %d times", g_hostCalls.load());
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
    }

    // fork / join: two branches on two streams joined by events
    {
        CHECK(cudaMemset(a, 0, n * sizeof(int)));
        CHECK(cudaMemset(b, 0, n * sizeof(int)));
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t fork, join;
        CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        CHECK(cudaEventRecord(fork, s0));
        CHECK(cudaStreamWaitEvent(s1, fork, 0));
        cudaStreamCaptureStatus status;
        CHECK(cudaStreamIsCapturing(s1, &status));
        EXPECT(status == cudaStreamCaptureStatusActive, "forked stream not capturing");
        addValue<<<16, 256, 0, s0>>>(a, n, 10);
        addValue<<<16, 256, 0, s1>>>(b, n, 20);
        CHECK(cudaEventRecord(join, s1));
        CHECK(cudaStreamWaitEvent(s0, join, 0));
        sum2<<<16, 256, 0, s0>>>(a, b, c, n);
        cudaGraph_t graph;
        CHECK(cudaStreamEndCapture(s0, &graph));
        CHECK(cudaStreamIsCapturing(s1, &status));
        EXPECT(status == cudaStreamCaptureStatusNone, "joined stream still capturing");
        size_t roots = 0;
        CHECK(cudaGraphGetRootNodes(graph, nullptr, &roots));
        EXPECT(nodeCount(graph) == 3 && edgeCount(graph) == 2 && roots == 2, "fork / join: %zu nodes, %zu edges, %zu roots",
               nodeCount(graph), edgeCount(graph), roots);
        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CHECK(cudaGraphLaunch(exec, s1));
        CHECK(cudaGraphLaunch(exec, s1));
        CHECK(cudaStreamSynchronize(s1));
        EXPECT(download(c, n)[7] == 60, "fork / join result %d", download(c, n)[7]);
        CHECK(cudaGraphDebugDotPrint(graph, "/tmp/librecuda_graphs_test.dot", 0));
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaEventDestroy(fork));
        CHECK(cudaEventDestroy(join));
    }

    // events created, recorded, waited for and destroyed during the capture (as PyTorch's NCCL process group does
    // around every collective)
    {
        CHECK(cudaMemset(a, 0, n * sizeof(int)));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        const int rounds = 300;
        for (int i = 0; i < rounds; i++) {
            cudaEvent_t fork, join;
            CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
            CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
            CHECK(cudaEventRecord(fork, s0));
            CHECK(cudaStreamWaitEvent(s1, fork, 0));
            addValue<<<16, 256, 0, s1>>>(a, n, 1);
            CHECK(cudaEventRecord(join, s1));
            CHECK(cudaStreamWaitEvent(s0, join, 0));
            CHECK(cudaEventDestroy(fork));
            CHECK(cudaEventDestroy(join));
        }
        cudaGraph_t graph;
        CHECK(cudaStreamEndCapture(s0, &graph));
        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CHECK(cudaGraphLaunch(exec, s0));
        CHECK(cudaStreamSynchronize(s0));
        EXPECT(download(a, n)[7] == rounds, "events destroyed during the capture: %d", download(a, n)[7]);
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
    }

    // capture errors
    {
        cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(1);
        // a forked stream that isn't joined back
        cudaEvent_t fork;
        CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        CHECK(cudaEventRecord(fork, s0));
        CHECK(cudaStreamWaitEvent(s1, fork, 0));
        addValue<<<1, 32, 0, s1>>>(a, n, 1);
        EXPECT(cudaStreamEndCapture(s0, &graph) == cudaErrorStreamCaptureUnjoined && graph == nullptr,
               "unjoined fork not reported");
        cudaGetLastError();
        CHECK(cudaEventDestroy(fork));
        cudaStreamCaptureStatus status;
        CHECK(cudaStreamIsCapturing(s1, &status));
        EXPECT(status == cudaStreamCaptureStatusNone, "forked stream still capturing after the failed end");

        // synchronizing a capturing stream invalidates the capture
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        addValue<<<1, 32, 0, s0>>>(a, n, 1);
        EXPECT(cudaStreamSynchronize(s0) == cudaErrorStreamCaptureUnsupported, "synchronization during capture");
        cudaGetLastError();
        CHECK(cudaStreamIsCapturing(s0, &status));
        EXPECT(status == cudaStreamCaptureStatusInvalidated, "capture not invalidated (%d)", int(status));
        EXPECT(cudaStreamEndCapture(s0, &graph) == cudaErrorStreamCaptureInvalidated && graph == nullptr,
               "invalidated capture ended");
        cudaGetLastError();

        // cudaMalloc is unsafe during a global capture, allowed for a relaxed thread
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        void *p = nullptr;
        EXPECT(cudaMalloc(&p, 1024) == cudaErrorStreamCaptureUnsupported, "cudaMalloc during global capture");
        cudaGetLastError();
        EXPECT(cudaStreamEndCapture(s0, &graph) == cudaErrorStreamCaptureInvalidated, "capture after cudaMalloc");
        cudaGetLastError();
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal));
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        CHECK(cudaMalloc(&p, 1024));
        CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        EXPECT(mode == cudaStreamCaptureModeRelaxed, "exchanged mode");
        addValue<<<1, 32, 0, s0>>>(a, n, 1);
        CHECK(cudaStreamEndCapture(s0, &graph));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaFree(p));

        // the legacy default stream synchronizes with capturing blocking streams: not allowed
        cudaStream_t blocking;
        CHECK(cudaStreamCreate(&blocking));
        CHECK(cudaStreamBeginCapture(blocking, cudaStreamCaptureModeRelaxed));
        addValue<<<1, 32, 0, blocking>>>(a, n, 1);
        EXPECT(cudaMemsetAsync(b, 0, 16, nullptr) == cudaErrorStreamCaptureImplicit, "legacy stream work");
        cudaGetLastError();
        EXPECT(cudaStreamEndCapture(blocking, &graph) == cudaErrorStreamCaptureInvalidated,
               "capture after legacy stream work");
        cudaGetLastError();
        CHECK(cudaStreamDestroy(blocking));
        // the legacy stream itself can't be captured
        EXPECT(cudaStreamBeginCapture(nullptr, cudaStreamCaptureModeGlobal) != cudaSuccess, "legacy stream captured");
        cudaGetLastError();
    }

    // a graph built node by node: memset -> kernel -> (copy, kernel) -> host, plus an empty node
    {
        g_hostCalls = 0;
        cudaGraph_t graph;
        CHECK(cudaGraphCreate(&graph, 0));
        cudaMemsetParams memset{};
        memset.dst = a;
        memset.value = 0;
        memset.elementSize = 4;
        memset.width = n;
        memset.height = 1;
        cudaGraphNode_t set, add, copy, mul, host, empty;
        CHECK(cudaGraphAddMemsetNode(&set, graph, nullptr, 0, &memset));
        int count = n, value = 7;
        void *add_args[] = {&a, &count, &value};
        cudaKernelNodeParams kernel{};
        kernel.func = reinterpret_cast<void *>(addValue);
        kernel.gridDim = dim3(16);
        kernel.blockDim = dim3(256);
        kernel.kernelParams = add_args;
        CHECK(cudaGraphAddKernelNode(&add, graph, &set, 1, &kernel));
        CHECK(cudaGraphAddMemcpyNode1D(&copy, graph, &add, 1, b, a, n * sizeof(int), cudaMemcpyDeviceToDevice));
        int factor = 3;
        void *mul_args[] = {&a, &count, &factor};
        kernel.func = reinterpret_cast<void *>(scale);
        kernel.kernelParams = mul_args;
        // (after the copy, which reads what it scales)
        CHECK(cudaGraphAddKernelNode(&mul, graph, &copy, 1, &kernel));
        // the arguments were copied: changing them afterwards doesn't matter
        factor = 100;
        cudaHostNodeParams hostParams{hostIncrement, nullptr};
        const cudaGraphNode_t both[] = {copy, mul};
        CHECK(cudaGraphAddHostNode(&host, graph, both, 2, &hostParams));
        CHECK(cudaGraphAddEmptyNode(&empty, graph, nullptr, 0));
        EXPECT(nodeCount(graph) == 6 && edgeCount(graph) == 5, "built graph: %zu nodes, %zu edges", nodeCount(graph),
               edgeCount(graph));
        cudaGraphNodeType type;
        CHECK(cudaGraphNodeGetType(copy, &type));
        EXPECT(type == cudaGraphNodeTypeMemcpy, "node type %d", int(type));
        size_t dependents = 0;
        CHECK(cudaGraphNodeGetDependentNodes(copy, nullptr, nullptr, &dependents));
        EXPECT(dependents == 2, "%zu dependents", dependents);
        cudaKernelNodeParams got{};
        CHECK(cudaGraphKernelNodeGetParams(mul, &got));
        EXPECT(got.func == reinterpret_cast<void *>(scale) && *static_cast<int *>(got.kernelParams[2]) == 3,
               "kernel node parameters");

        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CHECK(cudaGraphLaunch(exec, s0));
        CHECK(cudaStreamSynchronize(s0));
        EXPECT(download(b, n)[1] == 7 && download(a, n)[1] == 21 && g_hostCalls == 1, "built graph run: b %d a %d",
               download(b, n)[1], download(a, n)[1]);

        // the executable graph's kernel node gets a new argument
        int new_value = 2;
        void *new_args[] = {&a, &count, &new_value};
        cudaKernelNodeParams changed = kernel;
        changed.func = reinterpret_cast<void *>(addValue);
        changed.kernelParams = new_args;
        CHECK(cudaGraphExecKernelNodeSetParams(exec, add, &changed));
        CHECK(cudaGraphLaunch(exec, s0));
        CHECK(cudaStreamSynchronize(s0));
        EXPECT(download(a, n)[1] == 6, "after exec node update: %d", download(a, n)[1]);

        // whole graph update with the same topology, then a changed topology
        cudaGraph_t clone;
        CHECK(cudaGraphClone(&clone, graph));
        cudaGraphNode_t nodes[6];
        size_t num = 6;
        CHECK(cudaGraphGetNodes(clone, nodes, &num));
        for (size_t i = 0; i < num; i++) {
            CHECK(cudaGraphNodeGetType(nodes[i], &type));
            cudaKernelNodeParams p{};
            if (type == cudaGraphNodeTypeKernel && (cudaGraphKernelNodeGetParams(nodes[i], &p), p.func) ==
                                                       reinterpret_cast<void *>(scale)) {
                int ten = 10;
                void *args[] = {&a, &count, &ten};
                p.kernelParams = args;
                CHECK(cudaGraphKernelNodeSetParams(nodes[i], &p));
            }
        }
        cudaGraphExecUpdateResultInfo info{};
        CHECK(cudaGraphExecUpdate(exec, clone, &info));
        EXPECT(info.result == cudaGraphExecUpdateSuccess, "update result %d", int(info.result));
        CHECK(cudaGraphLaunch(exec, s0));
        CHECK(cudaStreamSynchronize(s0));
        EXPECT(download(a, n)[1] == 70, "after graph update: %d", download(a, n)[1]);
        cudaGraphNode_t extra;
        CHECK(cudaGraphAddEmptyNode(&extra, clone, nullptr, 0));
        EXPECT(cudaGraphExecUpdate(exec, clone, &info) == cudaErrorGraphExecUpdateFailure &&
               info.result == cudaGraphExecUpdateErrorTopologyChanged, "changed topology accepted");
        cudaGetLastError();
        CHECK(cudaGraphDestroy(clone));

        // as a child graph of another graph
        cudaGraph_t parent;
        CHECK(cudaGraphCreate(&parent, 0));
        cudaGraphNode_t child_node, after;
        CHECK(cudaGraphAddChildGraphNode(&child_node, parent, nullptr, 0, graph));
        int thousand = 1000;
        void *after_args[] = {&a, &count, &thousand};
        kernel.func = reinterpret_cast<void *>(addValue);
        kernel.kernelParams = after_args;
        CHECK(cudaGraphAddKernelNode(&after, parent, &child_node, 1, &kernel));
        cudaGraph_t child_graph;
        CHECK(cudaGraphChildGraphNodeGetGraph(child_node, &child_graph));
        EXPECT(nodeCount(child_graph) == 6, "child graph has %zu nodes", nodeCount(child_graph));
        cudaGraphExec_t parent_exec;
        CHECK(cudaGraphInstantiate(&parent_exec, parent, 0));
        g_hostCalls = 0;
        CHECK(cudaGraphLaunch(parent_exec, s1));
        CHECK(cudaStreamSynchronize(s1));
        EXPECT(download(a, n)[1] == 1021 && g_hostCalls == 1, "child graph run: %d", download(a, n)[1]);

        // graph launches can't be captured
        CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
        EXPECT(cudaGraphLaunch(parent_exec, s1) == cudaErrorStreamCaptureUnsupported, "graph launch captured");
        cudaGetLastError();
        cudaGraph_t captured = nullptr;
        EXPECT(cudaStreamEndCapture(s1, &captured) == cudaErrorStreamCaptureInvalidated, "capture after graph launch");
        cudaGetLastError();

        for (cudaGraphExec_t e: {exec, parent_exec}) {
            CHECK(cudaGraphExecDestroy(e));
        }
        for (cudaGraph_t g: {graph, parent}) {
            CHECK(cudaGraphDestroy(g));
        }
    }

    // event record nodes
    {
        cudaEvent_t done;
        CHECK(cudaEventCreate(&done));
        cudaGraph_t graph;
        CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t record;
        CHECK(cudaGraphAddEventRecordNode(&record, graph, nullptr, 0, done));
        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CHECK(cudaGraphLaunch(exec, s0));
        CHECK(cudaEventSynchronize(done));
        CHECK(cudaEventQuery(done));
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaEventDestroy(done));
    }

    CHECK(cudaStreamDestroy(s0));
    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaDeviceSynchronize());
    PASS("graphs");
    return 0;
}
