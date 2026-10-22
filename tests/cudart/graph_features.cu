// CUDA graph features beyond graphs.cu: independent branches running concurrently (more branches than lanes too,
// ordering with work before and after the launch), memory allocation and free nodes (also captured cudaMallocAsync /
// cudaFreeAsync) and their restrictions, cudaGraphAddNode / cudaGraphNodeGetParams / cudaGraphNodeSetParams /
// cudaGraphExecNodeSetParams, per node type setters of executable graphs, event node getters and setters, copies to
// and from symbols, disabled nodes, identifiers, kernel node attributes, user objects and long kernel chains with
// a node changed between launches. Checked against NVIDIA's runtime too.
#include "check.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

__device__ int g_symbol[16];

__global__ void spin(long long cycles, int *flag, int value) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
    if (flag != nullptr) {
        *flag = value;
    }
}

__global__ void addValue(int *p, int n, int value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] += value;
    }
}

// out[i] = sum of in[0..count)[i]
__global__ void sumOf(int *out, const int *const *in, int count, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int total = 0;
        for (int k = 0; k < count; k++) {
            total += in[k][i];
        }
        out[i] = total;
    }
}

static cudaGraphNode_t addKernel(cudaGraph_t graph, const std::vector<cudaGraphNode_t> &deps, void *func, dim3 grid,
                                 dim3 block, void **args) {
    cudaKernelNodeParams params{};
    params.func = func;
    params.gridDim = grid;
    params.blockDim = block;
    params.kernelParams = args;
    cudaGraphNode_t node;
    CHECK(cudaGraphAddKernelNode(&node, graph, deps.data(), deps.size(), &params));
    return node;
}

static double launchMs(cudaGraphExec_t exec, cudaStream_t stream) {
    auto t0 = std::chrono::steady_clock::now();
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void testConcurrentBranches() {
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    // about 30 ms per branch (the clock rate may be an estimate: compare with one branch alone)
    long long cycles = static_cast<long long>(clock_khz) * 30;
    int *flags;
    CHECK(cudaMallocManaged(&flags, 16 * sizeof(int)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // reference: a single branch
    cudaGraph_t single;
    CHECK(cudaGraphCreate(&single, 0));
    int *none = nullptr;
    int zero = 0;
    void *spin_args[] = {&cycles, &none, &zero};
    addKernel(single, {}, (void *) spin, 1, 1, spin_args);
    cudaGraphExec_t single_exec;
    CHECK(cudaGraphInstantiate(&single_exec, single, 0));
    launchMs(single_exec, stream);
    const double one = launchMs(single_exec, stream);

    // 4 independent branches joined by a kernel that checks all of them finished
    const int branches = 4;
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    std::vector<cudaGraphNode_t> ends;
    std::vector<int> values(branches);
    std::vector<void *> args(branches * 3);
    int *flag_ptrs[branches];
    for (int b = 0; b < branches; b++) {
        values[b] = b + 1;
        flag_ptrs[b] = flags + b;
        void *a[] = {&cycles, &flag_ptrs[b], &values[b]};
        ends.push_back(addKernel(graph, {}, (void *) spin, 1, 1, a));
    }
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    for (int b = 0; b < branches; b++) {
        flags[b] = 0;
    }
    launchMs(exec, stream);
    const double four = launchMs(exec, stream);
    for (int b = 0; b < branches; b++) {
        EXPECT(flags[b] == b + 1, "branch %d didn't run", b);
    }
    printf("    one branch %.1f ms, %d independent branches %.1f ms\n", one, branches, four);
    EXPECT(four < one * branches * 0.75, "branches didn't overlap: %.1f ms vs %.1f ms per branch", four, one);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaGraphExecDestroy(single_exec));
    CHECK(cudaGraphDestroy(single));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(flags));
    PASS("graph concurrent branches");
}

// many branches with dependencies between them, ordered after earlier work and before later work on the stream
static void testDataflow() {
    const int n = 1 << 16, branches = 12;
    int *input, *partial[branches], *out, **partial_list;
    CHECK(cudaMalloc(&input, n * sizeof(int)));
    CHECK(cudaMalloc(&out, n * sizeof(int)));
    CHECK(cudaMallocManaged(&partial_list, branches * sizeof(int *)));
    for (int b = 0; b < branches; b++) {
        CHECK(cudaMalloc(&partial[b], n * sizeof(int)));
        partial_list[b] = partial[b];
    }
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    // branch b: partial[b] = input (copy), += b, += 1 ; then out = sum of all partials ; out += 1000
    std::vector<cudaGraphNode_t> tails;
    std::vector<int> values(branches);
    int one = 1, count = branches, size = n, thousand = 1000;
    for (int b = 0; b < branches; b++) {
        cudaGraphNode_t copy;
        CHECK(cudaGraphAddMemcpyNode1D(&copy, graph, nullptr, 0, partial[b], input, n * sizeof(int),
                                       cudaMemcpyDeviceToDevice));
        values[b] = b;
        void *add_b[] = {&partial[b], &size, &values[b]};
        cudaGraphNode_t add = addKernel(graph, {copy}, (void *) addValue, 64, 256, add_b);
        void *add_1[] = {&partial[b], &size, &one};
        tails.push_back(addKernel(graph, {add}, (void *) addValue, 64, 256, add_1));
    }
    void *sum_args[] = {&out, &partial_list, &count, &size};
    cudaGraphNode_t sum = addKernel(graph, tails, (void *) sumOf, 64, 256, sum_args);
    void *final_args[] = {&out, &size, &thousand};
    addKernel(graph, {sum}, (void *) addValue, 64, 256, final_args);
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<int> host(n);
    for (int round = 0; round < 5; round++) {
        // the graph reads input written just before it and its result is read right after it on the stream
        CHECK(cudaMemsetAsync(input, 0, n * sizeof(int), stream));
        void *set_args[] = {&input, &size, &round};
        CHECK(cudaLaunchKernel((void *) addValue, 64, 256, set_args, 0, stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaMemcpyAsync(host.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        // sum over b of (round + b + 1) + 1000
        const int expected = branches * round + branches * (branches - 1) / 2 + branches + 1000;
        for (int i = 0; i < n; i += 997) {
            EXPECT(host[i] == expected, "round %d: out[%d] = %d, expected %d", round, i, host[i], expected);
        }
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    for (int b = 0; b < branches; b++) {
        CHECK(cudaFree(partial[b]));
    }
    CHECK(cudaFree(partial_list));
    CHECK(cudaFree(out));
    CHECK(cudaFree(input));
    PASS("graph dataflow across branches");
}

static void testMemoryNodes() {
    const int n = 4096;
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaMemAllocNodeParams alloc{};
    alloc.poolProps.allocType = cudaMemAllocationTypePinned;
    alloc.poolProps.location = {cudaMemLocationTypeDevice, 0};
    alloc.bytesize = n * sizeof(int);
    cudaGraphNode_t alloc_node;
    CHECK(cudaGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc));
    EXPECT(alloc.dptr != nullptr, "allocation node address");
    int *p = static_cast<int *>(alloc.dptr);
    cudaMemsetParams fill{};
    fill.dst = p;
    fill.value = 0;
    fill.elementSize = 4;
    fill.width = n;
    fill.height = 1;
    cudaGraphNode_t fill_node, add_node, copy_node, free_node;
    CHECK(cudaGraphAddMemsetNode(&fill_node, graph, &alloc_node, 1, &fill));
    int size = n, seven = 7;
    void *add_args[] = {&p, &size, &seven};
    add_node = addKernel(graph, {fill_node}, (void *) addValue, 16, 256, add_args);
    int *result;
    CHECK(cudaMallocHost(&result, n * sizeof(int)));
    CHECK(cudaGraphAddMemcpyNode1D(&copy_node, graph, &add_node, 1, result, p, n * sizeof(int),
                                   cudaMemcpyDeviceToHost));
    CHECK(cudaGraphAddMemFreeNode(&free_node, graph, &copy_node, 1, p));

    cudaMemAllocNodeParams got{};
    CHECK(cudaGraphMemAllocNodeGetParams(alloc_node, &got));
    EXPECT(got.dptr == p && got.bytesize == alloc.bytesize, "allocation node parameters");
    void *freed = nullptr;
    CHECK(cudaGraphMemFreeNodeGetParams(free_node, &freed));
    EXPECT(freed == p, "free node parameters");

    // restrictions (like CUDA)
    int *plain;
    CHECK(cudaMalloc(&plain, 64));
    cudaGraphNode_t bad;
    EXPECT(cudaGraphAddMemFreeNode(&bad, graph, nullptr, 0, plain) == cudaErrorInvalidValue, "free of other memory");
    cudaGraph_t clone, parent;
    EXPECT(cudaGraphClone(&clone, graph) == cudaErrorNotSupported, "clone of a graph with memory nodes");
    CHECK(cudaGraphCreate(&parent, 0));
    EXPECT(cudaGraphAddChildGraphNode(&bad, parent, nullptr, 0, graph) == cudaErrorNotSupported,
           "child graph with memory nodes");
    cudaGetLastError();

    cudaGraphExec_t exec, second;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    EXPECT(cudaGraphInstantiate(&second, graph, 0) == cudaErrorNotSupported, "second instantiation");
    cudaGetLastError();
    for (int round = 0; round < 3; round++) {
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
        EXPECT(result[0] == 7 && result[n - 1] == 7, "round %d: %d %d", round, result[0], result[n - 1]);
    }
    // (NVIDIA's runtime commits the memory at the first launch)
    unsigned long long used = 0;
    CHECK(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used));
    EXPECT(used >= n * sizeof(int), "graph memory in use: %llu", used);

    // captured cudaMallocAsync / cudaFreeAsync
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t captured;
    int *q = nullptr;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMallocAsync(reinterpret_cast<void **>(&q), n * sizeof(int), stream));
    CHECK(cudaMemsetAsync(q, 0, n * sizeof(int), stream));
    void *add_q[] = {&q, &size, &seven};
    CHECK(cudaLaunchKernel((void *) addValue, 16, 256, add_q, 0, stream));
    CHECK(cudaMemcpyAsync(result, q, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaFreeAsync(q, stream));
    CHECK(cudaStreamEndCapture(stream, &captured));
    size_t count = 0;
    CHECK(cudaGraphGetNodes(captured, nullptr, &count));
    EXPECT(count == 5, "captured nodes: %zu", count);
    cudaGraphExec_t captured_exec;
    CHECK(cudaGraphInstantiate(&captured_exec, captured, 0));
    result[0] = 0;
    CHECK(cudaGraphLaunch(captured_exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(result[0] == 7, "captured allocation: %d", result[0]);
    // freeing other memory in a capture fails
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    EXPECT(cudaFreeAsync(plain, stream) == cudaErrorInvalidValue, "captured free of other memory");
    cudaGraph_t ignored;
    cudaStreamEndCapture(stream, &ignored);
    if (ignored != nullptr) {
        cudaGraphDestroy(ignored);
    }
    cudaGetLastError();

    CHECK(cudaGraphExecDestroy(captured_exec));
    CHECK(cudaGraphDestroy(captured));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(parent));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaDeviceGraphMemTrim(0));
    CHECK(cudaFree(plain));
    CHECK(cudaFreeHost(result));
    PASS("graph memory nodes");
}

static int g_hostValue = 0;

static void CUDART_CB setHostValue(void *value) {
    g_hostValue = *static_cast<int *>(value);
}

static void testGenericNodes() {
    const int n = 1024;
    int *d, *h;
    CHECK(cudaMalloc(&d, n * sizeof(int)));
    CHECK(cudaMallocHost(&h, n * sizeof(int)));
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));

    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeMemset;
    p.memset.dst = d;
    p.memset.value = 3;
    p.memset.elementSize = 4;
    p.memset.width = n;
    p.memset.height = 1;
    cudaGraphNode_t memset_node, kernel_node, copy_node, host_node, empty_node;
    CHECK(cudaGraphAddNode(&memset_node, graph, nullptr, nullptr, 0, &p));

    int size = n, value = 10;
    void *args[] = {&d, &size, &value};
    p = {};
    p.type = cudaGraphNodeTypeKernel;
    p.kernel.func = (void *) addValue;
    p.kernel.gridDim = 4;
    p.kernel.blockDim = 256;
    p.kernel.kernelParams = args;
    CHECK(cudaGraphAddNode(&kernel_node, graph, &memset_node, nullptr, 1, &p));

    p = {};
    p.type = cudaGraphNodeTypeMemcpy;
    p.memcpy.copyParams.srcPtr = make_cudaPitchedPtr(d, n * sizeof(int), n * sizeof(int), 1);
    p.memcpy.copyParams.dstPtr = make_cudaPitchedPtr(h, n * sizeof(int), n * sizeof(int), 1);
    p.memcpy.copyParams.extent = make_cudaExtent(n * sizeof(int), 1, 1);
    p.memcpy.copyParams.kind = cudaMemcpyDeviceToHost;
    CHECK(cudaGraphAddNode(&copy_node, graph, &kernel_node, nullptr, 1, &p));

    static int host_arg = 5;
    p = {};
    p.type = cudaGraphNodeTypeHost;
    p.host.fn = setHostValue;
    p.host.userData = &host_arg;
    CHECK(cudaGraphAddNode(&host_node, graph, &copy_node, nullptr, 1, &p));
    p = {};
    p.type = cudaGraphNodeTypeEmpty;
    CHECK(cudaGraphAddNode(&empty_node, graph, &host_node, nullptr, 1, &p));

    // a child graph moved into the parent
    cudaGraph_t child;
    CHECK(cudaGraphCreate(&child, 0));
    int child_value = 100;
    void *child_args[] = {&d, &size, &child_value};
    addKernel(child, {}, (void *) addValue, 4, 256, child_args);
    p = {};
    p.type = cudaGraphNodeTypeGraph;
    p.graph.graph = child;
    p.graph.ownership = cudaGraphChildGraphOwnershipMove;
    cudaGraphNode_t child_node;
    CHECK(cudaGraphAddNode(&child_node, graph, &empty_node, nullptr, 1, &p));

    // parameters read back
    cudaGraphNodeParams got{};
    CHECK(cudaGraphNodeGetParams(kernel_node, &got));
    EXPECT(got.type == cudaGraphNodeTypeKernel && got.kernel.func == (void *) addValue && got.kernel.gridDim.x == 4 &&
           got.kernel.blockDim.x == 256 && *static_cast<int *>(got.kernel.kernelParams[2]) == 10,
           "kernel node parameters");
    CHECK(cudaGraphNodeGetParams(memset_node, &got));
    EXPECT(got.type == cudaGraphNodeTypeMemset && got.memset.dst == d && got.memset.value == 3 &&
           got.memset.width == n, "memset node parameters");
    CHECK(cudaGraphNodeGetParams(host_node, &got));
    EXPECT(got.host.fn == setHostValue && got.host.userData == &host_arg, "host node parameters");
    CHECK(cudaGraphNodeGetParams(child_node, &got));
    EXPECT(got.type == cudaGraphNodeTypeGraph && got.graph.graph == child, "moved child graph");

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(h[0] == 13 && h[n - 1] == 13 && g_hostValue == 5, "first launch: %d %d", h[0], g_hostValue);

    // generic setters on the graph (takes effect in new instantiations) and the executable graph
    value = 20;
    p = {};
    p.type = cudaGraphNodeTypeKernel;
    p.kernel.func = (void *) addValue;
    p.kernel.gridDim = 4;
    p.kernel.blockDim = 256;
    p.kernel.kernelParams = args;
    CHECK(cudaGraphExecNodeSetParams(exec, kernel_node, &p));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(h[0] == 23, "after cudaGraphExecNodeSetParams: %d", h[0]);
    value = 30;
    CHECK(cudaGraphNodeSetParams(kernel_node, &p));
    CHECK(cudaGraphNodeGetParams(kernel_node, &got));
    EXPECT(*static_cast<int *>(got.kernel.kernelParams[2]) == 30, "cudaGraphNodeSetParams");
    p.type = cudaGraphNodeTypeMemset;
    EXPECT(cudaGraphNodeSetParams(kernel_node, &p) == cudaErrorInvalidValue, "changing the type of a node");
    cudaGetLastError();

    // per type setters of the executable graph
    cudaMemsetParams memset_params{d, 0, 4, 4, static_cast<size_t>(n), 1};
    CHECK(cudaGraphExecMemsetNodeSetParams(exec, memset_node, &memset_params));
    static int host_arg2 = 9;
    cudaHostNodeParams host_params{setHostValue, &host_arg2};
    CHECK(cudaGraphExecHostNodeSetParams(exec, host_node, &host_params));
    int *h2;
    CHECK(cudaMallocHost(&h2, n * sizeof(int)));
    CHECK(cudaGraphExecMemcpyNodeSetParams1D(exec, copy_node, h2, d, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaGraph_t child2;
    CHECK(cudaGraphCreate(&child2, 0));
    int child_value2 = 1000;
    void *child_args2[] = {&d, &size, &child_value2};
    addKernel(child2, {}, (void *) addValue, 4, 256, child_args2);
    CHECK(cudaGraphExecChildGraphNodeSetParams(exec, child_node, child2));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    // memset 4, + 20 (kernel), copied to h2, then the child adds 1000 in d
    EXPECT(h2[0] == 24 && g_hostValue == 9, "per type setters: %d %d", h2[0], g_hostValue);
    CHECK(cudaMemcpy(h, d, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT(h[0] == 1024, "child graph update: %d", h[0]);

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(child2));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaFreeHost(h2));
    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
    PASS("graph generic node parameters");
}

static void testEventsSymbolsEnable() {
    cudaEvent_t e1, e2;
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventCreate(&e2));
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t record, wait;
    CHECK(cudaGraphAddEventRecordNode(&record, graph, nullptr, 0, e1));
    CHECK(cudaGraphAddEventWaitNode(&wait, graph, &record, 1, e1));
    cudaEvent_t got;
    CHECK(cudaGraphEventRecordNodeGetEvent(record, &got));
    EXPECT(got == e1, "record node event");
    CHECK(cudaGraphEventRecordNodeSetEvent(record, e2));
    CHECK(cudaGraphEventRecordNodeGetEvent(record, &got));
    EXPECT(got == e2, "record node event after set");
    CHECK(cudaGraphEventWaitNodeSetEvent(wait, e2));
    CHECK(cudaGraphEventWaitNodeGetEvent(wait, &got));
    EXPECT(got == e2, "wait node event after set");

    // symbol copies: host -> symbol -> device -> host
    int src[16];
    for (int i = 0; i < 16; i++) {
        src[i] = 50 + i;
    }
    int *d, *pinned_src, *pinned_dst;
    CHECK(cudaMalloc(&d, sizeof(src)));
    CHECK(cudaMallocHost(&pinned_src, sizeof(src)));
    CHECK(cudaMallocHost(&pinned_dst, sizeof(src)));
    memcpy(pinned_src, src, sizeof(src));
    cudaGraphNode_t to_symbol, from_symbol, copy_back, add;
    CHECK(cudaGraphAddMemcpyNodeToSymbol(&to_symbol, graph, &wait, 1, g_symbol, pinned_src, 8 * sizeof(int),
                                         8 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaGraphAddMemcpyNodeFromSymbol(&from_symbol, graph, &to_symbol, 1, d, g_symbol, 8 * sizeof(int),
                                           8 * sizeof(int), cudaMemcpyDeviceToDevice));
    int size = 8, one = 1;
    void *add_args[] = {&d, &size, &one};
    add = addKernel(graph, {from_symbol}, (void *) addValue, 1, 32, add_args);
    CHECK(cudaGraphAddMemcpyNode1D(&copy_back, graph, &add, 1, pinned_dst, d, 8 * sizeof(int),
                                   cudaMemcpyDeviceToHost));
    cudaGraphNode_t bad;
    EXPECT(cudaGraphAddMemcpyNodeToSymbol(&bad, graph, nullptr, 0, g_symbol, pinned_src, 16 * sizeof(int),
                                          8 * sizeof(int), cudaMemcpyHostToDevice) == cudaErrorInvalidValue,
           "copy beyond the symbol");
    cudaGetLastError();

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++) {
        EXPECT(pinned_dst[i] == src[i] + 1, "symbol round trip [%d] = %d", i, pinned_dst[i]);
    }
    // a disabled kernel node doesn't run
    unsigned int enabled = 7;
    CHECK(cudaGraphNodeGetEnabled(exec, add, &enabled));
    EXPECT(enabled == 1, "enabled by default");
    CHECK(cudaGraphNodeSetEnabled(exec, add, 0));
    CHECK(cudaGraphNodeGetEnabled(exec, add, &enabled));
    EXPECT(enabled == 0, "disabled");
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(pinned_dst[0] == src[0], "disabled kernel ran: %d", pinned_dst[0]);
    CHECK(cudaGraphNodeSetEnabled(exec, add, 1));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(pinned_dst[0] == src[0] + 1, "re-enabled kernel: %d", pinned_dst[0]);
    EXPECT(cudaGraphNodeSetEnabled(exec, record, 0) == cudaErrorInvalidValue, "disabling an event node");
    cudaGetLastError();
    // exec event setters and symbol setters
    CHECK(cudaGraphExecEventRecordNodeSetEvent(exec, record, e1));
    CHECK(cudaGraphExecEventWaitNodeSetEvent(exec, wait, e1));
    for (int i = 0; i < 16; i++) {
        pinned_src[i] = 200 + i;
    }
    CHECK(cudaGraphExecMemcpyNodeSetParamsToSymbol(exec, to_symbol, g_symbol, pinned_src, 8 * sizeof(int), 0,
                                                   cudaMemcpyHostToDevice));
    CHECK(cudaGraphExecMemcpyNodeSetParamsFromSymbol(exec, from_symbol, d, g_symbol, 8 * sizeof(int), 0,
                                                     cudaMemcpyDeviceToDevice));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(pinned_dst[0] == 201 && pinned_dst[7] == 208, "exec symbol setters: %d %d", pinned_dst[0], pinned_dst[7]);
    CHECK(cudaGraphMemcpyNodeSetParamsToSymbol(to_symbol, g_symbol, pinned_src, 4, 0, cudaMemcpyHostToDevice));
    CHECK(cudaGraphMemcpyNodeSetParamsFromSymbol(from_symbol, d, g_symbol, 4, 0, cudaMemcpyDeviceToDevice));

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaFreeHost(pinned_dst));
    CHECK(cudaFreeHost(pinned_src));
    CHECK(cudaFree(d));
    CHECK(cudaEventDestroy(e2));
    CHECK(cudaEventDestroy(e1));
    PASS("graph event nodes, symbol copies, disabled nodes");
}

__global__ void step(unsigned int *x, int n, unsigned int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = x[i] * 3 + k;
    }
}

// long chains of kernels (replayed from prebuilt commands): the result depends on the order of the steps
static void testKernelChains() {
    const int n = 4096, chains = 3, length = 40;
    unsigned int *x;
    CHECK(cudaMalloc(&x, chains * n * sizeof(unsigned int)));
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    std::vector<cudaGraphNode_t> middle(chains);
    std::vector<unsigned int *> bases(chains);
    std::vector<unsigned int> ks(chains * length);
    int size = n;
    for (int c = 0; c < chains; c++) {
        bases[c] = x + c * n;
        cudaGraphNode_t prev = nullptr;
        for (int i = 0; i < length; i++) {
            ks[c * length + i] = c * 100 + i;
            void *args[] = {&bases[c], &size, &ks[c * length + i]};
            std::vector<cudaGraphNode_t> deps;
            if (prev != nullptr) {
                deps.push_back(prev);
            }
            prev = addKernel(graph, deps, (void *) step, 8, 256, args);
            if (i == length / 2) {
                middle[c] = prev;
            }
        }
    }
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    auto expected = [&](int c, unsigned int start, int middle_k) {
        unsigned int v = start;
        for (int i = 0; i < length; i++) {
            v = v * 3 + (i == length / 2 && middle_k >= 0 ? unsigned(middle_k) : unsigned(c * 100 + i));
        }
        return v;
    };
    std::vector<unsigned int> host(chains * n);
    unsigned int start[chains];
    for (int c = 0; c < chains; c++) {
        start[c] = 1;
    }
    CHECK(cudaMemset(x, 0, chains * n * sizeof(unsigned int)));
    std::vector<unsigned int> ones(chains * n, 1);
    CHECK(cudaMemcpy(x, ones.data(), ones.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    for (int round = 0; round < 20; round++) {
        if (round == 10) {
            // a changed middle node: the chain's prebuilt commands are rebuilt
            unsigned int k = 7777;
            void *args[] = {&bases[1], &size, &k};
            cudaKernelNodeParams p{};
            p.func = (void *) step;
            p.gridDim = 8;
            p.blockDim = 256;
            p.kernelParams = args;
            CHECK(cudaGraphExecKernelNodeSetParams(exec, middle[1], &p));
        }
        CHECK(cudaGraphLaunch(exec, 0));
        for (int c = 0; c < chains; c++) {
            start[c] = expected(c, start[c], c == 1 && round >= 10 ? 7777 : -1);
        }
    }
    CHECK(cudaMemcpy(host.data(), x, host.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int c = 0; c < chains; c++) {
        for (int i = 0; i < n; i += 511) {
            EXPECT(host[c * n + i] == start[c], "chain %d [%d] = %u, expected %u", c, i, host[c * n + i], start[c]);
        }
    }
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaFree(x));
    PASS("graph kernel chains");
}

static std::atomic<int> g_destroyed{0};

static void CUDART_CB destroyObject(void *) {
    g_destroyed++;
}

static void testIdsAttributesUserObjects() {
    cudaGraph_t graph, other;
    CHECK(cudaGraphCreate(&graph, 0));
    CHECK(cudaGraphCreate(&other, 0));
    unsigned int id1 = 0, id2 = 0;
    CHECK(cudaGraphGetId(graph, &id1));
    CHECK(cudaGraphGetId(other, &id2));
    EXPECT(id1 != id2, "graph ids");
    int *d;
    CHECK(cudaMalloc(&d, 256));
    int size = 64, one = 1;
    void *args[] = {&d, &size, &one};
    cudaGraphNode_t a = addKernel(graph, {}, (void *) addValue, 1, 64, args);
    cudaGraphNode_t b = addKernel(graph, {a}, (void *) addValue, 1, 64, args);
    unsigned int local_a = 9, local_b = 9;
    CHECK(cudaGraphNodeGetLocalId(a, &local_a));
    CHECK(cudaGraphNodeGetLocalId(b, &local_b));
    EXPECT(local_a == 0 && local_b == 1, "local ids %u %u", local_a, local_b);
    unsigned long long tools_a = 0, tools_b = 0;
    CHECK(cudaGraphNodeGetToolsId(a, &tools_a));
    CHECK(cudaGraphNodeGetToolsId(b, &tools_b));
    EXPECT(tools_a != tools_b, "tools ids");
    cudaGraph_t containing;
    CHECK(cudaGraphNodeGetContainingGraph(b, &containing));
    EXPECT(containing == graph, "containing graph");

    cudaGraph_t clone;
    CHECK(cudaGraphClone(&clone, graph));
    cudaGraphNode_t in_clone;
    CHECK(cudaGraphNodeFindInClone(&in_clone, b, clone));
    CHECK(cudaGraphNodeGetContainingGraph(in_clone, &containing));
    EXPECT(containing == clone && in_clone != b, "node in clone");
    EXPECT(cudaGraphNodeFindInClone(&in_clone, b, other) == cudaErrorInvalidValue, "node not in a clone");
    cudaGetLastError();

    // kernel node attributes
    cudaKernelNodeAttrValue value{};
    value.cooperative = 1;
    CHECK(cudaGraphKernelNodeSetAttribute(a, cudaKernelNodeAttributeCooperative, &value));
    CHECK(cudaGraphKernelNodeCopyAttributes(b, a));
    value = {};
    CHECK(cudaGraphKernelNodeGetAttribute(b, cudaKernelNodeAttributeCooperative, &value));
    EXPECT(value.cooperative == 1, "copied cooperative %d", value.cooperative);

    // executable graph flags and ids
    cudaGraphInstantiateParams params{};
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiateWithParams(&exec, graph, &params));
    EXPECT(params.result_out == cudaGraphInstantiateSuccess, "instantiate result %d", int(params.result_out));
    unsigned long long flags = 9;
    CHECK(cudaGraphExecGetFlags(exec, &flags));
    EXPECT(flags == 0, "exec flags %llu", flags);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));
    CHECK(cudaGraphExecGetFlags(exec, &flags));
    EXPECT(flags == cudaGraphInstantiateFlagAutoFreeOnLaunch, "exec flags %llu", flags);
    unsigned int exec_id = 0;
    CHECK(cudaGraphExecGetId(exec, &exec_id));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    int host[64];
    CHECK(cudaMemcpy(host, d, sizeof(host), cudaMemcpyDeviceToHost));
    CHECK(cudaGraphExecDestroy(exec));

    // user objects: references of the caller, a graph and an executable graph
    cudaUserObject_t object;
    CHECK(cudaUserObjectCreate(&object, nullptr, destroyObject, 1, cudaUserObjectNoDestructorSync));
    CHECK(cudaUserObjectRetain(object, 2));
    CHECK(cudaUserObjectRelease(object, 2));
    CHECK(cudaGraphRetainUserObject(other, object, 1, cudaGraphUserObjectMove));
    CHECK(cudaGraphRetainUserObject(other, object, 2, 0));
    CHECK(cudaGraphReleaseUserObject(other, object, 2));
    cudaGraphExec_t other_exec;
    CHECK(cudaGraphInstantiate(&other_exec, other, 0));
    CHECK(cudaGraphDestroy(other));
    CHECK(cudaDeviceSynchronize());
    EXPECT(g_destroyed == 0, "destroyed while an executable graph holds a reference");
    CHECK(cudaGraphExecDestroy(other_exec));
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 100 && g_destroyed == 0; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT(g_destroyed == 1, "destructor calls: %d", g_destroyed.load());

    CHECK(cudaGraphDestroy(clone));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaFree(d));
    PASS("graph ids, kernel attributes, user objects");
}

int main() {
    testConcurrentBranches();
    testDataflow();
    testKernelChains();
    testMemoryNodes();
    testGenericNodes();
    testEventsSymbolsEnable();
    testIdsAttributesUserObjects();
    CHECK(cudaGetLastError());
    return 0;
}
