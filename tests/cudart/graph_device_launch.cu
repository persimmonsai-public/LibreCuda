// NVCC_FLAGS: -arch=$ARCH -rdc=true
// Graphs launched from device code (cudaGraphLaunch in a kernel of a graph): fire and forget launches run right after
// the launching kernel, tail launches after the whole launching graph, a graph relaunching itself as its tail
// (cudaGetCurrentGraphExec) until a counter runs out, and instantiation for device launch. Edge data: programmatic
// edges between kernel nodes, invalid edges, queries with and without edge data (lossy), programmatic edges recorded
// by stream capture of programmatic dependent launches. Checked against NVIDIA's runtime too.
#include "check.h"

__global__ void addTo(int *p, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *p += value;
    }
}

// records p[0] and whether the kernel runs as part of a graph
__global__ void record(int *p, int slot) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[slot] = p[0];
        p[slot + 1] = cudaGetCurrentGraphExec() != nullptr;
    }
}

__global__ void launchGraph(cudaGraphExec_t graph, int tail, int *status) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *status = cudaGraphLaunch(graph, tail ? cudaStreamGraphTailLaunch : cudaStreamGraphFireAndForget);
    }
}

// counts down and relaunches its own graph as a tail launch until the counter is 0
__global__ void loop(int *counter, int *iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *iterations += 1;
        if (--*counter > 0) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        }
    }
}

__global__ void currentGraph(unsigned long long *out) {
    *out = reinterpret_cast<unsigned long long>(cudaGetCurrentGraphExec());
}

template <typename... Args>
static cudaGraphNode_t addKernel(cudaGraph_t graph, const cudaGraphNode_t *dep, void (*func)(Args...), Args... args) {
    void *params[] = {&args...};
    cudaKernelNodeParams p{};
    p.func = reinterpret_cast<void *>(func);
    p.gridDim = 1;
    p.blockDim = 32;
    p.kernelParams = params;
    cudaGraphNode_t node;
    CHECK(cudaGraphAddKernelNode(&node, graph, dep, dep != nullptr ? 1 : 0, &p));
    return node;
}

static void testDeviceLaunch(int *values) {
    // the device launched graph records p[0] in p[1] (and whether it runs in a graph in p[2])
    cudaGraph_t child;
    CHECK(cudaGraphCreate(&child, 0));
    addKernel(child, nullptr, record, values, 1);
    cudaGraphExec_t child_exec;
    CHECK(cudaGraphInstantiateWithFlags(&child_exec, child, cudaGraphInstantiateFlagDeviceLaunch));
    CHECK(cudaGraphUpload(child_exec, 0));
    unsigned long long flags = 0;
    CHECK(cudaGraphExecGetFlags(child_exec, &flags));
    EXPECT(flags == cudaGraphInstantiateFlagDeviceLaunch, "device launch flags %llu", flags);

    for (int tail = 0; tail < 2; tail++) {
        // launcher, then a kernel adding 100: a tail launch sees it, a fire and forget launch doesn't
        cudaGraph_t parent;
        CHECK(cudaGraphCreate(&parent, 0));
        cudaGraphNode_t launcher = addKernel(parent, nullptr, launchGraph, child_exec, tail, values + 10);
        addKernel(parent, &launcher, addTo, values, 100);
        cudaGraphExec_t exec;
        CHECK(cudaGraphInstantiate(&exec, parent, 0));
        values[0] = 0;
        values[1] = -1;
        values[2] = 0;
        values[10] = -1;
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
        EXPECT(values[10] == 0, "device side cudaGraphLaunch returned %d", values[10]);
        EXPECT(values[1] == (tail ? 100 : 0), "%s launch saw %d", tail ? "tail" : "fire and forget", values[1]);
        EXPECT(values[2] == 1, "cudaGetCurrentGraphExec in a device launched graph");
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(parent));
    }

    // a device graph also runs when the host launches it
    values[0] = 5;
    values[1] = -1;
    CHECK(cudaGraphLaunch(child_exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(values[1] == 5, "host launch of a device graph: %d", values[1]);

    // outside graphs there is no current graph
    unsigned long long *current;
    CHECK(cudaMallocManaged(&current, sizeof(*current)));
    *current = 7;
    currentGraph<<<1, 1>>>(current);
    CHECK(cudaDeviceSynchronize());
    EXPECT(*current == 0, "cudaGetCurrentGraphExec outside graphs: %llx", *current);
    CHECK(cudaFree(current));

    // graphs for device launch contain kernels, copies and fills only
    cudaGraph_t bad;
    CHECK(cudaGraphCreate(&bad, 0));
    cudaHostNodeParams host{[](void *) {}, nullptr};
    cudaGraphNode_t host_node;
    CHECK(cudaGraphAddHostNode(&host_node, bad, nullptr, 0, &host));
    cudaGraphExec_t bad_exec;
    EXPECT(cudaGraphInstantiateWithFlags(&bad_exec, bad, cudaGraphInstantiateFlagDeviceLaunch) ==
           cudaErrorInvalidValue, "device launch graph with a host node");
    cudaGetLastError();
    CHECK(cudaGraphDestroy(bad));
    CHECK(cudaGraphExecDestroy(child_exec));
    CHECK(cudaGraphDestroy(child));
    PASS("graphs launched from the device");
}

static void testSelfRelaunch(int *values) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    addKernel(graph, nullptr, loop, values + 20, values + 21);
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagDeviceLaunch));
    CHECK(cudaGraphUpload(exec, 0));
    values[20] = 6;
    values[21] = 0;
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(values[20] == 0 && values[21] == 6, "self relaunch: counter %d, %d iterations", values[20], values[21]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("graph relaunching itself as its tail");
}

static void testEdgeData(int *values) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t a = addKernel(graph, nullptr, addTo, values + 30, 1);
    cudaGraphNode_t b = addKernel(graph, nullptr, addTo, values + 30, 2);
    cudaHostNodeParams host{[](void *) {}, nullptr};
    cudaGraphNode_t h;
    CHECK(cudaGraphAddHostNode(&h, graph, nullptr, 0, &host));

    cudaGraphEdgeData programmatic{};
    programmatic.from_port = cudaGraphKernelNodePortProgrammatic;
    programmatic.type = cudaGraphDependencyTypeProgrammatic;
    CHECK(cudaGraphAddDependencies(graph, &a, &b, &programmatic, 1));
    // invalid: a destination port, a programmatic edge to a host node, a non-default port on a default edge
    cudaGraphEdgeData to_port = programmatic;
    to_port.to_port = 1;
    EXPECT(cudaGraphAddDependencies(graph, &b, &h, &to_port, 1) == cudaErrorInvalidValue, "to_port");
    EXPECT(cudaGraphAddDependencies(graph, &b, &h, &programmatic, 1) == cudaErrorInvalidValue,
           "programmatic edge to a host node");
    cudaGraphEdgeData default_port{};
    default_port.from_port = cudaGraphKernelNodePortProgrammatic;
    EXPECT(cudaGraphAddDependencies(graph, &b, &h, &default_port, 1) == cudaErrorInvalidValue,
           "programmatic port on a default edge");
    cudaGetLastError();
    CHECK(cudaGraphAddDependencies(graph, &b, &h, nullptr, 1));

    cudaGraphNode_t from[4], to[4];
    cudaGraphEdgeData data[4];
    size_t count = 4;
    CHECK(cudaGraphGetEdges(graph, from, to, data, &count));
    EXPECT(count == 2, "edges: %zu", count);
    for (size_t i = 0; i < count; i++) {
        const bool is_programmatic = from[i] == a && to[i] == b;
        EXPECT(data[i].type == (is_programmatic ? 1 : 0) && data[i].from_port == (is_programmatic ? 1 : 0),
               "edge %zu data", i);
    }
    count = 4;
    EXPECT(cudaGraphGetEdges(graph, from, to, nullptr, &count) == cudaErrorLossyQuery, "lossy edge query");
    cudaGetLastError();
    cudaGraphNode_t deps[2];
    cudaGraphEdgeData dep_data[2];
    count = 2;
    CHECK(cudaGraphNodeGetDependencies(b, deps, dep_data, &count));
    EXPECT(count == 1 && deps[0] == a && dep_data[0].type == cudaGraphDependencyTypeProgrammatic,
           "dependencies of b");
    count = 2;
    CHECK(cudaGraphNodeGetDependentNodes(a, deps, dep_data, &count));
    EXPECT(count == 1 && deps[0] == b && dep_data[0].from_port == cudaGraphKernelNodePortProgrammatic,
           "dependents of a");

    // programmatic edges run like full dependencies
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    values[30] = 0;
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(values[30] == 3, "graph with a programmatic edge: %d", values[30]);
    // a clone keeps the edge data; removing needs matching data
    cudaGraph_t clone;
    CHECK(cudaGraphClone(&clone, graph));
    count = 4;
    CHECK(cudaGraphGetEdges(clone, from, to, data, &count));
    int programmatic_edges = 0;
    for (size_t i = 0; i < count; i++) {
        programmatic_edges += data[i].type == cudaGraphDependencyTypeProgrammatic;
    }
    EXPECT(programmatic_edges == 1, "programmatic edges in the clone: %d", programmatic_edges);
    CHECK(cudaGraphRemoveDependencies(graph, &a, &b, &programmatic, 1));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(clone));
    CHECK(cudaGraphDestroy(graph));

    // stream capture of a programmatic dependent launch
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaGraph_t captured;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    addTo<<<1, 32, 0, stream>>>(values + 31, 1);
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = 1;
    config.blockDim = 32;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    CHECK(cudaLaunchKernelEx(&config, addTo, values + 31, 10));
    CHECK(cudaStreamEndCapture(stream, &captured));
    count = 4;
    CHECK(cudaGraphGetEdges(captured, from, to, data, &count));
    EXPECT(count == 1 && data[0].type == cudaGraphDependencyTypeProgrammatic &&
           data[0].from_port == cudaGraphKernelNodePortProgrammatic, "captured programmatic edge");
    cudaGraphExec_t captured_exec;
    CHECK(cudaGraphInstantiate(&captured_exec, captured, 0));
    values[31] = 0;
    CHECK(cudaGraphLaunch(captured_exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(values[31] == 11, "captured graph: %d", values[31]);
    // outside capture, a programmatic dependent launch runs in stream order
    CHECK(cudaLaunchKernelEx(&config, addTo, values + 31, 100));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(values[31] == 111, "programmatic dependent launch: %d", values[31]);
    CHECK(cudaGraphExecDestroy(captured_exec));
    CHECK(cudaGraphDestroy(captured));
    CHECK(cudaStreamDestroy(stream));
    PASS("graph edge data");
}

int main() {
    int *values;
    CHECK(cudaMallocManaged(&values, 64 * sizeof(int)));
    memset(values, 0, 64 * sizeof(int));
    testDeviceLaunch(values);
    testSelfRelaunch(values);
    testEdgeData(values);
    CHECK(cudaFree(values));
    CHECK(cudaGetLastError());
    return 0;
}
