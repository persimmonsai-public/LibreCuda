// CUDA graph conditional nodes: IF, IF/ELSE, WHILE and SWITCH bodies selected by conditions that kernels set with
// cudaGraphSetConditional, default values assigned at each launch or kept between launches, a conditional nested in
// a WHILE body, a body graph built by stream capture, node parameters and invalid sizes. Checked against NVIDIA's
// runtime too.
#include "check.h"

__global__ void setCondition(cudaGraphConditionalHandle handle, const int *value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaGraphSetConditional(handle, static_cast<unsigned int>(*value));
    }
}

__global__ void addTo(int *p, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *p += value;
    }
}

// one loop iteration: counts down and keeps the loop going while the counter is positive
__global__ void countDown(cudaGraphConditionalHandle handle, int *counter, int *iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *counter -= 1;
        *iterations += 1;
        cudaGraphSetConditional(handle, *counter > 0 ? 1 : 0);
    }
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

static cudaGraphNode_t addConditional(cudaGraph_t graph, const cudaGraphNode_t *dep, cudaGraphConditionalHandle handle,
                                      cudaGraphConditionalNodeType type, unsigned int size, cudaGraph_t *bodies) {
    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = type;
    p.conditional.size = size;
    cudaGraphNode_t node;
    CHECK(cudaGraphAddNode(&node, graph, dep, nullptr, dep != nullptr ? 1 : 0, &p));
    for (unsigned int i = 0; i < size; i++) {
        bodies[i] = p.conditional.phGraph_out[i];
    }
    return node;
}

static void testIfElse(int *values) {
    // values[0]: the condition input, values[1]: the IF branch counter, values[2]: the ELSE branch counter
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
    cudaGraphNode_t set = addKernel(graph, nullptr, setCondition, handle, (const int *) values);
    cudaGraph_t bodies[2];
    addConditional(graph, &set, handle, cudaGraphCondTypeIf, 2, bodies);
    addKernel(bodies[0], nullptr, addTo, values + 1, 1);
    addKernel(bodies[1], nullptr, addTo, values + 2, 1);
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    const int inputs[] = {1, 0, 5, 0, 0};
    for (int input: inputs) {
        values[0] = input;
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
    }
    EXPECT(values[1] == 2 && values[2] == 3, "IF ran %d times, ELSE %d times", values[1], values[2]);

    // a graph with an IF only
    cudaGraph_t graph2;
    CHECK(cudaGraphCreate(&graph2, 0));
    cudaGraphConditionalHandle handle2;
    CHECK(cudaGraphConditionalHandleCreate(&handle2, graph2, 0, cudaGraphCondAssignDefault));
    cudaGraphNode_t set2 = addKernel(graph2, nullptr, setCondition, handle2, (const int *) values);
    cudaGraph_t body;
    addConditional(graph2, &set2, handle2, cudaGraphCondTypeIf, 1, &body);
    addKernel(body, nullptr, addTo, values + 1, 10);
    cudaGraphExec_t exec2;
    CHECK(cudaGraphInstantiate(&exec2, graph2, 0));
    values[0] = 0;
    CHECK(cudaGraphLaunch(exec2, 0));
    CHECK(cudaDeviceSynchronize());
    values[0] = 3;
    CHECK(cudaGraphLaunch(exec2, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(values[1] == 12, "IF without ELSE: %d", values[1]);
    CHECK(cudaGraphExecDestroy(exec2));
    CHECK(cudaGraphDestroy(graph2));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("graph IF / ELSE");
}

static void testWhile(int *values) {
    // values[3]: the counter, values[4]: iterations, values[5]: inner IF count
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle loop;
    // the loop runs at least once: the condition starts as 1 at each launch
    CHECK(cudaGraphConditionalHandleCreate(&loop, graph, 1, cudaGraphCondAssignDefault));
    cudaGraph_t body;
    addConditional(graph, nullptr, loop, cudaGraphCondTypeWhile, 1, &body);
    cudaGraphNode_t step = addKernel(body, nullptr, countDown, loop, values + 3, values + 4);
    // a conditional nested in the loop body: on iterations where the counter is odd
    cudaGraphConditionalHandle odd;
    CHECK(cudaGraphConditionalHandleCreate(&odd, graph, 0, cudaGraphCondAssignDefault));
    cudaGraphNode_t set = addKernel(body, &step, setCondition, odd, (const int *) (values + 6));
    cudaGraph_t inner;
    addConditional(body, &set, odd, cudaGraphCondTypeIf, 1, &inner);
    addKernel(inner, nullptr, addTo, values + 5, 1);
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    values[3] = 5;
    values[4] = 0;
    values[5] = 0;
    values[6] = 1;
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(values[3] == 0 && values[4] == 5 && values[5] == 5, "WHILE: counter %d, %d iterations, inner %d",
           values[3], values[4], values[5]);
    values[3] = 2;
    values[6] = 0;
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(values[4] == 7 && values[5] == 5, "second WHILE launch: %d iterations, inner %d", values[4], values[5]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("graph WHILE with a nested IF");
}

static void testSwitch(int *values) {
    // values[0]: the selector, values[10..13]: per case counters
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
    cudaGraphNode_t set = addKernel(graph, nullptr, setCondition, handle, (const int *) values);
    cudaGraph_t bodies[4];
    addConditional(graph, &set, handle, cudaGraphCondTypeSwitch, 4, bodies);
    for (int k = 0; k < 4; k++) {
        addKernel(bodies[k], nullptr, addTo, values + 10 + k, k + 1);
    }
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    const int selectors[] = {2, 0, 3, 9, 2};
    for (int selector: selectors) {
        values[0] = selector;
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
    }
    EXPECT(values[10] == 1 && values[11] == 0 && values[12] == 6 && values[13] == 4, "SWITCH counters %d %d %d %d",
           values[10], values[11], values[12], values[13]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("graph SWITCH");
}

static void testDefaultsAndParams(int *values) {
    // without cudaGraphCondAssignDefault the value set by one launch is used by the next
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, 0));
    cudaGraph_t body;
    cudaGraphNode_t cond = addConditional(graph, nullptr, handle, cudaGraphCondTypeIf, 1, &body);
    // the body counts and switches the condition off for later launches
    cudaGraphNode_t count = addKernel(body, nullptr, addTo, values + 20, 1);
    addKernel(body, &count, setCondition, handle, (const int *) (values + 21));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    values[20] = 0;
    values[21] = 0;
    for (int i = 0; i < 3; i++) {
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
    }
    EXPECT(values[20] == 1, "condition kept between launches: body ran %d times", values[20]);

    cudaGraphNodeParams p{};
    CHECK(cudaGraphNodeGetParams(cond, &p));
    EXPECT(p.type == cudaGraphNodeTypeConditional && p.conditional.handle == handle &&
           p.conditional.type == cudaGraphCondTypeIf && p.conditional.size == 1 && p.conditional.phGraph_out[0] == body,
           "conditional node parameters");
    cudaGraphNodeType type;
    CHECK(cudaGraphNodeGetType(cond, &type));
    EXPECT(type == cudaGraphNodeTypeConditional, "node type %d", int(type));

    // invalid sizes
    cudaGraphNodeParams bad{};
    bad.type = cudaGraphNodeTypeConditional;
    bad.conditional.handle = handle;
    bad.conditional.type = cudaGraphCondTypeWhile;
    bad.conditional.size = 2;
    cudaGraphNode_t node;
    EXPECT(cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &bad) == cudaErrorInvalidValue, "WHILE with 2 bodies");
    bad.conditional.type = cudaGraphCondTypeIf;
    bad.conditional.size = 3;
    EXPECT(cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &bad) == cudaErrorInvalidValue, "IF with 3 bodies");
    cudaGetLastError();
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("graph conditional defaults and parameters");
}

static void testCapturedBody(int *values) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));
    cudaGraph_t body;
    addConditional(graph, nullptr, handle, cudaGraphCondTypeIf, 1, &body);
    // the body graph filled by stream capture
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    addTo<<<1, 32, 0, stream>>>(values + 30, 4);
    addTo<<<1, 32, 0, stream>>>(values + 30, 5);
    cudaGraph_t captured;
    CHECK(cudaStreamEndCapture(stream, &captured));
    EXPECT(captured == body, "capture into the body graph");
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    values[30] = 0;
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    EXPECT(values[30] == 9, "captured body: %d", values[30]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    PASS("graph conditional with a captured body");
}

int main() {
    int *values;
    CHECK(cudaMallocManaged(&values, 64 * sizeof(int)));
    memset(values, 0, 64 * sizeof(int));
    testIfElse(values);
    testWhile(values);
    testSwitch(values);
    testDefaultsAndParams(values);
    testCapturedBody(values);
    CHECK(cudaFree(values));
    CHECK(cudaGetLastError());
    return 0;
}
