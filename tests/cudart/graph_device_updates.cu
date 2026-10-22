// NVCC_FLAGS: -arch=$ARCH -rdc=true
// Device side updates of graph kernel nodes: device-updatable nodes (cudaLaunchAttributeDeviceUpdatableKernelNode,
// set on a node or by stream capture), cudaGraphKernelNodeSetParam / SetGridDim / SetEnabled / UpdatesApply from
// kernels outside the graph and from a kernel of the graph itself, taking effect at the next launch, parameters at
// their offsets in the parameter buffer (with padding), host updates replacing device updates, and the restrictions
// (no update before instantiation, one executable graph, no device launch, graphs only). Checked against NVIDIA's
// runtime too.
#include "check.h"
#include <cstring>

// p[0] += value once, p[1] counts threads
__global__ void count(int *p, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(&p[0], value);
    }
    atomicAdd(&p[1], 1);
}

// parameters with padding: c at 0, value at 4, p at 8
__global__ void padded(char c, int value, int *p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p[2] = c * 1000 + value;
    }
}

__global__ void setParam(cudaGraphDeviceNode_t node, size_t offset, int value, int *status) {
    *status = cudaGraphKernelNodeSetParam(node, offset, &value, sizeof(value));
}

__global__ void setChar(cudaGraphDeviceNode_t node, char value, int *status) {
    *status = cudaGraphKernelNodeSetParam(node, 0, &value, sizeof(value));
}

__global__ void setGrid(cudaGraphDeviceNode_t node, unsigned int grid, int *status) {
    *status = cudaGraphKernelNodeSetGridDim(node, dim3(grid));
}

__global__ void setEnabled(cudaGraphDeviceNode_t node, int enabled, int *status) {
    *status = cudaGraphKernelNodeSetEnabled(node, enabled != 0);
}

__global__ void applyUpdates(cudaGraphDeviceNode_t node, int value, unsigned int grid, int *status) {
    __align__(16) char storage[2 * sizeof(cudaGraphKernelNodeUpdate)];
    memset(storage, 0, sizeof(storage));
    auto *updates = reinterpret_cast<cudaGraphKernelNodeUpdate *>(storage);
    updates[0].node = node;
    updates[0].field = cudaGraphKernelNodeFieldParam;
    updates[0].updateData.param.pValue = &value;
    updates[0].updateData.param.offset = 8;
    updates[0].updateData.param.size = sizeof(value);
    updates[1].node = node;
    updates[1].field = cudaGraphKernelNodeFieldGridDim;
    updates[1].updateData.gridDim.x = grid;
    updates[1].updateData.gridDim.y = 1;
    updates[1].updateData.gridDim.z = 1;
    *status = cudaGraphKernelNodeUpdatesApply(updates, 2);
}

// a kernel of the graph doubling the value the counting node adds at the next launch
__global__ void doubleNext(cudaGraphDeviceNode_t node, int *current) {
    *current *= 2;
    cudaGraphKernelNodeSetParam(node, 8, current, sizeof(int));
}

static cudaGraphNode_t addCount(cudaGraph_t graph, const cudaGraphNode_t *dep, int *p, int value) {
    void *args[] = {&p, &value};
    cudaKernelNodeParams params{};
    params.func = (void *) count;
    params.gridDim = 1;
    params.blockDim = 32;
    params.kernelParams = args;
    cudaGraphNode_t node;
    CHECK(cudaGraphAddKernelNode(&node, graph, dep, dep != nullptr ? 1 : 0, &params));
    return node;
}

static cudaGraphDeviceNode_t makeUpdatable(cudaGraphNode_t node) {
    cudaKernelNodeAttrValue value{};
    value.deviceUpdatableKernelNode.deviceUpdatable = 1;
    CHECK(cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &value));
    cudaKernelNodeAttrValue got{};
    CHECK(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &got));
    EXPECT(got.deviceUpdatableKernelNode.deviceUpdatable == 1 && got.deviceUpdatableKernelNode.devNode != nullptr &&
           got.deviceUpdatableKernelNode.devNode == value.deviceUpdatableKernelNode.devNode, "device node handle");
    return got.deviceUpdatableKernelNode.devNode;
}

static void launchAndCheck(cudaGraphExec_t exec, int *p, int expected0, int expected1, const char *what) {
    p[0] = p[1] = 0;
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(p[0] == expected0 && p[1] == expected1, "%s: %d %d (expected %d %d)", what, p[0], p[1], expected0,
           expected1);
}

static void testUpdates(int *p, int *status) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t node = addCount(graph, nullptr, p, 1);
    cudaGraphDeviceNode_t device_node = makeUpdatable(node);

    // no updates before the graph is instantiated
    setParam<<<1, 1>>>(device_node, 8, 5, status);
    CHECK(cudaDeviceSynchronize());
    EXPECT(*status == cudaErrorInvalidValue, "update before instantiation: %d", *status);

    cudaGraphExec_t exec, second;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    EXPECT(cudaGraphInstantiate(&second, graph, 0) == cudaErrorNotSupported, "second instantiation");
    EXPECT(cudaGraphInstantiateWithFlags(&second, graph, cudaGraphInstantiateFlagDeviceLaunch) ==
           cudaErrorNotSupported, "device launch");
    cudaGetLastError();
    launchAndCheck(exec, p, 1, 32, "initial");

    setParam<<<1, 1>>>(device_node, 8, 5, status);
    CHECK(cudaDeviceSynchronize());
    EXPECT(*status == 0, "cudaGraphKernelNodeSetParam: %d", *status);
    launchAndCheck(exec, p, 5, 32, "after cudaGraphKernelNodeSetParam");
    setGrid<<<1, 1>>>(device_node, 4, status);
    CHECK(cudaDeviceSynchronize());
    launchAndCheck(exec, p, 5, 128, "after cudaGraphKernelNodeSetGridDim");
    setEnabled<<<1, 1>>>(device_node, 0, status);
    CHECK(cudaDeviceSynchronize());
    launchAndCheck(exec, p, 0, 0, "disabled");
    setEnabled<<<1, 1>>>(device_node, 1, status);
    applyUpdates<<<1, 1>>>(device_node, 7, 3, status);
    CHECK(cudaDeviceSynchronize());
    EXPECT(*status == 0, "cudaGraphKernelNodeUpdatesApply: %d", *status);
    launchAndCheck(exec, p, 7, 96, "after cudaGraphKernelNodeUpdatesApply");
    // parameters past the end are refused
    setParam<<<1, 1>>>(device_node, 12, 1, status);
    CHECK(cudaDeviceSynchronize());
    EXPECT(*status == cudaErrorInvalidValue, "parameter past the end: %d", *status);

    // host updates replace the device's
    int nine = 9;
    void *args[] = {&p, &nine};
    cudaKernelNodeParams params{};
    params.func = (void *) count;
    params.gridDim = 1;
    params.blockDim = 32;
    params.kernelParams = args;
    CHECK(cudaGraphExecKernelNodeSetParams(exec, node, &params));
    launchAndCheck(exec, p, 9, 32, "after a host update");
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("device side kernel node updates");
}

static void testPaddedParameters(int *p, int *status) {
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    char c = 3;
    int value = 4;
    void *args[] = {&c, &value, &p};
    cudaKernelNodeParams params{};
    params.func = (void *) padded;
    params.gridDim = 1;
    params.blockDim = 1;
    params.kernelParams = args;
    cudaGraphNode_t node;
    CHECK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    cudaGraphDeviceNode_t device_node = makeUpdatable(node);
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(p[2] == 3004, "padded parameters: %d", p[2]);
    setParam<<<1, 1>>>(device_node, 4, 56, status);
    setChar<<<1, 1>>>(device_node, 2, status + 1);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGraphLaunch(exec, 0));
    CHECK(cudaDeviceSynchronize());
    EXPECT(p[2] == 2056, "padded parameters after updates: %d", p[2]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("device side updates at parameter offsets");
}

static void testUpdateFromTheGraph(int *p) {
    // a counting node and a kernel after it that doubles what the counting node adds at the next launch
    cudaGraph_t graph;
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t node = addCount(graph, nullptr, p, 1);
    cudaGraphDeviceNode_t device_node = makeUpdatable(node);
    int *current = p + 10;
    void *args[] = {&device_node, &current};
    cudaKernelNodeParams params{};
    params.func = (void *) doubleNext;
    params.gridDim = 1;
    params.blockDim = 1;
    params.kernelParams = args;
    cudaGraphNode_t updater;
    CHECK(cudaGraphAddKernelNode(&updater, graph, &node, 1, &params));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    *current = 1;
    p[0] = p[1] = 0;
    for (int i = 0; i < 5; i++) {
        CHECK(cudaGraphLaunch(exec, 0));
        CHECK(cudaDeviceSynchronize());
    }
    // 1 + 2 + 4 + 8 + 16
    EXPECT(p[0] == 31, "graph updating its own node: %d", p[0]);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    PASS("graph updating its own node");
}

static void testCapture(int *p, int *status) {
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeDeviceUpdatableKernelNode;
    attribute.val.deviceUpdatableKernelNode.deviceUpdatable = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = 1;
    config.blockDim = 32;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    // graphs only
    EXPECT(cudaLaunchKernelEx(&config, count, p, 1) == cudaErrorNotSupported, "outside a capture");
    cudaGetLastError();
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaLaunchKernelEx(&config, count, p, 1));
    CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphDeviceNode_t device_node = attribute.val.deviceUpdatableKernelNode.devNode;
    EXPECT(device_node != nullptr, "device node of a captured launch");
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));
    // (NVIDIA's runtime sometimes refuses updates before the executable graph was uploaded or launched)
    CHECK(cudaGraphUpload(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    setParam<<<1, 1>>>(device_node, 8, 11, status);
    CHECK(cudaDeviceSynchronize());
    launchAndCheck(exec, p, 11, 32, "captured node after an update");
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    PASS("device-updatable node from stream capture");
}

int main() {
    int *p;
    CHECK(cudaMallocManaged(&p, 64 * sizeof(int)));
    memset(p, 0, 64 * sizeof(int));
    int *status = p + 32;
    testUpdates(p, status);
    testPaddedParameters(p, status);
    testUpdateFromTheGraph(p);
    testCapture(p, status);
    CHECK(cudaFree(p));
    CHECK(cudaGetLastError());
    return 0;
}
