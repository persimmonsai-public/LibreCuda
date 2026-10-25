// External event nodes captured from streams (cudaEventRecordWithFlags with cudaEventRecordExternal,
// cudaStreamWaitEvent with cudaEventWaitExternal), as GROMACS's MD step graphs use them: the graph records an event
// after copying results to the host, which the host waits for after every launch, and the graph waits for an event
// recorded outside it before its first kernel.
#include "check.h"

__global__ void increment(int *counter) {
    *counter += 1;
}

__global__ void spin(long long cycles, volatile int *flag, int value) {
    const long long start = clock64();
    while (clock64() - start < cycles) {}
    *flag = value;
}

__global__ void copyFlag(const volatile int *flag, int *out) {
    *out = *flag;
}

int main() {
    int *counter, *flag, *seen, *host;
    CHECK(cudaMalloc(&counter, sizeof(int)));
    CHECK(cudaMalloc(&flag, sizeof(int)));
    CHECK(cudaMalloc(&seen, sizeof(int)));
    CHECK(cudaMemset(counter, 0, sizeof(int)));
    CHECK(cudaMemset(flag, 0, sizeof(int)));
    CHECK(cudaMallocHost(&host, 2 * sizeof(int)));
    cudaStream_t s, other;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
    cudaEvent_t copied, before;
    CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&before, cudaEventDisableTiming));

    // a graph: wait for `before` (recorded outside), copy the flag, increment the counter, copy both to the host,
    // record `copied` for the host
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    CHECK(cudaStreamWaitEvent(s, before, cudaEventWaitExternal));
    copyFlag<<<1, 1, 0, s>>>(flag, seen);
    increment<<<1, 1, 0, s>>>(counter);
    CHECK(cudaMemcpyAsync(host, counter, sizeof(int), cudaMemcpyDeviceToHost, s));
    CHECK(cudaMemcpyAsync(host + 1, seen, sizeof(int), cudaMemcpyDeviceToHost, s));
    CHECK(cudaEventRecordWithFlags(copied, s, cudaEventRecordExternal));
    CHECK(cudaStreamEndCapture(s, &graph));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    for (int i = 1; i <= 200; i++) {
        // another stream sets the flag to i after a while and records `before`: the graph's first kernel must see it
        spin<<<1, 1, 0, other>>>(i % 10 == 0 ? 2000000 : 1000, flag, i);
        CHECK(cudaEventRecord(before, other));
        CHECK(cudaGraphLaunch(exec, s));
        // the host waits for the event the graph records, then reads what the graph copied
        CHECK(cudaEventSynchronize(copied));
        EXPECT(host[0] == i, "launch %d: the host read the counter as %d after waiting for the graph's event", i, host[0]);
        EXPECT(host[1] == i, "launch %d: the graph's kernel ran before the event it waits for (flag %d)", i, host[1]);
    }

    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(copied));
    CHECK(cudaEventDestroy(before));
    CHECK(cudaStreamDestroy(s));
    CHECK(cudaStreamDestroy(other));
    CHECK(cudaFreeHost(host));
    CHECK(cudaFree(counter));
    CHECK(cudaFree(flag));
    CHECK(cudaFree(seen));
    PASS("graph_external_events");
    return 0;
}
