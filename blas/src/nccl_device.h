// LibreCUDA's NCCL: the kernels of the device transport (nccl_device.cu)
#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>

// the reductions of the kernels (average and pre-multiplied sums are sums with a divisor / scale)
enum NcclDeviceOp {
    NCCL_DEVICE_SUM, NCCL_DEVICE_PROD, NCCL_DEVICE_MAX, NCCL_DEVICE_MIN
};

/**
 * out1 / out2 (either may be nullptr) [i < count] = combine(a[i] * scale (if scaled), b[i]) / divide (if > 1): with a
 * only, a copy (scaled); with b only, b's copy; with both, op(a, b). blocks > 0 caps the grid (a copy over a link it
 * shares with a copy engine leaves the engine more of the link with few blocks).
 */
struct NcclCombine {
    ncclDataType_t type;
    int op;
    size_t count;
    const void *a, *b;
    void *out1, *out2;
    bool scaled;
    double scale;
    int divide;
    int blocks;
    // synchronization inside the kernel (all optional): before it starts, wait until *wait[i] >= waitValue[i]; once
    // every block finished (counted in *finished, which the last block resets), *signal[i] = signalValue[i] (release,
    // system scope). Kernels sharing a counter must not run at the same time.
    const uint64_t *wait[2];
    uint64_t waitValue[2];
    uint64_t *signal[2];
    uint64_t signalValue[2];
    unsigned *finished;
};

cudaError_t ncclDeviceCombine(const NcclCombine &combine, cudaStream_t stream);

/**
 * A small all-reduce between two ranks in one kernel per rank: once the peer freed the slot (*freed >= freedValue),
 * copy count elements of mine into the peer's inbox and announce them (*peerReady = sendSeq, *sent = sendSeq), wait
 * for the peer's (*ready >= recvSeq; in place also for *sent: the copy must be done before mine is overwritten),
 * out = op(mine, inbox) / divide, then free the peer's slot (*peerFreed = recvSeq). counters: two finished block
 * counters (no other kernel using them may run at the same time).
 */
struct NcclExchange {
    ncclDataType_t type;
    int op;
    size_t count;
    int divide;
    const void *mine;
    void *peerInbox;
    const void *inbox;
    void *out;
    const uint64_t *freed;
    uint64_t freedValue;
    uint64_t *peerReady, *sent;
    uint64_t sendSeq;
    const uint64_t *ready;
    uint64_t recvSeq;
    uint64_t *peerFreed;
    unsigned *counters;
};

cudaError_t ncclDeviceExchange(const NcclExchange &exchange, cudaStream_t stream);
