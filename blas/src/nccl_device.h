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
    // system scope). Kernels sharing a counter must not run at the same time. toggle (operations captured in graphs,
    // whose values don't change from launch to launch): the flags waited for are set to 1 and the last block resets
    // them to 0 before it signals (finished needed)
    const uint64_t *wait[2];
    uint64_t waitValue[2];
    uint64_t *signal[2];
    uint64_t signalValue[2];
    unsigned *finished;
    bool toggle;
};

cudaError_t ncclDeviceCombine(const NcclCombine &combine, cudaStream_t stream);

/**
 * A small all-reduce between two ranks in one kernel per rank: once the peer freed the slot (*freed >= freedValue),
 * copy count elements of mine into the peer's inbox and announce them (*peerReady = sendSeq, *sent = sendSeq), wait
 * for the peer's (*ready >= recvSeq; in place also for *sent: the copy must be done before mine is overwritten),
 * out = op(mine, inbox) / divide, then free the peer's slot (*peerFreed = recvSeq). counters: two finished block
 * counters (no other kernel using them may run at the same time). toggle (graphs): the values are 1 and the kernel
 * resets the flags it waited for (*freed, *ready, *sent) to 0 before the signals that let them be set again.
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
    bool toggle;
};

cudaError_t ncclDeviceExchange(const NcclExchange &exchange, cudaStream_t stream);

/**
 * A small all-reduce between two ranks in one kernel of one block per rank, with LL stores (the flag in the data):
 * each 16 byte store into the peer's slots carries 8 bytes of mine and the epoch twice (one per 8 bytes, which the
 * link writes whole), the kernel waits until all of my slots show the epoch, then out = op(mine, peer's) / divide. No
 * other flags: epochs alternate between two slot sets, and a rank writes a set again only after it received the
 * peer's next epoch, which the peer sends after it read this one. *epoch: the last epoch (this rank's, in its own
 * memory; the ranks advance them together); kernels using it must not run at the same time. At most
 * NCCL_LL_SLOTS * 8 bytes.
 */
constexpr int NCCL_LL_SLOTS = 1024;
// bytes of a rank's LL slots (two sets)
constexpr size_t NCCL_LL_BYTES = 2 * NCCL_LL_SLOTS * 16;

struct NcclLLExchange {
    ncclDataType_t type;
    int op;
    size_t count;
    int divide;
    const void *mine;
    void *out;
    void *peerSlots;
    const void *slots;
    uint32_t *epoch;
};

cudaError_t ncclDeviceLLExchange(const NcclLLExchange &exchange, cudaStream_t stream);

/**
 * A small all-reduce between two ranks driven by one of them (the one whose kernels reach the other GPU fastest), one
 * kernel per rank. The driver writes its data into the passive rank's inbox (result: the slot of this exchange's
 * parity) right away; the passive rank copies its data into its staging buffer and announces it (*staged = seq, in
 * the driver's memory); the driver reads it across the link, combines it with its own into its out and announces
 * that it is done with both (*done = seq, in the passive rank's memory); the passive rank then combines its data with
 * the inbox into its out. staging / result: the passive rank's buffers (the driver's pointers reach them in the peer's
 * memory). counter: a finished block counter of the driver.
 */
constexpr size_t NCCL_DRIVEN_MAX_BYTES = size_t(1) << 20;

struct NcclDriven {
    ncclDataType_t type;
    int op;
    size_t count;
    int divide;
    bool driver;
    const void *mine;
    void *out;
    void *staging, *result;
    uint64_t *staged, *done;
    uint64_t seq;
    unsigned *counter;
};

cudaError_t ncclDeviceDriven(const NcclDriven &driven, cudaStream_t stream);
