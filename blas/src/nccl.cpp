/**
 * LibreCUDA's NCCL (libnccl.so.2): collectives (all-reduce, broadcast, reduce, all-gather, reduce-scatter, all-to-all)
 * and point-to-point send / receive between ranks in one or several processes.
 *
 * Two transports. The device transport (below, "the device transport") moves data GPU to GPU: when every rank's GPU
 * reaches every other's memory (the same GPU, or peer access) and stream memory operations exist, the ranks exchange
 * device memory regions at initialization and collectives become kernels and flag waits / writes enqueued on the
 * streams, with nothing waiting on the CPU. Between two ranks on different GPUs, all-reduces (tensor parallelism's
 * collective) are exchanges: one rank uploads its data with its copy engine, the other one's data is read with a few
 * blocks, both combine locally, so that both directions of the link stay busy with the engines driving them fastest.
 *
 * Otherwise (and for fp8 types) the transport is host shared memory (POSIX shm, named by the communicator's unique
 * id): every rank has a data slot and every ordered pair of ranks a mailbox. A collective proceeds in chunks: each
 * rank copies its chunk from the GPU into its slot and announces it (posted counter); once all ranks posted, each
 * computes its result from the slots (reductions on the CPU) and copies it to the GPU, then announces that it is done
 * with the slots (done counter); a slot is reused once all ranks are done. Operations are non-blocking steps of a
 * progress loop, so the operations of a group (several ranks in one process, matching sends and receives) advance
 * together. The calls return when the results are on the GPU (the stream's earlier work is waited for first).
 */
#include "nccl_device.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <nccl.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int MAX_RANKS = 64;
constexpr size_t SLOT_BYTES = size_t(16) << 20;
constexpr size_t MAILBOX_BYTES = size_t(4) << 20;
constexpr uint32_t COMM_MAGIC = 0x4c43434e;

// the shared segment (zero filled when created; atomics in shared memory are lock free)
struct Segment {
    std::atomic<int> joined;
    std::atomic<uint64_t> posted[MAX_RANKS];
    std::atomic<uint64_t> done[MAX_RANKS];
    std::atomic<uint64_t> sent[MAX_RANKS * MAX_RANKS];
    std::atomic<uint64_t> consumed[MAX_RANKS * MAX_RANKS];
    uint64_t mailboxBytes[MAX_RANKS * MAX_RANKS];
    // device transport setup: barriers passed, votes (1 yes, 2 no), the ranks' GPUs and regions, measured bandwidths
    std::atomic<uint64_t> arrived[MAX_RANKS];
    std::atomic<int> vote[MAX_RANKS], vote2[MAX_RANKS];
    unsigned char uuid[MAX_RANKS][16];
    int pid[MAX_RANKS];
    uint64_t regionPtr[MAX_RANKS];
    char regionHandle[MAX_RANKS][64];
    float pushBw[MAX_RANKS][MAX_RANKS], pullBw[MAX_RANKS][MAX_RANKS], copyBw[MAX_RANKS][MAX_RANKS];
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);

size_t segmentSize(int nranks) {
    return (sizeof(Segment) + 4095) / 4096 * 4096 + size_t(nranks) * SLOT_BYTES +
           size_t(nranks) * nranks * MAILBOX_BYTES;
}

struct PreMulSum {
    double scalar;
    const void *devicePtr;
    ncclDataType_t type;
};

struct Transport;

} // namespace

struct ncclComm {
    uint32_t magic;
    int rank, nranks, device;
    std::string name;
    Segment *seg;
    char *base;
    size_t size;
    // collective rounds this rank started; point-to-point messages per peer (sent / received)
    uint64_t round;
    std::vector<uint64_t> sentTo, receivedFrom;
    uint64_t splits;
    bool joined;
    std::map<int, PreMulSum> preMulSums;
    int nextOp;
    // the device transport (nullptr: the host transport carries everything)
    Transport *transport;
};

namespace {

thread_local std::string t_lastError;

ncclResult_t fail(ncclResult_t result, const std::string &message) {
    t_lastError = message;
    return result;
}

bool valid(ncclComm_t comm) {
    return comm != nullptr && comm->magic == COMM_MAGIC;
}

char *slot(ncclComm_t c, int rank) {
    return c->base + (sizeof(Segment) + 4095) / 4096 * 4096 + size_t(rank) * SLOT_BYTES;
}

char *mailbox(ncclComm_t c, int from, int to) {
    return c->base + (sizeof(Segment) + 4095) / 4096 * 4096 + size_t(c->nranks) * SLOT_BYTES +
           (size_t(from) * c->nranks + to) * MAILBOX_BYTES;
}

void pause(int &spins) {
    if (++spins < 1000) {
        std::this_thread::yield();
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

size_t typeSize(ncclDataType_t type) {
    switch (type) {
        case ncclInt8: case ncclUint8: case ncclFloat8e4m3: case ncclFloat8e5m2: return 1;
        case ncclFloat16: case ncclBfloat16: return 2;
        case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
        default: return 8;
    }
}

// ---- host arithmetic on the element types

float halfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // subnormal: normalize
            int e = -1;
            uint32_t m = mantissa;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x400) == 0);
            bits = sign | uint32_t(127 - 15 - e) << 23 | (m & 0x3ff) << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t floatToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exponent = int((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) {
        return uint16_t(sign | 0x7c00u | (mantissa ? 0x200 : 0));
    }
    if (exponent >= 31) {
        return uint16_t(sign | 0x7c00u);
    }
    if (exponent <= 0) {
        if (exponent < -10) return uint16_t(sign);
        mantissa |= 0x800000;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1), midpoint = 1u << (shift - 1);
        if (rest > midpoint || (rest == midpoint && (half & 1))) half++;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | uint32_t(exponent) << 10 | mantissa >> 13;
    const uint32_t rest = mantissa & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) half++;
    return uint16_t(half);
}

float bf16ToFloat(uint16_t b) {
    const uint32_t bits = uint32_t(b) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t floatToBf16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    if ((x & 0x7fffffff) > 0x7f800000) return uint16_t((x >> 16) | 0x40);
    const uint32_t rounding = 0x7fff + ((x >> 16) & 1);
    return uint16_t((x + rounding) >> 16);
}

double loadValue(const char *p, size_t i, ncclDataType_t type) {
    switch (type) {
        case ncclInt8: return reinterpret_cast<const int8_t *>(p)[i];
        case ncclUint8: return reinterpret_cast<const uint8_t *>(p)[i];
        case ncclInt32: return reinterpret_cast<const int32_t *>(p)[i];
        case ncclUint32: return reinterpret_cast<const uint32_t *>(p)[i];
        case ncclInt64: return double(reinterpret_cast<const int64_t *>(p)[i]);
        case ncclUint64: return double(reinterpret_cast<const uint64_t *>(p)[i]);
        case ncclFloat16: return halfToFloat(reinterpret_cast<const uint16_t *>(p)[i]);
        case ncclBfloat16: return bf16ToFloat(reinterpret_cast<const uint16_t *>(p)[i]);
        case ncclFloat32: return reinterpret_cast<const float *>(p)[i];
        default: return reinterpret_cast<const double *>(p)[i];
    }
}

void storeValue(char *p, size_t i, ncclDataType_t type, double v) {
    switch (type) {
        case ncclInt8: reinterpret_cast<int8_t *>(p)[i] = int8_t(v); break;
        case ncclUint8: reinterpret_cast<uint8_t *>(p)[i] = uint8_t(v); break;
        case ncclInt32: reinterpret_cast<int32_t *>(p)[i] = int32_t(v); break;
        case ncclUint32: reinterpret_cast<uint32_t *>(p)[i] = uint32_t(v); break;
        case ncclInt64: reinterpret_cast<int64_t *>(p)[i] = int64_t(v); break;
        case ncclUint64: reinterpret_cast<uint64_t *>(p)[i] = uint64_t(v); break;
        case ncclFloat16: reinterpret_cast<uint16_t *>(p)[i] = floatToHalf(float(v)); break;
        case ncclBfloat16: reinterpret_cast<uint16_t *>(p)[i] = floatToBf16(float(v)); break;
        case ncclFloat32: reinterpret_cast<float *>(p)[i] = float(v); break;
        default: reinterpret_cast<double *>(p)[i] = v; break;
    }
}

// native types are reduced in their own type (64 bit integers exactly, not through double)
template<typename I>
void reduceNative(const std::vector<const char *> &inputs, size_t count, ncclRedOp_t op, char *out) {
    auto *o = reinterpret_cast<I *>(out);
    memcpy(o, inputs[0], count * sizeof(I));
    // (rank by rank, so the loops vectorize)
    for (size_t r = 1; r < inputs.size(); r++) {
        const I *v = reinterpret_cast<const I *>(inputs[r]);
        switch (op) {
            case ncclProd: for (size_t i = 0; i < count; i++) o[i] *= v[i]; break;
            case ncclMax: for (size_t i = 0; i < count; i++) o[i] = std::max(o[i], v[i]); break;
            case ncclMin: for (size_t i = 0; i < count; i++) o[i] = std::min(o[i], v[i]); break;
            default: for (size_t i = 0; i < count; i++) o[i] += v[i]; break;
        }
    }
    if (op == ncclAvg) {
        for (size_t i = 0; i < count; i++) o[i] /= I(inputs.size());
    }
}

void reduce(const std::vector<const char *> &inputs, size_t count, ncclDataType_t type, ncclRedOp_t op, char *out) {
    switch (type) {
        case ncclInt32: return reduceNative<int32_t>(inputs, count, op, out);
        case ncclUint32: return reduceNative<uint32_t>(inputs, count, op, out);
        case ncclInt64: return reduceNative<int64_t>(inputs, count, op, out);
        case ncclUint64: return reduceNative<uint64_t>(inputs, count, op, out);
        case ncclFloat32: return reduceNative<float>(inputs, count, op, out);
        case ncclFloat64: return reduceNative<double>(inputs, count, op, out);
        default: break;
    }
    for (size_t i = 0; i < count; i++) {
        double acc = loadValue(inputs[0], i, type);
        for (size_t r = 1; r < inputs.size(); r++) {
            const double v = loadValue(inputs[r], i, type);
            switch (op) {
                case ncclProd: acc *= v; break;
                case ncclMax: acc = std::max(acc, v); break;
                case ncclMin: acc = std::min(acc, v); break;
                default: acc += v; break;
            }
        }
        if (op == ncclAvg) acc /= double(inputs.size());
        storeValue(out, i, type, acc);
    }
}

// ---- operations

enum class Kind {
    ALL_REDUCE, BROADCAST, REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL, SEND, RECV
};

struct Operation {
    Kind kind;
    ncclComm_t comm;
    const char *send;
    char *recv;
    size_t count;  // elements (per rank for all-gather / reduce-scatter / all-to-all)
    ncclDataType_t type;
    ncclRedOp_t op;
    int root, peer;
    cudaStream_t stream;
    // progress: next element, whether this rank posted the current chunk, its round
    size_t offset;
    bool posted;
    uint64_t round;
    bool finished;
    double scale;  // pre-multiplied sums
};

thread_local int t_groupDepth = 0;
thread_local std::vector<Operation> t_group;
thread_local std::vector<ncclComm_t> t_pendingJoins;
thread_local std::vector<Operation> t_deviceGroup;

// elements per chunk: the slot holds the rank's contribution of a chunk
size_t chunkElements(const Operation &op) {
    const size_t element = typeSize(op.type);
    switch (op.kind) {
        case Kind::ALL_TO_ALL: return std::max<size_t>(1, SLOT_BYTES / element / op.comm->nranks);
        case Kind::SEND:
        case Kind::RECV: return std::max<size_t>(1, MAILBOX_BYTES / element);
        default: return std::max<size_t>(1, SLOT_BYTES / element);
    }
}

// the elements an operation moves in total (reduce-scatter: the whole input, chunk by chunk)
size_t totalElements(const Operation &op) {
    return op.kind == Kind::REDUCE_SCATTER ? op.count * op.comm->nranks : op.count;
}

bool cudaOk(cudaError_t e) {
    if (e != cudaSuccess) {
        t_lastError = std::string("CUDA error: ") + cudaGetErrorString(e);
        cudaGetLastError();
        return false;
    }
    return true;
}

// the rank's contribution of the current chunk into its slot (pre-multiplied for PreMulSum)
bool post(Operation &op, size_t n) {
    ncclComm_t c = op.comm;
    const size_t element = typeSize(op.type);
    char *mine = slot(c, c->rank);
    bool ok = true;
    switch (op.kind) {
        case Kind::ALL_REDUCE:
        case Kind::REDUCE:
        case Kind::REDUCE_SCATTER:
            ok = cudaOk(cudaMemcpy(mine, op.send + op.offset * element, n * element, cudaMemcpyDeviceToHost));
            if (ok && op.scale != 1.0) {
                for (size_t i = 0; i < n; i++) storeValue(mine, i, op.type, loadValue(mine, i, op.type) * op.scale);
            }
            break;
        case Kind::BROADCAST:
            if (c->rank == op.root) {
                ok = cudaOk(cudaMemcpy(mine, op.send + op.offset * element, n * element, cudaMemcpyDeviceToHost));
            }
            break;
        case Kind::ALL_GATHER:
            ok = cudaOk(cudaMemcpy(mine, op.send + op.offset * element, n * element, cudaMemcpyDeviceToHost));
            break;
        case Kind::ALL_TO_ALL:
            // [peer][n]: elements offset.. of the block for each peer
            ok = cudaOk(cudaMemcpy2D(mine, n * element, op.send + op.offset * element, op.count * element, n * element,
                                     c->nranks, cudaMemcpyDeviceToHost));
            break;
        default:
            break;
    }
    return ok;
}

// the result of the current chunk from all slots
bool complete(Operation &op, size_t n) {
    ncclComm_t c = op.comm;
    const size_t element = typeSize(op.type);
    std::vector<const char *> slots(c->nranks);
    for (int r = 0; r < c->nranks; r++) slots[r] = slot(c, r);
    std::vector<char> result;
    switch (op.kind) {
        case Kind::ALL_REDUCE:
        case Kind::REDUCE:
            if (op.kind == Kind::REDUCE && c->rank != op.root) return true;
            result.resize(n * element);
            reduce(slots, n, op.type, op.op, result.data());
            return cudaOk(cudaMemcpy(op.recv + op.offset * element, result.data(), n * element,
                                     cudaMemcpyHostToDevice));
        case Kind::REDUCE_SCATTER: {
            // the part of [offset, offset + n) that is this rank's block
            const size_t first = std::max(op.offset, size_t(c->rank) * op.count);
            const size_t last = std::min(op.offset + n, size_t(c->rank + 1) * op.count);
            if (first >= last) return true;
            std::vector<const char *> part(c->nranks);
            for (int r = 0; r < c->nranks; r++) part[r] = slots[r] + (first - op.offset) * element;
            result.resize((last - first) * element);
            reduce(part, last - first, op.type, op.op, result.data());
            return cudaOk(cudaMemcpy(op.recv + (first - size_t(c->rank) * op.count) * element, result.data(),
                                     result.size(), cudaMemcpyHostToDevice));
        }
        case Kind::BROADCAST:
            return cudaOk(cudaMemcpy(op.recv + op.offset * element, slots[op.root], n * element,
                                     cudaMemcpyHostToDevice));
        case Kind::ALL_GATHER:
            for (int r = 0; r < c->nranks; r++) {
                if (!cudaOk(cudaMemcpy(op.recv + (size_t(r) * op.count + op.offset) * element, slots[r], n * element,
                                       cudaMemcpyHostToDevice))) {
                    return false;
                }
            }
            return true;
        case Kind::ALL_TO_ALL:
            // from each rank r: its block for this rank
            for (int r = 0; r < c->nranks; r++) {
                if (!cudaOk(cudaMemcpy(op.recv + (size_t(r) * op.count + op.offset) * element,
                                       slots[r] + size_t(c->rank) * n * element, n * element,
                                       cudaMemcpyHostToDevice))) {
                    return false;
                }
            }
            return true;
        default:
            return true;
    }
}

/**
 * One step of an operation; returns false on errors. A collective chunk: post (once the slots are free), then
 * complete (once all ranks posted). A message chunk: into the mailbox when it is empty / out when it is full.
 */
bool step(Operation &op) {
    ncclComm_t c = op.comm;
    Segment *s = c->seg;
    const size_t total = totalElements(op);
    if (op.kind == Kind::SEND || op.kind == Kind::RECV) {
        const size_t element = typeSize(op.type);
        const int from = op.kind == Kind::SEND ? c->rank : op.peer, to = op.kind == Kind::SEND ? op.peer : c->rank;
        const int pair = from * MAX_RANKS + to;
        const size_t n = std::min(chunkElements(op), total - op.offset);
        if (op.kind == Kind::SEND) {
            if (s->sent[pair].load(std::memory_order_acquire) != s->consumed[pair].load(std::memory_order_acquire)) {
                return true;  // the receiver hasn't taken the previous chunk
            }
            if (n > 0 && !cudaOk(cudaMemcpy(mailbox(c, from, to), op.send + op.offset * element, n * element,
                                            cudaMemcpyDeviceToHost))) {
                return false;
            }
            s->mailboxBytes[pair] = n * element;
            s->sent[pair].fetch_add(1, std::memory_order_acq_rel);
        } else {
            if (s->sent[pair].load(std::memory_order_acquire) == s->consumed[pair].load(std::memory_order_acquire)) {
                return true;  // nothing yet
            }
            if (n > 0 && !cudaOk(cudaMemcpy(op.recv + op.offset * element, mailbox(c, from, to), n * element,
                                            cudaMemcpyHostToDevice))) {
                return false;
            }
            s->consumed[pair].fetch_add(1, std::memory_order_acq_rel);
        }
        op.offset += n;
        op.finished = op.offset >= total;
        return true;
    }
    const size_t n = std::min(chunkElements(op), total - op.offset);
    if (!op.posted) {
        // a new round: the slots of the previous one must be free
        const uint64_t round = c->round + 1;
        for (int r = 0; r < c->nranks; r++) {
            if (s->done[r].load(std::memory_order_acquire) < round - 1) return true;
        }
        if (!post(op, n)) return false;
        c->round = round;
        op.round = round;
        op.posted = true;
        s->posted[c->rank].store(round, std::memory_order_release);
        return true;
    }
    for (int r = 0; r < c->nranks; r++) {
        if (s->posted[r].load(std::memory_order_acquire) < op.round) return true;
    }
    if (!complete(op, n)) return false;
    s->done[c->rank].store(op.round, std::memory_order_release);
    op.posted = false;
    op.offset += n;
    op.finished = op.offset >= total;
    return true;
}

// waits for the streams, then advances all operations until they finished
ncclResult_t run(std::vector<Operation> &ops) {
    for (Operation &op: ops) {
        if (!cudaOk(cudaSetDevice(op.comm->device)) || !cudaOk(cudaStreamSynchronize(op.stream))) {
            return ncclUnhandledCudaError;
        }
        op.finished = totalElements(op) == 0 && op.kind != Kind::SEND && op.kind != Kind::RECV;
    }
    int spins = 0;
    bool pending = true;
    while (pending) {
        pending = false;
        bool progressed = false;
        for (Operation &op: ops) {
            if (op.finished) continue;
            const size_t before = op.offset;
            const bool posted = op.posted;
            if (cudaSetDevice(op.comm->device) != cudaSuccess || !step(op)) {
                return ncclUnhandledCudaError;
            }
            progressed = progressed || op.offset != before || op.posted != posted;
            pending = pending || !op.finished;
        }
        if (pending && !progressed) {
            pause(spins);
        } else {
            spins = 0;
        }
    }
    return ncclSuccess;
}

// ---- the device transport
//
// Each rank has a region of device memory: 64 bit flags (ready[p]: the chunks rank p sent it; freed[q]: the chunks
// rank q consumed of what it sent), and per peer an inbox and an outbox of SLOTS chunk slots. A directed edge carries
// chunks by push (the sender's kernel writes into the receiver's inbox) or by pull (the sender's kernel writes into its
// own outbox, the receiver's kernel reads from there), whichever was faster when the communicator was set up. Streams
// order the steps across GPUs and processes with stream memory operations: a step waits until the local flag shows
// its chunk (or a free slot) and then announces its own progress in the peer's flags. Collectives are rings (all-reduce,
// reduce-scatter, all-gather) or chains (broadcast, reduce) of chunk steps, each a kernel that combines a received
// chunk with the rank's own data and writes the result to its output and / or the next rank's slot. A communicator's
// operations run in issue order on its own stream, which joins the callers' streams (events) and which they wait for:
// operations of one communicator on several streams must not overtake each other across the ranks. Nothing waits on
// the CPU.
//
// Operations captured into CUDA graphs can't use the sequence numbers: a graph replays the values it was captured
// with, and stream memory operations can't be captured on LibreCUDA. They have their own flags and slots in the
// region (graph mode): per edge and slot a ready flag (the sender sets it to 1 once its chunk is in the slot) and a
// freed flag (the receiver sets it to 1 once it consumed the chunk, 1 initially); whoever waits for a flag resets it
// to 0 before it signals what lets the other side set it again. The kernels wait and signal themselves, the slots
// follow per edge cursors that advance at capture (both ranks capture the same operations, so they agree on each
// chunk's slot, and a replay uses the same slots again). Graph mode runs on the captured stream (and a second one
// joining the capture for uploads); it leaves the uncaptured operations' state alone, so graphs and uncaptured
// operations mix in any order, as long as all ranks issue and replay them in the same order.

// chunk size: 2 MB between two ranks (the pair all-reduce's copies are faster in larger pieces), else 1 MB
size_t chunkBytes(int nranks) {
    return nranks == 2 ? size_t(2) << 20 : size_t(1) << 20;
}
constexpr int SLOTS = 4;
constexpr size_t LL_SLOTS = 16384;
constexpr size_t DRIVEN = LL_SLOTS + NCCL_LL_BYTES;
constexpr size_t FLAGS = DRIVEN + 3 * NCCL_DRIVEN_MAX_BYTES;

// the flags (and the LL slots), then per peer SLOTS inbox and outbox slots, then those of graph mode
size_t regionSize(int nranks) {
    return FLAGS + 4 * size_t(nranks) * SLOTS * chunkBytes(nranks);
}

uint64_t *readyFlag(char *region, int from) {
    return reinterpret_cast<uint64_t *>(region) + from;
}

uint64_t *freedFlag(char *region, int to) {
    return reinterpret_cast<uint64_t *>(region + 512) + to;
}

// the chunks of its input a rank has finished reading (staged or uploaded): its own sums may then overwrite them
uint64_t *readFlag(char *region) {
    return reinterpret_cast<uint64_t *>(region + 1024);
}

// finished block counters of the kernels that synchronize themselves (NcclCombine::finished): one per stream
unsigned *finishedCounter(char *region, bool upload) {
    return reinterpret_cast<unsigned *>(region + 1536) + (upload ? 1 : 0);
}

// the small exchanges (ncclDeviceExchange): the chunks a rank finished sending, and the kernel's block counters
uint64_t *sentFlag(char *region) {
    return reinterpret_cast<uint64_t *>(region + 1032);
}

unsigned *exchangeCounters(char *region) {
    return reinterpret_cast<unsigned *>(region + 1544);
}

// the LL exchanges (ncclDeviceLLExchange): the rank's last epoch, and its slots
uint32_t *llEpoch(char *region) {
    return reinterpret_cast<uint32_t *>(region + 2048);
}

char *llSlots(char *region) {
    return region + LL_SLOTS;
}

// the driven exchanges (ncclDeviceDriven): the passive rank's staging buffer and two inbox slots, the flags, the
// driver's block counter
char *drivenStaging(char *region) {
    return region + DRIVEN;
}

char *drivenInbox(char *region, uint64_t seq) {
    return region + DRIVEN + (1 + seq % 2) * NCCL_DRIVEN_MAX_BYTES;
}

uint64_t *drivenStaged(char *region) {
    return reinterpret_cast<uint64_t *>(region + 2056);
}

uint64_t *drivenDone(char *region) {
    return reinterpret_cast<uint64_t *>(region + 2064);
}

unsigned *drivenCounter(char *region) {
    return reinterpret_cast<unsigned *>(region + 2072);
}

// all-reduces between two ranks above 16 KB and up to this size (LIBRECUDA_NCCL_DRIVEN_BYTES, at most 1 MB; 0: none)
// are driven exchanges, in one piece: below, the driven exchange's chain of flags across the link costs more than
// the slow direction's pushes; above, the copy engine's chunks are faster
constexpr size_t DRIVEN_MIN_BYTES = size_t(16) << 10;

size_t drivenExchangeBytes() {
    static const size_t bytes = [] {
        const char *v = getenv("LIBRECUDA_NCCL_DRIVEN_BYTES");
        return std::min<size_t>(v != nullptr ? strtoull(v, nullptr, 0) : NCCL_DRIVEN_MAX_BYTES, NCCL_DRIVEN_MAX_BYTES);
    }();
    return bytes;
}

// all-reduces up to this size between two ranks are LL exchanges (LIBRECUDA_NCCL_LL_BYTES): above it (about 6 KB on
// the test system), sending twice the data (the flags) over a link that the slower GPU writes at 2.3 GB/s costs more
// than the flag round trip it saves
size_t llExchangeBytes() {
    static const size_t bytes = [] {
        const char *v = getenv("LIBRECUDA_NCCL_LL_BYTES");
        return std::min<size_t>(v != nullptr ? strtoull(v, nullptr, 0) : 5120, NCCL_LL_SLOTS * size_t(8));
    }();
    return bytes;
}

// all-reduces up to this size between two ranks are one kernel per rank (ncclDeviceExchange): above it, the push
// from the GPU whose writes into the other one are slow (2.3 GB/s on the test system) costs more than the latency
// it saves
size_t smallExchangeBytes() {
    return size_t(64) << 10;
}

// graph mode: the ready / freed flags of an edge's slot, the small exchange's sent flag, the block counters
uint64_t *graphReadyFlag(char *region, int from, int slot) {
    return reinterpret_cast<uint64_t *>(region + 4096) + from * SLOTS + slot;
}

uint64_t *graphFreedFlag(char *region, int to, int slot) {
    return reinterpret_cast<uint64_t *>(region + 4096 + MAX_RANKS * SLOTS * 8) + to * SLOTS + slot;
}

uint64_t *graphSentFlag(char *region) {
    return reinterpret_cast<uint64_t *>(region + 4096 + 2 * MAX_RANKS * SLOTS * 8);
}

unsigned *graphFinishedCounter(char *region, bool upload) {
    return reinterpret_cast<unsigned *>(region + 4096 + 2 * MAX_RANKS * SLOTS * 8 + 64) + (upload ? 1 : 0);
}

unsigned *graphExchangeCounters(char *region) {
    return reinterpret_cast<unsigned *>(region + 4096 + 2 * MAX_RANKS * SLOTS * 8 + 128);
}

char *inbox(char *region, int nranks, int from, uint64_t seq) {
    return region + FLAGS + (size_t(from) * SLOTS + (seq - 1) % SLOTS) * chunkBytes(nranks);
}

char *outbox(char *region, int nranks, int to, uint64_t seq) {
    return region + FLAGS + (size_t(nranks + to) * SLOTS + (seq - 1) % SLOTS) * chunkBytes(nranks);
}

char *graphInbox(char *region, int nranks, int from, int slot) {
    return region + FLAGS + (size_t(2 * nranks + from) * SLOTS + slot) * chunkBytes(nranks);
}

char *graphOutbox(char *region, int nranks, int to, int slot) {
    return region + FLAGS + (size_t(3 * nranks + to) * SLOTS + slot) * chunkBytes(nranks);
}

struct Transport {
    // the communicator's operations run in issue order on this stream (the callers' streams join it and wait for it)
    cudaStream_t stream;
    cudaEvent_t joined, done;
    // two ranks: the rank that sends its all-reduce data with its copy engine (the other one's kernels stage theirs),
    // or -1 (rings); the sends run on the upload stream
    int uploader = -1;
    uint64_t readSeq = 0;
    uint64_t drivenSeq = 0;  // the driven exchanges so far
    cudaStream_t upload;
    cudaEvent_t forked, uploaded;
    // the stream of the communicator's last operation; direct: it ran there, not on the transport's stream (a small
    // exchange: no events, which cost a GPU wait each), which therefore has to catch up before its next work
    cudaStream_t lastStream = nullptr;
    bool lastDirect = false;
    std::mutex mutex;
    char *region;
    std::vector<char *> regions;  // every rank's region, as this process reaches it
    std::vector<bool> opened;     // regions opened from interprocess handles
    std::vector<bool> pushTo;     // edge to rank q: push (else pull)
    std::vector<bool> pullFrom;   // edge from rank p: pull (else push)
    std::vector<uint64_t> sendSeq, recvSeq;
    // graph mode: the chunks captured per edge (slot cursors), the stream that joins captures for uploads and the
    // events forking it off / joining it back
    std::vector<uint64_t> graphSent, graphReceived;
    cudaStream_t captureUpload;
    cudaEvent_t captureFork, captureJoin;
    // why operations can't be captured (nullptr: they can)
    const char *noGraphs = nullptr;
};

// stream memory operations of the driver API (LibreCUDA's or NVIDIA's libcuda)
using WaitValue64 = int (*)(cudaStream_t, unsigned long long, unsigned long long, unsigned int);
using WriteValue64 = int (*)(cudaStream_t, unsigned long long, unsigned long long, unsigned int);
WaitValue64 g_waitValue = nullptr;
WriteValue64 g_writeValue = nullptr;

bool memOpsAvailable() {
    static const bool available = [] {
        void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (cuda == nullptr) return false;
        auto find = [&](const char *name) {
            const std::string v2 = std::string(name) + "_v2";
            void *f = dlsym(cuda, v2.c_str());
            return f != nullptr ? f : dlsym(cuda, name);
        };
        g_waitValue = reinterpret_cast<WaitValue64>(find("cuStreamWaitValue64"));
        g_writeValue = reinterpret_cast<WriteValue64>(find("cuStreamWriteValue64"));
        return g_waitValue != nullptr && g_writeValue != nullptr;
    }();
    return available;
}

// LIBRECUDA_NCCL_P2P=0 or NCCL_P2P_DISABLE=1: the host transport only
bool deviceTransportWanted() {
    const char *v = getenv("LIBRECUDA_NCCL_P2P");
    if (v != nullptr && strcmp(v, "0") == 0) return false;
    v = getenv("NCCL_P2P_DISABLE");
    return v == nullptr || strcmp(v, "1") != 0;
}

bool waitValue(cudaStream_t s, const uint64_t *flag, uint64_t value) {
    // (CU_STREAM_WAIT_VALUE_GEQ)
    return g_waitValue(s, reinterpret_cast<unsigned long long>(flag), value, 0) == 0;
}

bool writeValue(cudaStream_t s, uint64_t *flag, uint64_t value) {
    return g_writeValue(s, reinterpret_cast<unsigned long long>(flag), value, 0) == 0;
}

/**
 * One chunk step on the stream: receives a chunk from rank `from` (-1: none), combines it with the rank's own data
 * (k.a, nullptr: none), writes the result to k.out1 (nullptr: nowhere) and sends it to rank `to` (-1: nobody). graph:
 * in graph mode (the kernel waits and signals itself)
 */
bool chunkStep(ncclComm_t c, cudaStream_t s, int from, int to, NcclCombine k, bool graph) {
    Transport &t = *c->transport;
    if (graph) {
        k.toggle = true;
        k.finished = graphFinishedCounter(t.region, false);
        if (from >= 0) {
            const int slot = int(t.graphReceived[from]++ % SLOTS);
            k.wait[0] = graphReadyFlag(t.region, from, slot);
            k.waitValue[0] = 1;
            k.b = t.pullFrom[from] ? graphOutbox(t.regions[from], c->nranks, c->rank, slot)
                                   : graphInbox(t.region, c->nranks, from, slot);
            k.signal[1] = graphFreedFlag(t.regions[from], c->rank, slot);
            k.signalValue[1] = 1;
        }
        if (to >= 0) {
            const int slot = int(t.graphSent[to]++ % SLOTS);
            k.wait[1] = graphFreedFlag(t.region, to, slot);
            k.waitValue[1] = 1;
            k.out2 = t.pushTo[to] ? graphInbox(t.regions[to], c->nranks, c->rank, slot)
                                  : graphOutbox(t.region, c->nranks, to, slot);
            k.signal[0] = graphReadyFlag(t.regions[to], c->rank, slot);
            k.signalValue[0] = 1;
        }
        return cudaOk(ncclDeviceCombine(k, s));
    }
    uint64_t seqIn = 0, seqOut = 0;
    if (from >= 0) {
        seqIn = ++t.recvSeq[from];
        if (!waitValue(s, readyFlag(t.region, from), seqIn)) return false;
        k.b = t.pullFrom[from] ? outbox(t.regions[from], c->nranks, c->rank, seqIn) : inbox(t.region, c->nranks, from, seqIn);
    }
    if (to >= 0) {
        seqOut = ++t.sendSeq[to];
        // the slot's previous chunk must have been consumed
        if (seqOut > SLOTS && !waitValue(s, freedFlag(t.region, to), seqOut - SLOTS)) return false;
        k.out2 = t.pushTo[to] ? inbox(t.regions[to], c->nranks, c->rank, seqOut) : outbox(t.region, c->nranks, to, seqOut);
    }
    if (!cudaOk(ncclDeviceCombine(k, s))) return false;
    if (to >= 0 && !writeValue(s, readyFlag(t.regions[to], c->rank), seqOut)) return false;
    if (from >= 0 && !writeValue(s, freedFlag(t.regions[from], c->rank), seqIn)) return false;
    return true;
}

NcclCombine combineOf(const Operation &op, size_t count) {
    NcclCombine k{};
    k.type = op.type;
    switch (op.op) {
        case ncclProd: k.op = NCCL_DEVICE_PROD; break;
        case ncclMax: k.op = NCCL_DEVICE_MAX; break;
        case ncclMin: k.op = NCCL_DEVICE_MIN; break;
        default: k.op = NCCL_DEVICE_SUM; break;
    }
    k.count = count;
    k.scaled = op.scale != 1.0;
    k.scale = op.scale;
    k.divide = 1;
    return k;
}

/**
 * The rings and chains of an operation on the device transport, on stream s (the transport's, or in graph mode the
 * captured one; the communicator's device is current)
 */
bool deviceCollective(const Operation &op, cudaStream_t s, bool graph) {
    ncclComm_t c = op.comm;
    const int n = c->nranks, r = c->rank, right = (r + 1) % n, left = (r + n - 1) % n;
    const size_t element = typeSize(op.type), chunk = chunkBytes(n) / element;
    const bool average = op.op == ncclAvg;
    auto at = [&](const char *p, size_t elements) { return p + elements * element; };
    auto out = [&](char *p, size_t elements) { return p + elements * element; };
    switch (op.kind) {
        case Kind::ALL_REDUCE:
        case Kind::REDUCE_SCATTER: {
            // segment k of the input; all-reduce: rank r ends with segment r + 1 and passes it on (all-gather),
            // reduce-scatter: with segment r
            const bool all = op.kind == Kind::ALL_REDUCE;
            const size_t seg = all ? (op.count + n - 1) / n : op.count, total = all ? op.count : op.count * n;
            const int shift = all ? 0 : -1;
            auto length = [&](int k, size_t c0) -> size_t {
                const size_t start = size_t(k) * seg + c0, end = std::min(size_t(k + 1) * seg, total);
                return start < end ? std::min(chunk, end - start) : 0;
            };
            for (size_t c0 = 0; c0 < seg; c0 += chunk) {
                const int first = ((r + shift) % n + n) % n;
                if (length(first, c0) > 0) {
                    NcclCombine k = combineOf(op, length(first, c0));
                    k.a = at(op.send, size_t(first) * seg + c0);
                    if (!chunkStep(c, s, -1, right, k, graph)) return false;
                }
                for (int step = 0; step < n - 1; step++) {
                    const int k_seg = ((r + shift - step - 1) % n + n) % n;
                    const size_t len = length(k_seg, c0);
                    if (len == 0) continue;
                    const bool last = step == n - 2;
                    NcclCombine k = combineOf(op, len);
                    k.a = at(op.send, size_t(k_seg) * seg + c0);
                    if (last) {
                        k.out1 = all ? out(op.recv, size_t(k_seg) * seg + c0) : out(op.recv, c0);
                        if (average) k.divide = n;
                    }
                    // all-reduce: the final segment goes on to the next rank (all-gather's first step)
                    if (!chunkStep(c, s, left, !last || all ? right : -1, k, graph)) return false;
                }
                if (!all) continue;
                for (int step = 0; step < n - 1; step++) {
                    const int k_seg = ((r - step) % n + n) % n;
                    const size_t len = length(k_seg, c0);
                    if (len == 0) continue;
                    NcclCombine k = combineOf(op, len);
                    k.scaled = false;
                    k.out1 = out(op.recv, size_t(k_seg) * seg + c0);
                    if (!chunkStep(c, s, left, step < n - 2 ? right : -1, k, graph)) return false;
                }
            }
            return true;
        }
        case Kind::ALL_GATHER: {
            NcclCombine own = combineOf(op, op.count);
            own.scaled = false;
            own.a = op.send;
            own.out1 = out(op.recv, size_t(r) * op.count);
            if (!cudaOk(ncclDeviceCombine(own, s))) return false;
            for (size_t c0 = 0; c0 < op.count; c0 += chunk) {
                const size_t len = std::min(chunk, op.count - c0);
                NcclCombine k = combineOf(op, len);
                k.scaled = false;
                k.a = at(op.send, c0);
                if (!chunkStep(c, s, -1, right, k, graph)) return false;
                for (int step = 0; step < n - 1; step++) {
                    const int k_seg = ((r - step - 1) % n + n) % n;
                    NcclCombine g = combineOf(op, len);
                    g.scaled = false;
                    g.out1 = out(op.recv, size_t(k_seg) * op.count + c0);
                    if (!chunkStep(c, s, left, step < n - 2 ? right : -1, g, graph)) return false;
                }
            }
            return true;
        }
        case Kind::BROADCAST:
        case Kind::REDUCE: {
            // a chain: broadcast from the root on, reduce towards the root
            const bool broadcast = op.kind == Kind::BROADCAST;
            const int pos = broadcast ? (r - op.root + n) % n : (r - op.root - 1 + 2 * n) % n;
            for (size_t c0 = 0; c0 < op.count; c0 += chunk) {
                const size_t len = std::min(chunk, op.count - c0);
                NcclCombine k = combineOf(op, len);
                const int from = pos == 0 ? -1 : left, to = pos == n - 1 ? -1 : right;
                if (broadcast) {
                    k.scaled = false;
                    if (pos == 0) k.a = at(op.send, c0);
                    if (pos != 0 || op.send != op.recv) k.out1 = out(op.recv, c0);
                } else {
                    k.a = at(op.send, c0);
                    if (pos == n - 1) {
                        k.out1 = out(op.recv, c0);
                        if (average) k.divide = n;
                    }
                }
                if (!chunkStep(c, s, from, to, k, graph)) return false;
            }
            return true;
        }
        case Kind::ALL_TO_ALL: {
            NcclCombine own = combineOf(op, op.count);
            own.scaled = false;
            own.a = at(op.send, size_t(r) * op.count);
            own.out1 = out(op.recv, size_t(r) * op.count);
            if (!cudaOk(ncclDeviceCombine(own, s))) return false;
            for (size_t c0 = 0; c0 < op.count; c0 += chunk) {
                const size_t len = std::min(chunk, op.count - c0);
                for (int d = 1; d < n; d++) {
                    const int q = (r + d) % n;
                    NcclCombine k = combineOf(op, len);
                    k.scaled = false;
                    k.a = at(op.send, size_t(q) * op.count + c0);
                    if (!chunkStep(c, s, -1, q, k, graph)) return false;
                }
                for (int d = 1; d < n; d++) {
                    const int p = (r - d + n) % n;
                    NcclCombine k = combineOf(op, len);
                    k.scaled = false;
                    k.out1 = out(op.recv, size_t(p) * op.count + c0);
                    if (!chunkStep(c, s, p, -1, k, graph)) return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

/**
 * Point-to-point operations of one communicator (a group's, in issue order) chunk by chunk, sends and receives
 * interleaved so that ranks sending to each other don't wait for slots the other one frees only later
 */
bool devicePointToPoint(const std::vector<Operation> &ops, cudaStream_t s, bool graph) {
    ncclComm_t c = ops.front().comm;
    size_t chunks = 0;
    const size_t bytes = chunkBytes(c->nranks);
    for (const Operation &op: ops) chunks = std::max(chunks, (op.count * typeSize(op.type) + bytes - 1) / bytes);
    for (size_t k = 0; k < chunks; k++) {
        for (int pass = 0; pass < 2; pass++) {
            for (const Operation &op: ops) {
                if ((op.kind == Kind::SEND) != (pass == 0)) continue;
                const size_t element = typeSize(op.type), per = bytes / element, c0 = k * per;
                if (c0 >= op.count) continue;
                NcclCombine m = combineOf(op, std::min(per, op.count - c0));
                m.scaled = false;
                if (op.kind == Kind::SEND) {
                    m.a = op.send + c0 * element;
                    if (!chunkStep(c, s, -1, op.peer, m, graph)) return false;
                } else {
                    m.out1 = op.recv + c0 * element;
                    if (!chunkStep(c, s, op.peer, -1, m, graph)) return false;
                }
            }
        }
    }
    return true;
}

// whether the stream is capturing (its operations go into a graph: graph mode, if the transport allows it)
bool capturing(cudaStream_t s, bool &yes, const Transport *t = nullptr) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (!cudaOk(cudaStreamIsCapturing(s, &status))) return false;
    yes = status != cudaStreamCaptureStatusNone;
    if (yes && t != nullptr && t->noGraphs != nullptr) {
        t_lastError = t->noGraphs;
        return false;
    }
    return true;
}

// orders the transport's stream after the last operation if that ran directly on its caller's stream
bool catchUp(Transport &t) {
    if (!t.lastDirect) return true;
    // (an event recorded there now would be captured)
    bool captured = false;
    if (!capturing(t.lastStream, captured)) return false;
    if (captured) {
        t_lastError = "an uncaptured operation while the stream of the communicator's last one is capturing";
        return false;
    }
    t.lastDirect = false;
    return cudaOk(cudaEventRecord(t.joined, t.lastStream)) && cudaOk(cudaStreamWaitEvent(t.stream, t.joined, 0));
}

/**
 * Runs work on the communicator's stream after the callers' streams' preceding work, and makes the callers' streams
 * wait for it (as NCCL, which serializes a communicator's operations whatever streams they are issued on)
 */
template<typename Work>
bool onTransportStream(ncclComm_t c, const std::vector<cudaStream_t> &streams, Work work) {
    Transport &t = *c->transport;
    std::lock_guard lock(t.mutex);
    if (!catchUp(t)) return false;
    t.lastStream = streams.size() == 1 ? streams.front() : nullptr;
    for (cudaStream_t s: streams) {
        if (!cudaOk(cudaEventRecord(t.joined, s)) || !cudaOk(cudaStreamWaitEvent(t.stream, t.joined, 0))) return false;
    }
    if (!work()) return false;
    if (!cudaOk(cudaEventRecord(t.done, t.stream))) return false;
    for (cudaStream_t s: streams) {
        if (!cudaOk(cudaStreamWaitEvent(s, t.done, 0))) return false;
    }
    return true;
}

/**
 * Launches a combine with its synchronization (NcclCombine::wait / signal): in the kernel, or as stream memory
 * operations around it. Kernels waiting for their chunk notice it sooner (all-reduces of 1 to 4 MB are 9 to 12%
 * faster), the stream operations keep large all-reduces 1 to 2% faster.
 */
bool launchSynced(NcclCombine k, cudaStream_t s, bool inKernel) {
    if (!inKernel) {
        for (int i = 0; i < 2; i++) {
            if (k.wait[i] != nullptr && !waitValue(s, k.wait[i], k.waitValue[i])) return false;
            k.wait[i] = nullptr;
        }
    }
    uint64_t *signal[2] = {k.signal[0], k.signal[1]};
    if (!inKernel) {
        k.signal[0] = k.signal[1] = nullptr;
        k.finished = nullptr;
    }
    if (!cudaOk(ncclDeviceCombine(k, s))) return false;
    for (int i = 0; i < 2 && !inKernel; i++) {
        if (signal[i] != nullptr && !writeValue(s, signal[i], k.signalValue[i])) return false;
    }
    return true;
}

// all-reduces up to this size synchronize their kernels themselves (launchSynced)
constexpr size_t KERNEL_SYNC_BYTES = size_t(8) << 20;

bool pairCapable(const Operation &op) {
    // (pre-multiplied sums need each rank's own scalar on its own data: rings)
    return op.kind == Kind::ALL_REDUCE && op.comm->transport->uploader >= 0 && op.scale == 1.0;
}

// elements per pair chunk: about 8 chunks (pipelines of small messages are deep enough to keep both directions busy),
// 256 KB to a slot (2 MB)
size_t pairChunk(const Operation &op) {
    const size_t element = typeSize(op.type), bytes = op.count * element;
    const size_t quantum = size_t(64) << 10;
    const size_t chunk = std::clamp((bytes / 8 + quantum - 1) / quantum * quantum, size_t(256) << 10, chunkBytes(2));
    return chunk / element;
}

/**
 * All-reduces between two ranks by exchange: each rank gets the other's data and combines it with its own (a + b on
 * one rank, b + a on the other: the same results). The uploader sends its data chunk by chunk with its copy engine
 * into the other rank's inbox (on its upload stream); the other rank stages its data in its outbox, from where the
 * uploader reads it with a few blocks, combining as it reads, which leaves the copy engine most of the link. Both
 * directions of the link are busy, each with the engine that drives it fastest, and neither waits for the other.
 * The kernels wait for their chunk and announce their progress themselves (a thread spinning on the flag, the last
 * block writing the next one): a GPU's semaphore acquire takes several microseconds longer to notice a flag.
 *
 * The operations of one call (ranks one thread drives) are enqueued a chunk of each in turn: no stream may wait for
 * work that isn't enqueued yet, as NVIDIA's driver blocks the enqueuing thread on a stream full of unsatisfied waits.
 */
bool pairAllReduce(const std::vector<Operation> &ops) {
    std::vector<ncclComm_t> comms;
    for (const Operation &op: ops) comms.push_back(op.comm);
    std::sort(comms.begin(), comms.end());
    comms.erase(std::unique(comms.begin(), comms.end()), comms.end());
    std::vector<std::unique_lock<std::mutex>> locks;
    for (ncclComm_t c: comms) locks.emplace_back(c->transport->mutex);

    auto driven = [](const Operation &op) {
        const size_t bytes = op.count * typeSize(op.type);
        return bytes > DRIVEN_MIN_BYTES && bytes <= drivenExchangeBytes();
    };
    // one kernel per rank (no copy engine): the small and the driven exchanges
    auto small = [&](const Operation &op) {
        return op.count * typeSize(op.type) <= smallExchangeBytes() || driven(op);
    };
    auto chunkOf = [&](const Operation &op) { return driven(op) ? op.count : pairChunk(op); };
    // a single small exchange on the stream of the communicator's last operation runs directly there (that stream
    // waited for the transport's earlier work): no events around it. Larger ones keep the transport's high priority
    // stream: they overlap computations, which would otherwise hold their kernels back.
    std::vector<cudaStream_t> streamOf(ops.size());
    size_t rounds = 0;
    for (size_t i = 0; i < ops.size(); i++) {
        const Operation &op = ops[i];
        Transport &t = *op.comm->transport;
        const size_t chunk = chunkOf(op);
        rounds = std::max(rounds, (op.count + chunk - 1) / chunk);
        if (!cudaOk(cudaSetDevice(op.comm->device))) return false;
        if (ops.size() == 1 && op.count * typeSize(op.type) <= smallExchangeBytes() && op.stream == t.lastStream) {
            streamOf[i] = op.stream;
        } else {
            // the callers' streams' preceding work
            streamOf[i] = t.stream;
            if (!catchUp(t) || !cudaOk(cudaEventRecord(t.joined, op.stream)) ||
                !cudaOk(cudaStreamWaitEvent(t.stream, t.joined, 0))) return false;
        }
        // (large) the upload stream after the main stream's preceding work
        if (!small(op) && (!cudaOk(cudaEventRecord(t.forked, streamOf[i])) ||
                           !cudaOk(cudaStreamWaitEvent(t.upload, t.forked, 0)))) return false;
    }
    for (size_t k = 0; k < rounds; k++) {
        for (size_t i = 0; i < ops.size(); i++) {
            const Operation &op = ops[i];
            ncclComm_t c = op.comm;
            Transport &t = *c->transport;
            const int peer = 1 - c->rank;
            const bool uploader = c->rank == t.uploader;
            const size_t element = typeSize(op.type), chunk = chunkOf(op), c0 = k * chunk;
            if (c0 >= op.count) continue;
            const size_t len = std::min(chunk, op.count - c0);
            const char *mine = op.send + c0 * element;
            if (!cudaOk(cudaSetDevice(c->device))) return false;
            if (small(op) && len * element <= llExchangeBytes()) {
                // tiny: LL stores into the peer's slots, no flags
                NcclLLExchange x{};
                x.type = op.type;
                x.op = combineOf(op, len).op;
                x.count = len;
                x.divide = op.op == ncclAvg ? 2 : 1;
                x.mine = mine;
                x.out = op.recv + c0 * element;
                x.peerSlots = llSlots(t.regions[peer]);
                x.slots = llSlots(t.region);
                x.epoch = llEpoch(t.region);
                if (!cudaOk(ncclDeviceLLExchange(x, streamOf[i]))) return false;
                continue;
            }
            if (driven(op)) {
                // the uploader (whose kernels reach the peer fastest) reads the peer's staged data and writes back
                const int passive = 1 - t.uploader;
                NcclDriven x{};
                x.type = op.type;
                x.op = combineOf(op, len).op;
                x.count = len;
                x.divide = op.op == ncclAvg ? 2 : 1;
                x.driver = uploader;
                x.mine = mine;
                x.out = op.recv + c0 * element;
                x.staging = drivenStaging(t.regions[passive]);
                x.seq = ++t.drivenSeq;
                x.result = drivenInbox(t.regions[passive], x.seq);
                x.staged = drivenStaged(t.regions[t.uploader]);
                x.done = drivenDone(t.regions[passive]);
                x.counter = drivenCounter(t.region);
                if (!cudaOk(ncclDeviceDriven(x, streamOf[i]))) return false;
                continue;
            }
            if (small(op)) {
                // small: one kernel pushes into the peer's inbox, waits for the peer's data and combines
                const uint64_t seqOut = ++t.sendSeq[peer], seqIn = ++t.recvSeq[peer];
                NcclExchange x{};
                x.type = op.type;
                x.op = combineOf(op, len).op;
                x.count = len;
                x.divide = op.op == ncclAvg ? 2 : 1;
                x.mine = mine;
                x.peerInbox = inbox(t.regions[peer], 2, c->rank, seqOut);
                x.inbox = inbox(t.region, 2, peer, seqIn);
                x.out = op.recv + c0 * element;
                if (seqOut > SLOTS) {
                    x.freed = freedFlag(t.region, peer);
                    x.freedValue = seqOut - SLOTS;
                }
                x.peerReady = readyFlag(t.regions[peer], c->rank);
                x.sent = sentFlag(t.region);
                x.sendSeq = seqOut;
                x.ready = readyFlag(t.region, peer);
                x.recvSeq = seqIn;
                x.peerFreed = freedFlag(t.regions[peer], c->rank);
                x.counters = exchangeCounters(t.region);
                if (!cudaOk(ncclDeviceExchange(x, streamOf[i]))) return false;
                continue;
            }
            // send: the uploader's copy engine into the peer's inbox (once the slot is free), or a copy into the own
            // outbox for the peer
            const uint64_t seqOut = ++t.sendSeq[peer], seqIn = ++t.recvSeq[peer], read = ++t.readSeq;
            if (uploader) {
                if ((seqOut > SLOTS && !waitValue(t.upload, freedFlag(t.region, peer), seqOut - SLOTS)) ||
                    !cudaOk(cudaMemcpyAsync(inbox(t.regions[peer], 2, c->rank, seqOut), mine, len * element,
                                            cudaMemcpyDeviceToDevice, t.upload)) ||
                    !writeValue(t.upload, readyFlag(t.regions[peer], c->rank), seqOut) ||
                    !writeValue(t.upload, readFlag(t.region), read)) return false;
            } else {
                NcclCombine stage = combineOf(op, len);
                stage.scaled = false;
                stage.a = mine;
                stage.out1 = outbox(t.region, 2, peer, seqOut);
                if (seqOut > SLOTS) {
                    stage.wait[0] = freedFlag(t.region, peer);
                    stage.waitValue[0] = seqOut - SLOTS;
                }
                stage.signal[0] = readyFlag(t.regions[peer], c->rank);
                stage.signalValue[0] = seqOut;
                stage.signal[1] = readFlag(t.region);
                stage.signalValue[1] = read;
                stage.finished = finishedCounter(t.region, true);
                if (!launchSynced(stage, t.upload, op.count * element <= KERNEL_SYNC_BYTES)) return false;
            }
            // receive and combine: the uploader reads the peer's outbox, the other rank its inbox (in place: once
            // the chunk was read out, as the sum overwrites it)
            NcclCombine sum = combineOf(op, len);
            sum.a = mine;
            sum.b = uploader ? outbox(t.regions[peer], 2, c->rank, seqIn) : inbox(t.region, 2, peer, seqIn);
            sum.out1 = op.recv + c0 * element;
            if (op.op == ncclAvg) sum.divide = 2;
            // (the uploader's reads share the link with its copy engine)
            if (uploader) sum.blocks = 32;
            sum.wait[0] = readyFlag(t.region, peer);
            sum.waitValue[0] = seqIn;
            if (op.send == op.recv) {
                sum.wait[1] = readFlag(t.region);
                sum.waitValue[1] = read;
            }
            sum.signal[0] = freedFlag(t.regions[peer], c->rank);
            sum.signalValue[0] = seqIn;
            sum.finished = finishedCounter(t.region, false);
            if (!launchSynced(sum, streamOf[i], op.count * element <= KERNEL_SYNC_BYTES)) return false;
        }
    }
    for (size_t i = 0; i < ops.size(); i++) {
        const Operation &op = ops[i];
        Transport &t = *op.comm->transport;
        t.lastStream = op.stream;
        t.lastDirect = streamOf[i] == op.stream;
        if (!cudaOk(cudaSetDevice(op.comm->device)) ||
            (!small(op) && (!cudaOk(cudaEventRecord(t.uploaded, t.upload)) ||
                            !cudaOk(cudaStreamWaitEvent(streamOf[i], t.uploaded, 0))))) return false;
        if (t.lastDirect) continue;
        if (!cudaOk(cudaEventRecord(t.done, t.stream)) || !cudaOk(cudaStreamWaitEvent(op.stream, t.done, 0))) {
            return false;
        }
    }
    return true;
}

/**
 * A pair all-reduce in graph mode (pairAllReduce's algorithms; the communicator's device is current): small ones one
 * exchange kernel, larger ones chunks uploaded on a second stream joining the capture (the copy engine between a
 * kernel waiting for the slot and one announcing the chunk, or the staging kernel) and summed on the captured stream.
 * In place, a chunk's sum depends on its upload in the graph.
 */
bool graphPairAllReduce(const Operation &op) {
    ncclComm_t c = op.comm;
    Transport &t = *c->transport;
    std::lock_guard lock(t.mutex);
    const int peer = 1 - c->rank;
    const bool uploader = c->rank == t.uploader;
    const size_t element = typeSize(op.type), chunk = pairChunk(op);
    cudaStream_t s = op.stream, u = t.captureUpload;
    const int op2 = combineOf(op, 0).op, divide = op.op == ncclAvg ? 2 : 1;
    if (op.count * element <= smallExchangeBytes()) {
        const int out = int(t.graphSent[peer]++ % SLOTS), in = int(t.graphReceived[peer]++ % SLOTS);
        NcclExchange x{};
        x.type = op.type;
        x.op = op2;
        x.count = op.count;
        x.divide = divide;
        x.mine = op.send;
        x.peerInbox = graphInbox(t.regions[peer], 2, c->rank, out);
        x.inbox = graphInbox(t.region, 2, peer, in);
        x.out = op.recv;
        x.freed = graphFreedFlag(t.region, peer, out);
        x.freedValue = 1;
        x.peerReady = graphReadyFlag(t.regions[peer], c->rank, out);
        x.sent = graphSentFlag(t.region);
        x.sendSeq = 1;
        x.ready = graphReadyFlag(t.region, peer, in);
        x.recvSeq = 1;
        x.peerFreed = graphFreedFlag(t.regions[peer], c->rank, in);
        x.counters = graphExchangeCounters(t.region);
        x.toggle = true;
        return cudaOk(ncclDeviceExchange(x, s));
    }
    if (!cudaOk(cudaEventRecord(t.captureFork, s)) || !cudaOk(cudaStreamWaitEvent(u, t.captureFork, 0))) return false;
    for (size_t c0 = 0; c0 < op.count; c0 += chunk) {
        const size_t len = std::min(chunk, op.count - c0);
        const char *mine = op.send + c0 * element;
        const int out = int(t.graphSent[peer]++ % SLOTS), in = int(t.graphReceived[peer]++ % SLOTS);
        NcclCombine send = combineOf(op, uploader ? 0 : len);
        send.scaled = false;
        send.toggle = true;
        send.finished = graphFinishedCounter(t.region, true);
        send.wait[0] = graphFreedFlag(t.region, peer, out);
        send.waitValue[0] = 1;
        if (uploader) {
            NcclCombine announce = combineOf(op, 0);
            announce.toggle = true;
            announce.finished = send.finished;
            announce.signal[0] = graphReadyFlag(t.regions[peer], c->rank, out);
            announce.signalValue[0] = 1;
            if (!cudaOk(ncclDeviceCombine(send, u)) ||
                !cudaOk(cudaMemcpyAsync(graphInbox(t.regions[peer], 2, c->rank, out), mine, len * element,
                                        cudaMemcpyDeviceToDevice, u)) ||
                !cudaOk(ncclDeviceCombine(announce, u))) return false;
        } else {
            send.a = mine;
            send.out1 = graphOutbox(t.region, 2, peer, out);
            send.signal[0] = graphReadyFlag(t.regions[peer], c->rank, out);
            send.signalValue[0] = 1;
            if (!cudaOk(ncclDeviceCombine(send, u))) return false;
        }
        if (op.send == op.recv &&
            (!cudaOk(cudaEventRecord(t.captureJoin, u)) || !cudaOk(cudaStreamWaitEvent(s, t.captureJoin, 0)))) {
            return false;
        }
        NcclCombine sum = combineOf(op, len);
        sum.a = mine;
        sum.b = uploader ? graphOutbox(t.regions[peer], 2, c->rank, in) : graphInbox(t.region, 2, peer, in);
        sum.out1 = op.recv + c0 * element;
        sum.divide = divide;
        if (uploader) sum.blocks = 32;
        sum.toggle = true;
        sum.finished = graphFinishedCounter(t.region, false);
        sum.wait[0] = graphReadyFlag(t.region, peer, in);
        sum.waitValue[0] = 1;
        sum.signal[0] = graphFreedFlag(t.regions[peer], c->rank, in);
        sum.signalValue[0] = 1;
        if (!cudaOk(ncclDeviceCombine(sum, s))) return false;
    }
    return cudaOk(cudaEventRecord(t.captureJoin, u)) && cudaOk(cudaStreamWaitEvent(s, t.captureJoin, 0));
}

/**
 * Point-to-point operations of a communicator in graph mode, on the first one's stream (the others' join it there and
 * wait for it)
 */
bool graphPointToPoint(ncclComm_t c, const std::vector<cudaStream_t> &streams, const std::vector<Operation> &ops) {
    Transport &t = *c->transport;
    std::lock_guard lock(t.mutex);
    cudaStream_t s = streams.front();
    for (size_t i = 1; i < streams.size(); i++) {
        if (!cudaOk(cudaEventRecord(t.captureFork, streams[i])) || !cudaOk(cudaStreamWaitEvent(s, t.captureFork, 0))) {
            return false;
        }
    }
    if (!devicePointToPoint(ops, s, true) || !cudaOk(cudaEventRecord(t.captureJoin, s))) return false;
    for (size_t i = 1; i < streams.size(); i++) {
        if (!cudaOk(cudaStreamWaitEvent(streams[i], t.captureJoin, 0))) return false;
    }
    return true;
}

bool deviceCapable(const Operation &op) {
    return op.comm->transport != nullptr && op.comm->nranks > 1 && op.type >= ncclInt8 && op.type <= ncclBfloat16;
}

// runs the operations on their communicators' devices (the current device is restored)
ncclResult_t deviceRun(const std::vector<Operation> &ops) {
    int current = 0;
    cudaGetDevice(&current);
    std::map<ncclComm_t, std::vector<Operation>> pointToPoint;
    bool ok = true;
    std::vector<Operation> pairs;  // consecutive pair all-reduces, enqueued together
    auto flush = [&] {
        ok = ok && (pairs.empty() || pairAllReduce(pairs));
        pairs.clear();
    };
    for (const Operation &op: ops) {
        if (op.kind == Kind::SEND || op.kind == Kind::RECV) {
            pointToPoint[op.comm].push_back(op);
            continue;
        }
        bool graph = false;
        ok = ok && capturing(op.stream, graph, op.comm->transport);
        if (graph) {
            flush();
            ok = ok && cudaOk(cudaSetDevice(op.comm->device));
            if (ok && pairCapable(op)) {
                ok = graphPairAllReduce(op);
            } else if (ok) {
                std::lock_guard lock(op.comm->transport->mutex);
                ok = deviceCollective(op, op.stream, true);
            }
            continue;
        }
        if (pairCapable(op)) {
            pairs.push_back(op);
            continue;
        }
        flush();
        ok = ok && cudaOk(cudaSetDevice(op.comm->device)) &&
             onTransportStream(op.comm, {op.stream},
                               [&] { return deviceCollective(op, op.comm->transport->stream, false); });
    }
    flush();
    for (auto &[comm, list]: pointToPoint) {
        std::vector<cudaStream_t> streams;
        for (const Operation &op: list) {
            if (std::find(streams.begin(), streams.end(), op.stream) == streams.end()) streams.push_back(op.stream);
        }
        bool graph = false;
        ok = ok && cudaOk(cudaSetDevice(comm->device)) && capturing(streams.front(), graph, comm->transport);
        if (ok && graph) {
            ok = graphPointToPoint(comm, streams, list);
            continue;
        }
        ok = ok && onTransportStream(comm, streams,
                                     [&] { return devicePointToPoint(list, comm->transport->stream, false); });
    }
    cudaSetDevice(current);
    return ok ? ncclSuccess : fail(ncclUnhandledCudaError, "device transport: " + t_lastError);
}

// ---- device transport setup (the communicators of this thread together: one, or all ranks of ncclCommInitAll)

void setupBarrier(const std::vector<ncclComm_t> &comms, uint64_t number) {
    for (ncclComm_t c: comms) c->seg->arrived[c->rank].store(number, std::memory_order_release);
    Segment *seg = comms.front()->seg;
    const int n = comms.front()->nranks;
    int spins = 0;
    for (int r = 0; r < n; r++) {
        while (seg->arrived[r].load(std::memory_order_acquire) < number) pause(spins);
    }
}

bool allVoted(Segment *seg, std::atomic<int> *votes, int n) {
    for (int r = 0; r < n; r++) {
        if (votes[r].load(std::memory_order_acquire) != 1) return false;
    }
    return true;
}

void dropTransport(ncclComm_t c) {
    Transport *t = c->transport;
    if (t == nullptr) return;
    int current = 0;
    cudaGetDevice(&current);
    cudaSetDevice(c->device);
    cudaDeviceSynchronize();
    for (size_t r = 0; r < t->regions.size(); r++) {
        if (t->opened[r]) cudaIpcCloseMemHandle(t->regions[r]);
    }
    if (t->region != nullptr) cudaFree(t->region);
    if (t->stream != nullptr) cudaStreamDestroy(t->stream);
    if (t->joined != nullptr) cudaEventDestroy(t->joined);
    if (t->done != nullptr) cudaEventDestroy(t->done);
    if (t->upload != nullptr) cudaStreamDestroy(t->upload);
    if (t->forked != nullptr) cudaEventDestroy(t->forked);
    if (t->uploaded != nullptr) cudaEventDestroy(t->uploaded);
    if (t->captureUpload != nullptr) cudaStreamDestroy(t->captureUpload);
    if (t->captureFork != nullptr) cudaEventDestroy(t->captureFork);
    if (t->captureJoin != nullptr) cudaEventDestroy(t->captureJoin);
    cudaGetLastError();
    cudaSetDevice(current);
    delete t;
    c->transport = nullptr;
}

// GB/s of a copy kernel (bytes from src to dst, three times) on the communicator's device
float measureCopy(ncclComm_t c, const char *src, char *dst, size_t bytes) {
    cudaStream_t s;
    cudaEvent_t start, end;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    NcclCombine k{};
    k.type = ncclFloat32;
    k.count = bytes / 4;
    k.a = src;
    k.out1 = dst;
    k.divide = 1;
    ncclDeviceCombine(k, s);
    cudaEventRecord(start, s);
    for (int i = 0; i < 3; i++) ncclDeviceCombine(k, s);
    cudaEventRecord(end, s);
    float ms = 0;
    const bool ok = cudaEventSynchronize(end) == cudaSuccess && cudaEventElapsedTime(&ms, start, end) == cudaSuccess;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    cudaGetLastError();
    (void) c;
    return ok && ms > 0 ? float(3.0 * bytes / ms / 1e6) : 0;
}

// GB/s of the copy engine (cudaMemcpyAsync, three times) on the communicator's device
float measureCopyEngine(const char *src, char *dst, size_t bytes) {
    cudaStream_t s;
    cudaEvent_t start, end;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
    cudaEventRecord(start, s);
    for (int i = 0; i < 3; i++) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
    cudaEventRecord(end, s);
    float ms = 0;
    const bool ok = cudaEventSynchronize(end) == cudaSuccess && cudaEventElapsedTime(&ms, start, end) == cudaSuccess;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    cudaGetLastError();
    return ok && ms > 0 ? float(3.0 * bytes / ms / 1e6) : 0;
}

/**
 * Sets up the device transport of the communicators (all of them, or none: the ranks vote): each rank allocates its
 * region, shares it (interprocess handle, or its address within a process), opens the others' (peer access needed),
 * and the ranks measure for each edge whether push or pull is faster, one edge at a time
 */
void setupTransports(const std::vector<ncclComm_t> &comms) {
    if (comms.empty() || comms.front()->nranks < 2) return;
    ncclComm_t first = comms.front();
    Segment *seg = first->seg;
    const int n = first->nranks;
    const bool wanted = deviceTransportWanted() && memOpsAvailable();
    int current = 0;
    cudaGetDevice(&current);
    // the setup barriers follow the host transport's (none before)
    uint64_t barrier = 0;
    for (ncclComm_t c: comms) {
        bool ok = wanted && cudaSetDevice(c->device) == cudaSuccess;
        auto *t = new Transport{};
        c->transport = t;
        // the highest priority: the transport's kernels go ahead of computations they overlap with (tensor
        // parallel GEMMs), which would otherwise hold back the chunks of both directions
        int lowest = 0, highest = 0;
        cudaDeviceGetStreamPriorityRange(&lowest, &highest);
        ok = ok && cudaStreamCreateWithPriority(&t->stream, cudaStreamNonBlocking, highest) == cudaSuccess &&
             cudaEventCreateWithFlags(&t->joined, cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&t->done, cudaEventDisableTiming) == cudaSuccess &&
             cudaStreamCreateWithPriority(&t->upload, cudaStreamNonBlocking, highest) == cudaSuccess &&
             cudaEventCreateWithFlags(&t->forked, cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&t->uploaded, cudaEventDisableTiming) == cudaSuccess &&
             cudaStreamCreateWithPriority(&t->captureUpload, cudaStreamNonBlocking, highest) == cudaSuccess &&
             cudaEventCreateWithFlags(&t->captureFork, cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&t->captureJoin, cudaEventDisableTiming) == cudaSuccess;
        cudaDeviceProp prop{};
        ok = ok && cudaGetDeviceProperties(&prop, c->device) == cudaSuccess;
        ok = ok && cudaMalloc(&t->region, regionSize(n)) == cudaSuccess;
        // (graph mode: every slot free)
        const std::vector<uint64_t> ones(size_t(n) * SLOTS, 1);
        ok = ok && cudaMemset(t->region, 0, FLAGS) == cudaSuccess &&
             cudaMemcpy(graphFreedFlag(t->region, 0, 0), ones.data(), ones.size() * 8, cudaMemcpyHostToDevice) ==
                 cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        cudaIpcMemHandle_t handle{};
        ok = ok && cudaIpcGetMemHandle(&handle, t->region) == cudaSuccess;
        cudaGetLastError();
        memcpy(seg->uuid[c->rank], prop.uuid.bytes, 16);
        seg->pid[c->rank] = getpid();
        seg->regionPtr[c->rank] = reinterpret_cast<uint64_t>(t->region);
        memcpy(seg->regionHandle[c->rank], &handle, sizeof(handle));
        seg->vote[c->rank].store(ok ? 1 : 2, std::memory_order_release);
    }
    setupBarrier(comms, ++barrier);
    const bool everyone = allVoted(seg, seg->vote, n);
    for (ncclComm_t c: comms) {
        Transport &t = *c->transport;
        bool ok = everyone && cudaSetDevice(c->device) == cudaSuccess;
        t.regions.assign(n, nullptr);
        t.opened.assign(n, false);
        int devices = 0;
        cudaGetDeviceCount(&devices);
        for (int p = 0; p < n && ok; p++) {
            if (p == c->rank) {
                t.regions[p] = t.region;
                continue;
            }
            // the peer's GPU, as this process numbers it: the same one, or another one with peer access
            int local = -1;
            for (int d = 0; d < devices; d++) {
                cudaDeviceProp prop{};
                if (cudaGetDeviceProperties(&prop, d) == cudaSuccess && memcmp(prop.uuid.bytes, seg->uuid[p], 16) == 0) {
                    local = d;
                }
            }
            int access = local == c->device;
            ok = local >= 0 && (access || (cudaDeviceCanAccessPeer(&access, c->device, local) == cudaSuccess &&
                                           access == 1));
            if (!ok) break;
            if (seg->pid[p] == getpid()) {
                if (local != c->device) {
                    const cudaError_t e = cudaDeviceEnablePeerAccess(local, 0);
                    ok = e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled;
                }
                t.regions[p] = reinterpret_cast<char *>(seg->regionPtr[p]);
            } else {
                cudaIpcMemHandle_t handle;
                memcpy(&handle, seg->regionHandle[p], sizeof(handle));
                void *mapped = nullptr;
                ok = cudaIpcOpenMemHandle(&mapped, handle, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess;
                t.regions[p] = static_cast<char *>(mapped);
                t.opened[p] = ok;
            }
        }
        cudaGetLastError();
        seg->vote2[c->rank].store(ok ? 1 : 2, std::memory_order_release);
    }
    setupBarrier(comms, ++barrier);
    if (!allVoted(seg, seg->vote2, n)) {
        for (ncclComm_t c: comms) dropTransport(c);
        cudaSetDevice(current);
        return;
    }
    // one edge at a time: the sender measures writing into the receiver's inbox, the receiver reading the outbox
    const size_t probe = SLOTS * chunkBytes(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            for (ncclComm_t c: comms) {
                if (c->rank != i) continue;
                cudaSetDevice(c->device);
                Transport &t = *c->transport;
                seg->pushBw[i][j] = measureCopy(c, outbox(t.region, n, j, 1), inbox(t.regions[j], n, i, 1), probe);
                seg->copyBw[i][j] = n == 2 ? measureCopyEngine(outbox(t.region, n, j, 1),
                                                               inbox(t.regions[j], n, i, 1), probe) : 0;
            }
            setupBarrier(comms, ++barrier);
            for (ncclComm_t c: comms) {
                if (c->rank != j) continue;
                cudaSetDevice(c->device);
                Transport &t = *c->transport;
                seg->pullBw[i][j] = measureCopy(c, outbox(t.regions[i], n, j, 1), inbox(t.region, n, i, 1), probe);
            }
            setupBarrier(comms, ++barrier);
        }
    }
    // LIBRECUDA_NCCL_EDGES=push / pull: every edge so (for tests)
    const char *forced = getenv("LIBRECUDA_NCCL_EDGES");
    const bool allPush = forced != nullptr && strcmp(forced, "push") == 0;
    const bool allPull = forced != nullptr && strcmp(forced, "pull") == 0;
    // two ranks on different GPUs: exchange all-reduces, the uploader being the rank whose copy engine sends and whose
    // kernels read the other rank's memory fastest (LIBRECUDA_NCCL_ALGO=ring: rings)
    int uploader = -1;
    const char *algo = getenv("LIBRECUDA_NCCL_ALGO");
    if (n == 2 && memcmp(seg->uuid[0], seg->uuid[1], 16) != 0 && !(algo != nullptr && strcmp(algo, "ring") == 0)) {
        const float rank0 = std::min(seg->copyBw[0][1], seg->pullBw[1][0]);
        const float rank1 = std::min(seg->copyBw[1][0], seg->pullBw[0][1]);
        uploader = rank1 >= rank0 ? 1 : 0;
    }
    // LibreCUDA's runtime (it exports the driver API too) doesn't preempt a kernel for another process's: a graph's
    // kernel waiting for a rank of another process on its GPU would keep that rank's kernel from ever running
    // (all ranks refuse, so that none waits for the others in vain)
    bool shared = false;
    for (int p = 0; p < n; p++) {
        for (int q = 0; q < p; q++) {
            shared = shared || (seg->pid[p] != seg->pid[q] && memcmp(seg->uuid[p], seg->uuid[q], 16) == 0);
        }
    }
    const bool libreRuntime = dlvsym(RTLD_DEFAULT, "cuCtxGetDevice", "libcudart.so.13") != nullptr;
    for (ncclComm_t c: comms) {
        Transport &t = *c->transport;
        if (shared && libreRuntime) {
            t.noGraphs = "operations can't be captured into graphs on LibreCUDA's runtime when ranks of several "
                         "processes share a GPU";
        }
        t.uploader = uploader;
        t.pushTo.assign(n, true);
        t.pullFrom.assign(n, false);
        t.sendSeq.assign(n, 0);
        t.recvSeq.assign(n, 0);
        t.graphSent.assign(n, 0);
        t.graphReceived.assign(n, 0);
        for (int p = 0; p < n; p++) {
            // (pull only when clearly faster)
            t.pushTo[p] = !(seg->pullBw[c->rank][p] > 1.1f * seg->pushBw[c->rank][p]);
            t.pullFrom[p] = seg->pullBw[p][c->rank] > 1.1f * seg->pushBw[p][c->rank];
            if (allPush || allPull) {
                t.pushTo[p] = allPush;
                t.pullFrom[p] = allPull;
            }
            // the pair's edges: the uploader pushes into the other rank's inbox and pulls from its outbox
            if (uploader >= 0) {
                t.pushTo[p] = c->rank == uploader;
                t.pullFrom[p] = c->rank == uploader;
            }
        }
        if (getenv("LIBRECUDA_NCCL_DEBUG") != nullptr) {
            for (int p = 0; p < n; p++) {
                if (p == c->rank) continue;
                fprintf(stderr, "[nccl] rank %d -> %d: push %.2f GB/s, pull %.2f GB/s, copy engine %.2f GB/s: %s%s\n",
                        c->rank, p, seg->pushBw[c->rank][p], seg->pullBw[c->rank][p], seg->copyBw[c->rank][p],
                        t.pushTo[p] ? "push" : "pull", uploader < 0 ? "" : c->rank == uploader ? " (uploader)" : "");
            }
        }
    }
    cudaSetDevice(current);
}

ncclResult_t submit(Operation op) {
    if (!valid(op.comm)) return fail(ncclInvalidArgument, "invalid communicator");
    if (op.type < ncclInt8 || op.type >= ncclFloat8e4m3) return fail(ncclInvalidArgument, "unsupported data type");
    op.scale = 1.0;
    if (op.op >= ncclNumOps) {
        auto it = op.comm->preMulSums.find(int(op.op));
        if (it == op.comm->preMulSums.end()) return fail(ncclInvalidArgument, "unknown reduction operation");
        double scalar = it->second.scalar;
        if (it->second.devicePtr != nullptr) {
            char bytes[8];
            if (!cudaOk(cudaMemcpy(bytes, it->second.devicePtr, typeSize(it->second.type), cudaMemcpyDeviceToHost))) {
                return ncclUnhandledCudaError;
            }
            scalar = loadValue(bytes, 0, it->second.type);
        }
        op.scale = scalar;
        op.op = ncclSum;
    }
    op.offset = 0;
    op.posted = false;
    op.finished = false;
    if (deviceCapable(op)) {
        // collectives are enqueued right away (also in groups); point-to-point operations of a group together
        if (t_groupDepth > 0 && (op.kind == Kind::SEND || op.kind == Kind::RECV)) {
            t_deviceGroup.push_back(op);
            return ncclSuccess;
        }
        return deviceRun({op});
    }
    if (t_groupDepth > 0) {
        t_group.push_back(op);
        return ncclSuccess;
    }
    std::vector<Operation> single{op};
    return run(single);
}

// ---- communicators

std::string randomName() {
    unsigned char bytes[12];
    const int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0 || read(fd, bytes, sizeof(bytes)) != ssize_t(sizeof(bytes))) {
        const uint64_t t = std::chrono::steady_clock::now().time_since_epoch().count() ^ uint64_t(getpid()) << 32;
        memcpy(bytes, &t, 8);
    }
    if (fd >= 0) close(fd);
    std::string name = "/librecuda-nccl-";
    for (unsigned char b: bytes) {
        static const char hex[] = "0123456789abcdef";
        name += hex[b >> 4];
        name += hex[b & 15];
    }
    return name;
}

// waits until all ranks joined the segment (the communicator is then complete)
void joinBarrier(ncclComm_t c) {
    int spins = 0;
    while (c->seg->joined.load(std::memory_order_acquire) < c->nranks) pause(spins);
    c->joined = true;
}

ncclResult_t attach(ncclComm_t *out, int nranks, const std::string &name, int rank) {
    if (out == nullptr || nranks < 1 || nranks > MAX_RANKS || rank < 0 || rank >= nranks) {
        return fail(ncclInvalidArgument, "invalid rank or rank count (at most 64 ranks)");
    }
    const size_t size = segmentSize(nranks);
    const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) return fail(ncclSystemError, "shm_open " + name + ": " + strerror(errno));
    if (ftruncate(fd, off_t(size)) != 0) {
        close(fd);
        return fail(ncclSystemError, std::string("ftruncate: ") + strerror(errno));
    }
    void *base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return fail(ncclSystemError, std::string("mmap: ") + strerror(errno));
    int device = 0;
    cudaGetDevice(&device);
    auto *c = new ncclComm{COMM_MAGIC, rank, nranks, device, name, static_cast<Segment *>(base),
                           static_cast<char *>(base), size, 0, std::vector<uint64_t>(nranks),
                           std::vector<uint64_t>(nranks), 0, false, {}, ncclNumOps, nullptr};
    // the last rank to join removes the name (the segment stays until all ranks unmapped it, also if they crash)
    if (c->seg->joined.fetch_add(1, std::memory_order_acq_rel) + 1 == nranks) {
        shm_unlink(name.c_str());
    }
    *out = c;
    // inside a group, the ranks of this thread join together at the end
    if (t_groupDepth > 0) {
        t_pendingJoins.push_back(c);
    } else {
        joinBarrier(c);
        setupTransports({c});
    }
    return ncclSuccess;
}

std::string nameOf(const ncclUniqueId &id) {
    return std::string(id.internal, strnlen(id.internal, sizeof(id.internal)));
}

void release(ncclComm_t c) {
    dropTransport(c);
    munmap(c->base, c->size);
    c->magic = 0;
    delete c;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// the API

ncclResult_t ncclGetVersion(int *version) {
    if (version == nullptr) return ncclInvalidArgument;
    *version = NCCL_VERSION_CODE;
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
    if (uniqueId == nullptr) return ncclInvalidArgument;
    memset(uniqueId, 0, sizeof(*uniqueId));
    const std::string name = randomName();
    memcpy(uniqueId->internal, name.c_str(), std::min(name.size(), sizeof(uniqueId->internal) - 1));
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
    return attach(comm, nranks, nameOf(commId), rank);
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *) {
    // (non-blocking configurations complete immediately too)
    return attach(comm, nranks, nameOf(commId), rank);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks, int myrank, int nId, ncclUniqueId *commIds,
                                      ncclConfig_t *) {
    if (nId < 1 || commIds == nullptr) return ncclInvalidArgument;
    return attach(newcomm, nranks, nameOf(commIds[0]), myrank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
    if (comm == nullptr || ndev < 1) return ncclInvalidArgument;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    int current = 0;
    cudaGetDevice(&current);
    ncclGroupStart();
    for (int i = 0; i < ndev; i++) {
        if (cudaSetDevice(devlist != nullptr ? devlist[i] : i) != cudaSuccess) {
            cudaGetLastError();
            ncclGroupEnd();
            return ncclUnhandledCudaError;
        }
        const ncclResult_t result = attach(&comm[i], ndev, nameOf(id), i);
        if (result != ncclSuccess) {
            ncclGroupEnd();
            return result;
        }
    }
    const ncclResult_t result = ncclGroupEnd();
    cudaSetDevice(current);
    return result;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    return valid(comm) ? ncclSuccess : ncclInvalidArgument;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!valid(comm)) return ncclInvalidArgument;
    release(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    return ncclCommDestroy(comm);
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *) {
    if (!valid(comm) || newcomm == nullptr) return ncclInvalidArgument;
    // exchange (color, key) through an all-gather of the parent
    int *pair = nullptr;
    if (cudaMalloc(&pair, 2 * sizeof(int) * (comm->nranks + 1)) != cudaSuccess) {
        cudaGetLastError();
        return ncclUnhandledCudaError;
    }
    const int mine[2] = {color, key};
    cudaMemcpy(pair, mine, sizeof(mine), cudaMemcpyHostToDevice);
    Operation op{Kind::ALL_GATHER, comm, reinterpret_cast<const char *>(pair), reinterpret_cast<char *>(pair + 2), 2,
                 ncclInt32, ncclSum, 0, 0, nullptr};
    const int depth = t_groupDepth;
    t_groupDepth = 0;
    ncclResult_t result = submit(op);
    t_groupDepth = depth;
    std::vector<int> all(2 * comm->nranks);
    cudaMemcpy(all.data(), pair + 2, all.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(pair);
    if (result != ncclSuccess) return result;
    const uint64_t split = ++comm->splits;
    if (color == NCCL_SPLIT_NOCOLOR) {
        *newcomm = nullptr;
        return ncclSuccess;
    }
    // ranks of this color ordered by (key, parent rank)
    int nranks = 0, rank = 0;
    for (int r = 0; r < comm->nranks; r++) {
        if (all[2 * r] != color) continue;
        nranks++;
        if (all[2 * r + 1] < key || (all[2 * r + 1] == key && r < comm->rank)) rank++;
    }
    const std::string name = comm->name.substr(0, 40) + "-s" + std::to_string(split) + "c" + std::to_string(color);
    return attach(newcomm, nranks, name, rank);
}

ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm,
                            ncclConfig_t *, int) {
    if (!valid(comm) || newcomm == nullptr || (excludeRanksCount > 0 && excludeRanksList == nullptr)) {
        return ncclInvalidArgument;
    }
    int below = 0;
    for (int i = 0; i < excludeRanksCount; i++) {
        if (excludeRanksList[i] == comm->rank) return ncclInvalidArgument;
        below += excludeRanksList[i] < comm->rank;
    }
    const uint64_t split = ++comm->splits;
    const std::string name = comm->name.substr(0, 40) + "-x" + std::to_string(split);
    return attach(newcomm, comm->nranks - excludeRanksCount, name, comm->rank - below);
}

const char *ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "no error";
        case ncclUnhandledCudaError: return "unhandled cuda error";
        case ncclSystemError: return "unhandled system error";
        case ncclInternalError: return "internal error";
        case ncclInvalidArgument: return "invalid argument";
        case ncclInvalidUsage: return "invalid usage";
        case ncclRemoteError: return "remote process exited or there was a network error";
        case ncclInProgress: return "NCCL operation in progress";
        default: return "unknown result code";
    }
}

const char *ncclGetLastError(ncclComm_t) {
    return t_lastError.c_str();
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
    if (!valid(comm) || asyncError == nullptr) return ncclInvalidArgument;
    *asyncError = ncclSuccess;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
    if (!valid(comm) || count == nullptr) return ncclInvalidArgument;
    *count = comm->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
    if (!valid(comm) || device == nullptr) return ncclInvalidArgument;
    *device = comm->device;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
    if (!valid(comm) || rank == nullptr) return ncclInvalidArgument;
    *rank = comm->rank;
    return ncclSuccess;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t, void **handle) {
    if (!valid(comm) || handle == nullptr) return ncclInvalidArgument;
    // (buffers need no registration: data goes through host memory)
    *handle = buff;
    return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *) {
    return valid(comm) ? ncclSuccess : ncclInvalidArgument;
}

// (ncclMemAlloc, ncclMemFree and the symmetric memory windows: nccl_window.cu)

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
    if (!valid(comm) || op == nullptr || scalar == nullptr) return ncclInvalidArgument;
    PreMulSum p{1.0, nullptr, datatype};
    if (residence == ncclScalarDevice) {
        p.devicePtr = scalar;
    } else {
        p.scalar = loadValue(static_cast<const char *>(scalar), 0, datatype);
    }
    const int id = comm->nextOp++;
    comm->preMulSums[id] = p;
    *op = static_cast<ncclRedOp_t>(id);
    return ncclSuccess;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
    if (!valid(comm)) return ncclInvalidArgument;
    return comm->preMulSums.erase(int(op)) ? ncclSuccess : ncclInvalidArgument;
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    return submit({Kind::ALL_REDUCE, comm, static_cast<const char *>(sendbuff), static_cast<char *>(recvbuff), count,
                   datatype, op, 0, 0, stream});
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream) {
    if (valid(comm) && (root < 0 || root >= comm->nranks)) return ncclInvalidArgument;
    return submit({Kind::BROADCAST, comm, static_cast<const char *>(sendbuff), static_cast<char *>(recvbuff), count,
                   datatype, ncclSum, root, 0, stream});
}

ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
    return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                        int root, ncclComm_t comm, cudaStream_t stream) {
    if (valid(comm) && (root < 0 || root >= comm->nranks)) return ncclInvalidArgument;
    return submit({Kind::REDUCE, comm, static_cast<const char *>(sendbuff), static_cast<char *>(recvbuff), count,
                   datatype, op, root, 0, stream});
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    return submit({Kind::REDUCE_SCATTER, comm, static_cast<const char *>(sendbuff), static_cast<char *>(recvbuff),
                   recvcount, datatype, op, 0, 0, stream});
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
    return submit({Kind::ALL_GATHER, comm, static_cast<const char *>(sendbuff), static_cast<char *>(recvbuff),
                   sendcount, datatype, ncclSum, 0, 0, stream});
}

ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                          ncclComm_t comm, cudaStream_t stream) {
    return submit({Kind::ALL_TO_ALL, comm, static_cast<const char *>(sendbuff), static_cast<char *>(recvbuff), count,
                   datatype, ncclSum, 0, 0, stream});
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
    if (valid(comm) && (peer < 0 || peer >= comm->nranks)) return ncclInvalidArgument;
    return submit({Kind::SEND, comm, static_cast<const char *>(sendbuff), nullptr, count, datatype, ncclSum, 0, peer,
                   stream});
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
    if (valid(comm) && (peer < 0 || peer >= comm->nranks)) return ncclInvalidArgument;
    return submit({Kind::RECV, comm, nullptr, static_cast<char *>(recvbuff), count, datatype, ncclSum, 0, peer,
                   stream});
}

ncclResult_t ncclGroupStart() {
    t_groupDepth++;
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
    if (t_groupDepth == 0) return ncclInvalidUsage;
    if (--t_groupDepth > 0) return ncclSuccess;
    for (ncclComm_t c: t_pendingJoins) joinBarrier(c);
    // (the ranks of this thread's new communicators set up their transports together, per communicator)
    std::map<std::string, std::vector<ncclComm_t>> joined;
    for (ncclComm_t c: t_pendingJoins) joined[c->name].push_back(c);
    t_pendingJoins.clear();
    for (auto &[name, comms]: joined) setupTransports(comms);
    std::vector<Operation> deviceOps;
    deviceOps.swap(t_deviceGroup);
    if (!deviceOps.empty()) {
        const ncclResult_t result = deviceRun(deviceOps);
        if (result != ncclSuccess) return result;
    }
    std::vector<Operation> ops;
    ops.swap(t_group);
    return run(ops);
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *) {
    return ncclSuccess;
}

// (the device communicators of nccl_device.h: nccl_window.cu)
