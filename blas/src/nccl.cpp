/**
 * LibreCUDA's NCCL (libnccl.so.2): collectives (all-reduce, broadcast, reduce, all-gather, reduce-scatter, all-to-all)
 * and point-to-point send / receive between ranks in one or several processes.
 *
 * The transport is host shared memory (POSIX shm, named by the communicator's unique id): every rank has a data slot
 * and every ordered pair of ranks a mailbox. A collective proceeds in chunks: each rank copies its chunk from the GPU
 * into its slot and announces it (posted counter); once all ranks posted, each computes its result from the slots
 * (reductions on the CPU) and copies it to the GPU, then announces that it is done with the slots (done counter); a
 * slot is reused once all ranks are done. Operations are non-blocking steps of a progress loop, so the operations of a
 * group (several ranks in one process, matching sends and receives) advance together. The calls return when the
 * results are on the GPU (the stream's earlier work is waited for first).
 *
 * This needs no GPU peer access or interprocess GPU memory; the bandwidth is that of host copies.
 */
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
                           std::vector<uint64_t>(nranks), 0, false, {}, ncclNumOps};
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
    }
    return ncclSuccess;
}

std::string nameOf(const ncclUniqueId &id) {
    return std::string(id.internal, strnlen(id.internal, sizeof(id.internal)));
}

void release(ncclComm_t c) {
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

ncclResult_t ncclCommWindowRegister(ncclComm_t, void *, size_t, ncclWindow_t *, int) {
    return fail(ncclInvalidUsage, "symmetric memory windows are not supported");
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) {
    return fail(ncclInvalidUsage, "symmetric memory windows are not supported");
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
    if (ptr == nullptr) return ncclInvalidArgument;
    if (cudaMalloc(ptr, size) != cudaSuccess) {
        cudaGetLastError();
        return ncclUnhandledCudaError;
    }
    return ncclSuccess;
}

ncclResult_t ncclMemFree(void *ptr) {
    return cudaFree(ptr) == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

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
    t_pendingJoins.clear();
    std::vector<Operation> ops;
    ops.swap(t_group);
    return run(ops);
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *) {
    return ncclSuccess;
}

// (declared in nccl_device.h: device communicators are not supported)
extern "C" __attribute__((visibility("default"))) ncclResult_t ncclDevCommCreate(...) {
    return ncclInvalidUsage;
}

extern "C" __attribute__((visibility("default"))) ncclResult_t ncclDevCommDestroy(...) {
    return ncclInvalidUsage;
}
