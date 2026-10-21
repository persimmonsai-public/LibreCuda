// Device side runtime for LibreCUDA: printf, assert, malloc and free.
//
// nvcc compiles printf(fmt, ...) into a call of the external function vprintf(fmt, args), where args points to the
// arguments packed like a struct (each aligned to its size; float promoted to double). The LibreCUDA loader resolves
// vprintf to librecuda_vprintf below. It copies the format string, the arguments and the contents of %s strings into
// a record in the printf buffer (pinned host memory); the host formats and prints the records at synchronization
// points.
//
// Record layout (8 byte aligned):
//   uint32 size        total record size in bytes; written last, 0 while the record is incomplete
//   uint32 fmt_size    size of the format string including the terminating 0
//   format string      padded to 8 bytes
//   arguments          one 8 byte slot per argument (integers zero extended, doubles, pointers);
//                      %s: uint32 size (including the 0; 0 for a null pointer) followed by the bytes, padded to 8
//
// A failed assert writes a record with fmt_size ASSERT_RECORD instead:
//   uint32 size, uint32 ASSERT_RECORD, uint32 line, uint32 blockIdx.xyz, uint32 threadIdx.xyz, padding to 40 bytes,
//   then the assertion, file and function as strings (uint32 size followed by the bytes, padded to 8)
// and stops the kernel with a trap.
//
// Compiled with -rdc=true: the caller reaches it with CALL.ABS and an absolute return address, and only relocatable
// compilation makes ptxas emit the matching absolute return (RET.ABS; whole program compilation emits RET.REL).
//
// The functions run as part of the calling kernel, so it must not use the constant banks the toolchain manages:
// constant bank 4 (addresses of globals, string literals) would be the caller's. It finds its state through %envreg8 /
// %envreg9, which LibreCUDA writes into the driver area of constant bank 0 for every launch, and uses no string
// literals. It must also fit the caller's registers (compiled with -maxrregcount=24, the minimum ptxas gives a kernel
// calling vprintf; the loader raises callers to this function's register count) and the default 1 KiB stack.

#include <cstdint>

// lives in device memory (atomics on host memory would need PCIe atomics)
struct DeviceRuntimeState {
    // pinned host buffer receiving the records, and its size in bytes
    unsigned long long buffer;
    unsigned int capacity;
    // bytes allocated in the buffer so far; reset by the host when the context is idle
    unsigned int offset;
    // records that did not fit
    unsigned int dropped;
    unsigned int padding;

    // malloc heap in device memory (0 until a module calling malloc / free is loaded)
    unsigned long long heap_base;
    unsigned long long heap_size;
    // bytes of the heap carved into blocks so far
    unsigned long long heap_top;
    // free lists per size class and shard: (offset of the first free block) | (version tag << HEAP_OFFSET_BITS)
    unsigned long long free_lists[32][32];
};

static __device__ __forceinline__ DeviceRuntimeState *deviceRuntimeState() {
    unsigned int hi, lo;
    asm volatile("mov.u32 %0, %%envreg8;" : "=r"(hi));
    asm volatile("mov.u32 %0, %%envreg9;" : "=r"(lo));
    return reinterpret_cast<DeviceRuntimeState *>((static_cast<unsigned long long>(hi) << 32) | lo);
}

// fmt_size of an assert record (see above)
static constexpr unsigned int ASSERT_RECORD = 0xffffffffu;

static __device__ __forceinline__ unsigned int align8(unsigned int n) {
    return (n + 7u) & ~7u;
}

// a null string is recorded with size 0 (no string literal for "(null)" here, see above)
static __device__ __forceinline__ unsigned int stringSize(const char *s) {
    if (s == nullptr) {
        return 0;
    }
    unsigned int n = 0;
    while (s[n] != 0) {
        n++;
    }
    return n + 1;
}

// Walks the conversion specifications of fmt. For each argument consumed (including '*' widths/precisions) calls
// visit(kind, arg_ptr) with kind 0 = 4 byte integer, 1 = 8 byte value, 2 = string pointer. Returns the number of
// bytes the arguments occupy in the record (or 0 when visit only writes).
template<typename F>
static __device__ __forceinline__ void forEachArgument(const char *fmt, const char *args, F visit) {
    unsigned int arg_off = 0;
    auto take = [&](unsigned int size, int kind) {
        arg_off = (arg_off + size - 1) & ~(size - 1);
        visit(kind, args + arg_off);
        arg_off += size;
    };
    for (const char *p = fmt; *p != 0; p++) {
        if (*p != '%') {
            continue;
        }
        p++;
        if (*p == '%') {
            continue;
        }
        // flags
        while (*p == '-' || *p == '+' || *p == ' ' || *p == '#' || *p == '0' || *p == '\'') {
            p++;
        }
        // width
        if (*p == '*') {
            take(4, 0);
            p++;
        } else {
            while (*p >= '0' && *p <= '9') {
                p++;
            }
        }
        // precision
        if (*p == '.') {
            p++;
            if (*p == '*') {
                take(4, 0);
                p++;
            } else {
                while (*p >= '0' && *p <= '9') {
                    p++;
                }
            }
        }
        // length
        bool wide = false;
        while (*p == 'h' || *p == 'l' || *p == 'L' || *p == 'z' || *p == 'j' || *p == 't' || *p == 'q') {
            if (*p != 'h') {
                wide = true;
            }
            p++;
        }
        switch (*p) {
            case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': case 'c':
                take(wide ? 8 : 4, wide ? 1 : 0);
                break;
            case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A': case 'p': case 'n':
                take(8, 1);
                break;
            case 's':
                take(8, 2);
                break;
            case 0:
                return;
            default:
                break;
        }
    }
}

// Allocates a record of size bytes in the buffer. Returns nullptr if it doesn't fit (counted as dropped).
static __device__ __forceinline__ char *allocateRecord(DeviceRuntimeState *state, unsigned int size) {
    const unsigned int offset = atomicAdd(&state->offset, size);
    if (offset + size > state->capacity) {
        atomicAdd(&state->dropped, 1u);
        return nullptr;
    }
    return reinterpret_cast<char *>(state->buffer) + offset;
}

// Writes a string (uint32 size, bytes, padding) at record + w; returns the new write offset
static __device__ __forceinline__ unsigned int writeString(char *record, unsigned int w, const char *s) {
    const unsigned int n = stringSize(s);
    *reinterpret_cast<unsigned int *>(record + w) = n;
    for (unsigned int i = 0; i < n; i++) {
        record[w + 4 + i] = s[i];
    }
    return w + align8(4 + n);
}

// Makes a completely written record visible to the host: the size must become visible after the body
static __device__ __forceinline__ void publishRecord(char *record, unsigned int size) {
    __threadfence_system();
    *reinterpret_cast<volatile unsigned int *>(record) = size;
}

extern "C" __device__ __noinline__ int librecuda_vprintf(const char *fmt, const char *args) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (state == nullptr || state->buffer == 0 || fmt == nullptr) {
        return -1;
    }
    const unsigned int fmt_size = stringSize(fmt);

    // size of the record
    unsigned int size = 8 + align8(fmt_size);
    forEachArgument(fmt, args, [&](int kind, const char *arg) {
        size += kind == 2 ? align8(4 + stringSize(*reinterpret_cast<const char *const *>(arg))) : 8;
    });

    char *record = allocateRecord(state, size);
    if (record == nullptr) {
        return -1;
    }

    // body
    reinterpret_cast<unsigned int *>(record)[1] = fmt_size;
    unsigned int w = 8;
    for (unsigned int i = 0; i < fmt_size; i++) {
        record[w + i] = fmt[i];
    }
    w += align8(fmt_size);
    forEachArgument(fmt, args, [&](int kind, const char *arg) {
        if (kind == 2) {
            w = writeString(record, w, *reinterpret_cast<const char *const *>(arg));
        } else {
            *reinterpret_cast<unsigned long long *>(record + w) =
                kind == 0 ? *reinterpret_cast<const unsigned int *>(arg)
                          : *reinterpret_cast<const unsigned long long *>(arg);
            w += 8;
        }
    });

    publishRecord(record, size);
    return static_cast<int>(fmt_size - 1);
}

// Called by assert() when the assertion fails: records the message for the host and stops the kernel
extern "C" __device__ __noinline__ void librecuda_assertfail(const char *assertion, const char *file, unsigned int line,
                                                            const char *function, unsigned long long) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (state != nullptr && state->buffer != 0) {
        const unsigned int size = 40 + align8(4 + stringSize(assertion)) + align8(4 + stringSize(file)) +
                                  align8(4 + stringSize(function));
        char *record = allocateRecord(state, size);
        if (record != nullptr) {
            unsigned int *header = reinterpret_cast<unsigned int *>(record);
            header[1] = ASSERT_RECORD;
            header[2] = line;
            header[3] = blockIdx.x;
            header[4] = blockIdx.y;
            header[5] = blockIdx.z;
            header[6] = threadIdx.x;
            header[7] = threadIdx.y;
            header[8] = threadIdx.z;
            unsigned int w = 40;
            w = writeString(record, w, assertion);
            w = writeString(record, w, file);
            writeString(record, w, function);
            publishRecord(record, size);
        }
    }
    __trap();
}

// malloc / free
//
// Blocks are powers of two from 32 bytes (size class 0) up and start with a 16 byte header holding the size class, so
// the returned pointers are 16 byte aligned. A freed block goes onto a free list of its class (a lock-free stack; the
// next pointer is kept where the user data was) and is reused by later allocations of that class; otherwise blocks are
// carved from the heap with a bump pointer. Freed blocks are not merged, so memory freed in small blocks can't serve
// larger allocations.
//
// To keep threads from contending for one list head, every class has HEAP_SHARDS lists and a thread uses the one of
// its SM; only when the heap is exhausted does malloc look at the other shards. The list heads carry a version tag
// that changes with every push, so a pop can't be fooled by a block that was popped and pushed again meanwhile (ABA).

static constexpr unsigned int HEAP_MIN_BLOCK_LOG2 = 5;
static constexpr unsigned int HEAP_CLASSES = 32;
static constexpr unsigned int HEAP_SHARDS = 32;
static constexpr unsigned int HEAP_HEADER_SIZE = 16;
static constexpr unsigned int HEAP_OFFSET_BITS = 40;
static constexpr unsigned long long HEAP_OFFSET_MASK = (1ull << HEAP_OFFSET_BITS) - 1;
// offset 0 is never a block (the host starts the bump pointer at 256), so it marks an empty list

static __device__ __forceinline__ unsigned int heapShard() {
    unsigned int sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    return sm % HEAP_SHARDS;
}

// waits a little after a failed compare-and-swap, longer the more often it failed
static __device__ __forceinline__ void backoff(unsigned int &attempt) {
    __nanosleep(32u << min(attempt, 6u));
    attempt++;
}

// pops a block from a free list; returns its offset or 0
static __device__ __forceinline__ unsigned long long popBlock(char *heap, unsigned long long *list) {
    unsigned long long head = *reinterpret_cast<volatile unsigned long long *>(list);
    unsigned int attempt = 0;
    while ((head & HEAP_OFFSET_MASK) != 0) {
        const unsigned long long offset = head & HEAP_OFFSET_MASK;
        __threadfence();
        const unsigned long long next = *reinterpret_cast<volatile unsigned long long *>(heap + offset +
                                                                                         HEAP_HEADER_SIZE);
        const unsigned long long replacement = (next & HEAP_OFFSET_MASK) | (head & ~HEAP_OFFSET_MASK);
        const unsigned long long seen = atomicCAS(list, head, replacement);
        if (seen == head) {
            __threadfence();
            return offset;
        }
        head = seen;
        backoff(attempt);
    }
    return 0;
}

extern "C" __device__ __noinline__ void *librecuda_malloc(unsigned long long size) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (state == nullptr || state->heap_base == 0 || size == 0 || size > state->heap_size) {
        return nullptr;
    }
    // smallest class whose blocks hold the header and the data
    const unsigned long long needed = size + HEAP_HEADER_SIZE;
    unsigned int cls = 0;
    while ((1ull << (cls + HEAP_MIN_BLOCK_LOG2)) < needed) {
        cls++;
    }
    if (cls >= HEAP_CLASSES) {
        return nullptr;
    }
    const unsigned long long block_size = 1ull << (cls + HEAP_MIN_BLOCK_LOG2);
    char *heap = reinterpret_cast<char *>(state->heap_base);
    const unsigned int shard = heapShard();

    // reuse a block freed on this SM
    unsigned long long offset = popBlock(heap, &state->free_lists[cls][shard]);
    if (offset != 0) {
        return heap + offset + HEAP_HEADER_SIZE;
    }

    // carve a new block (aligned to its size, so that blocks never straddle a larger alignment)
    unsigned long long top = *reinterpret_cast<volatile unsigned long long *>(&state->heap_top);
    unsigned int attempt = 0;
    for (;;) {
        offset = (top + block_size - 1) & ~(block_size - 1);
        if (offset + block_size > state->heap_size) {
            break;
        }
        const unsigned long long seen = atomicCAS(&state->heap_top, top, offset + block_size);
        if (seen == top) {
            *reinterpret_cast<unsigned int *>(heap + offset) = cls;
            __threadfence();
            return heap + offset + HEAP_HEADER_SIZE;
        }
        top = seen;
        backoff(attempt);
    }

    // the heap is exhausted: take a block freed on another SM
    for (unsigned int i = 1; i < HEAP_SHARDS; i++) {
        offset = popBlock(heap, &state->free_lists[cls][(shard + i) % HEAP_SHARDS]);
        if (offset != 0) {
            return heap + offset + HEAP_HEADER_SIZE;
        }
    }
    return nullptr;
}

extern "C" __device__ __noinline__ void librecuda_free(void *pointer) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (pointer == nullptr || state == nullptr || state->heap_base == 0) {
        return;
    }
    char *heap = reinterpret_cast<char *>(state->heap_base);
    char *block = static_cast<char *>(pointer) - HEAP_HEADER_SIZE;
    const unsigned long long offset = block - heap;
    if (offset >= state->heap_size) {
        return; // not a heap pointer
    }
    const unsigned int cls = *reinterpret_cast<volatile unsigned int *>(block);
    if (cls >= HEAP_CLASSES) {
        return;
    }
    unsigned long long *list = &state->free_lists[cls][heapShard()];
    unsigned long long head = *reinterpret_cast<volatile unsigned long long *>(list);
    unsigned int attempt = 0;
    for (;;) {
        *reinterpret_cast<volatile unsigned long long *>(block + HEAP_HEADER_SIZE) = head & HEAP_OFFSET_MASK;
        __threadfence();
        // new version tag: the head changes even if the same block comes back
        const unsigned long long replacement = offset | ((head & ~HEAP_OFFSET_MASK) + (1ull << HEAP_OFFSET_BITS));
        const unsigned long long seen = atomicCAS(list, head, replacement);
        if (seen == head) {
            return;
        }
        head = seen;
        backoff(attempt);
    }
}
