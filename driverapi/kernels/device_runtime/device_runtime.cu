// Device side runtime for LibreCUDA: printf, assert, malloc and free, and device side kernel launches.
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

    // device side kernel launches (0 until a module launching kernels is loaded): queue of launch slots in pinned host
    // memory, its capacity in slots and the slots reserved so far (reset by the host)
    unsigned long long launch_queue;
    unsigned int launch_capacity;
    unsigned int launch_count;
    // cudaGetLastError on the device of threads without an error slot
    unsigned int last_error;
    int device;
    int device_count;
    unsigned int padding2;

    // per thread errors (ErrorSlot), by hardware thread slot: error_slot_sms x error_slot_warps x 32 lanes
    unsigned long long error_slots;
    unsigned int error_slot_sms;
    unsigned int error_slot_warps;

    // device attributes (attribute_count ints per device, indexed by cudaDeviceAttr) and limits (by cudaLimit)
    unsigned long long attributes;
    unsigned int attribute_count;
    unsigned int padding3;
    unsigned long long limits[8];
    // attributes of the context's kernels (FunctionRecord)
    unsigned long long functions;
    unsigned int function_count;
    unsigned int function_capacity;
};

// a kernel's attributes for device side cudaFuncGetAttributes / occupancy (filled by the host, printf.cpp)
struct FunctionRecord {
    unsigned long long address;
    int shared_size;
    int const_size;
    int local_size;
    int max_threads;
    int registers;
    int version;
    int max_dynamic_shared;
    int cluster_must_be_set;
    int cluster[3];
    int padding;
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

// Device side kernel launches (dynamic parallelism)
//
// For child<<<grid, block, shmem, stream>>>(args) in device code nvcc emits
//   buffer = __cudaCDP2GetParameterBufferV2(child, grid, block, shmem); (skipped if buffer is null:)
//   <arguments stored into buffer at their parameter offsets>; __cudaCDP2LaunchDeviceV2(buffer, stream)
// (declared in cuda_device_runtime_api.h). The loader redirects these calls to the functions below even when NVIDIA's
// libcudadevrt is linked into the image. A launch reserves a slot in the launch queue; the host launches the children
// after the parent completed (driverapi/src/device_launch.cpp).
//
// Slot layout: LaunchSlot (64 bytes) followed by LAUNCH_PARAM_SIZE bytes of parameters. Device side memcpy / memset
// (cudaMemcpyAsync, ...) are slots too (operation COPY / SET, a CopyOp in the parameter area), executed by the host in
// order with the launches. Device side events are no-ops: all device side streams are one ordered stream.

static constexpr unsigned int LAUNCH_HEADER_SIZE = 64;
static constexpr unsigned int LAUNCH_PARAM_SIZE = 4096;
static constexpr unsigned int LAUNCH_SLOT_SIZE = LAUNCH_HEADER_SIZE + LAUNCH_PARAM_SIZE;

// LaunchSlot::status
static constexpr unsigned int LAUNCH_RESERVED = 0;
static constexpr unsigned int LAUNCH_LAUNCHED = 1;
static constexpr unsigned int LAUNCH_TAIL = 2;

// LaunchSlot::operation
static constexpr unsigned int OPERATION_KERNEL = 0;
static constexpr unsigned int OPERATION_COPY = 1;
static constexpr unsigned int OPERATION_SET = 2;
// a device graph launch: the parameters are the executable graph and the stream (cudaStreamGraph*)
static constexpr unsigned int OPERATION_GRAPH = 3;
// a device side cudaMemcpy3DAsync involving CUDA arrays (the parameters): run by the host runtime
static constexpr unsigned int OPERATION_COPY_3D = 4;

struct LaunchSlot {
    unsigned long long function;
    unsigned int grid[3];
    unsigned int block[3];
    unsigned int shared_mem;
    unsigned int status;
    unsigned int operation;
};

// a 3D copy or fill: depth slices of height rows of width bytes
struct CopyOp {
    unsigned long long dst;
    unsigned long long src;
    unsigned long long width;
    unsigned long long height;
    unsigned long long depth;
    unsigned long long dst_pitch;
    unsigned long long src_pitch;
    unsigned long long dst_slice_pitch;
    unsigned long long src_slice_pitch;
    unsigned int value;
};

// cudaError_t values
static constexpr int ERROR_INVALID_VALUE = 1;
static constexpr int ERROR_MEMORY_ALLOCATION = 2;
static constexpr int ERROR_LAUNCH_PENDING_COUNT_EXCEEDED = 69;

// cudaStreamTailLaunch
static constexpr unsigned long long STREAM_TAIL_LAUNCH = 0x3;
// what cudaStreamCreateWithFlags / cudaEventCreateWithFlags return: all device side streams are one ordered stream
static constexpr unsigned long long DEVICE_STREAM = 0x10;
static constexpr unsigned long long DEVICE_EVENT = 0x20;

// The last error of a thread (cudaGetLastError) is kept in the entry of its hardware thread slot (SM, warp slot,
// lane), tagged with the thread's grid and position: a thread that later runs in the same slot doesn't see it. (A
// warp keeps its warp slot unless it is preempted, which would only lose the error.)
struct ErrorSlot {
    unsigned long long grid;
    unsigned int thread;
    unsigned int error;
};

static __device__ __forceinline__ unsigned long long gridId() {
    unsigned long long id;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(id));
    return id;
}

static __device__ __forceinline__ unsigned int threadTag() {
    const unsigned int block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const unsigned int thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    return block * (blockDim.x * blockDim.y * blockDim.z) + thread;
}

static __device__ __forceinline__ ErrorSlot *errorSlot(DeviceRuntimeState *state) {
    if (state->error_slots == 0) {
        return nullptr;
    }
    unsigned int sm, warp, lane;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (sm >= state->error_slot_sms || warp >= state->error_slot_warps) {
        return nullptr;
    }
    return reinterpret_cast<ErrorSlot *>(state->error_slots) + (sm * state->error_slot_warps + warp) * 32 + lane;
}

static __device__ __forceinline__ void setLastError(DeviceRuntimeState *state, int error) {
    ErrorSlot *slot = errorSlot(state);
    if (slot == nullptr) {
        atomicExch(&state->last_error, static_cast<unsigned int>(error));
        return;
    }
    slot->grid = gridId();
    slot->thread = threadTag();
    slot->error = static_cast<unsigned int>(error);
}

// the calling thread's last error; reset if clear
static __device__ __forceinline__ int lastError(DeviceRuntimeState *state, bool clear) {
    if (state == nullptr) {
        return 0;
    }
    ErrorSlot *slot = errorSlot(state);
    if (slot == nullptr) {
        return static_cast<int>(clear ? atomicExch(&state->last_error, 0u)
                                      : *reinterpret_cast<volatile unsigned int *>(&state->last_error));
    }
    if (slot->grid != gridId() || slot->thread != threadTag()) {
        return 0; // another thread's
    }
    const int error = static_cast<int>(slot->error);
    if (clear) {
        slot->error = 0;
    }
    return error;
}

// reserves a launch slot; returns its parameter area or nullptr
static __device__ __forceinline__ char *reserveLaunch(void *function, dim3 grid, dim3 block, unsigned int shared_mem,
                                                     unsigned int operation = OPERATION_KERNEL) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (state == nullptr || state->launch_queue == 0) {
        return nullptr;
    }
    const unsigned int index = atomicAdd(&state->launch_count, 1u);
    if (index >= state->launch_capacity) {
        setLastError(state, ERROR_LAUNCH_PENDING_COUNT_EXCEEDED);
        return nullptr;
    }
    char *slot = reinterpret_cast<char *>(state->launch_queue) + static_cast<unsigned long long>(index) * LAUNCH_SLOT_SIZE;
    auto *header = reinterpret_cast<LaunchSlot *>(slot);
    header->function = reinterpret_cast<unsigned long long>(function);
    header->grid[0] = grid.x;
    header->grid[1] = grid.y;
    header->grid[2] = grid.z;
    header->block[0] = block.x;
    header->block[1] = block.y;
    header->block[2] = block.z;
    header->shared_mem = shared_mem;
    header->status = LAUNCH_RESERVED;
    header->operation = operation;
    return slot + LAUNCH_HEADER_SIZE;
}

// the slot of a parameter buffer returned by reserveLaunch, nullptr if it isn't one
static __device__ __forceinline__ LaunchSlot *launchSlot(DeviceRuntimeState *state, void *buffer) {
    if (state == nullptr || state->launch_queue == 0 || buffer == nullptr) {
        return nullptr;
    }
    const unsigned long long offset = reinterpret_cast<unsigned long long>(buffer) - state->launch_queue;
    if (offset >= static_cast<unsigned long long>(state->launch_capacity) * LAUNCH_SLOT_SIZE ||
        offset % LAUNCH_SLOT_SIZE != LAUNCH_HEADER_SIZE) {
        return nullptr;
    }
    return reinterpret_cast<LaunchSlot *>(static_cast<char *>(buffer) - LAUNCH_HEADER_SIZE);
}

static __device__ __forceinline__ int commitLaunch(void *buffer, cudaStream_t stream) {
    DeviceRuntimeState *state = deviceRuntimeState();
    LaunchSlot *slot = launchSlot(state, buffer);
    if (slot == nullptr) {
        if (state != nullptr) {
            setLastError(state, ERROR_INVALID_VALUE);
        }
        return ERROR_INVALID_VALUE;
    }
    __threadfence_system();
    *reinterpret_cast<volatile unsigned int *>(&slot->status) =
        reinterpret_cast<unsigned long long>(stream) == STREAM_TAIL_LAUNCH ? LAUNCH_TAIL : LAUNCH_LAUNCHED;
    return 0;
}

extern "C" __device__ __noinline__ void *librecuda_get_parameter_buffer_v2(void *function, dim3 grid, dim3 block,
                                                                         unsigned int shared_mem) {
    return reserveLaunch(function, grid, block, shared_mem);
}

extern "C" __device__ __noinline__ int librecuda_launch_device_v2(void *buffer, cudaStream_t stream) {
    return commitLaunch(buffer, stream);
}

// the older interface: the function and configuration come with the launch
extern "C" __device__ __noinline__ void *librecuda_get_parameter_buffer(unsigned long long alignment,
                                                                      unsigned long long size) {
    if (alignment > LAUNCH_HEADER_SIZE || size > LAUNCH_PARAM_SIZE) {
        DeviceRuntimeState *state = deviceRuntimeState();
        if (state != nullptr) {
            setLastError(state, ERROR_INVALID_VALUE);
        }
        return nullptr;
    }
    return reserveLaunch(nullptr, dim3(), dim3(), 0);
}

extern "C" __device__ __noinline__ int librecuda_launch_device(void *function, void *buffer, dim3 grid, dim3 block,
                                                               unsigned int shared_mem, cudaStream_t stream) {
    LaunchSlot *slot = launchSlot(deviceRuntimeState(), buffer);
    if (slot != nullptr) {
        slot->function = reinterpret_cast<unsigned long long>(function);
        slot->grid[0] = grid.x;
        slot->grid[1] = grid.y;
        slot->grid[2] = grid.z;
        slot->block[0] = block.x;
        slot->block[1] = block.y;
        slot->block[2] = block.z;
        slot->shared_mem = shared_mem;
    }
    return commitLaunch(buffer, stream);
}

extern "C" __device__ __noinline__ int librecuda_get_last_error() {
    return lastError(deviceRuntimeState(), true);
}

extern "C" __device__ __noinline__ int librecuda_peek_at_last_error() {
    return lastError(deviceRuntimeState(), false);
}

extern "C" __device__ __noinline__ int librecuda_get_device(int *device) {
    DeviceRuntimeState *state = deviceRuntimeState();
    *device = state != nullptr ? state->device : 0;
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_get_device_count(int *count) {
    DeviceRuntimeState *state = deviceRuntimeState();
    *count = state != nullptr ? state->device_count : 1;
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_stream_create_with_flags(cudaStream_t *stream, unsigned int) {
    *stream = reinterpret_cast<cudaStream_t>(DEVICE_STREAM);
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_stream_destroy(cudaStream_t) {
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_cdp_malloc(void **pointer, unsigned long long size) {
    *pointer = librecuda_malloc(size);
    return *pointer != nullptr ? 0 : ERROR_MEMORY_ALLOCATION;
}

extern "C" __device__ __noinline__ int librecuda_cdp_free(void *pointer) {
    librecuda_free(pointer);
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_runtime_get_version(int *version) {
    *version = CUDART_VERSION;
    return 0;
}

// cudaGraphSetConditional: LibreCUDA's conditional handles are the device addresses of the condition values, which the
// host reads when the graph's conditional node runs
extern "C" __device__ __noinline__ void librecuda_graph_set_conditional(unsigned long long handle, unsigned int value) {
    *reinterpret_cast<volatile unsigned int *>(handle) = value;
    __threadfence_system();
}

// cudaGraphLaunch from device code: queued for the host, which runs the graph after the launching kernel (fire and
// forget) or after the launching graph (tail launch)
extern "C" __device__ __noinline__ int librecuda_graph_launch(unsigned long long graph_exec, unsigned long long stream) {
    if (graph_exec == 0) {
        return ERROR_INVALID_VALUE;
    }
    char *params = reserveLaunch(nullptr, dim3(), dim3(), 0, OPERATION_GRAPH);
    if (params == nullptr) {
        DeviceRuntimeState *state = deviceRuntimeState();
        return state != nullptr ? lastError(state, false) : ERROR_INVALID_VALUE;
    }
    reinterpret_cast<unsigned long long *>(params)[0] = graph_exec;
    reinterpret_cast<unsigned long long *>(params)[1] = stream;
    __threadfence_system();
    auto *slot = reinterpret_cast<LaunchSlot *>(params - LAUNCH_HEADER_SIZE);
    *reinterpret_cast<volatile unsigned int *>(&slot->status) = LAUNCH_LAUNCHED;
    return 0;
}

// ---- device queries: attributes and limits (tables written by the host), error strings, function attributes and
// occupancy (from the host's FunctionRecords, computed like the host runtime's cudaFuncGetAttributes /
// cudaOccupancyMaxActiveBlocksPerMultiprocessor)

extern "C" __device__ __noinline__ int librecuda_device_get_attribute(int *value, int attribute, int device) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (value == nullptr || state == nullptr || state->attributes == 0 || device < 0 ||
        device >= state->device_count || attribute <= 0 || attribute >= static_cast<int>(state->attribute_count)) {
        return ERROR_INVALID_VALUE;
    }
    *value = reinterpret_cast<const int *>(state->attributes)[device * state->attribute_count + attribute];
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_device_get_limit(unsigned long long *value, int limit) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (value == nullptr || state == nullptr || limit < 0 || limit >= 8) {
        return ERROR_INVALID_VALUE;
    }
    *value = state->limits[limit];
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_device_get_cache_config(int *config) {
    if (config == nullptr) {
        return ERROR_INVALID_VALUE;
    }
    *config = 0; // cudaFuncCachePreferNone
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_device_get_shared_mem_config(int *config) {
    if (config == nullptr) {
        return ERROR_INVALID_VALUE;
    }
    *config = 1; // cudaSharedMemBankSizeFourByte
    return 0;
}

struct ErrorString {
    int value;
    const char *name;
    const char *description;
};

// the host runtime's table (generated from the toolkit's driver_types.h)
static __device__ const ErrorString ERROR_STRINGS[] = {
#include "../../../runtime/src/error_strings.inc"
};

extern "C" __device__ __noinline__ const char *librecuda_get_error_name(int error) {
    for (const ErrorString &entry: ERROR_STRINGS) {
        if (entry.value == error) {
            return entry.name;
        }
    }
    return "cudaErrorUnknown";
}

extern "C" __device__ __noinline__ const char *librecuda_get_error_string(int error) {
    for (const ErrorString &entry: ERROR_STRINGS) {
        if (entry.value == error) {
            return entry.description;
        }
    }
    return "unknown error";
}

static __device__ const FunctionRecord *functionRecord(const void *function) {
    DeviceRuntimeState *state = deviceRuntimeState();
    if (state == nullptr || state->functions == 0) {
        return nullptr;
    }
    const auto *records = reinterpret_cast<const FunctionRecord *>(state->functions);
    for (unsigned int i = 0; i < state->function_count; i++) {
        if (records[i].address == reinterpret_cast<unsigned long long>(function)) {
            return &records[i];
        }
    }
    return nullptr;
}

extern "C" __device__ __noinline__ int librecuda_func_get_attributes(cudaFuncAttributes *attributes,
                                                                     const void *function) {
    const FunctionRecord *record = functionRecord(function);
    if (attributes == nullptr || record == nullptr) {
        return ERROR_INVALID_VALUE;
    }
    memset(attributes, 0, sizeof(*attributes));
    attributes->sharedSizeBytes = record->shared_size;
    attributes->constSizeBytes = record->const_size;
    attributes->localSizeBytes = record->local_size;
    attributes->maxThreadsPerBlock = record->max_threads;
    attributes->numRegs = record->registers;
    attributes->ptxVersion = record->version;
    attributes->binaryVersion = record->version;
    // like NVIDIA's device runtime, maxDynamicSharedSizeBytes and preferredShmemCarveout stay 0
    attributes->clusterDimMustBeSet = record->cluster_must_be_set;
    attributes->requiredClusterWidth = record->cluster[0];
    attributes->requiredClusterHeight = record->cluster[1];
    attributes->requiredClusterDepth = record->cluster[2];
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_occupancy_with_flags(int *blocks, const void *function, int block_size,
                                                                      unsigned long long dynamic_shared,
                                                                      unsigned int) {
    DeviceRuntimeState *state = deviceRuntimeState();
    const FunctionRecord *record = functionRecord(function);
    if (blocks == nullptr || record == nullptr || state == nullptr || state->attributes == 0 || block_size <= 0) {
        return ERROR_INVALID_VALUE;
    }
    if (block_size > record->max_threads) {
        *blocks = 0;
        return 0;
    }
    const int *attributes = reinterpret_cast<const int *>(state->attributes) + state->device * state->attribute_count;
    // cudaDevAttrMaxThreadsPerMultiProcessor, MaxBlocksPerMultiprocessor, MaxRegistersPerMultiprocessor,
    // MaxSharedMemoryPerMultiprocessor, ReservedSharedMemoryPerBlock
    const int max_threads_sm = attributes[39], max_blocks_sm = attributes[106], registers_sm = attributes[82];
    const unsigned long long shared_sm = attributes[81], reserved_shared = attributes[111];
    const int warps = (block_size + 31) / 32;
    int result = min(max_blocks_sm, (max_threads_sm / 32) / warps);
    const int registers_per_warp = ((max(1, record->registers) * 32 + 255) / 256) * 256;
    result = min(result, registers_sm / (registers_per_warp * warps));
    const unsigned long long shared = ((record->shared_size + dynamic_shared + reserved_shared + 127) / 128) * 128;
    if (shared > 0) {
        result = min(result, static_cast<int>(shared_sm / shared));
    }
    *blocks = max(0, result);
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_occupancy(int *blocks, const void *function, int block_size,
                                                           unsigned long long dynamic_shared) {
    return librecuda_occupancy_with_flags(blocks, function, block_size, dynamic_shared, 0);
}

// Device side updates of graph kernel nodes (cudaGraphKernelNodeSetParam, ...). A cudaGraphDeviceNode_t is the address
// of the node's record in pinned host memory (runtime/src/cudart.cpp, DeviceNodeRecord): the device writes the new
// values and flags, the host applies them at the graph's next launch.
struct DeviceNodeRecord {
    unsigned int valid;       // set by the host while an executable graph contains the node
    unsigned int param_size;  // bytes of the kernel's parameters
    unsigned int params_set;
    unsigned int grid_set;
    unsigned int enabled_set;
    unsigned int enabled;
    unsigned int grid[3];
    unsigned int reserved[3];
    unsigned char params[4096]; // the kernel's parameters at their offsets
};

static __device__ __forceinline__ DeviceNodeRecord *deviceNode(unsigned long long node) {
    auto *record = reinterpret_cast<DeviceNodeRecord *>(node);
    return record != nullptr && *reinterpret_cast<volatile unsigned int *>(&record->valid) ? record : nullptr;
}

extern "C" __device__ __noinline__ int librecuda_graph_kernel_node_set_param(unsigned long long node,
                                                                            unsigned long long offset,
                                                                            const void *value,
                                                                            unsigned long long size) {
    DeviceNodeRecord *record = deviceNode(node);
    if (record == nullptr || value == nullptr || offset + size > record->param_size) {
        return ERROR_INVALID_VALUE;
    }
    for (unsigned long long i = 0; i < size; i++) {
        record->params[offset + i] = static_cast<const unsigned char *>(value)[i];
    }
    __threadfence_system();
    *reinterpret_cast<volatile unsigned int *>(&record->params_set) = 1;
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_graph_kernel_node_set_enabled(unsigned long long node, bool enable) {
    DeviceNodeRecord *record = deviceNode(node);
    if (record == nullptr) {
        return ERROR_INVALID_VALUE;
    }
    record->enabled = enable ? 1 : 0;
    __threadfence_system();
    *reinterpret_cast<volatile unsigned int *>(&record->enabled_set) = 1;
    return 0;
}

extern "C" __device__ __noinline__ int librecuda_graph_kernel_node_set_grid_dim(unsigned long long node, dim3 grid) {
    DeviceNodeRecord *record = deviceNode(node);
    if (record == nullptr || grid.x == 0 || grid.y == 0 || grid.z == 0) {
        return ERROR_INVALID_VALUE;
    }
    record->grid[0] = grid.x;
    record->grid[1] = grid.y;
    record->grid[2] = grid.z;
    __threadfence_system();
    *reinterpret_cast<volatile unsigned int *>(&record->grid_set) = 1;
    return 0;
}

// cudaGraphKernelNodeUpdate (driver_types.h)
struct KernelNodeUpdate {
    unsigned long long node;
    int field; // 0: invalid, 1: grid dimensions, 2: parameter, 3: enabled
    union {
        unsigned int grid[3];
        struct {
            const void *value;
            unsigned long long offset;
            unsigned long long size;
        } param;
        unsigned int enabled;
    } data;
};

extern "C" __device__ __noinline__ int librecuda_graph_kernel_node_updates_apply(const KernelNodeUpdate *updates,
                                                                                unsigned long long count) {
    if (updates == nullptr && count > 0) {
        return ERROR_INVALID_VALUE;
    }
    for (unsigned long long i = 0; i < count; i++) {
        const KernelNodeUpdate &update = updates[i];
        int error = ERROR_INVALID_VALUE;
        switch (update.field) {
            case 1:
                error = librecuda_graph_kernel_node_set_grid_dim(update.node, dim3(update.data.grid[0],
                                                                                   update.data.grid[1],
                                                                                   update.data.grid[2]));
                break;
            case 2:
                error = librecuda_graph_kernel_node_set_param(update.node, update.data.param.offset,
                                                              update.data.param.value, update.data.param.size);
                break;
            case 3:
                error = librecuda_graph_kernel_node_set_enabled(update.node, update.data.enabled != 0);
                break;
            default:
                break;
        }
        if (error != 0) {
            return error;
        }
    }
    return 0;
}

// queues a copy (operation COPY) or fill (SET)
static __device__ __forceinline__ int queueCopy(unsigned int operation, const CopyOp &op, cudaStream_t stream) {
    if (op.width == 0 || op.height == 0 || op.depth == 0) {
        return 0;
    }
    char *params = reserveLaunch(nullptr, dim3(), dim3(), 0, operation);
    if (params == nullptr) {
        DeviceRuntimeState *state = deviceRuntimeState();
        return state != nullptr ? lastError(state, false) : ERROR_INVALID_VALUE;
    }
    *reinterpret_cast<CopyOp *>(params) = op;
    return commitLaunch(params, stream);
}

extern "C" __device__ __noinline__ int librecuda_memcpy_2d_async(void *dst, unsigned long long dst_pitch,
                                                                 const void *src, unsigned long long src_pitch,
                                                                 unsigned long long width, unsigned long long height,
                                                                 int, cudaStream_t stream) {
    CopyOp op{};
    op.dst = reinterpret_cast<unsigned long long>(dst);
    op.src = reinterpret_cast<unsigned long long>(src);
    op.width = width;
    op.height = height;
    op.depth = 1;
    op.dst_pitch = dst_pitch;
    op.src_pitch = src_pitch;
    return queueCopy(OPERATION_COPY, op, stream);
}

extern "C" __device__ __noinline__ int librecuda_memcpy_async(void *dst, const void *src, unsigned long long count,
                                                              int kind, cudaStream_t stream) {
    return librecuda_memcpy_2d_async(dst, count, src, count, count, 1, kind, stream);
}

extern "C" __device__ __noinline__ int librecuda_memcpy_3d_async(const cudaMemcpy3DParms *p, cudaStream_t stream) {
    if (p == nullptr) {
        return ERROR_INVALID_VALUE;
    }
    if (p->srcArray != nullptr || p->dstArray != nullptr) {
        // the arrays are the host runtime's objects: it does the copy
        char *params = reserveLaunch(nullptr, dim3(), dim3(), 0, OPERATION_COPY_3D);
        if (params == nullptr) {
            DeviceRuntimeState *state = deviceRuntimeState();
            return state != nullptr ? lastError(state, false) : ERROR_INVALID_VALUE;
        }
        *reinterpret_cast<cudaMemcpy3DParms *>(params) = *p;
        return commitLaunch(params, stream);
    }
    CopyOp op{};
    op.dst_pitch = p->dstPtr.pitch;
    op.src_pitch = p->srcPtr.pitch;
    op.dst_slice_pitch = p->dstPtr.pitch * p->dstPtr.ysize;
    op.src_slice_pitch = p->srcPtr.pitch * p->srcPtr.ysize;
    op.dst = reinterpret_cast<unsigned long long>(p->dstPtr.ptr) + p->dstPos.z * op.dst_slice_pitch +
             p->dstPos.y * op.dst_pitch + p->dstPos.x;
    op.src = reinterpret_cast<unsigned long long>(p->srcPtr.ptr) + p->srcPos.z * op.src_slice_pitch +
             p->srcPos.y * op.src_pitch + p->srcPos.x;
    op.width = p->extent.width;
    op.height = p->extent.height;
    op.depth = p->extent.depth;
    return queueCopy(OPERATION_COPY, op, stream);
}

extern "C" __device__ __noinline__ int librecuda_memset_2d_async(void *dst, unsigned long long pitch, int value,
                                                                 unsigned long long width, unsigned long long height,
                                                                 cudaStream_t stream) {
    CopyOp op{};
    op.dst = reinterpret_cast<unsigned long long>(dst);
    op.width = width;
    op.height = height;
    op.depth = 1;
    op.dst_pitch = pitch;
    op.value = static_cast<unsigned int>(value) & 0xff;
    return queueCopy(OPERATION_SET, op, stream);
}

extern "C" __device__ __noinline__ int librecuda_memset_async(void *dst, int value, unsigned long long count,
                                                              cudaStream_t stream) {
    return librecuda_memset_2d_async(dst, count, value, count, 1, stream);
}

extern "C" __device__ __noinline__ int librecuda_memset_3d_async(cudaPitchedPtr dst, int value, cudaExtent extent,
                                                                 cudaStream_t stream) {
    CopyOp op{};
    op.dst = reinterpret_cast<unsigned long long>(dst.ptr);
    op.width = extent.width;
    op.height = extent.height;
    op.depth = extent.depth;
    op.dst_pitch = dst.pitch;
    op.dst_slice_pitch = dst.pitch * dst.ysize;
    op.value = static_cast<unsigned int>(value) & 0xff;
    return queueCopy(OPERATION_SET, op, stream);
}

extern "C" __device__ __noinline__ int librecuda_event_create_with_flags(cudaEvent_t *event, unsigned int) {
    *event = reinterpret_cast<cudaEvent_t>(DEVICE_EVENT);
    return 0;
}

// recording, waiting for and destroying device side events: work in device side streams runs in order anyway
extern "C" __device__ __noinline__ int librecuda_event_no_op() {
    return 0;
}

// Replaces device runtime functions LibreCUDA doesn't implement (NVIDIA's libcudadevrt code in an image, which would
// call into NVIDIA's driver): stops the kernel, which reports an illegal instruction
extern "C" __device__ __noinline__ void librecuda_unsupported() {
    __trap();
}
