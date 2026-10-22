#ifndef LIBRECUDA_LIBRECUDA_INTERNAL_H
#define LIBRECUDA_LIBRECUDA_INTERNAL_H

#include <sys/ioctl.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <nvidia/ctrl2080gr.h>

#include "nvidia/nvtypes.h"
#include "nvidia/nvCpuUuid.h"
#include "nvidia/nvos.h"
#include "nvidia/nv_escape.h"
#include "nvidia/clc56f.h"

struct NvCommandQueue;
struct NvSignal;

enum DriverType {
    OPEN_KERNEL_MODULES, NVIDIA_PROPRIETARY
};

extern DriverType driver_type;

struct GPFifo {
    NvU64 *ring;
    NvU32 entries_count;
    NvU32 token;
    AmpereAControlGPFifo *controls;
    NvHandle channel_handle{};
    // compute class object of a GR channel (0 for copy engine channels)
    NvHandle compute_object{};
    NvU32 put_value = 0;

    /**
     * Error notifier of the channel (NvNotification array). RM writes notifier[0] when the channel is torn down
     * due to an error, e.g. an MMU fault or an illegal instruction in a kernel.
     */
    volatile NvU32 *error_notifier = nullptr;

    /**
     * For each ring entry, the timeline signal and value that are reached once the GPU executed the entry's
     * commands. The GPGet value in USERD is not reliably written back, so ring space is tracked this way.
     */
    std::vector<std::pair<NvSignal *, NvU32>> entry_done{};

    /**
     * Local memory buffer the channel was pointed at by the commands submitted so far (compute channels; protected by
     * the context's submit_mutex). Buffers only grow, so a channel is only ever pointed at a larger one.
     */
    NvU64 slm_va{};
    NvU64 slm_bytes_per_tpc{};
    NvU32 slm_per_thread{};
};

struct LibreCUdevice_ {
    int ordinal;
    NvU32 instance;
    NvU32 minor_number; // N of /dev/nvidiaN
    NvU32 gpu_id;
    NvU16 pci_device_id;
    NvU32 pci_domain;
    NvU8 pci_bus;
    NvU8 pci_slot;
    NvProcessorUuid uuid;
    NvU32 *gpu_mmio;
    NvU32 compute_class;
    NvU32 gpfifo_class;
    NvU32 dma_class;
    NV2080_CTRL_GR_INFO device_info[NV2080_CTRL_GR_INFO_INDEX_MAX + 1];

    // framebuffer info in bytes, populated on context creation
    NvU64 total_mem;
    NvU32 l2_cache_size;
    NvU32 mem_bus_width;
    NvU32 memory_clock_khz;
};

#define UVM_HEAP_START 0x1000000000

struct NvSignal;

/**
 * Pool for small device allocations: power-of-two size classes, each carving 2 MiB chunks into equal slots.
 * Allocating a chunk costs an RM allocation plus UVM range creation and mapping (~40 us); slots are handed out
 * without any driver call.
 */
struct SmallAllocPool {
    static constexpr NvU32 MIN_SLOT_SHIFT = 8; // 256 bytes, the alignment cudaMalloc guarantees
    static constexpr NvU32 MAX_SLOT_SHIFT = 19; // 512 KiB
    static constexpr NvU64 CHUNK_SIZE = 2 << 20;
    static constexpr NvU32 NUM_CLASSES = MAX_SLOT_SHIFT - MIN_SLOT_SHIFT + 1;

    struct Chunk {
        NvU64 base;
        NvU32 size_class;
        NvU32 used_slots;
        std::vector<NvU32> free_slots;
    };

    std::vector<Chunk *> chunks[NUM_CLASSES]{};

    /**
     * live slots: address -> chunk
     */
    std::unordered_map<NvU64, Chunk *> live{};

    ~SmallAllocPool() {
        for (auto &class_chunks: chunks) {
            for (Chunk *chunk: class_chunks) {
                delete chunk;
            }
        }
    }
};

struct LibreCUcontext_;

/**
 * @return all channels of the context (compute channels, then the copy channels)
 */
std::vector<GPFifo *> ctxChannels(LibreCUcontext_ *ctx);

/**
 * Thread safety: the driver API may be called from several threads at once (but a context must not be destroyed
 * while other threads use it). Locks, in the order they may be taken (a thread holding one only takes later ones):
 *   device_launch_mutex -> NvCommandQueue::mutex -> LibreCUcontext_::mutex -> allocation lock (librecuda.cpp)
 *   NvCommandQueue::mutex -> submit_mutex;  NvCommandQueue::mutex -> error_mutex -> printf_mutex
 * Waits for the GPU hold at most the waiting queue's lock (and device_launch_mutex), never a context lock; code
 * holding printf_mutex only try-locks queues.
 */
struct LibreCUcontext_ {
    int device_fd;
    int flags;
    LibreCUdevice device;
    NvHandle device_handle;
    NvHandle sub_device_handle;
    NvU64 uvm_vaddr = UVM_HEAP_START;
    NvHandle va_space_handle{};

    /**
     * Compute channels, each in its own subcontext of the GR channel group so that they run concurrently.
     * Streams are assigned to them round-robin (LIBRECUDA_COMPUTE_CHANNELS, default 4). Only channel 0 exists from the
     * start; the others are created by ctxEnsureComputeChannel when a stream is first assigned to them (an idle channel
     * in the group makes every submission slower). The vector is never resized: channel_handle 0 = not created yet.
     */
    std::vector<GPFifo> compute_gpfifos{};
    NvU32 next_compute_channel = 0;

    /**
     * Memory holding the rings and USERD of all channels (one slot per channel)
     */
    NvU64 gpfifo_area_va{};
    NvHandle gpfifo_area_handle{};

    /**
     * Copy engine channel used for host to device / device to device copies and memsets.
     * On GPUs with async copy engines it runs on its own copy engine (channel group), otherwise on the GR copy engine.
     */
    GPFifo dma_gpfifo;

    /**
     * Second copy engine channel used for device to host copies, so both directions can transfer concurrently.
     * Only valid if has_d2h_channel.
     */
    GPFifo dma_d2h_gpfifo;
    bool has_d2h_channel = false;

    NvHandle ctx_share;
    NvHandle channel_group;

    /**
     * Channel groups of the async copy engine channels (empty if the GR copy engine is used)
     */
    std::vector<NvHandle> copy_channel_groups{};

    /**
     * Command queues (streams) created on the context; waited for and detached when the context is destroyed
     */
    std::unordered_set<NvCommandQueue *> queues{};

    /**
     * Small device allocations (libreCuMemAlloc without CPU mapping); chunk memory is freed with the context
     */
    SmallAllocPool small_allocs{};

    /**
     * Shader local memory is channel state, so it is tracked per context (all streams share the compute channel).
     * It only ever grows; replaced allocations are retired rather than freed because in-flight kernels may use them.
     */
    NvU64 slm_va{};
    NvU32 slm_per_thread{};
    NvU64 slm_bytes_per_tpc{};
    std::vector<NvU64> retired_allocs{};

    /**
     * Per-thread stack size used for kernels with a statically unknown stack (recursion, indirect calls);
     * mirrors cudaLimitStackSize.
     */
    NvU32 stack_limit = 1024;

    /**
     * Context wide pool of signals for events. Events may be recorded and waited upon across streams.
     */
    NvSignal *event_signal_pool{};
    std::vector<NvU32> free_event_signals{};

    /**
     * Set once a channel of the context reported an error. The channels are unusable afterwards.
     */
    std::atomic<libreCudaStatus_t> channel_error{LIBRECUDA_SUCCESS};
    NvU32 channel_error_code = 0;

    /**
     * Context state that is set up lazily or changes after creation: modules, the device runtime, local memory, compute
     * channel creation, the queue set, event signals, texture pools, limits
     */
    std::recursive_mutex mutex{};
    // writing to the channels' GPFIFO rings (shared by the queues)
    std::mutex submit_mutex{};
    // finding out which error a channel reported
    std::mutex error_mutex{};
    // printing device side printf output
    std::recursive_mutex printf_mutex{};

    /**
     * Device side printf (see printf.cpp); set up when the first module calling vprintf is loaded
     */
    LibreCUmodule printf_module{};
    // address and register count of the device functions implementing external calls (vprintf, __assertfail), by
    // the name the kernels call them with
    std::unordered_map<std::string, std::pair<NvU64, NvU32>> device_runtime_functions{};
    // PrintfState in device memory (CPU mapped), its address is passed to kernels in %envreg8/9
    NvU64 printf_state_va{};
    // pinned host buffer receiving the records
    NvU8 *printf_buffer{};
    size_t printf_buffer_size = 1 << 20;
    // offset of the next record to print
    size_t printf_read_offset{};
    // a kernel's assert failed (an assert record was printed): explains the GR exception its trap causes
    bool device_assert_failed{};
    // size of the device side malloc heap (cudaLimitMallocHeapSize), allocated when a module calling malloc is loaded
    size_t malloc_heap_size = 8 << 20;

    /**
     * Device side kernel launches (device_launch.cpp): the queue of launch slots in pinned host memory (created when
     * the first module launching kernels is loaded) and its capacity (cudaLimitDevRuntimePendingLaunchCount). Launches
     * of kernels that may launch kernels hold the mutex until their children completed: the queue is the context's.
     */
    NvU8 *launch_queue{};
    NvU32 launch_queue_capacity = 2048;
    std::mutex device_launch_mutex{};
    // kernels by code address (the function pointers device code passes when it launches a kernel)
    std::unordered_map<NvU64, LibreCUFunction_ *> kernels_by_address{};

    /**
     * Texture header and sampler pools (textures.cpp), created with the first texture object
     */
    NvU64 texture_header_pool_va{};
    NvU64 texture_sampler_pool_va{};
    std::vector<NvU32> free_texture_headers{};

    /**
     * Kernel launches and the time of the last one (steady clock): RM reports SM exceptions of a kernel only some time
     * after it completed (sm_exceptions.cpp)
     */
    std::atomic<NvU64> launch_count{0};
    std::atomic<std::chrono::steady_clock::rep> last_launch_time{0};
};

/**
 * Checks the channels' error notifiers.
 * @return LIBRECUDA_SUCCESS or the (sticky) error the channels reported
 */
libreCudaStatus_t ctxCheckChannelError(LibreCUcontext ctx);

/**
 * Sets up device side printf for the context (module with the vprintf implementation, buffers) if not done yet
 */
libreCudaStatus_t ctxEnsurePrintf(LibreCUcontext ctx);

/**
 * Looks up the LibreCUDA implementation of an external function kernels call (vprintf, __assertfail), setting up the
 * device side runtime if necessary.
 * @return false if the function is not provided (or the setup failed)
 */
bool ctxDeviceRuntimeFunction(LibreCUcontext ctx, const std::string &name, NvU64 *pAddressOut, NvU32 *pRegistersOut);

/**
 * @return the highest register count of the device runtime's functions (the device runtime must be set up)
 */
NvU32 ctxDeviceRuntimeRegisters(LibreCUcontext ctx);

/**
 * Creates the queue for device side kernel launches if not done yet (device_launch.cpp)
 */
libreCudaStatus_t ctxEnsureDeviceLaunchQueue(LibreCUcontext ctx);

/**
 * Runs launch (a launch of function on stream) and, if the function may launch kernels from the device, waits for it
 * and launches the kernels it requested (and theirs) on the same stream before returning
 */
libreCudaStatus_t ctxLaunchWithDeviceLaunches(LibreCUcontext ctx, LibreCUstream stream, LibreCUFunction function,
                                              const std::function<libreCudaStatus_t()> &launch);

/**
 * Prints the device side printf output of completed records. Resets the buffer if all queues of the context are idle.
 */
libreCudaStatus_t ctxFlushPrintf(LibreCUcontext ctx);

/**
 * Frees the context's printf resources (called while destroying the context)
 */
void ctxReleasePrintf(LibreCUcontext ctx);

/**
 * The RM client and the control device file descriptor all RM objects of the process are created with
 */
NvHandle rmClient();

int rmControlFd();

/**
 * Issues an RM control on an object of the process's RM client (rm_ctrl for other translation units)
 */
libreCudaStatus_t rmControl(NvHandle object, NvV32 cmd, void *params, NvU32 paramSize);

// entries of the texture header and sampler pools (textures.cpp)
#define LIBRECUDA_TEXTURE_POOL_ENTRIES 4096

/**
 * Creates the context's texture header and sampler pools if not done yet
 */
libreCudaStatus_t ctxEnsureTexturePools(LibreCUcontext ctx);

/**
 * Frees an RM object of the process's RM client
 */
void rmFreeObject(NvHandle parent, NvHandle object);

/**
 * Reads the SM error state RM recorded for the context after its channels reported a GR exception (Xid 13) and maps
 * it to the specific error (e.g. a misaligned address). Takes tens of milliseconds.
 */
libreCudaStatus_t ctxSmExceptionStatus(LibreCUcontext ctx);

/**
 * Sleeps until RM must have reported SM exceptions of the kernels launched so far (up to 50 ms after the last launch)
 */
void ctxWaitForGrExceptionReports(LibreCUcontext ctx);

/**
 * Creates the context's compute channel with the given index if it doesn't exist yet
 */
libreCudaStatus_t ctxEnsureComputeChannel(LibreCUcontext ctx, NvU32 index);

/**
 * Waits until *value >= target; fails if a channel of the context reports an error or the wait exceeds
 * LIBRECUDA_WAIT_TIMEOUT_MS (if set)
 */
libreCudaStatus_t ctxWaitForValue(LibreCUcontext ctx, volatile NvU64 *value, NvU64 target);

// the calling thread's current context
extern thread_local LibreCUcontext current_ctx;

struct KernelConstantInfo {
    NvU32 const_nr;
    NvU64 address;
    NvU64 size;
};

struct KernelParamInfo {
    NvU16 param_index;
    /**
     * byte offset of the parameter relative to the start of the parameter area in constant bank 0
     */
    NvU16 param_offset;
    NvU16 param_size;
};

struct ModuleGlobalInfo {
    NvU64 address;
    NvU64 size;
};

struct LibreCUmodule_ {
    /**
     * List of all functions of the module
     */
    std::vector<LibreCUFunction_> functions{};

    /**
     * __device__ and __constant__ variables of the module, by symbol name
     */
    std::unordered_map<std::string, ModuleGlobalInfo> globals{};

    /**
     * GPU virtual address where the elf binary was copied.
     * This address is freed on unload.
     */
    NvU64 module_va_addr{};

    /**
     * Addresses of all function symbols (kernels and the device functions they call), by symbol name
     */
    std::unordered_map<std::string, NvU64> function_symbols{};

    /**
     * Register counts reported for functions (kernels and device functions), by symbol name
     */
    std::unordered_map<std::string, NvU32> function_registers{};
};


struct LibreCUFunction_ {
    std::string name;
    NvU64 func_va_addr;
    /**
     * static shared memory of the function (including the shared memory reserved by the toolchain).
     * Dynamic shared memory is allocated after this.
     */
    NvU32 shared_mem;
    NvU32 num_registers;
    NvU32 local_mem_req;
    NvU64 function_size;
    std::vector<KernelConstantInfo> constants;
    std::vector<KernelParamInfo> param_info;

    /**
     * Offset in constant bank 0 at which kernel parameters start (0x160 pre-Blackwell, 0x380 on Blackwell)
     */
    NvU32 param_base = 0x160;

    /**
     * Total size of the kernel parameter area in bytes
     */
    NvU32 param_total_size{};

    /**
     * Maximum dynamic shared memory opted in via cuFuncSetAttribute
     */
    NvU32 max_dynamic_shared_mem = 48 << 10;

    /**
     * Maximum threads per block the function was compiled for (__launch_bounds__ / register usage)
     */
    NvU32 max_threads_per_block = 1024;

    /**
     * Per-thread local memory (spill/stack frame) size reported by the toolchain
     */
    NvU32 local_size_bytes{};

    /**
     * The function calls other (non-inlined) functions, possibly recursively. Its stack size cannot be determined
     * statically, so it gets the context's stack limit like in CUDA.
     */
    bool has_calls{};

    /**
     * Thread block cluster dimensions the kernel was compiled with (__cluster_dims__; 1x1x1 if none)
     */
    NvU32 cluster_dims[3]{1, 1, 1};

    /**
     * The kernel was compiled for clusters (__cluster_dims__ or __cluster_dims__-less cluster kernel)
     */
    bool explicit_cluster{};

    /**
     * The kernel's module launches kernels from the device (dynamic parallelism): host launches wait for the kernel
     * and launch its children (device_launch.cpp)
     */
    bool launches_kernels{};
};

struct LibreCUstream_ {
    NvCommandQueue *command_queue;
};

struct LibreCuRoute {
    NvBool valid;
    NvU32 swizId;
    NvU32 engineId;
};


#define LIBRECUDA_VALIDATE_UVM_IOCTL(ret, data_ptr) {                                    \
    int return_value = ret;                                                              \
    int status = (data_ptr) != nullptr ? (data_ptr)->rmStatus : return_value;            \
    if (status != 0) {                                                                   \
        LIBRECUDA_DEBUG("UVM_IOCTL failed with return code " + std::to_string(status));  \
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);                                         \
    }                                                                                    \
}

static inline int uvm_ioctl(int fd, int cmd, void *data) {
    return ioctl(fd, cmd, data);
}

#define UVM_IOCTL(fd, nr, data, data_size) LIBRECUDA_VALIDATE_UVM_IOCTL(uvm_ioctl(fd, nr, data), data)

#define LIBRECUDA_VALIDATE_NVIOWR(ret) {                                                    \
    int return_value = ret;                                                                 \
    if (return_value != 0) {                                                                \
        LIBRECUDA_DEBUG("NVIOWR failed with return code " + std::to_string(return_value));  \
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);                                            \
    }                                                                                       \
}

static inline libreCudaStatus_t nv_iowr(int fd, int nr, void *data, size_t data_size) {
    return ioctl(fd, (3 << 30) | (data_size & 0x3FFF) << 16 | ('F' & 0xFF) << 8 | (nr & 0xFF), data);
}

#define LIBRECUDA_VALIDATE_INITIALIZED() { if (!librecudaInitialized) { LIBRECUDA_FAIL(LIBRECUDA_ERROR_NOT_INITIALIZED); } }

#define NV_IOWR(fd, nr, data, data_size) LIBRECUDA_VALIDATE_NVIOWR(nv_iowr(fd, nr, data, data_size))

static inline libreCudaStatus_t rm_alloc(int fd, NvV32 clss,
                                         NvHandle client, NvHandle parent, NvHandle object,
                                         void *params, NvU32 paramSize,
                                         NvHandle *pObjectNew) {
    LIBRECUDA_VALIDATE(fd > 0, LIBRECUDA_ERROR_INVALID_VALUE);
    NVOS21_PARAMETERS parameters{
        .hRoot = client,
        .hObjectParent = parent,
        .hObjectNew = object,
        .hClass = clss,
        .pAllocParms = params,
        .paramsSize = paramSize
    };
    NV_IOWR(fd, NV_ESC_RM_ALLOC, &parameters, sizeof(parameters));
    if (pObjectNew != nullptr) {
        *pObjectNew = parameters.hObjectNew;
    }
    if (parameters.status != 0) {
        LIBRECUDA_DEBUG("rm_alloc of class " + std::to_string(clss) + " failed with RM status " +
                        std::to_string(parameters.status));
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);
    }
    LIBRECUDA_SUCCEED();
}

#define LIBRECUDA_VALIDATE_RM_ALLOC(status) { libreCudaStatus_t status_val = status; if (status_val != LIBRECUDA_SUCCESS) { LIBRECUDA_DEBUG(__FILE__ ":" __LIBRECUDA_TOSTRING(__LINE__) ": rm_alloc failed with status " + std::to_string(status_val)); LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN); } }

#define RM_ALLOC(fd, clss, client, parent, object, params, param_size, pObjectNew) LIBRECUDA_VALIDATE_RM_ALLOC(rm_alloc(fd, clss, client, parent, object, params, param_size, pObjectNew));

static inline libreCudaStatus_t rm_ctrl(int fd,
                                        NvV32 cmd,
                                        NvHandle client, NvHandle object,
                                        void *params, NvU32 paramSize);

#define LIBRECUDA_VALIDATE_RM_CTRL(status) { libreCudaStatus_t status_val = status; if (status_val != LIBRECUDA_SUCCESS) { LIBRECUDA_DEBUG(__FILE__ ":" __LIBRECUDA_TOSTRING(__LINE__) ": rm_ctrl failed with status " + std::to_string(status_val)); LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN); } }

#define RM_CTRL(fd, cmd, client, object, params, paramSize) LIBRECUDA_VALIDATE_RM_CTRL(rm_ctrl(fd, cmd, client, object, params, paramSize))

#define LIBRECUDA_ENSURE_CTX_VALID() LIBRECUDA_VALIDATE(current_ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT);

libreCudaStatus_t
gpuAlloc(LibreCUcontext ctx, size_t size, bool physicalContiguity, bool hugePages, bool mapToCpu, NvU32 mapFlags,
         NvU64 *pVaOut, NvHandle *pMemoryHandleOut = nullptr);

/**
 * Allocates system memory mapped into the GPU's address space (and the CPU's, if mapToCpu).
 * cpuCached: the CPU mapping is cached instead of uncached. GPU writes are coherent with the CPU caches, so this suits
 * memory the CPU only reads.
 */
libreCudaStatus_t
gpuSystemAlloc(LibreCUcontext ctx, size_t size, bool mapToCpu, NvU32 mapFlags,
               NvU64 *pVaOut, NvHandle *pMemoryHandleOut = nullptr, bool cpuCached = false);

/**
 * Allocates page-locked, CPU cached host memory mapped at the same virtual address on the GPU
 */
libreCudaStatus_t gpuHostAlloc(LibreCUcontext ctx, size_t size, NvU64 *pVaOut);

libreCudaStatus_t gpuFree(LibreCUcontext ctx, NvU64 virtualAddress);

/**
 * @return true if ptr points into any memory the GPU can access (device memory, pinned host memory)
 */
bool isGpuAccessiblePtr(const void *ptr);

libreCudaStatus_t ctxObtainEventSignal(LibreCUcontext ctx, NvSignal **pSignalOut);

void ctxReleaseEventSignal(LibreCUcontext ctx, NvSignal *signal);

/**
 * @param ptr the pointer to check for being a device pointer
 * @return true if the pointer is a device pointer.
 * This does not mean the ptr is still allocated. It just means it is or was a device pointer.
 */
bool isDevicePtr(void *ptr);

/**
 * @param ptr the pointer to check for being mapped to host.
 * @return true if the pointer is a device pointer and mapped to host. will return false for host pointers.
 */
bool isHostMappedPtr(void *ptr);

#endif //LIBRECUDA_LIBRECUDA_INTERNAL_H
