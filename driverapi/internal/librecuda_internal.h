#ifndef LIBRECUDA_LIBRECUDA_INTERNAL_H
#define LIBRECUDA_LIBRECUDA_INTERNAL_H

#include <sys/ioctl.h>
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
    libreCudaStatus_t channel_error = LIBRECUDA_SUCCESS;
    NvU32 channel_error_code = 0;
};

/**
 * Checks the channels' error notifiers.
 * @return LIBRECUDA_SUCCESS or the (sticky) error the channels reported
 */
libreCudaStatus_t ctxCheckChannelError(LibreCUcontext ctx);

/**
 * Creates the context's compute channel with the given index if it doesn't exist yet
 */
libreCudaStatus_t ctxEnsureComputeChannel(LibreCUcontext ctx, NvU32 index);

/**
 * Waits until *value >= target; fails if a channel of the context reports an error or the wait exceeds
 * LIBRECUDA_WAIT_TIMEOUT_MS (if set)
 */
libreCudaStatus_t ctxWaitForValue(LibreCUcontext ctx, volatile NvU64 *value, NvU64 target);

extern LibreCUcontext current_ctx;

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

libreCudaStatus_t
gpuSystemAlloc(LibreCUcontext ctx, size_t size, bool mapToCpu, NvU32 mapFlags,
               NvU64 *pVaOut, NvHandle *pMemoryHandleOut = nullptr);

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
