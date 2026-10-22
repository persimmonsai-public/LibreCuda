#include "librecuda.h"

#include "librecuda_status.h"
#include "librecuda_status_internal.h"

#include "librecuda_internal.h"
#include "cmdqueue.h"

#include "nvidia/nv-unix-nvos-params-wrappers.h"
#include "nvidia/nv_escape.h"
#include "nvidia/nv-ioctl-numbers.h"
#include "nvidia/uvm_linux_ioctl.h"
#include "nvidia/ctrl0000gpu.h"
#include "nvidia/nvos.h"
#include "nvidia/alloc_channel.h"
#include "nvidia/nv-ioctl.h"
#include "nvidia/dev_mmu.h"
#include "nvidia/cl0080.h"
#include "nvidia/cl2080.h"
#include "nvidia/ctrl2080gpu.h"
#include "nvidia/ctrl2080mc.h"
#include "nvidia/ctrl2080mc.1.h"

#include "nvidia/clc56f.h"
#include "nvidia/clb0b5sw.h"
#include "nvidia/g_allclasses.h"
#include "nvidia/ctrl0080gpu.h"
#include "nvidia/ctrlc36f.h"
#include "nvidia/ctrla06c.h"
#include "memcopy.h"

#include <elfio/elfio.hpp>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
#include <cstring>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <thread>

// each channel's ring (0x10000 entries) and USERD occupy one slot of the gpfifo area
#define GPU_FIFO_SLOT_SIZE 0x100000
#define MAX_COMPUTE_CHANNELS 16
#define EVENT_SIGNAL_POOL_SIZE 4096

// subset of ctrl2080ce.h (open-gpu-kernel-modules)
#define NV2080_CTRL_CMD_CE_GET_CAPS_V2 (0x20802a03)
#define NV2080_CTRL_CE_CAPS_TBL_SIZE 2
#define NV2080_CTRL_CE_CAPS_CE_GRCE_BYTE0 0x01
#define NV2080_CTRL_CE_CAPS_CE_SYSMEM_READ_BYTE0 0x04
#define NV2080_CTRL_CE_CAPS_CE_SYSMEM_WRITE_BYTE0 0x08

typedef struct NV2080_CTRL_CE_GET_CAPS_V2_PARAMS {
    NvU32 ceEngineType;
    NvU8 capsTbl[NV2080_CTRL_CE_CAPS_TBL_SIZE];
} NV2080_CTRL_CE_GET_CAPS_V2_PARAMS;

// subset of ctrl2080fb.h (open-gpu-kernel-modules)
#define NV2080_CTRL_CMD_FB_GET_INFO_V2 (0x20801303U)
#define NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE (0x00000008U)
#define NV2080_CTRL_FB_INFO_INDEX_BUS_WIDTH (0x0000000BU)
#define NV2080_CTRL_FB_INFO_INDEX_L2CACHE_SIZE (0x0000001BU)
#define NV2080_CTRL_FB_INFO_INDEX_HEAP_FREE (0x00000016U)
#define NV2080_CTRL_FB_INFO_MAX_LIST_SIZE (0x00000080U)

typedef struct NV2080_CTRL_FB_INFO {
    NvU32 index;
    NvU32 data;
} NV2080_CTRL_FB_INFO;

typedef struct NV2080_CTRL_FB_GET_INFO_V2_PARAMS {
    NvU32 fbInfoListSize;
    NV2080_CTRL_FB_INFO fbInfoList[NV2080_CTRL_FB_INFO_MAX_LIST_SIZE];
} NV2080_CTRL_FB_GET_INFO_V2_PARAMS;

// subset of ctrl2080perf.h (open-gpu-kernel-modules)
#define NV2080_CTRL_CMD_PERF_GET_LEVEL_INFO_V2 (0x2080200b)
#define NV2080_CTRL_PERF_CLK_MAX_DOMAINS 32U

typedef struct NV2080_CTRL_PERF_GET_CLK_INFO {
    NvU32 flags;
    NvU32 domain;
    NvU32 currentFreq;
    NvU32 defaultFreq;
    NvU32 minFreq;
    NvU32 maxFreq;
} NV2080_CTRL_PERF_GET_CLK_INFO;

typedef struct NV2080_CTRL_PERF_GET_LEVEL_INFO_V2_PARAMS {
    NvU32 level;
    NvU32 flags;
    NV2080_CTRL_PERF_GET_CLK_INFO perfGetClkInfoList[NV2080_CTRL_PERF_CLK_MAX_DOMAINS];
    NvU32 perfGetClkInfoListSize;
} NV2080_CTRL_PERF_GET_LEVEL_INFO_V2_PARAMS;

NvHandle root{};

static int fd_ctl = 0;
static int fd_uvm = 0;

static int num_devices;
static std::vector<nv_ioctl_card_info_t> devices{};
static std::vector<NvU32> device_instances{};
static std::vector<LibreCUdevice_> public_device_handles;
// first context created on each device, by ordinal
static std::unordered_map<int, LibreCUcontext> device_contexts{};

static bool librecudaInitialized = false;

// allocation bookkeeping (allocations, the contexts' VA bump pointers and small allocation pools, OS descriptor
// handles); taken last, nothing is waited for while holding it
static std::recursive_mutex alloc_mutex{};
DriverType driver_type;

thread_local LibreCUcontext current_ctx = nullptr;

libreCudaStatus_t rm_ctrl(int fd, NvV32 cmd, NvHandle client, NvHandle object, void *params, NvU32 paramSize) {
    LIBRECUDA_VALIDATE(params != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    NVOS54_PARAMETERS parameters{
        .hClient = client,
        .hObject = object,
        .cmd = cmd,
        .params = params,
        .paramsSize = paramSize
    };
    NV_IOWR(fd, NV_ESC_RM_CONTROL, &parameters, sizeof(parameters));
    if (parameters.status != 0) {
        LIBRECUDA_DEBUG("rm_ctrl failed with status: " + std::to_string(parameters.status));
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);
    }
    LIBRECUDA_SUCCEED();
}

static void rmFree(NvHandle parent, NvHandle object) {
    NVOS00_PARAMETERS params{
        .hRoot = root,
        .hObjectParent = parent,
        .hObjectOld = object
    };
    nv_iowr(fd_ctl, NV_ESC_RM_FREE, &params, sizeof(params));
}

/**
 * Checks whether the GPU responds to RM calls by allocating a device & sub-device object and querying the GPU uuid.
 */
static bool probeGpu(const nv_ioctl_card_info_t &cardInfo, NvU32 deviceInstance) {
    // the GPU only becomes usable through RM once its device file has been opened. The fd is kept open for the
    // lifetime of the process so the GPU stays initialized.
    {
        std::string device_file = "/dev/nvidia" + std::to_string(cardInfo.minor_number);
        int device_fd = open(device_file.c_str(), O_RDWR | O_CLOEXEC);
        if (device_fd == -1) {
            return false;
        }
        nv_ioctl_register_fd_t params{.ctl_fd = fd_ctl};
        if (nv_iowr(device_fd, NV_ESC_REGISTER_FD, &params, sizeof(params)) != 0) {
            close(device_fd);
            return false;
        }
    }
    NvHandle device_handle{};
    NV0080_ALLOC_PARAMETERS params{
        .deviceId = deviceInstance,
        .hClientShare = root,
        .vaMode = NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES
    };
    if (rm_alloc(fd_ctl, NV01_DEVICE_0, root, root, 0, &params, sizeof(params), &device_handle) != LIBRECUDA_SUCCESS) {
        return false;
    }
    bool usable = false;
    NvHandle sub_device_handle{};
    if (rm_alloc(fd_ctl, NV20_SUBDEVICE_0, root, device_handle, 0, nullptr, 0, &sub_device_handle) ==
        LIBRECUDA_SUCCESS) {
        NV2080_CTRL_GPU_GET_GID_INFO_PARAMS gid_params{
            .flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY,
            .length = 16
        };
        usable = rm_ctrl(fd_ctl, NV2080_CTRL_CMD_GPU_GET_GID_INFO, root, sub_device_handle, &gid_params,
                         sizeof(gid_params)) == LIBRECUDA_SUCCESS;
    }
    rmFree(root, device_handle); // frees the sub-device as well
    return usable;
}

libreCudaStatus_t libreCuInit(int flags) {
    if (librecudaInitialized) {
        return LIBRECUDA_SUCCESS;
    }

    internalLibreCuInitStatusNames();
    fd_ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    LIBRECUDA_VALIDATE(fd_ctl != -1, LIBRECUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);

    fd_uvm = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    LIBRECUDA_VALIDATE(fd_uvm != -1, LIBRECUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);

    // create rm client
    RM_ALLOC(fd_ctl, NV01_ROOT_CLIENT, 0, 0, 0, nullptr, 0, &root);

    // initialize nvidia_uvm
    {
        UVM_INITIALIZE_PARAMS params{.flags = 0};
        UVM_IOCTL(fd_uvm, UVM_INITIALIZE, &params, sizeof(params));
    }

    // initialize mm
    {
        int fd_uvm_2 = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        LIBRECUDA_VALIDATE(fd_uvm_2 != -1, LIBRECUDA_ERROR_INVALID_DEVICE);
        
        UVM_MM_INITIALIZE_PARAMS params{
            .uvmFd = fd_uvm
        };
        // Not required by all platforms, status only ok when needed
        // On Linux, open-kernel-modules requires the fd, while the proprietary driver does not
        int ret = uvm_ioctl(fd_uvm_2, UVM_MM_INITIALIZE, &params);
        NV_STATUS status = params.rmStatus;
        if (ret != 0 || status != 0) {
            if (status == NV_WARN_NOTHING_TO_DO) {
                // we use this as an indicator that the proprietary driver is in use
                driver_type = NVIDIA_PROPRIETARY;
                LIBRECUDA_DEBUG("Proprietary driver detected. Consider using the open-kernel-modules driver.");
                LIBRECUDA_DEBUG("The proprietary driver is not recommended and is inferior to the open-kernel-modules driver in terms of supported features.");
                LIBRECUDA_DEBUG("Eg. DtoH memcpys cannot be performed via DMA, and a workaround is currently not implemented.");
                close(fd_uvm_2);
            } else {
                LIBRECUDA_DEBUG("UVM_MM_INITIALIZE failed with return code " + std::to_string(ret) + " and status " +
                    std::to_string(status));
                close(fd_uvm_2);
                LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);
            }
        } else {
            // Keep the mm fd open: closing it shuts down the va_space_mm and
            // newer drivers then refuse all further GPU registrations.
            driver_type = OPEN_KERNEL_MODULES;
        }
    }

    // obtaining basic card info
    {
        constexpr int n_infos = 32; // NV_MAX_DEVICES; size must be an exact multiple of sizeof(nv_ioctl_card_info_t) on newer drivers
        // kernel respects this size; see open-gpu-kernel-modules/kernel-open/nvidia/nv.c:2451
        nv_ioctl_card_info_t card_infos[n_infos];
        NV_IOWR(fd_ctl, NV_ESC_CARD_INFO, card_infos, sizeof(card_infos));

        // count number of valid entries
        int num_gpus = 0;
        for (; num_gpus < n_infos; num_gpus++) {
            if (!card_infos[num_gpus].valid) {
                break;
            }
        }

        // CUDA_VISIBLE_DEVICES selects GPUs by their index in the driver's list (like CUDA). Only selected GPUs are
        // opened: opening a GPU initializes it, which is best avoided for GPUs the process doesn't use.
        std::vector<int> selected{}; {
            const char *env_var_cstr = getenv("CUDA_VISIBLE_DEVICES");
            if (env_var_cstr != nullptr) {
                std::string env_var = std::string(env_var_cstr);
                size_t start = 0;
                while (start <= env_var.size()) {
                    size_t end = env_var.find(',', start);
                    if (end == std::string::npos) {
                        end = env_var.size();
                    }
                    std::string ordinal_string = env_var.substr(start, end - start);
                    char *parse_end{};
                    long parsed = strtol(ordinal_string.c_str(), &parse_end, 10);
                    // like CUDA, stop at the first invalid entry
                    if (ordinal_string.empty() || *parse_end != '\0' || parsed < 0 || parsed >= num_gpus) {
                        break;
                    }
                    if (std::find(selected.begin(), selected.end(), parsed) == selected.end()) {
                        selected.push_back(static_cast<int>(parsed));
                    }
                    start = end + 1;
                }
            } else {
                for (int i = 0; i < num_gpus; i++) {
                    selected.push_back(i);
                }
            }
        }

        // Probe each selected GPU: GPUs that have fallen off the bus or failed to initialize are still reported by
        // NV_ESC_CARD_INFO, but every RM call on them fails. Only usable GPUs get an ordinal.
        for (int index: selected) {
            const nv_ioctl_card_info_t &card_info = card_infos[index];
            NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS id_params{
                .gpuId = card_info.gpu_id,
            };
            if (rm_ctrl(fd_ctl, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, root, root, &id_params, sizeof(id_params)) !=
                LIBRECUDA_SUCCESS) {
                continue;
            }
            if (!probeGpu(card_info, id_params.deviceInstance)) {
                LIBRECUDA_DEBUG("Skipping unusable GPU at PCI " + std::to_string(card_info.pci_info.bus) + ":" +
                                std::to_string(card_info.pci_info.slot));
                continue;
            }
            devices.push_back(card_info);
            device_instances.push_back(id_params.deviceInstance);
        }
        num_devices = static_cast<int>(devices.size());
    }

    // creating public device handles
    {
        public_device_handles.reserve(num_devices);
        for (int ordinal = 0; ordinal < num_devices; ordinal++) {
            const nv_ioctl_card_info_t &card_info = devices[ordinal];
            public_device_handles.push_back(LibreCUdevice_{
                .ordinal = ordinal,
                .instance = device_instances[ordinal],
                .minor_number = card_info.minor_number,
                .gpu_id = card_info.gpu_id,
                .pci_device_id = card_info.pci_info.device_id,
                .pci_domain = card_info.pci_info.domain,
                .pci_bus = card_info.pci_info.bus,
                .pci_slot = card_info.pci_info.slot,
            });
        }
    }

    librecudaInitialized = true;
    LIBRECUDA_SUCCEED();
}

static std::string getDeviceFile(LibreCUdevice device) {
    return "/dev/nvidia" + std::to_string(device->minor_number);
}

libreCudaStatus_t libreCuDeviceGetCount(int *pDeviceCount) {
    LIBRECUDA_VALIDATE_INITIALIZED();
    LIBRECUDA_VALIDATE(pDeviceCount != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pDeviceCount = num_devices;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuDeviceGet(LibreCUdevice *device, int deviceOrdinal) {
    LIBRECUDA_VALIDATE_INITIALIZED();
    LIBRECUDA_VALIDATE(device != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(deviceOrdinal >= 0 && deviceOrdinal < num_devices, LIBRECUDA_ERROR_INVALID_DEVICE);
    *device = &public_device_handles[deviceOrdinal];
    LIBRECUDA_SUCCEED();
}


libreCudaStatus_t libreCuGetErrorString(libreCudaStatus_t status, const char **pStatusName) {
    if (!internalLibreCuInitStatusNamesInitialized()) {
        // we want to tolerate this failure to easy debugging for the user to not descend into madness
        internalLibreCuInitStatusNames();
    }
    const char *status_name = internalLibreCuGetStatusName(static_cast<int>(status));
    LIBRECUDA_VALIDATE(pStatusName != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pStatusName = status_name;
    LIBRECUDA_SUCCEED();
}

static inline NvU64 ceilDiv(NvU64 a, NvU64 b) {
    return (a + b - 1) / b;
}

static inline NvU32 maxOf(NvU32 a, NvU32 b) {
    return (a > b) ? a : b;
}

static inline NvU64 maxOf(NvU64 a, NvU64 b) {
    return (a > b) ? a : b;
}


// all contexts share the process' UVM address space, so virtual addresses are handed out globally
static NvU64 next_uvm_vaddr = UVM_HEAP_START;

NvU64 bump_alloc_virtual_addr(LibreCUcontext ctx, size_t size, NvU32 alignment = 4 << 10) {
    std::lock_guard alloc_lock(alloc_mutex);
    NvU64 va_address = next_uvm_vaddr;
    NvU64 slack = va_address % alignment;
    if (slack != 0) {
        va_address += (alignment - slack);
    }
    assert(va_address % alignment == 0);
    next_uvm_vaddr = va_address + size;
    ctx->uvm_vaddr = next_uvm_vaddr;
    return va_address;
}

libreCudaStatus_t
makeGpuFifo(LibreCUcontext ctx,
            NvU64 gpfifoAreaVa, NvHandle gpfifoAreaHandle,
            NvHandle ctxShare, NvHandle channelGroup,
            NvU64 offset, NvU32 entries,
            NvU32 copyEngineType,
            GPFifo *pGPFifoOut) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_UNKNOWN);
    LIBRECUDA_VALIDATE(gpfifoAreaVa != 0, LIBRECUDA_ERROR_UNKNOWN);
    LIBRECUDA_VALIDATE(gpfifoAreaHandle != 0, LIBRECUDA_ERROR_UNKNOWN);
    LIBRECUDA_VALIDATE(pGPFifoOut != nullptr, LIBRECUDA_ERROR_UNKNOWN);

    void *notifier_va{};
    NvHandle notifier_memory_handle{}; {
        LIBRECUDA_ERR_PROPAGATE(
            gpuSystemAlloc(
                ctx,
                // RM only writes NV_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1 (3) NvNotification entries
                4 << 10, true, 0,
                reinterpret_cast<NvU64 *>(&notifier_va),
                &notifier_memory_handle,
                // read on every synchronization (ctxCheckChannelError)
                true
            )
        );
    }

    NvHandle compute_object{};
    NvHandle gpfifo{}; {
        NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS params{
            .hObjectError = notifier_memory_handle,
            .hObjectBuffer = gpfifoAreaHandle,
            .gpFifoOffset = gpfifoAreaVa + offset,
            .gpFifoEntries = entries,
            .hContextShare = ctxShare,
            .hUserdMemory = {
                gpfifoAreaHandle
            },
            .userdOffset = {
                entries * 8 + offset
            },
            // copy engine channels attach to the VA space of their channel group
            .engineType = copyEngineType
        };
        RM_ALLOC(fd_ctl, ctx->device->gpfifo_class, root, channelGroup, 0, &params, sizeof(params), &gpfifo);
    }

    if (copyEngineType == 0) {
        // channel in the GR channel group: compute and (GR) copy engine objects
        RM_ALLOC(fd_ctl, ctx->device->compute_class, root, gpfifo, 0, nullptr, 0, &compute_object);
        RM_ALLOC(fd_ctl, ctx->device->dma_class, root, gpfifo, 0, nullptr, 0, nullptr);
    } else {
        // channel on an async copy engine
        NVB0B5_ALLOCATION_PARAMETERS params{
            .version = NVB0B5_ALLOCATION_PARAMETERS_VERSION_1,
            .engineType = copyEngineType
        };
        RM_ALLOC(fd_ctl, ctx->device->dma_class, root, gpfifo, 0, &params, sizeof(params), nullptr);
    }

    // get work submit token
    NvU32 work_submit_token; {
        NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS params{.workSubmitToken = 0xFFFFFFFF};
        RM_CTRL(fd_ctl, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, root, gpfifo, &params, sizeof(params));
        LIBRECUDA_VALIDATE(params.workSubmitToken != 0xFFFFFFFF, LIBRECUDA_ERROR_UNKNOWN);
        work_submit_token = params.workSubmitToken;
    }

    NvU64 channel_base_va = bump_alloc_virtual_addr(ctx, 0x4000000); {
        UVM_REGISTER_CHANNEL_PARAMS params{
            .gpuUuid = ctx->device->uuid,
            .rmCtrlFd = fd_ctl,
            .hClient = root,
            .hChannel = gpfifo,
            .base = channel_base_va,
            .length = 0x4000000
        };
        UVM_IOCTL(fd_uvm, UVM_REGISTER_CHANNEL, &params, sizeof(UVM_REGISTER_CHANNEL_PARAMS));
    }

    GPFifo fifo{
        .ring = reinterpret_cast<NvU64 *>(gpfifoAreaVa + offset),
        .entries_count = entries,
        .token = work_submit_token,
        .controls = reinterpret_cast<AmpereAControlGPFifo *>(gpfifoAreaVa + offset + entries * 8),
        .channel_handle = gpfifo,
        .compute_object = compute_object,
        .error_notifier = reinterpret_cast<volatile NvU32 *>(notifier_va)
    };
    fifo.entry_done.assign(entries, {nullptr, 0});
    *pGPFifoOut = fifo;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t
memMapToCpu(LibreCUcontext ctx, NvHandle memoryHandle, size_t size, NvU64 &targetVa, NvU32 mapFlags,
            bool isSystemAlloc) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(memoryHandle != 0, LIBRECUDA_ERROR_INVALID_VALUE);

    int device_fd;
    if (isSystemAlloc) {
        device_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
        LIBRECUDA_VALIDATE(device_fd != -1, LIBRECUDA_ERROR_INVALID_DEVICE);
    } else {
        std::string device_file = getDeviceFile(ctx->device);

        device_fd = open(device_file.c_str(), O_RDWR | O_CLOEXEC);
        LIBRECUDA_VALIDATE(device_fd != -1, LIBRECUDA_ERROR_INVALID_DEVICE);
        nv_ioctl_register_fd_t params{
            .ctl_fd = fd_ctl
        };
        NV_IOWR(device_fd, NV_ESC_REGISTER_FD, &params, sizeof(params));
    }

    nv_ioctl_nvos33_parameters_with_fd parameters{
        .params = {
            .hClient = root,
            .hDevice = ctx->device_handle,
            .hMemory = memoryHandle,
            .length = size,
            .flags = mapFlags
        },
        .fd = device_fd
    };
    NV_IOWR(fd_ctl, NV_ESC_RM_MAP_MEMORY, &parameters, sizeof(parameters));

    void *map_result = mmap(
        reinterpret_cast<void *>(targetVa), size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | (targetVa != 0 ? MAP_FIXED : 0),
        device_fd, 0
    );

    LIBRECUDA_VALIDATE(
        map_result != MAP_FAILED,
        LIBRECUDA_ERROR_UNKNOWN
    );
    if (targetVa == 0) {
        targetVa = reinterpret_cast<NvU64>(map_result);
    }

    close(device_fd);

    LIBRECUDA_SUCCEED();
}

enum class AllocKind {
    DEVICE, // video memory
    SYSTEM, // RM allocated system memory (internal use: signals, notifiers)
    HOST    // page-locked host memory (libreCuMemAllocHost), cached on the CPU and mapped at the same VA on the GPU
};

struct AllocInfo {
    LibreCUcontext ctx;
    NvU64 size;
    NvHandle memory_handle;
    AllocKind kind;
    bool cpu_mapped;
    // other contexts whose GPUs the allocation is mapped on (host and system memory: all; device memory: peers with
    // access)
    std::vector<LibreCUcontext> mapped_on{};
};

// all live allocations, keyed by their (GPU and, if mapped, CPU) virtual address
static std::map<NvU64, AllocInfo> allocations{};

// live contexts (page-locked host memory and system memory, e.g. event signals, are mapped on all of them) and the GPU
// pairs with UVM peer access
static std::vector<LibreCUcontext> live_contexts{};
static std::set<std::pair<NvU32, NvU32>> uvm_peer_pairs{};

/**
 * Maps an allocation (already mapped on its own GPU) on the GPU of another context, at the same address
 */
static libreCudaStatus_t mapOnContext(LibreCUcontext target, NvU64 va, AllocInfo &info) {
    UVM_MAP_EXTERNAL_ALLOCATION_PARAMS params{
        .base = va,
        .length = info.size,
        .perGpuAttributes = {
            {
                .gpuUuid = target->device->uuid,
                .gpuMappingType = UvmGpuMappingTypeReadWriteAtomic
            }
        },
        .gpuAttributesCount = 1,
        .rmCtrlFd = fd_ctl,
        .hClient = root,
        .hMemory = info.memory_handle
    };
    UVM_IOCTL(fd_uvm, UVM_MAP_EXTERNAL_ALLOCATION, &params, sizeof(params));
    info.mapped_on.push_back(target);
    LIBRECUDA_SUCCEED();
}

static void unmapFromContext(LibreCUcontext target, NvU64 va, AllocInfo &info) {
    UVM_UNMAP_EXTERNAL_PARAMS params{.base = va, .length = info.size, .gpuUuid = target->device->uuid};
    uvm_ioctl(fd_uvm, UVM_UNMAP_EXTERNAL, &params);
    info.mapped_on.erase(std::remove(info.mapped_on.begin(), info.mapped_on.end(), target), info.mapped_on.end());
}

static std::map<NvU64, AllocInfo>::iterator findAllocation(NvU64 va) {
    auto it = allocations.upper_bound(va);
    if (it == allocations.begin()) {
        return allocations.end();
    }
    --it;
    if (va >= it->first && va < it->first + it->second.size) {
        return it;
    }
    return allocations.end();
}

libreCudaStatus_t libreCuCtxCreate_v2(LibreCUcontext *pCtx, int flags, LibreCUdevice device) {
    LIBRECUDA_VALIDATE_INITIALIZED();
    LIBRECUDA_VALIDATE(pCtx != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(device != nullptr, LIBRECUDA_ERROR_INVALID_DEVICE);
    LIBRECUDA_VALIDATE(device->ordinal >= 0 && device->ordinal < num_devices, LIBRECUDA_ERROR_INVALID_DEVICE);
    LIBRECUDA_VALIDATE(
        std::find(device_instances.begin(), device_instances.end(), device->instance) != device_instances.end(),
        LIBRECUDA_ERROR_INVALID_DEVICE
    );

    std::string device_file = getDeviceFile(device);

    auto *ctx = new LibreCUcontext_{
        .flags = flags,
        .device = &public_device_handles[device->ordinal]
    };

    // create & register device device fd
    {
        int device_fd = open(device_file.c_str(), O_RDWR | O_CLOEXEC);
        LIBRECUDA_VALIDATE(device_fd != -1, LIBRECUDA_ERROR_INVALID_DEVICE);
        nv_ioctl_register_fd_t params{
            .ctl_fd = fd_ctl
        };
        NV_IOWR(device_fd, NV_ESC_REGISTER_FD, &params, sizeof(params));
        ctx->device_fd = device_fd;
    }

    // create device rm handle
    NvHandle device_handle{}; {
        NV0080_ALLOC_PARAMETERS params{
            .deviceId = device->instance,
            .hClientShare = root,
            .vaMode = NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES
        };
        RM_ALLOC(fd_ctl, NV01_DEVICE_0, root, root, 0, &params, sizeof(params), &device_handle);
        ctx->device_handle = device_handle;
    }

    // create nv20 sub-device handle
    NvHandle sub_device_handle{};
    RM_ALLOC(fd_ctl, NV20_SUBDEVICE_0, root, device_handle, 0, nullptr, 0, &sub_device_handle);
    ctx->sub_device_handle = sub_device_handle;

    // query framebuffer info
    {
        NV2080_CTRL_FB_GET_INFO_V2_PARAMS params{};
        params.fbInfoListSize = 3;
        params.fbInfoList[0].index = NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE;
        params.fbInfoList[1].index = NV2080_CTRL_FB_INFO_INDEX_BUS_WIDTH;
        params.fbInfoList[2].index = NV2080_CTRL_FB_INFO_INDEX_L2CACHE_SIZE;
        RM_CTRL(fd_ctl, NV2080_CTRL_CMD_FB_GET_INFO_V2, root, sub_device_handle, &params, sizeof(params));
        ctx->device->total_mem = NvU64(params.fbInfoList[0].data) << 10; // reported in KiB
        ctx->device->mem_bus_width = params.fbInfoList[1].data;
        ctx->device->l2_cache_size = params.fbInfoList[2].data;
    }

    // Memory clock: the highest performance level's default DRAM clock. The open headers don't define the clock
    // domain ids of discrete GPUs; on Blackwell domain 0x10 reports the DRAM clock (14001 MHz at the top level of the
    // RTX 5060). The GPC clock is not exposed by this control, so the core clock rate stays a fixed estimate.
    {
        constexpr NvU32 dram_clock_domain = 0x10;
        for (NvU32 level = 0; level < 16; level++) {
            NV2080_CTRL_PERF_GET_LEVEL_INFO_V2_PARAMS params{.level = level, .perfGetClkInfoListSize = 1};
            params.perfGetClkInfoList[0].domain = dram_clock_domain;
            if (rm_ctrl(fd_ctl, NV2080_CTRL_CMD_PERF_GET_LEVEL_INFO_V2, root, sub_device_handle, &params,
                        sizeof(params)) != LIBRECUDA_SUCCESS) {
                break;
            }
            ctx->device->memory_clock_khz = std::max(ctx->device->memory_clock_khz,
                                                     params.perfGetClkInfoList[0].defaultFreq);
        }
        LIBRECUDA_DEBUG("memory clock " + std::to_string(ctx->device->memory_clock_khz) + " kHz");
    }

    // create user mode handle (doorbell); Hopper+ (incl. Blackwell) use HOPPER_USERMODE_A, older GPUs TURING_USERMODE_A
    NvHandle user_mode_handle{};
    if (rm_alloc(fd_ctl, HOPPER_USERMODE_A, root, sub_device_handle, 0, nullptr, 0, &user_mode_handle) != LIBRECUDA_SUCCESS) {
        RM_ALLOC(fd_ctl, TURING_USERMODE_A, root, sub_device_handle, 0, nullptr, 0, &user_mode_handle);
    }

    // create gpu mmio region
    NvU64 gpu_mmio = 0; {
        size_t mmio_sz = 0x10000;
        LIBRECUDA_ERR_PROPAGATE(
            memMapToCpu(
                ctx,
                user_mode_handle, mmio_sz,
                gpu_mmio, 2,
                false
            )
        );
    }
    device->gpu_mmio = reinterpret_cast<NvU32 *>(gpu_mmio);

    // obtain device uuid
    {
        NV2080_CTRL_GPU_GET_GID_INFO_PARAMS params{
            .flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY,
            .length = 16
        };
        RM_CTRL(fd_ctl, NV2080_CTRL_CMD_GPU_GET_GID_INFO, root, sub_device_handle, &params, sizeof(params));
        memcpy(device->uuid.uuid, params.data, 16);
    }

    // create virtual address space
    NvHandle va_space{}; {
        NV_VASPACE_ALLOCATION_PARAMETERS params{
            .index = NV_VASPACE_ALLOCATION_INDEX_GPU_NEW,
            .flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING |
                     NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
            .vaSize = 0x1fffffb000000,
            .bigPageSize = 0, // use system default
            .vaBase = 0x1000,
        };
        RM_ALLOC(fd_ctl, FERMI_VASPACE_A, root, device_handle, 0, &params, sizeof(params), &va_space);
    }

    // register gpu
    {
        UVM_REGISTER_GPU_PARAMS params{
            .gpu_uuid = device->uuid,
            .rmCtrlFd = -1,
        };
        UVM_IOCTL(fd_uvm, UVM_REGISTER_GPU, &params, sizeof(params));
    }

    // register gpu virtual address space
    {
        UVM_REGISTER_GPU_VASPACE_PARAMS params{
            .gpuUuid = device->uuid,
            .rmCtrlFd = fd_ctl,
            .hClient = root,
            .hVaSpace = va_space
        };
        UVM_IOCTL(fd_uvm, UVM_REGISTER_GPU_VASPACE, &params, sizeof(params));
    }
    ctx->va_space_handle = va_space;

    // create channel group
    NvHandle channel_group{}; {
        NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS params{
            .engineType = NV2080_ENGINE_TYPE_GRAPHICS
        };
        RM_ALLOC(fd_ctl, KEPLER_CHANNEL_GROUP_A, root, device_handle, 0, &params, sizeof(params),
                 &channel_group);
    }
    ctx->channel_group = channel_group;

    // create ctx share
    NvHandle ctx_share{}; {
        NV_CTXSHARE_ALLOCATION_PARAMETERS params{
            .hVASpace = ctx->va_space_handle,
            .flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC
        };
        RM_ALLOC(fd_ctl, FERMI_CONTEXT_SHARE_A, root, channel_group, 0, &params, sizeof(params), &ctx_share);
    }
    ctx->ctx_share = ctx_share;

    // get compute class
    {
        NvU32 class_list[NV0080_CTRL_GPU_CLASSLIST_MAX_SIZE];
        NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS params{
            .numClasses = NV0080_CTRL_GPU_CLASSLIST_MAX_SIZE,
            .classList = class_list
        };
        RM_CTRL(fd_ctl, NV0080_CTRL_CMD_GPU_GET_CLASSLIST, root, device_handle, &params, sizeof(params));

        // pick the newest supported class of each kind exposed by the device
        NvU32 compute_class = 0, gpfifo_class = 0, dma_class = 0;
        for (NvU32 i = 0; i < params.numClasses && i < NV0080_CTRL_GPU_CLASSLIST_MAX_SIZE; i++) {
            const NvU32 class_id = class_list[i];
            switch (class_id) {
                case BLACKWELL_COMPUTE_B:
                case BLACKWELL_COMPUTE_A:
                case ADA_COMPUTE_A:
                case AMPERE_COMPUTE_B:
                    compute_class = std::max(compute_class, class_id);
                    break;
                case BLACKWELL_CHANNEL_GPFIFO_B:
                case BLACKWELL_CHANNEL_GPFIFO_A:
                case AMPERE_CHANNEL_GPFIFO_A:
                    gpfifo_class = std::max(gpfifo_class, class_id);
                    break;
                case BLACKWELL_DMA_COPY_B:
                case BLACKWELL_DMA_COPY_A:
                case AMPERE_DMA_COPY_B:
                    dma_class = std::max(dma_class, class_id);
                    break;
                default:
                    break;
            }
        }
        LIBRECUDA_VALIDATE(compute_class != 0 && gpfifo_class != 0 && dma_class != 0, LIBRECUDA_ERROR_NOT_FOUND);
        ctx->device->compute_class = compute_class;
        ctx->device->gpfifo_class = gpfifo_class;
        ctx->device->dma_class = dma_class;
    }

    // get device info
    {
        NV2080_CTRL_GR_GET_INFO_PARAMS params{};
        NV2080_CTRL_GR_INFO grInfoList[NV2080_CTRL_GR_INFO_INDEX_MAX + 1];
        for (int i = 0; i <= NV2080_CTRL_GR_INFO_INDEX_MAX; i++) {
            grInfoList[i].index = i;
        }
        params.grInfoList = grInfoList;
        params.grInfoListSize = NV2080_CTRL_GR_INFO_INDEX_MAX + 1;
        params.grRouteInfo = NV2080_CTRL_GR_ROUTE_INFO{
            .flags = NV2080_CTRL_GR_ROUTE_INFO_FLAGS_TYPE_NONE,
            .route = 0
        };
        RM_CTRL(fd_ctl, NV2080_CTRL_CMD_GR_GET_INFO, root, sub_device_handle, &params,
                sizeof(params));

        int i = 0;
        for (auto &grInfo: device->device_info) {
            grInfo = grInfoList[i++];
        }
        LIBRECUDA_DEBUG("GR info: sm version " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_SM_VERSION].data) +
                        ", cores " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_GPU_CORE_COUNT].data) +
                        ", max sp/sm " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_MAX_SP_PER_SM].data) +
                        ", shader pipe count " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT].data) +
                        ", shader pipe sub count " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT].data) +
                        ", gpcs " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS].data) +
                        ", tpc/gpc " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC].data) +
                        ", sm/tpc " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC].data) +
                        ", warps/sm " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM].data) +
                        ", tensor cores " + std::to_string(grInfoList[NV2080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT].data));
    }
    // Select async copy engines: one that can read system memory (host to device) and one that can write it (device
    // to host). Copies in both directions then run concurrently on separate engines. LIBRECUDA_ASYNC_COPY_ENGINES=0
    // uses the GR copy engine instead, as does a GPU without two such engines.
    NvU32 h2d_copy_engine = 0, d2h_copy_engine = 0; {
        const char *env = getenv("LIBRECUDA_ASYNC_COPY_ENGINES");
        const bool enabled = env == nullptr || std::string(env) != "0";
        NV2080_CTRL_GPU_GET_ENGINES_V2_PARAMS params{};
        if (enabled && rm_ctrl(fd_ctl, NV2080_CTRL_CMD_GPU_GET_ENGINES_V2, root, sub_device_handle, &params,
                               sizeof(params)) == LIBRECUDA_SUCCESS) {
            for (NvU32 i = 0; i < params.engineCount && i < NV2080_GPU_MAX_ENGINES_LIST_SIZE; i++) {
                const NvU32 engine = params.engineList[i];
                const bool is_copy = (engine >= NV2080_ENGINE_TYPE_COPY0 && engine < NV2080_ENGINE_TYPE_COPY0 + 10) ||
                                     (engine >= NV2080_ENGINE_TYPE_COPY10 && engine < NV2080_ENGINE_TYPE_COPY10 + 54);
                if (!is_copy) {
                    continue;
                }
                NV2080_CTRL_CE_GET_CAPS_V2_PARAMS caps{.ceEngineType = engine};
                if (rm_ctrl(fd_ctl, NV2080_CTRL_CMD_CE_GET_CAPS_V2, root, sub_device_handle, &caps, sizeof(caps)) !=
                    LIBRECUDA_SUCCESS || (caps.capsTbl[0] & NV2080_CTRL_CE_CAPS_CE_GRCE_BYTE0)) {
                    continue;
                }
                if (h2d_copy_engine == 0 && (caps.capsTbl[0] & NV2080_CTRL_CE_CAPS_CE_SYSMEM_READ_BYTE0)) {
                    h2d_copy_engine = engine;
                } else if (d2h_copy_engine == 0 && (caps.capsTbl[0] & NV2080_CTRL_CE_CAPS_CE_SYSMEM_WRITE_BYTE0)) {
                    d2h_copy_engine = engine;
                }
            }
        }
        if (h2d_copy_engine == 0 || d2h_copy_engine == 0) {
            h2d_copy_engine = d2h_copy_engine = 0;
        }
        LIBRECUDA_DEBUG("copy engines: host to device " + std::to_string(h2d_copy_engine) + ", device to host " +
                        std::to_string(d2h_copy_engine) + (h2d_copy_engine == 0 ? " (using the GR copy engine)" : ""));
    }

    // number of compute channels: streams are spread over them so that their kernels run concurrently
    const NvU32 n_compute_channels = [] {
        const char *env = getenv("LIBRECUDA_COMPUTE_CHANNELS");
        const long n = env != nullptr ? strtol(env, nullptr, 10) : 4;
        return static_cast<NvU32>(n < 1 ? 1 : (n > MAX_COMPUTE_CHANNELS ? MAX_COMPUTE_CHANNELS : n));
    }();

    // create compute & dma gpfifo
    std::vector<GPFifo> compute_gpfifos(n_compute_channels);
    GPFifo dma_gpfifo{};
    GPFifo dma_d2h_gpfifo{};
    std::vector<NvHandle> copy_channel_groups{}; {
        NvU64 gpfifo_area_va{};
        NvHandle gpfifo_area_handle{};

        // alloc gpfifo area
        {
            // CUDA maps with 0x10d0000
            // but NVOS33_FLAGS_CACHING_TYPE_UNCACHED is ignored in kernel when calling from userspace?
            // also NVOS33_FLAGS_RESERVE_ON_UNMAP_ENABLE allows dangling pointer to remain valid but not mapped to GPU? why?
            LIBRECUDA_ERR_PROPAGATE(
                gpuAlloc(ctx, (n_compute_channels + 2) * GPU_FIFO_SLOT_SIZE, true, true, true,
                    (NVOS33_FLAGS_MAPPING_DIRECT << 16) | (NVOS33_FLAGS_MAP_FIXED_ENABLE << 18) |
                    (NVOS33_FLAGS_RESERVE_ON_UNMAP_ENABLE << 19) |
                    (NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED << 23),
                    &gpfifo_area_va,
                    &gpfifo_area_handle
                )
            );
        }

        // create compute channel 0; it uses the context's subcontext (shared with the GR copy channel). The others are
        // created on first use (ctxEnsureComputeChannel).
        LIBRECUDA_ERR_PROPAGATE(
            makeGpuFifo(
                ctx,
                gpfifo_area_va, gpfifo_area_handle,
                ctx_share, channel_group,
                0, 0x10000, 0,
                &compute_gpfifos[0]
            )
        );
        ctx->gpfifo_area_va = gpfifo_area_va;
        ctx->gpfifo_area_handle = gpfifo_area_handle;

        // create dma gpfifo(s)
        if (h2d_copy_engine == 0) {
            LIBRECUDA_ERR_PROPAGATE(
                makeGpuFifo(
                    ctx,
                    gpfifo_area_va, gpfifo_area_handle,
                    ctx_share, channel_group,
                    n_compute_channels * GPU_FIFO_SLOT_SIZE, 0x10000, 0,
                    &dma_gpfifo
                )
            );
        } else {
            // each async copy engine channel lives in its own channel group bound to that engine
            const std::pair<NvU32, GPFifo *> copy_channels[] = {
                {h2d_copy_engine, &dma_gpfifo}, {d2h_copy_engine, &dma_d2h_gpfifo}
            };
            NvU64 offset = n_compute_channels * GPU_FIFO_SLOT_SIZE;
            for (const auto &[engine, fifo]: copy_channels) {
                NvHandle copy_channel_group{};
                NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS params{
                    .hVASpace = ctx->va_space_handle,
                    .engineType = engine
                };
                RM_ALLOC(fd_ctl, KEPLER_CHANNEL_GROUP_A, root, device_handle, 0, &params, sizeof(params),
                         &copy_channel_group);
                LIBRECUDA_ERR_PROPAGATE(
                    makeGpuFifo(
                        ctx,
                        gpfifo_area_va, gpfifo_area_handle,
                        0, copy_channel_group,
                        offset, 0x10000, engine,
                        fifo
                    )
                );
                copy_channel_groups.push_back(copy_channel_group);
                offset += GPU_FIFO_SLOT_SIZE;
            }
        }
    }
    // enable gpfifo schedule
    {
        NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS params{.bEnable = true};
        RM_CTRL(fd_ctl, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, root, channel_group, &params, sizeof(params));
        for (NvHandle copy_channel_group: copy_channel_groups) {
            RM_CTRL(fd_ctl, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, root, copy_channel_group, &params, sizeof(params));
        }
    }
    ctx->compute_gpfifos = std::move(compute_gpfifos);
    ctx->dma_gpfifo = dma_gpfifo;
    ctx->dma_d2h_gpfifo = dma_d2h_gpfifo;
    ctx->has_d2h_channel = d2h_copy_engine != 0;
    ctx->copy_channel_groups = copy_channel_groups;

    // allocate event signal pool
    {
        NvU64 pool_va{};
        LIBRECUDA_ERR_PROPAGATE(
            gpuSystemAlloc(ctx, sizeof(NvSignal) * EVENT_SIGNAL_POOL_SIZE, true, 0, &pool_va)
        );
        ctx->event_signal_pool = reinterpret_cast<NvSignal *>(pool_va);
        ctx->free_event_signals.reserve(EVENT_SIGNAL_POOL_SIZE);
        for (NvU32 i = EVENT_SIGNAL_POOL_SIZE; i > 0; i--) {
            ctx->free_event_signals.push_back(i - 1);
        }
    }

    *pCtx = ctx;
    current_ctx = ctx;
    device_contexts.emplace(ctx->device->ordinal, ctx);

    // load kernels
    LIBRECUDA_ERR_PROPAGATE(loadMemcpyKernelsIfNeeded(device));

    // page-locked host memory and system memory (event signals) of the other contexts are usable here too
    {
        std::lock_guard alloc_lock(alloc_mutex);
        for (auto &[va, info]: allocations) {
            if (info.kind != AllocKind::DEVICE && info.ctx != ctx) {
                LIBRECUDA_ERR_PROPAGATE(mapOnContext(ctx, va, info));
            }
        }
        live_contexts.push_back(ctx);
    }

    LIBRECUDA_SUCCEED();
}

static void freeContextAllocations(LibreCUcontext ctx);

libreCudaStatus_t libreCuCtxDestroy(LibreCUcontext ctx) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT);

    // Let the GPU finish the work of all streams; it must no longer access the memory freed below. Errors are
    // ignored: a context whose channels faulted must still be destroyable.
    for (NvCommandQueue *queue: ctx->queues) {
        if (ctx->channel_error == LIBRECUDA_SUCCESS) {
            queue->awaitExecution();
        }
        queue->detachFromContext();
    }
    ctx->queues.clear();

    // stop scheduling the channels
    {
        NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS params{.bEnable = false};
        rm_ctrl(fd_ctl, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, root, ctx->channel_group, &params, sizeof(params));
        for (NvHandle copy_channel_group: ctx->copy_channel_groups) {
            rm_ctrl(fd_ctl, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, root, copy_channel_group, &params, sizeof(params));
        }
    }

    // UVM keeps references to the channels and the VA space; release them before RM frees the objects
    for (GPFifo *fifo: ctxChannels(ctx)) {
        if (fifo->channel_handle != 0) {
            UVM_UNREGISTER_CHANNEL_PARAMS params{.hClient = root, .hChannel = fifo->channel_handle};
            uvm_ioctl(fd_uvm, UVM_UNREGISTER_CHANNEL, &params);
        }
    }
    // the queues are idle (or the channels faulted): print what the kernels wrote
    ctxFlushPrintf(ctx);
    ctxReleasePrintf(ctx);
    releaseMemcpyKernels(ctx->device);
    {
        // the context leaves the peer relations; unregistering its GPU's VA space removes the other contexts'
        // allocations mapped on it
        std::lock_guard alloc_lock(alloc_mutex);
        live_contexts.erase(std::remove(live_contexts.begin(), live_contexts.end(), ctx), live_contexts.end());
        for (auto &[va, info]: allocations) {
            info.mapped_on.erase(std::remove(info.mapped_on.begin(), info.mapped_on.end(), ctx), info.mapped_on.end());
        }
        for (LibreCUcontext other: live_contexts) {
            auto &accessors = other->peer_accessors;
            accessors.erase(std::remove(accessors.begin(), accessors.end(), ctx), accessors.end());
        }
        for (auto it = uvm_peer_pairs.begin(); it != uvm_peer_pairs.end();) {
            if (it->first == ctx->device->gpu_id || it->second == ctx->device->gpu_id) {
                LibreCUdevice_ *a{}, *b{};
                for (auto &d: public_device_handles) {
                    a = d.gpu_id == it->first ? &d : a;
                    b = d.gpu_id == it->second ? &d : b;
                }
                if (a != nullptr && b != nullptr) {
                    UVM_DISABLE_PEER_ACCESS_PARAMS params{.gpuUuidA = a->uuid, .gpuUuidB = b->uuid};
                    uvm_ioctl(fd_uvm, UVM_DISABLE_PEER_ACCESS, &params);
                }
                it = uvm_peer_pairs.erase(it);
            } else {
                ++it;
            }
        }
        ctx->peer_accessors.clear();
    }
    freeContextAllocations(ctx);
    {
        UVM_UNREGISTER_GPU_VASPACE_PARAMS params{.gpuUuid = ctx->device->uuid};
        uvm_ioctl(fd_uvm, UVM_UNREGISTER_GPU_VASPACE, &params);
    }

    // freeing the device frees all RM objects created under it: sub-device, user mode (doorbell), VA space,
    // channel groups, channels and remaining memory objects
    rmFree(root, ctx->device_handle);
    if (ctx->device->gpu_mmio != nullptr) {
        munmap(ctx->device->gpu_mmio, 0x10000);
        ctx->device->gpu_mmio = nullptr;
    }
    {
        UVM_UNREGISTER_GPU_PARAMS params{.gpu_uuid = ctx->device->uuid};
        uvm_ioctl(fd_uvm, UVM_UNREGISTER_GPU, &params);
    }

    // set current context to none, if current context is equal to the context to destroy
    if (current_ctx == ctx) {
        current_ctx = nullptr;
    }
    auto device_ctx_it = device_contexts.find(ctx->device->ordinal);
    if (device_ctx_it != device_contexts.end() && device_ctx_it->second == ctx) {
        device_contexts.erase(device_ctx_it);
    }

    // close device fd
    close(ctx->device_fd);

    delete ctx;

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxSetCurrent(LibreCUcontext ctx) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT);
    current_ctx = ctx;
    LIBRECUDA_SUCCEED();
}

static std::string hexString(NvU64 value) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(value));
    return buf;
}


libreCudaStatus_t memUVMMap(LibreCUcontext ctx, NvHandle memoryHandle, NvU64 virtualAddress, size_t size) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(memoryHandle != 0, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(virtualAddress != 0, LIBRECUDA_ERROR_INVALID_VALUE);

    // create external range
    {
        UVM_CREATE_EXTERNAL_RANGE_PARAMS params{
            .base = virtualAddress,
            .length = size
        };
        UVM_IOCTL(fd_uvm, UVM_CREATE_EXTERNAL_RANGE, &params, sizeof(params));
    }

    // map obtained memory handle to the target virtual address
    {
        UVM_MAP_EXTERNAL_ALLOCATION_PARAMS params{
            .base = virtualAddress,
            .length = size,
            .perGpuAttributes = {
                {
                    .gpuUuid = ctx->device->uuid,
                    .gpuMappingType = UvmGpuMappingTypeReadWriteAtomic
                }
            },
            .gpuAttributesCount = 1,
            .rmCtrlFd = fd_ctl,
            .hClient = root,
            .hMemory = memoryHandle
        };
        UVM_IOCTL(fd_uvm, UVM_MAP_EXTERNAL_ALLOCATION, &params, sizeof(params));
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t
gpuAlloc(LibreCUcontext ctx, size_t size, bool physicalContiguity, bool hugePages, bool mapToCpu, NvU32 mapFlags,
         NvU64 *pVaOut, NvHandle *pMemoryHandleOut) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(pVaOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    NvU64 alignment = hugePages ? (2 << 20) : (4 << 10);
    size = ceilDiv(size, alignment) * alignment; // round up to next multiple of alignment

    // perform allocation
    NvHandle memory_handle{}; {
        NV_MEMORY_ALLOCATION_PARAMS params{
            .owner = root,
            .attr = static_cast<NvU32>((hugePages ? (NVOS32_ATTR_PAGE_SIZE_HUGE << 23) : 0) |
                                       ((physicalContiguity
                                             ? NVOS32_ATTR_PHYSICALITY_CONTIGUOUS
                                             : NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27)),
            .attr2 = static_cast<NvU32>(((NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) |
                                         (NVOS32_ATTR2_GPU_CACHEABLE_YES << 2) |
                                         (hugePages ? (NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB << 20) : 0))),
            .format = NV_MMU_PTE_KIND_GENERIC_MEMORY,
            .size = size,
            .alignment = alignment,
            .offset = 0,
            .limit = size - 1
        };
        if (rm_alloc(fd_ctl, NV1_MEMORY_USER, root, ctx->device_handle, 0, &params, sizeof(params),
                     &memory_handle) != LIBRECUDA_SUCCESS) {
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_OUT_OF_MEMORY);
        }
    }

    // bump allocate virtual address
    NvU64 va_address = bump_alloc_virtual_addr(ctx, size, alignment);

    // map obtained virtual address to refer to the memory allocated
    LIBRECUDA_ERR_PROPAGATE(memUVMMap(ctx, memory_handle, va_address, size));

    // map to cpu if requested
    if (mapToCpu) {
        LIBRECUDA_ERR_PROPAGATE(memMapToCpu(ctx, memory_handle, size, va_address, mapFlags, false));
    }

    *pVaOut = va_address;

    if (pMemoryHandleOut != nullptr) {
        *pMemoryHandleOut = memory_handle;
    }
    LIBRECUDA_DEBUG("alloc device [" + hexString(va_address) + ", " + hexString(va_address + size) + ")");
    AllocInfo &info = allocations[va_address] = AllocInfo{
        .ctx = ctx,
        .size = size,
        .memory_handle = memory_handle,
        .kind = AllocKind::DEVICE,
        .cpu_mapped = mapToCpu
    };
    // peers with access to the context's memory see it too
    for (LibreCUcontext accessor: ctx->peer_accessors) {
        LIBRECUDA_ERR_PROPAGATE(mapOnContext(accessor, va_address, info));
    }
    LIBRECUDA_SUCCEED();
}

static bool isSmallAllocChunk(LibreCUcontext ctx, NvU64 va) {
    for (const auto &class_chunks: ctx->small_allocs.chunks) {
        for (const SmallAllocPool::Chunk *chunk: class_chunks) {
            if (chunk->base == va) {
                return true;
            }
        }
    }
    return false;
}

static bool smallAllocsEnabled() {
    static const bool enabled = [] {
        const char *env = getenv("LIBRECUDA_SMALL_ALLOC_POOL");
        return env == nullptr || std::string(env) != "0";
    }();
    return enabled;
}

static libreCudaStatus_t smallAlloc(LibreCUcontext ctx, size_t size, NvU64 *pVaOut) {
    SmallAllocPool &pool = ctx->small_allocs;
    NvU32 shift = SmallAllocPool::MIN_SLOT_SHIFT;
    while ((NvU64(1) << shift) < size) {
        shift++;
    }
    const NvU32 size_class = shift - SmallAllocPool::MIN_SLOT_SHIFT;
    SmallAllocPool::Chunk *chunk = nullptr;
    for (SmallAllocPool::Chunk *candidate: pool.chunks[size_class]) {
        if (!candidate->free_slots.empty()) {
            chunk = candidate;
            break;
        }
    }
    if (chunk == nullptr) {
        NvU64 base{};
        LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, SmallAllocPool::CHUNK_SIZE, false, true, false, 0, &base));
        const NvU32 num_slots = NvU32(SmallAllocPool::CHUNK_SIZE >> shift);
        chunk = new SmallAllocPool::Chunk{.base = base, .size_class = size_class, .used_slots = 0};
        // hand out low addresses first
        chunk->free_slots.reserve(num_slots);
        for (NvU32 i = num_slots; i > 0; i--) {
            chunk->free_slots.push_back(i - 1);
        }
        pool.chunks[size_class].push_back(chunk);
    }
    const NvU32 slot = chunk->free_slots.back();
    chunk->free_slots.pop_back();
    chunk->used_slots++;
    const NvU64 va = chunk->base + (NvU64(slot) << shift);
    pool.live.emplace(va, chunk);
    *pVaOut = va;
    LIBRECUDA_SUCCEED();
}

/**
 * @return true if va was a small allocation (it is then freed)
 */
static bool smallFree(LibreCUcontext ctx, NvU64 va) {
    SmallAllocPool &pool = ctx->small_allocs;
    auto it = pool.live.find(va);
    if (it == pool.live.end()) {
        return false;
    }
    SmallAllocPool::Chunk *chunk = it->second;
    pool.live.erase(it);
    const NvU32 shift = chunk->size_class + SmallAllocPool::MIN_SLOT_SHIFT;
    chunk->free_slots.push_back(NvU32((va - chunk->base) >> shift));
    chunk->used_slots--;
    if (chunk->used_slots == 0) {
        // keep one empty chunk per size class to avoid reallocating it on alloc/free cycles
        auto &class_chunks = pool.chunks[chunk->size_class];
        const bool other_empty = std::any_of(class_chunks.begin(), class_chunks.end(),
                                             [chunk](const SmallAllocPool::Chunk *c) {
                                                 return c != chunk && c->used_slots == 0;
                                             });
        if (other_empty) {
            class_chunks.erase(std::find(class_chunks.begin(), class_chunks.end(), chunk));
            gpuFree(ctx, chunk->base);
            delete chunk;
        }
    }
    return true;
}

libreCudaStatus_t libreCuMemAlloc(void **pDevicePointer, size_t bytesize, bool mapToCpu) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(pDevicePointer != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(bytesize > 0, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    NvU64 va_address{};
    if (!mapToCpu && bytesize <= (NvU64(1) << SmallAllocPool::MAX_SLOT_SHIFT) && smallAllocsEnabled()) {
        LIBRECUDA_ERR_PROPAGATE(smallAlloc(current_ctx, bytesize, &va_address));
    } else {
        bool hugePages = bytesize > (16 << 20);
        LIBRECUDA_ERR_PROPAGATE(
            gpuAlloc(current_ctx, bytesize, false, hugePages, mapToCpu, 0, &va_address)
        );
    }
    *pDevicePointer = reinterpret_cast<void *>(va_address);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t gpuFree(LibreCUcontext ctx, NvU64 virtualAddress) {
    std::lock_guard alloc_lock(alloc_mutex);
    auto it = allocations.find(virtualAddress);
    LIBRECUDA_VALIDATE(it != allocations.end(), LIBRECUDA_ERROR_INVALID_VALUE);
    const AllocInfo info = it->second;
    allocations.erase(it);
    LIBRECUDA_DEBUG("free [" + hexString(virtualAddress) + ", " + hexString(virtualAddress + info.size) + ")");

    // unmap from the GPU; this drops UVM's reference to the memory so that it is actually released
    {
        UVM_FREE_PARAMS params{.base = virtualAddress};
        UVM_IOCTL(fd_uvm, UVM_FREE, &params, sizeof(params));
    }
    if (info.cpu_mapped) {
        munmap(reinterpret_cast<void *>(virtualAddress), info.size);
    }
    rmFree(info.ctx->device_handle, info.memory_handle);
    LIBRECUDA_SUCCEED();
}

/**
 * Frees all allocations made on the context (device, system and host memory)
 */
static void freeContextAllocations(LibreCUcontext ctx) {
    std::lock_guard alloc_lock(alloc_mutex);
    std::vector<NvU64> addresses{};
    for (const auto &[va, info]: allocations) {
        if (info.ctx == ctx) {
            addresses.push_back(va);
        }
    }
    for (NvU64 va: addresses) {
        gpuFree(ctx, va);
    }
}

libreCudaStatus_t
gpuSystemAlloc(LibreCUcontext ctx, size_t size, bool mapToCpu, NvU32 mapFlags, NvU64 *pVaOut,
               NvHandle *pMemHandleOut, bool cpuCached) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(pVaOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    size = ceilDiv(size, 4 << 10) * (4 << 10);

    NvHandle memory_handle{}; {
        NV_MEMORY_ALLOCATION_PARAMS params{
            .owner = root,
            .type = NVOS32_TYPE_NOTIFIER,
            .flags = (NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED |
                      NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED),
            .attr = (NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27) | (NVOS32_ATTR_LOCATION_PCI << 25) |
                    (NvU32(cpuCached ? NVOS32_ATTR_COHERENCY_CACHED : NVOS32_ATTR_COHERENCY_UNCACHED) << 29),
            .attr2 = (NVOS32_ATTR2_ZBC_PREFER_NO_ZBC << 0) | (NVOS32_ATTR2_GPU_CACHEABLE_NO << 2),
            .format = 6,
            .size = size,
            .alignment = (4 << 10),
            .offset = 0,
            .limit = size - 1
        };
        if (rm_alloc(fd_ctl, NV1_MEMORY_SYSTEM, root, ctx->device_handle, 0, &params, sizeof(params),
                     &memory_handle) != LIBRECUDA_SUCCESS) {
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_OUT_OF_MEMORY);
        }
    }

    // bump allocate virtual address
    NvU64 va_address = bump_alloc_virtual_addr(ctx, size, 4 << 10);

    // map to cpu if requested
    if (mapToCpu) {
        LIBRECUDA_ERR_PROPAGATE(memMapToCpu(ctx, memory_handle, size, va_address, mapFlags, true));
    }

    // map obtained virtual address to refer to the memory allocated
    LIBRECUDA_ERR_PROPAGATE(memUVMMap(ctx, memory_handle, va_address, size));

    *pVaOut = va_address;

    if (pMemHandleOut != nullptr) {
        *pMemHandleOut = memory_handle;
    }
    LIBRECUDA_DEBUG("alloc system [" + hexString(va_address) + ", " + hexString(va_address + size) + ")");
    AllocInfo &info = allocations[va_address] = AllocInfo{
        .ctx = ctx,
        .size = size,
        .memory_handle = memory_handle,
        .kind = AllocKind::SYSTEM,
        .cpu_mapped = mapToCpu
    };
    // other GPUs wait for and signal the context's events (signals live in system memory)
    for (LibreCUcontext other: live_contexts) {
        if (other != ctx) {
            LIBRECUDA_ERR_PROPAGATE(mapOnContext(other, va_address, info));
        }
    }
    LIBRECUDA_SUCCEED();
}

/**
 * Allocates page-locked host memory: anonymous memory is mapped at a reserved GPU virtual address,
 * registered with RM as an OS descriptor (pinning the pages) and then mapped into the GPU address space at the same
 * address. Unlike RM system memory mappings, the CPU accesses this memory through regular cached page tables.
 */
// handles for RM objects created with client provided handles; RM generated handles use a different range
static NvHandle next_os_descriptor_handle = 0xcafe0000;

libreCudaStatus_t gpuHostAlloc(LibreCUcontext ctx, size_t size, NvU64 *pVaOut) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(pVaOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    const NvU64 alignment = size >= (2 << 20) ? (2 << 20) : (4 << 10);
    size = ceilDiv(size, 4 << 10) * (4 << 10);
    NvU64 va_address = bump_alloc_virtual_addr(ctx, size, alignment);

    void *mapped = mmap(reinterpret_cast<void *>(va_address), size, PROT_READ | PROT_WRITE,
                        MAP_FIXED | MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    LIBRECUDA_VALIDATE(mapped != MAP_FAILED, LIBRECUDA_ERROR_OUT_OF_MEMORY);
    // Populate every page with a write access before RM pins them. Pinning pages that were never touched can
    // pin pages the GPU maps read-only, so copy engine writes into the buffer fault (Xid 31, PRIV_VIOLATION).
    memset(mapped, 0, size);

    nv_ioctl_nvos02_parameters_with_fd params{
        .params = {
            .hRoot = root,
            .hObjectParent = ctx->device_handle,
            // OS descriptors are created with a client provided handle
            .hObjectNew = next_os_descriptor_handle++,
            .hClass = NV01_MEMORY_SYSTEM_OS_DESCRIPTOR,
            .flags = (NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (NVOS02_FLAGS_COHERENCY_CACHED << 12) |
                     (NVOS02_FLAGS_MAPPING_NO_MAP << 30),
            .pMemory = reinterpret_cast<NvP64>(va_address),
            .limit = size - 1
        },
        .fd = -1
    };
    if (nv_iowr(ctx->device_fd, NV_ESC_RM_ALLOC_MEMORY, &params, sizeof(params)) != 0 || params.params.status != 0) {
        LIBRECUDA_DEBUG("Failed to register host memory with status " + std::to_string(params.params.status));
        munmap(mapped, size);
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_OUT_OF_MEMORY);
    }
    const NvHandle memory_handle = params.params.hObjectNew;
    LIBRECUDA_ERR_PROPAGATE(memUVMMap(ctx, memory_handle, va_address, size));

    LIBRECUDA_DEBUG("alloc host [" + hexString(va_address) + ", " + hexString(va_address + size) + ")");
    AllocInfo &info = allocations[va_address] = AllocInfo{
        .ctx = ctx,
        .size = size,
        .memory_handle = memory_handle,
        .kind = AllocKind::HOST,
        .cpu_mapped = true
    };
    // like CUDA's page-locked memory, usable from every device
    for (LibreCUcontext other: live_contexts) {
        if (other != ctx) {
            LIBRECUDA_ERR_PROPAGATE(mapOnContext(other, va_address, info));
        }
    }
    *pVaOut = va_address;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuMemAllocHost(void **pHostPointer, size_t bytesize) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(pHostPointer != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(bytesize > 0, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    NvU64 va{};
    LIBRECUDA_ERR_PROPAGATE(gpuHostAlloc(current_ctx, bytesize, &va));
    *pHostPointer = reinterpret_cast<void *>(va);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuMemFreeHost(void *hostPointer) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(hostPointer != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    auto it = allocations.find(reinterpret_cast<NvU64>(hostPointer));
    LIBRECUDA_VALIDATE(it != allocations.end() && it->second.kind == AllocKind::HOST, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(gpuFree(it->second.ctx, reinterpret_cast<NvU64>(hostPointer)));
    LIBRECUDA_SUCCEED();
}

bool libreCuMemIsPinnedHost(const void *ptr) {
    std::lock_guard alloc_lock(alloc_mutex);
    auto it = findAllocation(reinterpret_cast<NvU64>(ptr));
    return it != allocations.end() && it->second.kind == AllocKind::HOST;
}

bool libreCuMemIsDevice(const void *ptr) {
    std::lock_guard alloc_lock(alloc_mutex);
    auto it = findAllocation(reinterpret_cast<NvU64>(ptr));
    return it != allocations.end() && it->second.kind == AllocKind::DEVICE;
}

bool isDevicePtr(void *ptr) {
    std::lock_guard alloc_lock(alloc_mutex);
    return libreCuMemIsDevice(ptr);
}

bool isHostMappedPtr(void *ptr) {
    std::lock_guard alloc_lock(alloc_mutex);
    auto it = findAllocation(reinterpret_cast<NvU64>(ptr));
    return it != allocations.end() && it->second.kind == AllocKind::DEVICE && it->second.cpu_mapped;
}

LibreCUcontext libreCuMemGetContext(const void *ptr) {
    std::lock_guard alloc_lock(alloc_mutex);
    auto it = findAllocation(reinterpret_cast<NvU64>(ptr));
    if (it == allocations.end() || (it->second.kind != AllocKind::DEVICE && it->second.kind != AllocKind::HOST)) {
        return nullptr;
    }
    return it->second.ctx;
}

bool libreCuMemIsAccessibleFrom(const void *ptr, LibreCUcontext ctx) {
    std::lock_guard alloc_lock(alloc_mutex);
    auto it = findAllocation(reinterpret_cast<NvU64>(ptr));
    if (it == allocations.end()) {
        return false;
    }
    const AllocInfo &info = it->second;
    return info.ctx == ctx || std::find(info.mapped_on.begin(), info.mapped_on.end(), ctx) != info.mapped_on.end();
}

libreCudaStatus_t libreCuDeviceGetP2PCapabilities(LibreCUdevice device, LibreCUdevice peer, bool *pRead, bool *pWrite,
                                                  bool *pAtomics) {
    LIBRECUDA_VALIDATE(device != nullptr && peer != nullptr && pRead != nullptr && pWrite != nullptr &&
                       pAtomics != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pRead = *pWrite = *pAtomics = false;
    if (device->gpu_id == peer->gpu_id) {
        LIBRECUDA_SUCCEED();
    }
    NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS params{};
    params.gpuIds[0] = device->gpu_id;
    params.gpuIds[1] = peer->gpu_id;
    params.gpuCount = 2;
    RM_CTRL(fd_ctl, NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS, root, root, &params, sizeof(params));
    *pRead = (params.p2pCaps >> 1) & 1;    // READS_SUPPORTED
    *pWrite = params.p2pCaps & 1;          // WRITES_SUPPORTED
    *pAtomics = (params.p2pCaps >> 4) & 1; // ATOMICS_SUPPORTED
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxEnablePeerAccess(LibreCUcontext peerContext, unsigned int flags) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LibreCUcontext ctx = current_ctx;
    LIBRECUDA_VALIDATE(peerContext != nullptr && flags == 0, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(peerContext != ctx, LIBRECUDA_ERROR_INVALID_DEVICE);
    bool read{}, write{}, atomics{};
    LIBRECUDA_ERR_PROPAGATE(libreCuDeviceGetP2PCapabilities(ctx->device, peerContext->device, &read, &write, &atomics));
    LIBRECUDA_VALIDATE(read && write, LIBRECUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
    std::lock_guard alloc_lock(alloc_mutex);
    auto &accessors = peerContext->peer_accessors;
    LIBRECUDA_VALIDATE(std::find(accessors.begin(), accessors.end(), ctx) == accessors.end(),
                       LIBRECUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
    const auto pair = std::minmax(ctx->device->gpu_id, peerContext->device->gpu_id);
    if (uvm_peer_pairs.count(pair) == 0) {
        // UVM enables BAR1 (and NVLink) peers itself when it registers the GPUs: the request then fails with
        // NV_ERR_INVALID_DEVICE, and the pair stays UVM's to disable
        UVM_ENABLE_PEER_ACCESS_PARAMS params{.gpuUuidA = ctx->device->uuid, .gpuUuidB = peerContext->device->uuid};
        if (uvm_ioctl(fd_uvm, UVM_ENABLE_PEER_ACCESS, &params) != 0) {
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_UNKNOWN);
        }
        if (params.rmStatus == NV_OK) {
            uvm_peer_pairs.insert(pair);
        } else if (params.rmStatus != NV_ERR_INVALID_DEVICE) {
            LIBRECUDA_DEBUG("UVM_ENABLE_PEER_ACCESS failed with status " + std::to_string(params.rmStatus));
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
        }
    }
    for (auto &[va, info]: allocations) {
        if (info.ctx == peerContext && info.kind == AllocKind::DEVICE) {
            LIBRECUDA_ERR_PROPAGATE(mapOnContext(ctx, va, info));
        }
    }
    accessors.push_back(ctx);
    LIBRECUDA_DEBUG("peer access enabled: device " + std::to_string(ctx->device->ordinal) + " -> device " +
                    std::to_string(peerContext->device->ordinal));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxDisablePeerAccess(LibreCUcontext peerContext) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LibreCUcontext ctx = current_ctx;
    LIBRECUDA_VALIDATE(peerContext != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    std::lock_guard alloc_lock(alloc_mutex);
    auto &accessors = peerContext->peer_accessors;
    auto it = std::find(accessors.begin(), accessors.end(), ctx);
    LIBRECUDA_VALIDATE(it != accessors.end(), LIBRECUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
    accessors.erase(it);
    for (auto &[va, info]: allocations) {
        if (info.ctx == peerContext && info.kind == AllocKind::DEVICE) {
            unmapFromContext(ctx, va, info);
        }
    }
    LIBRECUDA_SUCCEED();
}

bool isGpuAccessiblePtr(const void *ptr) {
    std::lock_guard alloc_lock(alloc_mutex);
    return findAllocation(reinterpret_cast<NvU64>(ptr)) != allocations.end();
}

libreCudaStatus_t libreCuMemFree(void *devicePointer) {
    std::lock_guard alloc_lock(alloc_mutex);
    LIBRECUDA_VALIDATE(devicePointer != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    // memory of any context can be freed (unified addressing): the context that allocated it frees it
    auto owner_it = findAllocation(reinterpret_cast<NvU64>(devicePointer));
    LibreCUcontext ctx = owner_it != allocations.end() ? owner_it->second.ctx : current_ctx;
    if (smallFree(ctx, reinterpret_cast<NvU64>(devicePointer))) {
        LIBRECUDA_SUCCEED();
    }
    auto it = allocations.find(reinterpret_cast<NvU64>(devicePointer));
    LIBRECUDA_VALIDATE(it != allocations.end() && it->second.kind == AllocKind::DEVICE, LIBRECUDA_ERROR_INVALID_VALUE);
    // the base of a pool chunk is not a user allocation unless it is a live slot (handled above)
    LIBRECUDA_VALIDATE(!isSmallAllocChunk(ctx, it->first), LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(gpuFree(ctx, reinterpret_cast<NvU64>(devicePointer)));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuMemGetInfo(size_t *pFreeOut, size_t *pTotalOut) {
    LIBRECUDA_ENSURE_CTX_VALID();
    NV2080_CTRL_FB_GET_INFO_V2_PARAMS params{};
    params.fbInfoListSize = 1;
    params.fbInfoList[0].index = NV2080_CTRL_FB_INFO_INDEX_HEAP_FREE;
    RM_CTRL(fd_ctl, NV2080_CTRL_CMD_FB_GET_INFO_V2, root, current_ctx->sub_device_handle, &params, sizeof(params));
    if (pFreeOut != nullptr) {
        *pFreeOut = size_t(params.fbInfoList[0].data) << 10;
    }
    if (pTotalOut != nullptr) {
        *pTotalOut = current_ctx->device->total_mem;
    }
    LIBRECUDA_SUCCEED();
}


libreCudaStatus_t libreCuMemCpy(void *dst, void *src, size_t byteCount, LibreCUstream stream, bool async) {
    LIBRECUDA_VALIDATE(dst != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(src != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    const bool dst_gpu = isGpuAccessiblePtr(dst), src_gpu = isGpuAccessiblePtr(src);
    if (isDevicePtr(dst) && isDevicePtr(src) && memcpyD2DKernelsAvailable()) {
        // within device memory, kernels reach much higher bandwidth than the copy engine
        LIBRECUDA_ERR_PROPAGATE(memcpyD2D(dst, src, byteCount, stream, false));
    } else if (dst_gpu && src_gpu) {
        LIBRECUDA_ERR_PROPAGATE(stream->command_queue->gpuMemcpy(dst, src, byteCount, async));
    } else if (dst_gpu || src_gpu) {
        // pageable host memory: stage through pinned buffers
        LIBRECUDA_ERR_PROPAGATE(stream->command_queue->stagedMemcpy(dst, src, byteCount, !dst_gpu));
    } else {
        memcpy(dst, src, byteCount);
    }
    LIBRECUDA_SUCCEED();
}


// Relocation types of CUDA ELF images
#define R_CUDA_32 0x1
#define R_CUDA_64 0x2
// 32 / 64 bit generic address in data (e.g. tables of string pointers; generic = global address here)
#define R_CUDA_G32 0x3
#define R_CUDA_G64 0x4
#define R_CUDA_ABS32_LO_32 0x38
#define R_CUDA_ABS32_HI_32 0x39
// code address in a 128 bit CALL.ABS instruction (separately compiled code calling across translation units)
#define R_CUDA_ABS55_16_34 0x4b
// the address of a kernel (device code taking a kernel's address to launch it): 32 bit immediate in the upper half of
// a 64 bit instruction word, like R_CUDA_ABS32_LO_32 / HI_32 (seen on UMOVs nvcc emits for device side launches)
#define R_CUDA_KERNEL_ADDR32_LO_32 0x3e
#define R_CUDA_KERNEL_ADDR32_HI_32 0x3f
// the address of a kernel in data (e.g. __device__ void *table[] = {kernel}): 64 bit
#define R_CUDA_KERNEL_ADDR64 0x23

// Attributes in .nv.info / .nv.info.<func_name> sections.
// Each attribute starts with a 32-bit key: low byte = format, next byte = attribute id, high 16 bits = value or size.
// Formats 0x01-0x03 carry everything inline; format 0x04 is followed by a payload of `size` bytes.
#define EIATTR_FORMAT_FREE 0x04

// module wide attributes (payload: symbol index, value)
#define EIATTR_FRAME_SIZE 0x11
#define EIATTR_MIN_STACK_SIZE 0x12
#define EIATTR_MAX_STACK_SIZE 0x23
#define EIATTR_REGCOUNT 0x2f

// per function attributes
#define EIATTR_MAX_THREADS 0x05
#define EIATTR_PARAM_CBANK 0x0a
#define EIATTR_EXTERNS 0x0f
#define EIATTR_REQNTID 0x10
#define EIATTR_KPARAM_INFO 0x17
#define EIATTR_CBANK_PARAM_SIZE 0x19
#define EIATTR_CTA_PER_CLUSTER 0x3d
#define EIATTR_EXPLICIT_CLUSTER 0x3e

// Bytes reserved by the ABI at the top of the per thread local memory window; the initial stack pointer
// written to constant bank 0 is 0xfffdc0 = 16MiB - 0x240
#define LOCAL_MEM_ABI_RESERVED 0x240

#define MODULE_SECTION_ALIGN 256

static std::string elfSymbolName(const ELFIO::elfio &elf, size_t symtabIdx, NvU32 symIdx) {
    ELFIO::symbol_section_accessor symbols{elf, elf.sections[symtabIdx]};
    std::string name{};
    ELFIO::Elf64_Addr value{};
    ELFIO::Elf_Xword size{};
    unsigned char bind{}, type{}, other{};
    ELFIO::Elf_Half section_index{};
    symbols.get_symbol(symIdx, name, value, size, bind, type, section_index, other);
    return name;
}

/**
 * Iterates the attributes of an .nv.info section.
 * The callback receives (attribute id, format, inline value / payload size, payload pointer or nullptr)
 */
template<typename F>
static libreCudaStatus_t forEachNvInfoAttribute(const ELFIO::section *section, F callback) {
    const char *data = section->get_data();
    const size_t size = section->get_size();
    for (size_t off = 0; off + sizeof(NvU32) <= size;) {
        NvU32 key{};
        memcpy(&key, data + off, sizeof(NvU32));
        if (key == 0xffffffff) {
            break;
        }
        const NvU8 format = key & 0xff;
        const NvU8 attribute = (key >> 8) & 0xff;
        const NvU16 value = (key >> 16) & 0xffff;
        if (format == EIATTR_FORMAT_FREE) {
            LIBRECUDA_VALIDATE(off + sizeof(NvU32) + value <= size, LIBRECUDA_ERROR_INVALID_IMAGE);
            callback(attribute, format, value, reinterpret_cast<const NvU32 *>(data + off + sizeof(NvU32)));
            off += sizeof(NvU32) + value;
        } else if (format >= 0x01 && format <= 0x03) {
            callback(attribute, format, value, nullptr);
            off += sizeof(NvU32);
        } else {
            LIBRECUDA_DEBUG("Unknown .nv.info attribute format " + std::to_string(format));
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_IMAGE);
        }
    }
    LIBRECUDA_SUCCEED();
}

static bool startsWith(const std::string &str, const char *prefix) {
    return str.rfind(prefix, 0) == 0;
}

libreCudaStatus_t libreCuModuleLoadData(LibreCUmodule *pModule, const void *image, size_t imageSize) {
    LIBRECUDA_VALIDATE(pModule != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(image != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    // module state of the context (device runtime, kernel addresses) and the image upload
    std::lock_guard ctx_lock(current_ctx->mutex);

    // parse image as elf binary
    ELFIO::elfio elf_reader{}; {
        std::string s(reinterpret_cast<const char *>(image), imageSize);
        std::istringstream istream(s);
        LIBRECUDA_VALIDATE(elf_reader.load(istream), LIBRECUDA_ERROR_INVALID_IMAGE);
    }

    // find symbol table
    size_t symtab_idx = SIZE_MAX;
    for (size_t idx = 0; idx < elf_reader.sections.size(); idx++) {
        if (elf_reader.sections[idx]->get_type() == ELFIO::SHT_SYMTAB) {
            symtab_idx = idx;
            break;
        }
    }
    LIBRECUDA_VALIDATE(symtab_idx != SIZE_MAX, LIBRECUDA_ERROR_INVALID_IMAGE);

    // Lay out the sections that have to be resident in GPU memory: code, constant banks, initialized globals
    // and zero initialized globals (.nv.global is NOBITS). Shared and local memory sections only describe sizes.
    std::vector<NvU64> section_offsets(elf_reader.sections.size(), UINT64_MAX);
    size_t image_size = 0;
    for (size_t idx = 0; idx < elf_reader.sections.size(); idx++) {
        const auto &section = elf_reader.sections[idx];
        const std::string &name = section->get_name();
        bool loaded = (section->get_type() == ELFIO::SHT_PROGBITS && (section->get_flags() & ELFIO::SHF_ALLOC)) ||
                      (section->get_type() == ELFIO::SHT_NOBITS && startsWith(name, ".nv.global"));
        if (!loaded) {
            continue;
        }
        // constant banks are referenced by the QMD with 64 byte granularity, code is prefetched in 256 byte chunks
        const NvU64 align = std::max<NvU64>(section->get_addr_align(), MODULE_SECTION_ALIGN);
        image_size = ceilDiv(image_size, align) * align;
        section_offsets[idx] = image_size;
        image_size += section->get_size();
    }

    // Ensure some space after the program to mitigate instruction prefetch faults.
    const size_t allocated_size = ceilDiv(image_size, 0x1000) * 0x1000 + 0x1000;
    NvU64 module_gpu_va{};
    LIBRECUDA_ERR_PROPAGATE(
        gpuAlloc(current_ctx, allocated_size, false, false, true, 0, &module_gpu_va)
    );

    std::vector<uint8_t> image_data(image_size, 0);
    for (size_t idx = 0; idx < elf_reader.sections.size(); idx++) {
        const auto &section = elf_reader.sections[idx];
        if (section_offsets[idx] == UINT64_MAX || section->get_type() == ELFIO::SHT_NOBITS) {
            continue;
        }
        memcpy(image_data.data() + section_offsets[idx], section->get_data(), section->get_size());
    }

    ELFIO::symbol_section_accessor symbols{elf_reader, elf_reader.sections[symtab_idx]};

    // resolves a symbol to its address in GPU memory. Symbols in sections that are not loaded resolve to their value
    // (e.g. absolute symbols or offsets in shared memory)
    // the module calls the device runtime API / launches kernels from the device (set by resolve_symbol)
    bool uses_device_runtime_api = false, launches_kernels = false;
    auto resolve_symbol = [&](ELFIO::Elf_Xword symIdx, NvU64 *pAddrOut) -> bool {
        std::string name{};
        ELFIO::Elf64_Addr value{};
        ELFIO::Elf_Xword size{};
        unsigned char bind{}, type{}, other{};
        ELFIO::Elf_Half section_index{};
        if (!symbols.get_symbol(symIdx, name, value, size, bind, type, section_index, other)) {
            return false;
        }
        if (name.rfind("__cudaCDP2", 0) == 0 && ctxDeviceRuntimeFunction(current_ctx, name, pAddrOut, nullptr)) {
            // the device runtime API: LibreCUDA's implementation, even if the image contains NVIDIA's libcudadevrt
            uses_device_runtime_api = true;
            // device side copies run like launches (device_launch.cpp)
            launches_kernels = launches_kernels || name.find("LaunchDevice") != std::string::npos ||
                               name.find("Memcpy") != std::string::npos || name.find("Memset") != std::string::npos;
        } else if (section_index != ELFIO::SHN_UNDEF && section_index < section_offsets.size() &&
                   section_offsets[section_index] != UINT64_MAX) {
            *pAddrOut = module_gpu_va + section_offsets[section_index] + value;
        } else if (section_index == ELFIO::SHN_UNDEF && name.rfind("__cuda_syscall_", 0) == 0 &&
                   name != "__cuda_syscall_aligned_malloc" &&
                   ctxDeviceRuntimeFunction(current_ctx, "librecuda_unsupported", pAddrOut, nullptr)) {
            // a driver call of NVIDIA's device runtime code (only reachable through device runtime functions LibreCUDA
            // doesn't implement): stops the kernel
            LIBRECUDA_DEBUG("Warning: " + name + " (NVIDIA device runtime) is not supported");
        } else if (section_index == ELFIO::SHN_UNDEF &&
                   ctxDeviceRuntimeFunction(current_ctx, name, pAddrOut, nullptr)) {
            // device side printf / assert: provided by LibreCUDA's device runtime module
        } else {
            *pAddrOut = value;
        }
        return true;
    };

    auto *module = new LibreCUmodule_{
        .module_va_addr = module_gpu_va
    };

    std::unordered_map<std::string, LibreCUFunction_> functions{};
    std::vector<KernelConstantInfo> module_constants{};
    std::unordered_map<std::string, std::vector<KernelConstantInfo>> function_constants{};
    std::unordered_map<std::string, NvU32> function_shared_mem{};
    std::unordered_map<std::string, NvU32> function_regcount_attrs{};
    std::unordered_map<std::string, NvU32> function_stack_sizes{};
    // external functions (resolved by the driver, e.g. vprintf) called by each function
    std::unordered_map<std::string, std::vector<std::string>> function_externs{};

    for (size_t idx = 0; idx < elf_reader.sections.size(); idx++) {
        const auto &section = elf_reader.sections[idx];
        const std::string &section_name = section->get_name();

        if (section->get_type() == ELFIO::SHT_REL || section->get_type() == ELFIO::SHT_RELA) {
            // apply relocations targeting loaded sections. (relocations for e.g. debug sections are irrelevant)
            const NvU32 target_idx = section->get_info();
            if (target_idx >= section_offsets.size() || section_offsets[target_idx] == UINT64_MAX) {
                continue;
            }
            const bool has_addend = section->get_type() == ELFIO::SHT_RELA;
            ELFIO::relocation_section_accessor rel_accessor(elf_reader, section);
            for (ELFIO::Elf_Xword i = 0; i < rel_accessor.get_entries_num(); i++) {
                ELFIO::Elf64_Addr offset{};
                ELFIO::Elf_Word symbol{};
                uint32_t type{};
                ELFIO::Elf_Sxword addend{};
                LIBRECUDA_VALIDATE(rel_accessor.get_entry(i, offset, symbol, type, addend),
                                   LIBRECUDA_ERROR_INVALID_IMAGE);
                if (!has_addend) {
                    addend = 0;
                }
                NvU64 sym_addr{};
                LIBRECUDA_VALIDATE(resolve_symbol(symbol, &sym_addr), LIBRECUDA_ERROR_INVALID_IMAGE);
                const NvU64 value = sym_addr + addend;
                const NvU64 apply_off = section_offsets[target_idx] + offset;

                switch (type) {
                    case R_CUDA_32:
                    case R_CUDA_G32: {
                        LIBRECUDA_VALIDATE(apply_off + 4 <= image_size, LIBRECUDA_ERROR_INVALID_IMAGE);
                        NvU32 v = static_cast<NvU32>(value);
                        memcpy(image_data.data() + apply_off, &v, sizeof(v));
                        break;
                    }
                    case R_CUDA_64:
                    case R_CUDA_G64:
                    case R_CUDA_KERNEL_ADDR64: {
                        LIBRECUDA_VALIDATE(apply_off + 8 <= image_size, LIBRECUDA_ERROR_INVALID_IMAGE);
                        memcpy(image_data.data() + apply_off, &value, sizeof(value));
                        break;
                    }
                    case R_CUDA_ABS32_LO_32:
                    case R_CUDA_ABS32_HI_32:
                    case R_CUDA_KERNEL_ADDR32_LO_32:
                    case R_CUDA_KERNEL_ADDR32_HI_32: {
                        // 32 bit immediate in the upper half of a 64 bit instruction word
                        LIBRECUDA_VALIDATE(apply_off + 8 <= image_size, LIBRECUDA_ERROR_INVALID_IMAGE);
                        const bool low = type == R_CUDA_ABS32_LO_32 || type == R_CUDA_KERNEL_ADDR32_LO_32;
                        NvU32 v = static_cast<NvU32>(low ? value : value >> 32);
                        memcpy(image_data.data() + apply_off + 4, &v, sizeof(v));
                        break;
                    }
                    case R_CUDA_ABS55_16_34: {
                        // bits 2..56 of a 4 byte aligned address: bits 2..9 go to instruction bits 16..23, bits
                        // 10..56 to instruction bits 34..80 (found by disassembling patched instructions)
                        LIBRECUDA_VALIDATE(apply_off + 16 <= image_size && (value & 3) == 0 && (value >> 57) == 0,
                                           LIBRECUDA_ERROR_INVALID_IMAGE);
                        unsigned __int128 word{};
                        memcpy(&word, image_data.data() + apply_off, sizeof(word));
                        const unsigned __int128 low_mask = static_cast<unsigned __int128>(0xff) << 16;
                        const unsigned __int128 high_mask = ((static_cast<unsigned __int128>(1) << 47) - 1) << 34;
                        word &= ~(low_mask | high_mask);
                        word |= static_cast<unsigned __int128>((value >> 2) & 0xff) << 16;
                        word |= static_cast<unsigned __int128>(value >> 10) << 34;
                        memcpy(image_data.data() + apply_off, &word, sizeof(word));
                        break;
                    }
                    default: {
                        LIBRECUDA_DEBUG("Unsupported relocation type " + std::to_string(type) + " in section " +
                                        section_name + " for symbol " +
                                        elfSymbolName(elf_reader, symtab_idx, symbol));
                        delete module;
                        gpuFree(current_ctx, module_gpu_va);
                        LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_IMAGE);
                    }
                }
            }
        } else if (startsWith(section_name, ".nv.shared.")) {
            // per function static shared memory; includes the shared memory reserved by the toolchain (sm_90+)
            function_shared_mem[section_name.substr(11)] = section->get_size();
        } else if (startsWith(section_name, ".text.")) {
            std::string function_name = section_name.substr(6);
            auto &function = functions[function_name];
            function.name = function_name;
            function.func_va_addr = module_gpu_va + section_offsets[idx];
            // older toolchains encode the register count in the section info
            function.num_registers = section->get_info() >> 24;
            function.function_size = section->get_size();
        } else if (startsWith(section_name, ".nv.constant") && section_offsets[idx] != UINT64_MAX) {
            std::string constant_nr_str = section_name.substr(12);
            size_t end{};
            NvU32 constant_nr = std::stoi(constant_nr_str, &end);

            auto const_info = KernelConstantInfo{
                .const_nr = constant_nr,
                .address = module_gpu_va + section_offsets[idx],
                .size = section->get_size()
            };
            if (end != constant_nr_str.size()) {
                // .nv.constant<N>.<function name>
                function_constants[constant_nr_str.substr(end + 1)].push_back(const_info);
            } else {
                // module wide constant bank, applies to all functions
                module_constants.push_back(const_info);
            }
        } else if (section_name == ".nv.info") {
            // module wide attributes; per function ones reference the function by symbol index
            LIBRECUDA_ERR_PROPAGATE(forEachNvInfoAttribute(section, [&](NvU8 attr, NvU8 format, NvU16 size,
                                                                              const NvU32 *payload) {
                if (format != EIATTR_FORMAT_FREE || size < 2 * sizeof(NvU32)) {
                    return;
                }
                const NvU32 sym_idx = payload[0];
                const NvU32 value = payload[1];
                switch (attr) {
                    case EIATTR_REGCOUNT:
                        function_regcount_attrs[elfSymbolName(elf_reader, symtab_idx, sym_idx)] = value;
                        break;
                    case EIATTR_FRAME_SIZE:
                    case EIATTR_MIN_STACK_SIZE:
                    case EIATTR_MAX_STACK_SIZE: {
                        auto &stack_size = function_stack_sizes[elfSymbolName(elf_reader, symtab_idx, sym_idx)];
                        stack_size = std::max(stack_size, value);
                        break;
                    }
                    default:
                        break;
                }
            }));
        } else if (startsWith(section_name, ".nv.info.")) {
            auto &function = functions[section_name.substr(9)];
            LIBRECUDA_ERR_PROPAGATE(forEachNvInfoAttribute(section, [&](NvU8 attr, NvU8 format, NvU16 size,
                                                                              const NvU32 *payload) {
                switch (attr) {
                    case EIATTR_CBANK_PARAM_SIZE: {
                        function.param_total_size = size;
                        break;
                    }
                    case EIATTR_PARAM_CBANK: {
                        // payload: constant bank section symbol, (param area size << 16) | param area offset
                        if (payload != nullptr && size >= 2 * sizeof(NvU32)) {
                            function.param_base = payload[1] & 0xffff;
                        }
                        break;
                    }
                    case EIATTR_KPARAM_INFO: {
                        // payload: index, (ordinal | offset << 16), flags with the size in bits 18..31
                        if (payload != nullptr && size >= 3 * sizeof(NvU32)) {
                            function.param_info.push_back(KernelParamInfo{
                                .param_index = static_cast<NvU16>(payload[1] & 0xffff),
                                .param_offset = static_cast<NvU16>(payload[1] >> 16),
                                .param_size = static_cast<NvU16>(payload[2] >> 18)
                            });
                        }
                        break;
                    }
                    case EIATTR_MAX_THREADS:
                    case EIATTR_REQNTID: {
                        if (payload != nullptr && size >= 3 * sizeof(NvU32)) {
                            NvU32 n = payload[0] * std::max(1u, payload[1]) * std::max(1u, payload[2]);
                            if (n != 0) {
                                function.max_threads_per_block = std::min(function.max_threads_per_block, n);
                            }
                        }
                        break;
                    }
                    case EIATTR_CTA_PER_CLUSTER: {
                        // payload: cluster width, height, depth in blocks
                        if (payload != nullptr && size >= 3 * sizeof(NvU32)) {
                            for (int d = 0; d < 3; d++) {
                                function.cluster_dims[d] = std::max(1u, payload[d]);
                            }
                        }
                        break;
                    }
                    case EIATTR_EXPLICIT_CLUSTER: {
                        function.explicit_cluster = true;
                        break;
                    }
                    case EIATTR_EXTERNS: {
                        if (payload != nullptr) {
                            for (size_t k = 0; k < size / sizeof(NvU32); k++) {
                                const std::string extern_name = elfSymbolName(elf_reader, symtab_idx, payload[k]);
                                function_externs[section_name.substr(9)].push_back(extern_name);
                                if (!ctxDeviceRuntimeFunction(current_ctx, extern_name, nullptr, nullptr)) {
                                    LIBRECUDA_DEBUG("Warning: function " + section_name.substr(9) +
                                                    " references the external function " + extern_name +
                                                    ", which LibreCUDA doesn't provide");
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            }));
        }
    }

    // collect __device__ / __constant__ variables
    for (ELFIO::Elf_Xword i = 0; i < symbols.get_symbols_num(); i++) {
        std::string name{};
        ELFIO::Elf64_Addr value{};
        ELFIO::Elf_Xword size{};
        unsigned char bind{}, type{}, other{};
        ELFIO::Elf_Half section_index{};
        symbols.get_symbol(i, name, value, size, bind, type, section_index, other);
        if (type != ELFIO::STT_OBJECT || section_index == ELFIO::SHN_UNDEF || section_index >= section_offsets.size() ||
            section_offsets[section_index] == UINT64_MAX) {
            continue;
        }
        const std::string &section_name = elf_reader.sections[section_index]->get_name();
        if (!startsWith(section_name, ".nv.global") && !startsWith(section_name, ".nv.constant")) {
            continue;
        }
        module->globals[name] = ModuleGlobalInfo{
            .address = module_gpu_va + section_offsets[section_index] + value,
            .size = size
        };
    }

    // functions called by a kernel are emitted as additional function symbols in the kernel's text section
    for (ELFIO::Elf_Xword i = 0; i < symbols.get_symbols_num(); i++) {
        std::string name{};
        ELFIO::Elf64_Addr value{};
        ELFIO::Elf_Xword size{};
        unsigned char bind{}, type{}, other{};
        ELFIO::Elf_Half section_index{};
        symbols.get_symbol(i, name, value, size, bind, type, section_index, other);
        if (type != ELFIO::STT_FUNC || section_index >= elf_reader.sections.size()) {
            continue;
        }
        const std::string &section_name = elf_reader.sections[section_index]->get_name();
        if (startsWith(section_name, ".text.") && section_offsets[section_index] != UINT64_MAX) {
            module->function_symbols[name] = module_gpu_va + section_offsets[section_index] + value;
        }
        if (startsWith(section_name, ".text.") && section_name.substr(6) != name) {
            auto it = functions.find(section_name.substr(6));
            if (it != functions.end()) {
                it->second.has_calls = true;
            }
        }
    }

    module->function_registers = function_regcount_attrs;
    module->functions.reserve(functions.size());
    for (auto &[func_name, function]: functions) {
        if (function.func_va_addr == 0) {
            continue; // .nv.info.<name> without code
        }
        auto regcount_it = function_regcount_attrs.find(func_name);
        if (regcount_it != function_regcount_attrs.end()) {
            // newer toolchains no longer encode the register count in the .text section's sh_info
            function.num_registers = regcount_it->second;
        }
        if (uses_device_runtime_api) {
            // calls to the device runtime API were redirected to LibreCUDA's (see resolve_symbol). nvlink sized the
            // callers for NVIDIA's implementation, and which function calls it is only known for direct calls, so
            // every function of the module gets the device runtime's register count
            function.has_calls = true;
            function.num_registers = std::max(function.num_registers, ctxDeviceRuntimeRegisters(current_ctx));
            function.launches_kernels = launches_kernels;
        }
        for (const std::string &extern_name: function_externs[func_name]) {
            NvU32 callee_registers{};
            if (ctxDeviceRuntimeFunction(current_ctx, extern_name, nullptr, &callee_registers)) {
                // the call runs in the kernel's registers and on its stack (like nvlink does for separately compiled
                // device code, the kernel gets the callee's register count)
                function.has_calls = true;
                function.num_registers = std::max(function.num_registers, callee_registers);
            }
        }
        function.shared_mem = function_shared_mem[func_name];
        function.local_size_bytes = function_stack_sizes[func_name];
        function.local_mem_req = (function.local_size_bytes > 0 || function.has_calls)
                                     ? function.local_size_bytes + LOCAL_MEM_ABI_RESERVED
                                     : 0;

        // const0 must be the first in the constant list
        auto &consts = function.constants;
        for (const auto &c: function_constants[func_name]) {
            if (c.const_nr == 0) {
                consts.insert(consts.begin(), c);
            } else {
                consts.push_back(c);
            }
        }
        for (const auto &c: module_constants) {
            consts.push_back(c);
        }

        std::sort(function.param_info.begin(), function.param_info.end(),
                  [](const KernelParamInfo &a, const KernelParamInfo &b) { return a.param_index < b.param_index; });
        // toolchains that don't emit the size in KPARAM_INFO: derive it from the offset of the next parameter
        for (size_t i = 0; i < function.param_info.size(); i++) {
            auto &param = function.param_info[i];
            if (param.param_size == 0) {
                NvU32 next_offset = i + 1 < function.param_info.size()
                                        ? function.param_info[i + 1].param_offset
                                        : function.param_total_size;
                param.param_size = next_offset - param.param_offset;
            }
        }
        module->functions.push_back(std::move(function));
    }

    // device code launching kernels passes their code addresses
    for (auto &function: module->functions) {
        current_ctx->kernels_by_address[function.func_va_addr] = &function;
    }

    // copy to gpu
    memcpy(reinterpret_cast<void *>(module_gpu_va), image_data.data(), image_data.size());
    *pModule = module;
    LIBRECUDA_SUCCEED();
}



libreCudaStatus_t libreCuModuleGetFunctionCount(uint32_t *count, LibreCUmodule mod) {
    LIBRECUDA_VALIDATE(count != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *count = mod->functions.size();
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t
libreCuModuleEnumerateFunctions(LibreCUFunction *functions, uint32_t numFunctions, LibreCUmodule mod) {
    LIBRECUDA_VALIDATE(functions != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(mod != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(numFunctions >= mod->functions.size(), LIBRECUDA_ERROR_INVALID_VALUE);
    for (uint32_t i = 0; i < mod->functions.size(); i++) {
        functions[i] = &mod->functions[i];
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuModuleGetFunction(LibreCUFunction *pFunc, LibreCUmodule module, const char *name) {
    LIBRECUDA_VALIDATE(pFunc != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(module != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    bool found = false;

    for (auto &function: module->functions) {
        if (function.name == name) {
            found = true;
            *pFunc = &function;
            break;
        }
    }
    if (!found) {
        LIBRECUDA_FAIL(LIBRECUDA_ERROR_NOT_FOUND);
    }

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuLaunchKernel(LibreCUFunction function,
                                      uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                      uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                      uint32_t sharedMemBytes,
                                      LibreCUstream stream,
                                      void **kernelParams, size_t numParams,
                                      void **extra,
                                      bool async) {
    LIBRECUDA_VALIDATE(function != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(ctxLaunchWithDeviceLaunches(current_ctx, stream, function, [&] {
        return stream->command_queue->launchFunction(
                function,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes,
                kernelParams,
                numParams,
                async
            );
    }));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuLaunchKernelCluster(LibreCUFunction function,
                                             uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                             uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                             uint32_t clusterDimX, uint32_t clusterDimY, uint32_t clusterDimZ,
                                             uint32_t sharedMemBytes,
                                             LibreCUstream stream,
                                             void **kernelParams, size_t numParams) {
    LIBRECUDA_VALIDATE(function != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    const NvU32 cluster[3] = {clusterDimX, clusterDimY, clusterDimZ};
    LIBRECUDA_ERR_PROPAGATE(ctxLaunchWithDeviceLaunches(current_ctx, stream, function, [&] {
        return stream->command_queue->launchFunction(
                function,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes,
                kernelParams,
                numParams,
                false,
                false,
                cluster
            );
    }));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuLaunchCooperativeKernel(LibreCUFunction function,
                                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                                 uint32_t sharedMemBytes,
                                                 LibreCUstream stream,
                                                 void **kernelParams, size_t numParams) {
    LIBRECUDA_VALIDATE(function != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(ctxLaunchWithDeviceLaunches(current_ctx, stream, function, [&] {
        return stream->command_queue->launchFunction(
                function,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes,
                kernelParams,
                numParams,
                false,
                true
            );
    }));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuModuleUnload(LibreCUmodule module) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LIBRECUDA_VALIDATE(module != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    std::lock_guard ctx_lock(current_ctx->mutex);
    for (const auto &function: module->functions) {
        current_ctx->kernels_by_address.erase(function.func_va_addr);
    }
    LIBRECUDA_ERR_PROPAGATE(gpuFree(current_ctx, module->module_va_addr));
    delete module;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuFuncGetName(const char **pNameOut, LibreCUFunction func) {
    LIBRECUDA_VALIDATE(pNameOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pNameOut = func->name.c_str();
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuFuncGetLaunchesKernels(bool *pOut, LibreCUFunction func) {
    LIBRECUDA_VALIDATE(pOut != nullptr && func != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pOut = func->launches_kernels;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuFuncGetParamInfo(size_t *pOffsetOut, size_t *pSizeOut, LibreCUFunction func, size_t index) {
    LIBRECUDA_VALIDATE(pOffsetOut != nullptr && pSizeOut != nullptr && func != nullptr &&
                       index < func->param_info.size(), LIBRECUDA_ERROR_INVALID_VALUE);
    *pOffsetOut = func->param_info[index].param_offset;
    *pSizeOut = func->param_info[index].param_size;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuFuncGetParamCount(size_t *pCountOut, LibreCUFunction func) {
    LIBRECUDA_VALIDATE(pCountOut != nullptr && func != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pCountOut = func->param_info.size();
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuStreamCreate(LibreCUstream *pStreamOut, uint32_t flags) {
    LIBRECUDA_VALIDATE(pStreamOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    // create command queue
    auto command_queue = new NvCommandQueue(current_ctx);
    LIBRECUDA_ERR_PROPAGATE(command_queue->initializeQueue());

    *pStreamOut = new LibreCUstream_{
        .command_queue = command_queue
    };
    LIBRECUDA_SUCCEED();
}


libreCudaStatus_t libreCuStreamCommence(LibreCUstream stream) {
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(stream->command_queue->startExecution());
    LIBRECUDA_SUCCEED();
}


libreCudaStatus_t libreCuStreamAwait(LibreCUstream stream) {
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    const libreCudaStatus_t status = stream->command_queue->awaitExecution();
    // also when the work failed: a failed assert's message explains the error
    ctxFlushPrintf(current_ctx);
    return status;
}

libreCudaStatus_t libreCuStreamDestroy(LibreCUstream stream) {
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);

    // delete command queue
    delete stream->command_queue;

    delete stream;
    LIBRECUDA_SUCCEED();
}

/**
 * @return a context created on the device (attributes that require RM objects of the device need one)
 */
static LibreCUcontext contextOfDevice(LibreCUdevice device) {
    if (current_ctx != nullptr && current_ctx->device->ordinal == device->ordinal) {
        return current_ctx;
    }
    auto it = device_contexts.find(device->ordinal);
    return it != device_contexts.end() ? it->second : nullptr;
}

libreCudaStatus_t libreCuDeviceGetName(char *pDeviceName, int length, LibreCUdevice device) {
    LIBRECUDA_VALIDATE(pDeviceName != nullptr && length > 0, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(device != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LibreCUcontext ctx = contextOfDevice(device);
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT);

    NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS params{};
    params.gpuNameStringFlags = NV2080_CTRL_GPU_GET_NAME_STRING_FLAGS_TYPE_ASCII;
    RM_CTRL(fd_ctl, NV2080_CTRL_CMD_GPU_GET_NAME_STRING, root, ctx->sub_device_handle, &params, sizeof(params));
    const auto *name = reinterpret_cast<const char *>(params.gpuNameString.ascii);
    size_t n = strnlen(name, sizeof(params.gpuNameString.ascii));
    n = std::min(n, static_cast<size_t>(length - 1));
    memcpy(pDeviceName, name, n);
    pDeviceName[n] = '\0';

    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxGetCurrent(LibreCUcontext *pCtxOut) {
    LIBRECUDA_VALIDATE(pCtxOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pCtxOut = current_ctx;
    LIBRECUDA_SUCCEED();
}

static void computeCapability(LibreCUdevice device, int *pMajor, int *pMinor) {
    const NvU32 data = device->device_info[NV2080_CTRL_GR_INFO_INDEX_SM_VERSION].data;
    // GR reports the SM version as hex digits, e.g. 0x806 = 8.6
    int major = static_cast<int>((data >> 8) & 0xf);
    int minor = static_cast<int>(data & 0xff);
    minor = (minor >> 4) * 10 + (minor & 0xf);
    // consumer Blackwell (GB20x) reports GR SM version 10.4 but is CUDA compute capability 12.0
    if (data == 0xa04) {
        major = 12;
        minor = 0;
    }
    *pMajor = major;
    *pMinor = minor;
}

static int multiprocessorCount(LibreCUdevice device) {
    const auto &info = device->device_info;
    // SHADER_PIPE_SUB_COUNT is the number of (non floor-swept) TPCs
    const NvU32 tpcs = info[NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT].data;
    const NvU32 sm_per_tpc = std::max(1u, info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC].data);
    return static_cast<int>(tpcs * sm_per_tpc);
}

libreCudaStatus_t libreCuDeviceGetAttribute(int *pValOut, LibreCuDeviceAttribute attribute, LibreCUdevice device) {
    LIBRECUDA_VALIDATE(pValOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE)
    LIBRECUDA_VALIDATE(device != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    // device info is populated on context creation
    LIBRECUDA_VALIDATE(device->compute_class != 0, LIBRECUDA_ERROR_INVALID_CONTEXT);

    const auto &info = device->device_info;
    int major{}, minor{};
    computeCapability(device, &major, &minor);
    const bool is_blackwell = device->compute_class >= BLACKWELL_COMPUTE_A;
    const int max_warps_per_sm = static_cast<int>(info[NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM].data);
    const int smem_per_block_optin = is_blackwell ? 101376 : (major >= 8 ? 101376 : 49152);

    int value;
    switch (attribute) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: value = 1024; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: value = 1024; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: value = 1024; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: value = 64; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: value = 0x7fffffff; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: value = 65535; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: value = 65535; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: value = 49152; break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: value = 65536; break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE: value = static_cast<int>(info[NV2080_CTRL_GR_INFO_INDEX_MAX_THREADS_PER_WARP].data); break;
        case CU_DEVICE_ATTRIBUTE_MAX_PITCH: value = 0x7fffffff; break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: value = 65536; break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: value = 2000000; break; // kHz; not queried
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: value = 512; break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: value = multiprocessorCount(device); break;
        case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_INTEGRATED: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: value = device->pci_bus; break;
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: value = device->pci_slot; break;
        case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: value = static_cast<int>(device->pci_domain); break;
        case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: // kHz
            value = device->memory_clock_khz != 0 ? int(device->memory_clock_khz) : 14000000; // fallback: not reported
            break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: value = static_cast<int>(device->mem_bus_width); break;
        case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: value = static_cast<int>(device->l2_cache_size); break;
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: value = max_warps_per_sm * 32; break;
        case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: value = 32; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: value = major; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: value = minor; break;
        case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: value = smem_per_block_optin + 1024; break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: value = 65536; break;
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: value = 64; break;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: value = 1; break;
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: value = 1; break;
        case LIBRECUDA_DEVICE_ATTRIBUTE_GPC_COUNT:
            value = static_cast<int>(info[NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT].data);
            break;
        case CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH: {
            // LibreCUDA programs clusters only in the Blackwell QMD (V05)
            int major{}, minor{};
            computeCapability(device, &major, &minor);
            value = major >= 10 ? 1 : 0;
            break;
        }
        case CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: value = smem_per_block_optin; break;
        case CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: value = major >= 9 ? 32 : (major == 8 && minor == 9 ? 24 : 16); break;
        case CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE: value = 0; break;
        case CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: value = major >= 8 ? 1024 : 0; break;
        case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: value = 0; break;
        default: LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
    }
    *pValOut = value;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuDeviceTotalMem(size_t *pBytesOut, LibreCUdevice device) {
    LIBRECUDA_VALIDATE(pBytesOut != nullptr && device != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(device->compute_class != 0, LIBRECUDA_ERROR_INVALID_CONTEXT);
    *pBytesOut = device->total_mem;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuDeviceGetUuid(uint8_t uuidOut[16], LibreCUdevice device) {
    LIBRECUDA_VALIDATE(uuidOut != nullptr && device != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(device->compute_class != 0, LIBRECUDA_ERROR_INVALID_CONTEXT);
    memcpy(uuidOut, device->uuid.uuid, 16);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuFuncSetAttribute(LibreCUFunction function, LibreCuFunctionAttribute attribute, int value) {
    LIBRECUDA_VALIDATE(function != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    switch (attribute) {
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: {
            int max_smem{};
            LIBRECUDA_ERR_PROPAGATE(
                libreCuDeviceGetAttribute(
                    &max_smem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    current_ctx->device
                )
            );
            // the opt-in limit includes the function's static shared memory (minus the reserved part)
            const int static_smem = static_cast<int>(function->shared_mem) - 1024;
            if (value < 0 || value + std::max(0, static_smem) > max_smem) {
                LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
            }
            function->max_dynamic_shared_mem = value;
            break;
        }
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
            // the carveout is chosen per launch based on the shared memory usage
            break;
        default: LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuFuncGetAttribute(int *pValueOut, LibreCuFunctionAttribute attribute, LibreCUFunction function) {
    LIBRECUDA_VALIDATE(pValueOut != nullptr && function != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    int value;
    switch (attribute) {
        case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: {
            // limited by registers: 64K registers per block, allocated per warp in units of 256
            NvU32 regs_per_warp = ((std::max(1u, function->num_registers) * 32 + 255) / 256) * 256;
            NvU32 max_warps = 65536 / regs_per_warp;
            value = static_cast<int>(std::min({1024u, max_warps * 32, function->max_threads_per_block}));
            break;
        }
        case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
            // the reserved shared memory is not reported as part of the static size
            value = static_cast<int>(function->shared_mem >= 1024 ? function->shared_mem - 1024 : function->shared_mem);
            break;
        case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: {
            NvU64 total = 0;
            for (const auto &c: function->constants) {
                if (c.const_nr == 3) {
                    total += c.size;
                }
            }
            value = static_cast<int>(total);
            break;
        }
        case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: value = static_cast<int>(function->local_size_bytes); break;
        case CU_FUNC_ATTRIBUTE_NUM_REGS: value = static_cast<int>(function->num_registers); break;
        case CU_FUNC_ATTRIBUTE_PTX_VERSION:
        case CU_FUNC_ATTRIBUTE_BINARY_VERSION: {
            int major{}, minor{};
            LIBRECUDA_ENSURE_CTX_VALID();
            computeCapability(current_ctx->device, &major, &minor);
            value = major * 10 + minor;
            break;
        }
        case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: value = 0; break;
        case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: value = static_cast<int>(function->max_dynamic_shared_mem); break;
        case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: value = -1; break;
        case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: value = function->explicit_cluster ? 1 : 0; break;
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH:
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
        case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: {
            const NvU32 *dims = function->cluster_dims;
            const bool compiled = dims[0] * dims[1] * dims[2] > 1;
            value = compiled ? static_cast<int>(dims[attribute - CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH]) : 0;
            break;
        }
        // clusters beyond LIBRECUDA_MAX_CLUSTER_SIZE blocks are not supported
        case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: value = 0; break;
        case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: value = 0; break;
        default: LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
    }
    *pValueOut = value;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t
libreCuModuleGetGlobal(void **pDevicePtrOut, size_t *pBytesOut, LibreCUmodule module, const char *name) {
    LIBRECUDA_VALIDATE(module != nullptr && name != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    auto it = module->globals.find(name);
    LIBRECUDA_VALIDATE(it != module->globals.end(), LIBRECUDA_ERROR_NOT_FOUND);
    if (pDevicePtrOut != nullptr) {
        *pDevicePtrOut = reinterpret_cast<void *>(it->second.address);
    }
    if (pBytesOut != nullptr) {
        *pBytesOut = it->second.size;
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuMemsetAsync(void *dst, uint32_t value, size_t elementSize, size_t numElements,
                                     LibreCUstream stream) {
    LIBRECUDA_VALIDATE(stream != nullptr && dst != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(elementSize == 1 || elementSize == 2 || elementSize == 4, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(isGpuAccessiblePtr(dst), LIBRECUDA_ERROR_INVALID_VALUE);
    if (numElements == 0) {
        LIBRECUDA_SUCCEED();
    }
    auto *bytes = static_cast<uint8_t *>(dst);
    if (elementSize == 1 && isDevicePtr(dst) && memsetKernelsAvailable()) {
        // within device memory, kernels reach much higher bandwidth than the copy engine
        LIBRECUDA_ERR_PROPAGATE(memsetD8Kernel(dst, static_cast<uint8_t>(value), numElements, stream, false));
    } else if (elementSize == 1) {
        // byte memsets are split into a head, a 4-byte element body and a tail; much faster on the copy engine
        const NvU32 value4 = (value & 0xff) * 0x01010101u;
        size_t head = (4 - (reinterpret_cast<uintptr_t>(bytes) & 3)) & 3;
        head = std::min(head, numElements);
        const size_t body = (numElements - head) / 4;
        const size_t tail = numElements - head - body * 4;
        if (head > 0) {
            LIBRECUDA_ERR_PROPAGATE(stream->command_queue->gpuMemset(bytes, value & 0xff, 1, head));
        }
        if (body > 0) {
            LIBRECUDA_ERR_PROPAGATE(stream->command_queue->gpuMemset(bytes + head, value4, 4, body));
        }
        if (tail > 0) {
            LIBRECUDA_ERR_PROPAGATE(stream->command_queue->gpuMemset(bytes + head + body * 4, value & 0xff, 1, tail));
        }
    } else {
        LIBRECUDA_ERR_PROPAGATE(stream->command_queue->gpuMemset(dst, value, elementSize, numElements));
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuStreamWaitEvent(LibreCUstream stream, LibreCUEvent event) {
    LIBRECUDA_VALIDATE(stream != nullptr && event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(stream->command_queue->waitEventGpu(event));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuStreamQuery(LibreCUstream stream) {
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    return stream->command_queue->queryExecution();
}

libreCudaStatus_t libreCuEventQuery(LibreCUEvent event) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    return NvCommandQueue::queryEvent(event);
}

libreCudaStatus_t libreCuCtxSetStackLimit(size_t bytes) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LIBRECUDA_VALIDATE(bytes <= (1 << 20), LIBRECUDA_ERROR_INVALID_VALUE);
    current_ctx->stack_limit = static_cast<NvU32>((bytes + 15) & ~size_t(15));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxGetStackLimit(size_t *pBytesOut) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LIBRECUDA_VALIDATE(pBytesOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    *pBytesOut = current_ctx->stack_limit;
    LIBRECUDA_SUCCEED();
}

LibreCUEvent_ *NewEvent();

LibreCUstream_ *EventGetStream(LibreCUEvent pEvent);

void DeleteEvent(LibreCUEvent_ *pEvent);

libreCudaStatus_t libreCuEventCreate(LibreCUEvent *pEventOut, uint32_t flags) {
    LIBRECUDA_VALIDATE(pEventOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    *pEventOut = NewEvent();
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuEventRecord(LibreCUEvent event, LibreCUstream stream) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(stream != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ERR_PROPAGATE(stream->command_queue->recordEvent(event, stream));
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuEventSynchronize(LibreCUEvent event) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    const libreCudaStatus_t status = NvCommandQueue::waitForEvent(event);
    // also when the work failed: a failed assert's message explains the error
    ctxFlushPrintf(current_ctx);
    return status;
}

libreCudaStatus_t libreCuEventElapsedTime(float *pMillisecondsOut, LibreCUEvent start, LibreCUEvent end) {
    LIBRECUDA_VALIDATE(pMillisecondsOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(start != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(end != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    uint64_t startTimestamp{}, endTimestamp{};
    LIBRECUDA_ERR_PROPAGATE(NvCommandQueue::getEventTimestamp(start, &startTimestamp));
    LIBRECUDA_ERR_PROPAGATE(NvCommandQueue::getEventTimestamp(end, &endTimestamp));
    *pMillisecondsOut = static_cast<float>(static_cast<double>(static_cast<int64_t>(endTimestamp - startTimestamp)) / 1e6); // ns to ms
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuEventDestroy(LibreCUEvent event) {
    LIBRECUDA_VALIDATE(event != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    DeleteEvent(event);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t ctxObtainEventSignal(LibreCUcontext ctx, NvSignal **pSignalOut) {
    std::lock_guard ctx_lock(ctx->mutex);
    LIBRECUDA_VALIDATE(ctx != nullptr && pSignalOut != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(!ctx->free_event_signals.empty(), LIBRECUDA_ERROR_OUT_OF_MEMORY);
    NvSignal *signal = ctx->event_signal_pool + ctx->free_event_signals.back();
    ctx->free_event_signals.pop_back();
    signal->value = 0;
    signal->time_stamp = 0;
    *pSignalOut = signal;
    LIBRECUDA_SUCCEED();
}

void ctxReleaseEventSignal(LibreCUcontext ctx, NvSignal *signal) {
    std::lock_guard ctx_lock(ctx->mutex);
    ctx->free_event_signals.push_back(static_cast<NvU32>(signal - ctx->event_signal_pool));
}


libreCudaStatus_t ctxEnsureComputeChannel(LibreCUcontext ctx, NvU32 index) {
    std::lock_guard ctx_lock(ctx->mutex);
    LIBRECUDA_VALIDATE(index < ctx->compute_gpfifos.size(), LIBRECUDA_ERROR_INVALID_VALUE);
    GPFifo &fifo = ctx->compute_gpfifos[index];
    if (fifo.channel_handle != 0) {
        LIBRECUDA_SUCCEED();
    }
    // channels in different subcontexts of a channel group run concurrently on GR
    NvHandle ctx_share{}; {
        NV_CTXSHARE_ALLOCATION_PARAMETERS params{
            .hVASpace = ctx->va_space_handle,
            .flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC
        };
        RM_ALLOC(fd_ctl, FERMI_CONTEXT_SHARE_A, root, ctx->channel_group, 0, &params, sizeof(params), &ctx_share);
    }
    LIBRECUDA_ERR_PROPAGATE(
        makeGpuFifo(
            ctx,
            ctx->gpfifo_area_va, ctx->gpfifo_area_handle,
            ctx_share, ctx->channel_group,
            index * GPU_FIFO_SLOT_SIZE, 0x10000, 0,
            &fifo
        )
    );
    LIBRECUDA_SUCCEED();
}

NvHandle rmClient() {
    return root;
}

int rmControlFd() {
    return fd_ctl;
}

void rmFreeObject(NvHandle parent, NvHandle object) {
    rmFree(parent, object);
}

libreCudaStatus_t libreCuCtxCheckErrors() {
    LIBRECUDA_ENSURE_CTX_VALID();
    std::vector<NvCommandQueue *> queues{};
    {
        std::lock_guard ctx_lock(current_ctx->mutex);
        queues.assign(current_ctx->queues.begin(), current_ctx->queues.end());
    }
    for (NvCommandQueue *queue: queues) {
        LIBRECUDA_ERR_PROPAGATE(queue->awaitExecution());
    }
    ctxWaitForGrExceptionReports(current_ctx);
    return ctxCheckChannelError(current_ctx);
}

libreCudaStatus_t rmControl(NvHandle object, NvV32 cmd, void *params, NvU32 paramSize) {
    return rm_ctrl(fd_ctl, cmd, root, object, params, paramSize);
}

std::vector<GPFifo *> ctxChannels(LibreCUcontext_ *ctx) {
    std::vector<GPFifo *> channels{};
    for (GPFifo &fifo: ctx->compute_gpfifos) {
        channels.push_back(&fifo);
    }
    channels.push_back(&ctx->dma_gpfifo);
    channels.push_back(&ctx->dma_d2h_gpfifo);
    return channels;
}

/**
 * @return the channel's error status from its error notifier (NvNotification: timeStamp[2], info32, info16 | status << 16)
 */
static libreCudaStatus_t channelErrorOf(const GPFifo &fifo, NvU32 *pInfo32) {
    if (fifo.error_notifier == nullptr) {
        return LIBRECUDA_SUCCESS;
    }
    const NvU32 info32 = fifo.error_notifier[2];
    const NvU32 status = fifo.error_notifier[3] >> 16;
    if (info32 == 0 && (status & 0x7fff) == 0) {
        return LIBRECUDA_SUCCESS;
    }
    *pInfo32 = info32;
    // 31 = ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT (MMU fault, e.g. an out of bounds access)
    return info32 == 31 ? LIBRECUDA_ERROR_ILLEGAL_ADDRESS : LIBRECUDA_ERROR_LAUNCH_FAILED;
}

libreCudaStatus_t ctxCheckChannelError(LibreCUcontext ctx) {
    if (ctx->channel_error != LIBRECUDA_SUCCESS) {
        return ctx->channel_error;
    }
    // called on every submission: no allocation (ctxChannels) and no lock unless a channel reported an error
    auto check = [&](const GPFifo &fifo, const char *kind) -> libreCudaStatus_t {
        NvU32 info32{};
        if (channelErrorOf(fifo, &info32) == LIBRECUDA_SUCCESS) {
            return LIBRECUDA_SUCCESS;
        }
        // one thread finds out which error it was
        std::lock_guard lock(ctx->error_mutex);
        if (ctx->channel_error != LIBRECUDA_SUCCESS) {
            return ctx->channel_error;
        }
        libreCudaStatus_t error = channelErrorOf(fifo, &info32);
        LIBRECUDA_DEBUG(std::string(kind) + " channel reported error " + std::to_string(info32));
        // 13 = ROBUST_CHANNEL_GR_EXCEPTION (an SM exception, e.g. a misaligned address): find out which. A failed
        // assert stops the kernel with a trap, which the SM reports as an illegal instruction; the assert record in
        // the printf buffer tells it apart.
        if (info32 == 13) {
            ctxFlushPrintf(ctx);
            error = ctx->device_assert_failed ? LIBRECUDA_ERROR_ASSERT : ctxSmExceptionStatus(ctx);
        }
        // published once classified: other threads read it without the lock
        ctx->channel_error_code = info32;
        ctx->channel_error = error;
        return error;
    };
    for (const GPFifo &fifo: ctx->compute_gpfifos) {
        LIBRECUDA_ERR_PROPAGATE(check(fifo, "compute"));
    }
    LIBRECUDA_ERR_PROPAGATE(check(ctx->dma_gpfifo, "copy"));
    LIBRECUDA_ERR_PROPAGATE(check(ctx->dma_d2h_gpfifo, "copy"));
    return LIBRECUDA_SUCCESS;
}

libreCudaStatus_t ctxWaitForValue(LibreCUcontext ctx, volatile NvU64 *value, NvU64 target) {
    if (*value >= target) {
        LIBRECUDA_SUCCEED();
    }
    static const NvU64 timeout_ms = [] {
        const char *env = getenv("LIBRECUDA_WAIT_TIMEOUT_MS");
        return env != nullptr ? strtoull(env, nullptr, 10) : 0ull;
    }();
    const auto start = std::chrono::steady_clock::now();
    for (NvU64 iteration = 0; *value < target; iteration++) {
        if (iteration < 4096) {
            __builtin_ia32_pause();
            continue;
        }
        if ((iteration & 0xff) == 0) {
            LIBRECUDA_ERR_PROPAGATE(ctxCheckChannelError(ctx));
            if (timeout_ms != 0 && std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms)) {
                LIBRECUDA_DEBUG("timed out waiting for the GPU: semaphore at " + std::to_string(*value) + ", waiting for " +
                                std::to_string(target));
                for (const GPFifo *fifo: ctxChannels(ctx)) {
                    if (fifo->controls != nullptr) {
                        const std::string name = fifo == &ctx->dma_gpfifo ? "copy" :
                                                 fifo == &ctx->dma_d2h_gpfifo ? "copy d2h" :
                                                 "compute " + std::to_string(fifo - ctx->compute_gpfifos.data());
                        LIBRECUDA_DEBUG("  " + name + " channel: put " +
                                        std::to_string(fifo->put_value % fifo->entries_count) + ", GPGet " +
                                        std::to_string(fifo->controls->GPGet) + ", GPPut " +
                                        std::to_string(fifo->controls->GPPut));
                    }
                }
                LIBRECUDA_FAIL(LIBRECUDA_ERROR_LAUNCH_TIMEOUT);
            }
        }
        if (!(ctx->flags & CU_CTX_SCHED_SPIN)) {
            std::this_thread::yield();
        }
    }
    LIBRECUDA_SUCCEED();
}
