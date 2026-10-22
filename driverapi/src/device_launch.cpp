// Device side kernel launches (dynamic parallelism): host part.
//
// Device code launches kernels through the device runtime API (__cudaCDP2GetParameterBufferV2 /
// __cudaCDP2LaunchDeviceV2, cuda_device_runtime_api.h). The loader redirects these calls to LibreCUDA's device runtime
// (driverapi/kernels/device_runtime/device_runtime.cu), which records each launch in a slot of the context's launch
// queue (pinned host memory). LibreCUDA has no way to launch kernels from the GPU, so the host launches them: a launch
// of a kernel whose module launches kernels waits for the kernel, then launches the kernels it requested on the same
// stream, depth first (each child and all its descendants complete before the next child starts; tail launches after
// the other children), and only then returns. This keeps CDP2's guarantees (a grid completes after its children; work
// launched into a stream runs in order) except that children only start once their parent completed. Device side
// cudaMemcpyAsync / cudaMemsetAsync (and their 2D / 3D forms) are queued the same way and run on the stream in order
// with the launches.

#include <algorithm>
#include <cstring>

#include "cmdqueue.h"
#include "device_runtime_state.h"
#include "librecuda_internal.h"

// slot layout (device_runtime.cu)
static constexpr NvU64 LAUNCH_HEADER_SIZE = 64;
static constexpr NvU64 LAUNCH_PARAM_SIZE = 4096;
static constexpr NvU64 LAUNCH_SLOT_SIZE = LAUNCH_HEADER_SIZE + LAUNCH_PARAM_SIZE;
static constexpr NvU32 LAUNCH_RESERVED = 0;
static constexpr NvU32 LAUNCH_TAIL = 2;

static constexpr NvU32 OPERATION_KERNEL = 0;
static constexpr NvU32 OPERATION_COPY = 1;
static constexpr NvU32 OPERATION_SET = 2;
static constexpr NvU32 OPERATION_GRAPH = 3;
static constexpr NvU32 OPERATION_COPY_3D = 4;
// sizeof(cudaMemcpy3DParms)
static constexpr size_t MEMCPY_3D_PARAMS_SIZE = 160;

// runs device side copies involving CUDA arrays (the runtime's objects; libreCuSetDeviceArrayCopyHandler)
static LibreCuDeviceArrayCopyHandler array_copy_handler = nullptr;

libreCudaStatus_t libreCuSetDeviceArrayCopyHandler(LibreCuDeviceArrayCopyHandler handler) {
    array_copy_handler = handler;
    LIBRECUDA_SUCCEED();
}

struct LaunchSlot {
    NvU64 function;
    NvU32 grid[3];
    NvU32 block[3];
    NvU32 shared_mem;
    NvU32 status;
    NvU32 operation;
};

struct CopyOp {
    NvU64 dst;
    NvU64 src;
    NvU64 width;
    NvU64 height;
    NvU64 depth;
    NvU64 dst_pitch;
    NvU64 src_pitch;
    NvU64 dst_slice_pitch;
    NvU64 src_slice_pitch;
    NvU32 value;
};

// like CUDA's maximum nesting depth of device side launches
static constexpr NvU32 MAX_LAUNCH_DEPTH = 24;

libreCudaStatus_t ctxEnsureDeviceLaunchQueue(LibreCUcontext ctx) {
    if (ctx->launch_queue != nullptr) {
        LIBRECUDA_SUCCEED();
    }
    std::lock_guard ctx_lock(ctx->mutex);
    if (ctx->launch_queue != nullptr) {
        LIBRECUDA_SUCCEED();
    }
    LIBRECUDA_VALIDATE(ctx->printf_state_va != 0, LIBRECUDA_ERROR_NOT_INITIALIZED);
    NvU64 queue_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuHostAlloc(ctx, NvU64(ctx->launch_queue_capacity) * LAUNCH_SLOT_SIZE, &queue_va));
    auto *state = reinterpret_cast<volatile DeviceRuntimeState *>(ctx->printf_state_va);

    // per thread errors of the device runtime API, one entry per hardware thread slot (SM ids may be sparse: room for
    // every SM the GPCs could have)
    const auto &info = ctx->device->device_info;
    const NvU32 sms = std::max(1u, info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS].data) *
                      std::max(1u, info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC].data) *
                      std::max(1u, info[NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC].data);
    const NvU32 warps = std::max(1u, info[NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM].data);
    const NvU64 error_slots_size = NvU64(sms) * warps * 32 * sizeof(DeviceErrorSlot);
    NvU64 error_slots_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, error_slots_size, false, false, true, 0, &error_slots_va));
    // grid 0 matches no grid
    memset(reinterpret_cast<void *>(error_slots_va), 0, error_slots_size);
    state->error_slot_sms = sms;
    state->error_slot_warps = warps;
    state->error_slots = error_slots_va;

    state->launch_count = 0;
    state->launch_capacity = ctx->launch_queue_capacity;
    state->launch_queue = queue_va;
    ctx->launch_queue = reinterpret_cast<NvU8 *>(queue_va);
    LIBRECUDA_DEBUG("device launch queue: " + std::to_string(ctx->launch_queue_capacity) + " slots at " +
                    std::to_string(queue_va));
    LIBRECUDA_SUCCEED();
}

namespace {
    struct PendingLaunch {
        NvU32 operation;
        // kernel launches
        LibreCUFunction function;
        NvU32 grid[3];
        NvU32 block[3];
        NvU32 shared_mem;
        std::vector<NvU8> params;
        // copies and fills
        CopyOp copy;
    };
}

/**
 * Runs a device side memcpy / memset on the stream: one operation per row, or one for all if the rows are contiguous
 */
static libreCudaStatus_t runCopy(const PendingLaunch &launch, LibreCUstream stream) {
    const CopyOp &op = launch.copy;
    const bool fill = launch.operation == OPERATION_SET;
    const bool rows_contiguous = op.dst_pitch == op.width && (fill || op.src_pitch == op.width);
    const bool slices_contiguous = op.depth == 1 || (op.dst_slice_pitch == op.width * op.height &&
                                                     (fill || op.src_slice_pitch == op.width * op.height));
    auto run = [&](NvU64 dst, NvU64 src, NvU64 bytes) {
        return fill ? libreCuMemsetAsync(reinterpret_cast<void *>(dst), op.value, 1, bytes, stream)
                    : libreCuMemCpy(reinterpret_cast<void *>(dst), reinterpret_cast<void *>(src), bytes, stream, true);
    };
    if (rows_contiguous && slices_contiguous) {
        return run(op.dst, op.src, op.width * op.height * op.depth);
    }
    for (NvU64 z = 0; z < op.depth; z++) {
        for (NvU64 y = 0; y < op.height; y++) {
            LIBRECUDA_ERR_PROPAGATE(run(op.dst + z * op.dst_slice_pitch + y * op.dst_pitch,
                                        op.src + z * op.src_slice_pitch + y * op.src_pitch, op.width));
        }
    }
    LIBRECUDA_SUCCEED();
}

/**
 * Launches the kernels requested by the work completed on queue (and their descendants)
 */
static libreCudaStatus_t launchRequested(LibreCUcontext ctx, LibreCUstream stream, NvU32 depth) {
    NvCommandQueue *queue = stream->command_queue;
    auto *state = reinterpret_cast<volatile DeviceRuntimeState *>(ctx->printf_state_va);
    const NvU32 requested = state->launch_count;
    if (requested == 0) {
        LIBRECUDA_SUCCEED();
    }
    // launches beyond the capacity failed on the device (cudaErrorLaunchPendingCountExceeded)
    const NvU32 count = std::min(requested, ctx->launch_queue_capacity);

    // take the requests out of the queue: the children's requests reuse it
    std::vector<PendingLaunch> launches{}, tail_launches{};
    for (NvU32 i = 0; i < count; i++) {
        const NvU8 *slot = ctx->launch_queue + NvU64(i) * LAUNCH_SLOT_SIZE;
        LaunchSlot header{};
        memcpy(&header, slot, sizeof(header));
        if (header.status == LAUNCH_RESERVED) {
            continue; // a parameter buffer that was never launched
        }
        if (header.operation == OPERATION_GRAPH) {
            // run by the runtime once the launching kernel's device launches are done
            NvU64 graph[2];
            memcpy(graph, slot + LAUNCH_HEADER_SIZE, sizeof(graph));
            std::lock_guard ctx_lock(ctx->mutex);
            ctx->device_graph_launches.emplace_back(graph[0], graph[1]);
            continue;
        }
        if (header.operation == OPERATION_COPY_3D) {
            PendingLaunch copy{.operation = OPERATION_COPY_3D};
            copy.params.assign(slot + LAUNCH_HEADER_SIZE, slot + LAUNCH_HEADER_SIZE + MEMCPY_3D_PARAMS_SIZE);
            (header.status == LAUNCH_TAIL ? tail_launches : launches).push_back(std::move(copy));
            continue;
        }
        if (header.operation == OPERATION_COPY || header.operation == OPERATION_SET) {
            PendingLaunch copy{.operation = header.operation};
            memcpy(&copy.copy, slot + LAUNCH_HEADER_SIZE, sizeof(copy.copy));
            (header.status == LAUNCH_TAIL ? tail_launches : launches).push_back(std::move(copy));
            continue;
        }
        std::unique_lock ctx_lock(ctx->mutex);
        auto function_it = ctx->kernels_by_address.find(header.function);
        if (function_it == ctx->kernels_by_address.end()) {
            LIBRECUDA_DEBUG("device side launch of an unknown kernel at " + std::to_string(header.function));
            state->launch_count = 0;
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
        }
        PendingLaunch launch{.operation = OPERATION_KERNEL, .function = function_it->second,
                             .shared_mem = header.shared_mem};
        ctx_lock.unlock();
        memcpy(launch.grid, header.grid, sizeof(launch.grid));
        memcpy(launch.block, header.block, sizeof(launch.block));
        const NvU32 param_size = std::min<NvU32>(launch.function->param_total_size, LAUNCH_PARAM_SIZE);
        launch.params.assign(slot + LAUNCH_HEADER_SIZE, slot + LAUNCH_HEADER_SIZE + param_size);
        (header.status == LAUNCH_TAIL ? tail_launches : launches).push_back(std::move(launch));
    }
    state->launch_count = 0;
    launches.insert(launches.end(), std::make_move_iterator(tail_launches.begin()),
                    std::make_move_iterator(tail_launches.end()));
    if (launches.empty()) {
        LIBRECUDA_SUCCEED();
    }
    LIBRECUDA_VALIDATE(depth < MAX_LAUNCH_DEPTH, LIBRECUDA_ERROR_LAUNCH_MAX_DEPTH_EXCEEDED);

    for (PendingLaunch &launch: launches) {
        if (launch.operation == OPERATION_COPY_3D) {
            LIBRECUDA_VALIDATE(array_copy_handler != nullptr, LIBRECUDA_ERROR_NOT_SUPPORTED);
            LIBRECUDA_ERR_PROPAGATE(array_copy_handler(launch.params.data(), stream));
            continue;
        }
        if (launch.operation != OPERATION_KERNEL) {
            LIBRECUDA_ERR_PROPAGATE(runCopy(launch, stream));
            continue;
        }
        std::vector<void *> params{};
        for (const KernelParamInfo &param: launch.function->param_info) {
            LIBRECUDA_VALIDATE(param.param_offset + param.param_size <= launch.params.size(),
                               LIBRECUDA_ERROR_INVALID_VALUE);
            params.push_back(launch.params.data() + param.param_offset);
        }
        LIBRECUDA_ERR_PROPAGATE(queue->launchFunction(launch.function,
                                                      launch.grid[0], launch.grid[1], launch.grid[2],
                                                      launch.block[0], launch.block[1], launch.block[2],
                                                      launch.shared_mem, params.data(), params.size(), false));
        if (launch.function->launches_kernels) {
            LIBRECUDA_ERR_PROPAGATE(queue->awaitExecution());
            LIBRECUDA_ERR_PROPAGATE(launchRequested(ctx, stream, depth + 1));
        }
    }
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t ctxLaunchWithDeviceLaunches(LibreCUcontext ctx, LibreCUstream stream, LibreCUFunction function,
                                              const std::function<libreCudaStatus_t()> &launch) {
    if (!function->launches_kernels) {
        return launch();
    }
    std::lock_guard lock(ctx->device_launch_mutex);
    LIBRECUDA_ERR_PROPAGATE(launch());
    LIBRECUDA_ERR_PROPAGATE(stream->command_queue->awaitExecution());
    return launchRequested(ctx, stream, 1);
}

libreCudaStatus_t libreCuCtxTakeDeviceGraphLaunches(uint64_t *pGraphExecs, uint64_t *pStreams, size_t capacity,
                                                   size_t *pCount) {
    LIBRECUDA_VALIDATE(pCount != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    std::lock_guard ctx_lock(current_ctx->mutex);
    auto &launches = current_ctx->device_graph_launches;
    const size_t n = std::min(capacity, launches.size());
    for (size_t i = 0; i < n; i++) {
        pGraphExecs[i] = launches[i].first;
        pStreams[i] = launches[i].second;
    }
    launches.erase(launches.begin(), launches.begin() + n);
    *pCount = n;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxSetDeviceLaunchQueueSize(uint32_t launches) {
    LIBRECUDA_ENSURE_CTX_VALID();
    std::lock_guard ctx_lock(current_ctx->mutex);
    // like CUDA, the size can't change once the queue exists
    LIBRECUDA_VALIDATE(current_ctx->launch_queue == nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(launches >= 1 && launches <= (1u << 20), LIBRECUDA_ERROR_INVALID_VALUE);
    current_ctx->launch_queue_capacity = launches;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxGetDeviceLaunchQueueSize(uint32_t *pLaunches) {
    LIBRECUDA_VALIDATE(pLaunches != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    *pLaunches = current_ctx->launch_queue_capacity;
    LIBRECUDA_SUCCEED();
}
