// Device side kernel launches (dynamic parallelism): host part.
//
// Device code launches kernels through the device runtime API (__cudaCDP2GetParameterBufferV2 /
// __cudaCDP2LaunchDeviceV2, cuda_device_runtime_api.h). The loader redirects these calls to LibreCUDA's device runtime
// (driverapi/kernels/device_runtime/device_runtime.cu), which records each launch in a slot of the context's launch
// queue (pinned host memory). LibreCUDA has no way to launch kernels from the GPU, so the host launches them: a launch
// of a kernel whose module launches kernels waits for the kernel, then launches the kernels it requested on the same
// stream, depth first (each child and all its descendants complete before the next child starts; tail launches after
// the other children), and only then returns. This keeps CDP2's guarantees (a grid completes after its children; work
// launched into a stream runs in order) except that children only start once their parent completed.

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

struct LaunchSlot {
    NvU64 function;
    NvU32 grid[3];
    NvU32 block[3];
    NvU32 shared_mem;
    NvU32 status;
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
        LibreCUFunction function;
        NvU32 grid[3];
        NvU32 block[3];
        NvU32 shared_mem;
        std::vector<NvU8> params;
    };
}

/**
 * Launches the kernels requested by the work completed on queue (and their descendants)
 */
static libreCudaStatus_t launchRequested(LibreCUcontext ctx, NvCommandQueue *queue, NvU32 depth) {
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
        std::unique_lock ctx_lock(ctx->mutex);
        auto function_it = ctx->kernels_by_address.find(header.function);
        if (function_it == ctx->kernels_by_address.end()) {
            LIBRECUDA_DEBUG("device side launch of an unknown kernel at " + std::to_string(header.function));
            state->launch_count = 0;
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_INVALID_VALUE);
        }
        PendingLaunch launch{.function = function_it->second, .shared_mem = header.shared_mem};
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
            LIBRECUDA_ERR_PROPAGATE(launchRequested(ctx, queue, depth + 1));
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
    return launchRequested(ctx, stream->command_queue, 1);
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
