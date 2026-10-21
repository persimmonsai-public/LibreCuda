// SM exceptions (warp errors such as a misaligned or out of range address).
//
// When a warp hits such an error, RM logs Xid 13 and kills the warp; the kernel completes. RM marks the channel's error
// notifier with ROBUST_CHANNEL_GR_EXCEPTION (13) only 10-40 ms later, so waits must check the notifier even when the
// awaited work has completed (ctxWaitForValue), and a check at reset/exit waits until that delay has passed.
//
// The notifier only says "GR exception". To report the specific error like CUDA does, the SM error state RM recorded
// for the context is read once through a GT200_DEBUGGER object (NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES). Every
// control on that object takes hundreds of microseconds (tens of milliseconds for the first), which only matters on
// this error path.

#include <chrono>
#include <cstring>
#include <thread>

#include "cmdqueue.h"
#include "librecuda_internal.h"
#include "nvidia/cl83de.h"
#include "nvidia/ctrl83dedebug.h"

// how long after a kernel completed RM may still mark the channel for an SM exception
static constexpr auto GR_EXCEPTION_REPORT_DELAY = std::chrono::milliseconds(50);

// error codes in the low bits of the warp ESR, as seen on the GPU together with the Xid 13 messages. Other codes are
// reported as a generic launch failure.
static constexpr NvU32 WARP_ESR_ERROR_OUT_OF_RANGE_ADDRESS = 0xe;
static constexpr NvU32 WARP_ESR_ERROR_MISALIGNED_ADDRESS = 0xf;

static libreCudaStatus_t statusOfWarpEsr(NvU32 warp_esr) {
    switch (warp_esr & 0xffff) {
        case WARP_ESR_ERROR_OUT_OF_RANGE_ADDRESS:
            return LIBRECUDA_ERROR_ILLEGAL_ADDRESS;
        case WARP_ESR_ERROR_MISALIGNED_ADDRESS:
            return LIBRECUDA_ERROR_MISALIGNED_ADDRESS;
        default:
            return LIBRECUDA_ERROR_LAUNCH_FAILED;
    }
}

libreCudaStatus_t ctxSmExceptionStatus(LibreCUcontext ctx) {
    int sm_count{};
    if (libreCuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ctx->device) !=
        LIBRECUDA_SUCCESS || sm_count <= 0 || ctx->compute_gpfifos.empty()) {
        return LIBRECUDA_ERROR_LAUNCH_FAILED;
    }
    NvHandle debugger{};
    NV83DE_ALLOC_PARAMETERS alloc_params{.hAppClient = rmClient(),
                                         .hClass3dObject = ctx->compute_gpfifos[0].compute_object};
    if (rm_alloc(rmControlFd(), GT200_DEBUGGER, rmClient(), ctx->device_handle, 0, &alloc_params,
                 sizeof(alloc_params), &debugger) != LIBRECUDA_SUCCESS) {
        return LIBRECUDA_ERROR_LAUNCH_FAILED;
    }
    libreCudaStatus_t status = LIBRECUDA_ERROR_LAUNCH_FAILED;
    bool found = false;
    static NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS params{};
    for (NvU32 first = 0; first < static_cast<NvU32>(sm_count) && !found;
         first += NV83DE_CTRL_DEBUG_MAX_SMS_PER_CALL) {
        memset(&params, 0, sizeof(params));
        // the error state belongs to the GR context, which all compute channels of the context share
        params.hTargetChannel = ctx->compute_gpfifos[0].channel_handle;
        params.startingSM = first;
        params.numSMsToRead = std::min<NvU32>(sm_count - first, NV83DE_CTRL_DEBUG_MAX_SMS_PER_CALL);
        if (rmControl(debugger, NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES, &params, sizeof(params)) !=
            LIBRECUDA_SUCCESS) {
            break;
        }
        for (NvU32 i = 0; i < params.numSMsToRead; i++) {
            const NvU32 warp_esr = params.smErrorStateArray[i].hwwWarpEsr;
            if ((warp_esr & 0xffff) != 0) {
                LIBRECUDA_DEBUG("SM " + std::to_string(first + i) + " reported warp error state " +
                                std::to_string(warp_esr) + " at PC " +
                                std::to_string(params.smErrorStateArray[i].hwwWarpEsrPc64));
                status = statusOfWarpEsr(warp_esr);
                found = true;
                break;
            }
        }
    }
    rmFreeObject(ctx->device_handle, debugger);
    return status;
}

void ctxWaitForGrExceptionReports(LibreCUcontext ctx) {
    if (ctx->launch_count.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const auto last_launch = std::chrono::steady_clock::time_point(
        std::chrono::steady_clock::duration(ctx->last_launch_time.load(std::memory_order_relaxed)));
    std::this_thread::sleep_until(last_launch + GR_EXCEPTION_REPORT_DELAY);
}
