// Device side printf: host part.
//
// Kernels call vprintf, which the loader resolves to librecuda_vprintf from the printf module
// (driverapi/kernels/device_runtime/device_runtime.cu). It writes records (format string + arguments) into a pinned host buffer; this
// file prints them when the host synchronizes with the GPU, like CUDA does.

#include <cstdio>
#include <cstring>
#include <string>

#include "cmdqueue.h"
#include "librecuda_internal.h"
#include "device_runtime_kernels.h"
#include "device_runtime_state.h"

// the device heap's bump pointer starts here: offset 0 marks empty free lists (device_runtime.cu)
static constexpr NvU64 HEAP_FIRST_BLOCK = 256;

// external functions kernels call and the device runtime functions implementing them
static const std::pair<const char *, const char *> DEVICE_RUNTIME_FUNCTIONS[] = {
    {"vprintf", "librecuda_vprintf"},
    {"__assertfail", "librecuda_assertfail"},
    {"malloc", "librecuda_malloc"},
    {"free", "librecuda_free"},
    // the device runtime API (cuda_device_runtime_api.h); calls are redirected here even when libcudadevrt is linked
    // into the image
    {"__cudaCDP2GetParameterBufferV2", "librecuda_get_parameter_buffer_v2"},
    {"__cudaCDP2LaunchDeviceV2", "librecuda_launch_device_v2"},
    {"__cudaCDP2LaunchDeviceV2_ptsz", "librecuda_launch_device_v2"},
    {"__cudaCDP2GetParameterBuffer", "librecuda_get_parameter_buffer"},
    {"__cudaCDP2LaunchDevice", "librecuda_launch_device"},
    {"__cudaCDP2LaunchDevice_ptsz", "librecuda_launch_device"},
    {"__cudaCDP2GetLastError", "librecuda_get_last_error"},
    {"__cudaCDP2PeekAtLastError", "librecuda_peek_at_last_error"},
    {"__cudaCDP2GetDevice", "librecuda_get_device"},
    {"__cudaCDP2GetDeviceCount", "librecuda_get_device_count"},
    {"__cudaCDP2StreamCreateWithFlags", "librecuda_stream_create_with_flags"},
    {"__cudaCDP2StreamDestroy", "librecuda_stream_destroy"},
    {"__cudaCDP2Malloc", "librecuda_cdp_malloc"},
    {"__cudaCDP2Free", "librecuda_cdp_free"},
    {"__cudaCDP2RuntimeGetVersion", "librecuda_runtime_get_version"},
    // CUDA graph conditional nodes
    {"cudaGraphSetConditional", "librecuda_graph_set_conditional"},
    {"cudaGraphLaunch", "librecuda_graph_launch"},
    {"cudaGraphKernelNodeSetParam", "librecuda_graph_kernel_node_set_param"},
    {"cudaGraphKernelNodeSetEnabled", "librecuda_graph_kernel_node_set_enabled"},
    {"cudaGraphKernelNodeSetGridDim", "librecuda_graph_kernel_node_set_grid_dim"},
    {"cudaGraphKernelNodeUpdatesApply", "librecuda_graph_kernel_node_updates_apply"},
    {"__cudaCDP2MemcpyAsync", "librecuda_memcpy_async"},
    {"__cudaCDP2MemcpyAsync_ptsz", "librecuda_memcpy_async"},
    {"__cudaCDP2Memcpy2DAsync", "librecuda_memcpy_2d_async"},
    {"__cudaCDP2Memcpy2DAsync_ptsz", "librecuda_memcpy_2d_async"},
    {"__cudaCDP2Memcpy3DAsync", "librecuda_memcpy_3d_async"},
    {"__cudaCDP2Memcpy3DAsync_ptsz", "librecuda_memcpy_3d_async"},
    {"__cudaCDP2MemsetAsync", "librecuda_memset_async"},
    {"__cudaCDP2MemsetAsync_ptsz", "librecuda_memset_async"},
    {"__cudaCDP2Memset2DAsync", "librecuda_memset_2d_async"},
    {"__cudaCDP2Memset2DAsync_ptsz", "librecuda_memset_2d_async"},
    {"__cudaCDP2Memset3DAsync", "librecuda_memset_3d_async"},
    {"__cudaCDP2Memset3DAsync_ptsz", "librecuda_memset_3d_async"},
    {"__cudaCDP2EventCreateWithFlags", "librecuda_event_create_with_flags"},
    {"__cudaCDP2EventRecord", "librecuda_event_no_op"},
    {"__cudaCDP2EventRecord_ptsz", "librecuda_event_no_op"},
    {"__cudaCDP2EventRecordWithFlags", "librecuda_event_no_op"},
    {"__cudaCDP2EventRecordWithFlags_ptsz", "librecuda_event_no_op"},
    {"__cudaCDP2StreamWaitEvent", "librecuda_event_no_op"},
    {"__cudaCDP2StreamWaitEvent_ptsz", "librecuda_event_no_op"},
    {"__cudaCDP2EventDestroy", "librecuda_event_no_op"},
    // replaces the driver calls of NVIDIA's device runtime code (functions not provided above)
    {"librecuda_unsupported", "librecuda_unsupported"},
};

// fmt_size of an assert record (device_runtime.cu)
static constexpr NvU32 ASSERT_RECORD = 0xffffffffu;

static size_t align8(size_t n) {
    return (n + 7) & ~size_t(7);
}

libreCudaStatus_t ctxEnsurePrintf(LibreCUcontext ctx) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT);
    if (ctx->printf_state_va != 0) {
        LIBRECUDA_SUCCEED();
    }
    std::lock_guard ctx_lock(ctx->mutex);
    if (ctx->printf_state_va != 0) {
        LIBRECUDA_SUCCEED();
    }
    LIBRECUDA_VALIDATE(ctx == current_ctx, LIBRECUDA_ERROR_INVALID_CONTEXT);

    int major{}, minor{};
    LIBRECUDA_ERR_PROPAGATE(
        libreCuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ctx->device));
    LIBRECUDA_ERR_PROPAGATE(
        libreCuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, ctx->device));
    const uint8_t *cubin{};
    size_t cubin_size{};
#define PRINTF_ARCH(cc) case cc: cubin = DEVICE_RUNTIME_SM_##cc; cubin_size = sizeof(DEVICE_RUNTIME_SM_##cc); break;
    switch (major * 10 + minor) {
        PRINTF_ARCH(75)
        PRINTF_ARCH(80)
        PRINTF_ARCH(86)
        PRINTF_ARCH(87)
        PRINTF_ARCH(89)
        PRINTF_ARCH(90)
        PRINTF_ARCH(120)
        default:
            LIBRECUDA_DEBUG("device side printf is not available for compute capability " + std::to_string(major) +
                            "." + std::to_string(minor));
            LIBRECUDA_FAIL(LIBRECUDA_ERROR_NOT_SUPPORTED);
    }
#undef PRINTF_ARCH

    LibreCUmodule module{};
    LIBRECUDA_ERR_PROPAGATE(libreCuModuleLoadData(&module, cubin, cubin_size));
    std::unordered_map<std::string, std::pair<NvU64, NvU32>> functions{};
    for (const auto &[extern_name, symbol]: DEVICE_RUNTIME_FUNCTIONS) {
        auto function_it = module->function_symbols.find(symbol);
        LIBRECUDA_VALIDATE(function_it != module->function_symbols.end(), LIBRECUDA_ERROR_INVALID_IMAGE);
        auto registers_it = module->function_registers.find(symbol);
        functions[extern_name] = {function_it->second,
                                  registers_it != module->function_registers.end() ? registers_it->second : 24};
    }

    // state in device memory (the GPU allocates records with atomics), records in pinned host memory
    NvU64 state_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, sizeof(DeviceRuntimeState), false, false, true, 0, &state_va));
    NvU64 buffer_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuHostAlloc(ctx, ctx->printf_buffer_size, &buffer_va));
    auto *state = reinterpret_cast<volatile DeviceRuntimeState *>(state_va);
    state->buffer = buffer_va;
    state->capacity = static_cast<NvU32>(ctx->printf_buffer_size);
    state->offset = 0;
    state->dropped = 0;
    // device memory isn't zeroed: no heap until a module calling malloc / free is loaded
    state->heap_base = 0;
    state->heap_size = 0;
    state->heap_top = 0;
    for (auto &shards: state->free_lists) {
        for (auto &list: shards) {
            list = 0;
        }
    }
    // no launch queue until a module launching kernels is loaded
    state->launch_queue = 0;
    state->launch_capacity = 0;
    state->launch_count = 0;
    state->last_error = 0;
    state->error_slots = 0;
    state->error_slot_sms = 0;
    state->error_slot_warps = 0;
    state->device = ctx->device->ordinal;
    int device_count = 1;
    libreCuDeviceGetCount(&device_count);
    state->device_count = device_count;

    ctx->printf_module = module;
    ctx->device_runtime_functions = std::move(functions);
    ctx->printf_buffer = reinterpret_cast<NvU8 *>(buffer_va);
    ctx->printf_read_offset = 0;
    LIBRECUDA_DEBUG("device runtime: state at " + std::to_string(state_va) + ", buffer at " +
                    std::to_string(buffer_va));
    // last: launches pass the state to kernels once this is set
    ctx->printf_state_va = state_va;
    LIBRECUDA_SUCCEED();
}

/**
 * Allocates the heap for device side malloc (cudaLimitMallocHeapSize) if not done yet
 */
static libreCudaStatus_t ctxEnsureDeviceHeap(LibreCUcontext ctx) {
    auto *state = reinterpret_cast<volatile DeviceRuntimeState *>(ctx->printf_state_va);
    if (state->heap_base != 0) {
        LIBRECUDA_SUCCEED();
    }
    NvU64 heap_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, ctx->malloc_heap_size, false, ctx->malloc_heap_size >= (2 << 20), false, 0,
                                     &heap_va));
    state->heap_size = ctx->malloc_heap_size;
    state->heap_top = HEAP_FIRST_BLOCK;
    state->heap_base = heap_va;
    LIBRECUDA_DEBUG("device heap: " + std::to_string(ctx->malloc_heap_size) + " bytes at " + std::to_string(heap_va));
    LIBRECUDA_SUCCEED();
}

bool ctxDeviceRuntimeFunction(LibreCUcontext ctx, const std::string &name, NvU64 *pAddressOut, NvU32 *pRegistersOut) {
    bool provided = false;
    for (const auto &function: DEVICE_RUNTIME_FUNCTIONS) {
        provided = provided || name == function.first;
    }
    std::lock_guard ctx_lock(ctx->mutex);
    if (!provided || ctxEnsurePrintf(ctx) != LIBRECUDA_SUCCESS) {
        return false;
    }
    const bool allocates = name == "malloc" || name == "free" || name == "__cudaCDP2Malloc" || name == "__cudaCDP2Free";
    if (allocates && ctxEnsureDeviceHeap(ctx) != LIBRECUDA_SUCCESS) {
        return false;
    }
    if ((name.find("LaunchDevice") != std::string::npos || name.find("GetParameterBuffer") != std::string::npos ||
         name.find("Memcpy") != std::string::npos || name.find("Memset") != std::string::npos ||
         name == "cudaGraphLaunch") &&
        ctxEnsureDeviceLaunchQueue(ctx) != LIBRECUDA_SUCCESS) {
        return false;
    }
    const auto &[address, registers] = ctx->device_runtime_functions.at(name);
    if (pAddressOut != nullptr) {
        *pAddressOut = address;
    }
    if (pRegistersOut != nullptr) {
        *pRegistersOut = registers;
    }
    return true;
}

NvU32 ctxDeviceRuntimeRegisters(LibreCUcontext ctx) {
    NvU32 registers = 0;
    for (const auto &[name, function]: ctx->device_runtime_functions) {
        registers = std::max(registers, function.second);
    }
    return registers;
}

/**
 * Formats one conversion with snprintf. The argument types match what printf.cu recorded: 4 byte integers as int,
 * 8 byte integers as long long, floating point as double.
 */
template<typename T>
static void appendFormatted(std::string &out, const std::string &spec, const int *stars, int n_stars, T value) {
    char small[256];
    int n;
    switch (n_stars) {
        case 0: n = snprintf(small, sizeof(small), spec.c_str(), value); break;
        case 1: n = snprintf(small, sizeof(small), spec.c_str(), stars[0], value); break;
        default: n = snprintf(small, sizeof(small), spec.c_str(), stars[0], stars[1], value); break;
    }
    if (n < 0) {
        return;
    }
    if (static_cast<size_t>(n) < sizeof(small)) {
        out.append(small, n);
        return;
    }
    std::string big(n + 1, '\0');
    switch (n_stars) {
        case 0: snprintf(big.data(), big.size(), spec.c_str(), value); break;
        case 1: snprintf(big.data(), big.size(), spec.c_str(), stars[0], value); break;
        default: snprintf(big.data(), big.size(), spec.c_str(), stars[0], stars[1], value); break;
    }
    out.append(big.data(), n);
}

/**
 * Formats a record written by librecuda_vprintf (see the layout in printf.cu)
 */
static void formatRecord(const NvU8 *record, NvU32 size, std::string &out) {
    NvU32 fmt_size{};
    memcpy(&fmt_size, record + 4, sizeof(fmt_size));
    if (fmt_size == 0 || 8 + align8(fmt_size) > size) {
        return;
    }
    const char *fmt = reinterpret_cast<const char *>(record + 8);
    const NvU8 *arg = record + 8 + align8(fmt_size);
    const NvU8 *end = record + size;

    auto next_slot = [&](NvU64 *pValue) {
        if (arg + 8 > end) {
            return false;
        }
        memcpy(pValue, arg, 8);
        arg += 8;
        return true;
    };

    for (const char *p = fmt; *p != 0; p++) {
        if (*p != '%') {
            out.push_back(*p);
            continue;
        }
        if (p[1] == '%') {
            out.push_back('%');
            p++;
            continue;
        }
        std::string spec = "%";
        int stars[2]{};
        int n_stars = 0;
        p++;
        while (*p == '-' || *p == '+' || *p == ' ' || *p == '#' || *p == '0' || *p == '\'') {
            spec.push_back(*p++);
        }
        auto read_star = [&] {
            NvU64 v{};
            next_slot(&v);
            stars[n_stars++] = static_cast<int>(static_cast<NvU32>(v));
            spec.push_back('*');
            p++;
        };
        if (*p == '*') {
            read_star();
        } else {
            while (*p >= '0' && *p <= '9') {
                spec.push_back(*p++);
            }
        }
        if (*p == '.') {
            spec.push_back(*p++);
            if (*p == '*') {
                read_star();
            } else {
                while (*p >= '0' && *p <= '9') {
                    spec.push_back(*p++);
                }
            }
        }
        // length modifiers: 'h'/'hh' are kept (they convert the int argument), the others become "ll"
        std::string length{};
        bool wide = false;
        while (*p == 'h' || *p == 'l' || *p == 'L' || *p == 'z' || *p == 'j' || *p == 't' || *p == 'q') {
            if (*p == 'h') {
                length.push_back('h');
            } else {
                wide = true;
            }
            p++;
        }
        const char conversion = *p;
        if (conversion == 0) {
            break;
        }
        NvU64 value{};
        switch (conversion) {
            case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': case 'c':
                next_slot(&value);
                if (wide && conversion != 'c') {
                    appendFormatted(out, spec + "ll" + conversion, stars, n_stars, static_cast<long long>(value));
                } else {
                    appendFormatted(out, spec + (conversion == 'c' ? "" : length) + conversion, stars, n_stars,
                                    static_cast<int>(static_cast<NvU32>(value)));
                }
                break;
            case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A': {
                next_slot(&value);
                double d;
                memcpy(&d, &value, sizeof(d));
                appendFormatted(out, spec + conversion, stars, n_stars, d);
                break;
            }
            case 'p':
                next_slot(&value);
                appendFormatted(out, spec + 'p', stars, n_stars, reinterpret_cast<void *>(value));
                break;
            case 'n':
                next_slot(&value); // writing back through the pointer is not supported
                break;
            case 's': {
                NvU32 string_size{};
                if (arg + 4 > end) {
                    return;
                }
                memcpy(&string_size, arg, sizeof(string_size));
                if (arg + 4 + string_size > end) {
                    return;
                }
                const char *s = string_size == 0 ? "(null)" : reinterpret_cast<const char *>(arg + 4);
                arg += align8(4 + string_size);
                appendFormatted(out, spec + 's', stars, n_stars, s);
                break;
            }
            default:
                // unknown conversion: print it as written
                out += spec;
                out.push_back(conversion);
                break;
        }
    }
}

/**
 * Formats an assert record (see device_runtime.cu) like CUDA's message
 */
static void formatAssertRecord(const NvU8 *record, NvU32 size, std::string &out) {
    NvU32 header[9];
    if (size < sizeof(header) + 8) {
        return;
    }
    memcpy(header, record, sizeof(header));
    const NvU8 *p = record + 40;
    const NvU8 *end = record + size;
    std::string strings[3];
    for (std::string &s: strings) {
        NvU32 n{};
        if (p + 4 > end) {
            return;
        }
        memcpy(&n, p, sizeof(n));
        if (p + 4 + n > end) {
            return;
        }
        s = n == 0 ? std::string("(null)") : std::string(reinterpret_cast<const char *>(p + 4), n - 1);
        p += align8(4 + n);
    }
    char line[512];
    snprintf(line, sizeof(line), ":%u: %s: block: [%u,%u,%u], thread: [%u,%u,%u] Assertion `", header[2],
             strings[2].c_str(), header[3], header[4], header[5], header[6], header[7], header[8]);
    out += strings[1] + line + strings[0] + "` failed.\n";
}

// only try-locks (see the lock order in librecuda_internal.h): a queue in use counts as busy
static bool allQueuesIdle(LibreCUcontext ctx) {
    std::unique_lock ctx_lock(ctx->mutex, std::try_to_lock);
    if (!ctx_lock.owns_lock()) {
        return false;
    }
    for (NvCommandQueue *queue: ctx->queues) {
        if (!queue->tryIsIdle()) {
            return false;
        }
    }
    return true;
}

libreCudaStatus_t ctxFlushPrintf(LibreCUcontext ctx) {
    if (ctx == nullptr || ctx->printf_state_va == 0) {
        LIBRECUDA_SUCCEED();
    }
    std::lock_guard printf_lock(ctx->printf_mutex);
    auto *state = reinterpret_cast<volatile DeviceRuntimeState *>(ctx->printf_state_va);
    // decided before reading the records: if the GPU was idle, every allocated record is complete
    const bool idle = allQueuesIdle(ctx);

    std::string out{}, assert_messages{};
    const size_t capacity = ctx->printf_buffer_size;
    while (ctx->printf_read_offset + 8 <= capacity) {
        const NvU8 *record = ctx->printf_buffer + ctx->printf_read_offset;
        const NvU32 size = *reinterpret_cast<const volatile NvU32 *>(record);
        if (size == 0) {
            break; // not written (yet)
        }
        if (size < 8 || ctx->printf_read_offset + size > capacity) {
            break; // corrupt
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        NvU32 fmt_size{};
        memcpy(&fmt_size, record + 4, sizeof(fmt_size));
        if (fmt_size == ASSERT_RECORD) {
            ctx->device_assert_failed = true;
            // printf output so far first, so that the output stays in order
            fwrite(out.data(), 1, out.size(), stdout);
            out.clear();
            fflush(stdout);
            assert_messages.clear();
            formatAssertRecord(record, size, assert_messages);
            fwrite(assert_messages.data(), 1, assert_messages.size(), stderr);
        } else {
            formatRecord(record, size, out);
        }
        ctx->printf_read_offset += size;
    }
    if (!out.empty()) {
        fwrite(out.data(), 1, out.size(), stdout);
    }

    if (idle) {
        const NvU32 dropped = state->dropped;
        if (dropped != 0) {
            fprintf(stderr, "LibreCUDA: %u device printf calls were dropped: the printf buffer (%zu bytes) is full "
                            "(cudaLimitPrintfFifoSize)\n", dropped, capacity);
            state->dropped = 0;
        }
        // start over at the beginning of the buffer
        memset(ctx->printf_buffer, 0, std::min<size_t>(capacity, std::max<size_t>(ctx->printf_read_offset,
                                                                                   state->offset)));
        state->offset = 0;
        ctx->printf_read_offset = 0;
    }
    LIBRECUDA_SUCCEED();
}

void ctxReleasePrintf(LibreCUcontext ctx) {
    // the module's, the state's and the buffer's memory are context allocations and freed with the context
    delete ctx->printf_module;
    ctx->printf_module = nullptr;
    ctx->printf_state_va = 0;
    ctx->printf_buffer = nullptr;
}

libreCudaStatus_t libreCuCtxSetPrintfFifoSize(size_t size) {
    LIBRECUDA_ENSURE_CTX_VALID();
    std::lock_guard ctx_lock(current_ctx->mutex);
    // like CUDA, the size can't change once the buffer exists
    LIBRECUDA_VALIDATE(current_ctx->printf_state_va == 0, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(size >= 4096 && size <= 0xffff0000ull, LIBRECUDA_ERROR_INVALID_VALUE);
    current_ctx->printf_buffer_size = align8(size);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxGetPrintfFifoSize(size_t *pSize) {
    LIBRECUDA_VALIDATE(pSize != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    *pSize = current_ctx->printf_buffer_size;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxSetMallocHeapSize(size_t size) {
    LIBRECUDA_ENSURE_CTX_VALID();
    std::lock_guard ctx_lock(current_ctx->mutex);
    // like CUDA, the size can't change once the heap exists
    const bool heap_exists = current_ctx->printf_state_va != 0 &&
                             reinterpret_cast<volatile DeviceRuntimeState *>(current_ctx->printf_state_va)->heap_base;
    LIBRECUDA_VALIDATE(!heap_exists, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(size >= 4096, LIBRECUDA_ERROR_INVALID_VALUE);
    current_ctx->malloc_heap_size = align8(size);
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxGetMallocHeapSize(size_t *pSize) {
    LIBRECUDA_VALIDATE(pSize != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    *pSize = current_ctx->malloc_heap_size;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuCtxFlushPrintf() {
    LIBRECUDA_ENSURE_CTX_VALID();
    return ctxFlushPrintf(current_ctx);
}
