// Device side printf: host part.
//
// Kernels call vprintf, which the loader resolves to librecuda_vprintf from the printf module
// (driverapi/kernels/printf/printf.cu). It writes records (format string + arguments) into a pinned host buffer; this
// file prints them when the host synchronizes with the GPU, like CUDA does.

#include <cstdio>
#include <cstring>
#include <string>

#include "cmdqueue.h"
#include "librecuda_internal.h"
#include "printf_kernels.h"

// matches PrintfState in printf.cu
struct PrintfState {
    NvU64 buffer;
    NvU32 capacity;
    NvU32 offset;
    NvU32 dropped;
};

static const char *const VPRINTF_SYMBOL = "librecuda_vprintf";

static size_t align8(size_t n) {
    return (n + 7) & ~size_t(7);
}

libreCudaStatus_t ctxEnsurePrintf(LibreCUcontext ctx) {
    LIBRECUDA_VALIDATE(ctx != nullptr, LIBRECUDA_ERROR_INVALID_CONTEXT);
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
#define PRINTF_ARCH(cc) case cc: cubin = PRINTF_SM_##cc; cubin_size = sizeof(PRINTF_SM_##cc); break;
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
    auto function_it = module->function_symbols.find(VPRINTF_SYMBOL);
    LIBRECUDA_VALIDATE(function_it != module->function_symbols.end(), LIBRECUDA_ERROR_INVALID_IMAGE);
    auto registers_it = module->function_registers.find(VPRINTF_SYMBOL);

    // state in device memory (the GPU allocates records with atomics), records in pinned host memory
    NvU64 state_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, sizeof(PrintfState), false, false, true, 0, &state_va));
    NvU64 buffer_va{};
    LIBRECUDA_ERR_PROPAGATE(gpuHostAlloc(ctx, ctx->printf_buffer_size, &buffer_va));
    auto *state = reinterpret_cast<volatile PrintfState *>(state_va);
    state->buffer = buffer_va;
    state->capacity = static_cast<NvU32>(ctx->printf_buffer_size);
    state->offset = 0;
    state->dropped = 0;

    ctx->printf_module = module;
    ctx->printf_function_va = function_it->second;
    ctx->printf_function_registers = registers_it != module->function_registers.end() ? registers_it->second : 24;
    ctx->printf_buffer = reinterpret_cast<NvU8 *>(buffer_va);
    ctx->printf_read_offset = 0;
    LIBRECUDA_DEBUG("device printf: vprintf at " + std::to_string(ctx->printf_function_va) + " (" +
                    std::to_string(ctx->printf_function_registers) + " registers), state at " +
                    std::to_string(state_va) + ", buffer at " + std::to_string(buffer_va));
    // last: launches pass the state to kernels once this is set
    ctx->printf_state_va = state_va;
    LIBRECUDA_SUCCEED();
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

static bool allQueuesIdle(LibreCUcontext ctx) {
    for (NvCommandQueue *queue: ctx->queues) {
        if (queue->queryExecution() != LIBRECUDA_SUCCESS) {
            return false;
        }
    }
    return true;
}

libreCudaStatus_t ctxFlushPrintf(LibreCUcontext ctx) {
    if (ctx == nullptr || ctx->printf_state_va == 0) {
        LIBRECUDA_SUCCEED();
    }
    auto *state = reinterpret_cast<volatile PrintfState *>(ctx->printf_state_va);
    // decided before reading the records: if the GPU was idle, every allocated record is complete
    const bool idle = allQueuesIdle(ctx);

    std::string out{};
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
        formatRecord(record, size, out);
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

libreCudaStatus_t libreCuCtxFlushPrintf() {
    LIBRECUDA_ENSURE_CTX_VALID();
    return ctxFlushPrintf(current_ctx);
}
