// Device side printf for LibreCUDA.
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
// Compiled with -rdc=true: the caller reaches it with CALL.ABS and an absolute return address, and only relocatable
// compilation makes ptxas emit the matching absolute return (RET.ABS; whole program compilation emits RET.REL).
//
// The function runs as part of the calling kernel, so it must not use the constant banks the toolchain manages:
// constant bank 4 (addresses of globals, string literals) would be the caller's. It finds its state through %envreg8 /
// %envreg9, which LibreCUDA writes into the driver area of constant bank 0 for every launch, and uses no string
// literals. It must also fit the caller's registers (compiled with -maxrregcount=24, the minimum ptxas gives a kernel
// calling vprintf; the loader raises callers to this function's register count) and the default 1 KiB stack.

#include <cstdint>

// lives in device memory (atomics on host memory would need PCIe atomics)
struct PrintfState {
    // pinned host buffer receiving the records, and its size in bytes
    unsigned long long buffer;
    unsigned int capacity;
    // bytes allocated in the buffer so far; reset by the host when the context is idle
    unsigned int offset;
    // records that did not fit
    unsigned int dropped;
};

static __device__ __forceinline__ PrintfState *printfState() {
    unsigned int hi, lo;
    asm volatile("mov.u32 %0, %%envreg8;" : "=r"(hi));
    asm volatile("mov.u32 %0, %%envreg9;" : "=r"(lo));
    return reinterpret_cast<PrintfState *>((static_cast<unsigned long long>(hi) << 32) | lo);
}

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

extern "C" __device__ __noinline__ int librecuda_vprintf(const char *fmt, const char *args) {
    PrintfState *state = printfState();
    if (state == nullptr || state->buffer == 0 || fmt == nullptr) {
        return -1;
    }
    const unsigned int fmt_size = stringSize(fmt);

    // size of the record
    unsigned int size = 8 + align8(fmt_size);
    forEachArgument(fmt, args, [&](int kind, const char *arg) {
        size += kind == 2 ? align8(4 + stringSize(*reinterpret_cast<const char *const *>(arg))) : 8;
    });

    const unsigned int offset = atomicAdd(&state->offset, size);
    if (offset + size > state->capacity) {
        atomicAdd(&state->dropped, 1u);
        return -1;
    }
    char *record = reinterpret_cast<char *>(state->buffer) + offset;

    // body
    reinterpret_cast<unsigned int *>(record)[1] = fmt_size;
    unsigned int w = 8;
    for (unsigned int i = 0; i < fmt_size; i++) {
        record[w + i] = fmt[i];
    }
    w += align8(fmt_size);
    forEachArgument(fmt, args, [&](int kind, const char *arg) {
        if (kind == 2) {
            const char *s = *reinterpret_cast<const char *const *>(arg);
            const unsigned int n = stringSize(s);
            *reinterpret_cast<unsigned int *>(record + w) = n;
            for (unsigned int i = 0; i < n; i++) {
                record[w + 4 + i] = s[i];
            }
            w += align8(4 + n);
        } else {
            *reinterpret_cast<unsigned long long *>(record + w) =
                kind == 0 ? *reinterpret_cast<const unsigned int *>(arg)
                          : *reinterpret_cast<const unsigned long long *>(arg);
            w += 8;
        }
    });

    // publish: the size must become visible after the body
    __threadfence_system();
    *reinterpret_cast<volatile unsigned int *>(record) = size;
    return static_cast<int>(fmt_size - 1);
}
