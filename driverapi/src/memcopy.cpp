#include "memcopy.h"

#include <cmdqueue.h>
#include <librecuda_internal.h>
#include <unordered_map>
#include <vector>
#include "memcopy_kernels.h"

#define BLOCK_SIZE 256u

// grid-stride kernels: enough blocks to fill the GPU, but no more than there is work for
#define BLOCKS_PER_SM 16u

struct MemopKernels {
    LibreCUmodule module{};
    // kernels for device memory copy / fill; null if the module of the architecture predates them
    LibreCUFunction copyVec16{}, copyVec4{}, copyBytes{}, setVec16{}, setBytes{};
    uint32_t maxBlocks{};
};

static std::unordered_map<int, MemopKernels> kernels;

libreCudaStatus_t loadMemcpyKernelsIfNeeded(const LibreCUdevice device) {
    if (kernels.count(device->ordinal) != 0) {
        return LIBRECUDA_SUCCESS;
    }

    int compute_capability_major{}, compute_capability_minor{};
    LIBRECUDA_ERR_PROPAGATE(
        libreCuDeviceGetAttribute(&compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    LIBRECUDA_ERR_PROPAGATE(
        libreCuDeviceGetAttribute(&compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    int compute_capability = compute_capability_major * 10 + compute_capability_minor;

    const uint8_t *cubin_data;
    size_t cubin_size{};

#define ARCH_KERNEL(capability) \
            case capability: \
            cubin_data = MEMCPY_SM_##capability; \
            cubin_size = sizeof(MEMCPY_SM_##capability); \
            break;

    switch (compute_capability) {
        ARCH_KERNEL(50)
        ARCH_KERNEL(53)
        ARCH_KERNEL(60)
        ARCH_KERNEL(61)
        ARCH_KERNEL(62)
        ARCH_KERNEL(70)
        ARCH_KERNEL(72)
        ARCH_KERNEL(75)
        ARCH_KERNEL(80)
        ARCH_KERNEL(86)
        ARCH_KERNEL(87)
        ARCH_KERNEL(89)
        ARCH_KERNEL(90)
        ARCH_KERNEL(120)
        default:
            LIBRECUDA_DEBUG("No memcpy kernels for compute capability " + std::to_string(compute_capability) +
                            ", using the copy engine only");
            kernels[device->ordinal] = MemopKernels{};
            LIBRECUDA_SUCCEED();
    }
    MemopKernels k{};
    LIBRECUDA_ERR_PROPAGATE(libreCuModuleLoadData(&k.module, cubin_data, cubin_size));
    auto optional = [&](LibreCUFunction *pOut, const char *name) {
        if (libreCuModuleGetFunction(pOut, k.module, name) != LIBRECUDA_SUCCESS) {
            *pOut = nullptr;
        }
    };
    optional(&k.copyVec16, "memcpyVec16");
    optional(&k.copyVec4, "memcpyVec4");
    optional(&k.copyBytes, "memcpyBytes");
    optional(&k.setVec16, "memsetVec16");
    optional(&k.setBytes, "memsetBytes");
    int sms{};
    LIBRECUDA_ERR_PROPAGATE(libreCuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    k.maxBlocks = static_cast<uint32_t>(std::max(1, sms)) * BLOCKS_PER_SM;
    kernels[device->ordinal] = k;
    LIBRECUDA_SUCCEED();
}

void releaseMemcpyKernels(const LibreCUdevice device) {
    auto it = kernels.find(device->ordinal);
    if (it != kernels.end()) {
        delete it->second.module;
        kernels.erase(it);
    }
}

struct KernelLaunch {
    LibreCUFunction function;
    void *dst;
    const void *src;
    uint32_t value;
    size_t n;
};

static libreCudaStatus_t launchAll(const std::vector<KernelLaunch> &launches, uint32_t maxBlocks,
                                   LibreCUstream stream, bool async) {
    for (size_t i = 0; i < launches.size(); i++) {
        const auto &l = launches[i];
        const uint32_t blocks = static_cast<uint32_t>(
            std::min<size_t>(maxBlocks, (l.n + BLOCK_SIZE - 1) / BLOCK_SIZE));
        void *dst = l.dst;
        const void *src = l.src;
        uint32_t value = l.value;
        size_t n = l.n;
        void *copy_params[] = {&dst, &src, &n};
        void *set_params[] = {&dst, &value, &n};
        // async launches may overlap with preceding kernels of the stream, so only use them if the caller allows it
        LIBRECUDA_ERR_PROPAGATE(libreCuLaunchKernel(l.function, blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, stream,
                                                    l.src != nullptr ? copy_params : set_params, 3, nullptr,
                                                    async));
    }
    LIBRECUDA_SUCCEED();
}

bool memcpyD2DKernelsAvailable() {
    auto it = kernels.find(current_ctx->device->ordinal);
    return it != kernels.end() && it->second.copyVec16 != nullptr && it->second.copyVec4 != nullptr &&
           it->second.copyBytes != nullptr;
}

bool memsetKernelsAvailable() {
    auto it = kernels.find(current_ctx->device->ordinal);
    return it != kernels.end() && it->second.setVec16 != nullptr && it->second.setBytes != nullptr;
}

libreCudaStatus_t memcpyD2D(void *dst, const void *src, size_t size, LibreCUstream stream, bool async) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LIBRECUDA_VALIDATE(memcpyD2DKernelsAvailable(), LIBRECUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);
    const MemopKernels &k = kernels[current_ctx->device->ordinal];
    if (size == 0) {
        LIBRECUDA_SUCCEED();
    }
    auto *d = static_cast<uint8_t *>(dst);
    const auto *s = static_cast<const uint8_t *>(src);
    const auto misalign = [](const void *p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a; };

    std::vector<KernelLaunch> launches{};
    // widest element size both pointers can be aligned to
    const size_t width = misalign(d, 16) == misalign(s, 16) ? 16 : misalign(d, 4) == misalign(s, 4) ? 4 : 1;
    if (width == 1) {
        launches.push_back({k.copyBytes, d, s, 0, size});
    } else {
        const size_t head = std::min(size, (width - misalign(d, width)) % width);
        const size_t body = (size - head) / width;
        const size_t tail = size - head - body * width;
        if (head > 0) {
            launches.push_back({k.copyBytes, d, s, 0, head});
        }
        if (body > 0) {
            launches.push_back({width == 16 ? k.copyVec16 : k.copyVec4, d + head, s + head, 0, body});
        }
        if (tail > 0) {
            launches.push_back({k.copyBytes, d + head + body * width, s + head + body * width, 0, tail});
        }
    }
    return launchAll(launches, k.maxBlocks, stream, async);
}

libreCudaStatus_t memsetD8Kernel(void *dst, uint8_t value, size_t size, LibreCUstream stream, bool async) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LIBRECUDA_VALIDATE(memsetKernelsAvailable(), LIBRECUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);
    const MemopKernels &k = kernels[current_ctx->device->ordinal];
    if (size == 0) {
        LIBRECUDA_SUCCEED();
    }
    auto *d = static_cast<uint8_t *>(dst);
    const size_t head = std::min(size, (16 - reinterpret_cast<uintptr_t>(d) % 16) % 16);
    const size_t body = (size - head) / 16;
    const size_t tail = size - head - body * 16;
    const uint32_t value4 = value * 0x01010101u;

    std::vector<KernelLaunch> launches{};
    if (head > 0) {
        launches.push_back({k.setBytes, d, nullptr, value, head});
    }
    if (body > 0) {
        launches.push_back({k.setVec16, d + head, nullptr, value4, body});
    }
    if (tail > 0) {
        launches.push_back({k.setBytes, d + head + body * 16, nullptr, value, tail});
    }
    return launchAll(launches, k.maxBlocks, stream, async);
}
