// Texture objects (bindless textures).
//
// Kernels fetch from textures with a handle (cudaTextureObject_t) that indexes the context's texture header pool;
// samplers come from the texture sampler pool (verified on the GPU: the handle is the header index; Blackwell reads
// V2 headers). Both pools live in device memory, mapped to the CPU so that headers are
// written directly. Every launch invalidates the texture header / sampler / data caches (QMD), so headers written
// before a launch are seen by it. Every compute channel must be pointed at the pools (programmed per queue, like
// local memory). Header layouts: clcb97tex.h (open-gpu-kernel-modules).

#include <cstring>

#include "cmdqueue.h"
#include "librecuda_internal.h"
#include "nvidia/g_allclasses.h"

static constexpr NvU32 TEXTURE_POOL_ENTRIES = LIBRECUDA_TEXTURE_POOL_ENTRIES;
static constexpr NvU32 TEXTURE_POOL_ENTRY_SIZE = 32;

// TEXHEAD_1D (one dimensional buffer) field values, from clcb97tex.h (also used for TEXHEAD_V2_1DRT's sources)
static constexpr NvU32 TEXHEAD_HEADER_VERSION_ONE_D_BUFFER = 0;
static constexpr NvU32 TEXHEAD_TEXTURE_TYPE_ONE_D_BUFFER = 6;
static constexpr NvU32 TEXHEAD_SOURCE_IN_ZERO = 0;
static constexpr NvU32 TEXHEAD_SOURCE_IN_R = 2;
static constexpr NvU32 TEXHEAD_SOURCE_IN_ONE_FLOAT = 7;
static constexpr NvU32 TEXHEAD_SOURCE_IN_ONE_INT = 6;
// TEXHEAD_V2_1DRT field values
static constexpr NvU32 TEXHEAD_V2_DATA_TYPE_UNORM = 0;
static constexpr NvU32 TEXHEAD_V2_DATA_TYPE_SNORM = 1;
static constexpr NvU32 TEXHEAD_V2_DATA_TYPE_FLOAT = 2;
static constexpr NvU32 TEXHEAD_V2_DATA_TYPE_UINT = 6;
static constexpr NvU32 TEXHEAD_V2_DATA_TYPE_SINT = 7;
static constexpr NvU32 TEXHEAD_V2_HEADER_VERSION_ONE_D_RAW_TYPED = 5;

static void setBits(NvU32 *words, NvU32 high, NvU32 low, NvU32 value) {
    for (NvU32 bit = low; bit <= high; bit++) {
        const NvU32 word = bit / 32, shift = bit % 32;
        words[word] = (words[word] & ~(1u << shift)) | (((value >> (bit - low)) & 1u) << shift);
    }
}

libreCudaStatus_t ctxEnsureTexturePools(LibreCUcontext ctx) {
    if (ctx->texture_header_pool_va != 0) {
        LIBRECUDA_SUCCEED();
    }
    const NvU64 pool_size = NvU64(TEXTURE_POOL_ENTRIES) * TEXTURE_POOL_ENTRY_SIZE;
    NvU64 header_pool{}, sampler_pool{};
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, pool_size, false, false, true, 0, &header_pool));
    LIBRECUDA_ERR_PROPAGATE(gpuAlloc(ctx, pool_size, false, false, true, 0, &sampler_pool));
    // device memory isn't zeroed; an all-zero header or sampler is invalid
    memset(reinterpret_cast<void *>(header_pool), 0, pool_size);
    memset(reinterpret_cast<void *>(sampler_pool), 0, pool_size);
    // index 0 is never handed out: handle 0 means "no texture"
    ctx->free_texture_headers.clear();
    for (NvU32 i = TEXTURE_POOL_ENTRIES - 1; i >= 1; i--) {
        ctx->free_texture_headers.push_back(i);
    }
    ctx->texture_sampler_pool_va = sampler_pool;
    ctx->texture_header_pool_va = header_pool;
    LIBRECUDA_SUCCEED();
}

/**
 * @return the TEXHEAD component size layout for channels of bits each, 0 if unsupported
 */
static NvU32 componentSizes(NvU32 channels, NvU32 bits) {
    switch (bits * 10 + channels) {
        case 321: return 0x0f; // R32
        case 322: return 0x04; // R32_G32
        case 324: return 0x01; // R32_G32_B32_A32
        case 161: return 0x1b; // R16
        case 162: return 0x0c; // R16_G16
        case 164: return 0x03; // R16_G16_B16_A16
        case 81: return 0x1d;  // R8
        case 84: return 0x08;  // A8B8G8R8
        default: return 0;
    }
}

// source of output channel c: the texture's channels, missing ones read as 0 (alpha as 1)
static NvU32 channelSource(NvU32 c, NvU32 channels, LibreCuTexFormat format) {
    if (c < channels) {
        return TEXHEAD_SOURCE_IN_R + c;
    }
    if (c == 3) {
        const bool integer = format == LIBRECUDA_TEX_FORMAT_SINT || format == LIBRECUDA_TEX_FORMAT_UINT;
        return integer ? TEXHEAD_SOURCE_IN_ONE_INT : TEXHEAD_SOURCE_IN_ONE_FLOAT;
    }
    return TEXHEAD_SOURCE_IN_ZERO;
}

/**
 * TEXHEAD_V2_1DRT (one dimensional raw typed buffer): the header format Blackwell uses (the V1 format below reads as
 * an invalid texture there: fetches return 0)
 */
static void buildHeaderV2OneDRaw(NvU32 *header, NvU64 address, NvU64 width, LibreCuTexFormat format, NvU32 channels,
                                 NvU32 components) {
    setBits(header, 31, 0, NvU32(address));
    setBits(header, 63, 32, NvU32(address >> 32));
    setBits(header, 95, 64, NvU32(width - 1));
    for (NvU32 c = 0; c < 4; c++) {
        setBits(header, 98 + 3 * c, 96 + 3 * c, channelSource(c, channels, format));
    }
    NvU32 data_type{};
    switch (format) {
        case LIBRECUDA_TEX_FORMAT_FLOAT: data_type = TEXHEAD_V2_DATA_TYPE_FLOAT; break;
        case LIBRECUDA_TEX_FORMAT_SINT: data_type = TEXHEAD_V2_DATA_TYPE_SINT; break;
        case LIBRECUDA_TEX_FORMAT_UINT: data_type = TEXHEAD_V2_DATA_TYPE_UINT; break;
        case LIBRECUDA_TEX_FORMAT_UNORM: data_type = TEXHEAD_V2_DATA_TYPE_UNORM; break;
        case LIBRECUDA_TEX_FORMAT_SNORM: data_type = TEXHEAD_V2_DATA_TYPE_SNORM; break;
    }
    setBits(header, 111, 108, data_type);
    setBits(header, 117, 112, components);
    setBits(header, 127, 124, TEXHEAD_V2_HEADER_VERSION_ONE_D_RAW_TYPED);
}

/**
 * TEXHEAD_1D (one dimensional buffer), the format before Hopper (untested: no such GPU here)
 */
static void buildHeaderOneDBuffer(NvU32 *header, NvU64 address, NvU64 width, LibreCuTexFormat format, NvU32 channels,
                                  NvU32 components) {
    setBits(header, 6, 0, components);
    NvU32 data_type{};
    switch (format) {
        case LIBRECUDA_TEX_FORMAT_FLOAT: data_type = 7; break;
        case LIBRECUDA_TEX_FORMAT_SINT: data_type = 3; break;
        case LIBRECUDA_TEX_FORMAT_UINT: data_type = 4; break;
        case LIBRECUDA_TEX_FORMAT_UNORM: data_type = 2; break;
        case LIBRECUDA_TEX_FORMAT_SNORM: data_type = 1; break;
    }
    for (NvU32 c = 0; c < 4; c++) {
        // R, G, B, A data types (bits 7..18) and X, Y, Z, W sources (bits 19..30)
        setBits(header, 9 + 3 * c, 7 + 3 * c, data_type);
        setBits(header, 21 + 3 * c, 19 + 3 * c, channelSource(c, channels, format));
    }
    setBits(header, 63, 32, NvU32(address));
    setBits(header, 80, 64, NvU32(address >> 32));
    setBits(header, 87, 85, TEXHEAD_HEADER_VERSION_ONE_D_BUFFER);
    setBits(header, 111, 96, NvU32((width - 1) >> 16));
    setBits(header, 143, 128, NvU32((width - 1) & 0xffff));
    setBits(header, 154, 151, TEXHEAD_TEXTURE_TYPE_ONE_D_BUFFER);
}

libreCudaStatus_t libreCuTexObjectCreateLinear(LibreCUtexObject *pTexObject, void *devPtr, size_t sizeBytes,
                                               LibreCuTexFormat format, uint32_t channels, uint32_t bitsPerChannel) {
    LIBRECUDA_VALIDATE(pTexObject != nullptr && devPtr != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    const NvU32 components = componentSizes(channels, bitsPerChannel);
    LIBRECUDA_VALIDATE(components != 0, LIBRECUDA_ERROR_NOT_SUPPORTED);
    const bool normalized = format == LIBRECUDA_TEX_FORMAT_UNORM || format == LIBRECUDA_TEX_FORMAT_SNORM;
    LIBRECUDA_VALIDATE(!normalized || bitsPerChannel <= 16, LIBRECUDA_ERROR_INVALID_VALUE);
    const NvU64 element_size = NvU64(channels) * bitsPerChannel / 8;
    const NvU64 width = sizeBytes / element_size;
    LIBRECUDA_VALIDATE(width >= 1 && width <= (NvU64(1) << 27), LIBRECUDA_ERROR_INVALID_VALUE);
    const NvU64 address = reinterpret_cast<NvU64>(devPtr);
    LIBRECUDA_VALIDATE(address % LIBRECUDA_TEXTURE_LINEAR_ALIGNMENT == 0 && address < (NvU64(1) << 49),
                       LIBRECUDA_ERROR_INVALID_VALUE);

    LIBRECUDA_ERR_PROPAGATE(ctxEnsureTexturePools(current_ctx));
    LIBRECUDA_VALIDATE(!current_ctx->free_texture_headers.empty(), LIBRECUDA_ERROR_OUT_OF_MEMORY);
    const NvU32 index = current_ctx->free_texture_headers.back();
    current_ctx->free_texture_headers.pop_back();

    NvU32 header[8]{};
    if (current_ctx->device->compute_class >= BLACKWELL_COMPUTE_A) {
        buildHeaderV2OneDRaw(header, address, width, format, channels, components);
    } else {
        buildHeaderOneDBuffer(header, address, width, format, channels, components);
    }
    memcpy(reinterpret_cast<NvU8 *>(current_ctx->texture_header_pool_va) + NvU64(index) * TEXTURE_POOL_ENTRY_SIZE,
           header, sizeof(header));

    *pTexObject = index;
    LIBRECUDA_SUCCEED();
}

libreCudaStatus_t libreCuTexObjectDestroy(LibreCUtexObject texObject) {
    LIBRECUDA_ENSURE_CTX_VALID();
    LIBRECUDA_VALIDATE(texObject > 0 && texObject < TEXTURE_POOL_ENTRIES && current_ctx->texture_header_pool_va != 0,
                       LIBRECUDA_ERROR_INVALID_VALUE);
    // kernels still using the header must complete first (like CUDA, the caller ensures that)
    memset(reinterpret_cast<NvU8 *>(current_ctx->texture_header_pool_va) + NvU64(texObject) * TEXTURE_POOL_ENTRY_SIZE,
           0, TEXTURE_POOL_ENTRY_SIZE);
    current_ctx->free_texture_headers.push_back(static_cast<NvU32>(texObject));
    LIBRECUDA_SUCCEED();
}
