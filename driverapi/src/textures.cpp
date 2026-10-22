// Texture objects (bindless textures).
//
// Kernels fetch from textures with a handle (cudaTextureObject_t, TEX / TLD use its low 32 bits) that is the index
// of the texture's header in the context's texture header pool. The QMD selects SAMPLER_INDEX_VIA_HEADER_INDEX, so the
// sampler comes from the same index in the sampler pool: every texture object has its own header and sampler. In
// that mode the whole handle is the header index; a handle with a sampler index in bits 20.. (the "independent"
// layout) points past the pool and reads as an invalid texture (all zeros, no error), which is what kept 2D textures
// from working at first. Both pools live in device memory, mapped to the CPU so that headers are written directly.
// Every launch invalidates the texture header / sampler / data caches (QMD), so headers written before a launch are
// seen by it. Every compute channel must be pointed at the pools (programmed per queue, like local memory).
//
// Header and sampler layouts: clcb97tex.h (open-gpu-kernel-modules; Hopper); NVIDIA's clcd97tex.h / clce97tex.h for
// Blackwell (open-gpu-doc) define the same V2 layouts. Blackwell reads only V2 headers (V1 headers fetch zeros).

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
static constexpr NvU32 TEXHEAD_V2_HEADER_VERSION_PITCH = 2;
static constexpr NvU32 TEXHEAD_V2_BORDER_SOURCE_BORDER_COLOR = 1;
static constexpr NvU32 TEXHEAD_TEXTURE_TYPE_TWO_D_NO_MIPMAP = 7;

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
    std::lock_guard ctx_lock(ctx->mutex);
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
        case 82: return 0x18;  // G8R8
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

static NvU32 v2DataType(LibreCuTexFormat format) {
    switch (format) {
        case LIBRECUDA_TEX_FORMAT_FLOAT: return TEXHEAD_V2_DATA_TYPE_FLOAT;
        case LIBRECUDA_TEX_FORMAT_SINT: return TEXHEAD_V2_DATA_TYPE_SINT;
        case LIBRECUDA_TEX_FORMAT_UINT: return TEXHEAD_V2_DATA_TYPE_UINT;
        case LIBRECUDA_TEX_FORMAT_UNORM: return TEXHEAD_V2_DATA_TYPE_UNORM;
        case LIBRECUDA_TEX_FORMAT_SNORM: return TEXHEAD_V2_DATA_TYPE_SNORM;
    }
    return TEXHEAD_V2_DATA_TYPE_FLOAT;
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
    setBits(header, 111, 108, v2DataType(format));
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

// The handle of a texture object: its index in both pools (see above)
static LibreCUtexObject textureHandle(NvU32 index) {
    return index;
}

static NvU8 *poolEntry(NvU64 pool, NvU32 index) {
    return reinterpret_cast<NvU8 *>(pool) + NvU64(index) * TEXTURE_POOL_ENTRY_SIZE;
}

/**
 * Writes the header and sampler of a new texture object and returns its handle
 */
static libreCudaStatus_t createTextureObject(const NvU32 *header, const NvU32 *sampler, LibreCUtexObject *pTexObject) {
    LIBRECUDA_ERR_PROPAGATE(ctxEnsureTexturePools(current_ctx));
    std::lock_guard ctx_lock(current_ctx->mutex);
    LIBRECUDA_VALIDATE(!current_ctx->free_texture_headers.empty(), LIBRECUDA_ERROR_OUT_OF_MEMORY);
    const NvU32 index = current_ctx->free_texture_headers.back();
    current_ctx->free_texture_headers.pop_back();
    memcpy(poolEntry(current_ctx->texture_header_pool_va, index), header, TEXTURE_POOL_ENTRY_SIZE);
    memcpy(poolEntry(current_ctx->texture_sampler_pool_va, index), sampler, TEXTURE_POOL_ENTRY_SIZE);
    *pTexObject = textureHandle(index);
    LIBRECUDA_SUCCEED();
}

/**
 * TEXSAMP (clcb97tex.h): address modes, point or linear filtering without mipmaps, border color
 */
static void buildSampler(NvU32 *sampler, const LibreCuTexSampler *description) {
    LibreCuTexSampler defaults{};
    const LibreCuTexSampler &d = description != nullptr ? *description : defaults;
    for (NvU32 axis = 0; axis < 3; axis++) {
        NvU32 mode{};
        switch (d.addressMode[axis]) {
            case LIBRECUDA_TEX_ADDRESS_WRAP: mode = 0; break;
            case LIBRECUDA_TEX_ADDRESS_MIRROR: mode = 1; break;
            case LIBRECUDA_TEX_ADDRESS_CLAMP: mode = 2; break; // clamp to edge
            case LIBRECUDA_TEX_ADDRESS_BORDER: mode = 3; break;
        }
        setBits(sampler, 2 + 3 * axis, 3 * axis, mode);
    }
    const NvU32 filter = d.linearFilter ? 2 : 1;
    setBits(sampler, 32 + 2, 32 + 0, filter);  // mag filter
    setBits(sampler, 32 + 5, 32 + 4, filter);  // min filter
    setBits(sampler, 32 + 7, 32 + 6, 1);       // no mipmaps
    for (NvU32 c = 0; c < 4; c++) {
        NvU32 bits;
        memcpy(&bits, &d.borderColor[c], sizeof(bits));
        sampler[4 + c] = bits;
    }
}

/**
 * TEXHEAD_V2_PITCH: a two dimensional texture in pitched linear memory
 */
static void buildHeaderV2Pitch(NvU32 *header, NvU64 address, NvU32 width, NvU32 height, NvU32 pitch,
                               LibreCuTexFormat format, NvU32 channels, NvU32 components, bool normalized_coords) {
    setBits(header, 31, 5, NvU32(address >> 5));
    setBits(header, 56, 32, NvU32(address >> 32));
    setBits(header, 80, 64, pitch >> 5);
    for (NvU32 c = 0; c < 4; c++) {
        setBits(header, 98 + 3 * c, 96 + 3 * c, channelSource(c, channels, format));
    }
    setBits(header, 111, 108, v2DataType(format));
    setBits(header, 118, 112, components);
    setBits(header, 127, 124, TEXHEAD_V2_HEADER_VERSION_PITCH);
    setBits(header, 144, 128, width - 1);
    setBits(header, 145, 145, normalized_coords ? 1 : 0);
    setBits(header, 153, 150, TEXHEAD_TEXTURE_TYPE_TWO_D_NO_MIPMAP);
    setBits(header, 156, 156, TEXHEAD_V2_BORDER_SOURCE_BORDER_COLOR);
    setBits(header, 176, 160, height - 1);
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

    NvU32 header[8]{}, sampler[8]{};
    if (current_ctx->device->compute_class >= BLACKWELL_COMPUTE_A) {
        buildHeaderV2OneDRaw(header, address, width, format, channels, components);
    } else {
        buildHeaderOneDBuffer(header, address, width, format, channels, components);
    }
    // tex1Dfetch (TLD) doesn't sample, but every object has a valid sampler
    buildSampler(sampler, nullptr);
    return createTextureObject(header, sampler, pTexObject);
}

libreCudaStatus_t libreCuTexObjectCreatePitch2D(LibreCUtexObject *pTexObject, void *devPtr, uint32_t width,
                                                uint32_t height, size_t pitchBytes, LibreCuTexFormat format,
                                                uint32_t channels, uint32_t bitsPerChannel,
                                                const LibreCuTexSampler *sampler) {
    LIBRECUDA_VALIDATE(pTexObject != nullptr && devPtr != nullptr, LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_ENSURE_CTX_VALID();
    // pitched textures are only implemented with the V2 headers of Blackwell
    LIBRECUDA_VALIDATE(current_ctx->device->compute_class >= BLACKWELL_COMPUTE_A, LIBRECUDA_ERROR_NOT_SUPPORTED);
    const NvU32 components = componentSizes(channels, bitsPerChannel);
    LIBRECUDA_VALIDATE(components != 0, LIBRECUDA_ERROR_NOT_SUPPORTED);
    const bool normalized = format == LIBRECUDA_TEX_FORMAT_UNORM || format == LIBRECUDA_TEX_FORMAT_SNORM;
    LIBRECUDA_VALIDATE(!normalized || bitsPerChannel <= 16, LIBRECUDA_ERROR_INVALID_VALUE);
    const NvU64 element_size = NvU64(channels) * bitsPerChannel / 8;
    const NvU64 address = reinterpret_cast<NvU64>(devPtr);
    LIBRECUDA_VALIDATE(width >= 1 && height >= 1 && width <= (1u << 17) && height <= (1u << 17),
                       LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(pitchBytes % LIBRECUDA_TEXTURE_PITCH_ALIGNMENT == 0 && pitchBytes >= width * element_size &&
                       pitchBytes < (size_t(1) << 22), LIBRECUDA_ERROR_INVALID_VALUE);
    LIBRECUDA_VALIDATE(address % LIBRECUDA_TEXTURE_PITCH_ALIGNMENT == 0 && address < (NvU64(1) << 57),
                       LIBRECUDA_ERROR_INVALID_VALUE);
    NvU32 header[8]{}, sampler_words[8]{};
    buildHeaderV2Pitch(header, address, width, height, static_cast<NvU32>(pitchBytes), format, channels, components,
                       sampler != nullptr && sampler->normalizedCoords);
    buildSampler(sampler_words, sampler);
    return createTextureObject(header, sampler_words, pTexObject);
}

libreCudaStatus_t libreCuTexObjectDestroy(LibreCUtexObject texObject) {
    LIBRECUDA_ENSURE_CTX_VALID();
    const NvU32 index = static_cast<NvU32>(texObject);
    LIBRECUDA_VALIDATE(index > 0 && index < TEXTURE_POOL_ENTRIES && texObject == textureHandle(index) &&
                       current_ctx->texture_header_pool_va != 0, LIBRECUDA_ERROR_INVALID_VALUE);
    std::lock_guard ctx_lock(current_ctx->mutex);
    // kernels still using the texture must complete first (like CUDA, the caller ensures that)
    memset(poolEntry(current_ctx->texture_header_pool_va, index), 0, TEXTURE_POOL_ENTRY_SIZE);
    memset(poolEntry(current_ctx->texture_sampler_pool_va, index), 0, TEXTURE_POOL_ENTRY_SIZE);
    current_ctx->free_texture_headers.push_back(index);
    LIBRECUDA_SUCCEED();
}
