/**
 * LibreCUDA's cuBLASLt (libcublasLt.so.13): matrix multiplication with descriptors on the GEMM engine (gemm.cu).
 *
 * Descriptors live in the opaque storage the header defines (so the *Init functions work with caller allocated
 * descriptors). cublasLtMatmulAlgoGetHeuristic offers one algorithm, which needs no workspace; the engine picks its
 * kernel per call. Supported: real and complex types the engine supports, column and row major layouts, strided
 * batches, host and device scalars, and the epilogues bias, ReLU, GELU and their combinations. 8 bit floats (e4m3 and
 * e5m2 A and B) with per-tensor scales of A, B, C and D and the absolute maximum of D. Not supported: int8, other
 * scaling modes (vectors, blocks), auxiliary outputs and gradient epilogues.
 */
#include "blas_internal.h"

#include <cstdio>
#include <cstring>
#include <new>

using namespace librecuda_blas;

struct cublasLtContext {
    uint32_t magic;
};

namespace {

constexpr uint32_t DESC_MAGIC = 0x4c43444d;   // "MDCL"
constexpr uint32_t LAYOUT_MAGIC = 0x4c43414c;
constexpr uint32_t PREF_MAGIC = 0x4c435250;
constexpr uint32_t HANDLE_MAGIC = 0x4c43484c;
// marks the algorithm cublasLtMatmulAlgoGetHeuristic returns
constexpr uint64_t ALGO_MAGIC = 0x4c696272654c4743ull;

struct MatmulDesc {
    uint32_t magic;
    int32_t compute;
    int32_t scaleType;
    int32_t pointerMode;
    int32_t transa, transb, transc;
    uint32_t epilogue;
    const void *bias;
    int32_t biasType;
    int64_t biasBatchStride;
    // 8 bit float matmuls: the scales (device floats) and their modes, the absolute maximum of D
    const float *scaleA, *scaleB, *scaleC, *scaleD;
    float *amax;
    int32_t scaleModes[4];
    // features that aren't supported when set (auxiliary outputs, D's block scales)
    const void *unsupportedPointers[4];
};
static_assert(sizeof(MatmulDesc) <= sizeof(cublasLtMatmulDescOpaque_t));

struct Layout {
    uint32_t magic;
    int32_t type;
    int32_t order;
    int32_t batchCount;
    uint64_t rows, cols;
    int64_t ld;
    int64_t stride;
    int32_t batchMode;
};
static_assert(sizeof(Layout) <= sizeof(cublasLtMatrixLayoutOpaque_t));

struct Preference {
    uint32_t magic;
    uint32_t searchMode;
    uint64_t maxWorkspace;
};
static_assert(sizeof(Preference) <= sizeof(cublasLtMatmulPreferenceOpaque_t));

bool isFp8(int32_t type) {
    return type == CUDA_R_8F_E4M3 || type == CUDA_R_8F_E5M2;
}

MatmulDesc *desc(cublasLtMatmulDesc_t d) {
    auto *m = reinterpret_cast<MatmulDesc *>(d);
    return m != nullptr && m->magic == DESC_MAGIC ? m : nullptr;
}

Layout *layout(cublasLtMatrixLayout_t l) {
    auto *m = reinterpret_cast<Layout *>(l);
    return m != nullptr && m->magic == LAYOUT_MAGIC ? m : nullptr;
}

Preference *preference(cublasLtMatmulPreference_t p) {
    auto *m = reinterpret_cast<Preference *>(p);
    return m != nullptr && m->magic == PREF_MAGIC ? m : nullptr;
}

// copies an attribute value of exactly the field's size
template<typename T>
cublasStatus_t setField(T &field, const void *buf, size_t size) {
    if (buf == nullptr || size != sizeof(T)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    memcpy(&field, buf, sizeof(T));
    return CUBLAS_STATUS_SUCCESS;
}

template<typename T>
cublasStatus_t getField(const T &field, void *buf, size_t size, size_t *written) {
    if (written != nullptr) {
        *written = sizeof(T);
    }
    // a size query
    if (size == 0 && buf == nullptr && written != nullptr) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (buf == nullptr || size < sizeof(T)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    memcpy(buf, &field, sizeof(T));
    return CUBLAS_STATUS_SUCCESS;
}

void initDesc(MatmulDesc *m, cublasComputeType_t compute, cudaDataType scaleType) {
    *m = MatmulDesc{};
    m->magic = DESC_MAGIC;
    m->compute = compute;
    m->scaleType = scaleType;
    m->pointerMode = CUBLASLT_POINTER_MODE_HOST;
    m->transa = m->transb = m->transc = CUBLAS_OP_N;
    m->epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    m->biasType = -1;
}

void initLayout(Layout *l, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
    *l = Layout{};
    l->magic = LAYOUT_MAGIC;
    l->type = type;
    l->order = CUBLASLT_ORDER_COL;
    l->batchCount = 1;
    l->rows = rows;
    l->cols = cols;
    l->ld = ld;
}

// the operand of a layout as a column-major matrix (a row-major matrix is its transpose stored column-major)
struct Matrix {
    uint64_t rows, cols;
    bool transposed;
};

Matrix columnMajor(const Layout *l) {
    if (l->order == CUBLASLT_ORDER_ROW) {
        return {l->cols, l->rows, true};
    }
    return {l->rows, l->cols, false};
}

cublasOperation_t flip(cublasOperation_t op, bool transposed) {
    if (!transposed) {
        return op;
    }
    return op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
}

/**
 * The engine's parameters for a matmul (pointers may be null when only checking whether it's supported)
 */
cublasStatus_t buildParams(const MatmulDesc *m, const Layout *la, const Layout *lb, const Layout *lc, const Layout *ld,
                           const void *alpha, const void *a, const void *b, const void *beta, const void *c, void *d,
                           cudaStream_t stream, GemmParams &p) {
    if (m == nullptr || la == nullptr || lb == nullptr || lc == nullptr || ld == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (const void *pointer: m->unsupportedPointers) {
        if (pointer != nullptr) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    // only per-tensor scales (CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
    for (int32_t mode: m->scaleModes) {
        if (mode != CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    for (const Layout *l: {la, lb, lc, ld}) {
        if (l->order != CUBLASLT_ORDER_COL && l->order != CUBLASLT_ORDER_ROW) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        if (l->batchMode != 0) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    const bool fp8 = isFp8(la->type) && isFp8(lb->type);
    // (8 bit floats: C may have another type when it isn't read, beta 0, which the engine checks)
    if (m->transc != CUBLAS_OP_N || lc->order != ld->order || (lc->type != ld->type && !fp8)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (m->pointerMode != CUBLASLT_POINTER_MODE_HOST && m->pointerMode != CUBLASLT_POINTER_MODE_DEVICE) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    Activation activation = Activation::NONE;
    bool bias = false;
    switch (m->epilogue) {
        case CUBLASLT_EPILOGUE_DEFAULT:
            break;
        case CUBLASLT_EPILOGUE_RELU:
            activation = Activation::RELU;
            break;
        case CUBLASLT_EPILOGUE_GELU:
            activation = Activation::GELU;
            break;
        case CUBLASLT_EPILOGUE_BIAS:
            bias = true;
            break;
        case CUBLASLT_EPILOGUE_RELU_BIAS:
            bias = true;
            activation = Activation::RELU;
            break;
        case CUBLASLT_EPILOGUE_GELU_BIAS:
            bias = true;
            activation = Activation::GELU;
            break;
        default:
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    // a row-major D: compute its transpose, D^T = op(B)^T op(A)^T (the bias, along D's rows, can't follow)
    const bool row_major = ld->order == CUBLASLT_ORDER_ROW;
    if (row_major && bias) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const Matrix ma = columnMajor(la), mb = columnMajor(lb), md = columnMajor(ld);
    auto ta = static_cast<cublasOperation_t>(m->transa), tb = static_cast<cublasOperation_t>(m->transb);
    // conjugation without transposition can't be expressed for a transposed (row-major) operand
    if ((ma.transposed && ta == CUBLAS_OP_C) || (mb.transposed && tb == CUBLAS_OP_C)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    ta = flip(ta, ma.transposed);
    tb = flip(tb, mb.transposed);
    p = GemmParams{};
    const uint64_t m_rows = ta == CUBLAS_OP_N ? ma.rows : ma.cols, k_a = ta == CUBLAS_OP_N ? ma.cols : ma.rows;
    const uint64_t k_b = tb == CUBLAS_OP_N ? mb.rows : mb.cols, n_cols = tb == CUBLAS_OP_N ? mb.cols : mb.rows;
    // in D's terms (row-major: the problem of the transpose)
    const uint64_t d_rows = row_major ? md.cols : md.rows, d_cols = row_major ? md.rows : md.cols;
    if (k_a != k_b || m_rows != d_rows || n_cols != d_cols) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    Operand oa{a, nullptr, static_cast<cudaDataType>(la->type), la->ld, la->stride, 1};
    Operand ob{b, nullptr, static_cast<cudaDataType>(lb->type), lb->ld, lb->stride, 1};
    if (row_major) {
        // D^T (n x m) = op'(B) op'(A) with the operations flipped
        p.m = static_cast<int>(n_cols);
        p.n = static_cast<int>(m_rows);
        p.transa = flip(tb, true);
        p.transb = flip(ta, true);
        if (tb == CUBLAS_OP_C || ta == CUBLAS_OP_C) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        p.a = ob;
        p.b = oa;
    } else {
        p.m = static_cast<int>(m_rows);
        p.n = static_cast<int>(n_cols);
        p.transa = ta;
        p.transb = tb;
        p.a = oa;
        p.b = ob;
    }
    p.k = static_cast<int>(k_a);
    p.batch = ld->batchCount;
    p.c = Operand{c, nullptr, static_cast<cudaDataType>(lc->type), lc->ld, lc->stride, 1};
    p.d = Operand{d, nullptr, static_cast<cudaDataType>(ld->type), ld->ld, ld->stride, 1};
    // batch counts must match (an operand with one entry and stride 0 is shared by all)
    for (const Layout *l: {la, lb, lc}) {
        if (l->batchCount != ld->batchCount && !(l->batchCount == 1 && l->stride == 0)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
    }
    p.compute = static_cast<cublasComputeType_t>(m->compute);
    const auto scale_type = static_cast<cudaDataType>(m->scaleType);
    const bool device_scalars = m->pointerMode == CUBLASLT_POINTER_MODE_DEVICE;
    p.alpha = Scalar{alpha, scale_type, device_scalars};
    p.beta = Scalar{beta, scale_type, device_scalars};
    p.allowTF32 = m->compute == CUBLAS_COMPUTE_32F_FAST_TF32 || m->compute == CUBLAS_COMPUTE_32F_FAST_16F ||
                  m->compute == CUBLAS_COMPUTE_32F_FAST_16BF;
    p.activation = activation;
    if (bias) {
        p.bias = m->bias;
        p.biasType = m->biasType >= 0 ? static_cast<cudaDataType>(m->biasType)
                                      : (p.d.type == CUDA_R_16F || p.d.type == CUDA_R_16BF ? p.d.type : CUDA_R_32F);
        p.biasStride = m->biasBatchStride;
        if (p.biasType != CUDA_R_16F && p.biasType != CUDA_R_16BF && p.biasType != CUDA_R_32F) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    if ((activation != Activation::NONE || bias) && p.d.type != CUDA_R_16F && p.d.type != CUDA_R_16BF &&
        p.d.type != CUDA_R_32F) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    const bool scaled = m->scaleA != nullptr || m->scaleB != nullptr || m->scaleC != nullptr ||
                        m->scaleD != nullptr || m->amax != nullptr;
    if (fp8) {
        // e4m3 / e5m2 A and B, any of them with 16 bit floats, float or 8 bit floats as D; float compute
        if (p.d.type != CUDA_R_16F && p.d.type != CUDA_R_16BF && p.d.type != CUDA_R_32F && !isFp8(p.d.type)) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        if (m->compute != CUBLAS_COMPUTE_32F || scale_type != CUDA_R_32F) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        // (as NVIDIA's cuBLASLt: only 8 bit D has a scale and an absolute maximum)
        if ((m->scaleD != nullptr || m->amax != nullptr) && !isFp8(p.d.type)) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        // the kernels fuse a bias of D's type; other epilogues run as a separate pass after them, which can't come
        // before D's scale and amax (8 bit D)
        const bool separate = activation != Activation::NONE || (bias && p.biasType != p.d.type);
        if (separate && isFp8(p.d.type)) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        p.scaleA = row_major ? m->scaleB : m->scaleA;
        p.scaleB = row_major ? m->scaleA : m->scaleB;
        p.scaleC = m->scaleC;
        p.scaleD = m->scaleD;
        p.amax = m->amax;
        p.stream = stream;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (scaled) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    // the engine's operand types: A and B alike, D of their type or float (16 bit operands)
    if (p.a.type != p.b.type) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    switch (p.a.type) {
        case CUDA_R_16F:
        case CUDA_R_16BF:
            if (p.d.type != p.a.type && p.d.type != CUDA_R_32F) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            break;
        case CUDA_R_32F:
        case CUDA_R_64F:
        case CUDA_C_32F:
        case CUDA_C_64F:
            if (p.d.type != p.a.type) {
                return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            break;
        default:
            return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    p.stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// library

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) {
    if (lightHandle == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *lightHandle = new cublasLtContext{HANDLE_MAGIC};
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    if (lightHandle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    // (cuBLAS handles are passed as cuBLASLt handles too: only free our own)
    if (lightHandle->magic == HANDLE_MAGIC) {
        lightHandle->magic = 0;
        delete lightHandle;
    }
    return CUBLAS_STATUS_SUCCESS;
}

size_t cublasLtGetVersion(void) {
    return CUBLAS_VERSION;
}

size_t cublasLtGetCudartVersion(void) {
    return CUDART_VERSION;
}

cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int *value) {
    if (value == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (type) {
        case MAJOR_VERSION:
            *value = CUBLAS_VER_MAJOR;
            return CUBLAS_STATUS_SUCCESS;
        case MINOR_VERSION:
            *value = CUBLAS_VER_MINOR;
            return CUBLAS_STATUS_SUCCESS;
        case PATCH_LEVEL:
            *value = CUBLAS_VER_PATCH;
            return CUBLAS_STATUS_SUCCESS;
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
}

const char *cublasLtGetStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

const char *cublasLtGetStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "the library was not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "the resource allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "an invalid numeric value was used as an argument";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "an absent device architectural feature is required";
        case CUBLAS_STATUS_MAPPING_ERROR: return "an access to GPU memory space failed";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "the GPU program failed to execute";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "an internal operation failed";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "the functionality requested is not supported";
        case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
        default: return "unknown error";
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// matmul descriptors

cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t matmulDesc, size_t size,
                                               cublasComputeType_t computeType, cudaDataType_t scaleType) {
    if (matmulDesc == nullptr || size < sizeof(MatmulDesc)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    initDesc(reinterpret_cast<MatmulDesc *>(matmulDesc), computeType, scaleType);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, cublasComputeType_t computeType,
                                        cudaDataType_t scaleType) {
    if (matmulDesc == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    auto *storage = new(std::nothrow) cublasLtMatmulDescOpaque_t{};
    if (storage == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    initDesc(reinterpret_cast<MatmulDesc *>(storage), computeType, scaleType);
    *matmulDesc = storage;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    MatmulDesc *m = desc(matmulDesc);
    if (m == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    m->magic = 0;
    delete matmulDesc;
    return CUBLAS_STATUS_SUCCESS;
}

namespace {

// the attributes of unsupported features that hold pointers, by their slot in unsupportedPointers
int unsupportedSlot(cublasLtMatmulDescAttributes_t attr) {
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER: return 0;
        case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER: return 1;
        case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER: return 2;
        case CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER: return 3;
        default: return -1;
    }
}

// the scale modes of A, B, C and D, by their index in scaleModes
int scaleModeSlot(cublasLtMatmulDescAttributes_t attr) {
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_A_SCALE_MODE: return 0;
        case CUBLASLT_MATMUL_DESC_B_SCALE_MODE: return 1;
        case CUBLASLT_MATMUL_DESC_C_SCALE_MODE: return 2;
        case CUBLASLT_MATMUL_DESC_D_SCALE_MODE: return 3;
        default: return -1;
    }
}

} // namespace

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                                              const void *buf, size_t sizeInBytes) {
    MatmulDesc *m = desc(matmulDesc);
    if (m == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: return setField(m->compute, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_SCALE_TYPE: return setField(m->scaleType, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_POINTER_MODE: return setField(m->pointerMode, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_TRANSA: return setField(m->transa, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_TRANSB: return setField(m->transb, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_TRANSC: return setField(m->transc, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_EPILOGUE: return setField(m->epilogue, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER: return setField(m->bias, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: return setField(m->biasType, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: return setField(m->biasBatchStride, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: return setField(m->scaleA, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: return setField(m->scaleB, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_C_SCALE_POINTER: return setField(m->scaleC, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_D_SCALE_POINTER: return setField(m->scaleD, buf, sizeInBytes);
        case CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: return setField(m->amax, buf, sizeInBytes);
        default: {
            const int slot = unsupportedSlot(attr);
            if (slot >= 0) {
                return setField(m->unsupportedPointers[slot], buf, sizeInBytes);
            }
            const int mode = scaleModeSlot(attr);
            if (mode >= 0) {
                return setField(m->scaleModes[mode], buf, sizeInBytes);
            }
            // tuning hints (SM count target, fast accumulation, ...) don't change the result
            return buf != nullptr ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
        }
    }
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                                              void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    MatmulDesc *m = desc(matmulDesc);
    if (m == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: return getField(m->compute, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_SCALE_TYPE: return getField(m->scaleType, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_POINTER_MODE: return getField(m->pointerMode, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_TRANSA: return getField(m->transa, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_TRANSB: return getField(m->transb, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_TRANSC: return getField(m->transc, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_EPILOGUE: return getField(m->epilogue, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER: return getField(m->bias, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: return getField(m->biasType, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE:
            return getField(m->biasBatchStride, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: return getField(m->scaleA, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: return getField(m->scaleB, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_C_SCALE_POINTER: return getField(m->scaleC, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_D_SCALE_POINTER: return getField(m->scaleD, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: return getField(m->amax, buf, sizeInBytes, sizeWritten);
        default: {
            const int slot = unsupportedSlot(attr);
            if (slot >= 0) {
                return getField(m->unsupportedPointers[slot], buf, sizeInBytes, sizeWritten);
            }
            const int mode = scaleModeSlot(attr);
            if (mode >= 0) {
                return getField(m->scaleModes[mode], buf, sizeInBytes, sizeWritten);
            }
            return CUBLAS_STATUS_INVALID_VALUE;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// matrix layouts

cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t matLayout, size_t size, cudaDataType type,
                                                 uint64_t rows, uint64_t cols, int64_t ld) {
    if (matLayout == nullptr || size < sizeof(Layout)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    initLayout(reinterpret_cast<Layout *>(matLayout), type, rows, cols, ld);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
    if (matLayout == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    auto *storage = new(std::nothrow) cublasLtMatrixLayoutOpaque_t{};
    if (storage == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    initLayout(reinterpret_cast<Layout *>(storage), type, rows, cols, ld);
    *matLayout = storage;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(const cublasLtMatrixLayout_t matLayout) {
    Layout *l = layout(matLayout);
    if (l == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    l->magic = 0;
    delete matLayout;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
                                                const void *buf, size_t sizeInBytes) {
    Layout *l = layout(matLayout);
    if (l == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
        case CUBLASLT_MATRIX_LAYOUT_TYPE: return setField(l->type, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_ORDER: return setField(l->order, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_ROWS: return setField(l->rows, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_COLS: return setField(l->cols, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_LD: return setField(l->ld, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: return setField(l->batchCount, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: return setField(l->stride, buf, sizeInBytes);
        case CUBLASLT_MATRIX_LAYOUT_BATCH_MODE: return setField(l->batchMode, buf, sizeInBytes);
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr,
                                                void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    Layout *l = layout(matLayout);
    if (l == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
        case CUBLASLT_MATRIX_LAYOUT_TYPE: return getField(l->type, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_ORDER: return getField(l->order, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_ROWS: return getField(l->rows, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_COLS: return getField(l->cols, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_LD: return getField(l->ld, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: return getField(l->batchCount, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            return getField(l->stride, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATRIX_LAYOUT_BATCH_MODE: return getField(l->batchMode, buf, sizeInBytes, sizeWritten);
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// preferences and heuristics

cublasStatus_t cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t pref, size_t size) {
    if (pref == nullptr || size < sizeof(Preference)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *reinterpret_cast<Preference *>(pref) = Preference{PREF_MAGIC, 0, 0};
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    if (pref == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    auto *storage = new(std::nothrow) cublasLtMatmulPreferenceOpaque_t{};
    if (storage == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    *reinterpret_cast<Preference *>(storage) = Preference{PREF_MAGIC, 0, 0};
    *pref = storage;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    Preference *p = preference(pref);
    if (p == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    p->magic = 0;
    delete pref;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, const void *buf,
                                                    size_t sizeInBytes) {
    Preference *p = preference(pref);
    if (p == nullptr || buf == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
        case CUBLASLT_MATMUL_PREF_SEARCH_MODE: return setField(p->searchMode, buf, sizeInBytes);
        case CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: return setField(p->maxWorkspace, buf, sizeInBytes);
        default:
            // alignment hints and limits: the engine checks alignment itself
            return CUBLAS_STATUS_SUCCESS;
    }
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr, void *buf,
                                                    size_t sizeInBytes, size_t *sizeWritten) {
    Preference *p = preference(pref);
    if (p == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
        case CUBLASLT_MATMUL_PREF_SEARCH_MODE: return getField(p->searchMode, buf, sizeInBytes, sizeWritten);
        case CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: return getField(p->maxWorkspace, buf, sizeInBytes, sizeWritten);
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference_, int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                              int *returnAlgoCount) {
    if (lightHandle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (requestedAlgoCount <= 0 || heuristicResultsArray == nullptr || returnAlgoCount == nullptr ||
        preference(preference_) == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    GemmParams p{};
    const cublasStatus_t status = buildParams(desc(operationDesc), layout(Adesc), layout(Bdesc), layout(Cdesc),
                                              layout(Ddesc), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                              nullptr, p);
    *returnAlgoCount = 0;
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    cublasLtMatmulHeuristicResult_t &result = heuristicResultsArray[0];
    result = cublasLtMatmulHeuristicResult_t{};
    result.algo.data[0] = ALGO_MAGIC;
    result.workspaceSize = 0;
    result.state = CUBLAS_STATUS_SUCCESS;
    result.wavesCount = 1.0f;
    *returnAlgoCount = 1;
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// matmul

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc,
                              const void *beta, const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
                              cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *, void *, size_t,
                              cudaStream_t stream) {
    if (lightHandle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (alpha == nullptr || beta == nullptr || D == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    GemmParams p{};
    const cublasStatus_t status = buildParams(desc(computeDesc), layout(Adesc), layout(Bdesc), layout(Cdesc),
                                              layout(Ddesc), alpha, A, B, beta, C, D, stream, p);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return librecuda_blas_gemm(&p);
}
