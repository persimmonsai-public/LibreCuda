/**
 * LibreCUDA's cuSPARSE (libcusparse.so.12): handles and matrix descriptors only, so that libraries which create a
 * cuSPARSE handle without using sparse operations (PyTorch's batched cholesky_solve does) work. Sparse operations
 * return CUSPARSE_STATUS_NOT_SUPPORTED (stubs).
 */
#include <cusparse.h>

#include <cstdint>

struct cusparseContext {
    uint32_t magic;
    cudaStream_t stream;
    cusparsePointerMode_t pointerMode;
};

struct cusparseMatDescr {
    cusparseMatrixType_t type;
    cusparseFillMode_t fillMode;
    cusparseDiagType_t diagType;
    cusparseIndexBase_t indexBase;
};

namespace {

constexpr uint32_t SPARSE_MAGIC = 0x4c435350;

bool valid(cusparseHandle_t handle) {
    return handle != nullptr && handle->magic == SPARSE_MAGIC;
}

} // namespace

cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
    if (handle == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *handle = new cusparseContext{SPARSE_MAGIC, nullptr, CUSPARSE_POINTER_MODE_HOST};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    if (!valid(handle)) {
        return CUSPARSE_STATUS_NOT_INITIALIZED;
    }
    handle->magic = 0;
    delete handle;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int *version) {
    if (!valid(handle)) {
        return CUSPARSE_STATUS_NOT_INITIALIZED;
    }
    if (version == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *version = CUSPARSE_VERSION;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int *value) {
    if (value == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    switch (type) {
        case MAJOR_VERSION: *value = CUSPARSE_VER_MAJOR; return CUSPARSE_STATUS_SUCCESS;
        case MINOR_VERSION: *value = CUSPARSE_VER_MINOR; return CUSPARSE_STATUS_SUCCESS;
        case PATCH_LEVEL: *value = CUSPARSE_VER_PATCH; return CUSPARSE_STATUS_SUCCESS;
        default: return CUSPARSE_STATUS_INVALID_VALUE;
    }
}

const char *cusparseGetErrorName(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
        case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        default: return "CUSPARSE_STATUS_UNKNOWN";
    }
}

const char *cusparseGetErrorString(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "success";
        case CUSPARSE_STATUS_NOT_SUPPORTED: return "operation not supported (LibreCUDA's cuSPARSE has no sparse operations)";
        default: return cusparseGetErrorName(status);
    }
}

cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
    if (!valid(handle)) {
        return CUSPARSE_STATUS_NOT_INITIALIZED;
    }
    handle->stream = streamId;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId) {
    if (!valid(handle)) {
        return CUSPARSE_STATUS_NOT_INITIALIZED;
    }
    if (streamId == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *streamId = handle->stream;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode) {
    if (!valid(handle)) {
        return CUSPARSE_STATUS_NOT_INITIALIZED;
    }
    if (mode == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *mode = handle->pointerMode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
    if (!valid(handle)) {
        return CUSPARSE_STATUS_NOT_INITIALIZED;
    }
    handle->pointerMode = mode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA) {
    if (descrA == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *descrA = new cusparseMatDescr{CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT,
                                   CUSPARSE_INDEX_BASE_ZERO};
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    delete descrA;
    return CUSPARSE_STATUS_SUCCESS;
}

#define DESCRIPTOR_FIELD(Name, Type, field)                                                                            \
cusparseStatus_t cusparseSet##Name(cusparseMatDescr_t descrA, Type value) {                                            \
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                                                       \
    descrA->field = value;                                                                                             \
    return CUSPARSE_STATUS_SUCCESS;                                                                                    \
}                                                                                                                      \
Type cusparseGet##Name(const cusparseMatDescr_t descrA) {                                                              \
    return descrA->field;                                                                                              \
}

DESCRIPTOR_FIELD(MatType, cusparseMatrixType_t, type)
DESCRIPTOR_FIELD(MatFillMode, cusparseFillMode_t, fillMode)
DESCRIPTOR_FIELD(MatDiagType, cusparseDiagType_t, diagType)
DESCRIPTOR_FIELD(MatIndexBase, cusparseIndexBase_t, indexBase)
