/**
 * LibreCUDA's cuDNN, the legacy API: handles, versions, errors, descriptors (tensor, filter, convolution, activation,
 * pooling, dropout, spatial transformer) and the operations on them (convolutions, batch normalization, pooling, the
 * spatial transformer). Algorithm queries offer one algorithm per convolution direction: im2col and GEMMs
 * (cudnn_conv.cu). Functions of the public headers not implemented here return CUDNN_STATUS_NOT_SUPPORTED (stubs).
 */
#include "dnn_internal.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

using namespace librecuda_dnn;

struct cudnnFilterStruct {
    Tensor tensor;
    cudnnTensorFormat_t format;
};

struct cudnnConvolutionStruct {
    int arrayLength;
    int pad[3], stride[3], dilation[3];
    cudnnConvolutionMode_t mode;
    cudnnDataType_t compute;
    int groups;
    cudnnMathType_t math;
};

struct cudnnActivationStruct {
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t nan;
    double coef;
};

struct cudnnPoolingStruct {
    Pooling pooling;
};

struct cudnnSpatialTransformerStruct {
    cudnnDataType_t type;
    int nbDims;
    int dims[8];
};

namespace {

thread_local std::string t_lastError;

cudnnStatus_t fail(cudnnStatus_t status, const char *message) {
    t_lastError = message;
    return status;
}

#define CHECK_DNN(handle) do { if (!valid(handle)) { return CUDNN_STATUS_BAD_PARAM; } } while (0)

// packed strides of a filter in its format (NCHW: K, C, spatial; NHWC: channels last)
void filterStrides(Tensor &t, cudnnTensorFormat_t format) {
    const int n = t.nbDims;
    if (format == CUDNN_TENSOR_NHWC && n >= 3) {
        int64_t stride = 1;
        t.strides[1] = stride;
        stride *= t.dims[1];
        for (int d = n - 1; d >= 2; d--) {
            t.strides[d] = stride;
            stride *= t.dims[d];
        }
        t.strides[0] = stride;
    } else {
        int64_t stride = 1;
        for (int d = n - 1; d >= 0; d--) {
            t.strides[d] = stride;
            stride *= t.dims[d];
        }
    }
}

ConvGeometry geometryOf(const cudnnConvolutionStruct *c) {
    ConvGeometry g{};
    g.spatial = c->arrayLength;
    for (int d = 0; d < c->arrayLength; d++) {
        g.padBefore[d] = g.padAfter[d] = c->pad[d];
        g.stride[d] = c->stride[d];
        g.dilation[d] = c->dilation[d];
    }
    g.flip = c->mode == CUDNN_CONVOLUTION;
    g.compute = c->compute;
    // (CUDNN_DEFAULT_MATH permits TF32 for float data on GPUs with TF32 tensor cores)
    g.allowTF32 = c->math != CUDNN_FMA_MATH && tf32Permitted();
    return g;
}

bool descriptorsValid(const void *a, const void *b, const void *c, const void *d) {
    return a != nullptr && b != nullptr && c != nullptr && d != nullptr;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// library

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    if (handle == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    // (like cuDNN: the handle belongs to the current device, which must work)
    if (cudaFree(nullptr) != cudaSuccess) {
        cudaGetLastError();
        return CUDNN_STATUS_NOT_INITIALIZED;
    }
    *handle = new cudnnContext{HANDLE_MAGIC, nullptr};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    CHECK_DNN(handle);
    handle->magic = 0;
    delete handle;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    CHECK_DNN(handle);
    handle->stream = streamId;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
    CHECK_DNN(handle);
    if (streamId == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    *streamId = handle->stream;
    return CUDNN_STATUS_SUCCESS;
}

size_t cudnnGetVersion(void) {
    return CUDNN_VERSION;
}

size_t cudnnGetCudartVersion(void) {
    return CUDART_VERSION;
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int *value) {
    if (value == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    switch (type) {
        case MAJOR_VERSION: *value = CUDNN_MAJOR; return CUDNN_STATUS_SUCCESS;
        case MINOR_VERSION: *value = CUDNN_MINOR; return CUDNN_STATUS_SUCCESS;
        case PATCH_LEVEL: *value = CUDNN_PATCHLEVEL; return CUDNN_STATUS_SUCCESS;
        default: return CUDNN_STATUS_BAD_PARAM;
    }
}

const char *cudnnGetErrorString(cudnnStatus_t status) {
    switch (status) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_BAD_PARAM_NULL_POINTER: return "CUDNN_STATUS_BAD_PARAM_NULL_POINTER";
        case CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED: return "CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED";
        case CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE: return "CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN: return "CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN";
        case CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE: return "CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
        default: return "CUDNN_STATUS_UNKNOWN";
    }
}

void cudnnGetLastErrorString(char *message, size_t max_size) {
    if (message != nullptr && max_size > 0) {
        snprintf(message, max_size, "%s", t_lastError.c_str());
        t_lastError.clear();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// tensor and filter descriptors

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    if (tensorDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    *tensorDesc = new cudnnTensorStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    delete tensorDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
                                         const int dimA[], const int strideA[]) {
    if (tensorDesc == nullptr || nbDims <= 0 || nbDims > MAX_DIMS || dimA == nullptr || strideA == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    Tensor &t = tensorDesc->tensor;
    t.nbDims = nbDims;
    t.type = dataType;
    for (int i = 0; i < nbDims; i++) {
        if (dimA[i] <= 0 || strideA[i] < 0) {
            return CUDNN_STATUS_BAD_PARAM;
        }
        t.dims[i] = dimA[i];
        t.strides[i] = strideA[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
                                         cudnnDataType_t dataType, int n, int c, int h, int w) {
    if (tensorDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    Tensor &t = tensorDesc->tensor;
    t = Tensor{4, {n, c, h, w}, {}, dataType};
    filterStrides(t, format);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
                                         cudnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]) {
    if (tensorDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const Tensor &t = tensorDesc->tensor;
    if (dataType != nullptr) *dataType = t.type;
    if (nbDims != nullptr) *nbDims = t.nbDims;
    for (int i = 0; i < nbDimsRequested && i < t.nbDims; i++) {
        if (dimA != nullptr) dimA[i] = static_cast<int>(t.dims[i]);
        if (strideA != nullptr) strideA[i] = static_cast<int>(t.strides[i]);
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c,
                                           int h, int w, int nStride, int cStride, int hStride, int wStride) {
    if (tensorDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    tensorDesc->tensor = Tensor{4, {n, c, h, w}, {nStride, cStride, hStride, wStride}, dataType};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType, int *n,
                                         int *c, int *h, int *w, int *nStride, int *cStride, int *hStride,
                                         int *wStride) {
    if (tensorDesc == nullptr || tensorDesc->tensor.nbDims != 4) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const Tensor &t = tensorDesc->tensor;
    if (dataType != nullptr) *dataType = t.type;
    int *dims[4] = {n, c, h, w}, *strides[4] = {nStride, cStride, hStride, wStride};
    for (int i = 0; i < 4; i++) {
        if (dims[i] != nullptr) *dims[i] = static_cast<int>(t.dims[i]);
        if (strides[i] != nullptr) *strides[i] = static_cast<int>(t.strides[i]);
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    if (filterDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    *filterDesc = new cudnnFilterStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    delete filterDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
    if (filterDesc == nullptr || nbDims <= 0 || nbDims > MAX_DIMS || filterDimA == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    Tensor &t = filterDesc->tensor;
    t.nbDims = nbDims;
    t.type = dataType;
    for (int i = 0; i < nbDims; i++) {
        t.dims[i] = filterDimA[i];
    }
    filterDesc->format = format;
    filterStrides(t, format);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
                                         cudnnDataType_t *dataType, cudnnTensorFormat_t *format, int *nbDims,
                                         int filterDimA[]) {
    if (filterDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const Tensor &t = filterDesc->tensor;
    if (dataType != nullptr) *dataType = t.type;
    if (format != nullptr) *format = filterDesc->format;
    if (nbDims != nullptr) *nbDims = t.nbDims;
    for (int i = 0; i < nbDimsRequested && i < t.nbDims && filterDimA != nullptr; i++) {
        filterDimA[i] = static_cast<int>(t.dims[i]);
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int k, int c, int h, int w) {
    const int dims[4] = {k, c, h, w};
    return cudnnSetFilterNdDescriptor(filterDesc, dataType, format, 4, dims);
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
                                         cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w) {
    if (filterDesc == nullptr || filterDesc->tensor.nbDims != 4) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    int dims[4];
    cudnnGetFilterNdDescriptor(filterDesc, 4, dataType, format, nullptr, dims);
    int *out[4] = {k, c, h, w};
    for (int i = 0; i < 4; i++) {
        if (out[i] != nullptr) *out[i] = dims[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// convolutions

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    if (convDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    *convDesc = new cudnnConvolutionStruct{};
    (*convDesc)->groups = 1;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    delete convDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength,
                                              const int padA[], const int filterStrideA[], const int dilationA[],
                                              cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
    if (convDesc == nullptr || arrayLength <= 0 || arrayLength > 3 || padA == nullptr || filterStrideA == nullptr ||
        dilationA == nullptr) {
        return fail(CUDNN_STATUS_BAD_PARAM, "convolutions have 1 to 3 spatial dimensions");
    }
    convDesc->arrayLength = arrayLength;
    for (int i = 0; i < arrayLength; i++) {
        convDesc->pad[i] = padA[i];
        convDesc->stride[i] = filterStrideA[i];
        convDesc->dilation[i] = dilationA[i];
    }
    convDesc->mode = mode;
    convDesc->compute = computeType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u,
                                              int v, int dilation_h, int dilation_w, cudnnConvolutionMode_t mode,
                                              cudnnDataType_t computeType) {
    const int pad[2] = {pad_h, pad_w}, stride[2] = {u, v}, dilation[2] = {dilation_h, dilation_w};
    return cudnnSetConvolutionNdDescriptor(convDesc, 2, pad, stride, dilation, mode, computeType);
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
                                              int *arrayLength, int padA[], int strideA[], int dilationA[],
                                              cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
    if (convDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (arrayLength != nullptr) *arrayLength = convDesc->arrayLength;
    for (int i = 0; i < arrayLengthRequested && i < convDesc->arrayLength; i++) {
        if (padA != nullptr) padA[i] = convDesc->pad[i];
        if (strideA != nullptr) strideA[i] = convDesc->stride[i];
        if (dilationA != nullptr) dilationA[i] = convDesc->dilation[i];
    }
    if (mode != nullptr) *mode = convDesc->mode;
    if (computeType != nullptr) *computeType = convDesc->compute;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w,
                                              int *u, int *v, int *dilation_h, int *dilation_w,
                                              cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
    if (convDesc == nullptr || convDesc->arrayLength != 2) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    int pad[2], stride[2], dilation[2];
    cudnnGetConvolutionNdDescriptor(convDesc, 2, nullptr, pad, stride, dilation, mode, computeType);
    int *out[6] = {pad_h, pad_w, u, v, dilation_h, dilation_w};
    const int values[6] = {pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1]};
    for (int i = 0; i < 6; i++) {
        if (out[i] != nullptr) *out[i] = values[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                    const cudnnTensorDescriptor_t inputTensorDesc,
                                                    const cudnnFilterDescriptor_t filterDesc, int nbDims,
                                                    int tensorOuputDimA[]) {
    if (convDesc == nullptr || inputTensorDesc == nullptr || filterDesc == nullptr || tensorOuputDimA == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const Tensor &x = inputTensorDesc->tensor, &w = filterDesc->tensor;
    if (nbDims != x.nbDims || w.nbDims != x.nbDims || convDesc->arrayLength != nbDims - 2 ||
        x.dims[1] != w.dims[1] * convDesc->groups) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    tensorOuputDimA[0] = static_cast<int>(x.dims[0]);
    tensorOuputDimA[1] = static_cast<int>(w.dims[0]);
    for (int d = 0; d < nbDims - 2; d++) {
        tensorOuputDimA[2 + d] = static_cast<int>(convOutputSize(x.dims[2 + d], w.dims[2 + d], convDesc->pad[d],
                                                                 convDesc->pad[d], convDesc->stride[d],
                                                                 convDesc->dilation[d]));
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                    const cudnnTensorDescriptor_t inputTensorDesc,
                                                    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h,
                                                    int *w) {
    int dims[4];
    const cudnnStatus_t status = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, 4, dims);
    if (status != CUDNN_STATUS_SUCCESS) {
        return status;
    }
    int *out[4] = {n, c, h, w};
    for (int i = 0; i < 4; i++) {
        if (out[i] != nullptr) *out[i] = dims[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
    if (convDesc == nullptr || groupCount <= 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    convDesc->groups = groupCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
    if (convDesc == nullptr || groupCount == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    *groupCount = convDesc->groups;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
    if (convDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    convDesc->math = mathType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
    if (convDesc == nullptr || mathType == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    *mathType = convDesc->math;
    return CUDNN_STATUS_SUCCESS;
}

namespace {

// (the math type is the descriptor's: the kernels compute exactly in any case)
template<typename Perf, typename Algo>
cudnnStatus_t oneAlgorithm(int requested, int *returned, Perf *perf, Algo algo, size_t workspace,
                           cudnnMathType_t math) {
    if (returned == nullptr || perf == nullptr || requested <= 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    perf[0] = Perf{};
    perf[0].algo = algo;
    perf[0].status = CUDNN_STATUS_SUCCESS;
    perf[0].time = 0;
    perf[0].memory = workspace;
    perf[0].determinism = CUDNN_DETERMINISTIC;
    perf[0].mathType = math;
    *returned = 1;
    return CUDNN_STATUS_SUCCESS;
}

size_t forwardWorkspace(const cudnnTensorStruct *x, const cudnnFilterStruct *w, const cudnnConvolutionStruct *c,
                        const cudnnTensorStruct *y) {
    return convWorkspaceSize(ConvKind::FORWARD, x->tensor, w->tensor, y->tensor, geometryOf(c));
}

size_t dataWorkspace(const cudnnFilterStruct *w, const cudnnTensorStruct *dy, const cudnnConvolutionStruct *c,
                     const cudnnTensorStruct *dx) {
    return convWorkspaceSize(ConvKind::BACKWARD_DATA, dx->tensor, w->tensor, dy->tensor, geometryOf(c));
}

size_t filterWorkspace(const cudnnTensorStruct *x, const cudnnTensorStruct *dy, const cudnnConvolutionStruct *c,
                       const cudnnFilterStruct *dw) {
    return convWorkspaceSize(ConvKind::BACKWARD_FILTER, x->tensor, dw->tensor, dy->tensor, geometryOf(c));
}

} // namespace

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
                                                     const cudnnFilterDescriptor_t filterDesc,
                                                     const cudnnConvolutionDescriptor_t convDesc,
                                                     const cudnnTensorDescriptor_t destDesc,
                                                     const int requestedAlgoCount, int *returnedAlgoCount,
                                                     cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    CHECK_DNN(handle);
    if (!descriptorsValid(srcDesc, filterDesc, convDesc, destDesc)) return CUDNN_STATUS_BAD_PARAM;
    return oneAlgorithm(requestedAlgoCount, returnedAlgoCount, perfResults, CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                        forwardWorkspace(srcDesc, filterDesc, convDesc, destDesc), convDesc->math);
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
                                                          const cudnnFilterDescriptor_t filterDesc,
                                                          const cudnnTensorDescriptor_t diffDesc,
                                                          const cudnnConvolutionDescriptor_t convDesc,
                                                          const cudnnTensorDescriptor_t gradDesc,
                                                          const int requestedAlgoCount, int *returnedAlgoCount,
                                                          cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    CHECK_DNN(handle);
    if (!descriptorsValid(filterDesc, diffDesc, convDesc, gradDesc)) return CUDNN_STATUS_BAD_PARAM;
    return oneAlgorithm(requestedAlgoCount, returnedAlgoCount, perfResults, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                        dataWorkspace(filterDesc, diffDesc, convDesc, gradDesc), convDesc->math);
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                                            const cudnnTensorDescriptor_t srcDesc,
                                                            const cudnnTensorDescriptor_t diffDesc,
                                                            const cudnnConvolutionDescriptor_t convDesc,
                                                            const cudnnFilterDescriptor_t gradDesc,
                                                            const int requestedAlgoCount, int *returnedAlgoCount,
                                                            cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    CHECK_DNN(handle);
    if (!descriptorsValid(srcDesc, diffDesc, convDesc, gradDesc)) return CUDNN_STATUS_BAD_PARAM;
    return oneAlgorithm(requestedAlgoCount, returnedAlgoCount, perfResults, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                        filterWorkspace(srcDesc, diffDesc, convDesc, gradDesc), convDesc->math);
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
                                                     const void *, const cudnnFilterDescriptor_t wDesc, const void *,
                                                     const cudnnConvolutionDescriptor_t convDesc,
                                                     const cudnnTensorDescriptor_t yDesc, void *,
                                                     const int requestedAlgoCount, int *returnedAlgoCount,
                                                     cudnnConvolutionFwdAlgoPerf_t *perfResults, void *, size_t) {
    return cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount,
                                                  returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
                                                          const void *, const cudnnTensorDescriptor_t dyDesc,
                                                          const void *, const cudnnConvolutionDescriptor_t convDesc,
                                                          const cudnnTensorDescriptor_t dxDesc, void *,
                                                          const int requestedAlgoCount, int *returnedAlgoCount,
                                                          cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *,
                                                          size_t) {
    return cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount,
                                                       returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
                                                            const void *, const cudnnTensorDescriptor_t dyDesc,
                                                            const void *, const cudnnConvolutionDescriptor_t convDesc,
                                                            const cudnnFilterDescriptor_t dwDesc, void *,
                                                            const int requestedAlgoCount, int *returnedAlgoCount,
                                                            cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *,
                                                            size_t) {
    return cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount,
                                                         returnedAlgoCount, perfResults);
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
                                                      const cudnnFilterDescriptor_t wDesc,
                                                      const cudnnConvolutionDescriptor_t convDesc,
                                                      const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t,
                                                      size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (!descriptorsValid(xDesc, wDesc, convDesc, yDesc) || sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = forwardWorkspace(xDesc, wDesc, convDesc, yDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
                                                           const cudnnTensorDescriptor_t dyDesc,
                                                           const cudnnConvolutionDescriptor_t convDesc,
                                                           const cudnnTensorDescriptor_t dxDesc,
                                                           cudnnConvolutionBwdDataAlgo_t, size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (!descriptorsValid(wDesc, dyDesc, convDesc, dxDesc) || sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = dataWorkspace(wDesc, dyDesc, convDesc, dxDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
                                                             const cudnnTensorDescriptor_t dyDesc,
                                                             const cudnnConvolutionDescriptor_t convDesc,
                                                             const cudnnFilterDescriptor_t gradDesc,
                                                             cudnnConvolutionBwdFilterAlgo_t, size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (!descriptorsValid(xDesc, dyDesc, convDesc, gradDesc) || sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = filterWorkspace(xDesc, dyDesc, convDesc, gradDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc,
                                      const void *x, const cudnnFilterDescriptor_t wDesc, const void *w,
                                      const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t,
                                      void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
                                      const cudnnTensorDescriptor_t yDesc, void *y) {
    CHECK_DNN(handle);
    if (!descriptorsValid(xDesc, wDesc, convDesc, yDesc) || alpha == nullptr || beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = yDesc->tensor.type;
    return convRun(ConvKind::FORWARD, handle->stream, xDesc->tensor, const_cast<void *>(x), wDesc->tensor,
                   const_cast<void *>(w), yDesc->tensor, y, geometryOf(convDesc), scalarArg(alpha, type),
                   scalarArg(beta, type), workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc,
                                           const void *w, const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                           const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t,
                                           void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
                                           const cudnnTensorDescriptor_t dxDesc, void *dx) {
    CHECK_DNN(handle);
    if (!descriptorsValid(wDesc, dyDesc, convDesc, dxDesc) || alpha == nullptr || beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = dxDesc->tensor.type;
    return convRun(ConvKind::BACKWARD_DATA, handle->stream, dxDesc->tensor, dx, wDesc->tensor, const_cast<void *>(w),
                   dyDesc->tensor, const_cast<void *>(dy), geometryOf(convDesc), scalarArg(alpha, type),
                   scalarArg(beta, type), workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t handle, const void *alpha,
                                             const cudnnTensorDescriptor_t xDesc, const void *x,
                                             const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                             const cudnnConvolutionDescriptor_t convDesc,
                                             cudnnConvolutionBwdFilterAlgo_t, void *workSpace,
                                             size_t workSpaceSizeInBytes, const void *beta,
                                             const cudnnFilterDescriptor_t dwDesc, void *dw) {
    CHECK_DNN(handle);
    if (!descriptorsValid(xDesc, dyDesc, convDesc, dwDesc) || alpha == nullptr || beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = dwDesc->tensor.type;
    return convRun(ConvKind::BACKWARD_FILTER, handle->stream, xDesc->tensor, const_cast<void *>(x), dwDesc->tensor, dw,
                   dyDesc->tensor, const_cast<void *>(dy), geometryOf(convDesc), scalarArg(alpha, type),
                   scalarArg(beta, type), workSpace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnConvolutionBiasActivationForward(cudnnHandle_t handle, const void *alpha1,
                                                    const cudnnTensorDescriptor_t xDesc, const void *x,
                                                    const cudnnFilterDescriptor_t wDesc, const void *w,
                                                    const cudnnConvolutionDescriptor_t convDesc,
                                                    cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                    size_t workSpaceSizeInBytes, const void *alpha2,
                                                    const cudnnTensorDescriptor_t zDesc, const void *z,
                                                    const cudnnTensorDescriptor_t biasDesc, const void *bias,
                                                    const cudnnActivationDescriptor_t activationDesc,
                                                    const cudnnTensorDescriptor_t yDesc, void *y) {
    CHECK_DNN(handle);
    if (!descriptorsValid(xDesc, wDesc, convDesc, yDesc) || zDesc == nullptr || activationDesc == nullptr ||
        alpha1 == nullptr || alpha2 == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    // y = act(alpha1 conv(x) + alpha2 z + bias): z first (unless it is y), then the convolution added
    const cudnnDataType_t type = yDesc->tensor.type;
    const double a2 = scalarArg(alpha2, type);
    if (a2 != 0 && z != y) {
        const cudnnStatus_t copied = copyTensor(handle->stream, zDesc->tensor, z, yDesc->tensor, y);
        if (copied != CUDNN_STATUS_SUCCESS) return copied;
    }
    const cudnnStatus_t status = convRun(ConvKind::FORWARD, handle->stream, xDesc->tensor, const_cast<void *>(x),
                                         wDesc->tensor, const_cast<void *>(w), yDesc->tensor, y, geometryOf(convDesc),
                                         scalarArg(alpha1, type), a2, workSpace, workSpaceSizeInBytes);
    (void) algo;
    if (status != CUDNN_STATUS_SUCCESS) {
        return status;
    }
    return biasActivation(handle->stream, yDesc->tensor, y, biasDesc != nullptr ? bias : nullptr,
                          activationDesc->mode, activationDesc->coef);
}

// ---------------------------------------------------------------------------------------------------------------------
// activations

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
    if (activationDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *activationDesc = new cudnnActivationStruct{CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
    delete activationDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
                                           cudnnNanPropagation_t reluNanOpt, double coef) {
    if (activationDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *activationDesc = cudnnActivationStruct{mode, reluNanOpt, coef};
    return CUDNN_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// batch normalization

namespace {

BatchNorm batchNormOf(cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops, cudnnActivationDescriptor_t activation,
                      double epsilon) {
    BatchNorm bn{};
    bn.mode = mode;
    bn.ops = ops == CUDNN_BATCHNORM_OPS_BN ? 0 : ops == CUDNN_BATCHNORM_OPS_BN_ACTIVATION ? 1 : 2;
    bn.activation = activation != nullptr ? activation->mode : CUDNN_ACTIVATION_IDENTITY;
    bn.activationCoef = activation != nullptr ? activation->coef : 0;
    bn.epsilon = epsilon;
    return bn;
}

} // namespace

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle, cudnnBatchNormMode_t,
                                                                       cudnnBatchNormOps_t,
                                                                       const cudnnTensorDescriptor_t,
                                                                       const cudnnTensorDescriptor_t,
                                                                       const cudnnTensorDescriptor_t,
                                                                       const cudnnTensorDescriptor_t,
                                                                       const cudnnActivationDescriptor_t,
                                                                       size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle, cudnnBatchNormMode_t,
                                                                cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
                                                                const cudnnTensorDescriptor_t,
                                                                const cudnnTensorDescriptor_t,
                                                                const cudnnTensorDescriptor_t,
                                                                const cudnnTensorDescriptor_t,
                                                                const cudnnTensorDescriptor_t,
                                                                const cudnnActivationDescriptor_t,
                                                                size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle, cudnnBatchNormMode_t,
                                                                   cudnnBatchNormOps_t,
                                                                   const cudnnActivationDescriptor_t,
                                                                   const cudnnTensorDescriptor_t,
                                                                   size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    // (the backward pass recomputes the normalized values from x and the saved statistics)
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha,
        const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,
        const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData,
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
        double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon,
        void *resultSaveMean, void *resultSaveInvVariance, cudnnActivationDescriptor_t activationDesc, void *, size_t,
        void *, size_t) {
    CHECK_DNN(handle);
    if (xDesc == nullptr || yDesc == nullptr || bnScaleBiasMeanVarDesc == nullptr || alpha == nullptr ||
        beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = yDesc->tensor.type;
    return batchNormForwardTraining(handle->stream, batchNormOf(mode, bnOps, activationDesc, epsilon),
                                    scalarArg(alpha, type), scalarArg(beta, type), xDesc->tensor, xData,
                                    zDesc != nullptr ? zDesc->tensor : xDesc->tensor, zData, yDesc->tensor, yData,
                                    bnScaleBiasMeanVarDesc->tensor, bnScale, bnBias, exponentialAverageFactor,
                                    resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveInvVariance);
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta,
        const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y,
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
        double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon,
        void *resultSaveMean, void *resultSaveInvVariance) {
    return cudnnBatchNormalizationForwardTrainingEx(handle, mode, CUDNN_BATCHNORM_OPS_BN, alpha, beta, xDesc, x,
                                                    nullptr, nullptr, yDesc, y, bnScaleBiasMeanVarDesc, bnScale,
                                                    bnBias, exponentialAverageFactor, resultRunningMean,
                                                    resultRunningVariance, epsilon, resultSaveMean,
                                                    resultSaveInvVariance, nullptr, nullptr, 0, nullptr, 0);
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode,
                                                      const void *alpha, const void *beta,
                                                      const cudnnTensorDescriptor_t xDesc, const void *x,
                                                      const cudnnTensorDescriptor_t yDesc, void *y,
                                                      const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                      const void *bnScale, const void *bnBias,
                                                      const void *estimatedMean, const void *estimatedVariance,
                                                      double epsilon) {
    CHECK_DNN(handle);
    if (xDesc == nullptr || yDesc == nullptr || bnScaleBiasMeanVarDesc == nullptr || alpha == nullptr ||
        beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = yDesc->tensor.type;
    return batchNormForwardInference(handle->stream, batchNormOf(mode, CUDNN_BATCHNORM_OPS_BN, nullptr, epsilon),
                                     scalarArg(alpha, type), scalarArg(beta, type), xDesc->tensor, x, yDesc->tensor, y,
                                     bnScaleBiasMeanVarDesc->tensor, bnScale, bnBias, estimatedMean,
                                     estimatedVariance);
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(
        cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alphaDataDiff,
        const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff,
        const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t yDesc, const void *yData,
        const cudnnTensorDescriptor_t dyDesc, const void *dyData, const cudnnTensorDescriptor_t dzDesc, void *dzData,
        const cudnnTensorDescriptor_t dxDesc, void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
        const void *bnScaleData, const void *bnBiasData, void *dBnScaleData, void *dBnBiasData, double epsilon,
        const void *savedMean, const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc, void *,
        size_t, void *, size_t) {
    CHECK_DNN(handle);
    if (xDesc == nullptr || dyDesc == nullptr || dxDesc == nullptr || dBnScaleBiasDesc == nullptr ||
        alphaDataDiff == nullptr || betaDataDiff == nullptr || alphaParamDiff == nullptr || betaParamDiff == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = dxDesc->tensor.type;
    return batchNormBackward(handle->stream, batchNormOf(mode, bnOps, activationDesc, epsilon),
                             scalarArg(alphaDataDiff, type), scalarArg(betaDataDiff, type),
                             scalarArg(alphaParamDiff, type), scalarArg(betaParamDiff, type), xDesc->tensor, xData,
                             yDesc != nullptr ? yDesc->tensor : xDesc->tensor, yData, dyDesc->tensor, dyData,
                             dzDesc != nullptr ? dzDesc->tensor : dxDesc->tensor, dzData, dxDesc->tensor, dxData,
                             dBnScaleBiasDesc->tensor, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, savedMean,
                             savedInvVariance);
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t handle, cudnnBatchNormMode_t mode,
                                              const void *alphaDataDiff, const void *betaDataDiff,
                                              const void *alphaParamDiff, const void *betaParamDiff,
                                              const cudnnTensorDescriptor_t xDesc, const void *x,
                                              const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                              const cudnnTensorDescriptor_t dxDesc, void *dx,
                                              const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
                                              void *dBnScaleResult, void *dBnBiasResult, double epsilon,
                                              const void *savedMean, const void *savedInvVariance) {
    return cudnnBatchNormalizationBackwardEx(handle, mode, CUDNN_BATCHNORM_OPS_BN, alphaDataDiff, betaDataDiff,
                                             alphaParamDiff, betaParamDiff, xDesc, x, nullptr, nullptr, dyDesc, dy,
                                             nullptr, nullptr, dxDesc, dx, dBnScaleBiasDesc, bnScale, nullptr,
                                             dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance,
                                             nullptr, nullptr, 0, nullptr, 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// pooling

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
    if (poolingDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *poolingDesc = new cudnnPoolingStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
    delete poolingDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
                                          cudnnNanPropagation_t, int windowHeight, int windowWidth,
                                          int verticalPadding, int horizontalPadding, int verticalStride,
                                          int horizontalStride) {
    if (poolingDesc == nullptr || windowHeight <= 0 || windowWidth <= 0 || verticalStride <= 0 ||
        horizontalStride <= 0) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    Pooling &p = poolingDesc->pooling;
    p = Pooling{mode, 2, {windowHeight, windowWidth, 1}, {verticalPadding, horizontalPadding, 0},
                {verticalStride, horizontalStride, 1}};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                                  const cudnnTensorDescriptor_t yDesc, void *y) {
    CHECK_DNN(handle);
    if (poolingDesc == nullptr || xDesc == nullptr || yDesc == nullptr || alpha == nullptr || beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = yDesc->tensor.type;
    return poolingForward(handle->stream, poolingDesc->pooling, scalarArg(alpha, type), xDesc->tensor, x,
                          scalarArg(beta, type), yDesc->tensor, y);
}

// ---------------------------------------------------------------------------------------------------------------------
// dropout (for RNNs): the parameters are kept

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
    if (dropoutDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *dropoutDesc = new cudnnDropoutStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
    delete dropoutDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes) {
    CHECK_DNN(handle);
    if (sizeInBytes == nullptr) return CUDNN_STATUS_BAD_PARAM;
    // (the random state is the seed: a counter based generator needs no memory)
    *sizeInBytes = 64;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
                                        void *states, size_t stateSizeInBytes, unsigned long long seed) {
    CHECK_DNN(handle);
    if (dropoutDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *dropoutDesc = cudnnDropoutStruct{dropout, states, stateSizeInBytes, seed};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
                                            void *states, size_t stateSizeInBytes, unsigned long long seed) {
    return cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
}

// ---------------------------------------------------------------------------------------------------------------------
// spatial transformer

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
    if (stDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *stDesc = new cudnnSpatialTransformerStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
    delete stDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                                     cudnnSamplerType_t samplerType, cudnnDataType_t dataType,
                                                     const int nbDims, const int dimA[]) {
    if (stDesc == nullptr || samplerType != CUDNN_SAMPLER_BILINEAR || nbDims != 4 || dimA == nullptr) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    stDesc->type = dataType;
    stDesc->nbDims = nbDims;
    for (int i = 0; i < nbDims; i++) stDesc->dims[i] = dimA[i];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
                                                 const void *theta, void *grid) {
    CHECK_DNN(handle);
    if (stDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    return gridGeneratorForward(handle->stream, stDesc->type, stDesc->dims[0], stDesc->dims[2], stDesc->dims[3], theta,
                                grid);
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                                  const cudnnSpatialTransformerDescriptor_t stDesc, const void *dgrid,
                                                  void *dtheta) {
    CHECK_DNN(handle);
    if (stDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    return gridGeneratorBackward(handle->stream, stDesc->type, stDesc->dims[0], stDesc->dims[2], stDesc->dims[3],
                                 dgrid, dtheta);
}

cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
                                           const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                           const void *grid, const void *beta, cudnnTensorDescriptor_t yDesc,
                                           void *y) {
    CHECK_DNN(handle);
    if (stDesc == nullptr || xDesc == nullptr || yDesc == nullptr || alpha == nullptr || beta == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = yDesc->tensor.type;
    return samplerForward(handle->stream, scalarArg(alpha, type), xDesc->tensor, x, grid, scalarArg(beta, type),
                          yDesc->tensor, y);
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
                                            const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                                            const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx,
                                            const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
                                            const void *dy, const void *grid, const void *betaDgrid, void *dgrid) {
    CHECK_DNN(handle);
    if (stDesc == nullptr || xDesc == nullptr || dxDesc == nullptr || dyDesc == nullptr || alpha == nullptr ||
        beta == nullptr || alphaDgrid == nullptr || betaDgrid == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const cudnnDataType_t type = dxDesc->tensor.type;
    return samplerBackward(handle->stream, scalarArg(alpha, type), xDesc->tensor, x, scalarArg(beta, type),
                           dxDesc->tensor, dx, scalarArg(alphaDgrid, type), dyDesc->tensor, dy, grid,
                           scalarArg(betaDgrid, type), dgrid);
}
