/**
 * LibreCUDA's cuDNN (libcudnn.so.9): the parts shared by the legacy API (cudnn_legacy.cpp), the backend graph API
 * (cudnn_backend.cpp) and the kernels (cudnn_*.cu). Only the public cudnn*.h headers are used for types.
 */
#pragma once

#include <cudnn.h>

#include <cstddef>
#include <cstdint>

struct cudnnContext {
    uint32_t magic;
    cudaStream_t stream;
};

namespace librecuda_dnn {
struct Tensor;
}

namespace librecuda_dnn {

constexpr uint32_t HANDLE_MAGIC = 0x4c43444e;

inline bool valid(cudnnHandle_t handle) {
    return handle != nullptr && handle->magic == HANDLE_MAGIC;
}

constexpr int MAX_DIMS = 8;

// a tensor: N, C and up to 3 spatial dimensions (fewer are padded with size 1), element strides
struct Tensor {
    int nbDims;
    int64_t dims[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    cudnnDataType_t type;
};

struct ConvGeometry {
    int spatial;
    int64_t padBefore[3], padAfter[3], stride[3], dilation[3];
    // CUDNN_CONVOLUTION flips the filter (CUDNN_CROSS_CORRELATION doesn't)
    bool flip;
    // float or double accumulation
    cudnnDataType_t compute;
};

enum class ConvKind {
    FORWARD, BACKWARD_DATA, BACKWARD_FILTER
};

/**
 * Convolutions: x (N, C, spatial), w (K, C / groups, filter spatial), y (N, K, output spatial). FORWARD: y = alpha
 * conv(x, w) + beta y; BACKWARD_DATA: x = alpha conv^T(y, w) + beta x (y holds dy); BACKWARD_FILTER: w = alpha
 * dw(x, y) + beta w. The workspace size depends on the shapes only.
 */
cudnnStatus_t convCheck(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g);
size_t convWorkspaceSize(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g);
cudnnStatus_t convRun(ConvKind kind, cudaStream_t stream, const Tensor &x, void *xData, const Tensor &w, void *wData,
                      const Tensor &y, void *yData, const ConvGeometry &g, double alpha, double beta, void *workspace,
                      size_t workspaceSize);

// CUTLASS implicit GEMM convolutions of half / bfloat16 1D and 2D problems without groups (cudnn_implicit.cu): the
// workspace they need, 0 when they can't run the problem (convRun then uses im2col)
size_t implicitConvWorkspace(ConvKind kind, const Tensor &x, const Tensor &w, const Tensor &y, const ConvGeometry &g);
cudnnStatus_t implicitConvRun(ConvKind kind, cudaStream_t stream, const Tensor &x, void *xData, const Tensor &w,
                              void *wData, const Tensor &y, void *yData, const ConvGeometry &g, double alpha,
                              double beta, void *workspace);

// output dimension of a convolution
inline int64_t convOutputSize(int64_t input, int64_t filter, int64_t padBefore, int64_t padAfter, int64_t stride,
                              int64_t dilation) {
    return (input + padBefore + padAfter - dilation * (filter - 1) - 1) / stride + 1;
}

size_t dataTypeSize(cudnnDataType_t type);

} // namespace librecuda_dnn

// descriptors shared by the legacy API (cudnn_legacy.cpp) and the RNN functions (cudnn_rnn.cu)
struct cudnnTensorStruct {
    librecuda_dnn::Tensor tensor;
};

struct cudnnDropoutStruct {
    float dropout;
    void *states;
    size_t stateSize;
    unsigned long long seed;
};

namespace librecuda_dnn {

// ---- elementwise operations and normalization (cudnn_ops.cu)

// y = activation(y + bias (per channel, may be nullptr)); coef: clipping threshold / ELU alpha / swish beta
cudnnStatus_t biasActivation(cudaStream_t stream, const Tensor &y, void *yData, const void *bias,
                             cudnnActivationMode_t mode, double coef);
// dst = src (same shape, any strides and types)
cudnnStatus_t copyTensor(cudaStream_t stream, const Tensor &src, const void *srcData, const Tensor &dst,
                         void *dstData);

struct BatchNorm {
    cudnnBatchNormMode_t mode;
    // 0: none, 1: activation, 2: add z then activation (CUDNN_BATCHNORM_OPS_*)
    int ops;
    cudnnActivationMode_t activation;
    double activationCoef;
    double epsilon;
};

// training forward: batch statistics, running statistics update (factor), saved mean / inverse standard deviation
cudnnStatus_t batchNormForwardTraining(cudaStream_t stream, const BatchNorm &bn, double alpha, double beta,
                                       const Tensor &x, const void *xData, const Tensor &z, const void *zData,
                                       const Tensor &y, void *yData, const Tensor &param, const void *scale,
                                       const void *bias, double factor, void *runningMean, void *runningVar,
                                       void *saveMean, void *saveInvStd);
cudnnStatus_t batchNormForwardInference(cudaStream_t stream, const BatchNorm &bn, double alpha, double beta,
                                        const Tensor &x, const void *xData, const Tensor &y, void *yData,
                                        const Tensor &param, const void *scale, const void *bias, const void *mean,
                                        const void *var);
cudnnStatus_t batchNormBackward(cudaStream_t stream, const BatchNorm &bn, double alphaData, double betaData,
                                double alphaParam, double betaParam, const Tensor &x, const void *xData,
                                const Tensor &y, const void *yData, const Tensor &dy, const void *dyData,
                                const Tensor &dz, void *dzData, const Tensor &dx, void *dxData, const Tensor &param,
                                const void *scale, const void *bias, void *dScale, void *dBias, const void *savedMean,
                                const void *savedInvStd);

struct Pooling {
    cudnnPoolingMode_t mode;
    int spatial;
    int window[3], pad[3], stride[3];
};

cudnnStatus_t poolingForward(cudaStream_t stream, const Pooling &pool, double alpha, const Tensor &x,
                             const void *xData, double beta, const Tensor &y, void *yData);

// spatial transformer (bilinear sampling of x at the grid's [-1, 1] coordinates, corners aligned)
cudnnStatus_t gridGeneratorForward(cudaStream_t stream, cudnnDataType_t type, int n, int h, int w, const void *theta,
                                   void *grid);
cudnnStatus_t gridGeneratorBackward(cudaStream_t stream, cudnnDataType_t type, int n, int h, int w, const void *dgrid,
                                    void *dtheta);
cudnnStatus_t samplerForward(cudaStream_t stream, double alpha, const Tensor &x, const void *xData, const void *grid,
                             double beta, const Tensor &y, void *yData);
cudnnStatus_t samplerBackward(cudaStream_t stream, double alpha, const Tensor &x, const void *xData, double beta,
                              const Tensor &dx, void *dxData, double alphaGrid, const Tensor &dy, const void *dyData,
                              const void *grid, double betaGrid, void *dgrid);

// a scalar argument of cuDNN (double for double data, float otherwise)
inline double scalarArg(const void *p, cudnnDataType_t type) {
    return type == CUDNN_DATA_DOUBLE ? *static_cast<const double *>(p) : *static_cast<const float *>(p);
}

} // namespace librecuda_dnn
