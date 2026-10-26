/**
 * LibreCUDA's cuDNN: recurrent networks (RNN with ReLU / tanh, LSTM, GRU; several layers, both directions, sequences of
 * different lengths in the three data layouts, dropout between layers) and the CTC loss.
 *
 * RNNs: per layer and direction, the input projections of all time steps are one GEMM; each time step is a GEMM of the
 * recurrent weights and a cell kernel. The weights of a pseudo layer (layer * directions + direction) are stored as
 * W (all gates, gate after gate, each hidden x input row-major), R (likewise, hidden x hidden), then the input biases
 * and the recurrent biases, which is the layout PyTorch's weight flattening expects. Training keeps the layer outputs,
 * gates, cell states and (backward) gate gradients in the reserve space. Projections (projSize < hiddenSize) and the
 * skip input mode are not supported.
 *
 * CTC: log-space forward-backward per sequence (one thread block each), gradients with respect to the activations
 * (softmax normalization) or the probabilities.
 */
#include "blas_internal.h"
#include "dnn_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace librecuda_dnn;

struct cudnnRNNStruct {
    cudnnRNNMode_t cell;
    cudnnRNNBiasMode_t bias;
    cudnnDirectionMode_t direction;
    cudnnRNNInputMode_t input;
    cudnnDataType_t type;
    int inputSize, hiddenSize, projSize, layers;
    float dropout;
    unsigned long long seed;
    // float GEMMs on TF32 tensor cores: the math type isn't CUDNN_FMA_MATH (PyTorch passes CUDNN_DEFAULT_MATH unless
    // torch.backends.cudnn.allow_tf32 is off), as NVIDIA's cuDNN does; exact float GEMMs made a bidirectional GRU's
    // training step 5.6 times as long
    bool tf32;
};

struct cudnnRNNDataStruct {
    cudnnDataType_t type;
    cudnnRNNDataLayout_t layout;
    int maxLength, batch, vectorSize;
    std::vector<int> lengths;
};

struct cudnnCTCLossStruct {
    cudnnDataType_t compute;
    cudnnLossNormalizationMode_t normalization;
    int maxLabelLength;
};

namespace {

#define GRID_LOOP(i, total) \
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < (total); \
         i += static_cast<long long>(gridDim.x) * blockDim.x)

unsigned int gridFor(long long count) {
    return static_cast<unsigned int>(std::max<long long>(1, std::min<long long>((count + 255) / 256, 1 << 20)));
}

__device__ inline float loadf(const void *base, long long i, cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_HALF: return __half2float(static_cast<const __half *>(base)[i]);
        case CUDNN_DATA_BFLOAT16: return __bfloat162float(static_cast<const __nv_bfloat16 *>(base)[i]);
        case CUDNN_DATA_DOUBLE: return float(static_cast<const double *>(base)[i]);
        default: return static_cast<const float *>(base)[i];
    }
}

__device__ inline void storef(void *base, long long i, cudnnDataType_t type, float v) {
    switch (type) {
        case CUDNN_DATA_HALF: static_cast<__half *>(base)[i] = __float2half(v); break;
        case CUDNN_DATA_BFLOAT16: static_cast<__nv_bfloat16 *>(base)[i] = __float2bfloat16(v); break;
        case CUDNN_DATA_DOUBLE: static_cast<double *>(base)[i] = v; break;
        default: static_cast<float *>(base)[i] = v; break;
    }
}

__device__ inline float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

int gatesOf(cudnnRNNMode_t cell) {
    return cell == CUDNN_LSTM ? 4 : cell == CUDNN_GRU ? 3 : 1;
}

// ---- weight layout

struct WeightLayout {
    // per pseudo layer, element offsets of W, R, input bias, recurrent bias (-1: none)
    std::vector<long long> w, r, bw, br;
    long long total;
};

int layerInputSize(const cudnnRNNStruct &d, int layer) {
    const int dirs = d.direction == CUDNN_BIDIRECTIONAL ? 2 : 1;
    return layer == 0 ? d.inputSize : d.hiddenSize * dirs;
}

WeightLayout weightLayout(const cudnnRNNStruct &d) {
    const int dirs = d.direction == CUDNN_BIDIRECTIONAL ? 2 : 1, gates = gatesOf(d.cell);
    const bool input_bias = d.bias == CUDNN_RNN_DOUBLE_BIAS || d.bias == CUDNN_RNN_SINGLE_INP_BIAS;
    const bool recurrent_bias = d.bias == CUDNN_RNN_DOUBLE_BIAS || d.bias == CUDNN_RNN_SINGLE_REC_BIAS;
    WeightLayout l{};
    long long offset = 0;
    for (int layer = 0; layer < d.layers; layer++) {
        for (int dir = 0; dir < dirs; dir++) {
            const long long h = d.hiddenSize, in = layerInputSize(d, layer);
            l.w.push_back(offset);
            offset += gates * h * in;
            l.r.push_back(offset);
            offset += gates * h * h;
            l.bw.push_back(input_bias ? offset : -1);
            offset += input_bias ? gates * h : 0;
            l.br.push_back(recurrent_bias ? offset : -1);
            offset += recurrent_bias ? gates * h : 0;
        }
    }
    l.total = offset;
    return l;
}

// ---- sequences: the row (in the x / y buffers) of time step t of sequence b; -1 past its end

struct Sequences {
    int layout;
    int maxLength, batch;
    const int *lengths;  // device
    const int *offsets;  // device: rows before time step t (packed layout)
};

__device__ inline long long rowOf(const Sequences &s, int t, int b) {
    if (t >= s.lengths[b]) {
        return -1;
    }
    switch (s.layout) {
        case CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED: return s.offsets[t] + b;
        case CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED: return static_cast<long long>(b) * s.maxLength + t;
        default: return static_cast<long long>(t) * s.batch + b;
    }
}

// rows of a buffer laid out like x / y
long long rowCount(const cudnnRNNDataStruct &x) {
    if (x.layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED) {
        long long total = 0;
        for (int length: x.lengths) total += length;
        return total;
    }
    return static_cast<long long>(x.maxLength) * x.batch;
}

// ---- GEMMs (column-major; A and B of the data type, C of the data type or float)

cudaDataType blasType(cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_HALF: return CUDA_R_16F;
        case CUDNN_DATA_BFLOAT16: return CUDA_R_16BF;
        case CUDNN_DATA_DOUBLE: return CUDA_R_64F;
        default: return CUDA_R_32F;
    }
}

cudnnStatus_t gemm(cudaStream_t stream, cudnnDataType_t type, bool tf32, bool floatOutput, cublasOperation_t ta,
                   cublasOperation_t tb, long long m, long long n, long long k, const void *a, long long lda,
                   const void *b, long long ldb, void *c, long long ldc, float beta) {
    using namespace librecuda_blas;
    if (m <= 0 || n <= 0) {
        return CUDNN_STATUS_SUCCESS;
    }
    const bool doubles = type == CUDNN_DATA_DOUBLE;
    const double one_d = 1, beta_d = beta;
    const float one_f = 1;
    GemmParams p{};
    p.transa = ta;
    p.transb = tb;
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = 1;
    p.a = Operand{a, nullptr, blasType(type), lda, 0, 1};
    p.b = Operand{b, nullptr, blasType(type), ldb, 0, 1};
    p.c = Operand{c, nullptr, floatOutput && !doubles ? CUDA_R_32F : blasType(type), ldc, 0, 1};
    p.d = p.c;
    p.compute = doubles ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    p.alpha = Scalar{doubles ? static_cast<const void *>(&one_d) : &one_f, doubles ? CUDA_R_64F : CUDA_R_32F, false};
    p.beta = Scalar{doubles ? static_cast<const void *>(&beta_d) : &beta, doubles ? CUDA_R_64F : CUDA_R_32F, false};
    p.allowTF32 = tf32 && type == CUDNN_DATA_FLOAT;
    p.stream = stream;
    return librecuda_blas_gemm(&p) == CUBLAS_STATUS_SUCCESS ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

// ---- kernels

// dst[r * dstLd + j] = src[r * srcLd + j] for r < rows, j < cols (any data types: through float)
__global__ void copyRows(long long rows, int cols, const void *src, long long srcLd, cudnnDataType_t srcType, void *dst,
                         long long dstLd, cudnnDataType_t dstType) {
    GRID_LOOP(e, rows * cols) {
        const long long r = e / cols, j = e % cols;
        storef(dst, r * dstLd + j, dstType, loadf(src, r * srcLd + j, srcType));
    }
}

// one time step: gates of the active sequences, new states; out: rows of the layer output (dir block)
struct StepArgs {
    cudnnRNNMode_t cell;
    cudnnDataType_t type;
    int hidden, gates, t, dir, dirs;
    Sequences seq;
    float *gx;          // [row][gates * hidden]: input projections (+ bias), replaced by the activated gates
    const float *gh;    // [b][gates * hidden]: recurrent projections of this step
    const void *bw, *br;  // biases (may be nullptr), data type
    void *h;            // [b][hidden], data type: the state (updated)
    float *c;           // [b][hidden]: LSTM cell state (updated)
    float *cells;       // [row][hidden]: LSTM cell states kept for backward (may be nullptr)
    float *recurrentN;  // [row][hidden]: GRU recurrent part of n kept for backward (may be nullptr)
    void *out;          // layer output rows, data type
    long long outLd;
};

__global__ void cellForward(StepArgs a) {
    GRID_LOOP(e, static_cast<long long>(a.seq.batch) * a.hidden) {
        const int b = int(e / a.hidden), j = int(e % a.hidden);
        const long long row = rowOf(a.seq, a.t, b);
        if (row < 0) continue;
        float *g = a.gx + row * a.gates * a.hidden;
        const float *r = a.gh + static_cast<long long>(b) * a.gates * a.hidden;
        auto pre = [&](int gate, bool recurrent) {
            const int k = gate * a.hidden + j;
            float v = recurrent ? r[k] : g[k];
            const void *bias = recurrent ? a.br : a.bw;
            if (bias != nullptr) v += loadf(bias, k, a.type);
            return v;
        };
        const long long state = static_cast<long long>(b) * a.hidden + j;
        float h;
        switch (a.cell) {
            case CUDNN_LSTM: {
                const float i = sigmoid(pre(0, false) + pre(0, true)), f = sigmoid(pre(1, false) + pre(1, true));
                const float c_gate = tanhf(pre(2, false) + pre(2, true)), o = sigmoid(pre(3, false) + pre(3, true));
                const float c = f * a.c[state] + i * c_gate;
                a.c[state] = c;
                if (a.cells != nullptr) a.cells[row * a.hidden + j] = c;
                h = o * tanhf(c);
                g[j] = i;
                g[a.hidden + j] = f;
                g[2 * a.hidden + j] = c_gate;
                g[3 * a.hidden + j] = o;
                break;
            }
            case CUDNN_GRU: {
                const float rg = sigmoid(pre(0, false) + pre(0, true)), z = sigmoid(pre(1, false) + pre(1, true));
                const float rn = pre(2, true);
                const float n = tanhf(pre(2, false) + rg * rn);
                const float h_prev = loadf(a.h, state, a.type);
                h = (1 - z) * n + z * h_prev;
                if (a.recurrentN != nullptr) a.recurrentN[row * a.hidden + j] = rn;
                g[j] = rg;
                g[a.hidden + j] = z;
                g[2 * a.hidden + j] = n;
                break;
            }
            case CUDNN_RNN_RELU:
                h = fmaxf(pre(0, false) + pre(0, true), 0.0f);
                g[j] = h;
                break;
            default:
                h = tanhf(pre(0, false) + pre(0, true));
                g[j] = h;
                break;
        }
        storef(a.h, state, a.type, h);
        storef(a.out, row * a.outLd + a.dir * a.hidden + j, a.type, h);
    }
}

// one step backward: gate gradients (dgx, dgh: data type, [row][gates * hidden]) and the gradient of the state
struct StepBackArgs {
    cudnnRNNMode_t cell;
    cudnnDataType_t type;
    int hidden, gates, t, dir;
    Sequences seq;
    const float *gates_;   // [row][gates * hidden] activated gates
    const float *cells;    // LSTM cell states per row
    const float *recurrentN;
    const void *hx, *cx;   // initial states [b][hidden] of this layer / direction (may be nullptr)
    const void *layerOut;  // layer output rows (the previous states), data type
    long long outLd;
    const void *dy;        // gradient of the layer output rows, data type
    long long dyLd;
    float *dh;             // [b][hidden]: gradient flowing to the previous step (in: from the next step)
    float *dc;             // [b][hidden]
    void *dgx, *dgh;       // [row][gates * hidden]
    float *dhDirect;       // [b][hidden]: GRU: the part of dh_prev not through R
};

// the state before time step t of sequence b (row of the previous step, or the initial state)
__device__ inline float previousState(const StepBackArgs &a, int b, int j, bool cell) {
    const int t_prev = a.dir == 0 ? a.t - 1 : a.t + 1;
    const long long row = t_prev >= 0 ? rowOf(a.seq, t_prev, b) : -1;
    if (row >= 0) {
        return cell ? a.cells[row * a.hidden + j] : loadf(a.layerOut, row * a.outLd + a.dir * a.hidden + j, a.type);
    }
    const void *init = cell ? a.cx : a.hx;
    return init != nullptr ? loadf(init, static_cast<long long>(b) * a.hidden + j, a.type) : 0.0f;
}

__global__ void cellBackward(StepBackArgs a) {
    GRID_LOOP(e, static_cast<long long>(a.seq.batch) * a.hidden) {
        const int b = int(e / a.hidden), j = int(e % a.hidden);
        const long long row = rowOf(a.seq, a.t, b);
        const long long state = static_cast<long long>(b) * a.hidden + j;
        if (row < 0) {
            // the state passes unchanged: its gradient too
            if (a.dhDirect != nullptr) a.dhDirect[state] = 0;
            continue;
        }
        const float dh = a.dh[state] + (a.dy != nullptr ? loadf(a.dy, row * a.dyLd + a.dir * a.hidden + j, a.type) : 0);
        const float *g = a.gates_ + row * a.gates * a.hidden;
        const long long gr = row * a.gates * a.hidden;
        switch (a.cell) {
            case CUDNN_LSTM: {
                const float i = g[j], f = g[a.hidden + j], cg = g[2 * a.hidden + j], o = g[3 * a.hidden + j];
                const float c = a.cells[row * a.hidden + j], tc = tanhf(c);
                const float c_prev = previousState(a, b, j, true);
                const float dc = a.dc[state] + dh * o * (1 - tc * tc);
                const float d[4] = {dc * cg * i * (1 - i), dc * c_prev * f * (1 - f), dc * i * (1 - cg * cg),
                                    dh * tc * o * (1 - o)};
                for (int k = 0; k < 4; k++) {
                    storef(a.dgx, gr + k * a.hidden + j, a.type, d[k]);
                    storef(a.dgh, gr + k * a.hidden + j, a.type, d[k]);
                }
                a.dc[state] = dc * f;
                break;
            }
            case CUDNN_GRU: {
                const float r = g[j], z = g[a.hidden + j], n = g[2 * a.hidden + j];
                const float rn = a.recurrentN[row * a.hidden + j];
                const float h_prev = previousState(a, b, j, false);
                const float dn = dh * (1 - z) * (1 - n * n);
                const float dz = dh * (h_prev - n) * z * (1 - z);
                const float dr = dn * rn * r * (1 - r);
                storef(a.dgx, gr + j, a.type, dr);
                storef(a.dgh, gr + j, a.type, dr);
                storef(a.dgx, gr + a.hidden + j, a.type, dz);
                storef(a.dgh, gr + a.hidden + j, a.type, dz);
                storef(a.dgx, gr + 2 * a.hidden + j, a.type, dn);
                storef(a.dgh, gr + 2 * a.hidden + j, a.type, dn * r);
                a.dhDirect[state] = dh * z;
                break;
            }
            case CUDNN_RNN_RELU: {
                const float d = g[j] > 0 ? dh : 0;
                storef(a.dgx, gr + j, a.type, d);
                storef(a.dgh, gr + j, a.type, d);
                break;
            }
            default: {
                const float d = dh * (1 - g[j] * g[j]);
                storef(a.dgx, gr + j, a.type, d);
                storef(a.dgh, gr + j, a.type, d);
                break;
            }
        }
    }
}

// dh (of active sequences) = R^T dgh + direct part; inactive sequences keep their dh
__global__ void stateGradient(Sequences seq, int t, int hidden, const float *product, const float *direct,
                              float *dh) {
    GRID_LOOP(e, static_cast<long long>(seq.batch) * hidden) {
        const int b = int(e / hidden);
        if (rowOf(seq, t, b) < 0) continue;
        dh[e] = product[e] + (direct != nullptr ? direct[e] : 0);
    }
}

// gathers the rows of the active sequences of step t (gate gradients) into [b][cols] (zeros for inactive ones)
__global__ void gatherStep(Sequences seq, int t, int cols, const void *src, cudnnDataType_t type, void *dst) {
    GRID_LOOP(e, static_cast<long long>(seq.batch) * cols) {
        const int b = int(e / cols), k = int(e % cols);
        const long long row = rowOf(seq, t, b);
        storef(dst, e, type, row >= 0 ? loadf(src, row * cols + k, type) : 0.0f);
    }
}

// the previous states of all rows (for the recurrent weight gradient): [row][hidden]
__global__ void previousStates(Sequences seq, int hidden, int dir, const void *layerOut, long long outLd,
                               const void *hx, cudnnDataType_t type, void *dst) {
    GRID_LOOP(e, static_cast<long long>(seq.maxLength) * seq.batch * hidden) {
        const int j = int(e % hidden), b = int((e / hidden) % seq.batch), t = int(e / (hidden * seq.batch));
        const long long row = rowOf(seq, t, b);
        if (row < 0) continue;
        const int t_prev = dir == 0 ? t - 1 : t + 1;
        const long long prev = t_prev >= 0 ? rowOf(seq, t_prev, b) : -1;
        float v = 0;
        if (prev >= 0) {
            v = loadf(layerOut, prev * outLd + dir * hidden + j, type);
        } else if (hx != nullptr) {
            v = loadf(hx, static_cast<long long>(b) * hidden + j, type);
        }
        storef(dst, row * hidden + j, type, v);
    }
}

// db[k] (+)= sum over rows of g[row][k]
__global__ void biasGradient(long long rows, int cols, const void *g, cudnnDataType_t type, void *db, bool add) {
    GRID_LOOP(k, cols) {
        float sum = 0;
        for (long long r = 0; r < rows; r++) sum += loadf(g, r * cols + k, type);
        storef(db, k, type, add ? loadf(db, k, type) + sum : sum);
    }
}

// rows not produced by the RNN (padding of unpacked layouts) are zero
__global__ void zeroPadding(Sequences seq, int cols, void *data, cudnnDataType_t type) {
    if (seq.layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED) return;
    GRID_LOOP(e, static_cast<long long>(seq.maxLength) * seq.batch * cols) {
        const int k = int(e % cols), b = int((e / cols) % seq.batch), t = int(e / (cols * seq.batch));
        if (t < seq.lengths[b]) continue;
        const long long row = seq.layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
                              ? static_cast<long long>(b) * seq.maxLength + t
                              : static_cast<long long>(t) * seq.batch + b;
        storef(data, row * cols + k, type, 0.0f);
    }
}

// dropout between layers: keep with probability 1 - p (scaled), from a hash of (seed, layer, index)
__device__ inline bool keep(unsigned long long seed, int layer, long long index, float p) {
    unsigned long long x = seed ^ (static_cast<unsigned long long>(layer) << 48) ^ static_cast<unsigned long long>(index);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (x >> 40) * (1.0f / (1 << 24)) >= p;
}

__global__ void dropoutKernel(long long count, void *data, cudnnDataType_t type, unsigned long long seed, int layer,
                              float p) {
    GRID_LOOP(e, count) {
        storef(data, e, type, keep(seed, layer, e, p) ? loadf(data, e, type) / (1 - p) : 0.0f);
    }
}

// ---- buffers

struct Buffers {
    long long rows;
    int dirs, gates, hidden;
    // per layer: output (data type, rows x hidden*dirs), dropped output; per layer and direction: gates (float),
    // cells (float), GRU recurrent n (float), gate gradients (data type, input and recurrent)
    std::vector<char *> out, dropped, gates_, cells, recurrentN, dgx, dgh;
    // per call (workspace): states, step products, sequence info
    char *h, *c, *gh, *dh, *dc, *product, *direct, *stepGrad, *prev, *dxTemp;
    int *lengths, *offsets;
    size_t reserve, workspace;
};

size_t align(size_t bytes) {
    return (bytes + 255) / 256 * 256;
}

// lays out the reserve space (training) and the workspace; null bases only compute the sizes
Buffers buffersOf(const cudnnRNNStruct &d, const cudnnRNNDataStruct &x, bool training, char *reserve,
                  char *workspace) {
    Buffers b{};
    b.rows = rowCount(x);
    b.dirs = d.direction == CUDNN_BIDIRECTIONAL ? 2 : 1;
    b.gates = gatesOf(d.cell);
    b.hidden = d.hiddenSize;
    const size_t element = dataTypeSize(d.type);
    const long long rows = b.rows, h = b.hidden, gates = b.gates, n = x.batch;
    // (in inference, the per layer buffers live in the workspace and are reused across layers)
    char *base = training ? reserve : workspace;
    size_t offset = 0;
    auto take = [&](size_t bytes) {
        char *p = base != nullptr ? base + offset : nullptr;
        offset += align(bytes);
        return p;
    };
    const int stored_layers = training ? d.layers : std::min(d.layers, 2);
    for (int layer = 0; layer < d.layers; layer++) {
        const bool fresh = layer < stored_layers;
        b.out.push_back(fresh ? take(rows * h * b.dirs * element) : b.out[layer % 2]);
        b.dropped.push_back(training && d.dropout > 0 && layer + 1 < d.layers ? take(rows * h * b.dirs * element)
                                                                              : nullptr);
        for (int dir = 0; dir < b.dirs; dir++) {
            const bool fresh_dir = training || layer == 0;
            const int index = layer * b.dirs + dir;
            b.gates_.push_back(fresh_dir ? take(rows * gates * h * sizeof(float)) : b.gates_[dir]);
            b.cells.push_back(d.cell == CUDNN_LSTM && fresh_dir ? take(rows * h * sizeof(float))
                                                               : d.cell == CUDNN_LSTM ? b.cells[dir] : nullptr);
            b.recurrentN.push_back(d.cell == CUDNN_GRU && fresh_dir ? take(rows * h * sizeof(float))
                                                                   : d.cell == CUDNN_GRU ? b.recurrentN[dir] : nullptr);
            b.dgx.push_back(training ? take(rows * gates * h * element) : nullptr);
            b.dgh.push_back(training ? take(rows * gates * h * element) : nullptr);
            (void) index;
        }
    }
    if (training) {
        b.reserve = offset;
        base = workspace;
        offset = 0;
    }
    b.h = take(n * h * element);
    b.c = take(n * h * sizeof(float));
    b.gh = take(n * gates * h * sizeof(float));
    b.dh = take(n * h * sizeof(float));
    b.dc = take(n * h * sizeof(float));
    b.product = take(n * h * sizeof(float));
    b.direct = take(n * h * sizeof(float));
    b.stepGrad = take(n * gates * h * element);
    b.prev = take(rows * h * element);
    b.dxTemp = take(rows * std::max<long long>(d.inputSize, h * b.dirs) * element);
    b.lengths = reinterpret_cast<int *>(take(n * sizeof(int)));
    b.offsets = reinterpret_cast<int *>(take((x.maxLength + 1) * sizeof(int)));
    b.workspace = offset;
    if (!training) {
        b.reserve = 0;
    }
    return b;
}

cudnnStatus_t uploadSequences(cudaStream_t stream, const cudnnRNNDataStruct &x, const Buffers &b, Sequences &seq) {
    std::vector<int> offsets(x.maxLength + 1, 0);
    for (int t = 0; t < x.maxLength; t++) {
        int active = 0;
        for (int length: x.lengths) active += length > t;
        offsets[t + 1] = offsets[t] + active;
    }
    if (cudaMemcpyAsync(b.lengths, x.lengths.data(), x.lengths.size() * sizeof(int), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(b.offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream) !=
        cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        return CUDNN_STATUS_EXECUTION_FAILED;
    }
    seq = Sequences{x.layout, x.maxLength, x.batch, b.lengths, b.offsets};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t check(const cudnnRNNStruct *d, const cudnnRNNDataStruct *x) {
    if (d == nullptr || x == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (d->projSize != d->hiddenSize || d->input != CUDNN_LINEAR_INPUT) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    if (static_cast<int>(x->lengths.size()) != x->batch) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

#define DNN_TRY(expr) do { cudnnStatus_t dnn_try_status = (expr); if (dnn_try_status != CUDNN_STATUS_SUCCESS) { return dnn_try_status; } } while (0)

cudnnStatus_t launched() {
    return cudaGetLastError() == cudaSuccess ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_EXECUTION_FAILED;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// descriptors

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
    if (rnnDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *rnnDesc = new cudnnRNNStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    delete rnnDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t, cudnnRNNMode_t cellMode,
                                       cudnnRNNBiasMode_t biasMode, cudnnDirectionMode_t dirMode,
                                       cudnnRNNInputMode_t inputMode, cudnnDataType_t dataType, cudnnDataType_t,
                                       cudnnMathType_t mathType, int32_t inputSize, int32_t hiddenSize, int32_t projSize,
                                       int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc, uint32_t) {
    if (rnnDesc == nullptr || inputSize <= 0 || hiddenSize <= 0 || numLayers <= 0) return CUDNN_STATUS_BAD_PARAM;
    *rnnDesc = cudnnRNNStruct{cellMode, biasMode, dirMode, inputMode, dataType, inputSize, hiddenSize,
                              projSize > 0 ? projSize : hiddenSize, numLayers,
                              dropoutDesc != nullptr ? dropoutDesc->dropout : 0.0f,
                              dropoutDesc != nullptr ? dropoutDesc->seed : 0,
                              mathType != CUDNN_FMA_MATH && tf32Permitted()};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
    if (rnnDataDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *rnnDataDesc = new cudnnRNNDataStruct{};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
    delete rnnDataDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType,
                                        cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize, int vectorSize,
                                        const int seqLengthArray[], void *) {
    if (rnnDataDesc == nullptr || maxSeqLength <= 0 || batchSize <= 0 || vectorSize <= 0 ||
        seqLengthArray == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    rnnDataDesc->type = dataType;
    rnnDataDesc->layout = layout;
    rnnDataDesc->maxLength = maxSeqLength;
    rnnDataDesc->batch = batchSize;
    rnnDataDesc->vectorSize = vectorSize;
    rnnDataDesc->lengths.assign(seqLengthArray, seqLengthArray + batchSize);
    for (int length: rnnDataDesc->lengths) {
        if (length < 0 || length > maxSeqLength) return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, size_t *weightSpaceSize) {
    if (!valid(handle) || rnnDesc == nullptr || weightSpaceSize == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *weightSpaceSize = weightLayout(*rnnDesc).total * dataTypeSize(rnnDesc->type);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int32_t pseudoLayer,
                                      size_t, const void *weightSpace, int32_t linLayerID,
                                      cudnnTensorDescriptor_t mDesc, void **mAddr, cudnnTensorDescriptor_t bDesc,
                                      void **bAddr) {
    if (!valid(handle) || rnnDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    const cudnnRNNStruct &d = *rnnDesc;
    const int dirs = d.direction == CUDNN_BIDIRECTIONAL ? 2 : 1, gates = gatesOf(d.cell);
    if (pseudoLayer < 0 || pseudoLayer >= d.layers * dirs || linLayerID < 0 || linLayerID >= 2 * gates) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const WeightLayout l = weightLayout(d);
    const bool recurrent = linLayerID >= gates;
    const int gate = linLayerID % gates;
    const long long h = d.hiddenSize, in = recurrent ? h : layerInputSize(d, pseudoLayer / dirs);
    const size_t element = dataTypeSize(d.type);
    auto describe = [&](cudnnTensorDescriptor_t desc, long long rows, long long cols) {
        if (desc != nullptr) {
            desc->tensor = Tensor{3, {1, rows, cols}, {rows * cols, cols, 1}, d.type};
        }
    };
    const long long matrix = (recurrent ? l.r : l.w)[pseudoLayer] + gate * h * in;
    describe(mDesc, h, in);
    if (mAddr != nullptr) {
        *mAddr = weightSpace != nullptr ? const_cast<char *>(static_cast<const char *>(weightSpace)) + matrix * element
                                        : nullptr;
    }
    const long long bias_base = (recurrent ? l.br : l.bw)[pseudoLayer];
    describe(bDesc, bias_base >= 0 ? h : 0, 1);
    if (bAddr != nullptr) {
        *bAddr = bias_base >= 0 && weightSpace != nullptr
                 ? const_cast<char *>(static_cast<const char *>(weightSpace)) + (bias_base + gate * h) * element
                 : nullptr;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fwdMode,
                                        cudnnRNNDataDescriptor_t xDesc, size_t *workSpaceSize,
                                        size_t *reserveSpaceSize) {
    if (!valid(handle)) return CUDNN_STATUS_BAD_PARAM;
    DNN_TRY(check(rnnDesc, xDesc));
    const Buffers b = buffersOf(*rnnDesc, *xDesc, fwdMode == CUDNN_FWD_MODE_TRAINING, nullptr, nullptr);
    if (workSpaceSize != nullptr) *workSpaceSize = b.workspace;
    if (reserveSpaceSize != nullptr) *reserveSpaceSize = b.reserve;
    return CUDNN_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// forward and backward

cudnnStatus_t cudnnRNNForward(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fwdMode,
                              const int32_t[], cudnnRNNDataDescriptor_t xDesc, const void *x,
                              cudnnRNNDataDescriptor_t yDesc, void *y, cudnnTensorDescriptor_t, const void *hx,
                              void *hy, cudnnTensorDescriptor_t, const void *cx, void *cy, size_t,
                              const void *weightSpace, size_t, void *workSpace, size_t, void *reserveSpace) {
    if (!valid(handle) || x == nullptr || y == nullptr || weightSpace == nullptr || yDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    DNN_TRY(check(rnnDesc, xDesc));
    const cudnnRNNStruct &d = *rnnDesc;
    const bool training = fwdMode == CUDNN_FWD_MODE_TRAINING;
    const cudaStream_t stream = handle->stream;
    Buffers b = buffersOf(d, *xDesc, training, static_cast<char *>(reserveSpace), static_cast<char *>(workSpace));
    Sequences seq{};
    DNN_TRY(uploadSequences(stream, *xDesc, b, seq));
    const WeightLayout wl = weightLayout(d);
    const size_t element = dataTypeSize(d.type);
    const char *weights = static_cast<const char *>(weightSpace);
    const long long h = d.hiddenSize, gh = b.gates * h, n = xDesc->batch, rows = b.rows;
    for (int layer = 0; layer < d.layers; layer++) {
        const bool last = layer + 1 == d.layers;
        // the layer input: x, or the previous layer's (dropped) output
        const void *input = layer == 0 ? x : (b.dropped[layer - 1] != nullptr ? b.dropped[layer - 1] : b.out[layer - 1]);
        const long long in = layerInputSize(d, layer), in_ld = layer == 0 ? xDesc->vectorSize : h * b.dirs;
        void *out = last && !training ? y : b.out[layer];
        const long long out_ld = last && !training ? yDesc->vectorSize : h * b.dirs;
        for (int dir = 0; dir < b.dirs; dir++) {
            const int pl = layer * b.dirs + dir;
            float *gates = reinterpret_cast<float *>(b.gates_[pl]);
            // input projections of all rows: gates (gh x rows) = W^T(gh x in) x^T(in x rows)
            DNN_TRY(gemm(stream, d.type, d.tf32, true, CUBLAS_OP_T, CUBLAS_OP_N, gh, rows, in, weights + wl.w[pl] * element, in,
                         input, in_ld, gates, gh, 0));
            // initial states
            const long long state_offset = static_cast<long long>(pl) * n * h;
            if (hx != nullptr) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), static_cast<const char *>(hx) +
                                                             state_offset * element, h, d.type, b.h, h, d.type);
            } else {
                cudaMemsetAsync(b.h, 0, n * h * element, stream);
            }
            if (d.cell == CUDNN_LSTM) {
                if (cx != nullptr) {
                    copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), static_cast<const char *>(cx) +
                                                                 state_offset * element, h, d.type, b.c, h,
                                                                 CUDNN_DATA_FLOAT);
                } else {
                    cudaMemsetAsync(b.c, 0, n * h * sizeof(float), stream);
                }
            }
            StepArgs a{d.cell, d.type, int(h), b.gates, 0, dir, b.dirs, seq, gates,
                       reinterpret_cast<float *>(b.gh), wl.bw[pl] >= 0 ? weights + wl.bw[pl] * element : nullptr,
                       wl.br[pl] >= 0 ? weights + wl.br[pl] * element : nullptr, b.h, reinterpret_cast<float *>(b.c),
                       reinterpret_cast<float *>(b.cells[pl]), reinterpret_cast<float *>(b.recurrentN[pl]), out,
                       out_ld};
            for (int step = 0; step < xDesc->maxLength; step++) {
                a.t = dir == 0 ? step : xDesc->maxLength - 1 - step;
                // recurrent projections: gh (gh x n) = R^T(gh x h) h^T(h x n)
                DNN_TRY(gemm(stream, d.type, d.tf32, true, CUBLAS_OP_T, CUBLAS_OP_N, gh, n, h, weights + wl.r[pl] * element, h,
                             b.h, h, b.gh, gh, 0));
                cellForward<<<gridFor(n * h), 256, 0, stream>>>(a);
            }
            if (hy != nullptr) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), b.h, h, d.type,
                                                             static_cast<char *>(hy) + state_offset * element, h,
                                                             d.type);
            }
            if (cy != nullptr && d.cell == CUDNN_LSTM) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), b.c, h, CUDNN_DATA_FLOAT,
                                                             static_cast<char *>(cy) + state_offset * element, h,
                                                             d.type);
            }
        }
        zeroPadding<<<gridFor(rows * h * b.dirs), 256, 0, stream>>>(seq, int(h * b.dirs), out, d.type);
        if (training && b.dropped[layer] != nullptr) {
            cudaMemcpyAsync(b.dropped[layer], b.out[layer], rows * h * b.dirs * element, cudaMemcpyDeviceToDevice,
                            stream);
            dropoutKernel<<<gridFor(rows * h * b.dirs), 256, 0, stream>>>(rows * h * b.dirs, b.dropped[layer], d.type,
                                                                          d.seed, layer, d.dropout);
        }
        if (last && training) {
            copyRows<<<gridFor(rows * h * b.dirs), 256, 0, stream>>>(rows, int(h * b.dirs), b.out[layer], h * b.dirs,
                                                                     d.type, y, yDesc->vectorSize, d.type);
        }
    }
    return launched();
}

cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int32_t[],
                                      cudnnRNNDataDescriptor_t yDesc, const void *, const void *dy,
                                      cudnnRNNDataDescriptor_t xDesc, void *dx, cudnnTensorDescriptor_t,
                                      const void *hx, const void *dhy, void *dhx, cudnnTensorDescriptor_t,
                                      const void *cx, const void *dcy, void *dcx, size_t, const void *weightSpace,
                                      size_t, void *workSpace, size_t, void *reserveSpace) {
    if (!valid(handle) || dy == nullptr || dx == nullptr || weightSpace == nullptr || reserveSpace == nullptr ||
        yDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    DNN_TRY(check(rnnDesc, xDesc));
    const cudnnRNNStruct &d = *rnnDesc;
    const cudaStream_t stream = handle->stream;
    Buffers b = buffersOf(d, *xDesc, true, static_cast<char *>(reserveSpace), static_cast<char *>(workSpace));
    Sequences seq{};
    DNN_TRY(uploadSequences(stream, *xDesc, b, seq));
    const WeightLayout wl = weightLayout(d);
    const size_t element = dataTypeSize(d.type);
    const char *weights = static_cast<const char *>(weightSpace);
    const long long h = d.hiddenSize, gh = b.gates * h, n = xDesc->batch, rows = b.rows;
    // gradient of the current layer's output rows: dy, then the input gradients of the layer above
    const void *dout = dy;
    long long dout_ld = yDesc->vectorSize;
    for (int layer = d.layers - 1; layer >= 0; layer--) {
        for (int dir = 0; dir < b.dirs; dir++) {
            const int pl = layer * b.dirs + dir;
            const long long state_offset = static_cast<long long>(pl) * n * h;
            float *dh = reinterpret_cast<float *>(b.dh), *dc = reinterpret_cast<float *>(b.dc);
            if (dhy != nullptr) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), static_cast<const char *>(dhy) +
                                                             state_offset * element, h, d.type, dh, h,
                                                             CUDNN_DATA_FLOAT);
            } else {
                cudaMemsetAsync(dh, 0, n * h * sizeof(float), stream);
            }
            if (dcy != nullptr && d.cell == CUDNN_LSTM) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), static_cast<const char *>(dcy) +
                                                             state_offset * element, h, d.type, dc, h,
                                                             CUDNN_DATA_FLOAT);
            } else {
                cudaMemsetAsync(dc, 0, n * h * sizeof(float), stream);
            }
            StepBackArgs a{d.cell, d.type, int(h), b.gates, 0, dir, seq,
                           reinterpret_cast<const float *>(b.gates_[pl]), reinterpret_cast<const float *>(b.cells[pl]),
                           reinterpret_cast<const float *>(b.recurrentN[pl]),
                           hx != nullptr ? static_cast<const char *>(hx) + state_offset * element : nullptr,
                           cx != nullptr ? static_cast<const char *>(cx) + state_offset * element : nullptr,
                           b.out[layer], h * b.dirs, dout, dout_ld, dh, dc, b.dgx[pl], b.dgh[pl],
                           reinterpret_cast<float *>(b.direct)};
            for (int step = 0; step < xDesc->maxLength; step++) {
                a.t = dir == 0 ? xDesc->maxLength - 1 - step : step;
                cellBackward<<<gridFor(n * h), 256, 0, stream>>>(a);
                // dh_prev (h x n) = R(h x gh, as stored) dgh_step(gh x n)
                gatherStep<<<gridFor(n * gh), 256, 0, stream>>>(seq, a.t, int(gh), b.dgh[pl], d.type, b.stepGrad);
                DNN_TRY(gemm(stream, d.type, d.tf32, true, CUBLAS_OP_N, CUBLAS_OP_N, h, n, gh, weights + wl.r[pl] * element, h,
                             b.stepGrad, gh, b.product, h, 0));
                stateGradient<<<gridFor(n * h), 256, 0, stream>>>(seq, a.t, int(h),
                                                                  reinterpret_cast<const float *>(b.product),
                                                                  d.cell == CUDNN_GRU
                                                                  ? reinterpret_cast<const float *>(b.direct) : nullptr,
                                                                  dh);
            }
            if (dhx != nullptr) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), dh, h, CUDNN_DATA_FLOAT,
                                                             static_cast<char *>(dhx) + state_offset * element, h,
                                                             d.type);
            }
            if (dcx != nullptr && d.cell == CUDNN_LSTM) {
                copyRows<<<gridFor(n * h), 256, 0, stream>>>(n, int(h), dc, h, CUDNN_DATA_FLOAT,
                                                             static_cast<char *>(dcx) + state_offset * element, h,
                                                             d.type);
            }
        }
        // the input gradient: (in x rows) = sum over directions W(in x gh, as stored) dgx(gh x rows)
        const long long in = layerInputSize(d, layer);
        void *din = layer == 0 ? dx : b.dxTemp;
        const long long din_ld = layer == 0 ? xDesc->vectorSize : in;
        for (int dir = 0; dir < b.dirs; dir++) {
            const int pl = layer * b.dirs + dir;
            DNN_TRY(gemm(stream, d.type, d.tf32, false, CUBLAS_OP_N, CUBLAS_OP_N, in, rows, gh, weights + wl.w[pl] * element,
                         in, b.dgx[pl], gh, din, din_ld, dir == 0 ? 0.0f : 1.0f));
        }
        if (layer > 0) {
            if (b.dropped[layer - 1] != nullptr) {
                dropoutKernel<<<gridFor(rows * in), 256, 0, stream>>>(rows * in, b.dxTemp, d.type, d.seed, layer - 1,
                                                                      d.dropout);
            }
            // (the output gradient of the layer below; the next layer down writes its input gradient here only after
            // its steps have read it)
            dout = b.dxTemp;
            dout_ld = in;
        }
        zeroPadding<<<gridFor(rows * in), 256, 0, stream>>>(seq, int(in), din, d.type);
    }
    return launched();
}

cudnnStatus_t cudnnRNNBackwardWeights_v8(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnWgradMode_t addGrad,
                                         const int32_t[], cudnnRNNDataDescriptor_t xDesc, const void *x,
                                         cudnnTensorDescriptor_t, const void *hx, cudnnRNNDataDescriptor_t,
                                         const void *, size_t, void *dweightSpace, size_t, void *workSpace, size_t,
                                         void *reserveSpace) {
    if (!valid(handle) || x == nullptr || dweightSpace == nullptr || reserveSpace == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    DNN_TRY(check(rnnDesc, xDesc));
    const cudnnRNNStruct &d = *rnnDesc;
    const cudaStream_t stream = handle->stream;
    Buffers b = buffersOf(d, *xDesc, true, static_cast<char *>(reserveSpace), static_cast<char *>(workSpace));
    Sequences seq{};
    DNN_TRY(uploadSequences(stream, *xDesc, b, seq));
    const WeightLayout wl = weightLayout(d);
    const size_t element = dataTypeSize(d.type);
    char *dweights = static_cast<char *>(dweightSpace);
    const long long h = d.hiddenSize, gh = b.gates * h, n = xDesc->batch, rows = b.rows;
    const bool add = addGrad == CUDNN_WGRAD_MODE_ADD;
    for (int layer = 0; layer < d.layers; layer++) {
        const void *input = layer == 0 ? x : (b.dropped[layer - 1] != nullptr ? b.dropped[layer - 1] : b.out[layer - 1]);
        const long long in = layerInputSize(d, layer), in_ld = layer == 0 ? xDesc->vectorSize : h * b.dirs;
        for (int dir = 0; dir < b.dirs; dir++) {
            const int pl = layer * b.dirs + dir;
            // dW (in x gh, as stored) (+)= input^T(in x rows) dgx^T(rows x gh)
            DNN_TRY(gemm(stream, d.type, d.tf32, false, CUBLAS_OP_N, CUBLAS_OP_T, in, gh, rows, input, in_ld, b.dgx[pl], gh,
                         dweights + wl.w[pl] * element, in, add ? 1.0f : 0.0f));
            // dR (h x gh) (+)= previous states(h x rows) dgh^T
            const long long state_offset = static_cast<long long>(pl) * n * h;
            cudaMemsetAsync(b.prev, 0, rows * h * element, stream);
            previousStates<<<gridFor(static_cast<long long>(xDesc->maxLength) * n * h), 256, 0, stream>>>(
                seq, int(h), dir, b.out[layer], h * b.dirs,
                hx != nullptr ? static_cast<const char *>(hx) + state_offset * element : nullptr, d.type, b.prev);
            DNN_TRY(gemm(stream, d.type, d.tf32, false, CUBLAS_OP_N, CUBLAS_OP_T, h, gh, rows, b.prev, h, b.dgh[pl], gh,
                         dweights + wl.r[pl] * element, h, add ? 1.0f : 0.0f));
            if (wl.bw[pl] >= 0) {
                biasGradient<<<gridFor(gh), 256, 0, stream>>>(rows, int(gh), b.dgx[pl], d.type,
                                                              dweights + wl.bw[pl] * element, add);
            }
            if (wl.br[pl] >= 0) {
                biasGradient<<<gridFor(gh), 256, 0, stream>>>(rows, int(gh), b.dgh[pl], d.type,
                                                              dweights + wl.br[pl] * element, add);
            }
        }
    }
    return launched();
}

// ---------------------------------------------------------------------------------------------------------------------
// CTC loss

namespace {

__device__ inline double logAdd(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    return fmax(a, b) + log1p(exp(-fabs(a - b)));
}

/**
 * One block per sequence n: log-softmax normalizers, alpha / beta over the extended label (blanks between labels),
 * cost and gradient. probs: [T][N][C] (activations with softmax normalization, else probabilities).
 */
__global__ void ctcKernel(int T, int N, int C, const float *probs, const int *labels, const int *labelOffsets,
                          const int *labelLengths, const int *inputLengths, bool softmax, float *costs, float *grad,
                          double *alpha, double *beta, double *lse, int maxExtended) {
    const int n = blockIdx.x;
    const int L = labelLengths[n], S = 2 * L + 1, Tn = inputLengths[n];
    const int *label = labels + labelOffsets[n];
    double *a = alpha + static_cast<long long>(n) * T * maxExtended;
    double *bt = beta + static_cast<long long>(n) * T * maxExtended;
    double *norm = lse + static_cast<long long>(n) * T;
    __shared__ double shared[256];
    auto prob = [&](int t, int k) { return probs[(static_cast<long long>(t) * N + n) * C + k]; };
    // log normalizers per time step
    for (int t = 0; t < Tn; t++) {
        double m = -INFINITY;
        if (softmax) {
            for (int k = threadIdx.x; k < C; k += blockDim.x) m = fmax(m, double(prob(t, k)));
            shared[threadIdx.x] = m;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s /= 2) {
                if (threadIdx.x < s) shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
                __syncthreads();
            }
            m = shared[0];
            __syncthreads();
            double sum = 0;
            for (int k = threadIdx.x; k < C; k += blockDim.x) sum += exp(double(prob(t, k)) - m);
            shared[threadIdx.x] = sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s /= 2) {
                if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) norm[t] = m + log(shared[0]);
            __syncthreads();
        } else if (threadIdx.x == 0) {
            norm[t] = 0;
        }
    }
    __syncthreads();
    // log probability of extended label position s at time t
    auto lp = [&](int t, int s) {
        const int k = s % 2 == 0 ? 0 : label[s / 2];
        return softmax ? double(prob(t, k)) - norm[t] : log(fmax(double(prob(t, k)), 1e-300));
    };
    auto skip = [&](int s) { return s % 2 == 1 && s >= 2 && label[s / 2] != label[s / 2 - 1]; };
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        a[s] = s < 2 && Tn > 0 ? lp(0, s) : -INFINITY;
    }
    __syncthreads();
    for (int t = 1; t < Tn; t++) {
        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            double v = a[(t - 1) * maxExtended + s];
            if (s >= 1) v = logAdd(v, a[(t - 1) * maxExtended + s - 1]);
            if (skip(s)) v = logAdd(v, a[(t - 1) * maxExtended + s - 2]);
            a[t * maxExtended + s] = v + lp(t, s);
        }
        __syncthreads();
    }
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        bt[(Tn - 1) * maxExtended + s] = s >= S - 2 && Tn > 0 ? lp(Tn - 1, s) : -INFINITY;
    }
    __syncthreads();
    for (int t = Tn - 2; t >= 0; t--) {
        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            double v = bt[(t + 1) * maxExtended + s];
            if (s + 1 < S) v = logAdd(v, bt[(t + 1) * maxExtended + s + 1]);
            if (s + 2 < S && skip(s + 2)) v = logAdd(v, bt[(t + 1) * maxExtended + s + 2]);
            bt[t * maxExtended + s] = v + lp(t, s);
        }
        __syncthreads();
    }
    const double log_likelihood = Tn > 0 ? logAdd(a[(Tn - 1) * maxExtended + S - 1],
                                                  S >= 2 ? a[(Tn - 1) * maxExtended + S - 2] : -INFINITY)
                                         : (L == 0 ? 0 : -INFINITY);
    if (threadIdx.x == 0 && costs != nullptr) costs[n] = float(-log_likelihood);
    if (grad == nullptr) return;
    // gradient: softmax: p - posterior (w.r.t. the activations); probabilities: -posterior / p
    for (int t = 0; t < T; t++) {
        for (int k = threadIdx.x; k < C; k += blockDim.x) {
            float g = 0;
            if (t < Tn && softmax) g = float(exp(double(prob(t, k)) - norm[t]));
            grad[(static_cast<long long>(t) * N + n) * C + k] = g;
        }
    }
    __syncthreads();
    // (blocks of different sequences write different elements; within a block, extended positions may share a class)
    for (int t = 0; t < Tn; t++) {
        for (int s = threadIdx.x; s < S; s += blockDim.x) {
            const int k = s % 2 == 0 ? 0 : label[s / 2];
            // alpha * beta counts lp(t, s) twice
            const double posterior = exp(a[t * maxExtended + s] + bt[t * maxExtended + s] - lp(t, s) -
                                         log_likelihood);
            const float contribution = softmax ? float(-posterior) : float(-posterior / fmax(double(prob(t, k)), 1e-300));
            atomicAdd(&grad[(static_cast<long long>(t) * N + n) * C + k], contribution);
        }
        __syncthreads();
    }
}

size_t ctcWorkspace(int T, int N, int maxLabel) {
    const size_t extended = 2 * static_cast<size_t>(maxLabel) + 1;
    return align(2 * T * N * extended * sizeof(double)) + align(T * N * sizeof(double)) + align(4 * N * sizeof(int)) +
           align(static_cast<size_t>(N) * maxLabel * sizeof(int));
}

cudnnStatus_t ctcRun(cudaStream_t stream, const cudnnCTCLossStruct &desc, const Tensor &probs, const void *probsData,
                     const int *labels, const int *labelLengths, const int *inputLengths, void *costs,
                     void *gradients, void *workspace, size_t workspaceSize) {
    if (probs.nbDims != 3 || probs.type != CUDNN_DATA_FLOAT) {
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    const int T = int(probs.dims[0]), N = int(probs.dims[1]), C = int(probs.dims[2]);
    std::vector<int> offsets(N + 1, 0), all_labels;
    int max_label = 0;
    for (int n = 0; n < N; n++) {
        offsets[n + 1] = offsets[n] + labelLengths[n];
        max_label = std::max(max_label, labelLengths[n]);
    }
    all_labels.assign(labels, labels + offsets[N]);
    if (max_label > std::max(desc.maxLabelLength, 256) || workspaceSize < ctcWorkspace(T, N, max_label)) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    char *ws = static_cast<char *>(workspace);
    const size_t extended = 2 * static_cast<size_t>(max_label) + 1;
    auto *alpha = reinterpret_cast<double *>(ws);
    auto *beta = alpha + T * N * extended;
    ws += align(2 * T * N * extended * sizeof(double));
    auto *lse = reinterpret_cast<double *>(ws);
    ws += align(T * N * sizeof(double));
    int *ints = reinterpret_cast<int *>(ws);
    ws += align(4 * N * sizeof(int));
    int *device_labels = reinterpret_cast<int *>(ws);
    std::vector<int> header(4 * N, 0);
    for (int n = 0; n < N; n++) {
        header[n] = offsets[n];
        header[N + n] = labelLengths[n];
        header[2 * N + n] = inputLengths[n];
    }
    if (cudaMemcpyAsync(ints, header.data(), header.size() * sizeof(int), cudaMemcpyHostToDevice, stream) !=
        cudaSuccess ||
        (!all_labels.empty() && cudaMemcpyAsync(device_labels, all_labels.data(), all_labels.size() * sizeof(int),
                                                cudaMemcpyHostToDevice, stream) != cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return CUDNN_STATUS_EXECUTION_FAILED;
    }
    ctcKernel<<<N, 256, 0, stream>>>(T, N, C, static_cast<const float *>(probsData), device_labels, ints, ints + N,
                                     ints + 2 * N, desc.normalization == CUDNN_LOSS_NORMALIZATION_SOFTMAX,
                                     static_cast<float *>(costs), static_cast<float *>(gradients), alpha, beta, lse,
                                     int(extended));
    return launched();
}

std::vector<int> download(const int *device, size_t count, cudaStream_t stream) {
    std::vector<int> host(count);
    if (count > 0) {
        cudaMemcpyAsync(host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    return host;
}

} // namespace

cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
    if (ctcLossDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *ctcLossDesc = new cudnnCTCLossStruct{CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, 256};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
    delete ctcLossDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
                                          cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t) {
    if (ctcLossDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *ctcLossDesc = cudnnCTCLossStruct{compType, normMode, 256};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v9(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
                                           cudnnLossNormalizationMode_t normMode, cudnnCTCGradMode_t,
                                           int maxLabelLength) {
    if (ctcLossDesc == nullptr) return CUDNN_STATUS_BAD_PARAM;
    *ctcLossDesc = cudnnCTCLossStruct{compType, normMode, maxLabelLength > 0 ? maxLabelLength : 256};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
                                           const cudnnTensorDescriptor_t, const int *labels, const int *labelLengths,
                                           const int *, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t,
                                           size_t *sizeInBytes) {
    if (!valid(handle) || probsDesc == nullptr || labelLengths == nullptr || sizeInBytes == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    (void) labels;
    const Tensor &p = probsDesc->tensor;
    int max_label = 0;
    for (int n = 0; n < p.dims[1]; n++) max_label = std::max(max_label, labelLengths[n]);
    *sizeInBytes = ctcWorkspace(int(p.dims[0]), int(p.dims[1]), max_label);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t,
                                              cudnnCTCLossDescriptor_t ctcLossDesc,
                                              const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t,
                                              size_t *sizeInBytes) {
    if (!valid(handle) || ctcLossDesc == nullptr || probsDesc == nullptr || sizeInBytes == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const Tensor &p = probsDesc->tensor;
    *sizeInBytes = ctcWorkspace(int(p.dims[0]), int(p.dims[1]), ctcLossDesc->maxLabelLength);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const void *probs,
                           const int hostLabels[], const int hostLabelLengths[], const int hostInputLengths[],
                           void *costs, const cudnnTensorDescriptor_t, void *gradients, cudnnCTCLossAlgo_t,
                           cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, size_t workSpaceSizeInBytes) {
    if (!valid(handle) || probsDesc == nullptr || ctcLossDesc == nullptr || hostLabels == nullptr ||
        hostLabelLengths == nullptr || hostInputLengths == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    return ctcRun(handle->stream, *ctcLossDesc, probsDesc->tensor, probs, hostLabels, hostLabelLengths,
                  hostInputLengths, costs, gradients, workspace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t ctcLossDesc,
                              const cudnnTensorDescriptor_t probsDesc, const void *probs, const int labels[],
                              const int labelLengths[], const int inputLengths[], void *costs,
                              const cudnnTensorDescriptor_t, void *gradients, size_t workSpaceSizeInBytes,
                              void *workspace) {
    if (!valid(handle) || probsDesc == nullptr || ctcLossDesc == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    // (the lengths and labels are on the device here)
    const int N = int(probsDesc->tensor.dims[1]);
    const std::vector<int> label_lengths = download(labelLengths, N, handle->stream);
    const std::vector<int> input_lengths = download(inputLengths, N, handle->stream);
    size_t total = 0;
    for (int length: label_lengths) total += length;
    const std::vector<int> host_labels = download(labels, total, handle->stream);
    return ctcRun(handle->stream, *ctcLossDesc, probsDesc->tensor, probs, host_labels.data(), label_lengths.data(),
                  input_lengths.data(), costs, gradients, workspace, workSpaceSizeInBytes);
}
