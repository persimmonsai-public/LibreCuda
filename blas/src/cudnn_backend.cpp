/**
 * LibreCUDA's cuDNN, the backend (graph) API that cudnn-frontend builds on: descriptors are attribute stores; a few
 * attributes are computed (engine heuristics, engines and their knobs / notes, execution plan workspaces). Supported
 * operation graphs: one convolution (forward, backward data or backward filter), run by the convolution core
 * (cudnn_conv.cu). Engines have no knobs: engine 0 multiplies float data exactly, engine 1 (float graphs only) as TF32.
 *
 * LIBRECUDA_CUDNN_DEBUG=1 traces the calls.
 */
#include "dnn_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

using namespace librecuda_dnn;

namespace {

constexpr uint32_t DESCRIPTOR_MAGIC = 0x4c434244;

struct Attribute {
    cudnnBackendAttributeType_t type;
    int64_t count;
    std::vector<uint8_t> data;
};

struct Descriptor {
    uint32_t magic;
    cudnnBackendDescriptorType_t type;
    bool finalized;
    std::map<cudnnBackendAttributeName_t, Attribute> attributes;
    // execution plans: the workspace their operation needs
    int64_t workspace;
};

bool debug() {
    static const bool enabled = getenv("LIBRECUDA_CUDNN_DEBUG") != nullptr;
    return enabled;
}

#define TRACE(...) do { if (debug()) { fprintf(stderr, "[librecuda cudnn] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

Descriptor *descriptorOf(cudnnBackendDescriptor_t d) {
    auto *desc = static_cast<Descriptor *>(d);
    return desc != nullptr && desc->magic == DESCRIPTOR_MAGIC ? desc : nullptr;
}

size_t elementSize(cudnnBackendAttributeType_t type) {
    switch (type) {
        case CUDNN_TYPE_HANDLE: return sizeof(cudnnHandle_t);
        case CUDNN_TYPE_BOOLEAN: return sizeof(bool);
        case CUDNN_TYPE_INT64: return sizeof(int64_t);
        case CUDNN_TYPE_FLOAT: return sizeof(float);
        case CUDNN_TYPE_DOUBLE: return sizeof(double);
        case CUDNN_TYPE_VOID_PTR: return sizeof(void *);
        case CUDNN_TYPE_BACKEND_DESCRIPTOR: return sizeof(cudnnBackendDescriptor_t);
        case CUDNN_TYPE_INT32: return sizeof(int32_t);
        case CUDNN_TYPE_CHAR: return sizeof(char);
        case CUDNN_TYPE_FRACTION: return sizeof(cudnnFraction_t);
        default: return sizeof(int);  // enumerations
    }
}

// typed access to stored attributes
const Attribute *find(const Descriptor *d, cudnnBackendAttributeName_t name) {
    auto it = d->attributes.find(name);
    return it != d->attributes.end() ? &it->second : nullptr;
}

template<typename T>
bool get(const Descriptor *d, cudnnBackendAttributeName_t name, T &out, int64_t index = 0) {
    const Attribute *a = find(d, name);
    if (a == nullptr || index >= a->count || elementSize(a->type) != sizeof(T)) {
        return false;
    }
    memcpy(&out, a->data.data() + index * sizeof(T), sizeof(T));
    return true;
}

template<typename T>
std::vector<T> getAll(const Descriptor *d, cudnnBackendAttributeName_t name) {
    const Attribute *a = find(d, name);
    std::vector<T> out;
    if (a == nullptr || elementSize(a->type) != sizeof(T)) {
        return out;
    }
    out.resize(a->count);
    memcpy(out.data(), a->data.data(), a->count * sizeof(T));
    return out;
}

void set(Descriptor *d, cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t count,
         const void *elements) {
    Attribute a{type, count, std::vector<uint8_t>(count * elementSize(type))};
    if (count > 0) {
        memcpy(a.data.data(), elements, a.data.size());
    }
    d->attributes[name] = std::move(a);
}

Descriptor *referenced(const Descriptor *d, cudnnBackendAttributeName_t name, int64_t index = 0) {
    cudnnBackendDescriptor_t ref = nullptr;
    return get(d, name, ref, index) ? descriptorOf(ref) : nullptr;
}

Descriptor *create(cudnnBackendDescriptorType_t type) {
    return new Descriptor{DESCRIPTOR_MAGIC, type, false, {}, 0};
}

// ---- operation graphs

struct ConvOperation {
    ConvKind kind;
    Descriptor *x, *w, *y, *conv;
    double alpha, beta;
};

double scalarOf(const Descriptor *op, cudnnBackendAttributeName_t name, double fallback) {
    const Attribute *a = find(op, name);
    if (a == nullptr || a->count < 1) {
        return fallback;
    }
    if (a->type == CUDNN_TYPE_DOUBLE) {
        double v;
        memcpy(&v, a->data.data(), sizeof(v));
        return v;
    }
    float v;
    memcpy(&v, a->data.data(), sizeof(v));
    return v;
}

// the single convolution of an operation graph (false: not a supported graph)
bool convolutionOf(const Descriptor *graph, ConvOperation &out) {
    const Attribute *ops = find(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS);
    if (ops == nullptr || ops->count != 1) {
        return false;
    }
    const Descriptor *op = referenced(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS);
    if (op == nullptr) {
        return false;
    }
    switch (op->type) {
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            out = {ConvKind::FORWARD, referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC),
                   scalarOf(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1),
                   scalarOf(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 0)};
            break;
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            out = {ConvKind::BACKWARD_DATA, referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC),
                   scalarOf(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, 1),
                   scalarOf(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, 0)};
            break;
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            out = {ConvKind::BACKWARD_FILTER, referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY),
                   referenced(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC),
                   scalarOf(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, 1),
                   scalarOf(op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, 0)};
            break;
        default:
            return false;
    }
    return out.x != nullptr && out.w != nullptr && out.y != nullptr && out.conv != nullptr;
}

bool tensorOf(const Descriptor *d, Tensor &t, int64_t &uid) {
    const std::vector<int64_t> dims = getAll<int64_t>(d, CUDNN_ATTR_TENSOR_DIMENSIONS);
    const std::vector<int64_t> strides = getAll<int64_t>(d, CUDNN_ATTR_TENSOR_STRIDES);
    if (dims.empty() || dims.size() != strides.size() || dims.size() > MAX_DIMS) {
        return false;
    }
    t = Tensor{};
    t.nbDims = static_cast<int>(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        t.dims[i] = dims[i];
        t.strides[i] = strides[i];
    }
    int type = CUDNN_DATA_FLOAT;
    get(d, CUDNN_ATTR_TENSOR_DATA_TYPE, type);
    t.type = static_cast<cudnnDataType_t>(type);
    uid = 0;
    get(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, uid);
    return true;
}

bool geometryOf(const Descriptor *conv, ConvGeometry &g) {
    g = ConvGeometry{};
    int64_t spatial = 0;
    get(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, spatial);
    const auto pre = getAll<int64_t>(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS);
    const auto post = getAll<int64_t>(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS);
    const auto strides = getAll<int64_t>(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES);
    const auto dilations = getAll<int64_t>(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS);
    if (spatial == 0) {
        spatial = static_cast<int64_t>(strides.size());
    }
    if (spatial < 1 || spatial > 3 || static_cast<int64_t>(pre.size()) < spatial ||
        static_cast<int64_t>(strides.size()) < spatial || static_cast<int64_t>(dilations.size()) < spatial) {
        return false;
    }
    g.spatial = static_cast<int>(spatial);
    for (int i = 0; i < g.spatial; i++) {
        g.padBefore[i] = pre[i];
        g.padAfter[i] = static_cast<int64_t>(post.size()) > i ? post[i] : pre[i];
        g.stride[i] = strides[i];
        g.dilation[i] = dilations[i];
    }
    int mode = CUDNN_CROSS_CORRELATION;
    get(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, mode);
    g.flip = mode == CUDNN_CONVOLUTION;
    int compute = CUDNN_DATA_FLOAT;
    get(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, compute);
    g.compute = static_cast<cudnnDataType_t>(compute);
    return true;
}

struct Problem {
    ConvOperation op;
    Tensor x, w, y;
    int64_t xuid, wuid, yuid;
    ConvGeometry geometry;
};

cudnnStatus_t problemOf(const Descriptor *graph, Problem &p) {
    if (graph == nullptr || !convolutionOf(graph, p.op)) {
        return CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN;
    }
    if (!tensorOf(p.op.x, p.x, p.xuid) || !tensorOf(p.op.w, p.w, p.wuid) || !tensorOf(p.op.y, p.y, p.yuid) ||
        !geometryOf(p.op.conv, p.geometry)) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    return convCheck(p.op.kind, p.x, p.w, p.y, p.geometry);
}

// the operation graph of an engine configuration, engine or execution plan
const Descriptor *graphOf(const Descriptor *d) {
    while (d != nullptr) {
        switch (d->type) {
            case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
                return d;
            case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
                d = referenced(d, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
                break;
            case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
                d = referenced(d, CUDNN_ATTR_ENGINECFG_ENGINE);
                break;
            case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
                d = referenced(d, CUDNN_ATTR_ENGINE_OPERATION_GRAPH);
                break;
            case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
                d = referenced(d, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH);
                break;
            default:
                return nullptr;
        }
    }
    return nullptr;
}

// descriptors this library creates for results (engines, engine configurations): kept for the process
std::mutex g_internalMutex;
std::vector<Descriptor *> g_internal;

Descriptor *internal(cudnnBackendDescriptorType_t type) {
    Descriptor *d = create(type);
    std::lock_guard lock(g_internalMutex);
    g_internal.push_back(d);
    return d;
}

// an engine of the graph, finalized
Descriptor *engineFor(const Descriptor *graph, int64_t index) {
    Descriptor *engine = internal(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    const cudnnBackendDescriptor_t g = const_cast<Descriptor *>(graph);
    set(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &g);
    set(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
    engine->finalized = true;
    return engine;
}

/**
 * The engines of a convolution: 0 (exact float multiplication), and for float data 1, which multiplies as TF32 on
 * tensor cores (numerical note CUDNN_NUMERICAL_NOTE_TENSOR_CORE, which PyTorch filters out when TF32 isn't allowed).
 */
constexpr int64_t TF32_ENGINE = 1;

int64_t engineCount(const Problem &p) {
    return p.x.type == CUDNN_DATA_FLOAT && tf32Permitted() ? 2 : 1;
}

// the global index of the engine of an engine, engine configuration or execution plan (-1 without one)
int64_t engineIndexOf(const Descriptor *d) {
    while (d != nullptr && d->type != CUDNN_BACKEND_ENGINE_DESCRIPTOR) {
        if (d->type == CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) {
            d = referenced(d, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
        } else if (d->type == CUDNN_BACKEND_ENGINECFG_DESCRIPTOR) {
            d = referenced(d, CUDNN_ATTR_ENGINECFG_ENGINE);
        } else {
            return -1;
        }
    }
    int64_t index = -1;
    if (d != nullptr) get(d, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, index);
    return index;
}

// the problem an execution plan runs (with its engine's numerics)
cudnnStatus_t planProblemOf(const Descriptor *plan, Problem &p) {
    const cudnnStatus_t status = problemOf(graphOf(plan), p);
    p.geometry.allowTF32 = engineIndexOf(plan) == TF32_ENGINE;
    return status;
}

// a deep copy of a descriptor (owned by the library): descriptors referenced by attributes are copied when they are
// set, because callers (cudnn-frontend) destroy theirs once the referencing descriptor is finalized
Descriptor *clone(const Descriptor *from) {
    if (from == nullptr) {
        return nullptr;
    }
    Descriptor *copy = internal(from->type);
    copy->finalized = from->finalized;
    copy->attributes = from->attributes;
    copy->workspace = from->workspace;
    return copy;
}

// copies a descriptor into one the caller created (GetAttribute of descriptor attributes)
void copyInto(const Descriptor *from, cudnnBackendDescriptor_t to) {
    Descriptor *target = descriptorOf(to);
    if (from == nullptr || target == nullptr) {
        return;
    }
    target->type = from->type;
    target->finalized = from->finalized;
    target->attributes = from->attributes;
    target->workspace = from->workspace;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// the API

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType,
                                           cudnnBackendDescriptor_t *descriptor) {
    if (descriptor == nullptr) {
        return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
    }
    *descriptor = create(descriptorType);
    TRACE("create %d -> %p", int(descriptorType), *descriptor);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor) {
    Descriptor *d = descriptorOf(descriptor);
    if (d == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    d->magic = 0;
    delete d;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor) {
    Descriptor *d = descriptorOf(descriptor);
    if (d == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    d->attributes.clear();
    d->finalized = false;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
                                       cudnnBackendAttributeType_t attributeType, int64_t elementCount,
                                       const void *arrayOfElements) {
    Descriptor *d = descriptorOf(descriptor);
    TRACE("set %p (type %d) attribute %d type %d count %lld", descriptor, d ? int(d->type) : -1, int(attributeName),
          int(attributeType), (long long) elementCount);
    if (d == nullptr || elementCount < 0 || (elementCount > 0 && arrayOfElements == nullptr)) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (d->finalized) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (attributeType == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
        // (the referenced descriptors' own references are already copies)
        std::vector<cudnnBackendDescriptor_t> copies(elementCount);
        for (int64_t i = 0; i < elementCount; i++) {
            const auto *refs = static_cast<const cudnnBackendDescriptor_t *>(arrayOfElements);
            copies[i] = clone(descriptorOf(refs[i]));
        }
        set(d, attributeName, attributeType, elementCount, copies.data());
        return CUDNN_STATUS_SUCCESS;
    }
    set(d, attributeName, attributeType, elementCount, arrayOfElements);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
    Descriptor *d = descriptorOf(descriptor);
    if (d == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    Problem problem{};
    switch (d->type) {
        case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            status = problemOf(d, problem);
            break;
        case CUDNN_BACKEND_ENGINE_DESCRIPTOR: {
            int64_t index = -1;
            get(d, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, index);
            status = problemOf(graphOf(d), problem);
            if (status == CUDNN_STATUS_SUCCESS && (index < 0 || index >= engineCount(problem))) {
                status = CUDNN_STATUS_NOT_SUPPORTED;
            }
            break;
        }
        case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            status = planProblemOf(d, problem);
            if (status == CUDNN_STATUS_SUCCESS) {
                d->workspace = static_cast<int64_t>(convWorkspaceSize(problem.op.kind, problem.x, problem.w, problem.y,
                                                                      problem.geometry));
            }
            break;
        default:
            break;
    }
    TRACE("finalize %p (type %d): %d", descriptor, int(d->type), int(status));
    if (status == CUDNN_STATUS_SUCCESS) {
        d->finalized = true;
    }
    return status;
}

cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t const descriptor,
                                       cudnnBackendAttributeName_t attributeName,
                                       cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
                                       int64_t *elementCount, void *arrayOfElements) {
    Descriptor *d = descriptorOf(descriptor);
    TRACE("get %p (type %d) attribute %d type %d requested %lld", descriptor, d ? int(d->type) : -1,
          int(attributeName), int(attributeType), (long long) requestedElementCount);
    if (d == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    auto answer = [&](int64_t count, auto fill) {
        if (elementCount != nullptr) {
            *elementCount = count;
        }
        if (arrayOfElements != nullptr) {
            fill(std::min(count, requestedElementCount));
        }
        return CUDNN_STATUS_SUCCESS;
    };
    switch (attributeName) {
        case CUDNN_ATTR_ENGINEHEUR_RESULTS: {
            // an engine configuration per engine of a supported graph, TF32 first (like cuDNN's heuristics)
            const Descriptor *graph = graphOf(d);
            Problem problem{};
            const cudnnStatus_t check = graph != nullptr ? problemOf(graph, problem) : CUDNN_STATUS_BAD_PARAM;
            const int64_t count = check == CUDNN_STATUS_SUCCESS ? engineCount(problem) : 0;
            TRACE("  heuristics: graph %p, status %d", static_cast<const void *>(graph), int(check));
            return answer(count, [&](int64_t n) {
                auto *results = static_cast<cudnnBackendDescriptor_t *>(arrayOfElements);
                for (int64_t i = 0; i < n; i++) {
                    Descriptor *config = descriptorOf(results[i]);
                    if (config == nullptr) continue;
                    config->type = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
                    const cudnnBackendDescriptor_t engine = engineFor(graph, count - 1 - i);
                    set(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
                    config->finalized = true;
                }
            });
        }
        case CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT: {
            Problem problem{};
            const int64_t count = problemOf(d, problem) == CUDNN_STATUS_SUCCESS ? engineCount(problem) : 0;
            return answer(1, [&](int64_t n) { if (n > 0) memcpy(arrayOfElements, &count, sizeof(count)); });
        }
        case CUDNN_ATTR_ENGINE_NUMERICAL_NOTE: {
            const bool tf32 = engineIndexOf(d) == TF32_ENGINE;
            return answer(tf32 ? 1 : 0, [&](int64_t n) {
                const cudnnBackendNumericalNote_t note = CUDNN_NUMERICAL_NOTE_TENSOR_CORE;
                if (n > 0) memcpy(arrayOfElements, &note, sizeof(note));
            });
        }
        case CUDNN_ATTR_ENGINE_KNOB_INFO:
        case CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
        case CUDNN_ATTR_ENGINECFG_KNOB_CHOICES:
        case CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
        case CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
            return answer(0, [](int64_t) {});
        case CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
            return answer(1, [&](int64_t n) { if (n > 0) memcpy(arrayOfElements, &d->workspace, sizeof(int64_t)); });
        case CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION: {
            static const char exact[] = "{\"engine\": 0}", tf32[] = "{\"engine\": 1}";
            const char *json = engineIndexOf(d) == TF32_ENGINE ? tf32 : exact;
            return answer(int64_t(strlen(json)) + 1, [&](int64_t n) { memcpy(arrayOfElements, json, n); });
        }
        case CUDNN_ATTR_ENGINE_SM_COUNT_TARGET: {
            const int32_t zero = 0;
            return answer(1, [&](int64_t n) { if (n > 0) memcpy(arrayOfElements, &zero, sizeof(zero)); });
        }
        default:
            break;
    }
    const Attribute *a = find(d, attributeName);
    if (a == nullptr) {
        TRACE("  attribute %d not set", int(attributeName));
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (a->type != attributeType) {
        return CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE;
    }
    return answer(a->count, [&](int64_t n) {
        if (a->type == CUDNN_TYPE_BACKEND_DESCRIPTOR) {
            // descriptors are copied into the caller's
            for (int64_t i = 0; i < n; i++) {
                copyInto(referenced(d, attributeName, i), static_cast<cudnnBackendDescriptor_t *>(arrayOfElements)[i]);
            }
        } else {
            memcpy(arrayOfElements, a->data.data(), n * elementSize(a->type));
        }
    });
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan,
                                  cudnnBackendDescriptor_t variantPack) {
    if (!valid(handle)) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    const Descriptor *plan = descriptorOf(executionPlan), *pack = descriptorOf(variantPack);
    if (plan == nullptr || pack == nullptr || !plan->finalized) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    Problem p{};
    const cudnnStatus_t status = planProblemOf(plan, p);
    if (status != CUDNN_STATUS_SUCCESS) {
        return status;
    }
    const auto uids = getAll<int64_t>(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    const auto pointers = getAll<void *>(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    void *workspace = nullptr;
    get(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, workspace);
    auto pointerOf = [&](int64_t uid) -> void * {
        for (size_t i = 0; i < uids.size() && i < pointers.size(); i++) {
            if (uids[i] == uid) return pointers[i];
        }
        return nullptr;
    };
    void *x = pointerOf(p.xuid), *w = pointerOf(p.wuid), *y = pointerOf(p.yuid);
    TRACE("execute kind %d: x %p w %p y %p workspace %p (%lld)", int(p.op.kind), x, w, y, workspace,
          (long long) plan->workspace);
    if (x == nullptr || w == nullptr || y == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    return convRun(p.op.kind, handle->stream, p.x, x, p.w, w, p.y, y, p.geometry, p.op.alpha, p.op.beta, workspace,
                   static_cast<size_t>(plan->workspace));
}
