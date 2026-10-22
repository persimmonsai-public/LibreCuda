// Mipmapped arrays: every level of 2D, 3D and layered mipmapped arrays filled through cudaGetMipmappedArrayLevel and
// read back texel by texel with tex2DLod / tex3DLod / tex2DLayeredLod (the hardware finds the levels from level 0's
// size and block size, so this checks the level layout), linear filtering between levels, the maximum level clamp,
// and a surface write into a level.
#include "check.h"
#include <algorithm>
#include <cmath>
#include <vector>

struct Sample {
    float u, v, w, lod;
};

__global__ void sample2D(cudaTextureObject_t t, const Sample *samples, int n, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = tex2DLod<float>(t, samples[i].u, samples[i].v, samples[i].lod);
    }
}

__global__ void sample3D(cudaTextureObject_t t, const Sample *samples, int n, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = tex3DLod<float>(t, samples[i].u, samples[i].v, samples[i].w, samples[i].lod);
    }
}

__global__ void sampleLayered(cudaTextureObject_t t, const Sample *samples, int n, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = tex2DLayeredLod<float>(t, samples[i].u, samples[i].v, int(samples[i].w), samples[i].lod);
    }
}

__global__ void writeLevel(cudaSurfaceObject_t s, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        surf2Dwrite(-1.0f - i, s, (i % width) * sizeof(float), i / width);
    }
}

static float texelValue(int level, int layer, int index) {
    return level * 100000.0f + layer * 10000.0f + index;
}

static cudaTextureObject_t mipTexture(cudaMipmappedArray_t array, cudaTextureFilterMode mipFilter, float maxLevel) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeMipmappedArray;
    res.res.mipmap.mipmap = array;
    cudaTextureDesc tex{};
    tex.addressMode[0] = tex.addressMode[1] = tex.addressMode[2] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalizedCoords = 1;
    tex.mipmapFilterMode = mipFilter;
    tex.maxMipmapLevelClamp = maxLevel;
    cudaTextureObject_t t = 0;
    CHECK(cudaCreateTextureObject(&t, &res, &tex, nullptr));
    return t;
}

// runs the samples through a kernel and returns the results
template<typename Kernel>
static std::vector<float> run(Kernel kernel, cudaTextureObject_t t, const std::vector<Sample> &samples) {
    Sample *d_samples;
    float *out;
    CHECK(cudaMalloc(&d_samples, samples.size() * sizeof(Sample)));
    CHECK(cudaMalloc(&out, samples.size() * sizeof(float)));
    CHECK(cudaMemcpy(d_samples, samples.data(), samples.size() * sizeof(Sample), cudaMemcpyHostToDevice));
    kernel<<<8, 128>>>(t, d_samples, int(samples.size()), out);
    std::vector<float> result(samples.size());
    CHECK(cudaMemcpy(result.data(), out, samples.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_samples));
    CHECK(cudaFree(out));
    return result;
}

int main() {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    // 2D: every texel of every level, a level blend, the maximum level clamp and a surface write into a level
    {
        const int width = 37, height = 21, levels = 6;
        cudaMipmappedArray_t array;
        CHECK(cudaMallocMipmappedArray(&array, &desc, make_cudaExtent(width, height, 0), levels,
                                       cudaArraySurfaceLoadStore));
        std::vector<std::vector<float>> data(levels);
        std::vector<int> widths(levels), heights(levels);
        for (int l = 0; l < levels; l++) {
            widths[l] = std::max(1, width >> l);
            heights[l] = std::max(1, height >> l);
            cudaArray_t level;
            CHECK(cudaGetMipmappedArrayLevel(&level, array, l));
            cudaExtent extent{};
            CHECK(cudaArrayGetInfo(nullptr, &extent, nullptr, level));
            EXPECT(int(extent.width) == widths[l] && int(extent.height) == heights[l], "level %d extent %zu x %zu", l,
                   extent.width, extent.height);
            data[l].resize(widths[l] * heights[l]);
            for (size_t i = 0; i < data[l].size(); i++) {
                data[l][i] = texelValue(l, 0, int(i));
            }
            CHECK(cudaMemcpy2DToArray(level, 0, 0, data[l].data(), widths[l] * sizeof(float),
                                      widths[l] * sizeof(float), heights[l], cudaMemcpyHostToDevice));
        }
        const cudaTextureObject_t t = mipTexture(array, cudaFilterModePoint, levels - 1);
        std::vector<Sample> samples;
        std::vector<float> expected;
        for (int l = 0; l < levels; l++) {
            for (int i = 0; i < widths[l] * heights[l]; i++) {
                const int x = i % widths[l], y = i / widths[l];
                samples.push_back({(x + 0.5f) / widths[l], (y + 0.5f) / heights[l], 0, float(l)});
                expected.push_back(data[l][i]);
            }
        }
        std::vector<float> result = run(sample2D, t, samples);
        for (size_t i = 0; i < samples.size(); i++) {
            EXPECT(result[i] == expected[i], "2D level %g at (%g, %g) = %g, expected %g", samples[i].lod,
                   samples[i].u, samples[i].v, result[i], expected[i]);
        }

        // halfway between levels 1 and 2 with linear filtering between the levels
        const cudaTextureObject_t tl = mipTexture(array, cudaFilterModeLinear, levels - 1);
        samples.clear();
        expected.clear();
        for (int i = 0; i < 40; i++) {
            const float u = (i * 0.618f + 0.1f) - std::floor(i * 0.618f + 0.1f);
            const float v = (i * 0.414f + 0.3f) - std::floor(i * 0.414f + 0.3f);
            samples.push_back({u, v, 0, 1.5f});
            const float a = data[1][std::min(int(v * heights[1]), heights[1] - 1) * widths[1] +
                                    std::min(int(u * widths[1]), widths[1] - 1)];
            const float b = data[2][std::min(int(v * heights[2]), heights[2] - 1) * widths[2] +
                                    std::min(int(u * widths[2]), widths[2] - 1)];
            expected.push_back((a + b) / 2);
        }
        result = run(sample2D, tl, samples);
        for (size_t i = 0; i < samples.size(); i++) {
            EXPECT(std::fabs(result[i] - expected[i]) <= std::fabs(expected[i]) * 1e-3f,
                   "level 1.5 at (%g, %g) = %g, expected %g", samples[i].u, samples[i].v, result[i], expected[i]);
        }

        // the maximum level clamp: level 4 requested, level 2 read
        const cudaTextureObject_t tc = mipTexture(array, cudaFilterModePoint, 2);
        result = run(sample2D, tc, {{0.5f, 0.5f, 0, 4.0f}});
        const float clamped = data[2][(heights[2] / 2) * widths[2] + widths[2] / 2];
        EXPECT(result[0] == clamped, "clamped to level 2: %g, expected %g", result[0], clamped);

        // a surface on level 3, read back through the texture
        cudaArray_t level3;
        CHECK(cudaGetMipmappedArrayLevel(&level3, array, 3));
        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = level3;
        cudaSurfaceObject_t s;
        CHECK(cudaCreateSurfaceObject(&s, &res));
        writeLevel<<<1, 64>>>(s, widths[3], heights[3]);
        samples.clear();
        for (int i = 0; i < widths[3] * heights[3]; i++) {
            samples.push_back({(i % widths[3] + 0.5f) / widths[3], (i / widths[3] + 0.5f) / heights[3], 0, 3.0f});
        }
        result = run(sample2D, t, samples);
        for (size_t i = 0; i < samples.size(); i++) {
            EXPECT(result[i] == -1.0f - i, "surface written level 3 [%zu] = %g", i, result[i]);
        }
        CHECK(cudaDestroySurfaceObject(s));
        for (cudaTextureObject_t texture: {t, tl, tc}) {
            CHECK(cudaDestroyTextureObject(texture));
        }
        CHECK(cudaFreeMipmappedArray(array));
    }

    // 3D: depth halves too (block depth clamped per level)
    {
        const int width = 16, height = 12, depth = 10, levels = 4;
        cudaMipmappedArray_t array;
        CHECK(cudaMallocMipmappedArray(&array, &desc, make_cudaExtent(width, height, depth), levels));
        std::vector<Sample> samples;
        std::vector<float> expected;
        for (int l = 0; l < levels; l++) {
            const int w = std::max(1, width >> l), h = std::max(1, height >> l), d = std::max(1, depth >> l);
            std::vector<float> data(w * h * d);
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = texelValue(l, 0, int(i));
            }
            cudaArray_t level;
            CHECK(cudaGetMipmappedArrayLevel(&level, array, l));
            cudaMemcpy3DParms p{};
            p.srcPtr = make_cudaPitchedPtr(data.data(), w * sizeof(float), w, h);
            p.dstArray = level;
            p.extent = make_cudaExtent(w, h, d);
            p.kind = cudaMemcpyHostToDevice;
            CHECK(cudaMemcpy3D(&p));
            for (int i = 0; i < w * h * d; i++) {
                const int x = i % w, y = i / w % h, z = i / (w * h);
                samples.push_back({(x + 0.5f) / w, (y + 0.5f) / h, (z + 0.5f) / d, float(l)});
                expected.push_back(data[i]);
            }
        }
        const cudaTextureObject_t t = mipTexture(array, cudaFilterModePoint, levels - 1);
        std::vector<float> result = run(sample3D, t, samples);
        for (size_t i = 0; i < samples.size(); i++) {
            EXPECT(result[i] == expected[i], "3D level %g at (%g, %g, %g) = %g, expected %g", samples[i].lod,
                   samples[i].u, samples[i].v, samples[i].w, result[i], expected[i]);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaFreeMipmappedArray(array));
    }

    // layered: layers keep all their levels together (the layer stride covers every level)
    {
        const int width = 20, height = 9, layers = 3, levels = 5;
        cudaMipmappedArray_t array;
        CHECK(cudaMallocMipmappedArray(&array, &desc, make_cudaExtent(width, height, layers), levels,
                                       cudaArrayLayered));
        std::vector<Sample> samples;
        std::vector<float> expected;
        for (int l = 0; l < levels; l++) {
            const int w = std::max(1, width >> l), h = std::max(1, height >> l);
            std::vector<float> data(w * h * layers);
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = texelValue(l, int(i) / (w * h), int(i) % (w * h));
            }
            cudaArray_t level;
            CHECK(cudaGetMipmappedArrayLevel(&level, array, l));
            cudaMemcpy3DParms p{};
            p.srcPtr = make_cudaPitchedPtr(data.data(), w * sizeof(float), w, h);
            p.dstArray = level;
            p.extent = make_cudaExtent(w, h, layers);
            p.kind = cudaMemcpyHostToDevice;
            CHECK(cudaMemcpy3D(&p));
            for (size_t i = 0; i < data.size(); i++) {
                const int x = int(i) % w, y = int(i) / w % h, layer = int(i) / (w * h);
                samples.push_back({(x + 0.5f) / w, (y + 0.5f) / h, float(layer), float(l)});
                expected.push_back(data[i]);
            }
        }
        const cudaTextureObject_t t = mipTexture(array, cudaFilterModePoint, levels - 1);
        std::vector<float> result = run(sampleLayered, t, samples);
        for (size_t i = 0; i < samples.size(); i++) {
            EXPECT(result[i] == expected[i], "layered level %g layer %g at (%g, %g) = %g, expected %g",
                   samples[i].lod, samples[i].w, samples[i].u, samples[i].v, result[i], expected[i]);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaFreeMipmappedArray(array));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("mipmapped_arrays");
    return 0;
}
