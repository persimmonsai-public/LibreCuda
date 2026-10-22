// CUDA arrays (block linear texture memory): allocation, copies from and to pageable / pinned / device memory with
// offsets, array to array copies, and texture objects on 1D, 2D, 3D and layered 1D / 2D arrays with point / linear filtering,
// normalized coordinates and 8, 16 and 32 bit texels (Blackwell has GOB formats of its own for 8 and 16 bit texels).
#include "check.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

template<typename T>
__global__ void fetch2D(cudaTextureObject_t t, int width, int height, float dx, float dy, T *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += gridDim.x * blockDim.x) {
        out[i] = tex2D<T>(t, i % width + dx, i / width + dy);
    }
}

__global__ void fetch3D(cudaTextureObject_t t, int width, int height, int depth, bool normalized, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height * depth; i += gridDim.x * blockDim.x) {
        const float x = i % width + 0.5f, y = i / width % height + 0.5f, z = i / (width * height) + 0.5f;
        // normalized coordinates shifted by one full period: wrap addressing brings them back
        out[i] = normalized ? tex3D<float>(t, x / width + 1.0f, y / height - 1.0f, z / depth + 2.0f)
                            : tex3D<float>(t, x, y, z);
    }
}

__global__ void fetchLayered(cudaTextureObject_t t, int width, int height, int layers, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height * layers; i += gridDim.x * blockDim.x) {
        out[i] = tex2DLayered<float>(t, i % width + 0.5f, i / width % height + 0.5f, i / (width * height));
    }
}

__global__ void fetchLayered1D(cudaTextureObject_t t, int width, int layers, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * layers; i += gridDim.x * blockDim.x) {
        out[i] = tex1DLayered<float>(t, i % width + 0.5f, i / width);
    }
}

__global__ void fetch1D(cudaTextureObject_t t, int width, float *out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x) {
        out[i] = tex1D<float>(t, i + 0.5f);
    }
}

static cudaTextureObject_t arrayTexture(cudaArray_t array, cudaTextureFilterMode filter = cudaFilterModePoint,
                                        cudaTextureReadMode read = cudaReadModeElementType, bool normalized = false,
                                        cudaTextureAddressMode mode = cudaAddressModeClamp) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    cudaTextureDesc tex{};
    tex.addressMode[0] = tex.addressMode[1] = tex.addressMode[2] = mode;
    tex.filterMode = filter;
    tex.readMode = read;
    tex.normalizedCoords = normalized;
    cudaTextureObject_t object = 0;
    CHECK(cudaCreateTextureObject(&object, &res, &tex, nullptr));
    return object;
}

template<typename T>
static std::vector<T> download(const T *device, size_t n) {
    std::vector<T> host(n);
    CHECK(cudaMemcpy(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

int main() {
    // 2D float array: fill from pageable, pinned and device memory; read back by copies and by tex2D
    {
        const int width = 37, height = 21;
        std::vector<float> image(width * height);
        for (int i = 0; i < width * height; i++) {
            image[i] = (i % width) * 1.5f + (i / width) * 100.0f;
        }
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t array;
        CHECK(cudaMallocArray(&array, &desc, width, height));
        cudaChannelFormatDesc got_desc{};
        cudaExtent extent{};
        unsigned int flags = 99;
        CHECK(cudaArrayGetInfo(&got_desc, &extent, &flags, array));
        EXPECT(got_desc.x == 32 && got_desc.f == cudaChannelFormatKindFloat && extent.width == width &&
               extent.height == height && extent.depth == 0 && flags == 0, "cudaArrayGetInfo");
        CHECK(cudaGetChannelDesc(&got_desc, array));
        EXPECT(got_desc.x == 32, "cudaGetChannelDesc");

        float *out;
        CHECK(cudaMalloc(&out, width * height * sizeof(float)));
        const cudaTextureObject_t point = arrayTexture(array);
        const cudaTextureObject_t linear = arrayTexture(array, cudaFilterModeLinear);
        float *pinned;
        CHECK(cudaMallocHost(&pinned, width * height * sizeof(float)));
        float *pitched;
        size_t pitch;
        CHECK(cudaMallocPitch(reinterpret_cast<void **>(&pitched), &pitch, width * sizeof(float), height));
        for (int source = 0; source < 3; source++) {
            std::vector<float> data = image;
            for (float &v: data) {
                v += source * 0.25f;
            }
            if (source == 0) {
                CHECK(cudaMemcpy2DToArray(array, 0, 0, data.data(), width * sizeof(float), width * sizeof(float),
                                          height, cudaMemcpyHostToDevice));
            } else if (source == 1) {
                memcpy(pinned, data.data(), data.size() * sizeof(float));
                CHECK(cudaMemcpy2DToArrayAsync(array, 0, 0, pinned, width * sizeof(float), width * sizeof(float),
                                               height, cudaMemcpyHostToDevice, nullptr));
            } else {
                CHECK(cudaMemcpy2D(pitched, pitch, data.data(), width * sizeof(float), width * sizeof(float), height,
                                   cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy2DToArray(array, 0, 0, pitched, pitch, width * sizeof(float), height,
                                          cudaMemcpyDeviceToDevice));
            }
            fetch2D<float><<<8, 128>>>(point, width, height, 0.5f, 0.5f, out);
            std::vector<float> fetched = download(out, width * height);
            std::vector<float> copied(width * height, -1.0f);
            CHECK(cudaMemcpy2DFromArray(copied.data(), width * sizeof(float), array, 0, 0, width * sizeof(float),
                                        height, cudaMemcpyDeviceToHost));
            for (int i = 0; i < width * height; i++) {
                EXPECT(fetched[i] == data[i], "source %d: tex2D [%d] = %g, expected %g", source, i, fetched[i],
                       data[i]);
                EXPECT(copied[i] == data[i], "source %d: copied back [%d] = %g, expected %g", source, i, copied[i],
                       data[i]);
            }
        }
        // linear filtering halfway between texel centers in x (the last column clamps to the edge)
        fetch2D<float><<<8, 128>>>(linear, width, height, 1.0f, 0.5f, out);
        std::vector<float> filtered = download(out, width * height);
        for (int i = 0; i < width * height; i++) {
            const int x = i % width, right = std::min(x + 1, width - 1) + i / width * width;
            const float expected = (image[i] + image[right]) / 2 + 0.5f; // source 2's data
            EXPECT(std::fabs(filtered[i] - expected) < 1e-2f, "linear [%d] = %g, expected %g", i, filtered[i],
                   expected);
        }

        // a sub-rectangle at an offset, from device memory and back to pinned memory
        const int sx = 5, sy = 3, sw = 11, sh = 7;
        std::vector<float> patch(sw * sh);
        for (int i = 0; i < sw * sh; i++) {
            patch[i] = -1000.0f - i;
        }
        CHECK(cudaMemcpy(pitched, patch.data(), patch.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy2DToArray(array, sx * sizeof(float), sy, pitched, sw * sizeof(float), sw * sizeof(float), sh,
                                  cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy2DFromArray(pinned, width * sizeof(float), array, 0, 0, width * sizeof(float), height,
                                    cudaMemcpyDeviceToHost));
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const bool inside = x >= sx && x < sx + sw && y >= sy && y < sy + sh;
                const float expected = inside ? patch[(y - sy) * sw + (x - sx)] : image[y * width + x] + 0.5f;
                EXPECT(pinned[y * width + x] == expected, "patched (%d, %d) = %g, expected %g", x, y,
                       pinned[y * width + x], expected);
            }
        }

        // array to array, and the deprecated row-major copies across row ends
        cudaArray_t copy;
        CHECK(cudaMallocArray(&copy, &desc, width, height));
        CHECK(cudaMemcpy2DArrayToArray(copy, 0, 0, array, 0, 0, width * sizeof(float), height,
                                       cudaMemcpyDeviceToDevice));
        std::vector<float> row_major(width * 3 + 7, 0.0f);
        CHECK(cudaMemcpyFromArray(row_major.data(), copy, 4 * sizeof(float), 2, row_major.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < row_major.size(); i++) {
            const int x = int(4 + i) % width, y = 2 + int(4 + i) / width;
            const float expected = pinned[y * width + x];
            EXPECT(row_major[i] == expected, "row major [%zu] = %g, expected %g", i, row_major[i], expected);
        }
        CHECK(cudaDestroyTextureObject(point));
        CHECK(cudaDestroyTextureObject(linear));
        CHECK(cudaFreeArray(copy));
        CHECK(cudaFreeArray(array));
        CHECK(cudaFreeHost(pinned));
        CHECK(cudaFree(pitched));
        CHECK(cudaFree(out));
    }

    // 8 and 16 bit texels (GOB formats of their own on Blackwell) and uchar4 with filtering
    {
        const int width = 70, height = 19, n = width * height;
        std::vector<unsigned char> bytes(n);
        std::vector<unsigned short> shorts(n);
        std::vector<__half> halves(n);
        std::vector<uchar4> colors(n);
        for (int i = 0; i < n; i++) {
            bytes[i] = static_cast<unsigned char>(i * 7 + 3);
            shorts[i] = static_cast<unsigned short>(i * 31 + 5);
            halves[i] = __float2half(i * 0.25f);
            colors[i] = make_uchar4(i % 256, (i * 3) % 256, 255 - i % 256, 17);
        }
        float *out;
        float4 *out4;
        CHECK(cudaMalloc(&out, n * sizeof(float)));
        CHECK(cudaMalloc(&out4, n * sizeof(float4)));
        auto make = [&](cudaChannelFormatDesc desc, const void *data, size_t element) {
            cudaArray_t array;
            CHECK(cudaMallocArray(&array, &desc, width, height));
            CHECK(cudaMemcpy2DToArray(array, 0, 0, data, width * element, width * element, height,
                                      cudaMemcpyHostToDevice));
            return array;
        };
        cudaArray_t a8 = make(cudaCreateChannelDesc<unsigned char>(), bytes.data(), 1);
        cudaArray_t a16 = make(cudaCreateChannelDesc<unsigned short>(), shorts.data(), 2);
        cudaArray_t ah = make(cudaCreateChannelDescHalf(), halves.data(), 2);
        cudaArray_t a32 = make(cudaCreateChannelDesc<uchar4>(), colors.data(), 4);
        cudaTextureObject_t t8 = arrayTexture(a8, cudaFilterModePoint, cudaReadModeNormalizedFloat);
        cudaTextureObject_t t16 = arrayTexture(a16, cudaFilterModePoint, cudaReadModeNormalizedFloat);
        cudaTextureObject_t th = arrayTexture(ah);
        cudaTextureObject_t t32 = arrayTexture(a32, cudaFilterModeLinear, cudaReadModeNormalizedFloat);
        fetch2D<float><<<8, 128>>>(t8, width, height, 0.5f, 0.5f, out);
        std::vector<float> r8 = download(out, n);
        fetch2D<float><<<8, 128>>>(t16, width, height, 0.5f, 0.5f, out);
        std::vector<float> r16 = download(out, n);
        fetch2D<float><<<8, 128>>>(th, width, height, 0.5f, 0.5f, out);
        std::vector<float> rh = download(out, n);
        fetch2D<float4><<<8, 128>>>(t32, width, height, 0.5f, 1.0f, out4); // halfway to the next row
        std::vector<float4> r32 = download(out4, n);
        for (int i = 0; i < n; i++) {
            EXPECT(std::fabs(r8[i] - bytes[i] / 255.0f) < 1e-5f, "uchar [%d] = %g", i, r8[i]);
            EXPECT(std::fabs(r16[i] - shorts[i] / 65535.0f) < 1e-6f, "ushort [%d] = %g", i, r16[i]);
            EXPECT(rh[i] == i * 0.25f || std::fabs(rh[i] - i * 0.25f) <= i * 0.25f / 1024, "half [%d] = %g", i,
                   rh[i]);
            const int below = std::min(i + width, n - width + i % width);
            const float expected_x = (colors[i].x + colors[below].x) / 510.0f;
            const float expected_w = 17 / 255.0f;
            EXPECT(std::fabs(r32[i].x - expected_x) < 1.0f / 128 && std::fabs(r32[i].w - expected_w) < 1e-3f,
                   "uchar4 filtered [%d] = %g %g, expected %g %g", i, r32[i].x, r32[i].w, expected_x, expected_w);
        }
        // the 8 and 16 bit arrays copied back
        std::vector<unsigned char> bytes_back(n);
        std::vector<unsigned short> shorts_back(n);
        CHECK(cudaMemcpy2DFromArray(bytes_back.data(), width, a8, 0, 0, width, height, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy2DFromArray(shorts_back.data(), width * 2, a16, 0, 0, width * 2, height,
                                    cudaMemcpyDeviceToHost));
        EXPECT(bytes_back == bytes && shorts_back == shorts, "8 / 16 bit arrays copied back");
        for (cudaTextureObject_t t: {t8, t16, th, t32}) {
            CHECK(cudaDestroyTextureObject(t));
        }
        for (cudaArray_t a: {a8, a16, ah, a32}) {
            CHECK(cudaFreeArray(a));
        }
        CHECK(cudaFree(out));
        CHECK(cudaFree(out4));
    }

    // 3D array: cudaMemcpy3D from pageable memory, tex3D with texel and (wrapped) normalized coordinates
    {
        const int width = 9, height = 6, depth = 5, n = width * height * depth;
        std::vector<float> volume(n);
        for (int i = 0; i < n; i++) {
            volume[i] = i * 0.5f - 7.0f;
        }
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t array;
        CHECK(cudaMalloc3DArray(&array, &desc, make_cudaExtent(width, height, depth)));
        cudaMemcpy3DParms p{};
        p.srcPtr = make_cudaPitchedPtr(volume.data(), width * sizeof(float), width, height);
        p.dstArray = array;
        p.extent = make_cudaExtent(width, height, depth);
        p.kind = cudaMemcpyHostToDevice;
        CHECK(cudaMemcpy3D(&p));
        float *out;
        CHECK(cudaMalloc(&out, n * sizeof(float)));
        const cudaTextureObject_t t = arrayTexture(array);
        const cudaTextureObject_t tn = arrayTexture(array, cudaFilterModePoint, cudaReadModeElementType, true,
                                                    cudaAddressModeWrap);
        fetch3D<<<4, 128>>>(t, width, height, depth, false, out);
        std::vector<float> fetched = download(out, n);
        fetch3D<<<4, 128>>>(tn, width, height, depth, true, out);
        std::vector<float> wrapped = download(out, n);
        for (int i = 0; i < n; i++) {
            EXPECT(fetched[i] == volume[i], "tex3D [%d] = %g, expected %g", i, fetched[i], volume[i]);
            EXPECT(wrapped[i] == volume[i], "tex3D normalized wrap [%d] = %g, expected %g", i, wrapped[i], volume[i]);
        }
        // a box out of the middle, through cudaMemcpy3D into pitched device memory
        cudaPitchedPtr box;
        CHECK(cudaMalloc3D(&box, make_cudaExtent(4 * sizeof(float), 3, 2)));
        cudaMemcpy3DParms q{};
        q.srcArray = array;
        q.srcPos = make_cudaPos(2, 1, 2);
        q.dstPtr = box;
        q.extent = make_cudaExtent(4, 3, 2);
        q.kind = cudaMemcpyDeviceToDevice;
        CHECK(cudaMemcpy3D(&q));
        std::vector<float> host_box(4 * 3 * 2);
        cudaMemcpy3DParms r{};
        r.srcPtr = box;
        r.dstPtr = make_cudaPitchedPtr(host_box.data(), 4 * sizeof(float), 4, 3);
        r.extent = make_cudaExtent(4 * sizeof(float), 3, 2);
        r.kind = cudaMemcpyDeviceToHost;
        CHECK(cudaMemcpy3D(&r));
        for (int z = 0; z < 2; z++) {
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 4; x++) {
                    const float expected = volume[((z + 2) * height + y + 1) * width + x + 2];
                    EXPECT(host_box[(z * 3 + y) * 4 + x] == expected, "box (%d, %d, %d)", x, y, z);
                }
            }
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaDestroyTextureObject(tn));
        CHECK(cudaFreeArray(array));
        CHECK(cudaFree(box.ptr));
        CHECK(cudaFree(out));
    }

    // layered 2D array and tex2DLayered
    {
        const int width = 11, height = 6, layers = 3, n = width * height * layers;
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) {
            data[i] = i * 3.0f;
        }
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t array;
        CHECK(cudaMalloc3DArray(&array, &desc, make_cudaExtent(width, height, layers), cudaArrayLayered));
        cudaMemcpy3DParms p{};
        p.srcPtr = make_cudaPitchedPtr(data.data(), width * sizeof(float), width, height);
        p.dstArray = array;
        p.extent = make_cudaExtent(width, height, layers);
        p.kind = cudaMemcpyHostToDevice;
        CHECK(cudaMemcpy3D(&p));
        float *out;
        CHECK(cudaMalloc(&out, n * sizeof(float)));
        const cudaTextureObject_t t = arrayTexture(array);
        fetchLayered<<<4, 128>>>(t, width, height, layers, out);
        std::vector<float> fetched = download(out, n);
        for (int i = 0; i < n; i++) {
            EXPECT(fetched[i] == data[i], "tex2DLayered [%d] = %g, expected %g", i, fetched[i], data[i]);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaFreeArray(array));
        CHECK(cudaFree(out));
    }

    // layered 1D array and tex1DLayered
    {
        const int width = 50, layers = 4, n = width * layers;
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) {
            data[i] = i + 1.0f;
        }
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t array;
        CHECK(cudaMalloc3DArray(&array, &desc, make_cudaExtent(width, 0, layers), cudaArrayLayered));
        cudaMemcpy3DParms p{};
        p.srcPtr = make_cudaPitchedPtr(data.data(), width * sizeof(float), width, 1);
        p.dstArray = array;
        p.extent = make_cudaExtent(width, 1, layers);
        p.kind = cudaMemcpyHostToDevice;
        CHECK(cudaMemcpy3D(&p));
        float *out;
        CHECK(cudaMalloc(&out, n * sizeof(float)));
        const cudaTextureObject_t t = arrayTexture(array);
        fetchLayered1D<<<2, 128>>>(t, width, layers, out);
        std::vector<float> fetched = download(out, n);
        for (int i = 0; i < n; i++) {
            EXPECT(fetched[i] == data[i], "tex1DLayered [%d] = %g, expected %g", i, fetched[i], data[i]);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaFreeArray(array));
        CHECK(cudaFree(out));
    }

    // 1D array through the deprecated cudaMemcpyToArray
    {
        const int width = 300;
        std::vector<float> data(width);
        for (int i = 0; i < width; i++) {
            data[i] = std::sin(i * 0.1f);
        }
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaArray_t array;
        CHECK(cudaMallocArray(&array, &desc, width, 0));
        CHECK(cudaMemcpyToArray(array, 0, 0, data.data(), width * sizeof(float), cudaMemcpyHostToDevice));
        float *out;
        CHECK(cudaMalloc(&out, width * sizeof(float)));
        const cudaTextureObject_t t = arrayTexture(array);
        fetch1D<<<2, 128>>>(t, width, out);
        std::vector<float> fetched = download(out, width);
        for (int i = 0; i < width; i++) {
            EXPECT(fetched[i] == data[i], "tex1D [%d] = %g, expected %g", i, fetched[i], data[i]);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaFreeArray(array));
        CHECK(cudaFree(out));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("cuda_arrays");
    return 0;
}
