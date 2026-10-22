// Texture objects on linear memory (tex1Dfetch): several formats and channel counts, normalized reads of 8 and 16 bit
// integers, several textures in one kernel, reuse of destroyed handles, and unsupported descriptions. Texture objects
// on pitched 2D memory (tex2D): point and linear filtering, texel and normalized coordinates, clamp / border / wrap /
// mirror addressing, normalized reads of uchar4 and element reads of int, checked against a CPU reference.
#include "check.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

__global__ void fetchFloat(cudaTextureObject_t a, cudaTextureObject_t b, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tex1Dfetch<float>(a, i) + 1000.0f * tex1Dfetch<float>(b, i);
    }
}

__global__ void fetchFloat4(cudaTextureObject_t t, float4 *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tex1Dfetch<float4>(t, i);
    }
}

__global__ void fetchInt2(cudaTextureObject_t t, int2 *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tex1Dfetch<int2>(t, i);
    }
}

__global__ void fetchUint(cudaTextureObject_t t, unsigned *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tex1Dfetch<unsigned>(t, i);
    }
}

__global__ void fetchNormalized(cudaTextureObject_t uchar4_texture, cudaTextureObject_t short_texture, float4 *out4,
                                float *out1, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out4[i] = tex1Dfetch<float4>(uchar4_texture, i);
        out1[i] = tex1Dfetch<float>(short_texture, i);
    }
}

template<typename T>
__global__ void sample2D(cudaTextureObject_t t, const float2 *coords, T *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tex2D<T>(t, coords[i].x, coords[i].y);
    }
}

// CPU reference of a 2D texture of one channel: CUDA's addressing (wrap / mirror only with normalized coordinates)
struct Reference2D {
    const std::vector<float> &texels; // width * height, row major
    int width, height;
    cudaTextureAddressMode mode;
    float border;

    float texel(int x, int y) const {
        auto address = [&](int i, int n, bool *outside) {
            switch (mode) {
                case cudaAddressModeWrap: return ((i % n) + n) % n;
                case cudaAddressModeMirror: {
                    const int m = ((i % (2 * n)) + 2 * n) % (2 * n);
                    return m < n ? m : 2 * n - 1 - m;
                }
                case cudaAddressModeBorder:
                    *outside = *outside || i < 0 || i >= n;
                    return std::min(std::max(i, 0), n - 1);
                default: return std::min(std::max(i, 0), n - 1);
            }
        };
        bool outside = false;
        const int ax = address(x, width, &outside), ay = address(y, height, &outside);
        return outside ? border : texels[ay * width + ax];
    }

    // x, y in texels; *tolerance gets the allowed error (linear filtering uses 8 bit fractions in hardware)
    float sample(float x, float y, bool linear, float *tolerance) const {
        if (!linear) {
            *tolerance = 1e-6f;
            return texel(int(std::floor(x)), int(std::floor(y)));
        }
        const float xb = x - 0.5f, yb = y - 0.5f;
        const int i = int(std::floor(xb)), j = int(std::floor(yb));
        const float a = xb - i, b = yb - j;
        const float t00 = texel(i, j), t10 = texel(i + 1, j), t01 = texel(i, j + 1), t11 = texel(i + 1, j + 1);
        const float lo = std::min({t00, t10, t01, t11}), hi = std::max({t00, t10, t01, t11});
        *tolerance = (hi - lo) / 128.0f + 1e-4f;
        return (1 - a) * (1 - b) * t00 + a * (1 - b) * t10 + (1 - a) * b * t01 + a * b * t11;
    }
};

static cudaTextureObject_t pitchTexture(void *ptr, size_t pitch, int width, int height, cudaChannelFormatDesc desc,
                                        cudaTextureAddressMode mode, cudaTextureFilterMode filter, bool normalized,
                                        cudaTextureReadMode read = cudaReadModeElementType, float border = 0) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypePitch2D;
    res.res.pitch2D.devPtr = ptr;
    res.res.pitch2D.desc = desc;
    res.res.pitch2D.width = width;
    res.res.pitch2D.height = height;
    res.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc tex{};
    tex.addressMode[0] = tex.addressMode[1] = mode;
    tex.filterMode = filter;
    tex.normalizedCoords = normalized;
    tex.readMode = read;
    tex.borderColor[0] = border;
    cudaTextureObject_t object = 0;
    CHECK(cudaCreateTextureObject(&object, &res, &tex, nullptr));
    return object;
}

static cudaTextureObject_t linearTexture(void *ptr, size_t bytes, cudaChannelFormatDesc desc,
                                         cudaTextureReadMode mode = cudaReadModeElementType) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeLinear;
    res.res.linear.devPtr = ptr;
    res.res.linear.desc = desc;
    res.res.linear.sizeInBytes = bytes;
    cudaTextureDesc tex{};
    tex.readMode = mode;
    cudaTextureObject_t object = 0;
    CHECK(cudaCreateTextureObject(&object, &res, &tex, nullptr));
    return object;
}

template<typename T>
static T *upload(const std::vector<T> &host) {
    T *device;
    CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

int main() {
    const int n = 1000;
    const int blocks = (n + 127) / 128;

    // two float textures in one kernel; the second starts 256 bytes into its allocation
    {
        std::vector<float> a(n), b(n + 64);
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.5f;
        }
        for (int i = 0; i < n + 64; i++) {
            b[i] = static_cast<float>(i % 13);
        }
        float *da = upload(a), *db = upload(b), *out;
        CHECK(cudaMalloc(&out, n * sizeof(float)));
        const cudaTextureObject_t ta = linearTexture(da, n * sizeof(float), cudaCreateChannelDesc<float>());
        const cudaTextureObject_t tb = linearTexture(db + 64, n * sizeof(float), cudaCreateChannelDesc<float>());
        EXPECT(ta != 0 && tb != 0 && ta != tb, "texture handles %llu %llu", ta, tb);
        fetchFloat<<<blocks, 128>>>(ta, tb, out, n);
        std::vector<float> result(n);
        CHECK(cudaMemcpy(result.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            const float expected = a[i] + 1000.0f * b[i + 64];
            EXPECT(result[i] == expected, "float textures [%d]: %g, expected %g", i, result[i], expected);
        }

        cudaResourceDesc res{};
        CHECK(cudaGetTextureObjectResourceDesc(&res, tb));
        EXPECT(res.resType == cudaResourceTypeLinear && res.res.linear.devPtr == db + 64, "resource description");

        // a destroyed handle is reused for a new texture, which must see its own memory
        CHECK(cudaDestroyTextureObject(ta));
        const cudaTextureObject_t tc = linearTexture(db, n * sizeof(float), cudaCreateChannelDesc<float>());
        fetchFloat<<<blocks, 128>>>(tc, tb, out, n);
        CHECK(cudaMemcpy(result.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            const float expected = b[i] + 1000.0f * b[i + 64];
            EXPECT(result[i] == expected, "reused handle [%d]: %g, expected %g", i, result[i], expected);
        }
        CHECK(cudaDestroyTextureObject(tb));
        CHECK(cudaDestroyTextureObject(tc));
        CHECK(cudaFree(da));
        CHECK(cudaFree(db));
        CHECK(cudaFree(out));
    }

    // float4
    {
        std::vector<float4> v(n);
        for (int i = 0; i < n; i++) {
            v[i] = make_float4(i, -i, i * 0.25f, 7.0f);
        }
        float4 *dv = upload(v), *out;
        CHECK(cudaMalloc(&out, n * sizeof(float4)));
        const cudaTextureObject_t t = linearTexture(dv, n * sizeof(float4), cudaCreateChannelDesc<float4>());
        fetchFloat4<<<blocks, 128>>>(t, out, n);
        std::vector<float4> result(n);
        CHECK(cudaMemcpy(result.data(), out, n * sizeof(float4), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            EXPECT(memcmp(&result[i], &v[i], sizeof(float4)) == 0, "float4 [%d]: %g %g %g %g", i, result[i].x,
                   result[i].y, result[i].z, result[i].w);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaFree(dv));
        CHECK(cudaFree(out));
    }

    // int2 and unsigned
    {
        std::vector<int2> v(n);
        std::vector<unsigned> u(n);
        for (int i = 0; i < n; i++) {
            v[i] = make_int2(-i * 3, i * 7);
            u[i] = 0xf0000000u + i;
        }
        int2 *dv = upload(v), *out;
        unsigned *du = upload(u), *out_u;
        CHECK(cudaMalloc(&out, n * sizeof(int2)));
        CHECK(cudaMalloc(&out_u, n * sizeof(unsigned)));
        const cudaTextureObject_t t = linearTexture(dv, n * sizeof(int2), cudaCreateChannelDesc<int2>());
        const cudaTextureObject_t tu = linearTexture(du, n * sizeof(unsigned), cudaCreateChannelDesc<unsigned>());
        fetchInt2<<<blocks, 128>>>(t, out, n);
        fetchUint<<<blocks, 128>>>(tu, out_u, n);
        std::vector<int2> result(n);
        std::vector<unsigned> result_u(n);
        CHECK(cudaMemcpy(result.data(), out, n * sizeof(int2), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(result_u.data(), out_u, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            EXPECT(result[i].x == v[i].x && result[i].y == v[i].y, "int2 [%d]: %d %d", i, result[i].x, result[i].y);
            EXPECT(result_u[i] == u[i], "uint [%d]: %u", i, result_u[i]);
        }
        CHECK(cudaDestroyTextureObject(t));
        CHECK(cudaDestroyTextureObject(tu));
        CHECK(cudaFree(dv));
        CHECK(cudaFree(du));
        CHECK(cudaFree(out));
        CHECK(cudaFree(out_u));
    }

    // normalized reads: uchar4 -> [0, 1], short -> [-1, 1]
    {
        std::vector<uchar4> c(n);
        std::vector<short> s(n);
        for (int i = 0; i < n; i++) {
            c[i] = make_uchar4(i & 255, 255 - (i & 255), 0, 255);
            s[i] = static_cast<short>((i * 61) % 65536 - 32768);
        }
        uchar4 *dc = upload(c);
        short *ds = upload(s);
        float4 *out4;
        float *out1;
        CHECK(cudaMalloc(&out4, n * sizeof(float4)));
        CHECK(cudaMalloc(&out1, n * sizeof(float)));
        const cudaTextureObject_t tc = linearTexture(dc, n * sizeof(uchar4), cudaCreateChannelDesc<uchar4>(),
                                                     cudaReadModeNormalizedFloat);
        const cudaTextureObject_t ts = linearTexture(ds, n * sizeof(short), cudaCreateChannelDesc<short>(),
                                                     cudaReadModeNormalizedFloat);
        fetchNormalized<<<blocks, 128>>>(tc, ts, out4, out1, n);
        std::vector<float4> r4(n);
        std::vector<float> r1(n);
        CHECK(cudaMemcpy(r4.data(), out4, n * sizeof(float4), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(r1.data(), out1, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            EXPECT(nearlyEqual(r4[i].x, c[i].x / 255.0, 1e-6, 1e-6) && nearlyEqual(r4[i].y, c[i].y / 255.0, 1e-6, 1e-6) &&
                   r4[i].z == 0.0f && r4[i].w == 1.0f,
                   "uchar4 normalized [%d]: %g %g %g %g", i, r4[i].x, r4[i].y, r4[i].z, r4[i].w);
            const double expected = s[i] < -32767 ? -1.0 : s[i] / 32767.0;
            EXPECT(nearlyEqual(r1[i], expected, 1e-5, 1e-5), "short normalized [%d]: %g, expected %g", i, r1[i],
                   expected);
        }
        CHECK(cudaDestroyTextureObject(tc));
        CHECK(cudaDestroyTextureObject(ts));
        CHECK(cudaFree(dc));
        CHECK(cudaFree(ds));
        CHECK(cudaFree(out4));
        CHECK(cudaFree(out1));
    }

    // unsupported descriptions
    {
        float *d;
        CHECK(cudaMalloc(&d, 4096));
        cudaResourceDesc res{};
        cudaTextureDesc tex{};
        cudaTextureObject_t object = 0;
        res.resType = cudaResourceTypeLinear;
        res.res.linear.devPtr = d;
        res.res.linear.sizeInBytes = 4096;
        res.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
        EXPECT(cudaCreateTextureObject(&object, &res, &tex, nullptr) == cudaErrorInvalidChannelDescriptor,
               "3 channel texture accepted");
        cudaGetLastError();
        res.resType = cudaResourceTypePitch2D;
        res.res.pitch2D.devPtr = d;
        res.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        res.res.pitch2D.width = 16;
        res.res.pitch2D.height = 16;
        res.res.pitch2D.pitchInBytes = 40; // not a multiple of the pitch alignment
        EXPECT(cudaCreateTextureObject(&object, &res, &tex, nullptr) != cudaSuccess, "unaligned pitch accepted");
        cudaGetLastError();
        CHECK(cudaFree(d));
    }

    // pitched 2D float texture: filtering, coordinates and address modes against the CPU reference
    {
        const int width = 13, height = 7;
        std::vector<float> texels(width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                texels[y * width + x] = x * 1.5f + y * 10.0f - 20.0f;
            }
        }
        float *image;
        size_t pitch;
        CHECK(cudaMallocPitch(reinterpret_cast<void **>(&image), &pitch, width * sizeof(float), height));
        CHECK(cudaMemcpy2D(image, pitch, texels.data(), width * sizeof(float), width * sizeof(float), height,
                           cudaMemcpyHostToDevice));
        const int n = 512;
        float2 *d_coords;
        float *d_out;
        CHECK(cudaMalloc(&d_coords, n * sizeof(float2)));
        CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        struct Case {
            cudaTextureAddressMode mode;
            cudaTextureFilterMode filter;
            bool normalized;
        } cases[] = {
            {cudaAddressModeClamp, cudaFilterModePoint, false}, {cudaAddressModeClamp, cudaFilterModeLinear, false},
            {cudaAddressModeBorder, cudaFilterModePoint, false}, {cudaAddressModeBorder, cudaFilterModeLinear, false},
            {cudaAddressModeClamp, cudaFilterModeLinear, true}, {cudaAddressModeWrap, cudaFilterModePoint, true},
            {cudaAddressModeWrap, cudaFilterModeLinear, true}, {cudaAddressModeMirror, cudaFilterModePoint, true},
            {cudaAddressModeMirror, cudaFilterModeLinear, true},
        };
        unsigned seed = 12345;
        auto random = [&] { seed = seed * 1103515245u + 12345u; return (seed >> 8) & 0xffff; };
        for (const Case &c: cases) {
            const float border = 99.0f;
            const cudaTextureObject_t t = pitchTexture(image, pitch, width, height, cudaCreateChannelDesc<float>(),
                                                       c.mode, c.filter, c.normalized, cudaReadModeElementType,
                                                       border);
            // texel coordinates from -1.5 to 2.5 times the size; point samples away from texel edges (the hardware
            // rounds coordinates to 8 bit fractions)
            std::vector<float2> coords(n), texel_coords(n);
            for (int i = 0; i < n; i++) {
                const int range = c.mode == cudaAddressModeClamp || c.mode == cudaAddressModeBorder ? 2 : 4;
                const float fraction = c.filter == cudaFilterModePoint ? 0.2f + 0.6f * (random() % 1000) / 1000.0f
                                                                       : (random() % 1000) / 1000.0f;
                const float x = int(random() % (range * width)) - range * width / 2 + fraction;
                const float y = int(random() % (range * height)) - range * height / 2 + fraction;
                texel_coords[i] = make_float2(x, y);
                coords[i] = c.normalized ? make_float2(x / width, y / height) : texel_coords[i];
            }
            CHECK(cudaMemcpy(d_coords, coords.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
            sample2D<float><<<(n + 127) / 128, 128>>>(t, d_coords, d_out, n);
            std::vector<float> result(n);
            CHECK(cudaMemcpy(result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
            const Reference2D reference{texels, width, height, c.mode, border};
            for (int i = 0; i < n; i++) {
                float tolerance;
                const float expected = reference.sample(texel_coords[i].x, texel_coords[i].y,
                                                        c.filter == cudaFilterModeLinear, &tolerance);
                EXPECT(std::fabs(result[i] - expected) <= tolerance,
                       "pitch 2D mode %d filter %d normalized %d at (%g, %g): %g, expected %g", int(c.mode),
                       int(c.filter), int(c.normalized), texel_coords[i].x, texel_coords[i].y, result[i], expected);
            }
            CHECK(cudaDestroyTextureObject(t));
        }
        CHECK(cudaFree(image));
        CHECK(cudaFree(d_coords));
        CHECK(cudaFree(d_out));
    }

    // pitched 2D uchar4 read as normalized floats with linear filtering, int read as elements
    {
        const int width = 9, height = 5;
        std::vector<uchar4> colors(width * height);
        std::vector<int> ints(width * height);
        for (int i = 0; i < width * height; i++) {
            colors[i] = make_uchar4(i * 5, 255 - i * 5, i % 2 ? 255 : 0, 128);
            ints[i] = i * 1000 - 7;
        }
        uchar4 *d_colors;
        int *d_ints;
        size_t color_pitch, int_pitch;
        CHECK(cudaMallocPitch(reinterpret_cast<void **>(&d_colors), &color_pitch, width * sizeof(uchar4), height));
        CHECK(cudaMallocPitch(reinterpret_cast<void **>(&d_ints), &int_pitch, width * sizeof(int), height));
        CHECK(cudaMemcpy2D(d_colors, color_pitch, colors.data(), width * sizeof(uchar4), width * sizeof(uchar4),
                           height, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy2D(d_ints, int_pitch, ints.data(), width * sizeof(int), width * sizeof(int), height,
                           cudaMemcpyHostToDevice));
        const cudaTextureObject_t tc = pitchTexture(d_colors, color_pitch, width, height,
                                                    cudaCreateChannelDesc<uchar4>(), cudaAddressModeClamp,
                                                    cudaFilterModeLinear, false, cudaReadModeNormalizedFloat);
        const cudaTextureObject_t ti = pitchTexture(d_ints, int_pitch, width, height, cudaCreateChannelDesc<int>(),
                                                    cudaAddressModeClamp, cudaFilterModePoint, false);
        const int n = width * height;
        std::vector<float2> coords(n);
        for (int i = 0; i < n; i++) {
            // halfway between texel i and its right neighbour (clamped at the right edge); element reads use the
            // texel centers
            coords[i] = make_float2(i % width + 1.0f, i / width + 0.5f);
        }
        float2 *d_coords;
        float4 *d_out4;
        int *d_outi;
        CHECK(cudaMalloc(&d_coords, n * sizeof(float2)));
        CHECK(cudaMalloc(&d_out4, n * sizeof(float4)));
        CHECK(cudaMalloc(&d_outi, n * sizeof(int)));
        CHECK(cudaMemcpy(d_coords, coords.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
        sample2D<float4><<<1, 64>>>(tc, d_coords, d_out4, n);
        sample2D<int><<<1, 64>>>(ti, d_coords, d_outi, n);
        std::vector<float4> out4(n);
        std::vector<int> outi(n);
        CHECK(cudaMemcpy(out4.data(), d_out4, n * sizeof(float4), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(outi.data(), d_outi, n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            const int x = i % width, right = std::min(x + 1, width - 1) + (i / width) * width;
            const uchar4 a = colors[i], b = colors[right];
            const float expected[4] = {(a.x + b.x) / 510.0f, (a.y + b.y) / 510.0f, (a.z + b.z) / 510.0f,
                                       (a.w + b.w) / 510.0f};
            const float got[4] = {out4[i].x, out4[i].y, out4[i].z, out4[i].w};
            for (int c = 0; c < 4; c++) {
                EXPECT(std::fabs(got[c] - expected[c]) < 1.0f / 128, "uchar4 filtered [%d].%d: %g, expected %g", i, c,
                       got[c], expected[c]);
            }
            EXPECT(outi[i] == ints[std::min(x + 1, width - 1) + (i / width) * width], "int element [%d]: %d", i,
                   outi[i]);
        }
        cudaTextureDesc desc{};
        CHECK(cudaGetTextureObjectTextureDesc(&desc, tc));
        EXPECT(desc.filterMode == cudaFilterModeLinear && desc.readMode == cudaReadModeNormalizedFloat,
               "texture description");
        CHECK(cudaDestroyTextureObject(tc));
        CHECK(cudaDestroyTextureObject(ti));
        CHECK(cudaFree(d_colors));
        CHECK(cudaFree(d_ints));
        CHECK(cudaFree(d_coords));
        CHECK(cudaFree(d_out4));
        CHECK(cudaFree(d_outi));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("texture_objects");
    return 0;
}
