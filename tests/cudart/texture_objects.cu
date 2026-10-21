// Texture objects on linear memory (tex1Dfetch): several formats and channel counts, normalized reads of 8 and 16 bit
// integers, several textures in one kernel, reuse of destroyed handles, and unsupported descriptions.
#include "check.h"
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
        res.res.pitch2D.pitchInBytes = 256;
        EXPECT(cudaCreateTextureObject(&object, &res, &tex, nullptr) == cudaErrorNotSupported,
               "pitched texture accepted");
        cudaGetLastError();
        CHECK(cudaFree(d));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("texture_objects");
    return 0;
}
