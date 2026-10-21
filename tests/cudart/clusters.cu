// Thread block clusters: compile-time (__cluster_dims__) and launch-time (cudaLaunchKernelEx) cluster sizes, cluster
// barriers and distributed shared memory, cluster attributes and occupancy, and invalid cluster launches.
#include "check.h"
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

constexpr int THREADS = 128;

// every block sums its slice; rank 0 of each cluster adds up the partial sums of all blocks of its cluster through
// distributed shared memory
__global__ void __cluster_dims__(4, 1, 1) clusterSum(const int *in, int *cluster_sums, int *ranks) {
    __shared__ int partial;
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    if (threadIdx.x == 0) {
        partial = 0;
    }
    block.sync();
    atomicAdd(&partial, in[blockIdx.x * THREADS + threadIdx.x]);
    cluster.sync();
    if (threadIdx.x == 0) {
        ranks[blockIdx.x] = cluster.block_rank();
        if (cluster.block_rank() == 0) {
            int sum = 0;
            for (unsigned r = 0; r < cluster.num_blocks(); r++) {
                sum += *cluster.map_shared_rank(&partial, r);
            }
            cluster_sums[blockIdx.x / cluster.num_blocks()] = sum;
        }
    }
    // keep every block's shared memory alive until rank 0 read it
    cluster.sync();
}

// no compile-time cluster size: launched with clusters through cudaLaunchKernelEx. Each block exchanges a value with
// its partner (rank ^ 1) and records its position in the cluster.
__global__ void exchange(int *out) {
    __shared__ int value;
    cg::cluster_group cluster = cg::this_cluster();
    const int block = blockIdx.x + blockIdx.y * gridDim.x;
    if (threadIdx.x == 0) {
        value = block;
    }
    cluster.sync();
    const int partner = *cluster.map_shared_rank(&value, cluster.block_rank() ^ 1);
    cluster.sync();
    if (threadIdx.x == 0) {
        const dim3 index = cluster.block_index();
        out[block * 4 + 0] = cluster.block_rank();
        out[block * 4 + 1] = partner;
        out[block * 4 + 2] = index.x + 10 * index.y;
        out[block * 4 + 3] = cluster.num_blocks();
    }
}

// a regular launch: every block is a cluster of one
__global__ void noCluster(int *out) {
    if (threadIdx.x == 0) {
        cg::cluster_group cluster = cg::this_cluster();
        out[blockIdx.x * 2] = cluster.block_rank();
        out[blockIdx.x * 2 + 1] = cluster.num_blocks();
    }
}

int main() {
    int supported = 0;
    CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, 0));
    EXPECT(supported == 1, "cluster launch attribute is %d", supported);

    // compile-time cluster size
    {
        const int blocks = 64;
        std::vector<int> in(blocks * THREADS);
        for (size_t i = 0; i < in.size(); i++) {
            in[i] = static_cast<int>(i % 97);
        }
        int *d_in, *d_sums, *d_ranks;
        CHECK(cudaMalloc(&d_in, in.size() * sizeof(int)));
        CHECK(cudaMalloc(&d_sums, blocks / 4 * sizeof(int)));
        CHECK(cudaMalloc(&d_ranks, blocks * sizeof(int)));
        CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(int), cudaMemcpyHostToDevice));
        clusterSum<<<blocks, THREADS>>>(d_in, d_sums, d_ranks);
        CHECK(cudaGetLastError());
        std::vector<int> sums(blocks / 4), ranks(blocks);
        CHECK(cudaMemcpy(sums.data(), d_sums, sums.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(ranks.data(), d_ranks, ranks.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int c = 0; c < blocks / 4; c++) {
            int expected = 0;
            for (int i = c * 4 * THREADS; i < (c + 1) * 4 * THREADS; i++) {
                expected += in[i];
            }
            EXPECT(sums[c] == expected, "cluster %d: sum %d, expected %d", c, sums[c], expected);
        }
        for (int b = 0; b < blocks; b++) {
            EXPECT(ranks[b] == b % 4, "block %d has cluster rank %d", b, ranks[b]);
        }

        cudaFuncAttributes attrs{};
        CHECK(cudaFuncGetAttributes(&attrs, clusterSum));
        EXPECT(attrs.requiredClusterWidth == 4 && attrs.requiredClusterHeight == 1 && attrs.requiredClusterDepth == 1,
               "required cluster size %dx%dx%d", attrs.requiredClusterWidth, attrs.requiredClusterHeight,
               attrs.requiredClusterDepth);

        // a different launch-time cluster size conflicts with the compiled one
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = 2;
        attribute.val.clusterDim.y = 1;
        attribute.val.clusterDim.z = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(blocks);
        config.blockDim = dim3(THREADS);
        config.attrs = &attribute;
        config.numAttrs = 1;
        EXPECT(cudaLaunchKernelEx(&config, clusterSum, (const int *) d_in, d_sums, d_ranks) ==
               cudaErrorInvalidClusterSize, "conflicting cluster size accepted");
        cudaGetLastError();
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_sums));
        CHECK(cudaFree(d_ranks));
    }

    // launch-time 2x2 clusters on an 8x4 grid
    {
        const dim3 grid(8, 4);
        const int blocks = grid.x * grid.y;
        int *d_out;
        CHECK(cudaMalloc(&d_out, blocks * 4 * sizeof(int)));
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = 2;
        attribute.val.clusterDim.y = 2;
        attribute.val.clusterDim.z = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = grid;
        config.blockDim = dim3(64);
        config.attrs = &attribute;
        config.numAttrs = 1;
        CHECK(cudaLaunchKernelEx(&config, exchange, d_out));
        std::vector<int> out(blocks * 4);
        CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (unsigned y = 0; y < grid.y; y++) {
            for (unsigned x = 0; x < grid.x; x++) {
                const int block = x + y * grid.x;
                const int rank = (x % 2) + 2 * (y % 2);
                // the partner (rank ^ 1) is the neighbour in x within the cluster
                const int partner = (x ^ 1) + y * grid.x;
                const int *r = &out[block * 4];
                EXPECT(r[0] == rank && r[1] == partner && r[2] == int(x % 2 + 10 * (y % 2)) && r[3] == 4,
                       "block (%u,%u): rank %d partner %d index %d size %d", x, y, r[0], r[1], r[2], r[3]);
            }
        }

        int clusters = 0, potential = 0;
        config.blockDim = dim3(64);
        CHECK(cudaOccupancyMaxActiveClusters(&clusters, exchange, &config));
        CHECK(cudaOccupancyMaxPotentialClusterSize(&potential, exchange, &config));
        printf("occupancy: %d active 2x2 clusters, potential cluster size %d\n", clusters, potential);
        EXPECT(clusters > 0 && potential >= 4, "cluster occupancy %d / %d", clusters, potential);

        // invalid: grid not a multiple of the cluster, and a cluster above the portable size of 8
        config.gridDim = dim3(7, 4);
        EXPECT(cudaLaunchKernelEx(&config, exchange, d_out) == cudaErrorInvalidClusterSize,
               "grid not divisible by the cluster accepted");
        cudaGetLastError();
        attribute.val.clusterDim.x = 16;
        attribute.val.clusterDim.y = 1;
        config.gridDim = dim3(16);
        EXPECT(cudaLaunchKernelEx(&config, exchange, d_out) == cudaErrorInvalidClusterSize,
               "cluster of 16 accepted");
        cudaGetLastError();
        CHECK(cudaFree(d_out));
    }

    // regular launch: clusters of one block
    {
        int *d_out;
        CHECK(cudaMalloc(&d_out, 8 * 2 * sizeof(int)));
        noCluster<<<8, 32>>>(d_out);
        std::vector<int> out(16);
        CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int b = 0; b < 8; b++) {
            EXPECT(out[b * 2] == 0 && out[b * 2 + 1] == 1, "block %d: rank %d of %d", b, out[b * 2],
                   out[b * 2 + 1]);
        }
        CHECK(cudaFree(d_out));
    }

    CHECK(cudaDeviceSynchronize());
    PASS("clusters");
    return 0;
}
