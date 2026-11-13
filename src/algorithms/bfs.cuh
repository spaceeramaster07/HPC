#pragma once

#include "batch_update.cuh"
#include "hornet_graph.cuh"
#include "types.cuh"

#include <cub/cub.cuh>
#include <chrono>
#include <vector>

#ifndef HORNET_DS_NS
#define HORNET_DS_NS baseline
#endif

namespace hornet_alg {

using namespace hornet_common;
using namespace HORNET_DS_NS;

// =============================================================================
// BFS (Hornet Tier 1 & Static CSR Reference)
// =============================================================================

__global__ void gather_frontier_deg_kernel(int frontier_size,
                                           const vid_t* __restrict__ frontier,
                                           const vid_t* __restrict__ d_used,
                                           eoff_t* __restrict__ frontier_deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < frontier_size) {
        frontier_deg[i] = d_used[frontier[i]];
    }
}

__global__ void gather_frontier_deg_csr_kernel(int frontier_size,
                                               const vid_t* __restrict__ frontier,
                                               const eoff_t* __restrict__ csr_offsets,
                                               eoff_t* __restrict__ frontier_deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < frontier_size) {
        vid_t u = frontier[i];
        frontier_deg[i] = csr_offsets[u + 1] - csr_offsets[u];
    }
}

__global__ void bfs_expand_balanced_kernel(int frontier_size,
                                           const vid_t* __restrict__ frontier,
                                           const eoff_t* __restrict__ frontier_off,
                                           eoff_t total_edges,
                                           vid_t* const* __restrict__ d_ptr,
                                           int next_level,
                                           int* __restrict__ d_dist,
                                           vid_t* __restrict__ next_frontier,
                                           int* __restrict__ d_next_size) {
    eoff_t gid = static_cast<eoff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_edges) return;

    int lo = 0;
    int hi = frontier_size - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (frontier_off[mid + 1] <= gid) {
            lo = mid + 1;
        } else if (frontier_off[mid] > gid) {
            hi = mid - 1;
        } else {
            lo = mid;
            break;
        }
    }

    vid_t u = frontier[lo];
    eoff_t local_e = gid - frontier_off[lo];
    vid_t v = d_ptr[u][local_e];

    if (atomicCAS(&d_dist[v], -1, next_level) == -1) {
        int pos = atomicAdd(d_next_size, 1);
        next_frontier[pos] = v;
    }
}

__global__ void bfs_expand_balanced_csr_kernel(int frontier_size,
                                               const vid_t* __restrict__ frontier,
                                               const eoff_t* __restrict__ frontier_off,
                                               eoff_t total_edges,
                                               const eoff_t* __restrict__ csr_offsets,
                                               const vid_t* __restrict__ csr_edges,
                                               int next_level,
                                               int* __restrict__ d_dist,
                                               vid_t* __restrict__ next_frontier,
                                               int* __restrict__ d_next_size) {
    eoff_t gid = static_cast<eoff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_edges) return;

    int lo = 0;
    int hi = frontier_size - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (frontier_off[mid + 1] <= gid) {
            lo = mid + 1;
        } else if (frontier_off[mid] > gid) {
            hi = mid - 1;
        } else {
            lo = mid;
            break;
        }
    }

    vid_t u = frontier[lo];
    eoff_t local_e = gid - frontier_off[lo];
    vid_t v = csr_edges[csr_offsets[u] + local_e];

    if (atomicCAS(&d_dist[v], -1, next_level) == -1) {
        int pos = atomicAdd(d_next_size, 1);
        next_frontier[pos] = v;
    }
}

struct BFSResult {
    double time_ms = 0.0;
    long long traversed_edges = 0;
    double mteps = 0.0;
    int visited_vertices = 0;
};

inline BFSResult run_bfs(const HornetGraph& graph, vid_t source = 0) {
    vid_t nV = graph.nV();
    BFSResult res;
    if (nV == 0 || source < 0 || source >= nV) return res;

    int* d_dist = nullptr;
    vid_t* d_frontier = nullptr;
    vid_t* d_next_frontier = nullptr;
    eoff_t* d_frontier_deg = nullptr;
    eoff_t* d_frontier_off = nullptr;
    int* d_next_size = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dist, nV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, nV * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, nV * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_deg, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_off, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_dist, 0xFF, nV * sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(&d_dist[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(vid_t), cudaMemcpyHostToDevice));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_frontier_deg, d_frontier_off, nV + 1);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    int frontier_size = 1;
    int level = 0;
    long long total_traversed = 0;
    int visited_count = 1;

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    while (frontier_size > 0) {
        int blk = 256;
        gather_frontier_deg_kernel<<<(frontier_size + blk - 1) / blk, blk>>>(
            frontier_size, d_frontier, graph.d_used(), d_frontier_deg);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(&d_frontier_deg[frontier_size], 0, sizeof(eoff_t)));
        cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes, d_frontier_deg, d_frontier_off, frontier_size + 1);

        eoff_t total_edges = 0;
        CUDA_CHECK(cudaMemcpy(&total_edges, &d_frontier_off[frontier_size],
                              sizeof(eoff_t), cudaMemcpyDeviceToHost));
        total_traversed += total_edges;

        if (total_edges == 0) break;

        CUDA_CHECK(cudaMemset(d_next_size, 0, sizeof(int)));
        eoff_t grid = (total_edges + blk - 1) / blk;
        bfs_expand_balanced_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            frontier_size, d_frontier, d_frontier_off, total_edges,
            graph.d_ptr(), level + 1, d_dist, d_next_frontier, d_next_size);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        visited_count += frontier_size;
        std::swap(d_frontier, d_next_frontier);
        ++level;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.traversed_edges = total_traversed;
    res.visited_vertices = visited_count;
    res.mteps = (res.time_ms > 0.0) ? ((total_traversed / 1e6) / (res.time_ms / 1e3)) : 0.0;

    cudaFree(d_temp);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_deg);
    cudaFree(d_frontier_off);
    cudaFree(d_next_size);

    return res;
}

inline BFSResult run_bfs_csr(const CSRGraph& csr, vid_t source = 0) {
    vid_t nV = csr.nV;
    BFSResult res;
    if (nV == 0 || source < 0 || source >= nV) return res;

    eoff_t* d_csr_off = nullptr;
    vid_t* d_csr_edges = nullptr;
    int* d_dist = nullptr;
    vid_t* d_frontier = nullptr;
    vid_t* d_next_frontier = nullptr;
    eoff_t* d_frontier_deg = nullptr;
    eoff_t* d_frontier_off = nullptr;
    int* d_next_size = nullptr;

    CUDA_CHECK(cudaMalloc(&d_csr_off, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_csr_edges, std::max<eoff_t>(csr.nE, 1) * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_dist, nV * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, nV * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, nV * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_deg, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_off, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_csr_off, csr.row_offsets.data(), (nV + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));
    if (csr.nE > 0) {
        CUDA_CHECK(cudaMemcpy(d_csr_edges, csr.col_indices.data(), csr.nE * sizeof(vid_t), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(d_dist, 0xFF, nV * sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(&d_dist[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(vid_t), cudaMemcpyHostToDevice));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_frontier_deg, d_frontier_off, nV + 1);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    int frontier_size = 1;
    int level = 0;
    long long total_traversed = 0;
    int visited_count = 1;

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    while (frontier_size > 0) {
        int blk = 256;
        gather_frontier_deg_csr_kernel<<<(frontier_size + blk - 1) / blk, blk>>>(
            frontier_size, d_frontier, d_csr_off, d_frontier_deg);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(&d_frontier_deg[frontier_size], 0, sizeof(eoff_t)));
        cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes, d_frontier_deg, d_frontier_off, frontier_size + 1);

        eoff_t total_edges = 0;
        CUDA_CHECK(cudaMemcpy(&total_edges, &d_frontier_off[frontier_size],
                              sizeof(eoff_t), cudaMemcpyDeviceToHost));
        total_traversed += total_edges;

        if (total_edges == 0) break;

        CUDA_CHECK(cudaMemset(d_next_size, 0, sizeof(int)));
        eoff_t grid = (total_edges + blk - 1) / blk;
        bfs_expand_balanced_csr_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            frontier_size, d_frontier, d_frontier_off, total_edges,
            d_csr_off, d_csr_edges, level + 1, d_dist, d_next_frontier, d_next_size);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        visited_count += frontier_size;
        std::swap(d_frontier, d_next_frontier);
        ++level;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.traversed_edges = total_traversed;
    res.visited_vertices = visited_count;
    res.mteps = (res.time_ms > 0.0) ? ((total_traversed / 1e6) / (res.time_ms / 1e3)) : 0.0;

    cudaFree(d_temp);
    cudaFree(d_csr_off);
    cudaFree(d_csr_edges);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_deg);
    cudaFree(d_frontier_off);
    cudaFree(d_next_size);

    return res;
}


} // namespace hornet_alg
