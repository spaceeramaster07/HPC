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

// =============================================================================
// SpMV (Hornet Tier 1 & Static CSR Reference)
// =============================================================================

__global__ void spmv_warp_kernel(vid_t nV,
                                 const vid_t* __restrict__ d_used,
                                 vid_t* const* __restrict__ d_ptr,
                                 const float* __restrict__ x,
                                 float* __restrict__ y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vid_t v = tid >> 5;
    int lane = tid & 31;
    if (v >= nV) return;

    vid_t deg = d_used[v];
    const vid_t* adj = d_ptr[v];

    float sum = 0.0f;
    for (vid_t i = lane; i < deg; i += 32) {
        sum += x[adj[i]];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    if (lane == 0) {
        y[v] = sum;
    }
}

__global__ void spmv_warp_csr_kernel(vid_t nV,
                                     const eoff_t* __restrict__ csr_off,
                                     const vid_t* __restrict__ csr_edges,
                                     const float* __restrict__ x,
                                     float* __restrict__ y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    vid_t v = tid >> 5;
    int lane = tid & 31;
    if (v >= nV) return;

    eoff_t start = csr_off[v];
    eoff_t end = csr_off[v + 1];

    float sum = 0.0f;
    for (eoff_t i = start + lane; i < end; i += 32) {
        sum += x[csr_edges[i]];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    if (lane == 0) {
        y[v] = sum;
    }
}

struct SpMVResult {
    double time_ms = 0.0;
};

inline SpMVResult run_spmv(const HornetGraph& graph, int num_iters = 10) {
    vid_t nV = graph.nV();
    SpMVResult res;
    if (nV == 0) return res;

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, nV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, nV * sizeof(float)));

    std::vector<float> h_x(nV, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), nV * sizeof(float), cudaMemcpyHostToDevice));

    int blk = 256;
    int warps_per_blk = blk / 32;
    int grid = (nV + warps_per_blk - 1) / warps_per_blk;

    spmv_warp_kernel<<<grid, blk>>>(nV, graph.d_used(), graph.d_ptr(), d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < num_iters; ++it) {
        spmv_warp_kernel<<<grid, blk>>>(nV, graph.d_used(), graph.d_ptr(), d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_iters;

    cudaFree(d_x);
    cudaFree(d_y);
    return res;
}

inline SpMVResult run_spmv_csr(const CSRGraph& csr, int num_iters = 10) {
    vid_t nV = csr.nV;
    SpMVResult res;
    if (nV == 0) return res;

    eoff_t* d_csr_off = nullptr;
    vid_t* d_csr_edges = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_csr_off, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_csr_edges, std::max<eoff_t>(csr.nE, 1) * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_x, nV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, nV * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_csr_off, csr.row_offsets.data(), (nV + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));
    if (csr.nE > 0) {
        CUDA_CHECK(cudaMemcpy(d_csr_edges, csr.col_indices.data(), csr.nE * sizeof(vid_t), cudaMemcpyHostToDevice));
    }
    std::vector<float> h_x(nV, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), nV * sizeof(float), cudaMemcpyHostToDevice));

    int blk = 256;
    int warps_per_blk = blk / 32;
    int grid = (nV + warps_per_blk - 1) / warps_per_blk;

    spmv_warp_csr_kernel<<<grid, blk>>>(nV, d_csr_off, d_csr_edges, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < num_iters; ++it) {
        spmv_warp_csr_kernel<<<grid, blk>>>(nV, d_csr_off, d_csr_edges, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / num_iters;

    cudaFree(d_csr_off);
    cudaFree(d_csr_edges);
    cudaFree(d_x);
    cudaFree(d_y);
    return res;
}

// =============================================================================
// Dynamic Maximal K-Truss (Paper Section IV-D, Fig. 9)
// =============================================================================

__global__ void ktruss_count_and_flag_kernel(vid_t nV,
                                             const eoff_t* __restrict__ d_offsets,
                                             eoff_t total_edges,
                                             const vid_t* __restrict__ d_used,
                                             vid_t* const* __restrict__ d_ptr,
                                             int min_triangles,
                                             vid_t* __restrict__ out_u,
                                             vid_t* __restrict__ out_v,
                                             uint8_t* __restrict__ out_prune) {
    eoff_t gid = static_cast<eoff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_edges) return;

    // Binary search vertex u owning edge gid
    vid_t lo = 0;
    vid_t hi = nV - 1;
    while (lo <= hi) {
        vid_t mid = lo + ((hi - lo) >> 1);
        if (d_offsets[mid + 1] <= gid) {
            lo = mid + 1;
        } else if (d_offsets[mid] > gid) {
            hi = mid - 1;
        } else {
            lo = mid;
            break;
        }
    }
    vid_t u = lo;
    eoff_t local_e = gid - d_offsets[u];
    const vid_t* adj_u = d_ptr[u];
    vid_t v = adj_u[local_e];

    out_u[gid] = u;
    out_v[gid] = v;

    vid_t deg_u = d_used[u];
    vid_t deg_v = d_used[v];
    const vid_t* adj_v = d_ptr[v];

    // Iterate over smaller sorted list and binary search larger sorted list
    const vid_t* small_adj = (deg_u <= deg_v) ? adj_u : adj_v;
    vid_t small_deg = (deg_u <= deg_v) ? deg_u : deg_v;
    const vid_t* large_adj = (deg_u <= deg_v) ? adj_v : adj_u;
    vid_t large_deg = (deg_u <= deg_v) ? deg_v : deg_u;

    int triangles = 0;
    for (vid_t i = 0; i < small_deg; ++i) {
        vid_t w = small_adj[i];
        vid_t l = 0;
        vid_t r = large_deg - 1;
        while (l <= r) {
            vid_t m = l + ((r - l) >> 1);
            vid_t val = large_adj[m];
            if (val == w) {
                ++triangles;
                if (triangles >= min_triangles) {
                    out_prune[gid] = 0;
                    return;
                }
                break;
            } else if (val < w) {
                l = m + 1;
            } else {
                if (m == 0) break;
                r = m - 1;
            }
        }
    }

    out_prune[gid] = (triangles < min_triangles) ? 1 : 0;
}

struct KTrussStep {
    int k = 0;
    int deleted_edges = 0;
    double step_ms = 0.0;
    double update_rate_per_sec = 0.0;
};

inline std::vector<KTrussStep> run_ktruss(const CSRGraph& sym_csr, int max_k = 100) {
    std::vector<KTrussStep> steps;
    vid_t nV = sym_csr.nV;
    eoff_t init_nE = sym_csr.nE;
    if (nV == 0 || init_nE == 0) return steps;

    HornetGraph hg(sym_csr);
    BatchWorkspace ws;
    ws.ensure_capacity(init_nE);

    eoff_t* d_offsets = nullptr;
    vid_t* d_all_u = nullptr;
    vid_t* d_all_v = nullptr;
    uint8_t* d_prune = nullptr;
    int* d_num_pruned = nullptr;

    CUDA_CHECK(cudaMalloc(&d_offsets, (nV + 1) * sizeof(eoff_t)));
    CUDA_CHECK(cudaMalloc(&d_all_u, init_nE * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_all_v, init_nE * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_prune, init_nE * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_num_pruned, sizeof(int)));

    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, hg.d_used(), d_offsets, nV + 1);
    CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));

    int k = 3;
    int blk = 256;

    while (k <= max_k) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Compute active edge offsets
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, hg.d_used(), d_offsets, nV + 1);
        eoff_t total_edges = 0;
        CUDA_CHECK(cudaMemcpy(&total_edges, &d_offsets[nV], sizeof(eoff_t), cudaMemcpyDeviceToHost));
        if (total_edges == 0) break;

        // 2. Count triangles per edge & flag edges with triangles < k - 2
        eoff_t grid = (total_edges + blk - 1) / blk;
        ktruss_count_and_flag_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            nV, d_offsets, total_edges, hg.d_used(), hg.d_ptr(), k - 2,
            d_all_u, d_all_v, d_prune);
        CUDA_CHECK(cudaGetLastError());

        // 3. Compact flagged edges directly into ws.d_raw_src / ws.d_raw_dst
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, d_all_u, d_prune, ws.d_raw_src, d_num_pruned, total_edges);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, d_all_v, d_prune, ws.d_raw_dst, d_num_pruned, total_edges);

        int num_pruned = 0;
        CUDA_CHECK(cudaMemcpy(&num_pruned, d_num_pruned, sizeof(int), cudaMemcpyDeviceToHost));

        if (num_pruned == 0) {
            ++k;
            continue;
        }

        // 4. Batch delete pruned edges from Hornet
        ws.batch_size = num_pruned;
        batch_delete(hg, ws);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double rate = (step_ms > 0.0) ? (static_cast<double>(num_pruned) / (step_ms / 1000.0)) : 0.0;

        steps.push_back({k, num_pruned, step_ms, rate});
    }

    cudaFree(d_scan_temp);
    cudaFree(d_offsets);
    cudaFree(d_all_u);
    cudaFree(d_all_v);
    cudaFree(d_prune);
    cudaFree(d_num_pruned);

    return steps;
}

// =============================================================================
// Dynamic Triangle Counting (Batch Insert + Batch Triangle Query)
// =============================================================================

struct DynamicTCResult {
    double insert_ms = 0.0;
    double query_ms = 0.0;
    double total_ms = 0.0;
    unsigned long long delta_triangles = 0;
};

__global__ void tc_batch_intersect_kernel(int num_edges,
                                            const vid_t* __restrict__ batch_u,
                                            const vid_t* __restrict__ batch_v,
                                            vid_t* const* __restrict__ d_ptrs,
                                            const vid_t* __restrict__ d_used,
                                            unsigned long long* __restrict__ d_total_tri) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (warp_id >= num_edges) return;

    vid_t u = batch_u[warp_id];
    vid_t v = batch_v[warp_id];
    if (u == v) return;

    vid_t du = d_used[u];
    vid_t dv = d_used[v];
    if (du == 0 || dv == 0) return;

    const vid_t* small_ptr = (du <= dv) ? d_ptrs[u] : d_ptrs[v];
    const vid_t* large_ptr = (du <= dv) ? d_ptrs[v] : d_ptrs[u];
    vid_t small_len = (du <= dv) ? du : dv;
    vid_t large_len = (du <= dv) ? dv : du;

    unsigned long long local_cnt = 0;
    for (vid_t i = lane; i < small_len; i += 32) {
        vid_t target = small_ptr[i];
        int lo = 0, hi = large_len - 1;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            vid_t val = large_ptr[mid];
            if (val == target) {
                ++local_cnt;
                break;
            }
            if (val < target) lo = mid + 1;
            else hi = mid - 1;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_cnt += __shfl_down_sync(0xffffffff, local_cnt, offset);
    }
    if (lane == 0 && local_cnt > 0) {
        atomicAdd(d_total_tri, local_cnt);
    }
}

inline DynamicTCResult run_dynamic_tc(const CSRGraph& sym_csr,
                                      int batch_size,
                                      uint64_t seed) {
    DynamicTCResult res;
    if (sym_csr.nV <= 1 || batch_size <= 0) return res;

    HornetGraph hg(sym_csr);

    // Generate undirected batch (u, v) and its symmetrized form for Hornet insertion
    std::vector<vid_t> raw_u, raw_v;
    generate_random_batch(sym_csr.nV, batch_size, seed, raw_u, raw_v);

    std::vector<vid_t> sym_u, sym_v;
    sym_u.reserve(batch_size * 2);
    sym_v.reserve(batch_size * 2);
    std::vector<vid_t> q_u, q_v;
    q_u.reserve(batch_size);
    q_v.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        if (raw_u[i] == raw_v[i]) continue;
        q_u.push_back(raw_u[i]);
        q_v.push_back(raw_v[i]);
        sym_u.push_back(raw_u[i]);
        sym_v.push_back(raw_v[i]);
        sym_u.push_back(raw_v[i]);
        sym_v.push_back(raw_u[i]);
    }
    int num_q = static_cast<int>(q_u.size());
    if (num_q == 0) return res;

    BatchWorkspace ws;
    ws.ensure_capacity(static_cast<int>(sym_u.size()));
    ws.prepare(sym_u, sym_v);

    vid_t *d_qu = nullptr, *d_qv = nullptr;
    unsigned long long* d_tri = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qu, num_q * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_qv, num_q * sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc(&d_tri, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_qu, q_u.data(), num_q * sizeof(vid_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qv, q_v.data(), num_q * sizeof(vid_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tri, 0, sizeof(unsigned long long)));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    batch_insert(hg, ws, true);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (num_q + warps_per_block - 1) / warps_per_block;
    tc_batch_intersect_kernel<<<blocks, threads>>>(
        num_q, d_qu, d_qv, hg.d_ptr(), hg.d_used(), d_tri);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(&res.delta_triangles, d_tri, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    res.insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.query_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    res.total_ms = res.insert_ms + res.query_ms;

    cudaFree(d_qu);
    cudaFree(d_qv);
    cudaFree(d_tri);
    return res;
}

} // namespace hornet_alg
