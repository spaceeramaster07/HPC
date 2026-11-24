#pragma once

#include "batch_workspace.cuh"
#include <thrust/iterator/counting_iterator.h>

namespace improved {

__global__ void gather_delete_graph_info_kernel(int num_unique,
                                                const vid_t* __restrict__ unique_srcs,
                                                const vid_t* __restrict__ d_used,
                                                vid_t* const* __restrict__ d_ptr,
                                                vid_t* __restrict__ all_old_deg,
                                                vid_t* __restrict__ all_new_deg,
                                                vid_t** __restrict__ all_old_ptr,
                                                uint8_t* __restrict__ need_shrink,
                                                uint8_t* __restrict__ need_free_only,
                                                eoff_t* __restrict__ graph_deg,
                                                int* __restrict__ med_indices,
                                                int* __restrict__ med_count,
                                                int* __restrict__ large_indices,
                                                int* __restrict__ large_count,
                                                int* __restrict__ huge_indices,
                                                int* __restrict__ huge_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == num_unique) {
        graph_deg[num_unique] = 0;
    }

    bool in_bounds = (i < num_unique);
    vid_t old_deg = 0;
    vid_t* adj = nullptr;
    if (in_bounds) {
        vid_t v = unique_srcs[i];
        old_deg = d_used[v];
        adj = d_ptr[v];
        all_old_deg[i] = old_deg;
        all_new_deg[i] = 0;
        all_old_ptr[i] = adj;
        need_shrink[i] = 0;
        need_free_only[i] = 0;
        graph_deg[i] = (adj != nullptr) ? static_cast<eoff_t>(old_deg) : 0;
    }

    bool valid = (in_bounds && adj != nullptr);
    warp_append_idx(valid && old_deg > 256 && old_deg <= 4096,
                    i, med_indices, med_count);
    warp_append_idx(valid && old_deg > 4096 && old_deg <= 16384,
                    i, large_indices, large_count);
    warp_append_idx(valid && old_deg > 16384,
                    i, huge_indices, huge_count);
}

// DS-5: O(batch_deg * log old_deg) deletion marking over sorted adjacency lists
__global__ void delete_mark_sorted_kernel(int num_unique,
                                          const eoff_t* __restrict__ batch_offsets,
                                          eoff_t total_batch_edges,
                                          const eoff_t* __restrict__ graph_offsets,
                                          const vid_t* __restrict__ all_old_deg,
                                          vid_t* const* __restrict__ all_old_ptr,
                                          const vid_t* __restrict__ batch_dst,
                                          uint8_t* __restrict__ del_keep) {
    int pos = 0;
    eoff_t local_idx = 0;
    eoff_t gid = 0;
    if (!binarySearchLB(num_unique, batch_offsets, total_batch_edges, pos, local_idx, gid)) {
        return;
    }

    vid_t old_deg = all_old_deg[pos];
    const vid_t* adj = all_old_ptr[pos];
    if (old_deg <= 0 || adj == nullptr) return;

    eoff_t g_base = graph_offsets[pos];
    vid_t target = batch_dst[gid];

    vid_t idx = dev_lower_bound(adj, old_deg, target);
    while (idx < old_deg && adj[idx] == target) {
        del_keep[g_base + idx] = 0;
        ++idx;
    }
}

__device__ __forceinline__ void finalize_delete_vertex_meta(int i,
                                                            vid_t v,
                                                            vid_t old_deg,
                                                            vid_t new_deg,
                                                            vid_t* __restrict__ d_used,
                                                            vid_t** __restrict__ d_ptr,
                                                            uint8_t* __restrict__ need_shrink,
                                                            uint8_t* __restrict__ need_free_only,
                                                            vid_t* __restrict__ all_new_deg,
                                                            int* __restrict__ any_free_only) {
    all_new_deg[i] = new_deg;
    d_used[v] = new_deg;

    int old_cap = 1 << dev_ceil_log2(old_deg);
    if (new_deg == 0) {
        d_ptr[v] = nullptr;
        need_free_only[i] = 1;
        need_shrink[i] = 0;
        *any_free_only = 1;
    } else if (new_deg < old_cap / 2) {
        need_free_only[i] = 0;
        need_shrink[i] = 1;
    } else {
        need_free_only[i] = 0;
        need_shrink[i] = 0;
    }
}

__device__ __forceinline__ int compact_1024_chunk_ballot(const vid_t* __restrict__ adj,
                                                         eoff_t g_base,
                                                         vid_t base,
                                                         vid_t old_deg,
                                                         const uint8_t* __restrict__ del_keep,
                                                         vid_t* __restrict__ out_buf,
                                                         eoff_t out_base,
                                                         bool skip_if_unchanged,
                                                         int s_warp_sums[4][8],
                                                         int& s_chunk_total) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    unsigned lower_mask = (1u << lane) - 1u;

    uint8_t keep[4];
    int warp_rank[4];

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        vid_t idx = base + static_cast<vid_t>(k * 256 + tid);
        bool active = (idx < old_deg);
        keep[k] = active ? del_keep[g_base + idx] : 0;

        unsigned ballot = __ballot_sync(0xFFFFFFFFu, keep[k] != 0);
        warp_rank[k] = __popc(ballot & lower_mask);
        if (lane == 0) {
            s_warp_sums[k][warp_id] = __popc(ballot);
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        int k_idx = lane >> 3;
        int w_idx = lane & 7;
        int val = s_warp_sums[k_idx][w_idx];
        int inc = val;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(0xFFFFFFFFu, inc, offset);
            if (lane >= offset) inc += n;
        }
        s_warp_sums[k_idx][w_idx] = inc - val;
        if (lane == 31) {
            s_chunk_total = inc;
        }
    }
    __syncthreads();

    int total = s_chunk_total;
    int expected_len = min(1024, static_cast<int>(old_deg - base));
    bool unchanged = skip_if_unchanged &&
                     (out_base == static_cast<eoff_t>(base)) &&
                     (total == expected_len);

    if (total > 0 && !unchanged) {
        vid_t nbr[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (keep[k] != 0) {
                nbr[k] = adj[base + static_cast<vid_t>(k * 256 + tid)];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (keep[k] != 0) {
                out_buf[out_base + s_warp_sums[k][warp_id] + warp_rank[k]] = nbr[k];
            }
        }
    }
    __syncthreads();
    return total;
}

__global__ void delete_compact_warp_kernel(int num_unique,
                                           const vid_t* __restrict__ unique_srcs,
                                           const eoff_t* __restrict__ graph_offsets,
                                           const vid_t* __restrict__ all_old_deg,
                                           vid_t* const* __restrict__ all_old_ptr,
                                           const uint8_t* __restrict__ del_keep,
                                           vid_t* __restrict__ d_used,
                                           vid_t** __restrict__ d_ptr,
                                           uint8_t* __restrict__ need_shrink,
                                           uint8_t* __restrict__ need_free_only,
                                           vid_t* __restrict__ all_new_deg,
                                           int* __restrict__ any_free_only) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid >> 5;
    int lane = tid & 31;
    if (i >= num_unique) return;

    vid_t old_deg = all_old_deg[i];
    vid_t* adj = all_old_ptr[i];
    if (old_deg == 0 || old_deg > 256 || adj == nullptr) return;

    vid_t v = unique_srcs[i];
    eoff_t g_base = graph_offsets[i];
    unsigned lower_mask = (1u << lane) - 1u;
    vid_t write_base = 0;

    for (vid_t base = 0; base < old_deg; base += 32) {
        vid_t idx = base + lane;
        bool active = (idx < old_deg);
        vid_t nbr = active ? adj[idx] : 0;
        uint8_t keep = active ? del_keep[g_base + idx] : 0;

        unsigned ballot = __ballot_sync(0xFFFFFFFFu, keep != 0);
        int local_rank = __popc(ballot & lower_mask);
        int chunk_kept = __popc(ballot);

        __syncwarp();
        if (keep != 0) {
            adj[write_base + local_rank] = nbr;
        }
        write_base += chunk_kept;
        __syncwarp();
    }

    if (lane == 0) {
        finalize_delete_vertex_meta(i, v, old_deg, write_base,
                                    d_used, d_ptr, need_shrink, need_free_only,
                                    all_new_deg, any_free_only);
    }
}

__global__ void delete_compact_block_kernel(int num_items,
                                            const int* __restrict__ item_indices,
                                            const vid_t* __restrict__ unique_srcs,
                                            const eoff_t* __restrict__ graph_offsets,
                                            const vid_t* __restrict__ all_old_deg,
                                            vid_t* const* __restrict__ all_old_ptr,
                                            const uint8_t* __restrict__ del_keep,
                                            vid_t* __restrict__ d_used,
                                            vid_t** __restrict__ d_ptr,
                                            uint8_t* __restrict__ need_shrink,
                                            uint8_t* __restrict__ need_free_only,
                                            vid_t* __restrict__ all_new_deg,
                                            int* __restrict__ any_free_only) {
    int b = blockIdx.x;
    if (b >= num_items) return;

    int i = item_indices[b];
    vid_t v = unique_srcs[i];
    vid_t old_deg = all_old_deg[i];
    vid_t* adj = all_old_ptr[i];
    eoff_t g_base = graph_offsets[i];

    __shared__ int s_warp_sums[4][8];
    __shared__ int s_chunk_total;

    vid_t cur_base = 0;
    for (vid_t base = 0; base < old_deg; base += 1024) {
        int chunk_kept = compact_1024_chunk_ballot(
            adj, g_base, base, old_deg, del_keep,
            adj, static_cast<eoff_t>(cur_base), true, s_warp_sums, s_chunk_total);
        cur_base += static_cast<vid_t>(chunk_kept);
    }

    if (threadIdx.x == 0) {
        finalize_delete_vertex_meta(i, v, old_deg, cur_base,
                                    d_used, d_ptr, need_shrink, need_free_only,
                                    all_new_deg, any_free_only);
    }
}

__global__ void gather_huge_delete_deg_kernel(int num_huge,
                                              const int* __restrict__ huge_indices,
                                              const vid_t* __restrict__ all_old_deg,
                                              vid_t* __restrict__ huge_old_deg) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < num_huge) {
        huge_old_deg[k] = all_old_deg[huge_indices[k]];
    } else if (k == num_huge) {
        huge_old_deg[num_huge] = 0;
    }
}

__global__ void gather_hub_chunks_kernel(int cnt,
                                         const int* __restrict__ chunk_huge_idx,
                                         const vid_t* __restrict__ all_old_deg,
                                         eoff_t* __restrict__ chunks_per_hub) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < cnt) {
        int i = chunk_huge_idx[k];
        chunks_per_hub[k] = (static_cast<int>(all_old_deg[i]) + 1023) >> 10;
    } else if (k == cnt) {
        chunks_per_hub[cnt] = 0;
    }
}

__global__ void hub_compact_count_chunks_kernel(int cnt,
                                                const eoff_t* __restrict__ hub_chunk_offsets,
                                                const int* __restrict__ chunk_huge_idx,
                                                const vid_t* __restrict__ all_old_deg,
                                                const eoff_t* __restrict__ graph_offsets,
                                                const uint8_t* __restrict__ del_keep,
                                                int* __restrict__ chunk_counts) {
    __shared__ int s_meta[2];
    __shared__ int s_warp[8];

    int tid = threadIdx.x;
    if (tid == 0) {
        int c_idx = dev_upper_bound_offsets(hub_chunk_offsets, cnt, static_cast<eoff_t>(blockIdx.x));
        s_meta[0] = c_idx;
        s_meta[1] = static_cast<int>(static_cast<eoff_t>(blockIdx.x) - hub_chunk_offsets[c_idx]);
    }
    __syncthreads();

    int c_idx = s_meta[0];
    int local_chunk = s_meta[1];
    int i = chunk_huge_idx[c_idx];
    vid_t old_deg = all_old_deg[i];
    eoff_t g_base = graph_offsets[i];
    vid_t base = static_cast<vid_t>(local_chunk) * 1024;

    int local_sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        vid_t idx = base + static_cast<vid_t>(k * 256 + tid);
        if (idx < old_deg && del_keep[g_base + idx] != 0) {
            ++local_sum;
        }
    }

    int warp_id = tid >> 5;
    int lane = tid & 31;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }
    if (lane == 0) {
        s_warp[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        int val = (lane < 8) ? s_warp[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        }
        if (lane == 0) {
            chunk_counts[blockIdx.x] = val;
        }
    }
}

__global__ void hub_compact_to_scratch_kernel(int cnt,
                                              const eoff_t* __restrict__ hub_chunk_offsets,
                                              const int* __restrict__ chunk_counts,
                                              const eoff_t* __restrict__ chunk_prefix,
                                              const int* __restrict__ chunk_huge_idx,
                                              const vid_t* __restrict__ unique_srcs,
                                              const eoff_t* __restrict__ graph_offsets,
                                              const vid_t* __restrict__ all_old_deg,
                                              vid_t* const* __restrict__ all_old_ptr,
                                              const uint8_t* __restrict__ del_keep,
                                              vid_t* __restrict__ d_merge_scratch,
                                              vid_t* __restrict__ d_used,
                                              vid_t** __restrict__ d_ptr,
                                              uint8_t* __restrict__ need_shrink,
                                              uint8_t* __restrict__ need_free_only,
                                              vid_t* __restrict__ all_new_deg,
                                              int* __restrict__ any_free_only) {
    __shared__ int s_meta[4];
    __shared__ int s_warp_sums[4][8];
    __shared__ int s_chunk_total;

    if (threadIdx.x == 0) {
        int c_idx = dev_upper_bound_offsets(hub_chunk_offsets, cnt, static_cast<eoff_t>(blockIdx.x));
        s_meta[0] = c_idx;
        s_meta[1] = static_cast<int>(static_cast<eoff_t>(blockIdx.x) - hub_chunk_offsets[c_idx]);
        s_meta[2] = static_cast<int>(hub_chunk_offsets[c_idx]);
        s_meta[3] = static_cast<int>(hub_chunk_offsets[c_idx + 1]);
    }
    __syncthreads();

    int c_idx = s_meta[0];
    int local_chunk = s_meta[1];
    int c0 = s_meta[2];
    int c1 = s_meta[3];

    int i = chunk_huge_idx[c_idx];
    vid_t old_deg = all_old_deg[i];
    vid_t new_deg = static_cast<vid_t>(chunk_prefix[c1] - chunk_prefix[c0]);

    if (local_chunk == 0 && threadIdx.x == 0) {
        finalize_delete_vertex_meta(i, unique_srcs[i], old_deg, new_deg,
                                    d_used, d_ptr, need_shrink, need_free_only,
                                    all_new_deg, any_free_only);
    }

    int chunk_kept = chunk_counts[blockIdx.x];
    eoff_t local_write_base = chunk_prefix[blockIdx.x] - chunk_prefix[c0];
    int expected_len = min(1024, static_cast<int>(old_deg) - local_chunk * 1024);
    if (chunk_kept == 0 || (local_write_base == static_cast<eoff_t>(local_chunk) * 1024 && chunk_kept == expected_len)) {
        return;
    }

    compact_1024_chunk_ballot(
        all_old_ptr[i], graph_offsets[i], static_cast<vid_t>(local_chunk) * 1024,
        old_deg, del_keep, d_merge_scratch, chunk_prefix[blockIdx.x], false,
        s_warp_sums, s_chunk_total);
}

__global__ void hub_copy_from_scratch_kernel(int cnt,
                                             const eoff_t* __restrict__ hub_chunk_offsets,
                                             const int* __restrict__ chunk_counts,
                                             const eoff_t* __restrict__ chunk_prefix,
                                             const int* __restrict__ chunk_huge_idx,
                                             const vid_t* __restrict__ all_old_deg,
                                             vid_t* const* __restrict__ all_old_ptr,
                                             const vid_t* __restrict__ d_merge_scratch) {
    __shared__ int s_meta[3];

    if (threadIdx.x == 0) {
        int c_idx = dev_upper_bound_offsets(hub_chunk_offsets, cnt, static_cast<eoff_t>(blockIdx.x));
        s_meta[0] = c_idx;
        s_meta[1] = static_cast<int>(static_cast<eoff_t>(blockIdx.x) - hub_chunk_offsets[c_idx]);
        s_meta[2] = static_cast<int>(hub_chunk_offsets[c_idx]);
    }
    __syncthreads();

    int c_idx = s_meta[0];
    int local_chunk = s_meta[1];
    int c0 = s_meta[2];

    int i = chunk_huge_idx[c_idx];
    vid_t old_deg = all_old_deg[i];
    int chunk_kept = chunk_counts[blockIdx.x];
    eoff_t local_write_base = chunk_prefix[blockIdx.x] - chunk_prefix[c0];
    int expected_len = min(1024, static_cast<int>(old_deg) - local_chunk * 1024);
    if (chunk_kept == 0 || (local_write_base == static_cast<eoff_t>(local_chunk) * 1024 && chunk_kept == expected_len)) {
        return;
    }

    vid_t* adj = all_old_ptr[i];
    eoff_t src_base = chunk_prefix[blockIdx.x];
    for (int k = threadIdx.x; k < chunk_kept; k += 256) {
        adj[local_write_base + k] = d_merge_scratch[src_base + k];
    }
}

inline void launch_hub_compact_chunk(HornetGraph& graph,
                                     BatchWorkspace& ws,
                                     const int* d_indices,
                                     int c_start,
                                     int cnt,
                                     int* d_any_free_only) {
    if (cnt <= 0) return;

    int blk = 256;
    const int* d_chunk_huge_idx = d_indices + c_start;
    eoff_t* d_chunks_per_hub = ws.hub_chunks_per_hub_buf();
    eoff_t* d_hub_chunk_offsets = ws.d_move_offsets;

    gather_hub_chunks_kernel<<<(cnt + 1 + blk - 1) / blk, blk>>>(
        cnt, d_chunk_huge_idx, ws.d_all_old_deg, d_chunks_per_hub);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceScan::ExclusiveSum(
        ws.d_cub_temp, ws.cub_temp_bytes, d_chunks_per_hub, d_hub_chunk_offsets, cnt + 1);

    eoff_t total_chunks = 0;
    CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &d_hub_chunk_offsets[cnt],
                          sizeof(eoff_t), cudaMemcpyDeviceToHost));
    total_chunks = ws.h_pinned_counts[0];
    if (total_chunks <= 0) return;

    ws.ensure_hub_chunk_capacity(static_cast<size_t>(total_chunks) + 1);

    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, ws.d_hub_chunk_counts, ws.d_hub_chunk_prefix, total_chunks + 1);
    ws.ensure_cub_temp(scan_bytes);

    hub_compact_count_chunks_kernel<<<static_cast<unsigned int>(total_chunks), 256>>>(
        cnt, d_hub_chunk_offsets, d_chunk_huge_idx, ws.d_all_old_deg,
        ws.d_graph_offsets, ws.d_del_keep, ws.d_hub_chunk_counts);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(&ws.d_hub_chunk_counts[total_chunks], 0, sizeof(int)));
    cub::DeviceScan::ExclusiveSum(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_hub_chunk_counts,
        ws.d_hub_chunk_prefix, total_chunks + 1);

    hub_compact_to_scratch_kernel<<<static_cast<unsigned int>(total_chunks), 256>>>(
        cnt, d_hub_chunk_offsets, ws.d_hub_chunk_counts, ws.d_hub_chunk_prefix,
        d_chunk_huge_idx, ws.d_unique_srcs, ws.d_graph_offsets,
        ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_del_keep,
        ws.d_merge_scratch, graph.d_used(), graph.d_ptr(),
        ws.d_need_realloc, ws.d_need_free_only, ws.d_all_new_deg, d_any_free_only);
    CUDA_CHECK(cudaGetLastError());

    hub_copy_from_scratch_kernel<<<static_cast<unsigned int>(total_chunks), 256>>>(
        cnt, d_hub_chunk_offsets, ws.d_hub_chunk_counts, ws.d_hub_chunk_prefix,
        d_chunk_huge_idx, ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_merge_scratch);
    CUDA_CHECK(cudaGetLastError());
}

inline void compact_hubs_multiblock(HornetGraph& graph,
                                    BatchWorkspace& ws,
                                    const int* d_indices,
                                    int count,
                                    int* d_any_free_only) {
    if (count <= 0) return;

    int blk = 256;
    vid_t* d_huge_old_deg = ws.huge_old_deg_buf();
    eoff_t* d_huge_old_offsets = reinterpret_cast<eoff_t*>(ws.d_keys_sorted);

    gather_huge_delete_deg_kernel<<<(count + 1 + blk - 1) / blk, blk>>>(
        count, d_indices, ws.d_all_old_deg, d_huge_old_deg);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceScan::ExclusiveSum(
        ws.d_cub_temp, ws.cub_temp_bytes, d_huge_old_deg,
        d_huge_old_offsets, count + 1);

    eoff_t total_huge_edges = 0;
    CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &d_huge_old_offsets[count],
                          sizeof(eoff_t), cudaMemcpyDeviceToHost));
    total_huge_edges = ws.h_pinned_counts[0];

    if (total_huge_edges <= static_cast<eoff_t>(BatchWorkspace::MAX_SCRATCH_EDGES)) {
        ws.ensure_scratch_capacity(static_cast<size_t>(total_huge_edges));
        launch_hub_compact_chunk(graph, ws, d_indices, 0, count, d_any_free_only);
    } else {
        std::vector<vid_t> h_deg(count);
        CUDA_CHECK(cudaMemcpy(h_deg.data(), d_huge_old_deg,
                              count * sizeof(vid_t), cudaMemcpyDeviceToHost));
        vid_t max_single = *std::max_element(h_deg.begin(), h_deg.end());
        size_t cap = std::max(BatchWorkspace::MAX_SCRATCH_EDGES,
                              static_cast<size_t>(max_single));
        ws.ensure_scratch_capacity(cap);

        int c_start = 0;
        while (c_start < count) {
            size_t sum = 0;
            int c_end = c_start;
            while (c_end < count) {
                size_t d = static_cast<size_t>(h_deg[c_end]);
                if (c_end > c_start && sum + d > cap) break;
                sum += d;
                ++c_end;
            }
            launch_hub_compact_chunk(graph, ws, d_indices, c_start, c_end - c_start, d_any_free_only);
            c_start = c_end;
        }
    }
}

inline void record_zero_delete_remaining_phases(bool record_mark,
                                                bool record_compact) {
    if (!HornetProfiler::instance().is_enabled()) return;
    if (record_mark) HornetProfiler::instance().record_phase("delete_mark", 0.0);
    if (record_compact) HornetProfiler::instance().record_phase("delete_compact", 0.0);
    HornetProfiler::instance().record_phase("delete_gpu_alloc", 0.0);
    HornetProfiler::instance().record_phase("delete_host_alloc", 0.0);
    HornetProfiler::instance().record_phase("delete_h2d_copy", 0.0);
    HornetProfiler::instance().record_phase("delete_adj_move", 0.0);
    HornetProfiler::instance().record_phase("delete_host_free", 0.0);
}

/**
 * @brief Algorithm 1: Dynamic Batch Deletion (Paper Section III-E, Footnote 2).
 *
 * DS-5 + DS-7: O(batch_deg * log old_deg) edge marking over sorted adjacency lists +
 * order-preserving parallel compaction + GPU BitTree shrink allocation & block
 * reclamation + balanced shrinking moves.
 */
inline void batch_delete(HornetGraph& graph, BatchWorkspace& ws) {
    ScopedPhaseTimer total_timer("delete_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    const bool prof = HornetProfiler::instance().is_enabled();
    int blk = 256;
    int vid_bits = (graph.nV() > 1) ? hornet_common::ceil_log2(graph.nV()) : 1;
    if (vid_bits < 1) vid_bits = 1;
    if (vid_bits > 32) vid_bits = 32;

    int uniq_count = 0;
    int num_unique = 0;
    eoff_t total_graph_work = 0;
    int num_med = 0;
    int num_large = 0;
    int num_huge = 0;

    int* d_med_indices = ws.d_realloc_v;
    int* d_large_indices = ws.d_selected_idx;
    int* d_huge_indices = reinterpret_cast<int*>(ws.d_keys_uniq) + (ws.capacity + 1);

    // -------------------------------------------------------------------------
    // Phase 1: Preprocessing (Sort, Deduplicate, Batch CSR, Graph Work Offsets)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer prep_timer("delete_preprocess", true);

        if (batch_size <= 256) {
            // DS-3: Single-Block Shared-Memory Preprocessing Fast Path
            preprocess_small_batch_kernel<<<1, 256>>>(
                batch_size, ws.d_raw_src, ws.d_raw_dst, vid_bits,
                ws.d_src_uniq, ws.d_dst_uniq, ws.d_unique_srcs,
                ws.d_batch_deg, ws.d_batch_offsets, ws.d_scalar_count);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  2 * sizeof(int), cudaMemcpyDeviceToHost));
            uniq_count = ws.h_pinned_counts[0];
            num_unique = ws.h_pinned_counts[1];
        } else {
            // DS-2: Tight Bit-Packing & Partial end_bit
            pack_edges_kernel<<<(batch_size + blk - 1) / blk, blk>>>(
                batch_size, ws.d_raw_src, ws.d_raw_dst, ws.d_keys_in, vid_bits);
            CUDA_CHECK(cudaGetLastError());

            int sort_bits = std::min(64, 2 * vid_bits);
            cub::DeviceRadixSort::SortKeys(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_in, ws.d_keys_sorted,
                batch_size, 0, sort_bits);
            cub::DeviceSelect::Unique(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_sorted, ws.d_keys_uniq,
                ws.d_scalar_count, batch_size);

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            uniq_count = ws.h_pinned_counts[0];
            if (uniq_count == 0) {
                record_zero_delete_remaining_phases(true, true);
                return;
            }

            unpack_edges_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
                uniq_count, ws.d_keys_uniq, ws.d_src_uniq, ws.d_dst_uniq, vid_bits);
            CUDA_CHECK(cudaGetLastError());

            cub::DeviceRunLengthEncode::Encode(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_unique_srcs,
                ws.d_batch_deg, ws.d_scalar_count, uniq_count);

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            num_unique = ws.h_pinned_counts[0];
            if (num_unique == 0) {
                record_zero_delete_remaining_phases(true, true);
                return;
            }

            CUDA_CHECK(cudaMemsetAsync(&ws.d_batch_deg[num_unique], 0, sizeof(int)));
            cub::DeviceScan::ExclusiveSum(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets,
                num_unique + 1);
        }

        if (uniq_count == 0 || num_unique == 0) {
            record_zero_delete_remaining_phases(true, true);
            return;
        }

        CUDA_CHECK(cudaMemsetAsync(ws.d_scalar_count, 0, 3 * sizeof(int)));
        gather_delete_graph_info_kernel<<<(num_unique + 1 + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, graph.d_used(), graph.d_ptr(),
            ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr,
            ws.d_need_realloc, ws.d_need_free_only, ws.d_graph_deg,
            d_med_indices, &ws.d_scalar_count[0],
            d_large_indices, &ws.d_scalar_count[1],
            d_huge_indices, &ws.d_scalar_count[2]);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_graph_deg, ws.d_graph_offsets,
            num_unique + 1);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              3 * sizeof(int), cudaMemcpyDeviceToHost));
        num_med = ws.h_pinned_counts[0];
        num_large = ws.h_pinned_counts[1];
        num_huge = ws.h_pinned_counts[2];
        CUDA_CHECK(cudaMemcpy(&ws.h_pinned_counts[3], &ws.d_graph_offsets[num_unique],
                              sizeof(eoff_t), cudaMemcpyDeviceToHost));
        total_graph_work = ws.h_pinned_counts[3];
    }

    if (total_graph_work == 0) {
        record_zero_delete_remaining_phases(true, true);
        return;
    }

    // -------------------------------------------------------------------------
    // Phase 2: DS-5 O(batch_deg * log old_deg) Mark Surviving vs Deleted Edges
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer mark_timer("delete_mark", true);
        ws.ensure_del_keep_capacity(static_cast<size_t>(total_graph_work));
        CUDA_CHECK(cudaMemsetAsync(ws.d_del_keep, 1,
                                   static_cast<size_t>(total_graph_work) * sizeof(uint8_t)));
        eoff_t grid = (static_cast<eoff_t>(uniq_count) + blk - 1) / blk;
        delete_mark_sorted_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            num_unique, ws.d_batch_offsets, static_cast<eoff_t>(uniq_count),
            ws.d_graph_offsets, ws.d_all_old_deg, ws.d_all_old_ptr,
            ws.d_dst_uniq, ws.d_del_keep);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------------------------------------------------------------------------
    // Phase 3: Order-Preserving Parallel Compaction + Collect Shrink/Free Lists
    // -------------------------------------------------------------------------
    int any_free_only = 0;
    int num_shrink = 0;

    {
        ScopedPhaseTimer compact_timer("delete_compact", true);
        CUDA_CHECK(cudaMemsetAsync(ws.d_scalar_count, 0, 2 * sizeof(int)));
        int* d_any_free_only = &ws.d_scalar_count[1];

        int warps_per_blk = blk / 32;
        int warp_grid = (num_unique + warps_per_blk - 1) / warps_per_blk;
        delete_compact_warp_kernel<<<warp_grid, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_graph_offsets,
            ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_del_keep,
            graph.d_used(), graph.d_ptr(),
            ws.d_need_realloc, ws.d_need_free_only, ws.d_all_new_deg, d_any_free_only);
        CUDA_CHECK(cudaGetLastError());

        if (num_med > 0) {
            delete_compact_block_kernel<<<num_med, 256>>>(
                num_med, d_med_indices, ws.d_unique_srcs, ws.d_graph_offsets,
                ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_del_keep,
                graph.d_used(), graph.d_ptr(),
                ws.d_need_realloc, ws.d_need_free_only, ws.d_all_new_deg, d_any_free_only);
            CUDA_CHECK(cudaGetLastError());
        }

        if (num_large > 0) {
            if (num_large <= 512) {
                compact_hubs_multiblock(graph, ws, d_large_indices, num_large, d_any_free_only);
            } else {
                delete_compact_block_kernel<<<num_large, 256>>>(
                    num_large, d_large_indices, ws.d_unique_srcs, ws.d_graph_offsets,
                    ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_del_keep,
                    graph.d_used(), graph.d_ptr(),
                    ws.d_need_realloc, ws.d_need_free_only, ws.d_all_new_deg, d_any_free_only);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (num_huge > 0) {
            compact_hubs_multiblock(graph, ws, d_huge_indices, num_huge, d_any_free_only);
        }

        // DS-2: Collect shrink indices where new_deg < bsize(old_deg) / 2
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
            ws.d_need_realloc, ws.d_selected_idx, &ws.d_scalar_count[0], num_unique);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              2 * sizeof(int), cudaMemcpyDeviceToHost));
        num_shrink = ws.h_pinned_counts[0];
        any_free_only = ws.h_pinned_counts[1];

        if (num_shrink > 0) {
            gather_realloc_metadata_kernel<<<(num_shrink + blk - 1) / blk, blk>>>(
                num_shrink, ws.d_selected_idx, ws.d_unique_srcs,
                ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr,
                ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_deg,
                ws.d_realloc_old_ptr);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4-7: DS-7 GPU Shrink Allocation, Balanced Move, GPU Reclamation
    // -------------------------------------------------------------------------
    if (num_shrink > 0) {
        double gpu_alloc_ms = 0.0;
        std::chrono::high_resolution_clock::time_point alloc_t0;
        bool alloc_nvtx = false;
        if (prof) {
            if (HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePushA("delete_gpu_alloc");
                alloc_nvtx = true;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            alloc_t0 = std::chrono::high_resolution_clock::now();
        }

        ds7_gpu_allocate_blocks(
            graph.memory_manager(), ws.ds7_alloc_view(),
            num_shrink, ws.d_realloc_new_deg, ws.d_realloc_new_ptr,
            prof, "delete_gpu_alloc", "delete_host_alloc", "delete_h2d_copy",
            &alloc_t0, &alloc_nvtx, &gpu_alloc_ms);

        if (prof) {
            if (alloc_nvtx) {
                cudaDeviceSynchronize();
                auto alloc_t1 = std::chrono::high_resolution_clock::now();
                gpu_alloc_ms += std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();
                nvtxRangePop();
            }
            HornetProfiler::instance().record_phase("delete_gpu_alloc", gpu_alloc_ms);
        }

        {
            ScopedPhaseTimer move_timer("delete_adj_move", true);

            // DS-2: GPU ExclusiveSum for move_offsets
            CUDA_CHECK(cudaMemsetAsync(&ws.d_realloc_new_deg[num_shrink], 0, sizeof(vid_t)));
            cub::DeviceScan::ExclusiveSum(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_realloc_new_deg,
                ws.d_move_offsets, num_shrink + 1);

            eoff_t total_moves = 0;
            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &ws.d_move_offsets[num_shrink],
                                  sizeof(eoff_t), cudaMemcpyDeviceToHost));
            total_moves = ws.h_pinned_counts[0];

            if (total_moves > 0) {
                eoff_t grid = (total_moves + blk - 1) / blk;
                adj_move_balanced_kernel<<<static_cast<unsigned int>(grid), blk>>>(
                    num_shrink, ws.d_move_offsets, total_moves,
                    ws.d_realloc_v, ws.d_realloc_old_ptr, ws.d_realloc_new_ptr, graph.d_ptr());
                CUDA_CHECK(cudaGetLastError());
            }

            // Separate GPU reclamation phase after move kernels finish
            if (prof && HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePushA("delete_gpu_reclaim");
            }
            ds7_gpu_reclaim_blocks(
                graph.memory_manager(), num_shrink, ws.d_realloc_old_deg, ws.d_realloc_old_ptr);
            if (any_free_only != 0) {
                ds7_gpu_reclaim_masked_blocks(
                    graph.memory_manager(), num_unique, ws.d_need_free_only,
                    ws.d_all_old_deg, ws.d_all_old_ptr);
            }
            if (prof && HornetProfiler::instance().is_nvtx_enabled()) {
                nvtxRangePop();
            }
        }

        if (prof) {
            HornetProfiler::instance().record_phase("delete_host_free", 0.0);
        }
    } else if (any_free_only != 0) {
        if (prof) {
            HornetProfiler::instance().record_phase("delete_gpu_alloc", 0.0);
            HornetProfiler::instance().record_phase("delete_host_alloc", 0.0);
            HornetProfiler::instance().record_phase("delete_h2d_copy", 0.0);
        }
        {
            ScopedPhaseTimer move_timer("delete_adj_move", true);
            ds7_gpu_reclaim_masked_blocks(
                graph.memory_manager(), num_unique, ws.d_need_free_only,
                ws.d_all_old_deg, ws.d_all_old_ptr);
        }
        if (prof) {
            HornetProfiler::instance().record_phase("delete_host_free", 0.0);
        }
    } else {
        record_zero_delete_remaining_phases(false, false);
    }
}

} // namespace improved
