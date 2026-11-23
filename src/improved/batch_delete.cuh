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
                                                int* __restrict__ large_indices,
                                                int* __restrict__ large_count) {
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

    bool is_large = (in_bounds && old_deg > 256 && adj != nullptr);
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_large);
    if (is_large) {
        int lane = threadIdx.x & 31;
        int leader = __ffs(mask) - 1;
        int base = 0;
        if (lane == leader) {
            base = atomicAdd(large_count, __popc(mask));
        }
        base = __shfl_sync(mask, base, leader);
        int rank = __popc(mask & ((1u << lane) - 1u));
        large_indices[base + rank] = i;
    }
}

__global__ void delete_mark_lb_kernel(int num_unique,
                                      const eoff_t* __restrict__ graph_offsets,
                                      eoff_t total_graph_work,
                                      vid_t* const* __restrict__ all_old_ptr,
                                      const eoff_t* __restrict__ batch_offsets,
                                      const int* __restrict__ batch_deg,
                                      const vid_t* __restrict__ batch_dst,
                                      uint8_t* __restrict__ del_keep) {
    int pos = 0;
    eoff_t local_edge = 0;
    eoff_t gid = 0;
    if (!binarySearchLB(num_unique, graph_offsets, total_graph_work, pos, local_edge, gid)) {
        return;
    }

    uint32_t nbr = static_cast<uint32_t>(all_old_ptr[pos][local_edge]);
    eoff_t b_start = batch_offsets[pos];
    int b_len = batch_deg[pos];

    int lo = 0;
    int hi = b_len - 1;
    uint8_t keep = 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        uint32_t val = static_cast<uint32_t>(batch_dst[b_start + mid]);
        if (val == nbr) {
            keep = 0;
            break;
        } else if (val < nbr) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    del_keep[gid] = keep;
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
                                           vid_t* __restrict__ all_new_deg) {
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
        vid_t new_deg = write_base;
        all_new_deg[i] = new_deg;
        d_used[v] = new_deg;

        int old_cap = 1 << dev_ceil_log2(old_deg);
        if (new_deg == 0) {
            d_ptr[v] = nullptr;
            need_free_only[i] = 1;
            need_shrink[i] = 0;
        } else if (new_deg < old_cap / 2) {
            need_free_only[i] = 0;
            need_shrink[i] = 1;
        } else {
            need_free_only[i] = 0;
            need_shrink[i] = 0;
        }
    }
}

__global__ void delete_compact_block_kernel(int num_large,
                                            const int* __restrict__ large_indices,
                                            const vid_t* __restrict__ unique_srcs,
                                            const eoff_t* __restrict__ graph_offsets,
                                            const vid_t* __restrict__ all_old_deg,
                                            vid_t* const* __restrict__ all_old_ptr,
                                            const uint8_t* __restrict__ del_keep,
                                            vid_t* __restrict__ d_used,
                                            vid_t** __restrict__ d_ptr,
                                            uint8_t* __restrict__ need_shrink,
                                            uint8_t* __restrict__ need_free_only,
                                            vid_t* __restrict__ all_new_deg) {
    int b = blockIdx.x;
    if (b >= num_large) return;

    int i = large_indices[b];
    vid_t v = unique_srcs[i];
    vid_t old_deg = all_old_deg[i];
    vid_t* adj = all_old_ptr[i];
    eoff_t g_base = graph_offsets[i];

    __shared__ int s_warp_sums[4][8];
    __shared__ int s_chunk_total;
    __shared__ vid_t s_write_base;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    unsigned lower_mask = (1u << lane) - 1u;

    if (tid == 0) {
        s_write_base = 0;
    }
    __syncthreads();

    for (vid_t base = 0; base < old_deg; base += 1024) {
        vid_t nbr[4];
        uint8_t keep[4];
        int warp_rank[4];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            vid_t idx = base + static_cast<vid_t>(k * 256 + tid);
            bool active = (idx < old_deg);
            nbr[k] = active ? adj[idx] : 0;
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

        vid_t cur_base = s_write_base;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (keep[k] != 0) {
                adj[cur_base + s_warp_sums[k][warp_id] + warp_rank[k]] = nbr[k];
            }
        }
        if (tid == 0) {
            s_write_base = cur_base + s_chunk_total;
        }
        __syncthreads();
    }

    if (tid == 0) {
        vid_t new_deg = s_write_base;
        all_new_deg[i] = new_deg;
        d_used[v] = new_deg;

        int old_cap = 1 << dev_ceil_log2(old_deg);
        if (new_deg == 0) {
            d_ptr[v] = nullptr;
            need_free_only[i] = 1;
            need_shrink[i] = 0;
        } else if (new_deg < old_cap / 2) {
            need_free_only[i] = 0;
            need_shrink[i] = 1;
        } else {
            need_free_only[i] = 0;
            need_shrink[i] = 0;
        }
    }
}

/**
 * @brief Algorithm 1: Dynamic Batch Deletion (Paper Section III-E, Footnote 2).
 *
 * Parallel edge-level marking via binarySearchLB + order-preserving parallel
 * in-place compaction (maintaining sorted adjacency lists for K-Truss) +
 * balanced shrinking moves via binarySearchLB.
 */
inline void batch_delete(HornetGraph& graph, BatchWorkspace& ws) {
    ScopedPhaseTimer total_timer("delete_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    int blk = 256;
    int vid_bits = (graph.nV() > 1) ? hornet_common::ceil_log2(graph.nV()) : 1;
    if (vid_bits < 1) vid_bits = 1;
    if (vid_bits > 32) vid_bits = 32;

    int uniq_count = 0;
    int num_unique = 0;
    eoff_t total_graph_work = 0;
    int num_large = 0;

    // -------------------------------------------------------------------------
    // Phase 1: Preprocessing (Sort, Deduplicate, Batch CSR, Graph Work Offsets)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer prep_timer("delete_preprocess", true);

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
        if (uniq_count == 0) return;

        unpack_edges_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
            uniq_count, ws.d_keys_uniq, ws.d_src_uniq, ws.d_dst_uniq, vid_bits);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceRunLengthEncode::Encode(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_unique_srcs,
            ws.d_batch_deg, ws.d_scalar_count, uniq_count);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              sizeof(int), cudaMemcpyDeviceToHost));
        num_unique = ws.h_pinned_counts[0];
        if (num_unique == 0) return;

        CUDA_CHECK(cudaMemsetAsync(&ws.d_batch_deg[num_unique], 0, sizeof(int)));
        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets,
            num_unique + 1);

        CUDA_CHECK(cudaMemsetAsync(ws.d_scalar_count, 0, sizeof(int)));
        gather_delete_graph_info_kernel<<<(num_unique + 1 + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, graph.d_used(), graph.d_ptr(),
            ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr,
            ws.d_need_realloc, ws.d_need_free_only, ws.d_graph_deg,
            ws.d_realloc_v, ws.d_scalar_count);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_graph_deg, ws.d_graph_offsets,
            num_unique + 1);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &ws.d_graph_offsets[num_unique],
                              sizeof(eoff_t), cudaMemcpyDeviceToHost));
        total_graph_work = ws.h_pinned_counts[0];
        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              sizeof(int), cudaMemcpyDeviceToHost));
        num_large = ws.h_pinned_counts[0];
    }

    if (total_graph_work == 0) return;

    // -------------------------------------------------------------------------
    // Phase 2: Mark Surviving vs Deleted Edges (binarySearchLB)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer mark_timer("delete_mark", true);
        ws.ensure_del_keep_capacity(static_cast<size_t>(total_graph_work));
        eoff_t grid = (total_graph_work + blk - 1) / blk;
        delete_mark_lb_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            num_unique, ws.d_graph_offsets, total_graph_work, ws.d_all_old_ptr,
            ws.d_batch_offsets, ws.d_batch_deg, ws.d_dst_uniq, ws.d_del_keep);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------------------------------------------------------------------------
    // Phase 3: Order-Preserving Parallel Compaction + Collect Shrink/Free Lists
    // -------------------------------------------------------------------------
    int num_free_only = 0;
    int num_shrink = 0;

    {
        ScopedPhaseTimer compact_timer("delete_compact", true);
        int warps_per_blk = blk / 32;
        int warp_grid = (num_unique + warps_per_blk - 1) / warps_per_blk;
        delete_compact_warp_kernel<<<warp_grid, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_graph_offsets,
            ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_del_keep,
            graph.d_used(), graph.d_ptr(),
            ws.d_need_realloc, ws.d_need_free_only, ws.d_all_new_deg);
        CUDA_CHECK(cudaGetLastError());

        if (num_large > 0) {
            delete_compact_block_kernel<<<num_large, 256>>>(
                num_large, ws.d_realloc_v, ws.d_unique_srcs, ws.d_graph_offsets,
                ws.d_all_old_deg, ws.d_all_old_ptr, ws.d_del_keep,
                graph.d_used(), graph.d_ptr(),
                ws.d_need_realloc, ws.d_need_free_only, ws.d_all_new_deg);
            CUDA_CHECK(cudaGetLastError());
        }

        // DS-2: Reclaim fully emptied blocks (new_deg == 0) via 1 Flagged index pass + 1 gather kernel
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
            ws.d_need_free_only, ws.d_selected_idx, ws.d_scalar_count, num_unique);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              sizeof(int), cudaMemcpyDeviceToHost));
        num_free_only = ws.h_pinned_counts[0];

        if (num_free_only > 0) {
            gather_free_metadata_kernel<<<(num_free_only + blk - 1) / blk, blk>>>(
                num_free_only, ws.d_selected_idx, ws.d_all_old_deg, ws.d_all_old_ptr,
                ws.d_realloc_old_deg, ws.d_realloc_old_ptr);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(ws.h_free_deg, ws.d_realloc_old_deg,
                                  num_free_only * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_free_ptr, ws.d_realloc_old_ptr,
                                  num_free_only * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        }

        // DS-2: Shrink blocks where new_deg < bsize(old_deg) / 2 via 1 Flagged index pass + 1 gather kernel
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
            ws.d_need_realloc, ws.d_selected_idx, ws.d_scalar_count, num_unique);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              sizeof(int), cudaMemcpyDeviceToHost));
        num_shrink = ws.h_pinned_counts[0];

        if (num_shrink > 0) {
            gather_realloc_metadata_kernel<<<(num_shrink + blk - 1) / blk, blk>>>(
                num_shrink, ws.d_selected_idx, ws.d_unique_srcs,
                ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr,
                ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_deg,
                ws.d_realloc_old_ptr);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(ws.h_realloc_v, ws.d_realloc_v,
                                  num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_old_deg, ws.d_realloc_old_deg,
                                  num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_new_deg, ws.d_realloc_new_deg,
                                  num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_old_ptr, ws.d_realloc_old_ptr,
                                  num_shrink * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4-6: Shrink Block Allocation, H2D Pointer Copy, Balanced Move
    // -------------------------------------------------------------------------
    if (num_shrink > 0) {
        {
            ScopedPhaseTimer alloc_timer("delete_host_alloc", false);
            auto& mgr = graph.memory_manager();
            for (int i = 0; i < num_shrink; ++i) {
                int log_b = hornet_common::ceil_log2(ws.h_realloc_new_deg[i]);
                ws.h_realloc_new_ptr[i] = mgr.allocate_block(log_b, false).ptr;
            }
        }

        {
            ScopedPhaseTimer h2d_timer("delete_h2d_copy", true);
            CUDA_CHECK(cudaMemcpy(ws.d_realloc_new_ptr, ws.h_realloc_new_ptr,
                                  num_shrink * sizeof(vid_t*), cudaMemcpyHostToDevice));
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
        }
    }

    // -------------------------------------------------------------------------
    // Phase 7: Host Block Reclamation (Emptied + Old Shrunk Blocks)
    // -------------------------------------------------------------------------
    if (num_free_only > 0 || num_shrink > 0) {
        ScopedPhaseTimer free_timer("delete_host_free", false);
        auto& mgr = graph.memory_manager();
        for (int i = 0; i < num_free_only; ++i) {
            if (ws.h_free_deg[i] > 0 && ws.h_free_ptr[i] != nullptr) {
                mgr.reclaim_block(ws.h_free_ptr[i], hornet_common::ceil_log2(ws.h_free_deg[i]));
            }
        }
        for (int i = 0; i < num_shrink; ++i) {
            if (ws.h_realloc_old_deg[i] > 0 && ws.h_realloc_old_ptr[i] != nullptr) {
                mgr.reclaim_block(ws.h_realloc_old_ptr[i], hornet_common::ceil_log2(ws.h_realloc_old_deg[i]));
            }
        }
    }
}

} // namespace improved
