#pragma once

#include "batch_workspace.cuh"
#include <thrust/iterator/counting_iterator.h>

namespace improved {

__global__ void gather_insert_graph_deg_kernel(int num_unique,
                                               const vid_t* __restrict__ unique_srcs,
                                               const vid_t* __restrict__ d_used,
                                               eoff_t* __restrict__ graph_deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_unique) {
        graph_deg[i] = static_cast<eoff_t>(d_used[unique_srcs[i]]);
    } else if (i == num_unique) {
        graph_deg[num_unique] = 0;
    }
}

// Paper Sec. III-E / Footnote 2: binarySearchLB over existing edges of touched vertices
__global__ void check_cross_dups_lb_kernel(int num_unique,
                                           const eoff_t* __restrict__ graph_offsets,
                                           eoff_t total_graph_work,
                                           const vid_t* __restrict__ unique_srcs,
                                           const eoff_t* __restrict__ batch_offsets,
                                           const int* __restrict__ batch_deg,
                                           const vid_t* __restrict__ batch_dst,
                                           vid_t* const* __restrict__ d_ptr,
                                           uint8_t* __restrict__ d_keep) {
    int pos = 0;
    eoff_t local_edge = 0;
    eoff_t gid = 0;
    if (!binarySearchLB(num_unique, graph_offsets, total_graph_work, pos, local_edge, gid)) {
        return;
    }

    vid_t u = unique_srcs[pos];
    uint32_t nbr = static_cast<uint32_t>(d_ptr[u][local_edge]);

    eoff_t b_start = batch_offsets[pos];
    int b_len = batch_deg[pos];

    int lo = 0;
    int hi = b_len - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        uint32_t val = static_cast<uint32_t>(batch_dst[b_start + mid]);
        if (val == nbr) {
            d_keep[b_start + mid] = 0;
            break;
        } else if (val < nbr) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
}

__global__ void check_insert_realloc_kernel(int num_unique,
                                            const vid_t* __restrict__ unique_srcs,
                                            const int* __restrict__ batch_deg,
                                            const vid_t* __restrict__ d_used,
                                            vid_t* const* __restrict__ d_ptr,
                                            uint8_t* __restrict__ need_realloc,
                                            vid_t* __restrict__ all_old_deg,
                                            vid_t* __restrict__ all_new_deg,
                                            vid_t** __restrict__ all_old_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    vid_t v = unique_srcs[i];
    vid_t old_deg = d_used[v];
    vid_t new_deg = old_deg + batch_deg[i];
    int old_cap = (old_deg == 0) ? 0 : (1 << dev_ceil_log2(old_deg));

    all_old_deg[i] = old_deg;
    all_new_deg[i] = new_deg;
    all_old_ptr[i] = d_ptr[v];
    need_realloc[i] = (new_deg > old_cap) ? 1 : 0;
}

__global__ void update_zero_deg_ptrs_kernel(int num_realloc,
                                            const vid_t* __restrict__ realloc_v,
                                            const vid_t* __restrict__ old_deg,
                                            vid_t* const* __restrict__ new_ptrs,
                                            vid_t** __restrict__ d_ptr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_realloc && old_deg[r] == 0) {
        d_ptr[realloc_v[r]] = new_ptrs[r];
    }
}

__global__ void append_batch_edges_lb_kernel(int num_unique,
                                             const eoff_t* __restrict__ batch_offsets,
                                             eoff_t final_count,
                                             const vid_t* __restrict__ unique_srcs,
                                             const vid_t* __restrict__ all_old_deg,
                                             const vid_t* __restrict__ all_new_deg,
                                             const vid_t* __restrict__ batch_dst,
                                             vid_t* __restrict__ d_used,
                                             vid_t* const* __restrict__ d_ptr) {
    int pos = 0;
    eoff_t local_offset = 0;
    eoff_t gid = 0;
    if (!binarySearchLB(num_unique, batch_offsets, final_count, pos, local_offset, gid)) {
        return;
    }

    vid_t v = unique_srcs[pos];
    vid_t old_deg = all_old_deg[pos];
    vid_t* adj = d_ptr[v];
    adj[old_deg + local_offset] = batch_dst[gid];

    if (local_offset == 0) {
        d_used[v] = all_new_deg[pos];
    }
}


inline void batch_insert(HornetGraph& graph,
                         BatchWorkspace& ws,
                         bool remove_graph_dups) {
    ScopedPhaseTimer total_timer("insert_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    int blk = 256;
    int vid_bits = (graph.nV() > 1) ? hornet_common::ceil_log2(graph.nV()) : 1;
    if (vid_bits < 1) vid_bits = 1;
    if (vid_bits > 32) vid_bits = 32;

    int num_unique = 0;
    int final_count = 0;
    vid_t* d_final_dst = nullptr;

    // -------------------------------------------------------------------------
    // Phase 1: Preprocessing (Sort, Deduplicate, Batch CSR)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer prep_timer("insert_preprocess", true);
        int uniq_count = 0;

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
        }

        if (uniq_count == 0 || num_unique == 0) return;

        vid_t* d_final_src = ws.d_src_uniq;
        d_final_dst = ws.d_dst_uniq;
        final_count = uniq_count;

        if (remove_graph_dups) {
            gather_insert_graph_deg_kernel<<<(num_unique + 1 + blk - 1) / blk, blk>>>(
                num_unique, ws.d_unique_srcs, graph.d_used(), ws.d_graph_deg);
            CUDA_CHECK(cudaGetLastError());

            cub::DeviceScan::ExclusiveSum(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_graph_deg, ws.d_graph_offsets,
                num_unique + 1);

            eoff_t total_graph_work = 0;
            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &ws.d_graph_offsets[num_unique],
                                  sizeof(eoff_t), cudaMemcpyDeviceToHost));
            total_graph_work = ws.h_pinned_counts[0];

            CUDA_CHECK(cudaMemsetAsync(ws.d_keep, 1, uniq_count * sizeof(uint8_t)));
            if (total_graph_work > 0) {
                eoff_t grid = (total_graph_work + blk - 1) / blk;
                check_cross_dups_lb_kernel<<<static_cast<unsigned int>(grid), blk>>>(
                    num_unique, ws.d_graph_offsets, total_graph_work,
                    ws.d_unique_srcs, ws.d_batch_offsets, ws.d_batch_deg, ws.d_dst_uniq,
                    graph.d_ptr(), ws.d_keep);
                CUDA_CHECK(cudaGetLastError());
            }

            // DS-2: 1 Flagged pass on counting_iterator + 1 gather kernel
            cub::DeviceSelect::Flagged(
                ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
                ws.d_keep, ws.d_selected_idx, ws.d_scalar_count, uniq_count);

            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            final_count = ws.h_pinned_counts[0];
            if (final_count == 0) return;

            gather_selected_edges_kernel<<<(final_count + blk - 1) / blk, blk>>>(
                final_count, ws.d_selected_idx, ws.d_src_uniq, ws.d_dst_uniq,
                ws.d_filt_src, ws.d_filt_dst);
            CUDA_CHECK(cudaGetLastError());

            d_final_src = ws.d_filt_src;
            d_final_dst = ws.d_filt_dst;

            if (final_count < uniq_count) {
                cub::DeviceRunLengthEncode::Encode(
                    ws.d_cub_temp, ws.cub_temp_bytes, d_final_src, ws.d_unique_srcs,
                    ws.d_batch_deg, ws.d_scalar_count, final_count);
                CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                num_unique = ws.h_pinned_counts[0];
                CUDA_CHECK(cudaMemsetAsync(&ws.d_batch_deg[num_unique], 0, sizeof(int)));
                cub::DeviceScan::ExclusiveSum(
                    ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets,
                    num_unique + 1);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 2: Build Reallocation List (Device -> Host)
    // -------------------------------------------------------------------------
    int num_realloc = 0;

    {
        ScopedPhaseTimer realloc_timer("insert_build_realloc", true);
        int blk = 256;

        check_insert_realloc_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_batch_deg, graph.d_used(), graph.d_ptr(),
            ws.d_need_realloc, ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr);
        CUDA_CHECK(cudaGetLastError());

        // DS-2: 1 Flagged pass on counting_iterator + 1 gather kernel
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, thrust::counting_iterator<int>(0),
            ws.d_need_realloc, ws.d_selected_idx, ws.d_scalar_count, num_unique);

        CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, ws.d_scalar_count,
                              sizeof(int), cudaMemcpyDeviceToHost));
        num_realloc = ws.h_pinned_counts[0];

        if (num_realloc > 0) {
            gather_realloc_metadata_kernel<<<(num_realloc + blk - 1) / blk, blk>>>(
                num_realloc, ws.d_selected_idx, ws.d_unique_srcs,
                ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr,
                ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_deg,
                ws.d_realloc_old_ptr);
            CUDA_CHECK(cudaGetLastError());

            // DS-2: Copy to persistent pinned host buffers
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_v, ws.d_realloc_v,
                                  num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_old_deg, ws.d_realloc_old_deg,
                                  num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_new_deg, ws.d_realloc_new_deg,
                                  num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ws.h_realloc_old_ptr, ws.d_realloc_old_ptr,
                                  num_realloc * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        }
    }

    if (num_realloc > 0) {
        // ---------------------------------------------------------------------
        // Phase 3: Host Block Allocation via B+Tree & VecTree
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer alloc_timer("insert_host_alloc", false);
            auto& mgr = graph.memory_manager();
            for (int r = 0; r < num_realloc; ++r) {
                int log_b = hornet_common::ceil_log2(ws.h_realloc_new_deg[r]);
                ws.h_realloc_new_ptr[r] = mgr.allocate_block(log_b, false).ptr;
            }
        }

        // ---------------------------------------------------------------------
        // Phase 4: Copy New Pointers Host -> Device
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer h2d_timer("insert_h2d_copy", true);
            CUDA_CHECK(cudaMemcpy(ws.d_realloc_new_ptr, ws.h_realloc_new_ptr,
                                  num_realloc * sizeof(vid_t*), cudaMemcpyHostToDevice));
        }

        // ---------------------------------------------------------------------
        // Phase 5: Adjacency List Move (Load-Balanced)
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer move_timer("insert_adj_move", true);

            // DS-2: GPU ExclusiveSum for move_offsets
            CUDA_CHECK(cudaMemsetAsync(&ws.d_realloc_old_deg[num_realloc], 0, sizeof(vid_t)));
            cub::DeviceScan::ExclusiveSum(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_realloc_old_deg,
                ws.d_move_offsets, num_realloc + 1);

            eoff_t total_moves = 0;
            CUDA_CHECK(cudaMemcpy(ws.h_pinned_counts, &ws.d_move_offsets[num_realloc],
                                  sizeof(eoff_t), cudaMemcpyDeviceToHost));
            total_moves = ws.h_pinned_counts[0];

            int blk = 256;
            update_zero_deg_ptrs_kernel<<<(num_realloc + blk - 1) / blk, blk>>>(
                num_realloc, ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_ptr, graph.d_ptr());
            CUDA_CHECK(cudaGetLastError());

            if (total_moves > 0) {
                eoff_t grid = (total_moves + blk - 1) / blk;
                adj_move_balanced_kernel<<<static_cast<unsigned int>(grid), blk>>>(
                    num_realloc, ws.d_move_offsets, total_moves,
                    ws.d_realloc_v, ws.d_realloc_old_ptr, ws.d_realloc_new_ptr, graph.d_ptr());
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // ---------------------------------------------------------------------
        // Phase 6: Host Block Reclamation via B+Tree & VecTree
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer free_timer("insert_host_free", false);
            auto& mgr = graph.memory_manager();
            for (int r = 0; r < num_realloc; ++r) {
                if (ws.h_realloc_old_deg[r] > 0 && ws.h_realloc_old_ptr[r] != nullptr) {
                    int old_log_b = hornet_common::ceil_log2(ws.h_realloc_old_deg[r]);
                    mgr.reclaim_block(ws.h_realloc_old_ptr[r], old_log_b);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 7: Append New Edges (binarySearchLB, Paper Alg. 1 lines 15-16)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer append_timer("insert_append", true);
        int blk = 256;
        eoff_t grid = (static_cast<eoff_t>(final_count) + blk - 1) / blk;
        append_batch_edges_lb_kernel<<<static_cast<unsigned int>(grid), blk>>>(
            num_unique, ws.d_batch_offsets, static_cast<eoff_t>(final_count),
            ws.d_unique_srcs, ws.d_all_old_deg, ws.d_all_new_deg,
            d_final_dst, graph.d_used(), graph.d_ptr());
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace improved
