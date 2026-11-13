#pragma once

#include "hornet_graph.cuh"
#include "types.cuh"

#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace baseline {

// Pack/unpack (src, dst) into 64-bit sortable keys
__global__ void pack_edges_kernel(int n,
                                  const vid_t* __restrict__ src,
                                  const vid_t* __restrict__ dst,
                                  uint64_t* __restrict__ keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = (static_cast<uint64_t>(static_cast<uint32_t>(src[i])) << 32) |
                  static_cast<uint32_t>(dst[i]);
    }
}

__global__ void unpack_edges_kernel(int n,
                                    const uint64_t* __restrict__ keys,
                                    vid_t* __restrict__ src,
                                    vid_t* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint64_t k = keys[i];
        src[i] = static_cast<vid_t>(static_cast<uint32_t>(k >> 32));
        dst[i] = static_cast<vid_t>(static_cast<uint32_t>(k & 0xFFFFFFFFULL));
    }
}

// 1 warp per batch edge: checks if dst already exists in d_ptr[src][0..d_used[src]-1]
__global__ void check_cross_dups_warp_kernel(int n_edges,
                                             const vid_t* __restrict__ batch_src,
                                             const vid_t* __restrict__ batch_dst,
                                             const vid_t* __restrict__ d_used,
                                             vid_t* const* __restrict__ d_ptr,
                                             uint8_t* __restrict__ d_keep) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (warp_id >= n_edges) return;

    vid_t u = batch_src[warp_id];
    vid_t v = batch_dst[warp_id];
    vid_t deg = d_used[u];
    const vid_t* adj = d_ptr[u];

    bool found = false;
    for (vid_t i = lane; i < deg; i += 32) {
        if (adj[i] == v) {
            found = true;
            break;
        }
    }
    unsigned mask = __ballot_sync(0xFFFFFFFFu, found);
    if (lane == 0) {
        d_keep[warp_id] = (mask == 0u) ? 1 : 0;
    }
}

__device__ __forceinline__ int dev_ceil_log2(int x) {
    if (x <= 1) return 0;
    return 32 - __clz(static_cast<unsigned int>(x - 1));
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

// Binary-search load-balanced adjacency list copy (Paper Footnote 2)
__global__ void adj_move_balanced_kernel(int num_realloc,
                                         const eoff_t* __restrict__ move_offsets,
                                         eoff_t total_moves,
                                         const vid_t* __restrict__ realloc_v,
                                         vid_t* const* __restrict__ old_ptrs,
                                         vid_t* const* __restrict__ new_ptrs,
                                         vid_t** __restrict__ d_ptr) {
    eoff_t gid = static_cast<eoff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_moves) return;

    int lo = 0;
    int hi = num_realloc - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (move_offsets[mid + 1] <= gid) {
            lo = mid + 1;
        } else if (move_offsets[mid] > gid) {
            hi = mid - 1;
        } else {
            lo = mid;
            break;
        }
    }
    int r = lo;
    eoff_t local_idx = gid - move_offsets[r];
    new_ptrs[r][local_idx] = old_ptrs[r][local_idx];

    if (local_idx == 0) {
        d_ptr[realloc_v[r]] = new_ptrs[r];
    }
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

__global__ void append_batch_edges_kernel(int num_unique,
                                          const vid_t* __restrict__ unique_srcs,
                                          const eoff_t* __restrict__ batch_offsets,
                                          const int* __restrict__ batch_deg,
                                          const vid_t* __restrict__ batch_dst,
                                          vid_t* __restrict__ d_used,
                                          vid_t* const* __restrict__ d_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    vid_t v = unique_srcs[i];
    vid_t old_deg = d_used[v];
    vid_t* adj = d_ptr[v];
    eoff_t b_off = batch_offsets[i];
    int add_deg = batch_deg[i];

    for (int k = 0; k < add_deg; ++k) {
        adj[old_deg + k] = batch_dst[b_off + k];
    }
    d_used[v] = old_deg + add_deg;
}

__global__ void delete_and_check_shrink_kernel(int num_unique,
                                               const vid_t* __restrict__ unique_srcs,
                                               const eoff_t* __restrict__ batch_offsets,
                                               const int* __restrict__ batch_deg,
                                               const vid_t* __restrict__ batch_dst,
                                               vid_t* __restrict__ d_used,
                                               vid_t** __restrict__ d_ptr,
                                               uint8_t* __restrict__ need_shrink,
                                               uint8_t* __restrict__ need_free_only,
                                               vid_t* __restrict__ all_old_deg,
                                               vid_t* __restrict__ all_new_deg,
                                               vid_t** __restrict__ all_old_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    vid_t v = unique_srcs[i];
    vid_t old_deg = d_used[v];
    vid_t* adj = d_ptr[v];
    all_old_deg[i] = old_deg;
    all_old_ptr[i] = adj;
    need_shrink[i] = 0;
    need_free_only[i] = 0;

    if (old_deg == 0 || adj == nullptr) {
        all_new_deg[i] = 0;
        return;
    }

    eoff_t b_off = batch_offsets[i];
    int del_deg = batch_deg[i];

    vid_t write_pos = 0;
    for (vid_t k = 0; k < old_deg; ++k) {
        vid_t nbr = adj[k];
        int lo = 0;
        int hi = del_deg - 1;
        bool deleted = false;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            vid_t val = batch_dst[b_off + mid];
            if (val == nbr) {
                deleted = true;
                break;
            } else if (val < nbr) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (!deleted) {
            adj[write_pos++] = nbr;
        }
    }

    vid_t new_deg = write_pos;
    all_new_deg[i] = new_deg;
    d_used[v] = new_deg;

    int old_cap = 1 << dev_ceil_log2(old_deg);
    if (new_deg == 0) {
        d_ptr[v] = nullptr;
        need_free_only[i] = 1;
    } else if (new_deg < old_cap / 2) {
        need_shrink[i] = 1;
    }
}

/**
 * @brief Reusable batch scratch workspace (matches original Hornet's BatchUpdate buffers).
 * Pre-allocates temporary CUB/batch arrays before insert()/erase() is timed.
 */
struct BatchWorkspace {
    int capacity = 0;
    int batch_size = 0;

    vid_t* d_raw_src = nullptr;
    vid_t* d_raw_dst = nullptr;
    uint64_t* d_keys_in = nullptr;
    uint64_t* d_keys_sorted = nullptr;
    uint64_t* d_keys_uniq = nullptr;
    vid_t* d_src_uniq = nullptr;
    vid_t* d_dst_uniq = nullptr;
    uint8_t* d_keep = nullptr;
    vid_t* d_filt_src = nullptr;
    vid_t* d_filt_dst = nullptr;

    vid_t* d_unique_srcs = nullptr;
    int* d_batch_deg = nullptr;
    eoff_t* d_batch_offsets = nullptr;

    uint8_t* d_need_realloc = nullptr;
    uint8_t* d_need_free_only = nullptr;
    vid_t* d_all_old_deg = nullptr;
    vid_t* d_all_new_deg = nullptr;
    vid_t** d_all_old_ptr = nullptr;

    vid_t* d_realloc_v = nullptr;
    vid_t* d_realloc_old_deg = nullptr;
    vid_t* d_realloc_new_deg = nullptr;
    vid_t** d_realloc_old_ptr = nullptr;
    vid_t** d_realloc_new_ptr = nullptr;
    eoff_t* d_move_offsets = nullptr;

    int* d_scalar_count = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    ~BatchWorkspace() { free_all(); }

    void free_all() {
        cudaFree(d_raw_src);
        cudaFree(d_raw_dst);
        cudaFree(d_keys_in);
        cudaFree(d_keys_sorted);
        cudaFree(d_keys_uniq);
        cudaFree(d_src_uniq);
        cudaFree(d_dst_uniq);
        cudaFree(d_keep);
        cudaFree(d_filt_src);
        cudaFree(d_filt_dst);
        cudaFree(d_unique_srcs);
        cudaFree(d_batch_deg);
        cudaFree(d_batch_offsets);
        cudaFree(d_need_realloc);
        cudaFree(d_need_free_only);
        cudaFree(d_all_old_deg);
        cudaFree(d_all_new_deg);
        cudaFree(d_all_old_ptr);
        cudaFree(d_realloc_v);
        cudaFree(d_realloc_old_deg);
        cudaFree(d_realloc_new_deg);
        cudaFree(d_realloc_old_ptr);
        cudaFree(d_realloc_new_ptr);
        cudaFree(d_move_offsets);
        cudaFree(d_scalar_count);
        cudaFree(d_cub_temp);

        d_raw_src = d_raw_dst = d_src_uniq = d_dst_uniq = nullptr;
        d_filt_src = d_filt_dst = d_unique_srcs = nullptr;
        d_keys_in = d_keys_sorted = d_keys_uniq = nullptr;
        d_keep = d_need_realloc = d_need_free_only = nullptr;
        d_batch_deg = nullptr;
        d_batch_offsets = d_move_offsets = nullptr;
        d_all_old_deg = d_all_new_deg = d_realloc_v = d_realloc_old_deg = d_realloc_new_deg = nullptr;
        d_all_old_ptr = d_realloc_old_ptr = d_realloc_new_ptr = nullptr;
        d_scalar_count = nullptr;
        d_cub_temp = nullptr;
        capacity = 0;
        cub_temp_bytes = 0;
    }

    void ensure_capacity(int n) {
        if (n <= capacity) return;
        free_all();
        capacity = std::max(n, 1);

        CUDA_CHECK(cudaMalloc(&d_raw_src, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_raw_dst, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_in, capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_sorted, capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_uniq, capacity * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_src_uniq, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_dst_uniq, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_keep, capacity * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_filt_src, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_filt_dst, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_unique_srcs, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_batch_deg, capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_batch_offsets, (capacity + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_need_realloc, capacity * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_need_free_only, capacity * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_all_old_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_all_new_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_all_old_ptr, capacity * sizeof(vid_t*)));
        CUDA_CHECK(cudaMalloc(&d_realloc_v, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_realloc_old_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_realloc_new_deg, capacity * sizeof(vid_t)));
        CUDA_CHECK(cudaMalloc(&d_realloc_old_ptr, capacity * sizeof(vid_t*)));
        CUDA_CHECK(cudaMalloc(&d_realloc_new_ptr, capacity * sizeof(vid_t*)));
        CUDA_CHECK(cudaMalloc(&d_move_offsets, (capacity + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_scalar_count, sizeof(int)));

        // Pre-compute max CUB temporary storage needed across all operations
        size_t req = 0, tmp = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, tmp, d_keys_in, d_keys_sorted, capacity);
        req = std::max(req, tmp);
        cub::DeviceSelect::Unique(nullptr, tmp, d_keys_sorted, d_keys_uniq, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceSelect::Flagged(nullptr, tmp, d_all_old_ptr, d_need_realloc, d_realloc_old_ptr, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceSelect::Flagged(nullptr, tmp, d_src_uniq, d_keep, d_filt_src, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceRunLengthEncode::Encode(nullptr, tmp, d_filt_src, d_unique_srcs, d_batch_deg, d_scalar_count, capacity);
        req = std::max(req, tmp);
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_batch_deg, d_batch_offsets, capacity);
        req = std::max(req, tmp);

        cub_temp_bytes = std::max<size_t>(req, 256);
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
    }

    void prepare(const std::vector<vid_t>& h_src, const std::vector<vid_t>& h_dst) {
        batch_size = static_cast<int>(h_src.size());
        ensure_capacity(batch_size);
        if (batch_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_raw_src, h_src.data(), batch_size * sizeof(vid_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_raw_dst, h_dst.data(), batch_size * sizeof(vid_t), cudaMemcpyHostToDevice));
        }
    }
};

/**
 * @brief Algorithm 1: Dynamic Batch Insertion (Paper Section III-E).
 */
inline void batch_insert(HornetGraph& graph,
                         BatchWorkspace& ws,
                         bool remove_graph_dups) {
    ScopedPhaseTimer total_timer("insert_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    int num_unique = 0;
    vid_t* d_final_dst = nullptr;

    // -------------------------------------------------------------------------
    // Phase 1: Preprocessing (Sort, Deduplicate, Batch CSR)
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer prep_timer("insert_preprocess", true);
        int blk = 256;

        pack_edges_kernel<<<(batch_size + blk - 1) / blk, blk>>>(
            batch_size, ws.d_raw_src, ws.d_raw_dst, ws.d_keys_in);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceRadixSort::SortKeys(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_in, ws.d_keys_sorted, batch_size);

        cub::DeviceSelect::Unique(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_sorted, ws.d_keys_uniq, ws.d_scalar_count, batch_size);

        int uniq_count = 0;
        CUDA_CHECK(cudaMemcpy(&uniq_count, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (uniq_count == 0) return;

        unpack_edges_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
            uniq_count, ws.d_keys_uniq, ws.d_src_uniq, ws.d_dst_uniq);
        CUDA_CHECK(cudaGetLastError());

        vid_t* d_final_src = ws.d_src_uniq;
        d_final_dst = ws.d_dst_uniq;
        int final_count = uniq_count;

        if (remove_graph_dups) {
            int warps_per_blk = blk / 32;
            int grid = (uniq_count + warps_per_blk - 1) / warps_per_blk;
            check_cross_dups_warp_kernel<<<grid, blk>>>(
                uniq_count, ws.d_src_uniq, ws.d_dst_uniq, graph.d_used(), graph.d_ptr(), ws.d_keep);
            CUDA_CHECK(cudaGetLastError());

            cub::DeviceSelect::Flagged(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_keep, ws.d_filt_src, ws.d_scalar_count, uniq_count);
            cub::DeviceSelect::Flagged(
                ws.d_cub_temp, ws.cub_temp_bytes, ws.d_dst_uniq, ws.d_keep, ws.d_filt_dst, ws.d_scalar_count, uniq_count);

            CUDA_CHECK(cudaMemcpy(&final_count, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
            d_final_src = ws.d_filt_src;
            d_final_dst = ws.d_filt_dst;
        }

        if (final_count == 0) return;

        cub::DeviceRunLengthEncode::Encode(
            ws.d_cub_temp, ws.cub_temp_bytes, d_final_src, ws.d_unique_srcs, ws.d_batch_deg, ws.d_scalar_count, final_count);
        CUDA_CHECK(cudaMemcpy(&num_unique, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));

        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets, num_unique);
    }

    // -------------------------------------------------------------------------
    // Phase 2: Build Reallocation List (Device -> Host)
    // -------------------------------------------------------------------------
    int num_realloc = 0;
    std::vector<vid_t> h_realloc_v;
    std::vector<vid_t> h_realloc_old_deg;
    std::vector<vid_t> h_realloc_new_deg;
    std::vector<vid_t*> h_realloc_old_ptr;

    {
        ScopedPhaseTimer realloc_timer("insert_build_realloc", true);
        int blk = 256;

        check_insert_realloc_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_batch_deg, graph.d_used(), graph.d_ptr(),
            ws.d_need_realloc, ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_unique_srcs, ws.d_need_realloc, ws.d_realloc_v, ws.d_scalar_count, num_unique);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_deg, ws.d_need_realloc, ws.d_realloc_old_deg, ws.d_scalar_count, num_unique);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_new_deg, ws.d_need_realloc, ws.d_realloc_new_deg, ws.d_scalar_count, num_unique);
        cub::DeviceSelect::Flagged(
            ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_ptr, ws.d_need_realloc, ws.d_realloc_old_ptr, ws.d_scalar_count, num_unique);

        CUDA_CHECK(cudaMemcpy(&num_realloc, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (num_realloc > 0) {
            h_realloc_v.resize(num_realloc);
            h_realloc_old_deg.resize(num_realloc);
            h_realloc_new_deg.resize(num_realloc);
            h_realloc_old_ptr.resize(num_realloc);

            CUDA_CHECK(cudaMemcpy(h_realloc_v.data(), ws.d_realloc_v, num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_realloc_old_deg.data(), ws.d_realloc_old_deg, num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_realloc_new_deg.data(), ws.d_realloc_new_deg, num_realloc * sizeof(vid_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_realloc_old_ptr.data(), ws.d_realloc_old_ptr, num_realloc * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        }
    }

    if (num_realloc > 0) {
        std::vector<vid_t*> h_realloc_new_ptr(num_realloc, nullptr);

        // ---------------------------------------------------------------------
        // Phase 3: Host Block Allocation via B+Tree & VecTree
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer alloc_timer("insert_host_alloc", false);
            auto& mgr = graph.memory_manager();
            for (int r = 0; r < num_realloc; ++r) {
                int log_b = ceil_log2(h_realloc_new_deg[r]);
                h_realloc_new_ptr[r] = mgr.allocate_block(log_b, false).ptr;
            }
        }

        // ---------------------------------------------------------------------
        // Phase 4: Copy New Pointers Host -> Device
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer h2d_timer("insert_h2d_copy", true);
            CUDA_CHECK(cudaMemcpy(ws.d_realloc_new_ptr, h_realloc_new_ptr.data(),
                                  num_realloc * sizeof(vid_t*), cudaMemcpyHostToDevice));
        }

        // ---------------------------------------------------------------------
        // Phase 5: Adjacency List Move (Load-Balanced)
        // ---------------------------------------------------------------------
        {
            ScopedPhaseTimer move_timer("insert_adj_move", true);

            std::vector<eoff_t> h_move_offsets(num_realloc + 1, 0);
            for (int r = 0; r < num_realloc; ++r) {
                h_move_offsets[r + 1] = h_move_offsets[r] + h_realloc_old_deg[r];
            }
            eoff_t total_moves = h_move_offsets[num_realloc];

            int blk = 256;
            update_zero_deg_ptrs_kernel<<<(num_realloc + blk - 1) / blk, blk>>>(
                num_realloc, ws.d_realloc_v, ws.d_realloc_old_deg, ws.d_realloc_new_ptr, graph.d_ptr());
            CUDA_CHECK(cudaGetLastError());

            if (total_moves > 0) {
                CUDA_CHECK(cudaMemcpy(ws.d_move_offsets, h_move_offsets.data(),
                                      (num_realloc + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));

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
                if (h_realloc_old_deg[r] > 0 && h_realloc_old_ptr[r] != nullptr) {
                    int old_log_b = ceil_log2(h_realloc_old_deg[r]);
                    mgr.reclaim_block(h_realloc_old_ptr[r], old_log_b);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 7: Append New Edges
    // -------------------------------------------------------------------------
    {
        ScopedPhaseTimer append_timer("insert_append", true);
        int blk = 256;
        append_batch_edges_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
            num_unique, ws.d_unique_srcs, ws.d_batch_offsets, ws.d_batch_deg, d_final_dst,
            graph.d_used(), graph.d_ptr());
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * @brief Algorithm 1: Dynamic Batch Deletion (Paper Section III-E).
 */
inline void batch_delete(HornetGraph& graph, BatchWorkspace& ws) {
    ScopedPhaseTimer total_timer("delete_total", true);
    int batch_size = ws.batch_size;
    if (batch_size == 0) return;

    int blk = 256;
    pack_edges_kernel<<<(batch_size + blk - 1) / blk, blk>>>(
        batch_size, ws.d_raw_src, ws.d_raw_dst, ws.d_keys_in);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceRadixSort::SortKeys(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_in, ws.d_keys_sorted, batch_size);
    cub::DeviceSelect::Unique(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_keys_sorted, ws.d_keys_uniq, ws.d_scalar_count, batch_size);

    int uniq_count = 0;
    CUDA_CHECK(cudaMemcpy(&uniq_count, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (uniq_count == 0) return;

    unpack_edges_kernel<<<(uniq_count + blk - 1) / blk, blk>>>(
        uniq_count, ws.d_keys_uniq, ws.d_src_uniq, ws.d_dst_uniq);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceRunLengthEncode::Encode(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_src_uniq, ws.d_unique_srcs, ws.d_batch_deg, ws.d_scalar_count, uniq_count);

    int num_unique = 0;
    CUDA_CHECK(cudaMemcpy(&num_unique, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));

    cub::DeviceScan::ExclusiveSum(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_batch_deg, ws.d_batch_offsets, num_unique);

    delete_and_check_shrink_kernel<<<(num_unique + blk - 1) / blk, blk>>>(
        num_unique, ws.d_unique_srcs, ws.d_batch_offsets, ws.d_batch_deg, ws.d_dst_uniq,
        graph.d_used(), graph.d_ptr(), ws.d_need_realloc, ws.d_need_free_only,
        ws.d_all_old_deg, ws.d_all_new_deg, ws.d_all_old_ptr);
    CUDA_CHECK(cudaGetLastError());

    // Reclaim fully emptied blocks (new_deg == 0)
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_deg, ws.d_need_free_only, ws.d_realloc_old_deg, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_ptr, ws.d_need_free_only, ws.d_realloc_old_ptr, ws.d_scalar_count, num_unique);

    int num_free_only = 0;
    CUDA_CHECK(cudaMemcpy(&num_free_only, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_free_only > 0) {
        std::vector<vid_t> h_deg(num_free_only);
        std::vector<vid_t*> h_ptr(num_free_only);
        CUDA_CHECK(cudaMemcpy(h_deg.data(), ws.d_realloc_old_deg, num_free_only * sizeof(vid_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ptr.data(), ws.d_realloc_old_ptr, num_free_only * sizeof(vid_t*), cudaMemcpyDeviceToHost));
        auto& mgr = graph.memory_manager();
        for (int i = 0; i < num_free_only; ++i) {
            if (h_deg[i] > 0 && h_ptr[i]) {
                mgr.reclaim_block(h_ptr[i], ceil_log2(h_deg[i]));
            }
        }
    }

    // Shrink blocks where new_deg < bsize(old_deg) / 2
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_unique_srcs, ws.d_need_realloc, ws.d_realloc_v, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_deg, ws.d_need_realloc, ws.d_realloc_old_deg, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_new_deg, ws.d_need_realloc, ws.d_realloc_new_deg, ws.d_scalar_count, num_unique);
    cub::DeviceSelect::Flagged(
        ws.d_cub_temp, ws.cub_temp_bytes, ws.d_all_old_ptr, ws.d_need_realloc, ws.d_realloc_old_ptr, ws.d_scalar_count, num_unique);

    int num_shrink = 0;
    CUDA_CHECK(cudaMemcpy(&num_shrink, ws.d_scalar_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_shrink > 0) {
        std::vector<vid_t> h_old_deg(num_shrink), h_new_deg(num_shrink);
        std::vector<vid_t*> h_old_ptr(num_shrink), h_new_ptr(num_shrink);
        CUDA_CHECK(cudaMemcpy(h_old_deg.data(), ws.d_realloc_old_deg, num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_new_deg.data(), ws.d_realloc_new_deg, num_shrink * sizeof(vid_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_old_ptr.data(), ws.d_realloc_old_ptr, num_shrink * sizeof(vid_t*), cudaMemcpyDeviceToHost));

        auto& mgr = graph.memory_manager();
        for (int i = 0; i < num_shrink; ++i) {
            h_new_ptr[i] = mgr.allocate_block(ceil_log2(h_new_deg[i]), false).ptr;
        }

        CUDA_CHECK(cudaMemcpy(ws.d_realloc_new_ptr, h_new_ptr.data(), num_shrink * sizeof(vid_t*), cudaMemcpyHostToDevice));

        std::vector<eoff_t> h_move_offsets(num_shrink + 1, 0);
        for (int i = 0; i < num_shrink; ++i) {
            h_move_offsets[i + 1] = h_move_offsets[i] + h_new_deg[i];
        }
        eoff_t total_moves = h_move_offsets[num_shrink];
        if (total_moves > 0) {
            CUDA_CHECK(cudaMemcpy(ws.d_move_offsets, h_move_offsets.data(),
                                  (num_shrink + 1) * sizeof(eoff_t), cudaMemcpyHostToDevice));
            eoff_t grid = (total_moves + blk - 1) / blk;
            adj_move_balanced_kernel<<<static_cast<unsigned int>(grid), blk>>>(
                num_shrink, ws.d_move_offsets, total_moves,
                ws.d_realloc_v, ws.d_realloc_old_ptr, ws.d_realloc_new_ptr, graph.d_ptr());
            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < num_shrink; ++i) {
            mgr.reclaim_block(h_old_ptr[i], ceil_log2(h_old_deg[i]));
        }
    }
}

} // namespace baseline
