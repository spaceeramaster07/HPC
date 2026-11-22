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

__device__ __forceinline__ int dev_ceil_log2(int x) {
    if (x <= 1) return 0;
    return 32 - __clz(static_cast<unsigned int>(x - 1));
}

__device__ __forceinline__ int dev_upper_bound_offsets(const eoff_t* __restrict__ offsets,
                                                       int num_items,
                                                       eoff_t target) {
    int lo = 1;
    int hi = num_items;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (offsets[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

/**
 * @brief Block-cooperative shared-memory binarySearchLB (Paper Footnote 2 / Busato & Bombieri [9]).
 *
 * Two threads locate [block_start, block_end] in global `offsets`, the block cooperatively
 * loads that window into `__shared__` (chunked if span exceeds `SMEM_ITEMS`), and each thread
 * resolves `(out_pos, out_local_offset, out_gid)` in shared memory.
 */
template <int SMEM_ITEMS = 1024>
__device__ __forceinline__ bool binarySearchLB(int num_items,
                                               const eoff_t* __restrict__ offsets,
                                               eoff_t total_work,
                                               int& out_pos,
                                               eoff_t& out_local_offset,
                                               eoff_t& out_gid) {
    if (total_work <= 0 || num_items <= 0) return false;

    eoff_t block_first = static_cast<eoff_t>(blockIdx.x) * static_cast<eoff_t>(blockDim.x);
    if (block_first >= total_work) return false;

    eoff_t block_last = block_first + static_cast<eoff_t>(blockDim.x) - 1;
    if (block_last >= total_work) block_last = total_work - 1;

    __shared__ int s_range[2];
    __shared__ eoff_t s_offsets[SMEM_ITEMS + 1];

    if (threadIdx.x == 0) {
        s_range[0] = dev_upper_bound_offsets(offsets, num_items, block_first);
    }
    if (threadIdx.x == 1 || (blockDim.x == 1 && threadIdx.x == 0)) {
        s_range[1] = dev_upper_bound_offsets(offsets, num_items, block_last);
    }
    __syncthreads();

    int range_start = s_range[0];
    int range_end = s_range[1];
    eoff_t gid = block_first + static_cast<eoff_t>(threadIdx.x);

    int my_pos = -1;
    eoff_t my_local = 0;

    for (int chunk_start = range_start; chunk_start <= range_end; chunk_start += SMEM_ITEMS) {
        int chunk_items = range_end - chunk_start + 1;
        if (chunk_items > SMEM_ITEMS) chunk_items = SMEM_ITEMS;

        for (int k = static_cast<int>(threadIdx.x); k <= chunk_items; k += static_cast<int>(blockDim.x)) {
            s_offsets[k] = offsets[chunk_start + k];
        }
        __syncthreads();

        if (gid < total_work && my_pos < 0) {
            if (gid >= s_offsets[0] && gid < s_offsets[chunk_items]) {
                int lo = 1;
                int hi = chunk_items;
                while (lo < hi) {
                    int mid = lo + ((hi - lo) >> 1);
                    if (s_offsets[mid] <= gid) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int local_pos = lo - 1;
                my_pos = chunk_start + local_pos;
                my_local = gid - s_offsets[local_pos];
            }
        }

        if (chunk_start + SMEM_ITEMS <= range_end) {
            __syncthreads();
        }
    }

    if (gid < total_work && my_pos >= 0) {
        out_pos = my_pos;
        out_local_offset = my_local;
        out_gid = gid;
        return true;
    }
    return false;
}

__global__ void adj_move_balanced_kernel(int num_realloc,
                                         const eoff_t* __restrict__ move_offsets,
                                         eoff_t total_moves,
                                         const vid_t* __restrict__ realloc_v,
                                         vid_t* const* __restrict__ old_ptrs,
                                         vid_t* const* __restrict__ new_ptrs,
                                         vid_t** __restrict__ d_ptr) {
    int r = 0;
    eoff_t local_idx = 0;
    eoff_t gid = 0;
    if (!binarySearchLB(num_realloc, move_offsets, total_moves, r, local_idx, gid)) {
        return;
    }

    new_ptrs[r][local_idx] = old_ptrs[r][local_idx];
    if (local_idx == 0) {
        d_ptr[realloc_v[r]] = new_ptrs[r];
    }
}


struct BatchWorkspace {
    int capacity = 0;
    int batch_size = 0;
    size_t del_keep_capacity = 0;

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
    eoff_t* d_graph_deg = nullptr;
    eoff_t* d_graph_offsets = nullptr;
    uint8_t* d_del_keep = nullptr;

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
        cudaFree(d_graph_deg);
        cudaFree(d_graph_offsets);
        cudaFree(d_del_keep);
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
        d_keep = d_del_keep = d_need_realloc = d_need_free_only = nullptr;
        d_batch_deg = nullptr;
        d_batch_offsets = d_graph_deg = d_graph_offsets = d_move_offsets = nullptr;
        d_all_old_deg = d_all_new_deg = d_realloc_v = d_realloc_old_deg = d_realloc_new_deg = nullptr;
        d_all_old_ptr = d_realloc_old_ptr = d_realloc_new_ptr = nullptr;
        d_scalar_count = nullptr;
        d_cub_temp = nullptr;
        capacity = 0;
        del_keep_capacity = 0;
        cub_temp_bytes = 0;
    }

    void ensure_del_keep_capacity(size_t n) {
        if (n <= del_keep_capacity) return;
        if (d_del_keep) {
            cudaFree(d_del_keep);
            d_del_keep = nullptr;
            del_keep_capacity = 0;
        }
        size_t alloc_n = std::max<size_t>(n + n / 16, 1);
        CUDA_CHECK(cudaMalloc(&d_del_keep, alloc_n * sizeof(uint8_t)));
        del_keep_capacity = alloc_n;
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
        CUDA_CHECK(cudaMalloc(&d_batch_deg, (capacity + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_batch_offsets, (capacity + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_graph_deg, (capacity + 1) * sizeof(eoff_t)));
        CUDA_CHECK(cudaMalloc(&d_graph_offsets, (capacity + 1) * sizeof(eoff_t)));
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
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_batch_deg, d_batch_offsets, capacity + 1);
        req = std::max(req, tmp);
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_graph_deg, d_graph_offsets, capacity + 1);
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

} // namespace baseline
