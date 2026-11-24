#pragma once

#include "memory_manager.cuh"
#include "types.cuh"

#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <cstdint>

namespace improved {

struct SlabAllocTask {
    int bin;
    int ba_idx;
    int req_offset;
    int take;
};

struct DS7AllocWorkspaceView {
    // Assumes serialized stream-0 execution per BatchWorkspace; reserves h_pinned_counts[16..47]
    // for the 32-bin deficit check while leaving [0..3] and [4..5] untouched for callers.
    static constexpr int PINNED_COUNTS_CAP = 64;
    static constexpr int PINNED_BIN_DEFICIT_OFFSET = 16;
    static_assert(PINNED_BIN_DEFICIT_OFFSET + MemoryManager::MAX_BINS <= PINNED_COUNTS_CAP,
                  "DS-7 bin deficit slots exceed pinned scalar buffer capacity");

    uint8_t* d_req_bin = nullptr;
    int* d_bucket_req_idx = nullptr;
    int* d_single_task_idx = nullptr;
    int* d_med_task_idx = nullptr;
    int* d_huge_task_idx = nullptr;
    SlabAllocTask* d_slab_tasks = nullptr;
    int* d_huge_chunk_counts = nullptr;
    int* d_bin_needed = nullptr;    // [32]
    int* d_bin_offsets = nullptr;   // [33]
    int* d_bin_cursor = nullptr;    // [32]
    int* d_bin_deficit = nullptr;   // [32]
    int* d_alloc_counts = nullptr;  // [4]: single, med, huge, total
    int* h_pinned_counts = nullptr; // [PINNED_COUNTS_CAP]
};

__device__ __forceinline__ int ds7_ceil_log2(int x) {
    if (x <= 1) return 0;
    return 32 - __clz(static_cast<unsigned int>(x - 1));
}

__device__ __forceinline__ void ds7_warp_append_idx(bool pred,
                                                    int value,
                                                    int* __restrict__ out_indices,
                                                    int* __restrict__ out_count) {
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, pred);
    if (pred) {
        int lane = threadIdx.x & 31;
        int leader = __ffs(mask) - 1;
        int base = 0;
        if (lane == leader) {
            base = atomicAdd(out_count, __popc(mask));
        }
        base = __shfl_sync(mask, base, leader);
        int rank = __popc(mask & ((1u << lane) - 1u));
        out_indices[base + rank] = value;
    }
}

__global__ void ds7_count_requests_per_bin_kernel(int num_reqs,
                                                  const vid_t* __restrict__ d_new_deg,
                                                  uint8_t* __restrict__ d_req_bin,
                                                  int* __restrict__ d_bin_needed) {
    __shared__ int s_hist[32];
    if (threadIdx.x < 32) {
        s_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_reqs) {
        int b = ds7_ceil_log2(static_cast<int>(d_new_deg[r]));
        if (b < 0) b = 0;
        if (b >= MemoryManager::MAX_BINS) b = MemoryManager::MAX_BINS - 1;
        d_req_bin[r] = static_cast<uint8_t>(b);
        atomicAdd(&s_hist[b], 1);
    }
    __syncthreads();

    if (threadIdx.x < 32 && s_hist[threadIdx.x] > 0) {
        atomicAdd(&d_bin_needed[threadIdx.x], s_hist[threadIdx.x]);
    }
}

// 1024 threads = 32 warps (one warp per bin 0..31)
__global__ void ds7_check_bin_deficit_kernel(DeviceMemoryView mem,
                                             const int* __restrict__ d_bin_needed,
                                             int* __restrict__ d_bin_offsets,
                                             int* __restrict__ d_bin_cursor,
                                             int* __restrict__ d_bin_deficit) {
    __shared__ int s_needed[32];

    int b = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int needed = d_bin_needed[b];
    if (lane == 0) {
        s_needed[b] = needed;
    }

    long long avail = 0;
    if (needed > 0) {
        int nbas = mem.d_bin_num_bas[b];
        int first = mem.d_bin_first_nonfull[b];
        const int* free_counts = mem.d_bin_free_counts[b];
        for (int base = first; base < nbas && avail < static_cast<long long>(needed); base += 32) {
            int idx = base + lane;
            int fc = (idx < nbas) ? free_counts[idx] : 0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                fc += __shfl_down_sync(0xFFFFFFFFu, fc, offset);
            }
            int batch_sum = __shfl_sync(0xFFFFFFFFu, fc, 0);
            avail += static_cast<long long>(batch_sum);
        }
    }

    if (lane == 0) {
        long long def = (avail < static_cast<long long>(needed))
                            ? (static_cast<long long>(needed) - avail)
                            : 0LL;
        d_bin_deficit[b] = static_cast<int>(def);
    }
    __syncthreads();

    if (b == 0) {
        int val = s_needed[lane];
        int inc = val;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(0xFFFFFFFFu, inc, offset);
            if (lane >= offset) inc += n;
        }
        int exc = inc - val;
        d_bin_offsets[lane] = exc;
        d_bin_cursor[lane] = exc;
        if (lane == 31) {
            d_bin_offsets[32] = inc;
        }
    }
}

__global__ void ds7_bucket_scatter_kernel(int num_reqs,
                                          const uint8_t* __restrict__ d_req_bin,
                                          int* __restrict__ d_bin_cursor,
                                          int* __restrict__ d_bucket_req_idx) {
    __shared__ int s_bin_count[32];
    __shared__ int s_bin_base[32];

    if (threadIdx.x < 32) {
        s_bin_count[threadIdx.x] = 0;
    }
    __syncthreads();

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = (r < num_reqs);
    int b = 0;
    int local_rank = 0;
    if (valid) {
        b = static_cast<int>(d_req_bin[r]);
        local_rank = atomicAdd(&s_bin_count[b], 1);
    }
    __syncthreads();

    if (threadIdx.x < 32 && s_bin_count[threadIdx.x] > 0) {
        s_bin_base[threadIdx.x] = atomicAdd(&d_bin_cursor[threadIdx.x], s_bin_count[threadIdx.x]);
    }
    __syncthreads();

    if (valid) {
        d_bucket_req_idx[s_bin_base[b] + local_rank] = r;
    }
}

// 32 blocks of 32 threads (1 warp per bin)
__global__ void ds7_plan_slab_allocs_kernel(DeviceMemoryView mem,
                                            const int* __restrict__ d_bin_needed,
                                            const int* __restrict__ d_bin_offsets,
                                            SlabAllocTask* __restrict__ d_slab_tasks,
                                            int* __restrict__ d_single_task_idx,
                                            int* __restrict__ d_med_task_idx,
                                            int* __restrict__ d_huge_task_idx,
                                            int* __restrict__ d_alloc_counts) {
    int b = blockIdx.x;
    int lane = threadIdx.x;

    int needed = d_bin_needed[b];
    if (needed <= 0) return;

    int rem = needed;
    int req_off = d_bin_offsets[b];
    int nbas = mem.d_bin_num_bas[b];
    int first = mem.d_bin_first_nonfull[b];
    int* free_counts = mem.d_bin_free_counts[b];
    const BlockArrayDesc* descs = mem.d_bin_descs[b];

    int new_first = nbas;
    int next_unscanned = first;
    bool found_first = false;

    for (int base = first; base < nbas && rem > 0; base += 32) {
        int idx = base + lane;
        int fc = (idx < nbas) ? free_counts[idx] : 0;

        int inc = fc;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int val = __shfl_up_sync(0xFFFFFFFFu, inc, offset);
            if (lane >= offset) inc += val;
        }
        int exc = inc - fc;
        int batch_avail = __shfl_sync(0xFFFFFFFFu, inc, 31);

        int before = min(rem, exc);
        int after = min(rem, inc);
        int take = after - before;

        int rem_after_fc = fc - take;
        if (take > 0) {
            free_counts[idx] = rem_after_fc;
        }

        if (!found_first) {
            unsigned nonfull_mask = __ballot_sync(0xFFFFFFFFu, (idx < nbas) && (rem_after_fc > 0));
            if (nonfull_mask != 0u) {
                new_first = base + (__ffs(nonfull_mask) - 1);
                found_first = true;
            } else {
                next_unscanned = min(nbas, base + 32);
            }
        }

        bool has_task = (take > 0);
        unsigned task_mask = __ballot_sync(0xFFFFFFFFu, has_task);
        int task_slot = -1;
        if (has_task) {
            int leader = __ffs(task_mask) - 1;
            int base_slot = 0;
            if (lane == leader) {
                base_slot = atomicAdd(&d_alloc_counts[3], __popc(task_mask));
            }
            base_slot = __shfl_sync(task_mask, base_slot, leader);
            int rank = __popc(task_mask & ((1u << lane) - 1u));
            task_slot = base_slot + rank;
            d_slab_tasks[task_slot] = {b, idx, req_off + before, take};
        }

        bool is_single = has_task && (take == 1);
        bool is_huge = false;
        if (has_task && take > 4096) {
            int L = descs[idx].num_levels;
            int leaf_words = descs[idx].level_offsets[L] - descs[idx].level_offsets[L - 1];
            if (leaf_words > 1024) {
                is_huge = true;
            }
        }
        bool is_med = has_task && !is_single && !is_huge;

        ds7_warp_append_idx(is_single, task_slot, d_single_task_idx, &d_alloc_counts[0]);
        ds7_warp_append_idx(is_med, task_slot, d_med_task_idx, &d_alloc_counts[1]);
        ds7_warp_append_idx(is_huge, task_slot, d_huge_task_idx, &d_alloc_counts[2]);

        int used_batch = min(rem, batch_avail);
        rem -= used_batch;
        req_off += used_batch;
    }

    if (lane == 0) {
        mem.d_bin_first_nonfull[b] = found_first ? new_first : next_unscanned;
    }
}

__device__ __forceinline__ void ds7_claim_single_block(const BlockArrayDesc& desc,
                                                       int req_offset,
                                                       const int* __restrict__ d_bucket_req_idx,
                                                       vid_t** __restrict__ d_out_new_ptr) {
    uint64_t* tree = desc.d_bittree;
    int L = desc.num_levels;
    int idx = 0;
    for (int l = 0; l < L - 1; ++l) {
        uint64_t w = tree[desc.level_offsets[l] + idx];
        int bit = __ffsll(static_cast<long long>(w)) - 1;
        idx = (idx << 6) + bit;
    }

    int leaf_off = desc.level_offsets[L - 1];
    int leaf_word = idx;
    uint64_t lw = tree[leaf_off + leaf_word];
    int leaf_bit = __ffsll(static_cast<long long>(lw)) - 1;
    int block_idx = (leaf_word << 6) + leaf_bit;

    lw &= ~(1ULL << leaf_bit);
    tree[leaf_off + leaf_word] = lw;

    int child_word = leaf_word;
    uint64_t child_val = lw;
    for (int l = L - 2; l >= 0; --l) {
        if (child_val != 0ULL) break;
        int p_word = child_word >> 6;
        int p_bit = child_word & 63;
        uint64_t pw = tree[desc.level_offsets[l] + p_word];
        pw &= ~(1ULL << p_bit);
        tree[desc.level_offsets[l] + p_word] = pw;
        child_word = p_word;
        child_val = pw;
    }

    int req = d_bucket_req_idx[req_offset];
    d_out_new_ptr[req] = desc.d_edges + (static_cast<size_t>(block_idx) << desc.log_bsize);
}

__device__ __forceinline__ void ds7_rebuild_upper_levels_cta(const BlockArrayDesc& desc,
                                                             int start_level) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    for (int l = start_level; l >= 0; --l) {
        int parent_words = desc.level_offsets[l + 1] - desc.level_offsets[l];
        int child_words = desc.level_offsets[l + 2] - desc.level_offsets[l + 1];
        uint64_t* parent = desc.d_bittree + desc.level_offsets[l];
        const uint64_t* child = desc.d_bittree + desc.level_offsets[l + 1];

        for (int pw = warp_id; pw < parent_words; pw += 8) {
            int c0 = (pw << 6) + lane;
            int c1 = c0 + 32;
            bool nz0 = (c0 < child_words) && (child[c0] != 0ULL);
            bool nz1 = (c1 < child_words) && (child[c1] != 0ULL);
            uint32_t m0 = __ballot_sync(0xFFFFFFFFu, nz0);
            uint32_t m1 = __ballot_sync(0xFFFFFFFFu, nz1);
            if (lane == 0) {
                parent[pw] = (static_cast<uint64_t>(m1) << 32) | static_cast<uint64_t>(m0);
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void ds7_extract_set_bits_64(uint64_t mask,
                                                        int base_idx,
                                                        int* s_out,
                                                        int& s_cnt,
                                                        uint32_t& s_m0,
                                                        uint32_t& s_m1) {
    int tid = threadIdx.x;
    bool has = (tid < 64) && (((mask >> tid) & 1ULL) != 0ULL);
    if (tid < 32) {
        uint32_t m0 = __ballot_sync(0xFFFFFFFFu, has);
        if (tid == 0) s_m0 = m0;
    } else if (tid < 64) {
        uint32_t m1 = __ballot_sync(0xFFFFFFFFu, has);
        if (tid == 32) s_m1 = m1;
    }
    __syncthreads();

    uint32_t m0 = s_m0;
    uint32_t m1 = s_m1;
    int cnt0 = __popc(m0);
    if (tid < 32 && ((m0 >> tid) & 1u)) {
        int rank = __popc(m0 & ((1u << tid) - 1u));
        s_out[rank] = base_idx + tid;
    } else if (tid >= 32 && tid < 64 && ((m1 >> (tid - 32)) & 1u)) {
        int b = tid - 32;
        int rank = cnt0 + __popc(m1 & ((1u << b) - 1u));
        s_out[rank] = base_idx + tid;
    }
    if (tid == 0) {
        s_cnt = cnt0 + __popc(m1);
    }
    __syncthreads();
}

template <typename BlockScanTemp>
__device__ __forceinline__ void ds7_claim_med_cta(const BlockArrayDesc& desc,
                                                  int req_offset,
                                                  int take,
                                                  const int* __restrict__ d_bucket_req_idx,
                                                  vid_t** __restrict__ d_out_new_ptr,
                                                  BlockScanTemp& s_scan_temp) {
    using BlockScan = cub::BlockScan<int, 256>;

    __shared__ int s_pwords[64];
    __shared__ int s_np;
    __shared__ uint32_t s_m0, s_m1;
    __shared__ uint64_t s_pmask[4];
    __shared__ uint32_t s_halves[8];
    __shared__ int s_rem;
    __shared__ int s_out_pos;
    __shared__ int s_any_cleared_p;

    int tid = threadIdx.x;
    uint64_t* tree = desc.d_bittree;
    int L = desc.num_levels;

    if (L == 1) {
        uint64_t lw = tree[0];
        bool has = (tid < 64) && (((lw >> tid) & 1ULL) != 0ULL);
        int rank = 0;
        int total_avail = 0;
        BlockScan(s_scan_temp).ExclusiveSum(has ? 1 : 0, rank, total_avail);
        __syncthreads();

        bool take_this = has && (rank < take);
        if (take_this) {
            int req = d_bucket_req_idx[req_offset + rank];
            d_out_new_ptr[req] = desc.d_edges + (static_cast<size_t>(tid) << desc.log_bsize);
        }
        if (tid < 32) {
            uint32_t m0 = __ballot_sync(0xFFFFFFFFu, take_this);
            if (tid == 0) s_m0 = m0;
        } else if (tid < 64) {
            uint32_t m1 = __ballot_sync(0xFFFFFFFFu, take_this);
            if (tid == 32) s_m1 = m1;
        }
        __syncthreads();
        if (tid == 0) {
            uint64_t cleared = (static_cast<uint64_t>(s_m1) << 32) | static_cast<uint64_t>(s_m0);
            tree[0] = lw & ~cleared;
        }
        return;
    }

    if (tid == 0) {
        s_rem = take;
        s_out_pos = 0;
        s_any_cleared_p = 0;
    }
    __syncthreads();

    int leaf_off = desc.level_offsets[L - 1];
    int p_off = desc.level_offsets[L - 2];

    // Outer loop over grandparent words (single step for L==2 or L==3)
    int ngp = (L >= 3) ? (desc.level_offsets[L - 2] - desc.level_offsets[L - 3]) : 1;
    int gp_off = (L >= 3) ? desc.level_offsets[L - 3] : 0;

    for (int gp = 0; gp < ngp && s_rem > 0; ++gp) {
        if (L == 2) {
            if (tid == 0) {
                s_pwords[0] = 0;
                s_np = 1;
            }
            __syncthreads();
        } else if (L == 3) {
            uint64_t root_mask = tree[0];
            ds7_extract_set_bits_64(root_mask, 0, s_pwords, s_np, s_m0, s_m1);
        } else {
            uint64_t gp_mask = tree[gp_off + gp];
            if (gp_mask == 0ULL) continue;
            ds7_extract_set_bits_64(gp_mask, gp << 6, s_pwords, s_np, s_m0, s_m1);
        }

        int np_avail = s_np;
        for (int p_idx = 0; p_idx < np_avail && s_rem > 0; p_idx += 4) {
            if (tid < 4) {
                s_pmask[tid] = (p_idx + tid < np_avail) ? tree[p_off + s_pwords[p_idx + tid]] : 0ULL;
            }
            __syncthreads();

            int g = tid >> 6;
            int bit = tid & 63;
            bool active = (p_idx + g < np_avail) && (((s_pmask[g] >> bit) & 1ULL) != 0ULL);
            int leaf_w = active ? ((s_pwords[p_idx + g] << 6) + bit) : 0;

            uint64_t lw = active ? tree[leaf_off + leaf_w] : 0ULL;
            int pop = __popcll(lw);

            int exc_pop = 0;
            int batch_avail = 0;
            BlockScan(s_scan_temp).ExclusiveSum(pop, exc_pop, batch_avail);
            __syncthreads();

            uint64_t new_lw = lw;
            int rem_now = s_rem;
            if (active && exc_pop < rem_now) {
                int claim = min(pop, rem_now - exc_pop);
                int out_base = req_offset + s_out_pos + exc_pop;
                for (int k = 0; k < claim; ++k) {
                    int bpos = __ffsll(static_cast<long long>(new_lw)) - 1;
                    new_lw &= (new_lw - 1ULL);
                    int block_idx = (leaf_w << 6) + bpos;
                    int req = d_bucket_req_idx[out_base + k];
                    d_out_new_ptr[req] = desc.d_edges + (static_cast<size_t>(block_idx) << desc.log_bsize);
                }
                if (claim > 0) {
                    tree[leaf_off + leaf_w] = new_lw;
                }
            }

            int warp_id = tid >> 5;
            int lane = tid & 31;
            uint32_t non_empty_half = __ballot_sync(0xFFFFFFFFu, active && (new_lw != 0ULL));
            if (lane == 0) {
                s_halves[warp_id] = non_empty_half;
            }
            __syncthreads();

            if (tid < 4 && (p_idx + tid < np_avail)) {
                uint64_t new_pmask =
                    (static_cast<uint64_t>(s_halves[2 * tid + 1]) << 32) |
                    static_cast<uint64_t>(s_halves[2 * tid]);
                if (new_pmask != s_pmask[tid]) {
                    tree[p_off + s_pwords[p_idx + tid]] = new_pmask;
                    if (new_pmask == 0ULL) {
                        s_any_cleared_p = 1;
                    }
                }
            }
            if (tid == 0) {
                int used_now = min(rem_now, batch_avail);
                s_out_pos += used_now;
                s_rem -= used_now;
            }
            __syncthreads();
        }

        if (L <= 3) break;
    }

    if (L >= 3 && s_any_cleared_p != 0) {
        ds7_rebuild_upper_levels_cta(desc, L - 3);
    }
}

__global__ void ds7_alloc_single_bittree_kernel(int num_single,
                                                const int* __restrict__ d_single_task_idx,
                                                const SlabAllocTask* __restrict__ d_slab_tasks,
                                                DeviceMemoryView mem,
                                                const int* __restrict__ d_bucket_req_idx,
                                                vid_t** __restrict__ d_out_new_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_single) return;
    SlabAllocTask task = d_slab_tasks[d_single_task_idx[i]];
    const BlockArrayDesc& desc = mem.d_bin_descs[task.bin][task.ba_idx];
    ds7_claim_single_block(desc, task.req_offset, d_bucket_req_idx, d_out_new_ptr);
}

__global__ void ds7_alloc_med_bittree_kernel(int num_med,
                                             const int* __restrict__ d_med_task_idx,
                                             const SlabAllocTask* __restrict__ d_slab_tasks,
                                             DeviceMemoryView mem,
                                             const int* __restrict__ d_bucket_req_idx,
                                             vid_t** __restrict__ d_out_new_ptr) {
    if (blockIdx.x >= static_cast<unsigned int>(num_med)) return;
    SlabAllocTask task = d_slab_tasks[d_med_task_idx[blockIdx.x]];
    const BlockArrayDesc& desc = mem.d_bin_descs[task.bin][task.ba_idx];

    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage s_scan_temp;
    ds7_claim_med_cta(desc, task.req_offset, task.take,
                      d_bucket_req_idx, d_out_new_ptr, s_scan_temp);
}

__global__ void ds7_alloc_small_or_med_unified_kernel(const int* __restrict__ d_num_tasks,
                                                        const SlabAllocTask* __restrict__ d_slab_tasks,
                                                        DeviceMemoryView mem,
                                                        const int* __restrict__ d_bucket_req_idx,
                                                        vid_t** __restrict__ d_out_new_ptr) {
    int total_tasks = *d_num_tasks;
    if (blockIdx.x >= static_cast<unsigned int>(total_tasks)) return;

    SlabAllocTask task = d_slab_tasks[blockIdx.x];
    const BlockArrayDesc& desc = mem.d_bin_descs[task.bin][task.ba_idx];
    if (task.take == 1) {
        if (threadIdx.x == 0) {
            ds7_claim_single_block(desc, task.req_offset, d_bucket_req_idx, d_out_new_ptr);
        }
        return;
    }

    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage s_scan_temp;
    ds7_claim_med_cta(desc, task.req_offset, task.take,
                      d_bucket_req_idx, d_out_new_ptr, s_scan_temp);
}

// 32 CTAs per huge task
__global__ void ds7_huge_slab_count_kernel(int num_huge,
                                           const int* __restrict__ d_huge_task_idx,
                                           const SlabAllocTask* __restrict__ d_slab_tasks,
                                           DeviceMemoryView mem,
                                           int* __restrict__ d_huge_chunk_counts) {
    int h = blockIdx.x >> 5;
    int c = blockIdx.x & 31;
    if (h >= num_huge) return;

    SlabAllocTask task = d_slab_tasks[d_huge_task_idx[h]];
    const BlockArrayDesc& desc = mem.d_bin_descs[task.bin][task.ba_idx];
    int L = desc.num_levels;
    int leaf_off = desc.level_offsets[L - 1];
    int leaf_words = desc.level_offsets[L] - leaf_off;
    const uint64_t* leaf = desc.d_bittree + leaf_off;

    int w_start = static_cast<int>((static_cast<int64_t>(leaf_words) * c) >> 5);
    int w_end = static_cast<int>((static_cast<int64_t>(leaf_words) * (c + 1)) >> 5);

    int local_sum = 0;
    for (int w = w_start + threadIdx.x; w < w_end; w += 256) {
        local_sum += __popcll(leaf[w]);
    }

    using BlockReduce = cub::BlockReduce<int, 256>;
    __shared__ typename BlockReduce::TempStorage s_red;
    int total = BlockReduce(s_red).Sum(local_sum);
    if (threadIdx.x == 0) {
        d_huge_chunk_counts[h * 32 + c] = total;
    }
}

__global__ void ds7_huge_slab_claim_kernel(int num_huge,
                                           const int* __restrict__ d_huge_task_idx,
                                           const SlabAllocTask* __restrict__ d_slab_tasks,
                                           DeviceMemoryView mem,
                                           const int* __restrict__ d_huge_chunk_counts,
                                           const int* __restrict__ d_bucket_req_idx,
                                           vid_t** __restrict__ d_out_new_ptr) {
    int h = blockIdx.x >> 5;
    int c = blockIdx.x & 31;
    if (h >= num_huge) return;

    __shared__ int s_chunk_start;
    __shared__ int s_chunk_avail;
    using BlockScan = cub::BlockScan<int, 256>;
    __shared__ typename BlockScan::TempStorage s_scan_temp;

    int tid = threadIdx.x;
    if (tid < 32) {
        int cnt = d_huge_chunk_counts[h * 32 + tid];
        int inc = cnt;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int n = __shfl_up_sync(0xFFFFFFFFu, inc, offset);
            if (tid >= offset) inc += n;
        }
        int exc = inc - cnt;
        if (tid == c) {
            s_chunk_start = exc;
            s_chunk_avail = cnt;
        }
    }
    __syncthreads();

    SlabAllocTask task = d_slab_tasks[d_huge_task_idx[h]];
    int chunk_start = s_chunk_start;
    int chunk_avail = s_chunk_avail;
    if (chunk_start >= task.take || chunk_avail == 0) return;

    const BlockArrayDesc& desc = mem.d_bin_descs[task.bin][task.ba_idx];
    int L = desc.num_levels;
    int leaf_off = desc.level_offsets[L - 1];
    int leaf_words = desc.level_offsets[L] - leaf_off;
    uint64_t* leaf = desc.d_bittree + leaf_off;

    int w_start = static_cast<int>((static_cast<int64_t>(leaf_words) * c) >> 5);
    int w_end = static_cast<int>((static_cast<int64_t>(leaf_words) * (c + 1)) >> 5);

    int local_need = min(chunk_avail, task.take - chunk_start);
    int local_out = 0;

    for (int base = w_start; base < w_end && local_need > 0; base += 256) {
        int w = base + tid;
        uint64_t lw = (w < w_end) ? leaf[w] : 0ULL;
        int pop = __popcll(lw);

        int exc_pop = 0;
        int round_avail = 0;
        BlockScan(s_scan_temp).ExclusiveSum(pop, exc_pop, round_avail);
        __syncthreads();

        if (round_avail == 0) continue;

        if (w < w_end && exc_pop < local_need) {
            int claim = min(pop, local_need - exc_pop);
            if (claim > 0) {
                int out_base = task.req_offset + chunk_start + local_out + exc_pop;
                uint64_t new_lw = lw;
                for (int k = 0; k < claim; ++k) {
                    int bpos = __ffsll(static_cast<long long>(new_lw)) - 1;
                    new_lw &= (new_lw - 1ULL);
                    int block_idx = (w << 6) + bpos;
                    int req = d_bucket_req_idx[out_base + k];
                    d_out_new_ptr[req] = desc.d_edges + (static_cast<size_t>(block_idx) << desc.log_bsize);
                }
                leaf[w] = new_lw;
            }
        }

        int used_now = min(local_need, round_avail);
        local_out += used_now;
        local_need -= used_now;
    }
}

__global__ void ds7_huge_slab_rebuild_kernel(int num_huge,
                                             const int* __restrict__ d_huge_task_idx,
                                             const SlabAllocTask* __restrict__ d_slab_tasks,
                                             DeviceMemoryView mem) {
    int h = blockIdx.x;
    if (h >= num_huge) return;
    SlabAllocTask task = d_slab_tasks[d_huge_task_idx[h]];
    const BlockArrayDesc& desc = mem.d_bin_descs[task.bin][task.ba_idx];
    if (desc.num_levels >= 2) {
        ds7_rebuild_upper_levels_cta(desc, desc.num_levels - 2);
    }
}

// =============================================================================
// GPU Block Reclamation (Separate Kernel Phase After Move/Merge)
// =============================================================================

__device__ __forceinline__ int ds7_find_ba_for_ptr(const PtrLookupEntry* __restrict__ table,
                                                   int num_bas,
                                                   uintptr_t target) {
    int lo = 0;
    int hi = num_bas;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (table[mid].base_ptr <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == 0) return -1;
    return table[lo - 1].ba_idx;
}

__device__ __forceinline__ void ds7_reclaim_one_block(bool valid,
                                                      int old_log_b,
                                                      const vid_t* old_ptr,
                                                      const DeviceMemoryView& mem) {
    bool newly_freed = false;
    int ba_idx = -1;

    if (valid && old_ptr != nullptr && old_log_b >= 0 && old_log_b < MemoryManager::MAX_BINS) {
        int nbas = mem.d_bin_num_bas[old_log_b];
        const PtrLookupEntry* table = mem.d_bin_ptr_lookup[old_log_b];
        uintptr_t target = reinterpret_cast<uintptr_t>(old_ptr);
        ba_idx = ds7_find_ba_for_ptr(table, nbas, target);
        if (ba_idx >= 0) {
            const BlockArrayDesc& desc = mem.d_bin_descs[old_log_b][ba_idx];
            uintptr_t base = reinterpret_cast<uintptr_t>(desc.d_edges);
            size_t byte_off = static_cast<size_t>(target - base);
            size_t block_idx_sz = (byte_off / sizeof(vid_t)) >> desc.log_bsize;
            if (target >= base && block_idx_sz < static_cast<size_t>(desc.num_blocks)) {
                int block_idx = static_cast<int>(block_idx_sz);
                int L = desc.num_levels;
                uint64_t* tree = desc.d_bittree;
                int word = block_idx >> 6;
                int bit = block_idx & 63;
                uint64_t mask = 1ULL << bit;
                unsigned long long prev = atomicOr(
                    reinterpret_cast<unsigned long long*>(&tree[desc.level_offsets[L - 1] + word]),
                    static_cast<unsigned long long>(mask));
                newly_freed = ((prev & mask) == 0ULL);
                if (newly_freed && prev == 0ULL) {
                    int child_word = word;
                    for (int l = L - 2; l >= 0; --l) {
                        int p_word = child_word >> 6;
                        int p_bit = child_word & 63;
                        uint64_t p_mask = 1ULL << p_bit;
                        unsigned long long p_prev = atomicOr(
                            reinterpret_cast<unsigned long long*>(&tree[desc.level_offsets[l] + p_word]),
                            static_cast<unsigned long long>(p_mask));
                        if (p_prev != 0ULL) break;
                        child_word = p_word;
                    }
                }
            }
        }
    }

    if (newly_freed) {
        unsigned active = __activemask();
        uint64_t key = (static_cast<uint64_t>(old_log_b) << 32) | static_cast<uint32_t>(ba_idx);
        unsigned peer_mask = __match_any_sync(active, key);
        int lane = threadIdx.x & 31;
        int leader = __ffs(peer_mask) - 1;
        if (lane == leader) {
            atomicAdd(&mem.d_bin_free_counts[old_log_b][ba_idx], __popc(peer_mask));
            if (*reinterpret_cast<const volatile int*>(&mem.d_bin_first_nonfull[old_log_b]) > ba_idx) {
                atomicMin(&mem.d_bin_first_nonfull[old_log_b], ba_idx);
            }
        }
    }
}

__global__ void ds7_reclaim_blocks_kernel(int num_items,
                                          DeviceMemoryView mem,
                                          const vid_t* __restrict__ d_old_deg,
                                          vid_t* const* __restrict__ d_old_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = false;
    int old_log_b = 0;
    const vid_t* ptr = nullptr;
    if (i < num_items) {
        vid_t deg = d_old_deg[i];
        ptr = d_old_ptr[i];
        if (deg > 0 && ptr != nullptr) {
            valid = true;
            old_log_b = ds7_ceil_log2(static_cast<int>(deg));
        }
    }
    ds7_reclaim_one_block(valid, old_log_b, ptr, mem);
}

__global__ void ds7_reclaim_masked_blocks_kernel(int num_items,
                                                 DeviceMemoryView mem,
                                                 const uint8_t* __restrict__ d_mask,
                                                 const vid_t* __restrict__ d_old_deg,
                                                 vid_t* const* __restrict__ d_old_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = false;
    int old_log_b = 0;
    const vid_t* ptr = nullptr;
    if (i < num_items && d_mask[i] != 0) {
        vid_t deg = d_old_deg[i];
        ptr = d_old_ptr[i];
        if (deg > 0 && ptr != nullptr) {
            valid = true;
            old_log_b = ds7_ceil_log2(static_cast<int>(deg));
        }
    }
    ds7_reclaim_one_block(valid, old_log_b, ptr, mem);
}

// =============================================================================
// Host Drivers for DS-7 GPU Alloc & Reclaim
// =============================================================================

inline void ds7_launch_claim_kernels(const MemoryManager& mgr,
                                     const DS7AllocWorkspaceView& aw,
                                     int num_reqs,
                                     vid_t** d_out_new_ptr) {
    DeviceMemoryView mem = mgr.device_view();
    ds7_plan_slab_allocs_kernel<<<MemoryManager::MAX_BINS, 32>>>(
        mem, aw.d_bin_needed, aw.d_bin_offsets, aw.d_slab_tasks,
        aw.d_single_task_idx, aw.d_med_task_idx, aw.d_huge_task_idx, aw.d_alloc_counts);
    CUDA_CHECK(cudaGetLastError());

    if (num_reqs <= 256) {
        ds7_alloc_small_or_med_unified_kernel<<<num_reqs, 256>>>(
            &aw.d_alloc_counts[3], aw.d_slab_tasks, mem, aw.d_bucket_req_idx, d_out_new_ptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    CUDA_CHECK(cudaMemcpy(aw.h_pinned_counts, aw.d_alloc_counts,
                          3 * sizeof(int), cudaMemcpyDeviceToHost));
    int num_single = aw.h_pinned_counts[0];
    int num_med = aw.h_pinned_counts[1];
    int num_huge = aw.h_pinned_counts[2];

    if (num_single > 0) {
        int blk = 256;
        ds7_alloc_single_bittree_kernel<<<(num_single + blk - 1) / blk, blk>>>(
            num_single, aw.d_single_task_idx, aw.d_slab_tasks,
            mem, aw.d_bucket_req_idx, d_out_new_ptr);
        CUDA_CHECK(cudaGetLastError());
    }
    if (num_med > 0) {
        ds7_alloc_med_bittree_kernel<<<num_med, 256>>>(
            num_med, aw.d_med_task_idx, aw.d_slab_tasks,
            mem, aw.d_bucket_req_idx, d_out_new_ptr);
        CUDA_CHECK(cudaGetLastError());
    }
    if (num_huge > 0) {
        ds7_huge_slab_count_kernel<<<num_huge * 32, 256>>>(
            num_huge, aw.d_huge_task_idx, aw.d_slab_tasks, mem, aw.d_huge_chunk_counts);
        CUDA_CHECK(cudaGetLastError());
        ds7_huge_slab_claim_kernel<<<num_huge * 32, 256>>>(
            num_huge, aw.d_huge_task_idx, aw.d_slab_tasks, mem,
            aw.d_huge_chunk_counts, aw.d_bucket_req_idx, d_out_new_ptr);
        CUDA_CHECK(cudaGetLastError());
        ds7_huge_slab_rebuild_kernel<<<num_huge, 256>>>(
            num_huge, aw.d_huge_task_idx, aw.d_slab_tasks, mem);
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * @brief Runs DS-7 pre-check, optional slab-exhaustion fallback, and GPU BitTree allocation.
 *
 * When `track_phases` is true, accumulates GPU allocation time into `inout_gpu_phase_ms`
 * and records `host_alloc_phase` + `h2d_phase` (0.0 ms when deficit == 0).
 */
inline void ds7_gpu_allocate_blocks(MemoryManager& mgr,
                                    const DS7AllocWorkspaceView& aw,
                                    int num_reqs,
                                    const vid_t* d_new_deg,
                                    vid_t** d_out_new_ptr,
                                    bool track_phases,
                                    const char* gpu_alloc_nvtx_name,
                                    const char* host_alloc_phase,
                                    const char* h2d_phase,
                                    std::chrono::high_resolution_clock::time_point* gpu_part1_start,
                                    bool* gpu_nvtx_active,
                                    double* inout_gpu_phase_ms) {
    if (num_reqs <= 0) return;

    int blk = 256;
    CUDA_CHECK(cudaMemsetAsync(aw.d_bin_needed, 0, MemoryManager::MAX_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(aw.d_alloc_counts, 0, 4 * sizeof(int)));

    ds7_count_requests_per_bin_kernel<<<(num_reqs + blk - 1) / blk, blk>>>(
        num_reqs, d_new_deg, aw.d_req_bin, aw.d_bin_needed);
    CUDA_CHECK(cudaGetLastError());

    ds7_check_bin_deficit_kernel<<<1, 1024>>>(
        mgr.device_view(), aw.d_bin_needed, aw.d_bin_offsets, aw.d_bin_cursor, aw.d_bin_deficit);
    CUDA_CHECK(cudaGetLastError());

    ds7_bucket_scatter_kernel<<<(num_reqs + blk - 1) / blk, blk>>>(
        num_reqs, aw.d_req_bin, aw.d_bin_cursor, aw.d_bucket_req_idx);
    CUDA_CHECK(cudaGetLastError());

    // Pre-claim check: copy only the 32-int shortfall array to reserved pinned slots
    int* h_deficit = aw.h_pinned_counts + DS7AllocWorkspaceView::PINNED_BIN_DEFICIT_OFFSET;
    CUDA_CHECK(cudaMemcpy(h_deficit, aw.d_bin_deficit,
                          MemoryManager::MAX_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    bool has_deficit = false;
    for (int b = 0; b < MemoryManager::MAX_BINS; ++b) {
        if (h_deficit[b] > 0) {
            has_deficit = true;
            break;
        }
    }

    if (!has_deficit) {
        if (track_phases) {
            HornetProfiler::instance().record_phase(host_alloc_phase, 0.0);
            HornetProfiler::instance().record_phase(h2d_phase, 0.0);
        }
        ds7_launch_claim_kernels(mgr, aw, num_reqs, d_out_new_ptr);
        return;
    }

    // Pause part-1 GPU timer cleanly before host fallback
    if (track_phases && gpu_part1_start && inout_gpu_phase_ms) {
        cudaDeviceSynchronize();
        auto now = std::chrono::high_resolution_clock::now();
        *inout_gpu_phase_ms += std::chrono::duration<double, std::milli>(now - *gpu_part1_start).count();
        if (gpu_nvtx_active && *gpu_nvtx_active && HornetProfiler::instance().is_nvtx_enabled()) {
            nvtxRangePop();
            *gpu_nvtx_active = false;
        }
    }

    PendingFallbackUpload pending;
    {
        ScopedPhaseTimer host_alloc_timer(host_alloc_phase, false);
        mgr.prepare_fallback_slabs(h_deficit, pending);
    }
    {
        ScopedPhaseTimer h2d_timer(h2d_phase, true);
        mgr.commit_fallback_slabs(pending);
    }

    std::chrono::high_resolution_clock::time_point alloc_t0;
    if (track_phases) {
        if (HornetProfiler::instance().is_nvtx_enabled()) {
            nvtxRangePushA(gpu_alloc_nvtx_name);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        alloc_t0 = std::chrono::high_resolution_clock::now();
    }

    ds7_launch_claim_kernels(mgr, aw, num_reqs, d_out_new_ptr);

    if (track_phases && inout_gpu_phase_ms) {
        cudaDeviceSynchronize();
        auto alloc_t1 = std::chrono::high_resolution_clock::now();
        *inout_gpu_phase_ms += std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();
        if (HornetProfiler::instance().is_nvtx_enabled()) {
            nvtxRangePop();
        }
    }
}

inline void ds7_gpu_reclaim_blocks(const MemoryManager& mgr,
                                   int num_items,
                                   const vid_t* d_old_deg,
                                   vid_t* const* d_old_ptr) {
    if (num_items <= 0) return;
    int blk = 256;
    ds7_reclaim_blocks_kernel<<<(num_items + blk - 1) / blk, blk>>>(
        num_items, mgr.device_view(), d_old_deg, d_old_ptr);
    CUDA_CHECK(cudaGetLastError());
}

inline void ds7_gpu_reclaim_masked_blocks(const MemoryManager& mgr,
                                          int num_items,
                                          const uint8_t* d_mask,
                                          const vid_t* d_old_deg,
                                          vid_t* const* d_old_ptr) {
    if (num_items <= 0) return;
    int blk = 256;
    ds7_reclaim_masked_blocks_kernel<<<(num_items + blk - 1) / blk, blk>>>(
        num_items, mgr.device_view(), d_mask, d_old_deg, d_old_ptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace improved
