#pragma once

#include "block_array.cuh"
#include "types.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace improved {

struct BlockArrayDesc {
    vid_t* d_edges = nullptr;
    uint64_t* d_bittree = nullptr;
    int num_blocks = 0;
    int num_levels = 0;
    int log_bsize = 0;
    int level_offsets[BitTree::MAX_LEVELS + 1] = {};
};

struct PtrLookupEntry {
    uintptr_t base_ptr = 0;
    int ba_idx = 0;
};

struct DeviceMemoryView {
    BlockArrayDesc* const* d_bin_descs = nullptr;
    int* const* d_bin_free_counts = nullptr;
    const PtrLookupEntry* const* d_bin_ptr_lookup = nullptr;
    const int* d_bin_num_bas = nullptr;
    int* d_bin_first_nonfull = nullptr;
    size_t max_edges_per_ba = 0;
};

struct PendingBinFallback {
    int bin = 0;
    int old_n = 0;
    int new_n = 0;
    std::vector<BlockArrayDesc> new_descs;
    std::vector<int> new_free_counts;
    std::vector<BlockArray*> new_bas;
};

struct PendingFallbackUpload {
    std::vector<PendingBinFallback> bins;
};

/**
 * @brief DS-7 GPU-Resident BlockArray Manager.
 *
 * Replaces host-side B+Tree/std::map lookups on the update hot path with flat
 * per-bin GPU BlockArray descriptors (`d_edges`, `d_bittree`, `num_blocks`),
 * contiguous GPU `free_count[]` arrays, `first_nonfull_ba[32]` cursors, and
 * sorted `{base_ptr, ba_idx}` device lookup tables for O(log #BAs) GPU reclaim.
 */
class MemoryManager {
public:
    static constexpr int MAX_BINS = 32;

    explicit MemoryManager(size_t max_edges_per_ba = (1ULL << 23))
        : max_edges_per_ba_(max_edges_per_ba) {
        CUDA_CHECK(cudaMalloc(&d_bin_descs_arr_, MAX_BINS * sizeof(BlockArrayDesc*)));
        CUDA_CHECK(cudaMalloc(&d_bin_free_counts_arr_, MAX_BINS * sizeof(int*)));
        CUDA_CHECK(cudaMalloc(&d_bin_ptr_lookup_arr_, MAX_BINS * sizeof(PtrLookupEntry*)));
        CUDA_CHECK(cudaMalloc(&d_bin_num_bas_, MAX_BINS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bin_first_nonfull_, MAX_BINS * sizeof(int)));
        reset_device_directory();
    }

    ~MemoryManager() {
        // Directory pointers are freed immediately below, so skip zeroing them.
        clear(false);
        if (d_bin_descs_arr_) cudaFree(d_bin_descs_arr_);
        if (d_bin_free_counts_arr_) cudaFree(d_bin_free_counts_arr_);
        if (d_bin_ptr_lookup_arr_) cudaFree(d_bin_ptr_lookup_arr_);
        if (d_bin_num_bas_) cudaFree(d_bin_num_bas_);
        if (d_bin_first_nonfull_) cudaFree(d_bin_first_nonfull_);
        d_bin_descs_arr_ = nullptr;
        d_bin_free_counts_arr_ = nullptr;
        d_bin_ptr_lookup_arr_ = nullptr;
        d_bin_num_bas_ = nullptr;
        d_bin_first_nonfull_ = nullptr;
    }

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    void clear(bool reset_dir = true) {
        for (int b = 0; b < MAX_BINS; ++b) {
            if (d_bin_descs_buf_[b]) {
                cudaFreeAsync(d_bin_descs_buf_[b], 0);
                d_bin_descs_buf_[b] = nullptr;
            }
            if (d_bin_free_counts_buf_[b]) {
                cudaFreeAsync(d_bin_free_counts_buf_[b], 0);
                d_bin_free_counts_buf_[b] = nullptr;
            }
            if (d_bin_ptr_lookup_buf_[b]) {
                cudaFreeAsync(d_bin_ptr_lookup_buf_[b], 0);
                d_bin_ptr_lookup_buf_[b] = nullptr;
            }
            d_bin_cap_[b] = 0;
            h_ptr_lookup_[b].clear();
            all_bas_[b].clear();
        }
        if (reset_dir && d_bin_descs_arr_) {
            reset_device_directory();
        }
    }

    DeviceMemoryView device_view() const {
        return DeviceMemoryView{
            d_bin_descs_arr_,
            d_bin_free_counts_arr_,
            d_bin_ptr_lookup_arr_,
            d_bin_num_bas_,
            d_bin_first_nonfull_,
            max_edges_per_ba_
        };
    }

    void flush_all_staging() {
        for (int b = 0; b < MAX_BINS; ++b) {
            for (auto& ba : all_bas_[b]) {
                ba->flush_staging_to_device();
            }
        }
    }

    std::vector<BlockArray*> preallocate_bin_for_init(int log_bsize, int num_needed_blocks,
                                                       int& blocks_per_ba) {
        std::vector<BlockArray*> result;
        if (num_needed_blocks <= 0) {
            blocks_per_ba = 0;
            return result;
        }

        blocks_per_ba = static_cast<int>(std::max<size_t>(
            1, max_edges_per_ba_ / (1ULL << log_bsize)));
        int remaining = num_needed_blocks;
        while (remaining > 0) {
            auto new_ba = std::make_unique<BlockArray>(log_bsize, max_edges_per_ba_);
            const int take = std::min(remaining, blocks_per_ba);
            new_ba->enable_host_staging(take);
            new_ba->mark_prefix_used(take);
            BlockArray* ba = new_ba.get();
            int ba_idx = static_cast<int>(all_bas_[log_bsize].size());
            h_ptr_lookup_[log_bsize].push_back(
                {reinterpret_cast<uintptr_t>(ba->device_base()), ba_idx});
            all_bas_[log_bsize].push_back(std::move(new_ba));
            result.push_back(ba);
            remaining -= take;
        }
        return result;
    }

    void sync_all_bins_to_device() {
        BlockArrayDesc* h_descs_ptrs[MAX_BINS] = {};
        int* h_free_ptrs[MAX_BINS] = {};
        PtrLookupEntry* h_lookup_ptrs[MAX_BINS] = {};
        int h_num_bas[MAX_BINS] = {};
        int h_first_nonfull[MAX_BINS] = {};

        for (int b = 0; b < MAX_BINS; ++b) {
            int n = static_cast<int>(all_bas_[b].size());
            h_num_bas[b] = n;
            if (n == 0) {
                h_first_nonfull[b] = 0;
                continue;
            }

            ensure_bin_device_capacity(b, n, 0, false);

            std::vector<BlockArrayDesc> h_descs(n);
            std::vector<int> h_free(n);
            int first_nonfull = n;

            for (int i = 0; i < n; ++i) {
                BlockArray* ba = all_bas_[b][i].get();
                ba->upload_bittree_to_device(0);
                h_descs[i] = make_desc(ba);
                int fc = ba->free_count();
                h_free[i] = fc;
                if (first_nonfull == n && fc > 0) {
                    first_nonfull = i;
                }
            }

            std::sort(h_ptr_lookup_[b].begin(), h_ptr_lookup_[b].end(),
                      [](const PtrLookupEntry& a, const PtrLookupEntry& c) {
                          return a.base_ptr < c.base_ptr;
                      });

            CUDA_CHECK(cudaMemcpyAsync(d_bin_descs_buf_[b], h_descs.data(),
                                       n * sizeof(BlockArrayDesc), cudaMemcpyHostToDevice, 0));
            CUDA_CHECK(cudaMemcpyAsync(d_bin_free_counts_buf_[b], h_free.data(),
                                       n * sizeof(int), cudaMemcpyHostToDevice, 0));
            CUDA_CHECK(cudaMemcpyAsync(d_bin_ptr_lookup_buf_[b], h_ptr_lookup_[b].data(),
                                       n * sizeof(PtrLookupEntry), cudaMemcpyHostToDevice, 0));

            h_descs_ptrs[b] = d_bin_descs_buf_[b];
            h_free_ptrs[b] = d_bin_free_counts_buf_[b];
            h_lookup_ptrs[b] = d_bin_ptr_lookup_buf_[b];
            h_first_nonfull[b] = first_nonfull;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_bin_descs_arr_, h_descs_ptrs,
                                   MAX_BINS * sizeof(BlockArrayDesc*), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_free_counts_arr_, h_free_ptrs,
                                   MAX_BINS * sizeof(int*), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_ptr_lookup_arr_, h_lookup_ptrs,
                                   MAX_BINS * sizeof(PtrLookupEntry*), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_num_bas_, h_num_bas,
                                   MAX_BINS * sizeof(int), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_first_nonfull_, h_first_nonfull,
                                   MAX_BINS * sizeof(int), cudaMemcpyHostToDevice, 0));
    }

    void prepare_fallback_slabs(const int h_deficit[MAX_BINS], PendingFallbackUpload& out) {
        out.bins.clear();
        for (int b = 0; b < MAX_BINS; ++b) {
            if (h_deficit[b] <= 0) continue;

            PendingBinFallback pb;
            pb.bin = b;
            pb.old_n = static_cast<int>(all_bas_[b].size());
            int blocks_per_ba = static_cast<int>(std::max<size_t>(
                1, max_edges_per_ba_ / (1ULL << b)));
            int add_cnt = (h_deficit[b] + blocks_per_ba - 1) / blocks_per_ba;
            pb.new_n = pb.old_n + add_cnt;

            pb.new_descs.reserve(add_cnt);
            pb.new_free_counts.reserve(add_cnt);
            pb.new_bas.reserve(add_cnt);

            for (int k = 0; k < add_cnt; ++k) {
                auto new_ba = std::make_unique<BlockArray>(b, max_edges_per_ba_);
                BlockArray* ba = new_ba.get();
                int ba_idx = pb.old_n + k;
                h_ptr_lookup_[b].push_back(
                    {reinterpret_cast<uintptr_t>(ba->device_base()), ba_idx});
                pb.new_descs.push_back(make_desc(ba));
                pb.new_free_counts.push_back(ba->free_count());
                pb.new_bas.push_back(ba);
                all_bas_[b].push_back(std::move(new_ba));
            }

            std::sort(h_ptr_lookup_[b].begin(), h_ptr_lookup_[b].end(),
                      [](const PtrLookupEntry& a, const PtrLookupEntry& c) {
                          return a.base_ptr < c.base_ptr;
                      });

            out.bins.push_back(std::move(pb));
        }
    }

    void commit_fallback_slabs(const PendingFallbackUpload& pending) {
        if (pending.bins.empty()) return;

        BlockArrayDesc* h_descs_ptrs[MAX_BINS] = {};
        int* h_free_ptrs[MAX_BINS] = {};
        PtrLookupEntry* h_lookup_ptrs[MAX_BINS] = {};
        int h_num_bas[MAX_BINS] = {};
        for (int b = 0; b < MAX_BINS; ++b) {
            h_descs_ptrs[b] = d_bin_descs_buf_[b];
            h_free_ptrs[b] = d_bin_free_counts_buf_[b];
            h_lookup_ptrs[b] = d_bin_ptr_lookup_buf_[b];
            h_num_bas[b] = static_cast<int>(all_bas_[b].size());
        }

        for (const auto& pb : pending.bins) {
            int b = pb.bin;
            for (BlockArray* ba : pb.new_bas) {
                ba->upload_bittree_to_device(0);
            }

            // Preserve existing live GPU descriptors & free_counts when expanding
            ensure_bin_device_capacity(b, pb.new_n, pb.old_n, true);

            int add_cnt = pb.new_n - pb.old_n;
            CUDA_CHECK(cudaMemcpyAsync(d_bin_descs_buf_[b] + pb.old_n, pb.new_descs.data(),
                                       add_cnt * sizeof(BlockArrayDesc), cudaMemcpyHostToDevice, 0));
            CUDA_CHECK(cudaMemcpyAsync(d_bin_free_counts_buf_[b] + pb.old_n, pb.new_free_counts.data(),
                                       add_cnt * sizeof(int), cudaMemcpyHostToDevice, 0));
            CUDA_CHECK(cudaMemcpyAsync(d_bin_ptr_lookup_buf_[b], h_ptr_lookup_[b].data(),
                                       pb.new_n * sizeof(PtrLookupEntry), cudaMemcpyHostToDevice, 0));

            h_descs_ptrs[b] = d_bin_descs_buf_[b];
            h_free_ptrs[b] = d_bin_free_counts_buf_[b];
            h_lookup_ptrs[b] = d_bin_ptr_lookup_buf_[b];
            h_num_bas[b] = pb.new_n;
        }

        // Update directory pointers & counts while keeping live d_bin_first_nonfull_ untouched
        CUDA_CHECK(cudaMemcpyAsync(d_bin_descs_arr_, h_descs_ptrs,
                                   MAX_BINS * sizeof(BlockArrayDesc*), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_free_counts_arr_, h_free_ptrs,
                                   MAX_BINS * sizeof(int*), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_ptr_lookup_arr_, h_lookup_ptrs,
                                   MAX_BINS * sizeof(PtrLookupEntry*), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(d_bin_num_bas_, h_num_bas,
                                   MAX_BINS * sizeof(int), cudaMemcpyHostToDevice, 0));
    }

    size_t get_ba_allocated_edges() const {
        size_t sum = 0;
        for (int b = 0; b < MAX_BINS; ++b) {
            for (const auto& ba : all_bas_[b]) {
                sum += ba->total_edges();
            }
        }
        return sum;
    }

    size_t get_ba_allocated_bytes() const {
        return get_ba_allocated_edges() * sizeof(vid_t);
    }

    size_t get_ba_count() const {
        size_t count = 0;
        for (int b = 0; b < MAX_BINS; ++b) {
            count += all_bas_[b].size();
        }
        return count;
    }

    const std::vector<std::unique_ptr<BlockArray>>& bin_bas(int b) const {
        return all_bas_[b];
    }

    size_t max_edges_per_ba() const { return max_edges_per_ba_; }

private:
    static BlockArrayDesc make_desc(const BlockArray* ba) {
        BlockArrayDesc d{};
        d.d_edges = ba->device_base();
        d.d_bittree = ba->device_bittree();
        d.num_blocks = ba->num_blocks();
        d.num_levels = ba->bit_tree().num_levels();
        d.log_bsize = ba->log_bsize();
        const int* offs = ba->bit_tree().level_offsets();
        for (int l = 0; l <= d.num_levels; ++l) {
            d.level_offsets[l] = offs[l];
        }
        return d;
    }

    void reset_device_directory() {
        CUDA_CHECK(cudaMemset(d_bin_descs_arr_, 0, MAX_BINS * sizeof(BlockArrayDesc*)));
        CUDA_CHECK(cudaMemset(d_bin_free_counts_arr_, 0, MAX_BINS * sizeof(int*)));
        CUDA_CHECK(cudaMemset(d_bin_ptr_lookup_arr_, 0, MAX_BINS * sizeof(PtrLookupEntry*)));
        CUDA_CHECK(cudaMemset(d_bin_num_bas_, 0, MAX_BINS * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_bin_first_nonfull_, 0, MAX_BINS * sizeof(int)));
    }

    void ensure_bin_device_capacity(int b, int needed, int preserve_old_n, bool preserve_live_gpu) {
        if (needed <= d_bin_cap_[b]) return;
        int new_cap = std::max(needed, std::max(4, d_bin_cap_[b] * 2));

        BlockArrayDesc* new_descs = nullptr;
        int* new_free = nullptr;
        PtrLookupEntry* new_lookup = nullptr;

        cuda_malloc_async_with_pool(
            reinterpret_cast<void**>(&new_descs), new_cap * sizeof(BlockArrayDesc), 0);
        cuda_malloc_async_with_pool(
            reinterpret_cast<void**>(&new_free), new_cap * sizeof(int), 0);
        cuda_malloc_async_with_pool(
            reinterpret_cast<void**>(&new_lookup), new_cap * sizeof(PtrLookupEntry), 0);

        if (preserve_live_gpu && preserve_old_n > 0) {
            CUDA_CHECK(cudaMemcpyAsync(new_descs, d_bin_descs_buf_[b],
                                       preserve_old_n * sizeof(BlockArrayDesc),
                                       cudaMemcpyDeviceToDevice, 0));
            CUDA_CHECK(cudaMemcpyAsync(new_free, d_bin_free_counts_buf_[b],
                                       preserve_old_n * sizeof(int),
                                       cudaMemcpyDeviceToDevice, 0));
        }

        if (d_bin_descs_buf_[b]) cudaFreeAsync(d_bin_descs_buf_[b], 0);
        if (d_bin_free_counts_buf_[b]) cudaFreeAsync(d_bin_free_counts_buf_[b], 0);
        if (d_bin_ptr_lookup_buf_[b]) cudaFreeAsync(d_bin_ptr_lookup_buf_[b], 0);

        d_bin_descs_buf_[b] = new_descs;
        d_bin_free_counts_buf_[b] = new_free;
        d_bin_ptr_lookup_buf_[b] = new_lookup;
        d_bin_cap_[b] = new_cap;
    }

    size_t max_edges_per_ba_;
    std::vector<std::unique_ptr<BlockArray>> all_bas_[MAX_BINS];
    std::vector<PtrLookupEntry> h_ptr_lookup_[MAX_BINS];

    BlockArrayDesc* d_bin_descs_buf_[MAX_BINS] = {};
    int* d_bin_free_counts_buf_[MAX_BINS] = {};
    PtrLookupEntry* d_bin_ptr_lookup_buf_[MAX_BINS] = {};
    int d_bin_cap_[MAX_BINS] = {};

    BlockArrayDesc** d_bin_descs_arr_ = nullptr;
    int** d_bin_free_counts_arr_ = nullptr;
    PtrLookupEntry** d_bin_ptr_lookup_arr_ = nullptr;
    int* d_bin_num_bas_ = nullptr;
    int* d_bin_first_nonfull_ = nullptr;
};

} // namespace improved
