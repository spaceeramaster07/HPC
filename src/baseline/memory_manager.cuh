#pragma once

#include "block_array.cuh"
#include "btree.hpp"
#include "types.cuh"

#include <cstdint>
#include <memory>
#include <vector>

namespace baseline {

/**
 * @brief B+Tree BlockArray Manager (Paper Section III-C, Fig. 3).
 *
 * Maintains an array of B+Trees (one per log_bsize in [0, 31]).
 * Each B+Tree stores <free_blocks, BlockArray*> ordered by free_blocks.
 */
class MemoryManager {
public:
    static constexpr int MAX_BINS = 32;

    struct BTreeKey {
        int free_blocks = 0;
        uintptr_t ba_id = 0;

        bool operator<(const BTreeKey& other) const {
            if (free_blocks != other.free_blocks) {
                return free_blocks < other.free_blocks;
            }
            return ba_id < other.ba_id;
        }
    };

    struct AllocResult {
        vid_t* ptr = nullptr;
        BlockArray* ba = nullptr;
        int block_idx = -1;
    };

    explicit MemoryManager(size_t max_edges_per_ba = (1ULL << 23))
        : max_edges_per_ba_(max_edges_per_ba) {}

    ~MemoryManager() { clear(); }

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    void clear() {
        for (int b = 0; b < MAX_BINS; ++b) {
            btrees_[b].clear();
            all_bas_[b].clear();
        }
    }

    AllocResult allocate_block(int log_bsize, bool enable_staging = false) {
        BTreeKey found_key{};
        BlockArray* ba = nullptr;

        if (!btrees_[log_bsize].lower_bound({1, 0}, found_key, ba)) {
            auto new_ba = std::make_unique<BlockArray>(log_bsize, max_edges_per_ba_);
            if (enable_staging) {
                new_ba->enable_host_staging();
            }
            ba = new_ba.get();
            all_bas_[log_bsize].push_back(std::move(new_ba));
            btrees_[log_bsize].insert({ba->free_count(), reinterpret_cast<uintptr_t>(ba)}, ba);
        }

        int old_free = ba->free_count();
        int block_idx = -1;
        vid_t* ptr = ba->allocate_block(&block_idx);

        // Update B+Tree key for this BlockArray
        btrees_[log_bsize].erase({old_free, reinterpret_cast<uintptr_t>(ba)});
        btrees_[log_bsize].insert({ba->free_count(), reinterpret_cast<uintptr_t>(ba)}, ba);

        return {ptr, ba, block_idx};
    }

    void reclaim_block(const vid_t* ptr, int old_log_bsize) {
        if (!ptr) return;
        BlockArray* target = nullptr;
        for (const auto& ba_uptr : all_bas_[old_log_bsize]) {
            if (ba_uptr->owns_ptr(ptr)) {
                target = ba_uptr.get();
                break;
            }
        }
        if (!target) return;

        int old_free = target->free_count();
        target->free_block_by_ptr(ptr);

        btrees_[old_log_bsize].erase({old_free, reinterpret_cast<uintptr_t>(target)});
        btrees_[old_log_bsize].insert({target->free_count(), reinterpret_cast<uintptr_t>(target)}, target);
    }

    void flush_all_staging() {
        for (int b = 0; b < MAX_BINS; ++b) {
            for (auto& ba : all_bas_[b]) {
                ba->flush_staging_to_device();
            }
        }
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

private:
    size_t max_edges_per_ba_;
    BPlusTree<BTreeKey, BlockArray*> btrees_[MAX_BINS];
    std::vector<std::unique_ptr<BlockArray>> all_bas_[MAX_BINS];
};

} // namespace baseline
