#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

namespace baseline {

/**
 * @brief Vectorized Bit Tree (Paper Section III-B, Fig. 2).
 *
 * Tracks free (1) vs used (0) blocks in a BlockArray using a hierarchical
 * bitwise-OR tree of 64-bit machine words.
 * - Root check is O(1).
 * - Empty block lookup and reclamation are O(log_64(N)) using __builtin_ctzll.
 */
class BitTree {
public:
    BitTree() = default;

    explicit BitTree(int num_blocks) { init(num_blocks); }

    void init(int num_blocks) {
        assert(num_blocks >= 1);
        num_blocks_ = num_blocks;
        free_count_ = num_blocks;
        levels_.clear();

        // Build bottom-up 64-ary OR tree
        std::vector<std::vector<uint64_t>> bottom_up;

        int leaf_words = (num_blocks + 63) / 64;
        std::vector<uint64_t> leaf(leaf_words, ~0ULL);
        int rem = num_blocks % 64;
        if (rem != 0) {
            leaf.back() = (1ULL << rem) - 1ULL;
        }
        bottom_up.push_back(std::move(leaf));

        while (bottom_up.back().size() > 1) {
            const auto& child = bottom_up.back();
            int parent_words = (static_cast<int>(child.size()) + 63) / 64;
            std::vector<uint64_t> parent(parent_words, 0ULL);
            for (size_t i = 0; i < child.size(); ++i) {
                if (child[i] != 0ULL) {
                    parent[i >> 6] |= (1ULL << (i & 63));
                }
            }
            bottom_up.push_back(std::move(parent));
        }

        // Store top-down: levels_[0] is root, levels_.back() is leaf
        levels_.assign(bottom_up.rbegin(), bottom_up.rend());
    }

    bool has_free() const {
        return !levels_.empty() && levels_[0][0] != 0ULL;
    }

    int free_count() const { return free_count_; }
    int num_blocks() const { return num_blocks_; }
    int used_count() const { return num_blocks_ - free_count_; }

    int allocate_block() {
        if (!has_free()) return -1;

        int num_levels = static_cast<int>(levels_.size());
        int idx = 0;
        for (int l = 0; l < num_levels - 1; ++l) {
            int bit = __builtin_ctzll(levels_[l][idx]);
            idx = (idx << 6) + bit;
        }

        int leaf_word = idx;
        int leaf_bit = __builtin_ctzll(levels_[num_levels - 1][leaf_word]);
        int block_idx = (leaf_word << 6) + leaf_bit;

        // Clear bit at leaf and propagate emptiness upward
        levels_[num_levels - 1][leaf_word] &= ~(1ULL << leaf_bit);
        int child_word = leaf_word;
        for (int l = num_levels - 2; l >= 0; --l) {
            if (levels_[l + 1][child_word] != 0ULL) {
                break;
            }
            int p_word = child_word >> 6;
            int p_bit = child_word & 63;
            levels_[l][p_word] &= ~(1ULL << p_bit);
            child_word = p_word;
        }

        --free_count_;
        return block_idx;
    }

    void free_block(int block_idx) {
        assert(block_idx >= 0 && block_idx < num_blocks_);
        int num_levels = static_cast<int>(levels_.size());

        int word = block_idx >> 6;
        int bit = block_idx & 63;
        bool was_free = ((levels_[num_levels - 1][word] >> bit) & 1ULL) != 0ULL;
        if (was_free) return;

        levels_[num_levels - 1][word] |= (1ULL << bit);
        int child_word = word;
        for (int l = num_levels - 2; l >= 0; --l) {
            int p_word = child_word >> 6;
            int p_bit = child_word & 63;
            levels_[l][p_word] |= (1ULL << p_bit);
            child_word = p_word;
        }

        ++free_count_;
    }

private:
    int num_blocks_ = 0;
    int free_count_ = 0;
    std::vector<std::vector<uint64_t>> levels_;
};

} // namespace baseline
