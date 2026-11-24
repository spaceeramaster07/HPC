#pragma once

#include <cassert>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace improved {

/**
 * @brief Vectorized Bit Tree (Paper Section III-B, Fig. 2).
 *
 * Tracks free (1) vs used (0) blocks in a BlockArray using a flat-packed
 * hierarchical bitwise-OR tree of 64-bit machine words (levels_[0] = root,
 * levels_[num_levels - 1] = leaf), suitable for direct upload to GPU VRAM.
 */
class BitTree {
public:
    static constexpr int MAX_LEVELS = 6;

    BitTree() = default;

    explicit BitTree(int num_blocks) { init(num_blocks); }

    void init(int num_blocks) {
        assert(num_blocks >= 1);
        num_blocks_ = num_blocks;

        int sizes[MAX_LEVELS];
        int L = 0;
        int n = num_blocks;
        while (true) {
            int w = (n + 63) / 64;
            assert(L < MAX_LEVELS);
            sizes[L++] = w;
            if (w <= 1) break;
            n = w;
        }
        num_levels_ = L;

        level_offsets_[0] = 0;
        for (int l = 0; l < L; ++l) {
            level_offsets_[l + 1] = level_offsets_[l] + sizes[L - 1 - l];
        }
        words_.assign(level_offsets_[L], 0ULL);
        mark_prefix_used(0);
    }

    bool has_free() const {
        return num_levels_ > 0 && !words_.empty() && words_[0] != 0ULL;
    }

    int free_count() const { return free_count_; }
    int num_blocks() const { return num_blocks_; }
    int used_count() const { return num_blocks_ - free_count_; }
    int num_levels() const { return num_levels_; }
    const int* level_offsets() const { return level_offsets_; }
    int level_size(int level) const { return level_offsets_[level + 1] - level_offsets_[level]; }
    int total_words() const { return level_offsets_[num_levels_]; }
    size_t total_bytes() const { return static_cast<size_t>(total_words()) * sizeof(uint64_t); }
    const uint64_t* words_data() const { return words_.data(); }

    void mark_prefix_used(int used_blocks) {
        const int clamped = used_blocks < 0 ? 0 : (used_blocks > num_blocks_ ? num_blocks_ : used_blocks);
        const int leaf_off = level_offsets_[num_levels_ - 1];
        const int leaf_words = level_size(num_levels_ - 1);

        for (int word = 0; word < leaf_words; ++word) {
            const int first_block = word * 64;
            const int used_in_word = std::max(0, std::min(64, clamped - first_block));
            uint64_t free_mask = used_in_word >= 64 ? 0ULL : (~0ULL << used_in_word);
            const int valid_bits = std::min(64, num_blocks_ - first_block);
            if (valid_bits < 64) free_mask &= (1ULL << valid_bits) - 1ULL;
            words_[leaf_off + word] = free_mask;
        }
        free_count_ = num_blocks_ - clamped;

        for (int level = num_levels_ - 2; level >= 0; --level) {
            const int p_off = level_offsets_[level];
            const int p_len = level_size(level);
            const int c_off = level_offsets_[level + 1];
            const int c_len = level_size(level + 1);
            std::fill(words_.begin() + p_off, words_.begin() + p_off + p_len, 0ULL);
            for (int i = 0; i < c_len; ++i) {
                if (words_[c_off + i] != 0ULL) {
                    words_[p_off + (i >> 6)] |= (1ULL << (i & 63));
                }
            }
        }
    }

    int allocate_block() {
        if (!has_free()) return -1;

        int idx = 0;
        for (int l = 0; l < num_levels_ - 1; ++l) {
            int bit = __builtin_ctzll(words_[level_offsets_[l] + idx]);
            idx = (idx << 6) + bit;
        }

        const int leaf_off = level_offsets_[num_levels_ - 1];
        int leaf_word = idx;
        int leaf_bit = __builtin_ctzll(words_[leaf_off + leaf_word]);
        int block_idx = (leaf_word << 6) + leaf_bit;

        // Clear bit at leaf and propagate emptiness upward
        words_[leaf_off + leaf_word] &= ~(1ULL << leaf_bit);
        int child_word = leaf_word;
        for (int l = num_levels_ - 2; l >= 0; --l) {
            if (words_[level_offsets_[l + 1] + child_word] != 0ULL) {
                break;
            }
            int p_word = child_word >> 6;
            int p_bit = child_word & 63;
            words_[level_offsets_[l] + p_word] &= ~(1ULL << p_bit);
            child_word = p_word;
        }

        --free_count_;
        return block_idx;
    }

    void free_block(int block_idx) {
        assert(block_idx >= 0 && block_idx < num_blocks_);

        const int leaf_off = level_offsets_[num_levels_ - 1];
        int word = block_idx >> 6;
        int bit = block_idx & 63;
        bool was_free = ((words_[leaf_off + word] >> bit) & 1ULL) != 0ULL;
        if (was_free) return;

        words_[leaf_off + word] |= (1ULL << bit);
        int child_word = word;
        for (int l = num_levels_ - 2; l >= 0; --l) {
            int p_word = child_word >> 6;
            int p_bit = child_word & 63;
            words_[level_offsets_[l] + p_word] |= (1ULL << p_bit);
            child_word = p_word;
        }

        ++free_count_;
    }

private:
    int num_blocks_ = 0;
    int free_count_ = 0;
    int num_levels_ = 0;
    int level_offsets_[MAX_LEVELS + 1] = {};
    std::vector<uint64_t> words_;
};

} // namespace improved
