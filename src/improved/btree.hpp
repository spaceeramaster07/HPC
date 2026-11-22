#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace improved {

/**
 * @brief Clean in-memory B+Tree (Paper Section III-C, Fig. 3).
 *
 * Stores <key, value> pairs in sorted leaf nodes linked sequentially.
 * Used by MemoryManager (one B+Tree per block size) to look up BlockArrays
 * with available free blocks in O(log(#BlockArrays)) time.
 */
template <typename Key, typename Value, int MAX_KEYS = 31>
class BPlusTree {
    static_assert(MAX_KEYS >= 3, "B+Tree MAX_KEYS must be at least 3");
    static constexpr int MIN_KEYS = MAX_KEYS / 2;

    struct Node {
        bool is_leaf;
        int num_keys = 0;
        Key keys[MAX_KEYS + 1];

        explicit Node(bool leaf) : is_leaf(leaf) {}
        virtual ~Node() = default;
    };

    struct LeafNode : public Node {
        Value vals[MAX_KEYS + 1];
        LeafNode* next = nullptr;

        LeafNode() : Node(true) {}
    };

    struct InternalNode : public Node {
        Node* children[MAX_KEYS + 2]{};

        InternalNode() : Node(false) {}
        ~InternalNode() override {
            for (int i = 0; i <= this->num_keys; ++i) {
                delete children[i];
            }
        }
    };

    struct SplitResult {
        bool split = false;
        Key promoted_key{};
        Node* right_node = nullptr;
    };

public:
    BPlusTree() = default;
    ~BPlusTree() { clear(); }

    BPlusTree(const BPlusTree&) = delete;
    BPlusTree& operator=(const BPlusTree&) = delete;

    BPlusTree(BPlusTree&& other) noexcept
        : root_(other.root_), size_(other.size_) {
        other.root_ = nullptr;
        other.size_ = 0;
    }

    BPlusTree& operator=(BPlusTree&& other) noexcept {
        if (this != &other) {
            clear();
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void clear() {
        delete root_;
        root_ = nullptr;
        size_ = 0;
    }

    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }

    void insert(const Key& key, const Value& val) {
        if (!root_) {
            auto* leaf = new LeafNode();
            leaf->keys[0] = key;
            leaf->vals[0] = val;
            leaf->num_keys = 1;
            root_ = leaf;
            size_ = 1;
            return;
        }

        SplitResult res = insert_recursive(root_, key, val);
        if (res.split) {
            auto* new_root = new InternalNode();
            new_root->keys[0] = res.promoted_key;
            new_root->children[0] = root_;
            new_root->children[1] = res.right_node;
            new_root->num_keys = 1;
            root_ = new_root;
        }
    }

    bool erase(const Key& key) {
        if (!root_) return false;
        bool erased = erase_recursive(root_, key);
        if (!erased) return false;

        if (!root_->is_leaf && root_->num_keys == 0) {
            auto* old_root = static_cast<InternalNode*>(root_);
            root_ = old_root->children[0];
            old_root->children[0] = nullptr;
            old_root->num_keys = -1; // prevent destructor from deleting children
            delete old_root;
        } else if (root_->is_leaf && root_->num_keys == 0) {
            delete root_;
            root_ = nullptr;
        }
        return true;
    }

    // Finds the first entry with entry_key >= key. Returns false if none exists.
    bool lower_bound(const Key& key, Key& out_key, Value& out_val) const {
        if (!root_) return false;

        const Node* cur = root_;
        while (!cur->is_leaf) {
            const auto* internal = static_cast<const InternalNode*>(cur);
            int idx = 0;
            while (idx < internal->num_keys && !(key < internal->keys[idx])) {
                ++idx;
            }
            cur = internal->children[idx];
        }

        const auto* leaf = static_cast<const LeafNode*>(cur);
        while (leaf) {
            for (int i = 0; i < leaf->num_keys; ++i) {
                if (!(leaf->keys[i] < key)) {
                    out_key = leaf->keys[i];
                    out_val = leaf->vals[i];
                    return true;
                }
            }
            leaf = leaf->next;
        }
        return false;
    }

private:
    SplitResult insert_recursive(Node* node, const Key& key, const Value& val) {
        if (node->is_leaf) {
            auto* leaf = static_cast<LeafNode*>(node);
            int pos = 0;
            while (pos < leaf->num_keys && leaf->keys[pos] < key) {
                ++pos;
            }

            if (pos < leaf->num_keys && !(key < leaf->keys[pos]) && !(leaf->keys[pos] < key)) {
                // Key already exists: overwrite value
                leaf->vals[pos] = val;
                return {};
            }

            for (int i = leaf->num_keys; i > pos; --i) {
                leaf->keys[i] = leaf->keys[i - 1];
                leaf->vals[i] = leaf->vals[i - 1];
            }
            leaf->keys[pos] = key;
            leaf->vals[pos] = val;
            ++leaf->num_keys;
            ++size_;

            if (leaf->num_keys <= MAX_KEYS) {
                return {};
            }

            // Split leaf
            int mid = leaf->num_keys / 2;
            auto* right = new LeafNode();
            right->num_keys = leaf->num_keys - mid;
            for (int i = 0; i < right->num_keys; ++i) {
                right->keys[i] = leaf->keys[mid + i];
                right->vals[i] = leaf->vals[mid + i];
            }
            leaf->num_keys = mid;
            right->next = leaf->next;
            leaf->next = right;

            return {true, right->keys[0], right};
        }

        auto* internal = static_cast<InternalNode*>(node);
        int idx = 0;
        while (idx < internal->num_keys && !(key < internal->keys[idx])) {
            ++idx;
        }

        SplitResult child_split = insert_recursive(internal->children[idx], key, val);
        if (!child_split.split) {
            return {};
        }

        for (int i = internal->num_keys; i > idx; --i) {
            internal->keys[i] = internal->keys[i - 1];
            internal->children[i + 1] = internal->children[i];
        }
        internal->keys[idx] = child_split.promoted_key;
        internal->children[idx + 1] = child_split.right_node;
        ++internal->num_keys;

        if (internal->num_keys <= MAX_KEYS) {
            return {};
        }

        // Split internal node
        int mid = internal->num_keys / 2;
        Key up_key = internal->keys[mid];
        auto* right = new InternalNode();
        right->num_keys = internal->num_keys - mid - 1;
        for (int i = 0; i < right->num_keys; ++i) {
            right->keys[i] = internal->keys[mid + 1 + i];
            right->children[i] = internal->children[mid + 1 + i];
        }
        right->children[right->num_keys] = internal->children[internal->num_keys];
        internal->num_keys = mid;

        return {true, up_key, right};
    }

    bool erase_recursive(Node* node, const Key& key) {
        if (node->is_leaf) {
            auto* leaf = static_cast<LeafNode*>(node);
            int pos = 0;
            while (pos < leaf->num_keys && leaf->keys[pos] < key) {
                ++pos;
            }
            if (pos >= leaf->num_keys || key < leaf->keys[pos] || leaf->keys[pos] < key) {
                return false;
            }
            for (int i = pos; i < leaf->num_keys - 1; ++i) {
                leaf->keys[i] = leaf->keys[i + 1];
                leaf->vals[i] = leaf->vals[i + 1];
            }
            --leaf->num_keys;
            --size_;
            return true;
        }

        auto* internal = static_cast<InternalNode*>(node);
        int idx = 0;
        while (idx < internal->num_keys && !(key < internal->keys[idx])) {
            ++idx;
        }

        bool erased = erase_recursive(internal->children[idx], key);
        if (!erased) return false;

        Node* child = internal->children[idx];
        if (child->num_keys < MIN_KEYS) {
            rebalance_child(internal, idx);
        }
        return true;
    }

    void rebalance_child(InternalNode* parent, int idx) {
        Node* child = parent->children[idx];

        if (child->is_leaf) {
            auto* leaf = static_cast<LeafNode*>(child);

            // Try borrow from left sibling
            if (idx > 0 && parent->children[idx - 1]->num_keys > MIN_KEYS) {
                auto* left = static_cast<LeafNode*>(parent->children[idx - 1]);
                for (int i = leaf->num_keys; i > 0; --i) {
                    leaf->keys[i] = leaf->keys[i - 1];
                    leaf->vals[i] = leaf->vals[i - 1];
                }
                leaf->keys[0] = left->keys[left->num_keys - 1];
                leaf->vals[0] = left->vals[left->num_keys - 1];
                --left->num_keys;
                ++leaf->num_keys;
                parent->keys[idx - 1] = leaf->keys[0];
                return;
            }

            // Try borrow from right sibling
            if (idx < parent->num_keys && parent->children[idx + 1]->num_keys > MIN_KEYS) {
                auto* right = static_cast<LeafNode*>(parent->children[idx + 1]);
                leaf->keys[leaf->num_keys] = right->keys[0];
                leaf->vals[leaf->num_keys] = right->vals[0];
                ++leaf->num_keys;
                for (int i = 0; i < right->num_keys - 1; ++i) {
                    right->keys[i] = right->keys[i + 1];
                    right->vals[i] = right->vals[i + 1];
                }
                --right->num_keys;
                parent->keys[idx] = right->keys[0];
                return;
            }

            // Merge with left or right leaf
            if (idx > 0) {
                auto* left = static_cast<LeafNode*>(parent->children[idx - 1]);
                for (int i = 0; i < leaf->num_keys; ++i) {
                    left->keys[left->num_keys + i] = leaf->keys[i];
                    left->vals[left->num_keys + i] = leaf->vals[i];
                }
                left->num_keys += leaf->num_keys;
                left->next = leaf->next;
                delete leaf;

                for (int i = idx - 1; i < parent->num_keys - 1; ++i) {
                    parent->keys[i] = parent->keys[i + 1];
                    parent->children[i + 1] = parent->children[i + 2];
                }
                --parent->num_keys;
            } else {
                auto* right = static_cast<LeafNode*>(parent->children[idx + 1]);
                for (int i = 0; i < right->num_keys; ++i) {
                    leaf->keys[leaf->num_keys + i] = right->keys[i];
                    leaf->vals[leaf->num_keys + i] = right->vals[i];
                }
                leaf->num_keys += right->num_keys;
                leaf->next = right->next;
                delete right;

                for (int i = idx; i < parent->num_keys - 1; ++i) {
                    parent->keys[i] = parent->keys[i + 1];
                    parent->children[i + 1] = parent->children[i + 2];
                }
                --parent->num_keys;
            }
            return;
        }

        // Internal node rebalance
        auto* inode = static_cast<InternalNode*>(child);

        if (idx > 0 && parent->children[idx - 1]->num_keys > MIN_KEYS) {
            auto* left = static_cast<InternalNode*>(parent->children[idx - 1]);
            for (int i = inode->num_keys; i > 0; --i) {
                inode->keys[i] = inode->keys[i - 1];
                inode->children[i + 1] = inode->children[i];
            }
            inode->children[1] = inode->children[0];
            inode->keys[0] = parent->keys[idx - 1];
            inode->children[0] = left->children[left->num_keys];
            parent->keys[idx - 1] = left->keys[left->num_keys - 1];
            left->children[left->num_keys] = nullptr;
            --left->num_keys;
            ++inode->num_keys;
            return;
        }

        if (idx < parent->num_keys && parent->children[idx + 1]->num_keys > MIN_KEYS) {
            auto* right = static_cast<InternalNode*>(parent->children[idx + 1]);
            inode->keys[inode->num_keys] = parent->keys[idx];
            inode->children[inode->num_keys + 1] = right->children[0];
            parent->keys[idx] = right->keys[0];
            for (int i = 0; i < right->num_keys - 1; ++i) {
                right->keys[i] = right->keys[i + 1];
                right->children[i] = right->children[i + 1];
            }
            right->children[right->num_keys - 1] = right->children[right->num_keys];
            right->children[right->num_keys] = nullptr;
            --right->num_keys;
            ++inode->num_keys;
            return;
        }

        if (idx > 0) {
            auto* left = static_cast<InternalNode*>(parent->children[idx - 1]);
            left->keys[left->num_keys] = parent->keys[idx - 1];
            for (int i = 0; i < inode->num_keys; ++i) {
                left->keys[left->num_keys + 1 + i] = inode->keys[i];
                left->children[left->num_keys + 1 + i] = inode->children[i];
            }
            left->children[left->num_keys + 1 + inode->num_keys] = inode->children[inode->num_keys];
            left->num_keys += 1 + inode->num_keys;
            inode->num_keys = -1; // prevent child deletion
            delete inode;

            for (int i = idx - 1; i < parent->num_keys - 1; ++i) {
                parent->keys[i] = parent->keys[i + 1];
                parent->children[i + 1] = parent->children[i + 2];
            }
            --parent->num_keys;
        } else {
            auto* right = static_cast<InternalNode*>(parent->children[idx + 1]);
            inode->keys[inode->num_keys] = parent->keys[idx];
            for (int i = 0; i < right->num_keys; ++i) {
                inode->keys[inode->num_keys + 1 + i] = right->keys[i];
                inode->children[inode->num_keys + 1 + i] = right->children[i];
            }
            inode->children[inode->num_keys + 1 + right->num_keys] = right->children[right->num_keys];
            inode->num_keys += 1 + right->num_keys;
            right->num_keys = -1;
            delete right;

            for (int i = idx; i < parent->num_keys - 1; ++i) {
                parent->keys[i] = parent->keys[i + 1];
                parent->children[i + 1] = parent->children[i + 2];
            }
            --parent->num_keys;
        }
    }

    Node* root_ = nullptr;
    size_t size_ = 0;
};

} // namespace improved
