# Hornet Baseline Hotspot Analysis & Pure-DS Optimization Roadmap

**Target Architecture:** NVIDIA GeForce RTX 4060 Laptop GPU (`sm_89`, `8187.5 MiB` VRAM), AMD Ryzen 7 7840HS (16 threads), CUDA `13.3`, C++17.  
**Baseline Reference Commit:** `98fa57d` (`fixed to faithful adaptation`, merged into `analysis` via `--ff-only`).  
**Constraint:** All optimizations (`DS-1`–`DS-7`) must reside strictly in `src/improved/` (`-DHORNET_DS_NS=improved`). `src/algorithms/` must remain untouched.

---

## 1. Post-Option-B Baseline Verification Summary (`98fa57d`)

Commit `98fa57d` resolved paper-faithfulness and measurement correctness gaps in `src/baseline/` and `src/algorithms/` (`MemoryManager::ptr_maps_` $O(\log B)$ `std::map` pointer lookup, block-cooperative shared-memory `binarySearchLB`, parallel `batch_delete` with order-preserving compaction, `sum_pow2` zero-degree fix, `verify_small` decoupling from timed trials, and `MAX_KTRUSS_BATCH = 8,000,000` chunking).

### Verified Paper-Faithfulness & Stability Fixes

| Metric / Case | Pre-Option-B (`7851848` / `2bd68a4`) | Post-Option-B (`98fa57d`) | Root Fix & Impact |
| :--- | :---: | :---: | :--- |
| **`ba_util` upper bound (`memory_results.txt`)** | `1.0003` (`soc-LiveJournal1`, `BA_EXP=16`) | **`<= 0.9992`** (`cage15`: `0.9992`, `soc-LiveJournal1`: `0.9948`) | `sum_pow2` in `src/algorithms/benchmark.cu` skips `deg == 0` vertices. |
| **`kron_g500-logn21` delete (`1M` batch)** | `145.0174 ms` (`7851848`) / `151.3019 ms` (`2bd68a4`) | **`21.4092 ms`** (`46.71M upd/s`) | **`6.77x` faster** (`7.07x` vs `2bd68a4`) via `binarySearchLB` + parallel compaction + `ptr_maps_`. |
| **`kron_g500-logn21` delete (`10M` batch)** | `226.0266 ms` (`7851848`) / `231.9415 ms` (`2bd68a4`) | **`40.4676 ms`** (`247.11M upd/s`) | **`5.59x` faster** (`5.73x` vs `2bd68a4`); across `B = 100..10M`, speedup vs `7851848` is **`5.59x–21.69x`** (`0.4821–40.4676 ms`). |
| **`soc-LiveJournal1` K-Truss (`ktruss_results.txt`)** | `306,694.55 ms` (`306.69s`, peak `10.30M upd/s`) | **`40,852.43 ms`** (`40.85s`, peak `40.16M upd/s`) | **`7.51x` faster** (`1,581` steps, `80,761,284` deleted edges). |
| **`cage15` K-Truss (`ktruss_results.txt`)** | `7,633.16 ms` (`7.63s`, peak `39.96M upd/s`) | **`925.86 ms`** (`0.93s`, peak `148.77M upd/s`) | **`8.24x` faster** (`12` steps, `94,044,692` deleted edges). |
| **`in-2004` insert `1M` IQR (`analysis_summary.md`)** | `11.4778 ms` (`7851848`) / `16.6369 ms` (`2bd68a4`) | **`1.5063 ms`** (`median 27.2789 ms`) | Decoupled `verify_small` + shared-mem `binarySearchLB` eliminated trial variance. |
| **`dblp-2010` `1M` `check_cross_dups` (`nsys`)** | `990.2 us` avg (`58.5%` GPU kern, warp kernel) | **`185.8 us` avg** (`17.7%` GPU kern, `check_cross_dups_lb_kernel`) | **`5.33x` faster** via `binarySearchLB` load-balanced edge scan. |

*Other K-Truss totals on `98fa57d`:* `dblp-2010` = `200.70 ms` (`133` steps, `1,597,644` deleted edges, peak `53.77M upd/s`); `in-2004` = `22,034.85 ms` (`1,109` steps, `20,226,050` deleted edges, peak `36.24M upd/s`).

### Why `adj_move` Is Not a Standalone Optimization Target

In `src/baseline/batch_workspace.cuh`, `adj_move_balanced_kernel` already uses block-cooperative `binarySearchLB` to distribute adjacency-list copies uniformly across threads (`new_ptrs[r][local_idx] = old_ptrs[r][local_idx]`), achieving coalesced per-edge moves:
- **Measured `nsys` GPU kernel time (`adj_move_balanced_kernel`):** `138.7 us` avg (`13.2%` of GPU kernel time) on `dblp-2010` `1M`, and `199.4 us` avg (`8.6%` of GPU kernel time) on `in-2004` `1M`.
- **Measured `insert_adj_move` phase share (`raw.csv`, `B = 1M`):** `0.3876 ms` (`3.9%`) on `cage15`, `0.6178 ms` (`3.2%`) on `dblp-2010`, `0.9420 ms` (`3.5%`) on `in-2004`, `0.8042 ms` (`2.4%`) on `kron_g500-logn21`, and `0.9049 ms` (`2.7%`) on `soc-LiveJournal1`.
- **Measured `delete_adj_move` phase share (`raw.csv`, `B = 1M`):** `0.0393–0.1325 ms` (`0.2%–1.9%` of `delete_total`).
- The small host-side prefix sum + `cudaMemcpy` for `d_move_offsets` inside `insert_adj_move` / `delete_h2d_copy` is already folded into `DS-2` via GPU `cub::DeviceScan::ExclusiveSum`.

---

## 2. Verified Remaining Pure-DS Bottlenecks

### 2.1 Host Serial Block Allocation/Reclamation & Synchronization (Primary Amdahl Bottleneck)

In `src/baseline/batch_insert.cuh`, `src/baseline/batch_delete.cuh`, and `src/baseline/hornet_graph.cuh`, every block allocation (`MemoryManager::allocate_block` via `BPlusTree` + `BitTree`) and reclamation (`MemoryManager::reclaim_block` via `std::map::upper_bound` + `BitTree` + `BPlusTree`) runs sequentially on a single host thread, bracketed by synchronous D2H/H2D transfers.

**Amdahl Ceiling on `B = 1,000,000` Batch Insert (`analysis_summary.md`, Task 8):**

| Graph | Measured Total (ms) | `insert_host_alloc` (ms) | `insert_host_free` (ms) | Host Serial Share (%) | Amdahl Ceiling (Eliminate Host) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `soc-LiveJournal1` | `33.0489` | `14.0458` (`42.5%`) | `12.3243` (`37.3%`) | **`79.8%`** | **`4.95x`** |
| `dblp-2010` | `19.2640` | `7.8831` (`40.9%`) | `7.1689` (`37.2%`) | **`78.1%`** | **`4.57x`** |
| `in-2004` | `26.5899` | `11.8211` (`44.5%`) | `6.9316` (`26.1%`) | **`70.5%`** | **`3.39x`** |
| `kron_g500-logn21` | `32.9391` | `13.7927` (`41.9%`) | `8.1505` (`24.7%`) | **`66.6%`** | **`3.00x`** |
| `cage15` | `10.0347` | `1.9635` (`19.6%`) | `2.1180` (`21.1%`) | `40.7%` | `1.69x` |

**Host Serial Dominance on Batch Delete (`raw.csv`, `--delete`):**
- `dblp-2010` (`B = 1M`): `delete_host_alloc` (`2.9944 ms`) + `delete_host_free` (`4.6876 ms`) = **`7.6820 / 11.3448 ms` (`67.7%`)**.
- `dblp-2010` (`B = 10M`): `10.4315 / 14.3468 ms` (`72.7%`).
- `in-2004` (`B = 10M`): `36.7368 / 57.0031 ms` (`64.4%`); `soc-LiveJournal1` (`B = 10M`): `20.8859 / 53.9655 ms` (`38.7%`); `cage15` (`B = 10M`): `20.4764 / 67.3526 ms` (`30.4%`).

**CSR Initialization (`analysis_summary.md`, Task 6):**
- `init_vertex_loop` (serial CPU loop over `nV` calling `allocate_block` + `stage_block_data`) consumes **`71.5%–79.3%`** of total initialization:
  - `dblp-2010`: `106.16 / 148.38 ms` (`71.5%`), `init_device_copy_loop`: `40.38 ms` (`27.2%`).
  - `in-2004`: `194.02 / 263.63 ms` (`73.6%`), `init_device_copy_loop`: `66.38 ms` (`25.2%`).
  - `soc-LiveJournal1`: `406.19 / 512.52 ms` (`79.3%`), `init_device_copy_loop`: `105.49 ms` (`20.6%`).
  - `cage15`: `246.71 / 323.93 ms` (`76.2%`), `init_device_copy_loop`: `74.44 ms` (`23.0%`).
  - `kron_g500-logn21`: `382.74 / 523.33 ms` (`73.1%`), `init_device_copy_loop`: `138.65 ms` (`26.5%`).

### 2.2 CUDA Driver Alloc/Free Overhead & Zero-Filled Host Staging (`nsys` Traces)

- **CUDA API time (`cuda_api_sum` in `nsys_*_stats.txt`):**
  - `dblp-2010` (`10k`): `cudaFree` = `130.23 ms` (`65.6%`), `cudaMemcpy` = `53.01 ms` (`26.7%`), `cudaMalloc` = `12.13 ms` (`6.1%`).
  - `in-2004` (`10k`): `cudaFree` = `145.79 ms` (`58.1%`), `cudaMemcpy` = `80.36 ms` (`32.0%`), `cudaMalloc` = `21.65 ms` (`8.6%`).
  - `dblp-2010` (`1M`): `cudaFree` = `127.92 ms` (`60.8%`), `cudaMemcpy` = `57.06 ms` (`27.1%`), `cudaMalloc` = `20.23 ms` (`9.6%`).
  - `in-2004` (`1M`): `cudaFree` = `159.02 ms` (`56.5%`), `cudaMemcpy` = `87.67 ms` (`31.1%`), `cudaMalloc` = `29.35 ms` (`10.4%`).
- **Root Causes in `src/baseline/`:**
  1. Every `BlockArray` constructor/destructor (`src/baseline/block_array.cuh`) calls synchronous `cudaMalloc` / `cudaFree` (`2^23 * 4B = 32 MiB` slabs at `BA_EXP=23`).
  2. `BlockArray::enable_host_staging()` allocates zero-initialized pageable `std::vector<vid_t> h_staging_` (`resize(total_edges_, 0)`), and `flush_staging_to_device()` performs pageable H2D `cudaMemcpy` of the entire slab (`612–997 MB` total H2D across the 2 profiled `HornetGraph` instances in `nsys`), even when `ba_util` is `2.8%–19.7%` (`dblp-2010`, `in-2004` at `BA_EXP=23`).
  3. `batch_insert.cuh` and `batch_delete.cuh` allocate temporary pageable `std::vector`s per call, launch 64-bit `DeviceRadixSort` (`end_bit = 64`), run up to 4 separate `cub::DeviceSelect::Flagged` passes for reallocation/shrink metadata, and issue ~`15` separate CUB/kernel launches with synchronous D2H scalar readbacks (`0.38–0.89 ms` launch-latency floor for $B \le 1,000$).

### 2.3 Unsorted Adjacency Lists ($O(\text{deg})$ Edge Scans) & Single-Block Hub Compaction

- **Unsorted `adj` forcing $O(\text{deg})$ scans:**
  - `append_batch_edges_lb_kernel` (`src/baseline/batch_insert.cuh`) appends new edges unsorted at `adj[old_deg + local_offset]`. Consequently:
    - `check_cross_dups_lb_kernel` must scan **all** `total_graph_work` existing edges of touched vertices and binary-search the batch: **`2.071 ms` (`44.5%` of GPU kernel time)** on `in-2004` `1M` (`1.035 ms` avg per call) and **`371.6 us` (`17.7%`)** on `dblp-2010` `1M`.
    - `delete_mark_lb_kernel` (`src/baseline/batch_delete.cuh`) must scan **all** existing edges of touched vertices:
      - `kron_g500-logn21` `B=1M`: **`13.785 ms` (`64.4%` of `delete_total`)**; `B=10M`: **`20.107 ms` (`49.7%`)**; `B=10k`: **`2.114 ms` (`49.8%`)**.
      - `soc-LiveJournal1` `B=1M`: **`4.379 ms` (`47.8%`)**; `cage15` `B=1M`: **`3.094 ms` (`45.0%`)**; `in-2004` `B=1M`: **`1.802 ms` (`27.2%`)**.
    - In `src/algorithms/triangle_counting.cuh`, `tc_batch_intersect_kernel` uses binary search on `hg.d_ptr()`, so unsorted adjacency lists after `batch_insert` degrade dynamic TC accuracy.
- **Single-block-per-hub compaction bottleneck (`delete_compact_block_kernel`):**
  - Vertices with `old_deg > 256` are compacted by **one 256-thread block per vertex** (`<<<num_large, 256>>>`) iterating serially in 1024-edge chunks. On scale-free `kron_g500-logn21`, `delete_compact` consumes **`6.645 ms` (`31.0%` of `delete_total`)** at `B=1M`, **`1.624 ms` (`38.3%`)** at `B=10k`, and **`7.757 ms` (`19.2%`)** at `B=10M` (`soc-LiveJournal1` `1M`: `2.722 ms` / `29.7%`; `cage15` `1M`: `2.094 ms` / `30.5%`).

### 2.4 8 GB VRAM Safety Boundary (`ktruss` & Allocator Headroom)

- Total GPU VRAM is `8187.5 MiB` (~8 GiB).
- `kron_g500-logn21` `ktruss` is skipped by `src/algorithms/benchmark.cu` (`[KTRUSS_SKIP]`) because the unchanged `src/algorithms/ktruss.cuh` performs static per-$k$ recounting over all `init_nE` edges and allocates 5 `init_nE`-sized buffers alongside Hornet's `1304.0 MB` (`BA_EXP=23`) graph representation.
- For the 4 active `ktruss` graphs (`dblp-2010`, `in-2004`, `soc-LiveJournal1`, `cage15`), `MAX_KTRUSS_BATCH = 8000000` chunking bounds `BatchWorkspace` memory within 8 GB VRAM.
- **Implication for `DS-1`–`DS-7`:** Any slab reuse pool or device-resident allocator (`DS-1`, `DS-7`) must enforce bounded slab retention / explicit release so `HornetGraph` + `BatchWorkspace` + `ktruss` auxiliary buffers stay safely under `8187.5 MiB`.

---

## 3. Difficulty-Sorted Pure-DS Optimization Roadmap (`DS-1`–`DS-7`)

All optimizations target `src/improved/` only (`HORNET_DS_NS=improved`), preserving exact `HornetGraph` Tier-1 ABI (`nV()`, `d_used()`, `d_ptr()`, `get_coo()`, `get_ba_*()`) and unchanged `src/algorithms/`.

| ID | Difficulty | Target Files (`src/improved/`) | Optimization & Mechanism | Bottleneck Addressed |
| :--- | :--- | :--- | :--- | :--- |
| **DS-1** | **Easy** | `block_array.cuh` | **Stream-ordered / pooled GPU slab allocator & non-zeroed pinned staging:** Replace raw `cudaMalloc`/`cudaFree` with CUDA stream-ordered `cudaMallocAsync`/`cudaFreeAsync` (configured `cudaMemPool_t` with bounded release threshold for 8 GB VRAM safety) or bounded slab reuse pool; replace zero-filled `std::vector<vid_t> h_staging_` with uninitialized `cudaMallocHost` pinned staging (copying only populated prefix/blocks or DMA-ready slabs). | `cudaFree` (`58.1%–65.6%` CUDA API time), `cudaMalloc` (`6.1%–10.4%`), `init_device_copy_loop` (`20.6%–27.2%`), zero-fill CPU overhead. |
| **DS-2** | **Easy** | `batch_workspace.cuh`, `batch_insert.cuh`, `batch_delete.cuh` | **Tight key bit-packing, 1-pass `Flagged`, GPU `move_offsets` scan, pinned metadata:** Pack `(src, dst)` using $B_v = \lceil \log_2 |V| \rceil$ bits and pass `begin_bit = 0, end_bit = 2 * B_v` ($\le 46$ bits for $|V| \le 5.15\text{M}$) to `cub::DeviceRadixSort::SortKeys`; fuse multi-array `DeviceSelect::Flagged` into index/struct 1-pass `Flagged`; compute `d_move_offsets` on GPU via `cub::DeviceScan::ExclusiveSum`; use reusable pinned host buffers (`cudaMallocHost`) in `BatchWorkspace` for realloc/shrink metadata. | `DeviceRadixSort` (`30.7%` kernel time on `dblp` `1M`), redundant `Flagged` launches, pageable H2D/D2H transfers, host `h_move_offsets` loop. |
| **DS-3** | **Easy** | `batch_insert.cuh`, `batch_delete.cuh` | **Adaptive small-batch ($B \le 1024$) fast path:** Bypass the ~15 CUB/helper kernel launches and synchronous scalar D2H readbacks via a fused `__shared__` + `__shfl_sync` warp/block kernel for sort, dedup, duplicate check, and compaction/append decisions when `batch_size <= 1024`. | Small-batch launch & sync latency floor (`0.38–0.89 ms` across $B = 1..1,000$; `ktruss` tail steps with small pruned batches). |
| **DS-4** | **Easy** | `hornet_graph.cuh`, `CMakeLists.txt` | **OpenMP parallel degree-histogram CSR init:** Partition `nV` across 16 CPU threads (Ryzen 7 7840HS) to compute per-degree-bin vertex lists / histograms in parallel, pre-size `BlockArray` slabs per `log_b` bin, assign block offsets in parallel, and copy adjacency lists into staging buffers concurrently (`-fopenmp`). | `init_vertex_loop` (`71.5%–79.3%` / `106.2–406.2 ms` of CSR init). |
| **DS-5** | **Medium** | `batch_insert.cuh`, `batch_delete.cuh` | **Sorted adjacency lists via GPU parallel merge + multi-block hub compaction:** Maintain sorted adjacency lists on insert via parallel block/warp merge of sorted `batch_dst` with sorted `adj` (`old_deg` + `batch_deg`); replace $O(\text{deg})$ edge scans in `check_cross_dups_lb_kernel` and `delete_mark_lb_kernel` with $O(\text{batch\_deg} \cdot \log \text{deg})$ binary search from batch edges into sorted `adj` (automatically fixing `tc_batch_intersect_kernel` Dynamic TC accuracy); split `old_deg > 256` hub compaction across multiple blocks using two-pass chunked count + global prefix scan + scatter. | `check_cross_dups` (`44.5%` GPU kernel time on `in-2004` `1M`), `delete_mark` (`64.4%` on `kron` `1M`), `delete_compact` (`31.0%` on `kron` `1M`), Dynamic TC accuracy. |
| **DS-6** | **Medium** | `hornet_graph.cuh` | **Full GPU-side CSR $\to$ Hornet initialization:** Upload CSR (`row_offsets`, `col_indices`) or build directly on device: compute per-vertex `log_b` bins on GPU, batch-allocate `BlockArray` slabs per bin (`1` `cudaMalloc` / slab per bin instead of per-vertex host loop), compute prefix offsets per bin on GPU, and scatter CSR adjacency lists directly into `BlockArray`s on GPU. | Remaining CSR `init_vertex_loop` (`71.5%–79.3%`) + host-to-device staging copies (`init_device_copy_loop`, `20.6%–27.2%`). |
| **DS-7** | **Hard** | `memory_manager.cuh`, `block_array.cuh`, `batch_insert.cuh`, `batch_delete.cuh` | **GPU-resident `BitTree` & device block allocator:** Move `BitTree` bitmasks and per-bin `BlockArray` descriptor tables to device memory; allocate and reclaim blocks directly inside CUDA kernels via warp-aggregated atomics (`__ffsll`, `atomicAnd`, `atomicOr`, `__activemask()`/`__ballot_sync()`), spawning new slabs only on bin overflow. Eliminates host `insert_host_alloc`/`insert_host_free` and `delete_host_alloc`/`delete_host_free` plus metadata D2H/H2D ping-pong. | Primary Amdahl bottleneck: `insert_host_alloc` + `insert_host_free` (`66.6%–79.8%`, up to `4.95x` Amdahl ceiling) and `delete_host_alloc` + `delete_host_free` (`67.7%` on `dblp-2010` `1M`). |
