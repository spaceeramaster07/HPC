# Hornet: Modernized CUDA Implementation & Paper Reproduction

A modernized, high-performance port and reproduction suite of **Hornet** (*Busato et al., IEEE HPEC 2018*) for modern GPUs (NVIDIA Ada Lovelace, SM 8.9) with CUDA 13.x, GCC 12, and C++17 under WSL2 Linux.

---

## 1. Key Updates & Modernization

1. **Toolchain Modernization:**
   - Upgraded build scripts and source headers for **CUDA 13.3**, **GCC 12.4.0**, and **C++17**.
   - Added native compute capability **`sm_89`** (RTX 4060 Laptop GPU, 8 GB GDDR6).
   - Modernized CCCL / Thrust / CUB device primitives for modern warp shuffle and memory fences.

2. **Crucial Bug Fixes:**
   - **RNG Edge Generation (`RandomGraphData.cuh`):** Fixed an upstream bug where device `thrust::transform` used identical sequence indices for `src` and `dst`, causing 100% generated batch edges to be self-loops `(u, u)`. Offset destination sequence to generate true uniform random edges `(u, v)`.
   - **Verification Acceleration (`HornetPaperBenchmark.cu`):** Replaced slow host sorting ($O(E \log E)$ taking 25s per trial on 182M edges) with device radix sorting and binary search verification, reducing correctness check latency to $<0.35$ s while preserving bit-exact verification.

3. **Instrumentation & Profiling:**
   - Added [`HornetProfiler.hpp`](file:///home/spaceeramaster/HPCproj/hornet/hornet/include/Util/HornetProfiler.hpp) with **NVTX3 ranges** and sub-millisecond host/device timers around all core sub-phases (preprocess, realloc list, host alloc, H2D copy, adjacency move, host free, edge append). Verified negligible overhead ($<0.4$ ms).

4. **Unified Paper Benchmark Program:**
   - Implemented `hornet_paper_benchmark` supporting `bench_insert`, `bench_delete`, `bench_init`, `bench_overhead`, and memory profiling.

---

## 2. Experimental Results Summary

Evaluated on 5 SuiteSparse datasets (`dblp-2010`, `in-2004`, `soc-LiveJournal1`, `cage15`, `kron_g500-logn21`) with 3 warmups + 10 timed trials per configuration:

- **Batch Insertion Rate (Fig. 5b):**
  - Replicated paper scaling curve across batch sizes $1 \dots 10^7$.
  - Reached **194.3M updates/s** on `cage15` ($B=10^7$) and **353.0M updates/s** on `dblp-2010` ($B=10^7$), matching and exceeding the paper's 200M upd/s baseline.
- **Dynamic Deletions (Fig. 5c):**
  - Scaled up to **115.8M upd/s** (`cage15`) and **127.8M upd/s** (`in-2004`) for batch size $10^6$.
- **Memory Utilization (Fig. 4):**
  - **Block-level:** $70.9\%\text{--}75.0\%$ across all graphs, matching theoretical power-of-2 rounding.
  - **BlockArray-level:** $S_{\text{BA}} = 2^{16}$ yields $91\%\text{--}99.9\%$ utilization; $2^{22}$ causes severe fragmentation on small graphs ($5.7\%$ on `dblp-2010`), matching paper trends.
  - **VRAM Footprint:** $5.68\text{--}7.82$ bytes/edge on large graphs, more compact than COO (8.00 bytes/edge).
- **Graph Traversal (Fig. 7 & 8):**
  - BFS2 achieved **10,525 MTEPS (10.5 GTEPS)** on `kron_g500-logn21`.
- **Amdahl's Law & Bottleneck Analysis (Task 7 & 8):**
  - Nsys profiler revealed the GPU is idle **63.8% of the time** at $B=10^6$ waiting for single-threaded CPU host B+Tree block allocations (`insert_host_alloc` + `insert_host_free`, $36\%\text{--}46\%$ of runtime).
  - Eliminating host serial provides an Amdahl upper bound speedup of **1.85x**, whereas an infinitely fast GPU is capped at **2.17x**.

---

## 3. Quick Start

### Build Benchmarks
```bash
cd hornet/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16 hornet_paper_benchmark

cd ../hornetsnest/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16 bfs2 spmv
```

### Run End-to-End Reproduction Suite
From workspace root:
```bash
bash results/run_all.sh
```
All outputs, CSV logs, publication plots, and technical report are generated in [`results/`](file:///home/spaceeramaster/HPCproj/results/).
