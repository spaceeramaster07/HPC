#!/usr/bin/env bash
# ==============================================================================
# Master Reproduction Script for Hornet Paper Replication
# Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, sm_89, WSL2)
# Baseline Commit: 2441ccabd44a09dbf16f521dbd0bc856ab5001f8
# ==============================================================================

set -euo pipefail

WORKSPACE="/home/spaceeramaster/HPCproj"
RESULTS_DIR="${WORKSPACE}/results"
PYTHON="${WORKSPACE}/.venv/bin/python3"

echo "======================================================================"
echo "Starting Hornet Paper End-to-End Reproduction Suite"
echo "Date: $(date)"
echo "Directory: ${WORKSPACE}"
echo "======================================================================"

# 1. Environment Capture
echo "[Step 1] Verifying environment..."
nvidia-smi
nvcc --version
gcc --version
cmake --version

# 2. Build binaries if needed
echo "[Step 2] Building benchmarks..."
cd "${WORKSPACE}/hornet/hornet/build"
make -j16 hornet_paper_benchmark
cd "${WORKSPACE}/hornet/hornetsnest/build"
make -j16 bfs2 spmv

# 3. Run all experiments
echo "[Step 3] Running full benchmark suite..."
cd "${WORKSPACE}"
${PYTHON} scripts/run_paper_experiments.py

# 4. Generate plots
echo "[Step 4] Generating publication-quality figures..."
${PYTHON} scripts/generate_plots.py

# 5. Run analysis and print tables
echo "[Step 5] Computing statistical summaries and Amdahl upper bounds..."
${PYTHON} scripts/analyze_results.py

echo "======================================================================"
echo "Hornet Paper Reproduction Complete! Results stored in ${RESULTS_DIR}"
echo "======================================================================"
