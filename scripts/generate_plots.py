#!/home/spaceeramaster/HPCproj/.venv/bin/python3
"""
Plotting Script for Hornet Reproduction
Generates all publication-quality figures for report.md
"""

import os
import sys
import csv
import re
from pathlib import Path
import numpy as np

# Ensure matplotlib runs headless and uses a writable cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_config"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/home/spaceeramaster/HPCproj/results")
RAW_CSV = RESULTS_DIR / "raw.csv"
MEM_FILE = RESULTS_DIR / "memory_results.txt"
OVERHEAD_FILE = RESULTS_DIR / "overhead_results.txt"
BFS_FILE = RESULTS_DIR / "bfs_spmv_results.txt"

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

COLORS = {
    "dblp-2010": "#2ca02c",
    "in-2004": "#1f77b4",
    "soc-LiveJournal1": "#ff7f0e",
    "cage15": "#d62728",
    "kron_g500-logn21": "#9467bd"
}

MARKERS = {
    "dblp-2010": "s",
    "in-2004": "o",
    "soc-LiveJournal1": "^",
    "cage15": "v",
    "kron_g500-logn21": "D"
}

# ==============================================================================
# Plot 1: Update Rate vs Batch Size (Fig. 5b Replication)
# ==============================================================================
def plot_insert_update_rate():
    print("[Plotting] Generating Fig 5: Update Rate vs Batch Size...")
    if not RAW_CSV.exists():
        print(f"Skipping: {RAW_CSV} not found.")
        return

    # Parse raw.csv for insert_total
    data = {} # (graph, batch_size) -> list of time_ms
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] == "insert_total" and row["flags"] == "--graph-dedup":
                g = row["graph"]
                b = int(row["batch_size"])
                t = float(row["time_ms"])
                data.setdefault((g, b), []).append(t)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Order of graphs matching paper
    graph_order = ["in-2004", "soc-LiveJournal1", "cage15", "kron_g500-logn21", "dblp-2010"]

    for g in graph_order:
        batches = sorted([b for (gr, b) in data.keys() if gr == g])
        if not batches:
            continue
        rates = []
        err_low = []
        err_high = []
        for b in batches:
            times = np.array(data[(g, b)])
            rates_arr = (b / (times / 1000.0))
            med = np.median(rates_arr)
            q25 = np.percentile(rates_arr, 25)
            q75 = np.percentile(rates_arr, 75)
            rates.append(med)
            err_low.append(med - q25)
            err_high.append(q75 - med)

        c = COLORS.get(g, "black")
        m = MARKERS.get(g, "o")
        ax.errorbar(batches, rates, yerr=[err_low, err_high], label=g,
                    marker=m, color=c, linewidth=2, markersize=7, capsize=3, capthick=1)

    # Reference lines
    ax.axhline(2e8, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Paper Claim: 200M upd/s (P100)")
    ax.axhline(8e8, color="red", linestyle=":", linewidth=1.2, alpha=0.6,
               label="Paper Peak (AIM config): 800M upd/s")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (edges per update)")
    ax.set_ylabel("Update Rate (edges / second)")
    ax.set_title("Hornet Dynamic Insertion Update Rate vs. Batch Size\n(RTX 4060 Laptop GPU, sm_89, WSL2)")
    ax.set_ylim(1e3, 2e9)
    ax.set_xlim(0.8, 2e7)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    out_path = RESULTS_DIR / "fig5_insert_rate_vs_batch.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Plot 2: Stacked Phase Breakdown per Batch Size
# ==============================================================================
def plot_phase_breakdown():
    print("[Plotting] Generating Stacked Phase Breakdown...")
    if not RAW_CSV.exists():
        return

    # Aggregate phases per (graph, batch_size)
    subphases = [
        ("insert_preprocess", "Preprocess (sort/CSR) [GPU]", "#1f77b4"),
        ("insert_build_realloc", "Build Realloc List [GPU]", "#aec7e8"),
        ("insert_host_alloc", "Host Alloc Blocks [CPU-serial]", "#ff7f0e"),
        ("insert_h2d_copy", "Host-to-Device Copy [PCIe]", "#ffbb78"),
        ("insert_adj_move", "Adjacency Move Kernel [GPU]", "#2ca02c"),
        ("insert_host_free", "Host Free Blocks [CPU-serial]", "#d62728"),
        ("insert_append", "Append Edges Kernel [GPU]", "#9467bd")
    ]

    phase_times = {} # (graph, batch_size, phase) -> list of time_ms
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["flags"] == "--graph-dedup":
                g = row["graph"]
                b = int(row["batch_size"])
                p = row["phase"]
                if b > 0 and p != "insert_total":
                    t = float(row["time_ms"])
                    phase_times.setdefault((g, b, p), []).append(t)

    # Plot for soc-LiveJournal1 as primary exemplar, or multi-panel
    graphs_to_plot = ["in-2004", "soc-LiveJournal1"]
    fig, axes = plt.subplots(1, len(graphs_to_plot), figsize=(14, 5.5), sharey=True)

    for ax_idx, g in enumerate(graphs_to_plot):
        ax = axes[ax_idx]
        batches = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
        valid_batches = [b for b in batches if (g, b, "insert_preprocess") in phase_times]

        bottoms = np.zeros(len(valid_batches))
        bar_width = 0.55
        x_indices = np.arange(len(valid_batches))

        # First compute total median time of subphases
        totals = np.zeros(len(valid_batches))
        for p_key, p_lbl, p_col in subphases:
            for i, b in enumerate(valid_batches):
                totals[i] += np.median(phase_times.get((g, b, p_key), [0.0]))

        # Normalize to percentage
        for p_key, p_lbl, p_col in subphases:
            shares = np.zeros(len(valid_batches))
            for i, b in enumerate(valid_batches):
                p_med = np.median(phase_times.get((g, b, p_key), [0.0]))
                shares[i] = (p_med / totals[i] * 100.0) if totals[i] > 0 else 0.0

            ax.bar(x_indices, shares, bottom=bottoms, width=bar_width,
                   label=p_lbl if ax_idx == 0 else "", color=p_col, edgecolor="white", linewidth=0.5)
            bottoms += shares

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"$10^{int(np.log10(b))}$" for b in valid_batches])
        ax.set_xlabel("Batch Size")
        ax.set_title(f"Phase Share Breakdown: {g}")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Execution Time Share (%)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True)
    plt.tight_layout()
    out_path = RESULTS_DIR / "fig_phase_breakdown_stacked.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Plot 3: Memory Utilization Analysis (Fig. 4 Replication)
# ==============================================================================
def plot_memory_utilization():
    print("[Plotting] Generating Fig 4: Memory Utilization...")
    if not MEM_FILE.exists():
        print(f"Skipping: {MEM_FILE} not found.")
        return

    # Parse memory_results.txt
    # [MEMORY_SUMMARY] Graph=... block_util=... ba_util=... overall_util=... hornet_bpe=... csr_bpe=... coo_bpe=...
    base_data = {}
    ba_exp_data = {} # exp -> dict of graph -> ba_util

    with open(MEM_FILE, "r") as f:
        for line in f:
            if "[MEMORY_SUMMARY]" in line and "BA_EXP" not in line:
                m_g = re.search(r"Graph=(\S+)", line)
                m_bu = re.search(r"block_util=([\d\.]+)", line)
                m_bau = re.search(r"ba_util=([\d\.]+)", line)
                m_ov = re.search(r"overall_util=([\d\.]+)", line)
                m_hbpe = re.search(r"hornet_bpe=([\d\.]+)", line)
                m_csrbpe = re.search(r"csr_bpe=([\d\.]+)", line)
                m_coobpe = re.search(r"coo_bpe=([\d\.]+)", line)
                if m_g and m_bu:
                    base_data[m_g.group(1)] = {
                        "block_util": float(m_bu.group(1)),
                        "ba_util": float(m_bau.group(1)),
                        "overall_util": float(m_ov.group(1)),
                        "hornet_bpe": float(m_hbpe.group(1)),
                        "csr_bpe": float(m_csrbpe.group(1)),
                        "coo_bpe": float(m_coobpe.group(1))
                    }
            elif "[MEMORY_SUMMARY BA_EXP=" in line:
                m_exp = re.search(r"BA_EXP=(\d+)", line)
                m_g = re.search(r"Graph=(\S+)", line)
                m_bau = re.search(r"ba_util=([\d\.]+)", line)
                if m_exp and m_g and m_bau:
                    exp = int(m_exp.group(1))
                    ba_exp_data.setdefault(exp, {})[m_g.group(1)] = float(m_bau.group(1))

    if not base_data:
        print("No memory data found.")
        return

    graphs = list(base_data.keys())
    x = np.arange(len(graphs))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11))

    # Subplot 1: Block-level fragmentation (Paper Fig 4 upper)
    block_util = [base_data[g]["block_util"] * 100 for g in graphs]
    over_alloc = [100.0 - u for u in block_util]
    ax1.bar(x, block_util, label="Utilized Space", color="#2ca02c", width=0.55)
    ax1.bar(x, over_alloc, bottom=block_util, label="Over-allocated (Power-of-2 rule)",
            color="#ff7f0e", width=0.55, alpha=0.7)
    ax1.set_ylabel("Edge Utilization (%)")
    ax1.set_title("Block-Level Memory Utilization (Paper Fig. 4 Top)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(graphs)
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Subplot 2: Block-array level utilization across BA_EXP (Paper Fig 4 middle)
    exps = [16, 19, 22]
    w = 0.25
    colors_ba = ["#1f77b4", "#aec7e8", "#c7c7c7"]
    for i, exp in enumerate(exps):
        vals = [ba_exp_data.get(exp, {}).get(g, base_data[g]["ba_util"]) * 100 for g in graphs]
        ax2.bar(x + (i - 1) * w, vals, width=w, label=f"EDGES_PER_BLOCKARRAY = $2^{{{exp}}}$",
                color=colors_ba[i])
    ax2.set_ylabel("Block Utilization (%)")
    ax2.set_title("Block-Array Level Fragmentation (Paper Fig. 4 Middle)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(graphs)
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # Subplot 3: Bytes per edge: Hornet vs CSR vs COO
    w3 = 0.26
    h_bpe = [base_data[g]["hornet_bpe"] for g in graphs]
    csr_bpe = [base_data[g]["csr_bpe"] for g in graphs]
    coo_bpe = [base_data[g]["coo_bpe"] for g in graphs]
    ax3.bar(x - w3, csr_bpe, width=w3, label="CSR (Static Minimal)", color="#2ca02c")
    ax3.bar(x, h_bpe, width=w3, label="Hornet (Measured Device)", color="#1f77b4")
    ax3.bar(x + w3, coo_bpe, width=w3, label="COO (Static Edge-List)", color="#d62728")
    ax3.set_ylabel("Bytes per Edge")
    ax3.set_title("Memory Efficiency: Bytes per Edge (Hornet vs Static CSR & COO)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(graphs)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    out_path = RESULTS_DIR / "fig4_memory_utilization.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Plot 4: Initialization Time from CSR
# ==============================================================================
def plot_initialization_time():
    print("[Plotting] Generating Fig: Initialization Time...")
    if not RAW_CSV.exists():
        return

    init_data = {} # graph -> dict of phase -> list of ms
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["flags"] == "--init":
                g = row["graph"]
                p = row["phase"]
                t = float(row["time_ms"])
                init_data.setdefault(g, {}).setdefault(p, []).append(t)

    if not init_data:
        print("No init data found.")
        return

    graphs = list(init_data.keys())
    x = np.arange(len(graphs))
    w = 0.5

    vloop = [np.median(init_data[g].get("init_vertex_loop", [0.0])) for g in graphs]
    dcopy = [np.median(init_data[g].get("init_device_copy_loop", [0.0])) for g in graphs]
    total = [np.median(init_data[g].get("init_total", [0.0])) for g in graphs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, vloop, width=w, label="Phase 1: Vertex Degree Loop (Host)", color="#ff7f0e")
    ax.bar(x, dcopy, width=w, bottom=vloop, label="Phase 2: Bulk Block-Array Copy (PCIe)", color="#1f77b4")

    # Annotate total ms above bars
    for i, t_val in enumerate(total):
        ax.text(i, t_val * 1.05, f"{t_val:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Initialization Time (ms)")
    ax.set_title("Hornet Initialization Time from CSR\n(Phase Breakdown & Total)")
    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path = RESULTS_DIR / "fig_init_time.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Plot 5: Dynamic Deletion Update Rate
# ==============================================================================
def plot_delete_update_rate():
    print("[Plotting] Generating Dynamic Deletion Update Rate...")
    if not RAW_CSV.exists():
        return

    data = {}
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] == "delete_total":
                g = row["graph"]
                b = int(row["batch_size"])
                t = float(row["time_ms"])
                data.setdefault((g, b), []).append(t)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    graph_order = ["in-2004", "soc-LiveJournal1", "cage15", "kron_g500-logn21", "dblp-2010"]

    for g in graph_order:
        batches = sorted([b for (gr, b) in data.keys() if gr == g])
        if not batches:
            continue
        rates = []
        for b in batches:
            times = np.array(data[(g, b)])
            rates_arr = (b / (times / 1000.0))
            rates.append(np.median(rates_arr))

        c = COLORS.get(g, "black")
        m = MARKERS.get(g, "o")
        ax.plot(batches, rates, label=g, marker=m, color=c, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (edges deleted)")
    ax.set_ylabel("Deletion Rate (edges / second)")
    ax.set_title("Hornet Dynamic Deletion Update Rate vs. Batch Size\n(RTX 4060 Laptop GPU, sm_89, WSL2)")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    out_path = RESULTS_DIR / "fig_delete_rate_vs_batch.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    print("=== GENERATING ALL BENCHMARK PLOTS ===")
    plot_insert_update_rate()
    plot_phase_breakdown()
    plot_memory_utilization()
    plot_initialization_time()
    plot_delete_update_rate()
    print("=== ALL PLOTS GENERATED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
