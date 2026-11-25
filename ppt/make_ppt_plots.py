#!/home/spaceeramaster/HPCproj_backup/.venv/bin/python3
"""
Isolated PPT plotting script for branch `ppt`.
Generates three clean, separate presentation figures inside `ppt/plots/`
using a cohesive color palette and clear Baseline vs. Improved naming:
  1. overlay_insert_delete_throughput.png
  2. csr_init_time_speedup.png
  3. in2004_phase_host_alloc.png
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "scripts"))
import analyze_results as ar

PPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PPT_DIR / "plots"

GRAPH_ORDER = ["dblp-2010", "in-2004", "soc-LiveJournal1", "cage15", "kron_g500-logn21"]
SHORT_LABELS = {
    "dblp-2010": "dblp-2010\n(1.6M)",
    "in-2004": "in-2004\n(16.9M)",
    "soc-LiveJournal1": "soc-LJ1\n(69.0M)",
    "cage15": "cage15\n(99.2M)",
    "kron_g500-logn21": "kron-21\n(182.1M)",
}
SHORT_SINGLE = {
    "dblp-2010": "dblp-2010",
    "in-2004": "in-2004",
    "soc-LiveJournal1": "soc-LJ1",
    "cage15": "cage15",
    "kron_g500-logn21": "kron-21",
}

COLORS = {
    "dblp-2010": "#0F766E",       # Deep teal
    "in-2004": "#1D4ED8",         # Royal blue
    "soc-LiveJournal1": "#D97706",# Warm amber
    "cage15": "#BE123C",          # Crimson rose
    "kron_g500-logn21": "#6D28D9",# Indigo violet
}
MARKERS = {
    "dblp-2010": "s",
    "in-2004": "o",
    "soc-LiveJournal1": "^",
    "cage15": "v",
    "kron_g500-logn21": "D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "figure.dpi": 300,
})


def load_phase_medians(mode: str, graph: str, batches: list[int]) -> dict:
    raw_csv = WORKSPACE / "results" / mode / "raw.csv"
    agg = defaultdict(list)
    with open(raw_csv, "r") as f:
        for r in csv.DictReader(f):
            if r["graph"] == graph and r["flags"] == "--graph-dedup":
                b = int(r["batch_size"])
                if b in batches:
                    agg[(b, r["phase"])].append(float(r["time_ms"]))
    return {k: float(np.median(v)) for k, v in agg.items()}


def load_mode(mode: str) -> dict:
    d = WORKSPACE / "results" / mode
    raw_csv = d / "raw.csv"
    mem_file = d / "memory_results.txt"
    ins_stats, _ = ar.analyze_insert_updates(raw_csv)
    del_stats = ar.analyze_delete_updates(raw_csv, mem_file)
    return {
        "init": ar.analyze_init(raw_csv),
        "insert": ins_stats,
        "delete": del_stats,
    }


def make_overlay_throughput_plot(base: dict, imp: dict) -> Path:
    """Combined Insert + Delete log-log plot overlaying Baseline vs. Improved."""
    fig, (ax_ins, ax_del) = plt.subplots(1, 2, figsize=(13.2, 4.95), sharey=True)

    panels = [
        (ax_ins, base["insert"], imp["insert"], "(a) Batch Insert Throughput (With Deduplication)", "Edges Inserted per Batch", "Improved Peak: 880.2M/s\n(10.7x at 1M on dblp-2010)"),
        (ax_del, base["delete"], imp["delete"], "(b) Batch Delete Throughput (Effective Batch)", "Effective Batch Size (min(B, |E|))", "Improved Peak: 1.03B/s* (dblp 1.62M*)\n626.5M/s at 10M (in-2004)"),
    ]

    for ax, b_dict, i_dict, title, xlabel, highlight in panels:
        for g in GRAPH_ORDER:
            col = COLORS[g]
            mkr = MARKERS[g]

            b_batches = sorted(b for (gr, b) in b_dict if gr == g)
            b_rates = [b_dict[(g, b)]["rate_med"] for b in b_batches]
            i_batches = sorted(b for (gr, b) in i_dict if gr == g)
            i_rates = [i_dict[(g, b)]["rate_med"] for b in i_batches]

            ax.plot(
                b_batches, b_rates,
                linestyle="--", linewidth=1.7, alpha=0.60,
                color=col, marker=mkr, markersize=4.8,
                markerfacecolor="white", markeredgewidth=1.4,
            )
            ax.plot(
                i_batches, i_rates,
                linestyle="-", linewidth=2.5, alpha=0.96,
                color=col, marker=mkr, markersize=6.2,
                markeredgecolor="white", markeredgewidth=0.6,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.7, 2.2e7)
        ax.set_ylim(1e3, 2.8e9)
        ax.grid(True, which="major", linestyle="-", color="#CBD5E1", alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", color="#E2E8F0", alpha=0.8)
        ax.set_title(title, fontsize=12.4, fontweight="bold", color="#0F172A", pad=10)
        ax.set_xlabel(xlabel, fontsize=11.0, color="#1E293B", labelpad=6)
        ax.tick_params(labelsize=10, colors="#334155")

        ax.text(
            0.04, 0.88, highlight,
            transform=ax.transAxes,
            fontsize=9.7, fontweight="bold", color="#0F172A",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#94A3B8", alpha=0.93),
        )

    ax_ins.set_ylabel("Median Update Rate (edges / sec, log)  ↑ Higher is Better", fontsize=11.0, color="#1E293B")

    style_handles = [
        mlines.Line2D([], [], color="#1E293B", linestyle="-", linewidth=2.6, marker="o", markersize=6, label="Improved (Solid)"),
        mlines.Line2D([], [], color="#1E293B", linestyle="--", linewidth=1.8, alpha=0.65, marker="o", markerfacecolor="white", markersize=5, label="Baseline (Dashed)"),
    ]
    graph_handles = [
        mlines.Line2D([], [], color=COLORS[g], linestyle="-", linewidth=2.5, marker=MARKERS[g], markersize=6.5, label=SHORT_SINGLE[g])
        for g in GRAPH_ORDER
    ]
    fig.legend(
        handles=style_handles + graph_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=7,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=9.8,
        columnspacing=1.2,
        handlelength=2.2,
    )

    fig.text(
        0.5, -0.02,
        "*On dblp-2010 (|E| = 1.62M), 10M delete batches clamp to min(B, |E|) = 1,615,400 edges; Baseline (117.2M/s) & Improved (1.03B/s) are plotted fairly at x = 1.62M.",
        ha="center", va="top", fontsize=9.2, color="#475569", style="italic",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    out = PLOTS_DIR / "overlay_insert_delete_throughput.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_csr_init_time_plot(base: dict, imp: dict) -> Path:
    """Standalone CSR Initialization actual execution time (ms) + speedups."""
    fig, ax = plt.subplots(figsize=(5.95, 4.25))
    x = np.arange(len(GRAPH_ORDER))
    w = 0.36

    b_ms = [base["init"][g]["total_ms"] for g in GRAPH_ORDER]
    i_vloop = [imp["init"][g]["vloop_ms"] for g in GRAPH_ORDER]
    i_dcopy = [imp["init"][g]["dcopy_ms"] for g in GRAPH_ORDER]
    i_tot = [imp["init"][g]["total_ms"] for g in GRAPH_ORDER]

    ax.bar(x - w / 2, b_ms, width=w, color="#64748B", edgecolor="white", linewidth=0.9, label="Baseline (Serial CPU + Sync Copy)")
    ax.bar(x + w / 2, i_vloop, width=w, color="#0D9488", edgecolor="white", linewidth=0.9, label="Improved: OpenMP Parallel Prep")
    ax.bar(x + w / 2, i_dcopy, bottom=i_vloop, width=w, color="#2563EB", edgecolor="white", linewidth=0.9, label="Improved: CUDA Mempool + Bulk Copy")

    for idx, (bm, im) in enumerate(zip(b_ms, i_tot)):
        spd = bm / im
        ax.text(idx - w / 2, bm + 10, f"{bm:.0f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#475569")
        ax.text(idx + 0.02, im + 10, f"{im:.1f} ms\n({spd:.1f}x)", ha="left", va="bottom", fontsize=8.0, fontweight="bold", color="#0F766E")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[g] for g in GRAPH_ORDER], fontsize=8.9)
    ax.set_xlim(-0.52, len(GRAPH_ORDER) - 0.12)
    ax.set_ylim(0, max(b_ms) * 1.15)
    ax.set_ylabel("Initialization Time (ms)  ↓ Lower is Better", fontsize=10.0, color="#1E293B")
    ax.set_title("CSR-to-Hornet Initialization Time & Speedup", fontsize=11.5, fontweight="bold", color="#0F172A", pad=44)
    ax.grid(axis="y", linestyle="--", color="#CBD5E1", alpha=0.65)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7.7,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        columnspacing=0.9,
    )

    fig.tight_layout()
    out = PLOTS_DIR / "csr_init_time_speedup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_normalized_phase_plot() -> Path:
    """
    Normalized stacked phase breakdown on in-2004 (10K, 1M, 10M Insert):
    Baseline Total = 100% per batch; Improved scaled relative to Baseline Total.
    """
    batches = [10_000, 1_000_000, 10_000_000]
    batch_labels = ["10K Edges", "1M Edges", "10M Edges\n(Slab Overflow)"]
    b_med = load_phase_medians("baseline", "in-2004", batches)
    i_med = load_phase_medians("improved", "in-2004", batches)

    def extract_pcts(med: dict, b: int, base_tot: float) -> tuple[float, float, float, float, float]:
        tot = med[(b, "insert_total")]
        prep = med[(b, "insert_preprocess")]
        h2d = med[(b, "insert_h2d_copy")]
        host = med[(b, "insert_host_alloc")] + med[(b, "insert_host_free")]
        gpu_work = max(0.0, tot - prep - h2d - host)
        scale = 100.0 / base_tot
        return prep * scale, gpu_work * scale, h2d * scale, host * scale, tot

    b_stacks, i_stacks, b_tots, i_tots = [], [], [], []
    for b in batches:
        btot = b_med[(b, "insert_total")]
        b_vals = extract_pcts(b_med, b, btot)
        i_vals = extract_pcts(i_med, b, btot)
        b_stacks.append(b_vals[:4])
        i_stacks.append(i_vals[:4])
        b_tots.append(b_vals[4])
        i_tots.append(i_vals[4])

    b_arr = np.array(b_stacks)
    i_arr = np.array(i_stacks)

    fig, ax = plt.subplots(figsize=(6.45, 4.25))
    x = np.arange(len(batches))
    w = 0.32
    offset = 0.19

    phase_specs = [
        ("GPU Preprocess (Sort/Dedup)", "#1E3A8A"),
        ("GPU Plan, Realloc & Merge", "#0D9488"),
        ("Slab H2D Metadata Copy", "#F59E0B"),
        ("CPU Host Alloc + Free", "#E11D48"),
    ]

    b_bottom = np.zeros(len(batches))
    i_bottom = np.zeros(len(batches))
    for k, (plabel, pcol) in enumerate(phase_specs):
        ax.bar(
            x - offset, b_arr[:, k], bottom=b_bottom, width=w,
            color=pcol, edgecolor="white", linewidth=0.8, label=plabel,
        )
        ax.bar(
            x + offset, i_arr[:, k], bottom=i_bottom, width=w,
            color=pcol, edgecolor="white", linewidth=0.8,
        )
        b_bottom += b_arr[:, k]
        i_bottom += i_arr[:, k]

    for idx in range(len(batches)):
        bt_ms = b_tots[idx]
        bt_fmt = f"{bt_ms:.2f}" if bt_ms < 10 else f"{bt_ms:.1f}"
        ax.text(
            idx - offset, 101.8, f"100%\n({bt_fmt} ms)",
            ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#334155",
        )
        b_host_pct = b_arr[idx, 3]
        b_host_mid = b_arr[idx, 0] + b_arr[idx, 1] + b_arr[idx, 2] + b_host_pct / 2.0
        ax.text(
            idx - offset, b_host_mid, f"{b_host_pct:.0f}%",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color="white",
        )

        it_pct = i_bottom[idx]
        it_ms = i_tots[idx]
        spd = bt_ms / it_ms
        it_fmt = f"{it_ms:.2f}" if it_ms < 10 else f"{it_ms:.1f}"
        lbl_extra = "\nHost: 0%" if idx == 0 else ""
        ax.text(
            idx + 0.03, it_pct + 2.0,
            f"{it_pct:.0f}% ({spd:.1f}x)\n{it_fmt} ms{lbl_extra}",
            ha="left", va="bottom", fontsize=7.6, fontweight="bold", color="#0F766E",
        )

    ax.annotate(
        "Host Alloc: 0%",
        xy=(1 + offset + w / 2, i_bottom[1] + 0.3),
        xytext=(1 + 0.04, 46),
        ha="left", va="bottom", fontsize=7.8, fontweight="bold", color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#0D9488", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.0, connectionstyle="arc3,rad=-0.38", shrinkB=2),
    )
    ax.annotate(
        "0.8% Host (0.72 ms)\n+2.2% H2D (1 slab)",
        xy=(2 + offset + w / 2, i_bottom[2] - 0.8),
        xytext=(2 + 0.03, 52),
        ha="left", va="bottom", fontsize=7.6, fontweight="bold", color="#BE123C",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF1F2", edgecolor="#E11D48", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#E11D48", lw=1.0, connectionstyle="arc3,rad=-0.38", shrinkB=2),
    )

    for idx in range(len(batches)):
        ax.text(idx - offset, -2.5, "Base", ha="center", va="top", fontsize=7.8, fontweight="bold", color="#475569")
        ax.text(idx + offset, -2.5, "Imp", ha="center", va="top", fontsize=7.8, fontweight="bold", color="#0F766E")

    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels, fontsize=9.0, fontweight="bold", color="#0F172A")
    ax.tick_params(axis="x", pad=13)
    ax.set_xlim(-0.52, 2.95)
    ax.set_ylim(0, 120)
    ax.set_ylabel("% of Baseline Batch Time  ↓ Lower is Better", fontsize=9.8, color="#1E293B")
    ax.set_title("Normalized Insert Phases (in-2004): Host Alloc Eliminated Until 10M", fontsize=11.0, fontweight="bold", color="#0F172A", pad=44)
    ax.grid(axis="y", linestyle="--", color="#CBD5E1", alpha=0.65)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=7.7,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        columnspacing=0.9,
    )

    fig.tight_layout()
    out = PLOTS_DIR / "in2004_phase_host_alloc.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "normalized_init_and_batch_reduction.png",
        "in2004_host_alloc_example.png",
        "nsys_in2004_1m_profile.png",
    ):
        stale = PLOTS_DIR / stale_name
        if stale.exists():
            stale.unlink()

    base = load_mode("baseline")
    imp = load_mode("improved")

    for p in (
        make_overlay_throughput_plot(base, imp),
        make_csr_init_time_plot(base, imp),
        make_normalized_phase_plot(),
    ):
        print(f"Generated: {p}")


if __name__ == "__main__":
    main()
