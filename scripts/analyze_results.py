#!/home/spaceeramaster/HPCproj/.venv/bin/python3
"""
Analysis script for Hornet Reproduction
Computes all statistical metrics, Amdahl's law upper bounds, and phase shares.
Outputs Markdown tables directly usable in report.md
"""

import os
import sys
import csv
import re
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("/home/spaceeramaster/HPCproj/results")
RAW_CSV = RESULTS_DIR / "raw.csv"
MEM_FILE = RESULTS_DIR / "memory_results.txt"
OVERHEAD_FILE = RESULTS_DIR / "overhead_results.txt"
BFS_FILE = RESULTS_DIR / "bfs_spmv_results.txt"

def analyze_insert_updates():
    if not RAW_CSV.exists():
        return {}

    data = {} # (graph, batch_size) -> list of total_ms
    phase_data = {} # (graph, batch_size, phase) -> list of ms

    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["flags"] == "--graph-dedup":
                g = row["graph"]
                b = int(row["batch_size"])
                p = row["phase"]
                t = float(row["time_ms"])
                if p == "insert_total" and b > 0:
                    data.setdefault((g, b), []).append(t)
                elif b > 0:
                    phase_data.setdefault((g, b, p), []).append(t)

    stats = {}
    for (g, b), times in sorted(data.items()):
        arr = np.array(times)
        med = np.median(arr)
        min_t = np.min(arr)
        max_t = np.max(arr)
        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        iqr = q75 - q25

        # Rates (updates/sec)
        rate_med = b / (med / 1000.0)
        rate_max = b / (min_t / 1000.0) # fastest time = max rate
        rate_min = b / (max_t / 1000.0)

        stats[(g, b)] = {
            "n": len(arr),
            "med_ms": med,
            "min_ms": min_t,
            "max_ms": max_t,
            "iqr_ms": iqr,
            "rate_med": rate_med,
            "rate_min": rate_min,
            "rate_max": rate_max
        }

    return stats, phase_data

def analyze_delete_updates():
    if not RAW_CSV.exists():
        return {}

    data = {} # (graph, batch_size) -> list of total_ms

    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("flags") == "--delete":
                g = row["graph"]
                b = int(row["batch_size"])
                p = row["phase"]
                t = float(row["time_ms"])
                if p == "delete_total" and b > 0:
                    data.setdefault((g, b), []).append(t)

    stats = {}
    for (g, b), times in sorted(data.items()):
        arr = np.array(times)
        med = np.median(arr)
        min_t = np.min(arr)
        max_t = np.max(arr)
        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        iqr = q75 - q25

        rate_med = b / (med / 1000.0)
        rate_max = b / (min_t / 1000.0)
        rate_min = b / (max_t / 1000.0)

        stats[(g, b)] = {
            "n": len(arr),
            "med_ms": med,
            "min_ms": min_t,
            "max_ms": max_t,
            "iqr_ms": iqr,
            "rate_med": rate_med,
            "rate_min": rate_min,
            "rate_max": rate_max
        }

    return stats

def analyze_dedup_comparison():
    if not RAW_CSV.exists():
        return {}
    data = {} # (graph, batch_size, flags) -> list of ms
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flags = row.get("flags", "")
            if flags in ["--graph-dedup", "--no-dedup"]:
                g = row["graph"]
                b = int(row["batch_size"])
                p = row["phase"]
                t = float(row["time_ms"])
                if p == "insert_total" and b in [10000, 1000000]:
                    data.setdefault((g, b, flags), []).append(t)
    res = {}
    for (g, b) in set((k[0], k[1]) for k in data.keys()):
        t_dedup = np.median(data.get((g, b, "--graph-dedup"), [0.0]))
        t_nodedup = np.median(data.get((g, b, "--no-dedup"), [0.0]))
        if t_dedup > 0 and t_nodedup > 0:
            speedup = t_dedup / t_nodedup
            res[(g, b)] = {
                "t_dedup": t_dedup,
                "t_nodedup": t_nodedup,
                "speedup": speedup,
                "rate_dedup": b / (t_dedup / 1000.0),
                "rate_nodedup": b / (t_nodedup / 1000.0)
            }
    return res


def analyze_amdahl_and_hotspots(phase_data):
    # For batch sizes 10^4 and 10^6 across graphs, compute phase shares and Amdahl upper bounds
    subphases = [
        "insert_preprocess",
        "insert_build_realloc",
        "insert_host_alloc",
        "insert_h2d_copy",
        "insert_adj_move",
        "insert_host_free",
        "insert_append"
    ]

    results = {}
    for (g, b, p), times in phase_data.items():
        results.setdefault((g, b), {})[p] = np.median(times)

    amdahl_table = []
    hotspots_table = []

    for (g, b), p_dict in sorted(results.items()):
        if b in [10000, 1000000]:
            total_sub = sum(p_dict.get(p, 0.0) for p in subphases)
            if total_sub == 0:
                continue

            host_serial = p_dict.get("insert_host_alloc", 0.0) + p_dict.get("insert_host_free", 0.0)
            p_host = host_serial / total_sub
            # Amdahl upper bound if host serial eliminated: S_max = 1 / (1 - P_host)
            s_eliminate_host = 1.0 / (1.0 - p_host) if p_host < 1.0 else 1.0
            # Amdahl upper bound if GPU parts infinitely accelerated: S_max = 1 / P_host
            s_gpu_inf = 1.0 / p_host if p_host > 0.0 else 999.0

            amdahl_table.append({
                "graph": g,
                "batch_size": b,
                "total_ms": total_sub,
                "host_serial_ms": host_serial,
                "host_share_pct": p_host * 100.0,
                "amdahl_eliminate_host_speedup": s_eliminate_host,
                "amdahl_gpu_inf_speedup": s_gpu_inf
            })

            # Ranked phases
            ranked = sorted([(p, p_dict.get(p, 0.0), p_dict.get(p, 0.0) / total_sub * 100.0)
                             for p in subphases], key=lambda x: x[1], reverse=True)
            hotspots_table.append({
                "graph": g,
                "batch_size": b,
                "ranking": ranked
            })

    return amdahl_table, hotspots_table

def analyze_memory():
    if not MEM_FILE.exists():
        return []
    records = []
    with open(MEM_FILE, "r") as f:
        for line in f:
            if "[MEMORY_SUMMARY]" in line and "BA_EXP" not in line:
                m = re.search(r"Graph=(\S+)\s+\|V\|=(\d+)\s+\|E\|=(\d+).*block_util=([\d\.]+).*ba_util=([\d\.]+).*overall_util=([\d\.]+).*ba_count=(\d+).*vram_used_bytes=(\d+).*hornet_bpe=([\d\.]+).*csr_bpe=([\d\.]+).*coo_bpe=([\d\.]+)", line)
                if m:
                    records.append({
                        "graph": m.group(1),
                        "nV": int(m.group(2)),
                        "nE": int(m.group(3)),
                        "block_util": float(m.group(4)) * 100,
                        "ba_util": float(m.group(5)) * 100,
                        "overall_util": float(m.group(6)) * 100,
                        "ba_count": int(m.group(7)),
                        "vram_mb": int(m.group(8)) / (1024 * 1024),
                        "hornet_bpe": float(m.group(9)),
                        "csr_bpe": float(m.group(10)),
                        "coo_bpe": float(m.group(11))
                    })
    return records

def analyze_init():
    if not RAW_CSV.exists():
        return {}
    init_data = {}
    with open(RAW_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["flags"] == "--init":
                g = row["graph"]
                p = row["phase"]
                t = float(row["time_ms"])
                init_data.setdefault(g, {}).setdefault(p, []).append(t)

    res = {}
    for g, p_dict in init_data.items():
        tot = np.median(p_dict.get("init_total", [0.0]))
        vloop = np.median(p_dict.get("init_vertex_loop", [0.0]))
        dcopy = np.median(p_dict.get("init_device_copy_loop", [0.0]))
        res[g] = {
            "total_ms": tot,
            "vloop_ms": vloop,
            "vloop_pct": (vloop / tot * 100) if tot > 0 else 0,
            "dcopy_ms": dcopy,
            "dcopy_pct": (dcopy / tot * 100) if tot > 0 else 0
        }
    return res

def analyze_overhead():
    if not OVERHEAD_FILE.exists():
        return []
    records = []
    with open(OVERHEAD_FILE, "r") as f:
        for line in f:
            if "[OVERHEAD_SUMMARY]" in line:
                m = re.search(r"Graph=(\S+)\s+BatchSize=(\d+)\s+Trials=(\d+)\s+Uninstrumented_ms=([\d\.]+)\s+Instrumented_ms=([\d\.]+)\s+Delta_ms=([-\d\.]+)\s+Overhead_pct=([-\d\.]+)%", line)
                if m:
                    records.append({
                        "graph": m.group(1),
                        "batch_size": int(m.group(2)),
                        "uninst_ms": float(m.group(4)),
                        "inst_ms": float(m.group(5)),
                        "delta_ms": float(m.group(6)),
                        "overhead_pct": float(m.group(7))
                    })
    return records

def print_markdown_tables():
    print("=== TASK 3: INSTRUMENTATION OVERHEAD ===")
    overhead_recs = analyze_overhead()
    print("| Graph | Batch Size | Uninstrumented (ms) | Instrumented (ms) | Overhead Delta (ms) | Overhead (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in overhead_recs:
        print(f"| {r['graph']} | {r['batch_size']:,} | {r['uninst_ms']:.4f} | {r['inst_ms']:.4f} | {r['delta_ms']:+.4f} | {r['overhead_pct']:+.2f}% |")

    print("\n=== TASK 5: MEMORY UTILIZATION ===")
    mem_recs = analyze_memory()
    print("| Graph | |V| | |E| | Block Util (%) | BlockArray Util (%) | Total Hornet VRAM (MB) | Hornet Bytes/Edge | CSR Bytes/Edge | COO Bytes/Edge |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in mem_recs:
        print(f"| {r['graph']} | {r['nV']:,} | {r['nE']:,} | {r['block_util']:.1f}% | {r['ba_util']:.1f}% | {r['vram_mb']:.1f} MB | {r['hornet_bpe']:.2f} | {r['csr_bpe']:.2f} | {r['coo_bpe']:.2f} |")

    print("\n=== TASK 6: INITIALIZATION TIME FROM CSR ===")
    init_res = analyze_init()
    print("| Graph | Total Init (ms) | Vertex Loop (Host) (ms) | Vertex Loop (%) | BlockArray D2H Copy (ms) | Copy (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for g, d in init_res.items():
        print(f"| {g} | {d['total_ms']:.2f} | {d['vloop_ms']:.2f} | {d['vloop_pct']:.1f}% | {d['dcopy_ms']:.2f} | {d['dcopy_pct']:.1f}% |")

    stats, phase_data = analyze_insert_updates()
    print("\n=== TASK 4: DYNAMIC INSERT UPDATE RATES (MEDIAN, MIN, MAX, IQR) ===")
    print("| Graph | Batch Size | Median Time (ms) | IQR (ms) | Min (ms) | Max (ms) | Median Rate (updates/s) | Peak Rate (updates/s) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for (g, b), s in sorted(stats.items()):
        print(f"| {g} | {b:,} | {s['med_ms']:.4f} | {s['iqr_ms']:.4f} | {s['min_ms']:.4f} | {s['max_ms']:.4f} | {s['rate_med']:,.0f} | {s['rate_max']:,.0f} |")

    amdahl_table, hotspots_table = analyze_amdahl_and_hotspots(phase_data)
    print("\n=== TASK 8: AMDAHL'S LAW UPPER BOUNDS ON HOST-SERIAL PHASES ===")
    print("| Graph | Batch Size | Measured Total (ms) | Host Serial (ms) | Host Share (%) | Amdahl Upper Bound (Eliminate Host) | Amdahl Upper Bound (GPU Inf Fast) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for a in amdahl_table:
        print(f"| {a['graph']} | {a['batch_size']:,} | {a['total_ms']:.4f} | {a['host_serial_ms']:.4f} | {a['host_share_pct']:.1f}% | {a['amdahl_eliminate_host_speedup']:.2f}x | {a['amdahl_gpu_inf_speedup']:.2f}x |")

    del_stats = analyze_delete_updates()
    print("\n=== TASK 4: DYNAMIC DELETE UPDATE RATES (MEDIAN, MIN, MAX, IQR) ===")
    print("| Graph | Batch Size | Median Time (ms) | IQR (ms) | Min (ms) | Max (ms) | Median Rate (updates/s) | Peak Rate (updates/s) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for (g, b), s in sorted(del_stats.items()):
        print(f"| {g} | {b:,} | {s['med_ms']:.4f} | {s['iqr_ms']:.4f} | {s['min_ms']:.4f} | {s['max_ms']:.4f} | {s['rate_med']:,.0f} | {s['rate_max']:,.0f} |")

    dedup_comp = analyze_dedup_comparison()
    print("\n=== TASK 4: DEDUPLICATION IMPACT (WITH VS WITHOUT DEDUP) ===")
    print("| Graph | Batch Size | Dedup ON Time (ms) | Dedup OFF Time (ms) | Dedup ON Rate (upd/s) | Dedup OFF Rate (upd/s) | Speedup Without Dedup |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for (g, b), d in sorted(dedup_comp.items()):
        print(f"| {g} | {b:,} | {d['t_dedup']:.4f} | {d['t_nodedup']:.4f} | {d['rate_dedup']:,.0f} | {d['rate_nodedup']:,.0f} | {d['speedup']:.2f}x |")

    print("\n=== BFS MTEPS AND SPMV TRAVERSAL BENCHMARKS ===")
    if BFS_FILE.exists():
        bfs_data = {}
        with open(BFS_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    algo, g, ms = parts[0], parts[1], float(parts[2])
                    bfs_data.setdefault(g, {})[algo] = ms
        print("| Graph | BFS2 Time (ms) | SpMV Time (ms) | BFS MTEPS (approx |E|/t) |")
        print("| :--- | :---: | :---: | :---: |")
        # Graph edge counts
        edges = {
            "dblp-2010": 1615400,
            "in-2004": 16917053,
            "soc-LiveJournal1": 68993773,
            "cage15": 99199551,
            "kron_g500-logn21": 182082942
        }
        for g, algos in sorted(bfs_data.items()):
            t_bfs = algos.get("BFS2", 0.0)
            t_spmv = algos.get("SPMV", 0.0)
            e_cnt = edges.get(g, 0)
            mteps = (e_cnt / (t_bfs * 1000.0)) if t_bfs > 0 else 0.0
            print(f"| {g} | {t_bfs:.2f} ms | {t_spmv:.2f} ms | {mteps:.1f} MTEPS |")

    print("\n=== TASK 7: PHASE HOTSPOT RANKINGS BY MEASURED SHARE ===")
    for h in hotspots_table:
        print(f"**{h['graph']} (Batch {h['batch_size']:,})**:")
        for rank, (p, p_ms, p_pct) in enumerate(h['ranking'], 1):
            print(f"  {rank}. `{p}`: {p_ms:.4f} ms ({p_pct:.1f}%)")

if __name__ == "__main__":
    print_markdown_tables()


