#!/usr/bin/env python3
"""
Downloads the exact datasets from Table II of the Hornet paper:
"Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs"
(Busato et al., IEEE HPEC 2018)

Datasets sourced directly from the SuiteSparse Matrix Collection (sparse.tamu.edu):
  - dblp-2010:   0.03M vertices,  1.6M edges (Collaboration network)
  - cantilever:  0.06M vertices,  4.1M edges (FEM mesh)
  - in-2004:     1.38M vertices, 16.7M edges (Web crawl)
  - soc-LiveJournal1: 4.85M vertices, 69.0M edges (Social network)
"""

import sys
import os
import urllib.request
import tarfile
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = WORKSPACE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Table II datasets from the paper
PAPER_DATASETS = {
    "dblp-2010": {
        "group": "LAW",
        "url": "https://sparse.tamu.edu/MM/LAW/dblp-2010.tar.gz",
        "vertices": "30,000",
        "edges": "1,600,000",
        "size_mb": 15,
        "description": "Collaboration network (1.6M edges - primary paper benchmark)"
    },
    "cantilever": {
        "group": "Williams",
        "url": "https://sparse.tamu.edu/MM/Williams/cantilever.tar.gz",
        "vertices": "60,000",
        "edges": "4,100,000",
        "size_mb": 35,
        "description": "FEM mechanical mesh (4.1M edges)"
    },
    "in-2004": {
        "group": "LAW",
        "url": "https://sparse.tamu.edu/MM/LAW/in-2004.tar.gz",
        "vertices": "1,380,000",
        "edges": "16,700,000",
        "size_mb": 100,
        "description": "Web crawl network (16.7M edges)"
    },
    "soc-LiveJournal1": {
        "group": "SNAP",
        "url": "https://sparse.tamu.edu/MM/SNAP/soc-LiveJournal1.tar.gz",
        "vertices": "4,850,000",
        "edges": "69,000,000",
        "size_mb": 450,
        "description": "Large social network (69M edges - fits comfortably in 8GB VRAM)"
    }
}

def download_and_extract(dataset_key):
    info = PAPER_DATASETS[dataset_key]
    url = info["url"]
    archive_path = DATASETS_DIR / f"{dataset_key}.tar.gz"
    mtx_target = DATASETS_DIR / f"{dataset_key}.mtx"

    if mtx_target.exists():
        print(f"[✓] {dataset_key}.mtx already exists in datasets/ ({mtx_target.stat().st_size / (1024*1024):.1f} MB)")
        return str(mtx_target)

    print(f"\n[*] Downloading {dataset_key} ({info['description']})...")
    print(f"    URL: {url} (~{info['size_mb']} MB)")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r    Downloading: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        else:
            sys.stdout.write(f"\r    Downloading: {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, archive_path, progress_hook)
        print("\n    Download complete. Extracting Matrix Market file (.mtx)...")

        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".mtx"):
                    # Extract directly as <dataset_key>.mtx
                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(mtx_target, "wb") as f_out:
                            f_out.write(f_in.read())
                        break

        # Remove the archive to save disk space
        if archive_path.exists():
            archive_path.unlink()

        print(f"[✓] Successfully extracted to: {mtx_target} ({mtx_target.stat().st_size / (1024*1024):.1f} MB)")
        return str(mtx_target)

    except Exception as e:
        print(f"\n[!] Error downloading {dataset_key}: {e}")
        return None

def main():
    print("=" * 70)
    print(" HORNET PAPER DATASET DOWNLOADER (Table II from IEEE HPEC 2018)")
    print(f" Target Directory: {DATASETS_DIR}")
    print("=" * 70)

    # Allow passing specific dataset or default to dblp-2010
    requested = sys.argv[1:] if len(sys.argv) > 1 else ["dblp-2010"]

    if requested == ["all"]:
        requested = list(PAPER_DATASETS.keys())

    for key in requested:
        if key in PAPER_DATASETS:
            download_and_extract(key)
        else:
            print(f"[!] Unknown dataset: {key}. Available: {list(PAPER_DATASETS.keys())} or 'all'")

    print("\n" + "=" * 70)
    print(" Download summary complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
