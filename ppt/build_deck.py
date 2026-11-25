#!/home/spaceeramaster/HPCproj_backup/.venv/bin/python3
"""
Minimalist, unboxed 7-slide PowerPoint generator for branch `ppt`.
Uses a formal 2-color palette (Oxford Navy + Steel Blue) on pure white,
zero footer, zero bordered boxes, and concise talk-over bullets covering:
  Slide 1: Title Slide
  Slide 2: 1. Introduction & Objectives
  Slide 3: 2. Individual Contributions (Brief hotspot ownership)
  Slide 4: 3. Experimental Setup & Hotspot-Profiling Methodology
  Slide 5: 4. Performance Results (Part 1: Insert/Delete Scaling & Analytics)
  Slide 6: 4. Performance Results (Part 2: CSR Init & Normalized Phase Chart)
  Slide 7: 5. Limitations, Future Scope, and Key Observations
"""

from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PPT_DIR / "plots"
OUT_PPTX = PPT_DIR / "hornet_evaluation.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
TOTAL_SLIDES = 7

# Formal 2-color academic theme: Oxford Navy + Steel Blue (plus neutral charcoal/slate/white)
PRIMARY = RGBColor(27, 54, 93)      # #1B365D Oxford Navy
SECONDARY = RGBColor(59, 105, 152)  # #3B6998 Steel Blue
CHARCOAL = RGBColor(31, 41, 55)     # #1F2937 Neutral body text
MUTED = RGBColor(75, 85, 99)        # #4B5563 Neutral secondary text
RULE = RGBColor(203, 213, 225)      # #CBD5E1 Thin divider
ROW_ALT = RGBColor(243, 246, 250)   # Subtle table row tint
WHITE = RGBColor(255, 255, 255)

FONT_NAME = "Calibri"


def add_header(slide, slide_num: int, title: str, subtitle: str) -> None:
    """Clean minimalist top header with thin accent rule and no footer."""
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE

    # Clean top band in Primary Navy
    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_W, Inches(0.88))
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = PRIMARY
    hdr.line.fill.background()

    # Thin Secondary accent stripe
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.88), SLIDE_W, Inches(0.04))
    bar.fill.solid()
    bar.fill.fore_color.rgb = SECONDARY
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.08), Inches(11.4), Inches(0.76))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    r0 = p0.add_run()
    r0.text = title
    r0.font.name = FONT_NAME
    r0.font.size = Pt(22)
    r0.font.bold = True
    r0.font.color.rgb = WHITE

    p1 = tf.add_paragraph()
    r1 = p1.add_run()
    r1.text = subtitle
    r1.font.name = FONT_NAME
    r1.font.size = Pt(11.5)
    r1.font.color.rgb = RGBColor(214, 228, 240)

    # Clean unboxed slide number in top-right header
    ntb = slide.shapes.add_textbox(Inches(11.95), Inches(0.24), Inches(0.85), Inches(0.40))
    ntf = ntb.text_frame
    ntf.vertical_anchor = MSO_ANCHOR.MIDDLE
    np0 = ntf.paragraphs[0]
    np0.alignment = PP_ALIGN.RIGHT
    nr = np0.add_run()
    nr.text = f"{slide_num} / {TOTAL_SLIDES}"
    nr.font.name = FONT_NAME
    nr.font.size = Pt(12)
    nr.font.bold = True
    nr.font.color.rgb = RGBColor(214, 228, 240)


def add_section(slide, x: float, y: float, w: float, h: float, title: str, accent: RGBColor,
                items: list, role_line: str = "", title_size: float = 15.5,
                body_size: float = 12.5, space_after: float = 8) -> None:
    """
    Unboxed minimalist content column: crisp heading, thin horizontal rule,
    and concise bold-lead bullets on pure white canvas.
    """
    # Thin horizontal underline under heading
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.36), Inches(w), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = accent
    line.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    hp = tf.paragraphs[0]
    hp.space_after = Pt(10)
    hr = hp.add_run()
    hr.text = title
    hr.font.name = FONT_NAME
    hr.font.size = Pt(title_size)
    hr.font.bold = True
    hr.font.color.rgb = accent

    if role_line:
        rp = tf.add_paragraph()
        rp.space_after = Pt(space_after)
        rr = rp.add_run()
        rr.text = role_line
        rr.font.name = FONT_NAME
        rr.font.size = Pt(body_size)
        rr.font.italic = True
        rr.font.bold = True
        rr.font.color.rgb = SECONDARY

    for lead, body in items:
        p = tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.18
        if lead:
            rl = p.add_run()
            rl.text = f"•  {lead}: "
            rl.font.name = FONT_NAME
            rl.font.size = Pt(body_size)
            rl.font.bold = True
            rl.font.color.rgb = CHARCOAL
            rb = p.add_run()
            rb.text = body
            rb.font.name = FONT_NAME
            rb.font.size = Pt(body_size)
            rb.font.color.rgb = MUTED
        else:
            rb = p.add_run()
            rb.text = f"•  {body}"
            rb.font.name = FONT_NAME
            rb.font.size = Pt(body_size)
            rb.font.color.rgb = CHARCOAL


def add_fitted_image(slide, img_path: Path, box_x: float, box_y: float, box_w: float, box_h: float) -> None:
    """Place PNG centered inside target box while preserving exact aspect ratio."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    aspect = iw / ih
    target_aspect = box_w / box_h
    if aspect >= target_aspect:
        w = box_w
        h = box_w / aspect
        x = box_x
        y = box_y + (box_h - h) / 2.0
    else:
        h = box_h
        w = box_h * aspect
        x = box_x + (box_w - w) / 2.0
        y = box_y
    slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def set_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def build_slide_1_title(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PRIMARY

    # Thin vertical rule on the left edge
    left_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), SLIDE_H)
    left_bar.fill.solid()
    left_bar.fill.fore_color.rgb = SECONDARY
    left_bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(1.05), Inches(1.55), Inches(11.2), Inches(3.2))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = 0

    kicker = tf.paragraphs[0]
    kicker.space_after = Pt(14)
    kr = kicker.add_run()
    kr.text = "HPC PROJECT EVALUATION"
    kr.font.name = FONT_NAME
    kr.font.size = Pt(13)
    kr.font.bold = True
    kr.font.color.rgb = RGBColor(166, 198, 228)

    tp = tf.add_paragraph()
    tp.space_after = Pt(16)
    tr = tp.add_run()
    tr.text = "Accelerating Dynamic Sparse Graphs\non GPUs (Hornet)"
    tr.font.name = FONT_NAME
    tr.font.size = Pt(42)
    tr.font.bold = True
    tr.font.color.rgb = WHITE

    sp = tf.add_paragraph()
    sr = sp.add_run()
    sr.text = "Paper-Faithful Baseline Reproduction & GPU Data-Structure Optimization"
    sp.line_spacing = 1.2
    sr.font.name = FONT_NAME
    sr.font.size = Pt(19)
    sr.font.color.rgb = RGBColor(214, 228, 240)

    # Clean horizontal divider above presenters
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.05), Inches(4.95), Inches(11.2), Inches(0.025))
    rule.fill.solid()
    rule.fill.fore_color.rgb = SECONDARY
    rule.line.fill.background()

    mtb = slide.shapes.add_textbox(Inches(1.05), Inches(5.22), Inches(11.2), Inches(1.4))
    mtf = mtb.text_frame
    mtf.word_wrap = True
    mtf.margin_left = mtf.margin_top = 0

    p_names = mtf.paragraphs[0]
    p_names.space_after = Pt(10)
    rn = p_names.add_run()
    rn.text = "Rahul (Spaceeramaster)   •   Mahanth (mahanth-naik)   •   Syed Shams Alam Bukhari"
    rn.font.name = FONT_NAME
    rn.font.size = Pt(20)
    rn.font.bold = True
    rn.font.color.rgb = WHITE

    p_meta = mtf.add_paragraph()
    rm = p_meta.add_run()
    rm.text = "Reference: Busato et al. (IEEE HPEC 2018)   |   RTX 4060 Laptop (CUDA 13.3)   |   SuiteSparse Graphs"
    rm.font.name = FONT_NAME
    rm.font.size = Pt(13)
    rm.font.color.rgb = RGBColor(166, 198, 228)

    set_notes(slide, "Title slide introducing Hornet GPU graph optimization and all three presenters.")


def build_slide_2_intro(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 2,
        "Introduction, Objectives & Profiled Hotspots",
        "1. Hornet Overview, Core Goals & Bottlenecks Found in Baseline",
    )

    add_section(
        slide, 0.55, 1.20, 5.55, 2.25,
        "What Is Hornet?", PRIMARY,
        [
            ("Dynamic vs. Static CSR", "Static CSR requires full rebuilds per batch; Hornet updates variable-size neighbor blocks in place."),
            ("Two-Tier Architecture", "Flat vertex degree/pointer arrays backed by power-of-2 memory slabs, OR BitTrees, and host B+Trees."),
        ],
        title_size=15.5, body_size=12.3, space_after=8,
    )

    add_section(
        slide, 0.55, 3.60, 5.55, 3.45,
        "Project Objectives", SECONDARY,
        [
            ("Paper-Faithful Baseline", "Reproduce 2018 Hornet topology operations reliably within an 8 GB VRAM budget."),
            ("Strict Isolation", "Optimize only internal data structures while keeping graph algorithm kernels identical across Baseline and Improved."),
            ("Verified Speedup", "Eliminate CPU/API stalls while preserving Static CSR traversal parity across large graphs."),
        ],
        title_size=15.5, body_size=12.3, space_after=9,
    )

    add_section(
        slide, 6.45, 1.20, 6.33, 5.95,
        "Hotspots Identified in Baseline", PRIMARY,
        [
            ("1. CPU-Serial BitTree Allocation (#1)", "Host slab tracking & D2H pointer ping-pong stall batch updates."),
            ("2. Sync CUDA Alloc/Free & Staging", "Synchronous cudaMalloc/cudaFree and zeroed slabs stall driver."),
            ("3. Single-Threaded CSR Loading", "Serial CPU vertex loops delay initial graph construction."),
            ("4. Unsorted Neighbor Tails", "Linear duplicate/delete scans slow updates and miss triangles."),
            ("5. Oversized Sort Keys & Filters", "Extra radix-sort passes and multi-pass filters add GPU overhead."),
            ("6. Tiny-Batch Kernel Overhead", "Multi-kernel preprocessing dominates small-batch update latency."),
        ],
        title_size=15.5, body_size=12.0, space_after=10,
    )

    set_notes(
        slide,
        "Left: Explain what Hornet is (top) and our strict data-structure-only objectives (bottom). "
        "Right: Walk through the 6 Baseline hotspots identified via profiling, setting up teammate contributions on Slide 3.",
    )


def build_slide_3_contrib(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 3,
        "Individual Contributions",
        "2. Team Ownership of Identified Hotspots",
    )

    add_section(
        slide, 0.55, 1.35, 3.80, 5.45,
        "Rahul  (Spaceeramaster)", PRIMARY,
        [
            ("", "Paper-faithful Baseline & analytics harness"),
            ("", "Nsight Systems profiling & hotspot diagnosis"),
            ("", "GPU-resident BitTree block allocator"),
            ("", "Sorted adjacency merge & multi-block reclaim"),
        ],
        role_line="Baseline, GPU Allocator & Sorted Adjacency",
        title_size=15.5, body_size=13, space_after=12,
    )

    add_section(
        slide, 4.75, 1.35, 3.80, 5.45,
        "Mahanth  (mahanth-naik)", SECONDARY,
        [
            ("", "Stream-ordered async CUDA memory pool"),
            ("", "Compact prefix host staging buffers"),
            ("", "OpenMP parallel CSR-to-Hornet loading"),
        ],
        role_line="Memory Pool & Parallel Initialization",
        title_size=15.5, body_size=13, space_after=12,
    )

    add_section(
        slide, 8.95, 1.35, 3.83, 5.45,
        "Syed Shams Alam Bukhari", PRIMARY,
        [
            ("", "Tight vertex bit-packing for radix sort"),
            ("", "Single-pass batch preprocessing consolidation"),
            ("", "Fused small-batch shared-memory kernel"),
            ("", "Delete compaction race condition fix"),
        ],
        role_line="Preprocessing & Small-Batch Fast Path",
        title_size=15.5, body_size=13, space_after=12,
    )

    set_notes(slide, "Walk briefly through each member's hotspot ownership without deep code details.")


def build_slide_4_setup(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 4,
        "Experimental Setup and Process",
        "3. End-to-End Profiling Workflow, Controlled Testbed & Benchmark Datasets",
    )

    add_section(
        slide, 0.55, 1.18, 5.65, 5.95,
        "How We Ran the Experiment", PRIMARY,
        [
            ("1. Isolate Baseline vs. Improved", "Compiled separate Baseline and Improved binaries linking the exact same unmodified algorithm kernels (BFS, SpMV, TC, K-Truss)."),
            ("2. Profile with Nsight & Phase Timers", "Tagged insert/delete phases with NVTX and captured nsys CUDA traces (dblp-2010, in-2004) to rank CPU host, API, and init stalls."),
            ("3. Single-Load Automated Suite", "Ran bench_suite once per graph across memory utilization, CSR init, batch insert/delete (1..10M), traversals, and dynamic analytics."),
            ("4. Iterative Fix-and-Profile Cycle", "Resolved each hotspot incrementally, re-profiling to catch regressions (tuning small-batch threshold at 256, fixing compaction WAR race)."),
            ("5. Fairness & Verification Protocol", "Logged 10-trial medians after 3 warmups under live nvidia-smi telemetry, verified structural accuracy offline (verify=1), and clamped deletes to min(B, |E|)."),
        ],
        title_size=15.0, body_size=11.4, space_after=8,
    )

    add_section(
        slide, 6.45, 1.18, 6.33, 1.35,
        "Testbed & Toolchain", SECONDARY,
        [
            ("Hardware", "NVIDIA RTX 4060 Laptop (sm_89, 8 GB VRAM) + AMD Ryzen 7 7840HS (16 threads)."),
            ("Software Stack", "CUDA 13.3 (nvcc -O3), GCC 12.4 (-std=c++17), OpenMP, CUB/Thrust, Nsight Systems."),
        ],
        title_size=14.5, body_size=11.2, space_after=5,
    )

    # Clean table title & thin underline (unboxed)
    tline = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.45), Inches(2.96), Inches(6.33), Inches(0.025))
    tline.fill.solid()
    tline.fill.fore_color.rgb = PRIMARY
    tline.line.fill.background()

    dt_hdr = slide.shapes.add_textbox(Inches(6.45), Inches(2.60), Inches(6.33), Inches(0.36))
    dtf = dt_hdr.text_frame
    dtf.margin_left = dtf.margin_top = 0
    dp = dtf.paragraphs[0]
    dr = dp.add_run()
    dr.text = "SuiteSparse Benchmark Datasets"
    dr.font.name = FONT_NAME
    dr.font.size = Pt(14.5)
    dr.font.bold = True
    dr.font.color.rgb = PRIMARY

    rows_data = [
        ["Graph", "|V|", "|E|", "Avg Deg", "Domain"],
        ["dblp-2010", "326K", "1.62M", "5.0", "Co-authorship"],
        ["in-2004", "1.38M", "16.92M", "12.2", "Web crawl"],
        ["soc-LiveJournal1", "4.85M", "68.99M", "14.2", "Social network"],
        ["cage15", "5.15M", "99.20M", "19.2", "DNA matrix"],
        ["kron_g500-logn21", "2.10M", "182.08M", "86.8", "Synthetic RMAT"],
    ]
    tbl_shape = slide.shapes.add_table(len(rows_data), 5, Inches(6.45), Inches(3.10), Inches(6.33), Inches(2.50))
    tbl = tbl_shape.table
    col_widths = [Inches(1.85), Inches(0.92), Inches(1.06), Inches(0.88), Inches(1.62)]
    for ci, cw in enumerate(col_widths):
        tbl.columns[ci].width = cw

    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = PRIMARY
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = ROW_ALT
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_idx in (0, 4) else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = val
            run.font.name = FONT_NAME
            run.font.size = Pt(10.5)
            run.font.bold = (r_idx == 0 or c_idx == 0)
            run.font.color.rgb = WHITE if r_idx == 0 else CHARCOAL

    add_section(
        slide, 6.45, 5.76, 6.33, 1.45,
        "Workloads Evaluated", SECONDARY,
        [
            ("Dynamic Operations", "CSR initialization, deduplicated edge insert, and clamped edge delete."),
            ("Unchanged Analytics", "Static BFS/SpMV vs. CSR, Dynamic Triangle Counting, and Maximal K-Truss."),
        ],
        title_size=14.2, body_size=11.2, space_after=5,
    )

    set_notes(
        slide,
        "Walk through the full experimental loop step-by-step: build identical algorithm baselines, trace hotspots with nsys "
        "and NVTX timers, automate benchmarks via bench_suite under nvidia-smi monitoring, re-profile after each fix, "
        "and enforce strict verify=1 + delete clamping + K-Truss VRAM safety bounds.",
    )


def build_slide_5_perf1(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 5,
        "Performance Metrics and Results  (Part 1)",
        "4. Update Throughput Scaling, Analytics Gains & Static Parity",
    )

    add_fitted_image(slide, PLOTS_DIR / "overlay_insert_delete_throughput.png", 0.45, 0.98, 12.43, 4.35)

    cols = [
        (
            "Batch Update Scaling", PRIMARY,
            [
                ("Insert Speedup", "Consistently faster across all batch sizes, reaching up to 10.7x at 1M and 880M edges/s peak."),
                ("Delete Speedup", "Up to 8.8x faster with 1.03B edges/s peak on clamped dblp-2010."),
            ],
        ),
        (
            "Unchanged Analytics", SECONDARY,
            [
                ("Triangle Counting", "Up to 5.8x faster while sorted blocks eliminated Baseline's missed-triangle error."),
                ("Maximal K-Truss", "Up to 2.4x faster (cage15) with identical pruned edge counts verified."),
            ],
        ),
        (
            "Traversal Parity & Slab Size", PRIMARY,
            [
                ("BFS & SpMV", "Matches Static CSR traversal speed with no regression from sorted blocks."),
                ("Smaller Slabs", "Keeps 5x–7x 1M insert speedups even under tighter slab capacities."),
            ],
        ),
    ]

    col_w = 3.88
    gap = 0.29
    start_x = 0.55
    for idx, (title, accent, items) in enumerate(cols):
        cx = start_x + idx * (col_w + gap)
        add_section(
            slide, cx, 5.45, col_w, 1.85,
            title, accent, items,
            title_size=13.8, body_size=11.0, space_after=6,
        )

    set_notes(
        slide,
        "Point directly to the left/right throughput curves as you speak: dashed = Baseline, solid = Improved. "
        "Highlight the jump above 10^4 edges where GPU allocation removes CPU bottlenecks, plus TC/K-Truss gains and BFS/SpMV parity.",
    )


def build_slide_6_perf2(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 6,
        "Performance Metrics and Results  (Part 2)",
        "4. CSR Initialization Latency & Normalized Phase Breakdown",
    )

    add_fitted_image(slide, PLOTS_DIR / "csr_init_time_speedup.png", 0.45, 0.98, 5.95, 4.28)
    add_fitted_image(slide, PLOTS_DIR / "in2004_phase_host_alloc.png", 6.55, 0.98, 6.33, 4.28)

    add_section(
        slide, 0.55, 5.38, 5.85, 1.90,
        "CSR Initialization Speedup (3.1x–17.2x)", PRIMARY,
        [
            ("Parallel Prep + Mempool", "OpenMP degree histograms and async CUDA pools replace serial CPU loops and synchronous copies, cutting startup across all five graphs."),
        ],
        title_size=13.8, body_size=11.4, space_after=6,
    )

    add_section(
        slide, 6.60, 5.38, 6.18, 1.90,
        "Why Host Allocation Disappears Until 10M", SECONDARY,
        [
            ("GPU Allocator + Rare Overflow", "CPU BitTree allocation vanishes completely at 10K and 1M (confirmed by zero D2H traffic in nsys); only a tiny 0.8% slab fallback appears when 10M edges overflow."),
        ],
        title_size=13.8, body_size=11.4, space_after=6,
    )

    set_notes(
        slide,
        "Point to the two charts directly above each note: Left shows actual ms saved in CSR Init; Right shows the "
        "red CPU Host Alloc+Free phase disappearing completely in Improved until the tiny 10M slab overflow sliver.",
    )


def build_slide_7_limits(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 7,
        "Limitations, Future Scope, and Key Observations",
        "5. Core Findings, Practical Constraints & Next Steps",
    )

    add_section(
        slide, 0.55, 1.28, 3.80, 5.65,
        "Key Observations", PRIMARY,
        [
            ("Host Stalls Dominated", "Moving BitTree allocation onto the GPU removed the largest batch bottleneck."),
            ("Hardware Confirmed", "Nsight traces verified zero host pointer transfers and near-zero cudaFree stalls at 1M."),
            ("Dual Benefit of Sorting", "Sorted blocks accelerated lookups and automatically fixed Baseline Triangle Counting."),
            ("No Traversal Penalty", "Dynamic improvements preserved Static CSR speeds for BFS and SpMV."),
        ],
        title_size=15.5, body_size=12.5, space_after=11,
    )

    add_section(
        slide, 4.75, 1.28, 3.80, 5.65,
        "Limitations", SECONDARY,
        [
            ("Fixed Slab Granularity", "Large default slabs under-utilize tiny graphs, whereas very small slabs increase slab count."),
            ("10M Overflow Fallback", "Very large batches exceeding existing slabs still perform a single host slab expansion."),
            ("Frequent Small Steps", "Sorting overhead slightly outweighs gains when peeling >1,000 tiny K-Truss steps (in-2004)."),
            ("Topology Scope", "Evaluation models edge connectivity without per-edge property attributes."),
        ],
        title_size=15.5, body_size=12.5, space_after=11,
    )

    add_section(
        slide, 8.95, 1.28, 3.83, 5.65,
        "Future Scope", PRIMARY,
        [
            ("Adaptive Slab Sizing", "Select slab capacity automatically from graph size and degree distribution."),
            ("Full GPU Initialization", "Move CSR degree counting and edge scatter completely onto CUDA kernels."),
            ("Device Slab Arena", "Allocate overflow slabs inside a pre-reserved GPU arena so even 10M+ batches stay host-free."),
            ("Incremental K-Truss", "Pair sorted blocks with incremental triangle updates to support denser graphs."),
        ],
        title_size=15.5, body_size=12.5, space_after=11,
    )

    set_notes(slide, "Conclude with key findings, honest limitations, and concrete engineering next steps.")


def main() -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_slide_1_title(prs)
    build_slide_2_intro(prs)
    build_slide_3_contrib(prs)
    build_slide_4_setup(prs)
    build_slide_5_perf1(prs)
    build_slide_6_perf2(prs)
    build_slide_7_limits(prs)

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"Saved {len(prs.slides)}-slide presentation: {OUT_PPTX} ({OUT_PPTX.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
