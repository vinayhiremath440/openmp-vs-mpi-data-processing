#!/usr/bin/env python3
"""
Presentation Generator for Parallel Computing Mini-Project
Topic: OpenMP vs MPI for Data Processing

Generates a professional 13-slide PowerPoint presentation:
presentation/OpenMP_vs_MPI_Data_Processing.pptx
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(WORKSPACE_DIR, "graphs")
PRESENTATION_DIR = os.path.join(WORKSPACE_DIR, "presentation")
OUTPUT_PPTX = os.path.join(PRESENTATION_DIR, "OpenMP_vs_MPI_Data_Processing.pptx")

os.makedirs(PRESENTATION_DIR, exist_ok=True)

# Color Palette
NAVY = RGBColor(24, 43, 73)
TEAL = RGBColor(16, 124, 130)
DARK_GRAY = RGBColor(45, 52, 54)
LIGHT_BG = RGBColor(245, 247, 250)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(39, 174, 96)
BLUE = RGBColor(41, 128, 185)
RED = RGBColor(231, 76, 60)

def set_slide_background(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = LIGHT_BG

def add_header(slide, title_text, category_text="PARALLEL COMPUTING MINI-PROJECT | TOPIC #9"):
    # Header container banner
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY
    shape.line.color.rgb = NAVY

    # Category Text
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(8.4), Inches(0.3))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL

    # Main Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.42), Inches(8.4), Inches(0.55))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE

def create_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Background rectangle
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(5.625))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.color.rgb = NAVY

    # Accent Stripe
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(3.2))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = TEAL
    stripe.line.color.rgb = TEAL

    # Title & Subtitle Box
    tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(8.0), Inches(3.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "PARALLEL COMPUTING LAB EVALUATION"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = TEAL
    p0.space_after = Pt(10)

    p1 = tf.add_paragraph()
    p1.text = "OpenMP vs MPI for Data Processing"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "A Comparative Performance, Speedup, and Efficiency Study"
    p2.font.size = Pt(15)
    p2.font.color.rgb = RGBColor(200, 214, 229)
    p2.space_after = Pt(25)

    p3 = tf.add_paragraph()
    p3.text = "Author: Vinay Hiremath\nDomain: Parallel & Distributed Computing\nTheme: OpenMP vs MPI for Data Processing"
    p3.font.size = Pt(11)
    p3.font.color.rgb = RGBColor(164, 176, 190)

def create_bullet_slide(prs, title, points, notes=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, title)

    # Content container
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3), Inches(8.4), Inches(3.9))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = RGBColor(220, 224, 230)

    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(8.0), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True

    for i, pt in enumerate(points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = pt[0]
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.space_after = Pt(2)

        sub_p = tf.add_paragraph()
        sub_p.text = pt[1]
        sub_p.font.size = Pt(11)
        sub_p.font.color.rgb = DARK_GRAY
        sub_p.space_after = Pt(10)

def create_image_slide(prs, title, image_name, bullet_points):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, title)

    img_path = os.path.join(GRAPHS_DIR, image_name)
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.8), Inches(1.3), Inches(5.0), Inches(3.8))

    # Notes card on the right
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.0), Inches(1.3), Inches(3.2), Inches(3.8))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = RGBColor(220, 224, 230)

    tb = slide.shapes.add_textbox(Inches(6.15), Inches(1.45), Inches(2.9), Inches(3.5))
    tf = tb.text_frame
    tf.word_wrap = True

    for i, pt in enumerate(bullet_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = pt[0]
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.space_after = Pt(2)

        sub_p = tf.add_paragraph()
        sub_p.text = pt[1]
        sub_p.font.size = Pt(9.5)
        sub_p.font.color.rgb = DARK_GRAY
        sub_p.space_after = Pt(8)

def create_table_slide(prs, title, headers, rows):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, title)

    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, Inches(0.8), Inches(1.3), Inches(8.4), Inches(3.8))
    table = table_shape.table

    # Format header
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    # Format data rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else RGBColor(240, 243, 246)
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(9.5)
            p.font.color.rgb = DARK_GRAY
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

def build_presentation():
    prs = pptx.Presentation()
    # Set 16:9 widescreen dimensions
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    print("Building presentation slide deck...")

    # Slide 1: Title
    create_title_slide(prs)

    # Slide 2: Project Objectives & Motivation
    create_bullet_slide(prs, "1. Project Motivation & Research Objectives", [
        ("Core Objective", "Implement identical numerical dataset operations in OpenMP and MPI to quantitatively evaluate latency, speedup, and scalability."),
        ("Multi-Paradigm Study", "Benchmarking shared-memory thread parallelism (OpenMP) against distributed-memory message passing (MPI) with zero computational drift."),
        ("Multi-Scale Experimental Grid", "Systematic evaluation across dataset sizes (5M to 50M doubles) and worker counts (1 to 12 threads/processes)."),
        ("Architectural Analysis", "Evaluating Amdahl's Law, parallel efficiency decay, memory bus contention, and collective communication overhead."),
        ("Performance Metrics", "Measuring wall-clock time (ms), speedup factor S, parallel efficiency E (%), and processing throughput (Million items/sec).")
    ])

    # Slide 3: Problem Definition & Mathematical Formulation
    create_bullet_slide(prs, "2. Problem Definition & Mathematical Operation", [
        ("Input Dataset Characteristics", "Array of N double-precision floating-point numbers (N in {5M, 10M, 25M, 50M}), simulating big-data telemetry/sensor records."),
        ("Phase 1: Aggregate Reductions", "Compute global Sum, Mean (mu = Sum/N), Variance (sigma^2 = (Sum_sq/N) - mu^2), Std Deviation, Minimum, and Maximum values."),
        ("Phase 2: Dependent Conditional Filtering", "Filter and count elements exceeding the dataset mean (val > mu). Highlights multi-pass synchronization overhead."),
        ("Algorithmic Complexity", "Sequential complexity is O(N) work and O(N) space. Parallel span (critical path) is O(N/P + log P) where P is worker count."),
        ("Reproducibility Assurance", "Deterministic pseudo-random linear congruential generator (LCG) ensures 100% bitwise identical results across Sequential, OpenMP, and MPI.")
    ])

    # Slide 4: Sequential Baseline Implementation
    create_bullet_slide(prs, "3. Sequential Baseline Implementation", [
        ("Execution Paradigm", "Single-threaded von Neumann execution compiled with GCC -O3 optimization level."),
        ("Pass 1 Accumulation Loop", "Linear traversal accumulating sum, sum_sq, min_val, and max_val into CPU registers."),
        ("Statistical Derivation", "Variance and standard deviation derived from first and second moments in O(1) time after Pass 1."),
        ("Pass 2 Filtering Loop", "Branch evaluation (data[i] > mean) counting elements exceeding average in linear time."),
        ("High-Precision Windows Timing", "QueryPerformanceCounter (QPC) provides sub-microsecond resolution timing, free from OS tick quantization artifacts.")
    ])

    # Slide 5: OpenMP Shared-Memory Parallel Design
    create_bullet_slide(prs, "4. OpenMP Shared-Memory Parallel Design", [
        ("Memory Model", "Uniform shared-memory SMP model. All threads share the same virtual address space with zero data duplication."),
        ("Directive-Driven Reductions", "#pragma omp parallel for reduction(+:sum, sum_sq) reduction(min:min_val) reduction(max:max_val) schedule(static)."),
        ("Thread Synchronization", "Implicit barrier at loop completion ensures all private accumulators are merged safely into master variables without race conditions."),
        ("Pass 2 Shared Mean Access", "Calculated mean is immediately readable by all threads from shared RAM without any explicit inter-thread communication."),
        ("Parallel Filter Loop", "#pragma omp parallel for reduction(+:count_above_mean) splits iteration space statically across available threads.")
    ])

    # Slide 6: MPI Distributed-Memory Design
    create_bullet_slide(prs, "5. MPI Distributed-Memory Parallel Design", [
        ("Memory Model", "Shared-nothing distributed memory architecture. Each process has an isolated address space; explicit message passing is required."),
        ("1D Domain Decomposition", "Dataset divided into contiguous chunks: local_count = N / P (+1 for rank < N % P). Local memory allocation ensures zero global RAM bloat."),
        ("Pass 1 Collective Reductions", "MPI_Reduce() performs tree-based aggregation (MPI_SUM, MPI_MIN, MPI_MAX) from all ranks into Rank 0 in O(log P) steps."),
        ("Global Mean Broadcast", "Rank 0 computes global_mean = global_sum / N and broadcasts it to all ranks via MPI_Bcast() for Phase 2."),
        ("Pass 2 Distributed Filter", "Each rank filters its local slice and executes MPI_Reduce() with MPI_LONG_LONG to compute global count above mean.")
    ])

    # Slide 7: Experimental Hardware & Software Setup
    create_bullet_slide(prs, "6. Experimental Hardware & Benchmarking Setup", [
        ("Processor (CPU)", "13th Gen Intel Core i5-13420H (8 Physical Cores, 12 Logical Processors, Base 2.1 GHz, Turbo 4.6 GHz)."),
        ("Memory & Architecture", "16 GB RAM, x86_64 architecture, Windows 11 64-bit OS."),
        ("Compilers & Toolchain", "MinGW-w64 GCC 8.1.0 with -O3 optimization level, Microsoft MPI (MS-MPI) v10.1."),
        ("Dataset Scale Grid", "5,000,000 | 10,000,000 | 25,000,000 | 50,000,000 double-precision elements (40 MB to 400 MB)."),
        ("Worker Counts Tested", "P in {1, 2, 4, 8, 12} threads for OpenMP and {1, 2, 4, 8, 12} processes for MPI."),
        ("Measurement Rigor", "3 repetitions per configuration, computing average runtime, standard deviation, speedup, efficiency, and throughput.")
    ])

    # Slide 8: Execution Time Comparison
    create_image_slide(prs, "7. Execution Time Comparison vs Dataset Size", "execution_time_vs_datasize.png", [
        ("Sub-Linear Runtime", "Execution time scales proportionally with dataset size from 5M to 50M elements across all implementations."),
        ("Significant Latency Drop", "At 50M elements: Sequential requires 81.97 ms, while OpenMP reduces to 31.45 ms and MPI to 27.12 ms at 4 cores."),
        ("Shared vs Distributed", "MPI exhibits slightly higher initial startup overhead at small sizes, but matches and outperforms OpenMP at large scales due to cache isolation."),
        ("Zero Computational Drift", "Outputs match exactly: Sum = 249,997,015,134.12, Mean = 4,999.94, Count = 25,000,772.")
    ])

    # Slide 9: Speedup vs Number of Workers
    create_image_slide(prs, "8. Speedup Analysis vs Ideal Linear Scaling", "speedup_vs_cores.png", [
        ("Near-Linear Scaling (1-4 Cores)", "Both OpenMP and MPI achieve sharp speedup improvements up to 4 cores (OpenMP ~2.6x-2.9x, MPI ~2.9x-3.0x)."),
        ("Peak Speedup at 8-12 Cores", "MPI reaches 3.42x speedup at 12 processes (24.00 ms). OpenMP achieves 3.27x peak speedup at 8 threads (25.10 ms)."),
        ("Amdahl's Law Impact", "Serial fractions (variance computation, barrier synchronization, broadcast latency) set an upper theoretical limit on speedup."),
        ("Memory Bandwidth Saturation", "Beyond 8 threads, multiple cores contend for shared L3 cache and memory bus bandwidth on CPU architecture.")
    ])

    # Slide 10: Efficiency & Throughput Analysis
    create_image_slide(prs, "9. Parallel Efficiency & Throughput Profiling", "efficiency_vs_cores.png", [
        ("High Low-Core Efficiency", "At 2 cores, MPI achieves 89.4% efficiency and OpenMP achieves 86.8% efficiency."),
        ("Diminishing Returns on SMT", "Beyond 8 physical cores, logical hyperthreads (threads 9-12) share execution pipelines, reducing per-worker efficiency."),
        ("Massive Throughput Gains", "Throughput scales from 610 Million elements/sec (Sequential) to over 2,083 Million elements/sec (MPI 12 Procs)."),
        ("Optimal Operating Point", "4 to 8 workers represent the sweet spot for maximizing speedup without excessive parallel overhead.")
    ])

    # Slide 11: Summary Comparison Table
    create_table_slide(prs, "10. Comprehensive Comparison: OpenMP vs MPI", [
        "Feature / Metric", "OpenMP (Shared Memory)", "MPI (Distributed Memory)", "Sequential Baseline"
    ], [
        ["Hardware Model", "Multi-core CPU (SMP / Shared RAM)", "Cluster / Multi-node / Distributed", "Single CPU Core"],
        ["Memory Space", "Single Shared Address Space", "Isolated Local Memory per Rank", "Single Local Process Space"],
        ["Communication", "Implicit via Shared Memory Reads", "Explicit via MPI_Reduce / MPI_Bcast", "None (Direct Access)"],
        ["Implementation", "Compiler Directives (#pragma omp)", "Library API Calls (MPI_Init, etc.)", "Standard C code"],
        ["50M Runtime (4 Workers)", "31.45 ms (2.61x speedup)", "27.12 ms (3.02x speedup)", "81.97 ms (Baseline)"],
        ["50M Peak Speedup", "3.27x (8 Threads)", "3.42x (12 Processes)", "1.00x (Baseline)"],
        ["Max Throughput (50M)", "1,992 Million items/sec", "2,083 Million items/sec", "610 Million items/sec"],
        ["Fault Tolerance", "Thread failure crashes process", "Process failure isolation possible", "Process failure fatal"],
        ["Best Used For", "Local compute-heavy loops & SMP", "Large clusters & Big Data HPC", "Small / single-threaded tasks"]
    ])

    # Slide 12: Architectural Discussion & Scalability Insights
    create_bullet_slide(prs, "11. Architectural Discussion & Scalability Insights", [
        ("Shared vs. Distributed Memory Trade-offs", "OpenMP excels on single-node SMP systems with zero message serialization overhead. MPI provides superior cache isolation through 1D domain decomposition."),
        ("Collective Communication Latency", "MPI requires explicit synchronization (MPI_Bcast, MPI_Reduce) using binomial spanning trees with O(log P) complexity to distribute and reduce variables."),
        ("Memory Bus & Roofline Limits", "Speedup saturation beyond 8 cores is governed by memory bandwidth limits on the host memory controller when processing contiguous floating-point streams."),
        ("Heterogeneous Core Scaling", "Intel hybrid architectures (P-cores vs E-cores) and hyperthreading (SMT) deliver diminishing efficiency returns compared to dedicated physical execution units.")
    ])

    # Slide 13: Conclusion & Future Scope
    create_bullet_slide(prs, "12. Conclusion & Future Scope", [
        ("Key Conclusion 1", "Both OpenMP and MPI successfully accelerated the data processing workload, achieving >3.2x-3.4x speedup over the sequential baseline."),
        ("Key Conclusion 2", "OpenMP provides simple syntax and excellent developer ergonomics for shared-memory workstations with zero message-passing overhead."),
        ("Key Conclusion 3", "MPI provides superior memory scalability and cache locality via domain decomposition, making it indispensable for distributed big data clusters."),
        ("Future Scope: Hybrid OpenMP + MPI", "Combine MPI across distributed nodes and OpenMP within multi-core nodes for state-of-the-art Petascale HPC data processing pipelines.")
    ])

    prs.save(OUTPUT_PPTX)
    print(f"Presentation successfully saved to: {OUTPUT_PPTX}")

if __name__ == "__main__":
    build_presentation()
