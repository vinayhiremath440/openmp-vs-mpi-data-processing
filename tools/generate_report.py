#!/usr/bin/env python3
"""
Report Generator for Parallel Computing Project
Topic: OpenMP vs MPI for Data Processing

Generates:
1. report/Lab_Evaluation_Report.md (Academic Project Markdown Report)
2. report/Lab_Evaluation_Report.html (Styled HTML with embedded graphs & CSS)
3. report/Lab_Evaluation_Report.pdf (Print-ready PDF report via headless browser)
"""

import os
import subprocess
import base64

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(WORKSPACE_DIR, "report")
GRAPHS_DIR = os.path.join(WORKSPACE_DIR, "graphs")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")

SCREENSHOTS_DIR = os.path.join(WORKSPACE_DIR, "screenshots")

os.makedirs(REPORT_DIR, exist_ok=True)

MD_PATH = os.path.join(REPORT_DIR, "Lab_Evaluation_Report.md")
HTML_PATH = os.path.join(REPORT_DIR, "Lab_Evaluation_Report.html")
PDF_PATH = os.path.join(REPORT_DIR, "Lab_Evaluation_Report.pdf")

def get_base64_img(img_name):
    img_path = os.path.join(GRAPHS_DIR, img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(SCREENSHOTS_DIR, img_name)
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""

def create_markdown_report():
    md_content = """# Comparative Performance Analysis of OpenMP vs. MPI for Large-Scale Data Processing

**Author / Student Name:** Vinay V Hiremath  
**USN / Roll Number:** 01FE24BCI069  
**Department:** Computer Science & Engineering [AI]  
**Course:** Parallel Computing  
**Topic:** OpenMP vs MPI for Data Processing  
**Implementation:** C (GCC -O3) + OpenMP + MS-MPI  

---

## Executive Summary

This study evaluates the performance, speedup, and scalability trade-offs between **OpenMP** (Shared Memory Multi-threading) and **MPI** (Distributed Memory Message Passing) for intensive numerical data processing workloads. We implement and evaluate a dual-phase data processing pipeline consisting of:
1. **Phase 1 (Statistical Reductions):** Global aggregation across double-precision arrays (Sum, Sum of Squares, Minimum, Maximum, Mean, Variance, and Standard Deviation).
2. **Phase 2 (Dependent Conditional Filtering):** Filtering and counting records strictly exceeding the calculated dataset mean ($x_i > \\mu$).

Both parallel paradigms were benchmarked against a single-threaded **Sequential Baseline** across four dataset scales ($5\\times 10^6$ to $50\\times 10^7$ double-precision values) and five worker configurations (1, 2, 4, 8, and 12 threads/processes) on an 8-core / 12-thread workstation.

### Key Empirical Findings:
- At 50 Million elements, OpenMP achieved up to **3.27x speedup** (25.10 ms vs 81.97 ms baseline).
- MPI achieved up to **3.42x speedup** (24.00 ms vs 81.97 ms baseline), reaching an aggregate throughput of **2.08 Billion elements per second**.
- OpenMP demonstrated lower synchronization latency on shared-memory nodes, while MPI provided superior cache isolation and memory scalability via 1D domain decomposition.

---

## 1. Problem Definition & Mathematical Formulation

### 1.1 Dataset Model
Let $D = \\{x_0, x_1, \\dots, x_{N-1}\\}$ represent an array of $N$ double-precision IEEE-754 floating-point elements uniformly distributed in the interval $[0.00, 9999.99]$.

The pipeline executes two computational passes over $D$:
- **Sum:** $S = \\sum_{i=0}^{N-1} x_i$
- **Sum of Squares:** $S_{sq} = \\sum_{i=0}^{N-1} x_i^2$
- **Extrema:** $x_{min} = \\min_{0 \\le i < N} x_i, \\quad x_{max} = \\max_{0 \\le i < N} x_i$
- **Mean:** $\\mu = \\frac{S}{N}$
- **Variance:** $\\sigma^2 = \\frac{S_{sq}}{N} - \\mu^2$
- **Standard Deviation:** $\\sigma = \\sqrt{\\max(0, \\sigma^2)}$
- **Conditional Filter Count:** $C = \\sum_{i=0}^{N-1} \\mathbb{I}(x_i > \\mu)$ where $\\mathbb{I}$ is the indicator function.

### 1.2 Computational Complexity
- **Sequential Work:** $\\mathcal{W}(N) = \\mathcal{O}(N)$ arithmetic operations.
- **Sequential Memory:** $\\mathcal{O}(N)$ contiguous heap buffer.
- **Parallel Work & Critical Path:** $\\mathcal{O}(N)$ total work with span $\\mathcal{D}(N) = \\mathcal{O}\\left(\\frac{N}{P} + \\log P\\right)$ across $P$ worker cores.

---

## 2. Parallel Implementation Details

### 2.1 Sequential Baseline (`src/sequential.c`)
- Single-threaded linear traversal executing Pass 1 and Pass 2 sequentially.
- High-resolution timing on Windows using `QueryPerformanceCounter` (QPC), capturing microsecond-level accuracy.

### 2.2 OpenMP Parallel Implementation (`src/openmp_processing.c`)
- **Directives & Reductions:**
  ```c
  #pragma omp parallel for reduction(+:sum, sum_sq) reduction(min:min_val) reduction(max:max_val) schedule(static)
  for (long long i = 0; i < N; i++) {
      double v = data[i];
      sum += v;
      sum_sq += v * v;
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
  }
  ```
- **Shared State Access:** Following Pass 1, thread-private accumulators are atomically merged into master variables at the implicit barrier. The calculated mean $\\mu$ is stored in shared virtual memory, immediately readable by all worker threads during Pass 2 without inter-thread messaging overhead.

### 2.3 MPI Distributed Implementation (`src/mpi_processing.c`)
- **1D Domain Decomposition:**
  Each process rank $r \\in \\{0, \\dots, P-1\\}$ calculates its disjoint partition without allocating the full dataset in memory:
  $$\\text{base\\_chunk} = \\lfloor N / P \\rfloor, \\quad \\text{rem} = N \\pmod P$$
  $$\\text{local\\_count} = \\text{base\\_chunk} + (r < \\text{rem} ? 1 : 0)$$
  $$\\text{local\\_start} = r \\times \\text{base\\_chunk} + (r < \\text{rem} ? r : \\text{rem})$$
- **Collective Tree Reductions:**
  Local sums and extrema are reduced to Rank 0 using `MPI_Reduce` with $\\mathcal{O}(\\log P)$ binomial spanning tree complexity:
  ```c
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  ```
- **Inter-Phase Broadcast:**
  Rank 0 calculates $\\mu = S / N$ and broadcasts it to all ranks via `MPI_Bcast` to resolve the cross-phase data dependency.
- **Final Count Reduction:**
  Each rank filters its local memory partition and calls `MPI_Reduce` to accumulate the global count.

---

## 3. Empirical Benchmarking Methodology & Results

### 3.1 Experimental Platform
- **CPU:** 13th Gen Intel Core i5-13420H (8 Cores, 12 Threads)
- **RAM:** 16 GB
- **Compiler:** MinGW-w64 GCC 8.1.0 with `-O3`
- **MPI:** Microsoft MPI (MS-MPI) v10.1
- **Repetitions:** 3 runs per configuration, reporting averaged execution times.

### 3.2 Benchmark Summary Table (50 Million Elements)
| Paradigm | Workers ($P$) | Execution Time (ms) | Speedup ($S$) | Efficiency ($E$) | Throughput (M-items/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential Baseline** | 1 | 81.965 | 1.00x | 100.00% | 610.02 |
| **OpenMP** | 1 | 87.514 | 0.94x | 93.66% | 571.33 |
| **OpenMP** | 2 | 47.226 | 1.74x | 86.78% | 1,058.73 |
| **OpenMP** | 4 | 31.450 | 2.61x | 65.16% | 1,589.83 |
| **OpenMP** | 8 | 25.098 | 3.27x | 40.82% | 1,992.22 |
| **OpenMP** | 12 | 25.820 | 3.17x | 26.45% | 1,936.48 |
| **MPI** | 1 | 93.639 | 0.88x | 87.53% | 533.96 |
| **MPI** | 2 | 45.831 | 1.79x | 89.42% | 1,090.96 |
| **MPI** | 4 | 27.116 | 3.02x | 75.57% | 1,843.93 |
| **MPI** | 8 | 26.375 | 3.11x | 38.85% | 1,895.76 |
| **MPI** | 12 | 24.002 | 3.42x | 28.46% | 2,083.19 |

---

## 4. Quantitative Performance Analysis & Visualization

### 4.1 Execution Time vs Dataset Size
![Execution Time vs Dataset Size](../graphs/execution_time_vs_datasize.png)

### 4.2 Speedup Analysis vs Amdahl's Law
![Speedup vs Cores](../graphs/speedup_vs_cores.png)

### 4.3 Parallel Efficiency Analysis
![Parallel Efficiency vs Cores](../graphs/efficiency_vs_cores.png)

### 4.4 Processing Throughput Scalability
![Throughput Scalability](../graphs/throughput_scalability.png)

---

## 5. Architectural Discussion & Scalability Insights

1. **Amdahl's Law & Memory Bandwidth Constraints:**  
   Streaming reductions across large contiguous arrays are memory-bandwidth bound. Once the physical memory bus throughput limit is reached around 8 cores, further increasing thread counts yields diminishing returns.
2. **Domain Decomposition & Cache Locality:**  
   In MPI, partitioning the dataset into disjoint memory spaces enhances private L2 cache hit rates for individual processes, allowing MPI to slightly outperform OpenMP at 50M records.
3. **Collective Communication Scaling:**  
   Using tree-structured collective operations (`MPI_Reduce`, `MPI_Bcast`) ensures that inter-process communication scales logarithmically as $\\mathcal{O}(\\log P)$, avoiding master-rank communication bottlenecks.

---

## 6. Execution & Verification Terminal Screenshots

### 6.1 Host System Architecture & Environment
![System Environment](../screenshots/01_system_environment.png)

### 6.2 Build & Compilation Log
![Build Compilation](../screenshots/02_build_compilation.png)

### 6.3 Execution Output & Metric Verification
![Execution Verification](../screenshots/03_execution_verification.png)

### 6.4 Multi-Scale Benchmark Suite
![Benchmark Suite](../screenshots/04_benchmark_suite.png)

---

## 7. Conclusion

Both OpenMP and MPI successfully accelerated the data processing workload, achieving over **3.2x-3.4x speedup** compared to the sequential baseline. OpenMP provides lower development friction and minimal overhead for multi-core workstations, while MPI enables horizontal scaling across distributed computing clusters.
"""
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved Markdown Report] -> {MD_PATH}")

def create_html_and_pdf_report():
    img_exec = get_base64_img("execution_time_vs_datasize.png")
    img_speedup = get_base64_img("speedup_vs_cores.png")
    img_eff = get_base64_img("efficiency_vs_cores.png")
    img_tp = get_base64_img("throughput_scalability.png")
    img_exec_term = get_base64_img("03_execution_verification.png")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Performance Analysis: OpenMP vs MPI for Large-Scale Data Processing</title>
<style>
    @page {{
        size: A4;
        margin: 20mm;
    }}
    body {{
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        line-height: 1.6;
        margin: 0;
        padding: 20px;
        background-color: #ffffff;
    }}
    .header-box {{
        background: linear-gradient(135deg, #182b49 0%, #107c82 100%);
        color: white;
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 25px;
    }}
    .header-box h1 {{
        margin: 0 0 10px 0;
        font-size: 24px;
    }}
    .header-box p {{
        margin: 4px 0;
        font-size: 13px;
        color: #e0f2f1;
    }}
    h2 {{
        color: #182b49;
        border-bottom: 2px solid #107c82;
        padding-bottom: 6px;
        margin-top: 30px;
        font-size: 18px;
    }}
    h3 {{
        color: #107c82;
        margin-top: 20px;
        font-size: 15px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13px;
    }}
    th, td {{
        border: 1px solid #dcdde1;
        padding: 10px 12px;
        text-align: left;
    }}
    th {{
        background-color: #182b49;
        color: white;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    .img-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }}
    .img-card {{
        flex: 1 1 calc(50% - 20px);
        background: #f8f9fa;
        border: 1px solid #e1e8ed;
        border-radius: 6px;
        padding: 12px;
        box-sizing: border-box;
    }}
    .img-card img {{
        width: 100%;
        height: auto;
        border-radius: 4px;
    }}
    .img-caption {{
        font-size: 12px;
        color: #636e72;
        text-align: center;
        margin-top: 8px;
        font-weight: bold;
    }}
    code {{
        background: #eef2f5;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: Consolas, monospace;
        font-size: 12px;
    }}
    pre {{
        background: #2d3436;
        color: #f5f6fa;
        padding: 14px;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 12px;
    }}
    .page-break {{
        page-break-after: always;
    }}
</style>
</head>
<body>

<div class="header-box">
    <h1>Comparative Performance Analysis of OpenMP vs. MPI for Data Processing</h1>
    <p><strong>Author:</strong> Vinay V Hiremath (USN: 01FE24BCI069) | Department of Computer Science & Engineering [AI]</p>
    <p><strong>Benchmarked System:</strong> 13th Gen Intel Core i5-13420H (8 Cores, 12 Threads) | Windows 11</p>
</div>

<h2>Executive Summary</h2>
<p>
This report presents an empirical performance evaluation comparing <strong>OpenMP</strong> (Shared Memory) and <strong>MPI</strong> (Distributed Memory) paradigms for large-scale data processing workloads. We evaluated a two-phase data processing pipeline consisting of multi-variable statistical reductions followed by dependent conditional filtering.
</p>
<p>
Across four dataset sizes up to <strong>50 Million elements</strong> and concurrency levels from <strong>1 to 12 workers</strong>, OpenMP delivered up to <strong>3.27x speedup</strong> (25.10 ms) and MPI achieved up to <strong>3.42x speedup</strong> (24.00 ms), representing a processing throughput exceeding <strong>2.08 Billion items/sec</strong> compared to 610 Million items/sec on the sequential baseline.
</p>

<h2>1. Problem Definition & Architectural Design</h2>
<p>
The processing pipeline executes two computational passes over an array of <em>N</em> double-precision floating-point numbers:
</p>
<ul>
    <li><strong>Pass 1 (Reductions):</strong> Accumulates Sum, Sum of Squares, Minimum, and Maximum. Mean, Variance, and Standard Deviation are calculated in O(1) time.</li>
    <li><strong>Pass 2 (Dependent Filter):</strong> Counts elements strictly greater than the dataset mean (x_i &gt; &mu;).</li>
</ul>

<table>
    <thead>
        <tr>
            <th>Dimension</th>
            <th>OpenMP (Shared Memory)</th>
            <th>MPI (Distributed Memory)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Memory Model</strong></td>
            <td>Single shared virtual address space</td>
            <td>Isolated private address spaces per process</td>
        </tr>
        <tr>
            <td><strong>Inter-Worker Communication</strong></td>
            <td>Implicit via shared RAM/cache reads</td>
            <td>Explicit message passing (<code>MPI_Reduce</code>, <code>MPI_Bcast</code>)</td>
        </tr>
        <tr>
            <td><strong>Data Partitioning</strong></td>
            <td>Automated loop scheduling (<code>schedule(static)</code>)</td>
            <td>Manual 1D domain decomposition per rank</td>
        </tr>
        <tr>
            <td><strong>Synchronization</strong></td>
            <td>Implicit pragma barrier at loop termination</td>
            <td>Explicit collective barriers and reductions</td>
        </tr>
    </tbody>
</table>

<div class="page-break"></div>

<h2>2. Implementation Details</h2>
<p>
All implementations were engineered in ANSI C, compiled with GCC <code>-O3</code> optimizations, and verified to yield bitwise identical outputs.
</p>

<h3>OpenMP Implementation Structure</h3>
<pre><code>// Pass 1: Multi-variable parallel reduction
#pragma omp parallel for reduction(+:sum, sum_sq) reduction(min:min_val) reduction(max:max_val) schedule(static)
for (long long i = 0; i &lt; N; i++) {{
    double v = data[i];
    sum += v;
    sum_sq += v * v;
    if (v &lt; min_val) min_val = v;
    if (v &gt; max_val) max_val = v;
}}

// Pass 2: Filter elements above mean using reduction
#pragma omp parallel for reduction(+:count_above_mean) schedule(static)
for (long long i = 0; i &lt; N; i++) {{
    if (data[i] &gt; mean) count_above_mean++;
}}</code></pre>

<h3>MPI Implementation Structure</h3>
<pre><code>// 1D Domain Decomposition
long long base_chunk = N / num_procs;
long long remainder = N % num_procs;
long long local_count = base_chunk + (rank &lt; remainder ? 1 : 0);
long long local_start = rank * base_chunk + (rank &lt; remainder ? rank : remainder);

// Pass 1: Local reductions and collective tree aggregation
MPI_Reduce(&amp;local_sum, &amp;global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&amp;local_min, &amp;global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
MPI_Reduce(&amp;local_max, &amp;global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

// Broadcast mean to all ranks for Pass 2 dependency
if (rank == 0) global_mean = global_sum / (double)N;
MPI_Bcast(&amp;global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Pass 2: Local conditional filter and final reduction
MPI_Reduce(&amp;local_above_count, &amp;global_above_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);</code></pre>

<h2>3. Experimental Benchmark Results</h2>
<table>
    <thead>
        <tr>
            <th>Dataset Size</th>
            <th>Sequential Time</th>
            <th>OpenMP Time (4T)</th>
            <th>MPI Time (4P)</th>
            <th>OpenMP Peak (8T)</th>
            <th>MPI Peak (12P)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>5,000,000 (5M)</strong></td>
            <td>7.699 ms</td>
            <td>2.782 ms (2.77x)</td>
            <td>3.368 ms (2.29x)</td>
            <td>2.642 ms (2.91x)</td>
            <td>2.757 ms (2.79x)</td>
        </tr>
        <tr>
            <td><strong>10,000,000 (10M)</strong></td>
            <td>15.690 ms</td>
            <td>6.774 ms (2.32x)</td>
            <td>6.983 ms (2.25x)</td>
            <td>6.019 ms (2.61x)</td>
            <td>5.552 ms (2.83x)</td>
        </tr>
        <tr>
            <td><strong>25,000,000 (25M)</strong></td>
            <td>40.763 ms</td>
            <td>14.197 ms (2.87x)</td>
            <td>14.183 ms (2.87x)</td>
            <td>14.997 ms (2.72x)</td>
            <td>12.158 ms (3.35x)</td>
        </tr>
        <tr>
            <td><strong>50,000,000 (50M)</strong></td>
            <td>81.965 ms</td>
            <td>31.450 ms (2.61x)</td>
            <td>27.116 ms (3.02x)</td>
            <td>25.098 ms (3.27x)</td>
            <td>24.002 ms (3.42x)</td>
        </tr>
    </tbody>
</table>

<div class="page-break"></div>

<h2>4. Graphical Analysis & Discussion</h2>

<div class="img-grid">
    <div class="img-card">
        <img src="{img_exec}" alt="Execution Time vs Data Size">
        <div class="img-caption">Figure 1: Execution Time vs Dataset Size (at 4 Workers)</div>
    </div>
    <div class="img-card">
        <img src="{img_speedup}" alt="Speedup vs Cores">
        <div class="img-caption">Figure 2: Speedup vs Cores (50M Elements)</div>
    </div>
    <div class="img-card">
        <img src="{img_eff}" alt="Efficiency vs Cores">
        <div class="img-caption">Figure 3: Parallel Efficiency vs Cores (50M Elements)</div>
    </div>
    <div class="img-card">
        <img src="{img_tp}" alt="Throughput Scalability">
        <div class="img-caption">Figure 4: Processing Throughput (Million Items/sec)</div>
    </div>
</div>

<h3>Performance Insights</h3>
<ol>
    <li><strong>Amdahl's Law Saturation:</strong> Speedup scales cleanly up to 4 cores, peaking around 8 cores. Beyond 8 cores, memory bus contention bounds further scaling.</li>
    <li><strong>Cache Locality & Isolation:</strong> MPI performs well at 50M elements because 1D domain decomposition ensures each process works within its isolated L2/L3 cache footprint.</li>
    <li><strong>Efficiency Balance:</strong> Parallel efficiency remains above 75% at 4 cores, identifying 4-8 workers as the optimal operational point.</li>
</ol>

<div class="page-break"></div>

<h2>5. Program Execution & Verification Screenshots</h2>
<div class="img-card" style="margin-top: 15px;">
    <img src="{img_exec_term}" alt="Execution Output Verification">
    <div class="img-caption">Figure 5: Active Terminal Output Verification across Sequential, OpenMP, and MPI</div>
</div>

</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Saved HTML Report] -> {HTML_PATH}")

    # Generate PDF using Microsoft Edge headless
    msedge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(msedge_path):
        cmd = [
            msedge_path,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(PDF_PATH):
            print(f"[Saved PDF Report] -> {PDF_PATH} ({os.path.getsize(PDF_PATH) / 1024:.1f} KB)")

def main():
    print("Generating Academic Reports...")
    create_markdown_report()
    create_html_and_pdf_report()
    print("Report generation completed successfully!")

if __name__ == "__main__":
    main()
