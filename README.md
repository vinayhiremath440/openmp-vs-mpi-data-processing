# Comparative Performance Analysis of OpenMP vs. MPI for Large-Scale Data Processing

A high-performance computing project benchmarking shared-memory multi-threading (**OpenMP**) against distributed-memory message passing (**MPI**) on numerical data processing pipelines.

---

## 👨‍💻 Project Overview
- **Author:** Vinay Hiremath
- **Domain:** Parallel & Distributed Computing
- **Parallel Models:** OpenMP (Shared Memory SMP) vs. MPI (Distributed Memory Message Passing)
- **Primary Language:** C (GCC with `-O3` optimization)
- **Runtime Environment:** Microsoft Windows 11 (64-bit), MinGW-w64, Microsoft MPI (MS-MPI)

---

## 📌 Abstract & Motivation

As dataset sizes scale exponentially in big data analytics, financial modeling, and scientific simulations, choosing the appropriate parallelization paradigm is critical for computational efficiency and scalability.

This project investigates the performance characteristics, scaling behavior, and architectural trade-offs between two fundamental parallel computing models:
1. **OpenMP (Shared Memory):** Thread-based parallelism operating within a single unified virtual memory space.
2. **MPI (Distributed Memory):** Process-based parallelism operating across isolated address spaces via explicit message-passing communication.

Both models are benchmarked against an optimized single-threaded **Sequential Baseline** across varying dataset sizes (from **5 Million** to **50 Million** double-precision floating-point records) and concurrency levels (1, 2, 4, 8, and 12 workers) on a multi-core processor.

---

## 🔬 Mathematical Workload & Algorithmic Formulation

To realistically evaluate parallel performance beyond trivial single-loop operations, this project implements a **two-phase data processing pipeline** representing typical aggregation and filtering workloads.

Let $D = \{x_0, x_1, x_2, \dots, x_{N-1}\}$ represent an array of $N$ double-precision values ($x_i \in [0.00, 9999.99]$).

### Phase 1: Statistical Reductions
Compute descriptive statistical moments and extrema across array $D$:
- **Sum:**  
  $$S = \sum_{i=0}^{N-1} x_i$$
- **Sum of Squares:**  
  $$S_{sq} = \sum_{i=0}^{N-1} x_i^2$$
- **Extrema:**  
  $$x_{min} = \min_{0 \le i < N} x_i, \quad x_{max} = \max_{0 \le i < N} x_i$$
- **Dataset Mean:**  
  $$\mu = \frac{S}{N}$$
- **Variance & Standard Deviation:**  
  $$\sigma^2 = \frac{S_{sq}}{N} - \mu^2, \quad \sigma = \sqrt{\max(0, \sigma^2)}$$

### Phase 2: Dependent Conditional Filtering
Filter and count all records strictly exceeding the computed global mean:
$$C = \sum_{i=0}^{N-1} \mathbb{I}(x_i > \mu)$$
*(where $\mathbb{I}$ is the indicator function)*

> **Architectural Significance:**  
> Phase 2 creates an explicit cross-phase dependency on $\mu$. In OpenMP, $\mu$ is stored in shared virtual memory and read directly by all worker threads. In MPI, each process has an isolated memory space; hence, Rank 0 must explicitly broadcast $\mu$ using `MPI_Bcast` before distributed worker ranks can execute Phase 2.

---

## 🏛️ Parallel Architectural Design

```text
+-----------------------------------------------------------------------------+
|                               PARALLEL MODELS                               |
+------------------------------------+----------------------------------------+
|       OpenMP (Shared Memory)       |       MPI (Distributed Memory)         |
+------------------------------------+----------------------------------------+
|                                    |                                        |
|         Unified Address Space      |   Rank 0       Rank 1       Rank P-1   |
|     +----------------------------+ |   +------+     +------+     +------+   |
|     |     Shared Array D[0..N]   | |   |Part 0|     |Part 1|     |Part P|   |
|     +----------------------------+ |   +------+     +------+     +------+   |
|       Thread 0  Thread 1  Thread P |      \            |            /       |
|          \         |         /     |       \           |           /        |
|           \        |        /      |        -->  MPI_Reduce (Sum) <--       |
|       #pragma omp parallel for     |                   |                    |
|       reduction(+:sum, sum_sq)     |             Compute Mean (mu)          |
|       reduction(min:min_val)       |                   |                    |
|       reduction(max:max_val)       |        <--   MPI_Bcast (mu)   -->      |
|                   |                |                   |                    |
|       Compute Mean (shared RAM)    |       Local Filtering (x > mu)         |
|                   |                |                   |                    |
|       Parallel Filter Reduction    |        -->  MPI_Reduce (Count) <--     |
+------------------------------------+----------------------------------------+
```

### 1. Sequential Baseline (`src/sequential.c`)
- Single-threaded execution compiled with `-O3`.
- Linear scan through memory with sub-microsecond precision timing via Windows `QueryPerformanceCounter` (QPC).

### 2. OpenMP Implementation (`src/openmp_processing.c`)
- Uses `#pragma omp parallel for reduction(+:sum, sum_sq) reduction(min:min_val) reduction(max:max_val) schedule(static)`.
- Eliminates race conditions by allocating thread-private accumulators merged at loop boundaries.
- Zero inter-thread message serialization overhead.

### 3. MPI Distributed Implementation (`src/mpi_processing.c`)
- Employs **1D Domain Decomposition**:
  $$\text{local\_count} = \lfloor N / P \rfloor + (\text{rank} < N \pmod P ? 1 : 0)$$
- Each rank allocates and processes only its local chunk, enabling execution on datasets that exceed single-machine RAM limits.
- Uses `MPI_Reduce` for tree-based reduction of partial metrics to Rank 0.
- Rank 0 broadcasts global mean $\mu$ to all processes via `MPI_Bcast`.
- Second `MPI_Reduce` aggregates total filtered count to Rank 0.

---

## 📂 Repository Layout

```text
openmp-vs-mpi-data-processing/
├── README.md                      # Comprehensive project documentation
├── build.bat                      # Build script for Sequential, OpenMP & MPI
├── run_benchmarks.py              # Automated multi-core benchmark suite
├── generate_graphs.py             # Script generating high-resolution metric plots
├── bin/                           # Compiled binaries & runtime libraries
│   ├── sequential.exe             # Sequential baseline executable
│   ├── openmp_processing.exe      # OpenMP executable
│   ├── mpi_processing.exe         # MPI executable
│   └── *.dll                      # Runtime dependencies (libgomp, msmpi, etc.)
├── src/                           # C source code
│   ├── sequential.c               # Baseline algorithm
│   ├── openmp_processing.c        # OpenMP parallel implementation
│   └── mpi_processing.c           # MPI distributed implementation
├── data/                          # Dataset generation utility
│   └── generate_dataset.py        # Generates synthetic data in CSV / binary formats
├── results/                       # Experimental benchmarking logs
│   ├── raw_benchmark_runs.csv     # Raw timing across all repetitions
│   ├── benchmark_summary.csv      # Statistical summary (mean, speedup, efficiency)
│   └── benchmark_summary.txt      # Formatted summary text table
├── graphs/                        # Performance visualization plots
│   ├── execution_time_vs_datasize.png
│   ├── speedup_vs_cores.png
│   ├── efficiency_vs_cores.png
│   └── throughput_scalability.png
├── report/                        # Technical project report
│   ├── Lab_Evaluation_Report.pdf  # PDF report
│   ├── Lab_Evaluation_Report.html # HTML report
│   └── Lab_Evaluation_Report.md   # Markdown source report
└── presentation/                  # Project presentation
    └── OpenMP_vs_MPI_Data_Processing.pptx  # Slide deck
```

---

## 🛠️ Build and Execution Instructions

### Prerequisites
- GCC Compiler (MinGW-w64 with OpenMP support)
- Microsoft MPI (MS-MPI v10.1 or higher)
- Python 3.x (with `pandas`, `numpy`, `matplotlib`, `python-pptx`)

### Step 1: Compile Executables
Execute the provided build script from the repository root:
```cmd
build.bat
```
This generates optimized `-O3` binaries in the `bin/` directory.

### Step 2: Run Individual Executables

#### Sequential Baseline
```cmd
bin\sequential.exe [dataset_size]
```
*Example:* `bin\sequential.exe 10000000`

#### OpenMP Parallel Execution
```cmd
bin\openmp_processing.exe [dataset_size] [num_threads]
```
*Example (10M elements on 4 threads):* `bin\openmp_processing.exe 10000000 4`

#### MPI Distributed Execution
```cmd
tools\mpi\bin\mpiexec.exe -n [num_processes] bin\mpi_processing.exe [dataset_size]
```
*Example (10M elements on 4 processes):*  
```cmd
tools\mpi\bin\mpiexec.exe -n 4 bin\mpi_processing.exe 10000000
```

### Step 3: Run Full Benchmark Suite
To execute the automated benchmark across all configurations (5M to 50M elements, 1 to 12 workers, 3 runs each):
```cmd
python run_benchmarks.py
```

### Step 4: Generate Graphs
```cmd
python generate_graphs.py
```

---

## 📊 Experimental Results & Performance Analysis

### Hardware Benchmark Platform
- **Processor:** 13th Gen Intel Core i5-13420H (8 Cores, 12 Threads)
- **Memory:** 16 GB RAM
- **Operating System:** Microsoft Windows 11 (64-bit)

---

### Benchmark Data (50 Million Elements - Largest Scale)

| Implementation | Workers ($P$) | Execution Time (ms) | Speedup ($S$) | Parallel Efficiency ($E$) | Throughput (M-items/sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sequential Baseline** | **1** | **81.965 ms** | **1.00x** | **100.00%** | **610.02** |
| **OpenMP** | 1 | 87.514 ms | 0.94x | 93.66% | 571.33 |
| **OpenMP** | 2 | 47.226 ms | 1.74x | 86.78% | 1,058.73 |
| **OpenMP** | 4 | 31.450 ms | 2.61x | 65.16% | 1,589.83 |
| **OpenMP** | 8 | 25.098 ms | **3.27x** | 40.82% | **1,992.22** |
| **OpenMP** | 12 | 25.820 ms | 3.17x | 26.45% | 1,936.48 |
| **MPI** | 1 | 93.639 ms | 0.88x | 87.53% | 533.96 |
| **MPI** | 2 | 45.831 ms | 1.79x | 89.42% | 1,090.96 |
| **MPI** | 4 | 27.116 ms | 3.02x | 75.57% | 1,843.93 |
| **MPI** | 8 | 26.375 ms | 3.11x | 38.85% | 1,895.76 |
| **MPI** | 12 | 24.002 ms | **3.42x** | 28.46% | **2,083.19** |

---

### Scaling Across Dataset Sizes (at 4 Workers)

| Dataset Size | Sequential Baseline | OpenMP (4 Threads) | MPI (4 Processes) |
| :---: | :---: | :---: | :---: |
| **5 Million (5M)** | 7.699 ms | **2.782 ms** (2.77x) | **3.368 ms** (2.29x) |
| **10 Million (10M)** | 15.690 ms | **6.774 ms** (2.32x) | **6.983 ms** (2.25x) |
| **25 Million (25M)** | 40.763 ms | **14.197 ms** (2.87x) | **14.183 ms** (2.87x) |
| **50 Million (50M)** | 81.965 ms | **31.450 ms** (2.61x) | **27.116 ms** (3.02x) |

---

## 📈 Performance Visualizations

### 1. Execution Time Comparison
![Execution Time vs Dataset Size](graphs/execution_time_vs_datasize.png)
*Figure 1: Wall-clock execution time across dataset sizes. Parallel models consistently demonstrate significant latency reductions.*

### 2. Speedup vs Number of Workers
![Speedup vs Cores](graphs/speedup_vs_cores.png)
*Figure 2: Scaling speedup vs Worker Concurrency. Near-linear acceleration up to 4 cores, reaching peak speedup of ~3.4x.*

### 3. Parallel Efficiency
![Parallel Efficiency vs Cores](graphs/efficiency_vs_cores.png)
*Figure 3: Parallel efficiency remains high (>75-89%) at 2-4 cores and gradually levels off as SMT hyperthreads share execution pipelines.*

### 4. Throughput Scalability
![Throughput Scalability](graphs/throughput_scalability.png)
*Figure 4: Data processing throughput increases from 610 Million elements/sec up to over 2.08 Billion elements/sec.*

---

## 💡 Key Findings & Architectural Insights

1. **Low Overhead of OpenMP:**  
   OpenMP demonstrates minimal startup latency on shared-memory nodes because threads reuse the existing virtual address space without memory duplication.
2. **Domain Decomposition Scalability in MPI:**  
   MPI achieves high throughput at large data scales (50M elements) because domain decomposition gives each process its own isolated memory chunk, optimizing L2/L3 cache utilization.
3. **Memory Bandwidth Bottleneck:**  
   Both paradigms experience speedup saturation beyond 8 cores. For streaming vector data processing, the shared memory bus bandwidth becomes the governing constraint (as described by Amdahl's Law and the Roofline model).
4. **Numerical Precision Consistency:**  
   All three implementations yield bitwise identical output values, confirming correctness across reduction and filtering passes.

---

## ⚖️ Architectural Comparison Summary

| Parameter | Sequential | OpenMP | MPI |
| :--- | :--- | :--- | :--- |
| **Programming Paradigm** | Single-threaded | Multi-threaded Directives | Message-Passing APIs |
| **Memory Model** | Single address space | Shared memory (SMP) | Distributed (Shared-nothing) |
| **Communication Mechanism** | None (Direct access) | Implicit (Shared RAM) | Explicit (`MPI_Reduce`, `MPI_Bcast`) |
| **Domain Partitioning** | None | Automatic loop scheduling | Manual 1D domain decomposition |
| **Scalability Bound** | 1 Core | Single node cores & RAM | Multi-node cluster interconnects |
| **Peak 50M Speedup** | 1.00x | **3.27x** (8 Threads) | **3.42x** (12 Processes) |
| **Peak Throughput** | 610 M items/sec | 1,992 M items/sec | 2,083 M items/sec |
