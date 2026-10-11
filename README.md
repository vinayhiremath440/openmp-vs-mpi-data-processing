# Parallel Computing Mini-Project: OpenMP vs MPI for Data Processing

**Course:** Parallel Computing (Semester 5 Lab Evaluation)  
**Topic #9:** OpenMP vs MPI for Data Processing (`OpenMP + MPI`)  
**Task:** Perform the same simple dataset operation using OpenMP and MPI and compare results.  
**Academic Year:** 2026-2027  

---

## 👥 Team Information
- **Team Number:** Topic 9 / Team 9
- **Student Name:** [Your Name / Team Lead]
- **USN / Roll Number:** [Your Roll / USN Number]
- **Department:** Computer Science & Engineering
- **Guide / Evaluator:** [Guide / Faculty Name]

---

## 🎯 Project Objective

The primary objective of this project is to perform a rigorous comparative performance evaluation between **OpenMP** (Shared-Memory Multi-threading) and **MPI** (Distributed-Memory Message Passing) when executing identical numerical data processing operations on large datasets.

We evaluate:
1. **Execution Time (Latency):** Wall-clock execution time across dataset scales ($5 \times 10^6$ to $50 \times 10^6$ double-precision numbers).
2. **Speedup ($S$):** Scaling factor relative to the single-threaded sequential baseline across 1 to 12 threads/processes.
3. **Parallel Efficiency ($E$):** Resource utilization efficiency ($S / P \times 100\%$).
4. **Processing Throughput:** Millions of dataset elements processed per second.
5. **Architectural Trade-offs:** Memory footprint, synchronization overhead, cache locality, and programming complexity.

---

## 🧮 Data Processing Workload & Algorithmic Design

The chosen benchmark performs a dual-phase data processing operation typical in statistical computing and data filtering:

### Phase 1: Aggregate Reductions (Statistical Metrics)
Given an array $D = \{x_0, x_1, \dots, x_{N-1}\}$ of $N$ double-precision values ($x_i \in [0.00, 9999.99]$):
- **Sum:** $S = \sum_{i=0}^{N-1} x_i$
- **Sum of Squares:** $S_{sq} = \sum_{i=0}^{N-1} x_i^2$
- **Extrema:** $x_{min} = \min_{0 \le i < N} x_i, \quad x_{max} = \max_{0 \le i < N} x_i$
- **Mean:** $\mu = \frac{S}{N}$
- **Variance:** $\sigma^2 = \frac{S_{sq}}{N} - \mu^2$
- **Standard Deviation:** $\sigma = \sqrt{\max(0, \sigma^2)}$

### Phase 2: Dependent Conditional Filtering
- **Count Elements Exceeding Mean:**  
  $$C = \sum_{i=0}^{N-1} \mathbb{I}(x_i > \mu)$$
  *(where $\mathbb{I}$ is the indicator condition)*

> **Why this operation is ideal for evaluation:**  
> Phase 2 has a strict data dependency on Phase 1 ($\mu$ must be completely resolved before filtering begins). This directly demonstrates how OpenMP handles implicit memory sharing versus how MPI requires an explicit collective broadcast (`MPI_Bcast`) to disseminate the global mean to worker ranks.

---

## 🏗️ Repository Structure

This repository strictly complies with the evaluation guidelines:

```text
exp-name-parallel-computing/
|-- README.md                    # Project documentation, build instructions & results summary
|-- build.bat                    # Automated Windows build script for all 3 targets
|-- run_benchmarks.py            # Automated multi-core benchmarking suite across data sizes
|-- generate_graphs.py           # Publication-quality plotting script for metrics
|-- bin/                         # Compiled executables & runtime DLLs
|   |-- sequential.exe           # Sequential baseline binary
|   |-- openmp_processing.exe    # OpenMP parallel binary
|   `-- mpi_processing.exe       # MPI distributed binary
|-- src/                         # Source code files
|   |-- sequential.c             # C sequential implementation
|   |-- openmp_processing.c      # C OpenMP multi-threaded implementation
|   `-- mpi_processing.c         # C MPI distributed implementation
|-- data/                        # Dataset scripts
|   `-- generate_dataset.py      # Python dataset generator (CSV & binary format)
|-- results/                     # Raw and summary experimental timing data
|   |-- raw_benchmark_runs.csv   # Raw execution logs (all repetitions)
|   |-- benchmark_summary.csv    # Aggregated results (averages, speedup, efficiency)
|   `-- benchmark_summary.txt    # Human-readable formatted summary table
|-- graphs/                      # High-resolution comparison plots
|   |-- execution_time_vs_datasize.png
|   |-- speedup_vs_cores.png
|   |-- efficiency_vs_cores.png
|   `-- throughput_scalability.png
|-- report/                      # Academic Project Report
|   |-- Lab_Evaluation_Report.md # Full Markdown report
|   |-- Lab_Evaluation_Report.html # Styled HTML report with embedded visuals
|   `-- Lab_Evaluation_Report.pdf  # Print-ready academic PDF report
`-- presentation/                # Lab Evaluation Slide Deck
    `-- OpenMP_vs_MPI_Data_Processing.pptx  # Single comprehensive PPTX presentation
```

---

## ⚙️ Hardware & Software Environment

- **Processor (CPU):** 13th Gen Intel Core i5-13420H (8 Cores, 12 Logical Processors, 2.10 GHz Base, 4.60 GHz Turbo)
- **Host Memory:** 16 GB RAM
- **Operating System:** Microsoft Windows 11 (64-bit)
- **C Compiler:** MinGW-w64 GCC 8.1.0 with `-O3` optimization
- **MPI Framework:** Microsoft MPI (MS-MPI) v10.1 (bundled in `tools/mpi/`)
- **Python Environment:** Python 3.14 with `pandas`, `numpy`, `matplotlib`, and `python-pptx`

---

## 🚀 How to Build and Run

### 1. Build Executables
Run the batch script from the repository root:
```cmd
build.bat
```
This compiles `src/sequential.c`, `src/openmp_processing.c`, and `src/mpi_processing.c` with `-O3` optimization into `bin/` and bundles required runtime DLLs.

### 2. Run Individual Executables

#### A. Sequential Baseline
```cmd
bin\sequential.exe [dataset_size]
```
*Example:* `bin\sequential.exe 10000000`

#### B. OpenMP Parallel
```cmd
bin\openmp_processing.exe [dataset_size] [num_threads]
```
*Example (10M elements on 4 threads):* `bin\openmp_processing.exe 10000000 4`

#### C. MPI Distributed
```cmd
tools\mpi\bin\mpiexec.exe -n [num_processes] bin\mpi_processing.exe [dataset_size]
```
*Example (10M elements on 4 processes):*  
```cmd
tools\mpi\bin\mpiexec.exe -n 4 bin\mpi_processing.exe 10000000
```

### 3. Run Automated Benchmark Suite
To re-run all combinations (5M to 50M elements, 1 to 12 workers, 3 iterations each):
```cmd
python run_benchmarks.py
```
This updates `results/benchmark_summary.csv` and `results/benchmark_summary.txt`.

### 4. Regenerate Graphs
```cmd
python generate_graphs.py
```

### 5. Regenerate Presentation & Reports
```cmd
python tools\build_presentation.py
python tools\generate_report.py
```

---

## 📊 Results Summary

All runs were repeated **3 times**; average execution times are shown below.

### Execution Results (50 Million Elements - Largest Scale)

| Paradigm | Workers ($P$) | Execution Time (ms) | Speedup ($S$) | Efficiency ($E$) | Throughput (M-items/sec) |
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

### Execution Time vs Dataset Size (at 4 Workers)

| Dataset Size | Sequential Baseline | OpenMP (4 Threads) | MPI (4 Processes) |
| :---: | :---: | :---: | :---: |
| **5 Million (5M)** | 7.699 ms | **2.782 ms** (2.77x) | **3.368 ms** (2.29x) |
| **10 Million (10M)** | 15.690 ms | **6.774 ms** (2.32x) | **6.983 ms** (2.25x) |
| **25 Million (25M)** | 40.763 ms | **14.197 ms** (2.87x) | **14.183 ms** (2.87x) |
| **50 Million (50M)** | 81.965 ms | **31.450 ms** (2.61x) | **27.116 ms** (3.02x) |

---

## 📈 Visual Performance Analysis

### 1. Execution Time Comparison
![Execution Time vs Dataset Size](graphs/execution_time_vs_datasize.png)
*Figure 1: Wall-clock execution time vs Dataset Size. Both OpenMP and MPI consistently cut execution latency across all dataset scales.*

### 2. Speedup vs Number of Workers
![Speedup vs Cores](graphs/speedup_vs_cores.png)
*Figure 2: Scaling speedup vs Worker Concurrency. Near-linear scaling up to 4 cores, peaking at 8-12 cores at ~3.4x speedup due to memory bus saturation.*

### 3. Parallel Efficiency
![Parallel Efficiency vs Cores](graphs/efficiency_vs_cores.png)
*Figure 3: Resource efficiency remains high (>75-89%) at 2-4 cores, dropping smoothly as SMT hyperthreads share core execution pipelines.*

### 4. Throughput Scalability
![Throughput Scalability](graphs/throughput_scalability.png)
*Figure 4: Aggregate processing throughput increases from 610 Million elements/sec (Sequential) to over 2.08 Billion elements/sec (MPI 12 Procs).*

---

## ⚖️ Architectural Comparison: OpenMP vs MPI

| Feature | OpenMP | MPI | Sequential |
| :--- | :--- | :--- | :--- |
| **Model** | Shared Memory Multi-threading | Distributed Memory Message Passing | Single-threaded |
| **Address Space** | Uniform shared memory | Private isolated memory per rank | Single process memory |
| **Communication** | Implicit (Shared RAM) | Explicit (`MPI_Reduce`, `MPI_Bcast`) | None |
| **Synchronization** | Directives (`#pragma omp barrier`) | Collectives (`MPI_Barrier`, etc.) | None |
| **Programming Difficulty** | Low (Incremental pragmas) | Moderate/High (Domain decomposition) | Minimal |
| **Peak 50M Speedup** | **3.27x** (8 Threads) | **3.42x** (12 Processes) | 1.00x |
| **Scaling Limit** | Limited by single node RAM & cores | Scales across supercomputer clusters | Single CPU core |

---

## 🎓 Checkpoint 5: Technical Viva Voce Questions & Answers

### Q1: What is the fundamental architectural difference between OpenMP and MPI?
> **Answer:**  
> **OpenMP** follows a **Shared Memory** model where all threads reside within the same operating system process and communicate implicitly via reads and writes to shared virtual memory.  
> **MPI** follows a **Distributed Memory (Shared-Nothing)** model where each process maintains its own private address space and communicates explicitly over a message-passing layer using function calls (`MPI_Send`, `MPI_Recv`, `MPI_Reduce`, `MPI_Bcast`).

### Q2: Why did MPI require `MPI_Bcast` between Pass 1 and Pass 2?
> **Answer:**  
> In Pass 1, all MPI ranks perform local reductions and reduce their sums to Rank 0 (`MPI_Reduce`). Rank 0 computes the dataset mean $\mu = \text{Sum} / N$. However, because MPI processes cannot inspect Rank 0's private memory, Rank 0 must broadcast $\mu$ to all other ranks via `MPI_Bcast` so they can filter their local partitions in Pass 2. In OpenMP, this broadcast is unnecessary because $\mu$ is stored in shared RAM, readable by all threads immediately.

### Q3: Why does speedup plateau beyond 8 cores on this CPU?
> **Answer:**  
> 1. **Core Topology:** The Intel Core i5-13420H has 8 physical cores and 12 logical threads (SMT/hyperthreading). Hyperthreads share physical ALU and vector execution units, providing lower incremental gains than physical cores.  
> 2. **Memory Bandwidth Bottleneck:** Array reductions and filtering are memory-bound streaming operations. Once the CPU's memory controllers and bus bandwidth are saturated, adding more worker threads cannot accelerate data transfer from RAM.

### Q4: How does OpenMP prevent race conditions in parallel reductions?
> **Answer:**  
> Through the `reduction(+:variable)` clause. The compiler generates thread-private accumulators for each thread. Each thread accumulates into its local variable without locks. At the end of the parallel loop, OpenMP merges the private accumulators into the shared variable using atomic hardware instructions or a binary reduction tree.

### Q5: What is Amdahl's Law, and how does it relate to your measured results?
> **Answer:**  
> Amdahl's Law states that maximum speedup is bounded by the serial fraction $s$:
> $$S_{max} = \frac{1}{s + \frac{1 - s}{P}}$$
> In our code, serial portions include process creation, global mean division, and barrier synchronization. As $P \to \infty$, speedup asymptotically approaches $1/s$.

---

## 📦 Submission Deliverables Checklist
- [x] **Checkpoint 1:** Mathematical definition, sequential algorithm & parallel architecture design.
- [x] **Checkpoint 2:** Tested, bitwise-identical C implementations for Sequential, OpenMP, and MPI.
- [x] **Checkpoint 3:** Complete benchmark runs across 5M, 10M, 25M, and 50M elements on 1 to 12 workers.
- [x] **Checkpoint 4:** 4 high-resolution graphs for execution time, speedup, efficiency, and throughput.
- [x] **Checkpoint 5:** Full technical viva preparation guide and final demonstration readiness.
- [x] **Presentation:** 13-slide PowerPoint slide deck in `presentation/OpenMP_vs_MPI_Data_Processing.pptx`.
- [x] **Report:** Academic Markdown report, HTML report, and PDF report in `report/`.
