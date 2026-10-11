# Parallel Computing Mini-Project: OpenMP vs MPI for Data Processing

**Topic #9:** OpenMP vs MPI for Data Processing (OpenMP + MPI)  
**Main Task:** Perform the same simple dataset operation using OpenMP and MPI and compare results.  
**Academic Year:** 2026-2027  
**Course:** Parallel Computing Lab Evaluation  

---

## Executive Summary

This project conducts an empirical and theoretical performance evaluation comparing **OpenMP** (Shared Memory Multi-threading) and **MPI** (Distributed Memory Message Passing) for large-scale data processing workloads. We implement a dual-phase data processing pipeline consisting of:
1. **Phase 1 (Reductions):** Global statistical aggregation (Sum, Sum of Squares, Minimum, Maximum, Mean, Variance, and Standard Deviation).
2. **Phase 2 (Dependent Conditional Filtering):** Counting elements exceeding the calculated global dataset mean ($x_i > \mu$).

Both parallel models were systematically evaluated against a single-threaded **Sequential Baseline** across four dataset sizes ($5\times 10^6$, $10\times 10^7$, $25\times 10^7$, and $50\times 10^7$ double-precision elements) and five core configurations (1, 2, 4, 8, and 12 threads/processes) on an 8-core / 12-thread Intel Core i5-13420H workstation.

The empirical results show that:
- For 50 Million elements, OpenMP achieves up to **3.27x speedup** (25.10 ms vs 81.97 ms baseline).
- MPI achieves up to **3.42x speedup** (24.00 ms vs 81.97 ms baseline), processing over **2.08 Billion elements per second**.
- OpenMP delivers lower synchronization overhead for shared-memory workstations, while MPI provides superior cache isolation and memory scalability via 1D domain decomposition.

---

## Checkpoint 1: Problem Definition, Sequential Algorithm & Parallel Design (2 Marks)

### 1.1 Problem Definition & Mathematical Formulation
Modern data analytics architectures frequently process massive telemetry streams, financial transactions, and sensor time-series datasets. We consider a dataset $D = \{x_0, x_1, \dots, x_{N-1}\}$ of $N$ double-precision IEEE-754 floating-point values ($x_i \in [0.00, 9999.99]$).

The workload executes two passes over $D$:
- **Sum:** $S = \sum_{i=0}^{N-1} x_i$
- **Sum of Squares:** $S_{sq} = \sum_{i=0}^{N-1} x_i^2$
- **Extrema:** $x_{min} = \min_{0 \le i < N} x_i, \quad x_{max} = \max_{0 \le i < N} x_i$
- **Mean:** $\mu = \frac{S}{N}$
- **Variance:** $\sigma^2 = \frac{S_{sq}}{N} - \mu^2$
- **Standard Deviation:** $\sigma = \sqrt{\max(0, \sigma^2)}$
- **Conditional Filter Count:** $C = \sum_{i=0}^{N-1} \mathbb{I}(x_i > \mu)$ where $\mathbb{I}$ is the indicator function.

### 1.2 Computational Complexity
- **Sequential Work Complexity:** $\mathcal{W}(N) = \mathcal{O}(N)$ arithmetic operations.
- **Sequential Space Complexity:** $\mathcal{O}(N)$ memory storage.
- **Parallel Work & Span:**
  - Parallel work: $\mathcal{O}(N)$
  - Parallel span (critical path): $\mathcal{D}(N) = \mathcal{O}\left(\frac{N}{P} + \log P\right)$ where $P$ is the number of worker cores.

### 1.3 Sequential Algorithm Design
The sequential implementation processes array $D$ in a single thread:
1. Pre-allocates contiguous heap buffer for $N$ doubles ($8N$ bytes).
2. Executes Pass 1 in a linear scan, accumulating scalar variables into hardware registers.
3. Derives $\mu, \sigma^2, \sigma$ in $\mathcal{O}(1)$ time.
4. Executes Pass 2 to count values $x_i > \mu$.
5. Employs `QueryPerformanceCounter` on Windows to record sub-microsecond timing.

### 1.4 Parallel Architectural Comparison
| Architectural Dimension | OpenMP Model | MPI Model |
| :--- | :--- | :--- |
| **Target Architecture** | Symmetric Multiprocessing (SMP) / Multi-core | Distributed Cluster / Multi-node HPC / Cloud |
| **Address Space** | Uniform Shared Memory | Disjoint / Isolated Local Address Space per Rank |
| **Communication** | Implicit via shared cache/RAM reads & writes | Explicit message passing via library APIs |
| **Data Partitioning** | Compiler handles thread scheduling across iterations | Manual 1D domain decomposition across ranks |
| **Hardware Boundary** | Single motherboard / node RAM limit | Network interconnect (InfiniBand / TCP / Ethernet) |

---

## Checkpoint 2: Working Parallel Implementation (2 Marks)

### 2.1 OpenMP Parallel Implementation
The OpenMP implementation leverages compiler pragmas to partition loop iterations across POSIX/Windows worker threads:
- **Directive Reductions:**
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
- **Implicit Barrier & Shared State:**
  Once Pass 1 finishes, the OpenMP runtime executes an implicit barrier, merging each thread's private accumulator into the master variables. The mean $\mu$ is computed and instantly visible to all threads in shared memory without communication overhead.
- **Pass 2 Parallel Filtering:**
  ```c
  #pragma omp parallel for reduction(+:count_above_mean) schedule(static)
  for (long long i = 0; i < N; i++) {
      if (data[i] > mean) count_above_mean++;
  }
  ```

### 2.2 MPI Distributed Implementation
The MPI implementation operates on a shared-nothing distributed architecture:
- **1D Domain Decomposition:**
  Each process with rank $r \in \{0, \dots, P-1\}$ calculates its disjoint partition:
  $$\text{base\_chunk} = \lfloor N / P \rfloor, \quad \text{rem} = N \pmod P$$
  $$\text{local\_count} = \text{base\_chunk} + (r < \text{rem} ? 1 : 0)$$
  $$\text{local\_start} = r \times \text{base\_chunk} + (r < \text{rem} ? r : \text{rem})$$
- **Collective Reduction for Phase 1:**
  Local sums and extrema are reduced to Rank 0 using collective tree algorithms:
  ```c
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  ```
- **Collective Broadcast for Phase 2 Dependency:**
  Because processes cannot read Rank 0's memory, Rank 0 computes $\mu = S / N$ and broadcasts it to all ranks:
  ```c
  MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  ```
- **Distributed Filter Reduction:**
  Each rank counts its local matches and invokes:
  ```c
  MPI_Reduce(&local_above_count, &global_above_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  ```

### 2.3 Verification of Numerical Correctness
All three implementations were validated to produce bitwise identical mathematical results:
- **Sum:** `249997015134.1200`
- **Mean:** `4999.9403`
- **Minimum:** `0.0000`
- **Maximum:** `9999.9900`
- **Standard Deviation:** `2886.7483`
- **Count > Mean:** `25000772`

---

## Checkpoint 3: Benchmark Runs Across Dataset Sizes & Workers (2 Marks)

### 3.1 Experimental Test Matrix
- **Dataset Sizes Tested:**
  1. $5\times 10^6$ (5 Million elements $\approx 40$ MB)
  2. $10\times 10^6$ (10 Million elements $\approx 80$ MB)
  3. $25\times 10^6$ (25 Million elements $\approx 200$ MB)
  4. $50\times 10^6$ (50 Million elements $\approx 400$ MB)
- **Worker Concurrency Grid:** $P \in \{1, 2, 4, 8, 12\}$ threads/processes.
- **Reps:** 3 independent runs per configuration; average execution time recorded.

### 3.2 Summary Benchmark Data Table (50 Million Elements)
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

## Checkpoint 4: Visualization, Graphs & Quantitative Analysis (2 Marks)

### 4.1 Execution Time vs Dataset Size
![Execution Time vs Dataset Size](../graphs/execution_time_vs_datasize.png)

**Key Observations:**
1. Execution time scales linearly with dataset size across all paradigms.
2. At 50M elements, parallelization reduces wall-clock latency from 81.97 ms down to 24.00 ms (MPI) and 25.10 ms (OpenMP).
3. The speedup margin widens significantly as dataset size increases, verifying that the parallel compute workload dominates fixed launch overhead.

### 4.2 Speedup Analysis vs Amdahl's Law
![Speedup vs Cores](../graphs/speedup_vs_cores.png)

**Speedup Formula:**
$$S(P) = \frac{T_{sequential}}{T_{parallel}(P)}$$

**Amdahl's Law Modeling:**
$$S_{Amdahl}(P) = \frac{1}{s + \frac{1 - s}{P}}$$
Where $s$ is the serial fraction (initialization, scalar variance calculation, barrier synchronization).
- Scaling is near-linear up to 4 cores ($S \approx 2.6x - 3.0x$).
- Beyond 8 cores, speedup levels off at $\sim 3.3x - 3.4x$ due to:
  1. **Memory Bus Saturation:** High-volume streaming of double-precision floats saturates L3 cache and dual-channel RAM bandwidth.
  2. **Heterogeneous Core Architecture:** The Intel Core i5-13420H features 4 Performance Cores (P-cores) and 4 Efficient Cores (E-cores). Hyperthreads share physical execution pipelines.

### 4.3 Parallel Efficiency Analysis
![Parallel Efficiency vs Cores](../graphs/efficiency_vs_cores.png)

**Efficiency Formula:**
$$E(P) = \frac{S(P)}{P} \times 100\%$$
- At $P = 2$, efficiency remains high: **89.4% (MPI)** and **86.8% (OpenMP)**.
- At $P = 4$, efficiency stays strong at **75.6% (MPI)**.
- At $P = 12$, efficiency drops to $\sim 26\% - 28\%$, which is characteristic of CPU-bound and memory-bound hybrid SMT architectures.

### 4.4 Processing Throughput Scalability
![Throughput Scalability](../graphs/throughput_scalability.png)

- Baseline sequential throughput: **610 Million elements/sec**.
- Peak OpenMP throughput: **1,992 Million elements/sec** (at 8 threads).
- Peak MPI throughput: **2,083 Million elements/sec** (at 12 processes).
- Demonstrates a **3.4x overall throughput enhancement**.

---

## Checkpoint 5: Technical Viva Preparation & Guide Q&A (2 Marks)

### Q1: Why did you choose OpenMP and MPI for comparison?
> **Answer:** OpenMP represents the industry standard for **Shared-Memory Multi-threading** (SMP), utilizing compiler pragmas and shared memory address space. MPI represents the standard for **Distributed-Memory Computing** (clusters, HPC supercomputers), using explicit message passing. Comparing them highlights trade-offs in communication latency, memory overhead, and programming complexity.

### Q2: Why did MPI require an `MPI_Bcast` between Pass 1 and Pass 2?
> **Answer:** In OpenMP, all threads share the same address space; once Pass 1 computes the dataset mean $\mu$, all threads can immediately access it from shared RAM. In MPI, each process has an isolated address space. Rank 0 calculates the global mean after `MPI_Reduce`, so it must explicitly broadcast $\mu$ to all other ranks via `MPI_Bcast` before they can perform local filtering in Pass 2.

### Q3: Why does speedup saturate beyond 8 cores?
> **Answer:** Two primary factors:
> 1. **Hardware Architecture:** The host CPU (Intel Core i5-13420H) has 8 physical cores and 12 logical threads. Hyperthreads share pipeline units, yielding sub-linear gains.
> 2. **Memory Bandwidth Bottleneck:** Linear reduction and filtering are memory-bound streaming kernels. Once the memory bus bandwidth is fully saturated, adding more worker threads cannot fetch data faster from RAM.

### Q4: How does OpenMP handle race conditions during reductions?
> **Answer:** OpenMP's `reduction(+:variable)` clause allocates private accumulator copies for each thread. During loop execution, each thread updates its own local accumulator without contention. Upon loop completion, the runtime automatically merges private copies into the shared master variable using hardware atomic instructions or a tree reduction.

### Q5: What collective communication algorithms are used by `MPI_Reduce` and `MPI_Bcast`?
> **Answer:** High-performance MPI implementations use binomial spanning trees or recursive doubling algorithms with $\mathcal{O}(\log P)$ communication complexity, rather than a naive $\mathcal{O}(P)$ star topology.

---

## Conclusion

This project successfully met all evaluation checkpoints:
1. **Checkpoint 1:** Complete mathematical definition, sequential baseline, and architectural design.
2. **Checkpoint 2:** Robust C implementations for OpenMP and MPI with bitwise identical numerical outputs.
3. **Checkpoint 3:** Full empirical benchmarking across dataset sizes (5M to 50M) and worker counts (1 to 12).
4. **Checkpoint 4:** Publication-quality graphical analysis of runtime, speedup, efficiency, and throughput.
5. **Checkpoint 5:** Thorough technical viva preparation and architectural answers for presentation.
