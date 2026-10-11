#!/usr/bin/env python3
"""
Graph Generation Script for Parallel Computing Mini-Project
Topic: OpenMP vs MPI for Data Processing

Generates publication-quality graphs:
1. execution_time_vs_datasize.png
2. speedup_vs_cores.png
3. efficiency_vs_cores.png
4. throughput_scalability.png
5. execution_time_breakdown.png (Multi-core comparison)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
GRAPHS_DIR = os.path.join(WORKSPACE_DIR, "graphs")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "benchmark_summary.csv")

os.makedirs(GRAPHS_DIR, exist_ok=True)

# Set high-quality visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

def load_data():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}. Please run run_benchmarks.py first.")
    return pd.read_csv(SUMMARY_CSV)

def plot_execution_time_vs_datasize(df):
    """Plot Execution Time vs Data Size at 4 Cores/Threads"""
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    # Filter for 4 cores
    sub_omp = df[(df["Paradigm"] == "OpenMP") & (df["Cores"] == 4)].sort_values("N")
    sub_mpi = df[(df["Paradigm"] == "MPI") & (df["Cores"] == 4)].sort_values("N")
    sub_seq = df[(df["Paradigm"] == "Sequential") & (df["Cores"] == 1)].sort_values("N")
    
    data_labels = sub_seq["N_Label"].tolist()
    x = np.arange(len(data_labels))
    width = 0.25

    plt.bar(x - width, sub_seq["AvgTimeMs"], width, label="Sequential (1 Core)", color="#e74c3c", alpha=0.9, edgecolor="black", linewidth=0.5)
    plt.bar(x, sub_omp["AvgTimeMs"], width, label="OpenMP (4 Threads)", color="#2ecc71", alpha=0.9, edgecolor="black", linewidth=0.5)
    plt.bar(x + width, sub_mpi["AvgTimeMs"], width, label="MPI (4 Processes)", color="#3498db", alpha=0.9, edgecolor="black", linewidth=0.5)

    for i in range(len(x)):
        plt.text(x[i] - width, sub_seq["AvgTimeMs"].iloc[i] + 1.0, f"{sub_seq['AvgTimeMs'].iloc[i]:.1f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(x[i], sub_omp["AvgTimeMs"].iloc[i] + 1.0, f"{sub_omp['AvgTimeMs'].iloc[i]:.1f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(x[i] + width, sub_mpi["AvgTimeMs"].iloc[i] + 1.0, f"{sub_mpi['AvgTimeMs'].iloc[i]:.1f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.title("Execution Time Comparison vs Dataset Size (4 Cores/Workers)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Dataset Size (Number of Elements)", fontsize=11, fontweight='bold')
    plt.ylabel("Execution Time (milliseconds) - Lower is Better", fontsize=11, fontweight='bold')
    plt.xticks(x, [f"{lbl}\n({n:,})" for lbl, n in zip(data_labels, sub_seq["N"])], fontsize=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(GRAPHS_DIR, "execution_time_vs_datasize.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved Graph] {out_path}")

def plot_speedup_vs_cores(df):
    """Plot Speedup vs Cores for Largest Dataset (50M elements)"""
    max_n = df["N"].max()
    sub_omp = df[(df["Paradigm"] == "OpenMP") & (df["N"] == max_n)].sort_values("Cores")
    sub_mpi = df[(df["Paradigm"] == "MPI") & (df["N"] == max_n)].sort_values("Cores")
    
    cores = sub_omp["Cores"].tolist()
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(cores, cores, 'k--', label="Ideal Linear Speedup (S = P)", linewidth=1.8, alpha=0.7)
    plt.plot(cores, sub_omp["Speedup"], 'o-', label="OpenMP Speedup", color="#27ae60", linewidth=2.5, markersize=8)
    plt.plot(cores, sub_mpi["Speedup"], 's-', label="MPI Speedup", color="#2980b9", linewidth=2.5, markersize=8)

    for c, s in zip(cores, sub_omp["Speedup"]):
        plt.annotate(f"{s:.2f}x", (c, s), textcoords="offset points", xytext=(-10, 8), fontsize=9, fontweight='bold', color="#27ae60")
    for c, s in zip(cores, sub_mpi["Speedup"]):
        plt.annotate(f"{s:.2f}x", (c, s), textcoords="offset points", xytext=(-10, -14), fontsize=9, fontweight='bold', color="#2980b9")

    plt.title(f"Speedup vs Number of Workers (Dataset: {max_n//1_000_000}M Elements)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Threads / Processes (P)", fontsize=11, fontweight='bold')
    plt.ylabel("Speedup (S = T_seq / T_par) - Higher is Better", fontsize=11, fontweight='bold')
    plt.xticks(cores, [str(c) for c in cores], fontsize=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(GRAPHS_DIR, "speedup_vs_cores.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved Graph] {out_path}")

def plot_efficiency_vs_cores(df):
    """Plot Parallel Efficiency vs Cores (50M elements)"""
    max_n = df["N"].max()
    sub_omp = df[(df["Paradigm"] == "OpenMP") & (df["N"] == max_n)].sort_values("Cores")
    sub_mpi = df[(df["Paradigm"] == "MPI") & (df["N"] == max_n)].sort_values("Cores")
    
    cores = sub_omp["Cores"].tolist()

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.axhline(100.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label="Ideal Efficiency (100%)")
    plt.plot(cores, sub_omp["EfficiencyPct"], 'o-', label="OpenMP Efficiency", color="#27ae60", linewidth=2.5, markersize=8)
    plt.plot(cores, sub_mpi["EfficiencyPct"], 's-', label="MPI Efficiency", color="#2980b9", linewidth=2.5, markersize=8)

    for c, e in zip(cores, sub_omp["EfficiencyPct"]):
        plt.annotate(f"{e:.1f}%", (c, e), textcoords="offset points", xytext=(-10, 8), fontsize=9, fontweight='bold', color="#27ae60")
    for c, e in zip(cores, sub_mpi["EfficiencyPct"]):
        plt.annotate(f"{e:.1f}%", (c, e), textcoords="offset points", xytext=(-10, -14), fontsize=9, fontweight='bold', color="#2980b9")

    plt.title(f"Parallel Efficiency vs Worker Count (Dataset: {max_n//1_000_000}M Elements)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Threads / Processes (P)", fontsize=11, fontweight='bold')
    plt.ylabel("Parallel Efficiency (%) - E = (S / P) * 100", fontsize=11, fontweight='bold')
    plt.xticks(cores, [str(c) for c in cores], fontsize=10)
    plt.ylim(0, 115)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(GRAPHS_DIR, "efficiency_vs_cores.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved Graph] {out_path}")

def plot_throughput_scalability(df):
    """Plot Processing Throughput (Million Items / sec)"""
    max_n = df["N"].max()
    sub_omp = df[(df["Paradigm"] == "OpenMP") & (df["N"] == max_n)].sort_values("Cores")
    sub_mpi = df[(df["Paradigm"] == "MPI") & (df["N"] == max_n)].sort_values("Cores")
    seq_time = df[(df["Paradigm"] == "Sequential") & (df["N"] == max_n)]["AvgTimeMs"].iloc[0]
    seq_throughput = (max_n / (seq_time / 1000.0)) / 1_000_000.0

    cores = sub_omp["Cores"].tolist()

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.axhline(seq_throughput, color='#e74c3c', linestyle=':', linewidth=2, label=f"Sequential (1 Core: {seq_throughput:.1f} M/s)")
    plt.plot(cores, sub_omp["ThroughputMItemsPerSec"], 'o-', label="OpenMP Throughput", color="#27ae60", linewidth=2.5, markersize=8)
    plt.plot(cores, sub_mpi["ThroughputMItemsPerSec"], 's-', label="MPI Throughput", color="#2980b9", linewidth=2.5, markersize=8)

    for c, tp in zip(cores, sub_omp["ThroughputMItemsPerSec"]):
        plt.annotate(f"{tp:.0f} M/s", (c, tp), textcoords="offset points", xytext=(-12, 8), fontsize=9, fontweight='bold', color="#27ae60")
    for c, tp in zip(cores, sub_mpi["ThroughputMItemsPerSec"]):
        plt.annotate(f"{tp:.0f} M/s", (c, tp), textcoords="offset points", xytext=(-12, -14), fontsize=9, fontweight='bold', color="#2980b9")

    plt.title(f"Processing Throughput vs Cores (Dataset: {max_n//1_000_000}M Elements)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Threads / Processes (P)", fontsize=11, fontweight='bold')
    plt.ylabel("Throughput (Million Elements / sec) - Higher is Better", fontsize=11, fontweight='bold')
    plt.xticks(cores, [str(c) for c in cores], fontsize=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(GRAPHS_DIR, "throughput_scalability.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved Graph] {out_path}")

def generate_all_graphs():
    df = load_data()
    print("Generating comprehensive evaluation graphs...")
    plot_execution_time_vs_datasize(df)
    plot_speedup_vs_cores(df)
    plot_efficiency_vs_cores(df)
    plot_throughput_scalability(df)
    print("All 4 graphs successfully generated in 'graphs/' folder!")

if __name__ == "__main__":
    generate_all_graphs()
