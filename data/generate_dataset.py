#!/usr/bin/env python3
"""
Dataset Generator for Parallel Computing Evaluation
Topic: OpenMP vs MPI for Data Processing

Generates synthetic numerical floating-point datasets simulating sensor/time-series data.
Can generate both CSV format and binary double-precision format.
"""

import sys
import os
import argparse
import numpy as np

def generate_csv_dataset(filename, num_records, seed=42):
    print(f"Generating CSV dataset with {num_records:,} elements -> {filename}")
    np.random.seed(seed)
    # Generate random data with normal distribution + uniform mix
    data = np.round(np.random.uniform(0.0, 9999.99, size=num_records), 2)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("id,value\n")
        for i, val in enumerate(data):
            f.write(f"{i},{val}\n")
    print(f"Successfully generated {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")

def generate_binary_dataset(filename, num_records, seed=42):
    print(f"Generating binary double dataset with {num_records:,} elements -> {filename}")
    np.random.seed(seed)
    data = np.random.uniform(0.0, 9999.99, size=num_records).astype(np.float64)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    data.tofile(filename)
    print(f"Successfully generated {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark datasets")
    parser.add_argument("--size", type=int, default=1000000, help="Number of data elements (default: 1,000,000)")
    parser.add_argument("--format", choices=["csv", "bin"], default="csv", help="Output format")
    parser.add_argument("--output", type=str, default="data/sample_dataset.csv", help="Output file path")
    
    args = parser.parse_args()
    if args.format == "csv":
        generate_csv_dataset(args.output, args.size)
    else:
        generate_binary_dataset(args.output, args.size)
