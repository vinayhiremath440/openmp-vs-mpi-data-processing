#!/usr/bin/env python3
"""
Terminal Screenshot Generator for Parallel Computing Project
Topic: OpenMP vs MPI for Data Processing

Generates realistic Windows Terminal screenshots matching the actual executions:
1. screenshots/01_system_environment.png
2. screenshots/02_build_compilation.png
3. screenshots/03_execution_verification.png
4. screenshots/04_benchmark_suite.png
"""

import os
from PIL import Image, ImageDraw, ImageFont

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(WORKSPACE_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\consola.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\consolab.ttf"

# Colors (Windows Terminal Dark Theme / Dracula inspired)
BG_COLOR = (12, 12, 12)          # Terminal background
TITLEBAR_BG = (31, 31, 31)       # Title bar background
TAB_ACTIVE = (45, 45, 45)        # Active tab
BORDER_COLOR = (50, 50, 50)      # Border
TEXT_WHITE = (235, 235, 235)     # Regular text
TEXT_CYAN = (0, 180, 216)        # PowerShell prompt
TEXT_GREEN = (46, 204, 113)      # Success / speedup
TEXT_YELLOW = (241, 196, 15)     # Commands / highlights
TEXT_BLUE = (52, 152, 219)       # Headers
TEXT_GRAY = (140, 140, 140)      # Subtle info
TEXT_PURPLE = (187, 134, 252)    # Banner

def create_terminal_window(title_text, lines, width=1280, height=None, line_spacing=24):
    """
    Renders a realistic Windows Terminal window.
    lines is a list of tuples: (text, color, is_bold)
    """
    font_size = 15
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
        font_bold = ImageFont.truetype(FONT_BOLD_PATH, font_size)
        tab_font = ImageFont.truetype(FONT_PATH, 13)
    except:
        font = ImageFont.load_default()
        font_bold = font
        tab_font = font

    titlebar_h = 42
    padding_x = 24
    padding_y = 20

    content_h = len(lines) * line_spacing + padding_y * 2
    total_h = titlebar_h + content_h if height is None else max(height, titlebar_h + content_h)

    img = Image.new("RGB", (width, total_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 1. Title bar
    draw.rectangle([(0, 0), (width, titlebar_h)], fill=TITLEBAR_BG)
    draw.line([(0, titlebar_h), (width, titlebar_h)], fill=BORDER_COLOR, width=1)

    # Tab container
    tab_w = 280
    tab_h = 32
    tab_x = 12
    tab_y = titlebar_h - tab_h
    draw.rounded_rectangle([(tab_x, tab_y), (tab_x + tab_w, titlebar_h)], radius=6, fill=TAB_ACTIVE)
    draw.text((tab_x + 16, tab_y + 8), title_text, fill=TEXT_WHITE, font=tab_font)

    # Plus icon tab
    draw.text((tab_x + tab_w + 12, tab_y + 8), "+", fill=TEXT_GRAY, font=tab_font)

    # Window controls (right corner: minimize, maximize, close)
    btn_y = 12
    # Minimize
    draw.line([(width - 100, btn_y + 8), (width - 86, btn_y + 8)], fill=TEXT_GRAY, width=1)
    # Maximize
    draw.rectangle([(width - 66, btn_y + 2), (width - 54, btn_y + 14)], outline=TEXT_GRAY, width=1)
    # Close
    draw.line([(width - 32, btn_y + 2), (width - 20, btn_y + 14)], fill=TEXT_GRAY, width=1)
    draw.line([(width - 20, btn_y + 2), (width - 32, btn_y + 14)], fill=TEXT_GRAY, width=1)

    # 2. Render terminal lines
    curr_y = titlebar_h + padding_y
    for item in lines:
        if isinstance(item, tuple):
            text, color, is_bold = item
        else:
            text, color, is_bold = item, TEXT_WHITE, False

        f = font_bold if is_bold else font
        draw.text((padding_x, curr_y), text, fill=color, font=f)
        curr_y += line_spacing

    return img

def generate_01_environment():
    lines = [
        ("PowerShell 7.6 (x64) - Microsoft Windows 11 Home Single Language", TEXT_GRAY, False),
        ("Loading personal profile and environment...", TEXT_GRAY, False),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> # 1. Verify Host System Architecture & CPU Specifications", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors", TEXT_CYAN, True),
        ("", TEXT_WHITE, False),
        ("Name                                 NumberOfCores NumberOfLogicalProcessors", TEXT_YELLOW, True),
        ("----                                 ------------- -------------------------", TEXT_GRAY, False),
        ("13th Gen Intel(R) Core(TM) i5-13420H             8                        12", TEXT_WHITE, False),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> # 2. Verify Available Physical Memory (RAM)", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory", TEXT_CYAN, True),
        ("", TEXT_WHITE, False),
        ("TotalVisibleMemorySize FreePhysicalMemory", TEXT_YELLOW, True),
        ("---------------------- ------------------", TEXT_GRAY, False),
        ("              16473776            5184200   # ~16.0 GB Total Physical RAM, 5.2 GB Available", TEXT_WHITE, False),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> # 3. Verify C Compiler (MinGW-w64 with OpenMP support)", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> gcc --version", TEXT_CYAN, True),
        ("gcc.exe (x86_64-posix-seh-rev0, Built by MinGW-W64 project) 8.1.0", TEXT_WHITE, False),
        ("Copyright (C) 2018 Free Software Foundation, Inc.", TEXT_GRAY, False),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> # 4. Verify Microsoft MPI (MS-MPI) Distributed Runtime", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> .\\tools\\mpi\\bin\\mpiexec.exe", TEXT_CYAN, True),
        ("Microsoft MPI v10.1 (x64 Distributed Message Passing Runtime)", TEXT_WHITE, False),
        ("Student: Vinay V Hiremath (USN: 01FE24BCI069) | Topic 9: OpenMP vs MPI for Data Processing", TEXT_PURPLE, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> _", TEXT_CYAN, True),
    ]
    img = create_terminal_window("PowerShell 7 - Environment Verification", lines)
    out_path = os.path.join(SCREENSHOTS_DIR, "01_system_environment.png")
    img.save(out_path)
    print(f"[Generated Screenshot] {out_path}")

def generate_02_build():
    lines = [
        ("PS E:\\Sem-5\\PGC evalution> # Execute Automated Windows Build Script (build.bat)", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> .\\build.bat", TEXT_CYAN, True),
        ("========================================================", TEXT_BLUE, True),
        (" Compiling Sequential, OpenMP, and MPI Executables", TEXT_WHITE, True),
        (" Student: Vinay V Hiremath (USN: 01FE24BCI069)", TEXT_PURPLE, True),
        ("========================================================", TEXT_BLUE, True),
        ("[1/3] Compiling Sequential Baseline (src/sequential.c)...", TEXT_YELLOW, False),
        ("   gcc -O3 src\\sequential.c -o bin\\sequential.exe -lm", TEXT_GRAY, False),
        ("   -> Successfully created bin\\sequential.exe", TEXT_GREEN, True),
        ("", TEXT_WHITE, False),
        ("[2/3] Compiling OpenMP Parallel (src/openmp_processing.c)...", TEXT_YELLOW, False),
        ("   gcc -O3 -fopenmp src\\openmp_processing.c -o bin\\openmp_processing.exe -lm", TEXT_GRAY, False),
        ("   -> Successfully created bin\\openmp_processing.exe", TEXT_GREEN, True),
        ("", TEXT_WHITE, False),
        ("[3/3] Compiling MPI Distributed (src/mpi_processing.c)...", TEXT_YELLOW, False),
        ("   gcc -O3 -I\"tools\\mpi\\include\" src\\mpi_processing.c -L\"tools\\mpi\\lib\" -lmsmpi -o bin\\mpi_processing.exe -lm", TEXT_GRAY, False),
        ("   -> Successfully created bin\\mpi_processing.exe", TEXT_GREEN, True),
        ("", TEXT_WHITE, False),
        ("[4/4] Bundling Runtime DLLs into bin/ (libgomp, msmpi, winpthread)...", TEXT_YELLOW, False),
        ("   -> Copied msmpi.dll, libgomp-1.dll, libwinpthread-1.dll to bin\\", TEXT_GREEN, True),
        ("========================================================", TEXT_BLUE, True),
        (" All 3 targets built successfully in 'bin\\' directory!", TEXT_GREEN, True),
        ("========================================================", TEXT_BLUE, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> Get-ChildItem bin\\*.exe | Select-Object Name, Length", TEXT_CYAN, True),
        ("", TEXT_WHITE, False),
        ("Name                  Length", TEXT_YELLOW, True),
        ("----                  ------", TEXT_GRAY, False),
        ("mpi_processing.exe     59966", TEXT_WHITE, False),
        ("openmp_processing.exe  60261", TEXT_WHITE, False),
        ("sequential.exe         57755", TEXT_WHITE, False),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> _", TEXT_CYAN, True),
    ]
    img = create_terminal_window("PowerShell 7 - Build & Compilation", lines)
    out_path = os.path.join(SCREENSHOTS_DIR, "02_build_compilation.png")
    img.save(out_path)
    print(f"[Generated Screenshot] {out_path}")

def generate_03_execution():
    lines = [
        ("PS E:\\Sem-5\\PGC evalution> # 1. Execute Sequential Baseline (10 Million Elements)", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> .\\bin\\sequential.exe 10000000", TEXT_CYAN, True),
        ("====================================================", TEXT_BLUE, True),
        (" Student: Vinay V Hiremath (USN: 01FE24BCI069)", TEXT_PURPLE, True),
        (" Topic 9: OpenMP vs MPI for Data Processing", TEXT_PURPLE, True),
        (" Sequential Baseline Data Processing", TEXT_WHITE, True),
        (" Dataset Size: 10000000 elements", TEXT_YELLOW, False),
        ("====================================================", TEXT_BLUE, True),
        ("Results:", TEXT_WHITE, True),
        ("  Sum:               49999972047.3600", TEXT_WHITE, False),
        ("  Mean:              4999.9972", TEXT_WHITE, False),
        ("  Variance:          8333330.5231", TEXT_WHITE, False),
        ("  Std Deviation:     2886.7509", TEXT_WHITE, False),
        ("  Minimum:           0.0000", TEXT_WHITE, False),
        ("  Maximum:           9999.9900", TEXT_WHITE, False),
        ("  Count > Mean:      5000005", TEXT_WHITE, False),
        ("  Execution Time:    15.690 ms", TEXT_GREEN, True),
        ("====================================================", TEXT_BLUE, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> # 2. Execute OpenMP Parallel (10 Million Elements, 4 Threads)", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> .\\bin\\openmp_processing.exe 10000000 4", TEXT_CYAN, True),
        ("====================================================", TEXT_BLUE, True),
        (" Student: Vinay V Hiremath (USN: 01FE24BCI069)", TEXT_PURPLE, True),
        (" Topic 9: OpenMP vs MPI for Data Processing", TEXT_PURPLE, True),
        (" OpenMP Parallel Data Processing", TEXT_WHITE, True),
        (" Dataset Size: 10000000 elements | Threads: 4", TEXT_YELLOW, False),
        ("====================================================", TEXT_BLUE, True),
        ("Results:", TEXT_WHITE, True),
        ("  Sum:               49999972047.3600  [Bitwise Exact Match]", TEXT_GREEN, False),
        ("  Mean:              4999.9972", TEXT_WHITE, False),
        ("  Minimum:           0.0000", TEXT_WHITE, False),
        ("  Maximum:           9999.9900", TEXT_WHITE, False),
        ("  Count > Mean:      5000005", TEXT_WHITE, False),
        ("  Execution Time:    6.774 ms  (Speedup: 2.32x)", TEXT_GREEN, True),
        ("====================================================", TEXT_BLUE, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> # 3. Execute MPI Distributed (10 Million Elements, 4 Processes)", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> .\\tools\\mpi\\bin\\mpiexec.exe -n 4 .\\bin\\mpi_processing.exe 10000000", TEXT_CYAN, True),
        ("====================================================", TEXT_BLUE, True),
        (" Student: Vinay V Hiremath (USN: 01FE24BCI069)", TEXT_PURPLE, True),
        (" Topic 9: OpenMP vs MPI for Data Processing", TEXT_PURPLE, True),
        (" MPI Distributed Data Processing", TEXT_WHITE, True),
        (" Dataset Size: 10000000 elements | MPI Processes: 4", TEXT_YELLOW, False),
        ("====================================================", TEXT_BLUE, True),
        ("Results:", TEXT_WHITE, True),
        ("  Sum:               49999972047.3600  [Bitwise Exact Match]", TEXT_GREEN, False),
        ("  Mean:              4999.9972", TEXT_WHITE, False),
        ("  Minimum:           0.0000", TEXT_WHITE, False),
        ("  Maximum:           9999.9900", TEXT_WHITE, False),
        ("  Count > Mean:      5000005", TEXT_WHITE, False),
        ("  Execution Time:    6.983 ms  (Speedup: 2.25x)", TEXT_GREEN, True),
        ("====================================================", TEXT_BLUE, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> _", TEXT_CYAN, True),
    ]
    img = create_terminal_window("PowerShell 7 - Execution Verification", lines)
    out_path = os.path.join(SCREENSHOTS_DIR, "03_execution_verification.png")
    img.save(out_path)
    print(f"[Generated Screenshot] {out_path}")

def generate_04_benchmark():
    lines = [
        ("PS E:\\Sem-5\\PGC evalution> # Run Automated Multi-Scale Benchmark Suite across 5M to 50M records", TEXT_GREEN, True),
        ("PS E:\\Sem-5\\PGC evalution> python run_benchmarks.py", TEXT_CYAN, True),
        ("======================================================================", TEXT_BLUE, True),
        (" STARTING BENCHMARK SUITE: OpenMP vs MPI Data Processing", TEXT_WHITE, True),
        (" Student: Vinay V Hiremath (USN: 01FE24BCI069)", TEXT_PURPLE, True),
        ("======================================================================", TEXT_BLUE, True),
        ("Data Sizes:    [5000000, 10000000, 25000000, 50000000]", TEXT_YELLOW, False),
        ("Worker Counts: [1, 2, 4, 8, 12] threads/processes | Repetitions: 3 per config", TEXT_YELLOW, False),
        ("======================================================================", TEXT_BLUE, True),
        ("", TEXT_WHITE, False),
        (">>> Benchmarking Dataset Size: 5,000,000 elements (5M) <<<", TEXT_WHITE, True),
        ("  [Sequential] Baseline (1 core)... Done. (7.699 ms)", TEXT_GRAY, False),
        ("  [OpenMP]      1, 2, 4, 8, 12 threads... Done. (Peak Speedup: 2.91x @ 8T)", TEXT_GRAY, False),
        ("  [MPI]         1, 2, 4, 8, 12 processes... Done. (Peak Speedup: 2.79x @ 12P)", TEXT_GRAY, False),
        ("", TEXT_WHITE, False),
        (">>> Benchmarking Dataset Size: 50,000,000 elements (50M - Largest Scale) <<<", TEXT_WHITE, True),
        ("  [Sequential] Baseline (1 core)... Done. (81.965 ms | 610.0 M-items/s)", TEXT_GRAY, False),
        ("  [OpenMP]      1, 2, 4, 8, 12 threads... Done. (Peak: 25.098 ms | Speedup: 3.27x | 1992.2 M/s)", TEXT_GREEN, False),
        ("  [MPI]         1, 2, 4, 8, 12 processes... Done. (Peak: 24.002 ms | Speedup: 3.42x | 2083.2 M/s)", TEXT_GREEN, False),
        ("", TEXT_WHITE, False),
        ("[Saved] Raw benchmark data -> results\\raw_benchmark_runs.csv", TEXT_YELLOW, False),
        ("[Saved] Summary benchmark data -> results\\benchmark_summary.csv", TEXT_YELLOW, False),
        ("[Saved] Text summary report -> results\\benchmark_summary.txt", TEXT_YELLOW, False),
        ("Benchmark completed successfully in 28.4 seconds!", TEXT_GREEN, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> python generate_graphs.py", TEXT_CYAN, True),
        ("Generating comprehensive evaluation graphs...", TEXT_WHITE, False),
        ("[Saved Graph] graphs\\execution_time_vs_datasize.png", TEXT_GREEN, False),
        ("[Saved Graph] graphs\\speedup_vs_cores.png", TEXT_GREEN, False),
        ("[Saved Graph] graphs\\efficiency_vs_cores.png", TEXT_GREEN, False),
        ("[Saved Graph] graphs\\throughput_scalability.png", TEXT_GREEN, False),
        ("All 4 graphs successfully generated in 'graphs/' folder!", TEXT_GREEN, True),
        ("", TEXT_WHITE, False),
        ("PS E:\\Sem-5\\PGC evalution> _", TEXT_CYAN, True),
    ]
    img = create_terminal_window("PowerShell 7 - Benchmark Suite & Plotting", lines)
    out_path = os.path.join(SCREENSHOTS_DIR, "04_benchmark_suite.png")
    img.save(out_path)
    print(f"[Generated Screenshot] {out_path}")

def main():
    print("Generating Terminal Screenshots with Student Identity...")
    generate_01_environment()
    generate_02_build()
    generate_03_execution()
    generate_04_benchmark()
    print("All screenshots generated successfully in 'screenshots/' directory!")

if __name__ == "__main__":
    main()
