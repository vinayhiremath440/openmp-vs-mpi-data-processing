#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

/* High-resolution timer for Windows */
static double get_time_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

/* Linear Congruential Generator for reproducible pseudo-random data */
static inline double generate_value(long long index) {
    unsigned long long seed = (unsigned long long)index * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(seed % 1000000) / 100.0; /* Float values in [0.00, 9999.99] */
}

int main(int argc, char* argv[]) {
    long long N = 10000000LL; /* Default: 10 Million elements */
    if (argc > 1) {
        N = atoll(argv[1]);
    }

    printf("====================================================\n");
    printf(" Sequential Baseline Data Processing\n");
    printf(" Dataset Size: %lld elements\n", N);
    printf("====================================================\n");

    /* Allocate memory for dataset */
    double* data = (double*)malloc(N * sizeof(double));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for %lld doubles.\n", N);
        return 1;
    }

    /* Initialize dataset */
    for (long long i = 0; i < N; i++) {
        data[i] = generate_value(i);
    }

    /* Start benchmark timing */
    double t_start = get_time_sec();

    /* Pass 1: Reduction for Sum, Sum of Squares, Min, Max */
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_val = data[0];
    double max_val = data[0];

    for (long long i = 0; i < N; i++) {
        double v = data[i];
        sum += v;
        sum_sq += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    double mean = sum / (double)N;
    double variance = (sum_sq / (double)N) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    double std_dev = sqrt(variance);

    /* Pass 2: Conditional filter - count values above mean */
    long long count_above_mean = 0;
    for (long long i = 0; i < N; i++) {
        if (data[i] > mean) {
            count_above_mean++;
        }
    }

    double t_end = get_time_sec();
    double exec_time_ms = (t_end - t_start) * 1000.0;

    printf("Results:\n");
    printf("  Sum:               %.4f\n", sum);
    printf("  Mean:              %.4f\n", mean);
    printf("  Variance:          %.4f\n", variance);
    printf("  Std Deviation:     %.4f\n", std_dev);
    printf("  Minimum:           %.4f\n", min_val);
    printf("  Maximum:           %.4f\n", max_val);
    printf("  Count > Mean:      %lld\n", count_above_mean);
    printf("  Execution Time:    %.3f ms\n", exec_time_ms);
    printf("====================================================\n");

    /* Machine readable output line for benchmarking scripts */
    printf("RESULT_CSV:Sequential,%lld,1,%.4f,%.4f,%.4f,%.4f,%.4f,%lld,%.3f\n",
           N, sum, mean, min_val, max_val, std_dev, count_above_mean, exec_time_ms);

    free(data);
    return 0;
}
