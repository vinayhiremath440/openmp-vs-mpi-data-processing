#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    return omp_get_wtime();
}
#endif

/* Linear Congruential Generator for reproducible pseudo-random data */
static inline double generate_value(long long index) {
    unsigned long long seed = (unsigned long long)index * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(seed % 1000000) / 100.0; /* Float values in [0.00, 9999.99] */
}

int main(int argc, char* argv[]) {
    long long N = 10000000LL; /* Default: 10 Million elements */
    int num_threads = 4;      /* Default: 4 threads */

    if (argc > 1) {
        N = atoll(argv[1]);
    }
    if (argc > 2) {
        num_threads = atoi(argv[2]);
    }

    omp_set_num_threads(num_threads);

    printf("====================================================\n");
    printf(" OpenMP Parallel Data Processing\n");
    printf(" Dataset Size: %lld elements\n", N);
    printf(" Threads:      %d\n", num_threads);
    printf("====================================================\n");

    /* Allocate memory for dataset */
    double* data = (double*)malloc(N * sizeof(double));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for %lld doubles.\n", N);
        return 1;
    }

    /* Initialize dataset in parallel */
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++) {
        data[i] = generate_value(i);
    }

    /* Start benchmark timing */
    double t_start = get_time_sec();

    /* Pass 1: Parallel reduction for Sum, Sum of Squares, Min, Max */
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_val = data[0];
    double max_val = data[0];

    #pragma omp parallel for reduction(+:sum, sum_sq) reduction(min:min_val) reduction(max:max_val) schedule(static)
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

    /* Pass 2: Parallel conditional filter - count values above mean */
    long long count_above_mean = 0;
    #pragma omp parallel for reduction(+:count_above_mean) schedule(static)
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
    printf("RESULT_CSV:OpenMP,%lld,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%lld,%.3f\n",
           N, num_threads, sum, mean, min_val, max_val, std_dev, count_above_mean, exec_time_ms);

    free(data);
    return 0;
}
