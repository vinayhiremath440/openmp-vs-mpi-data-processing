#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* Linear Congruential Generator for reproducible pseudo-random data */
static inline double generate_value(long long index) {
    unsigned long long seed = (unsigned long long)index * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(seed % 1000000) / 100.0; /* Float values in [0.00, 9999.99] */
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    long long N = 10000000LL; /* Default: 10 Million elements */
    if (argc > 1) {
        N = atoll(argv[1]);
    }

    /* Compute chunk size and offset for domain decomposition */
    long long base_chunk = N / num_procs;
    long long remainder = N % num_procs;
    long long local_count = base_chunk + (rank < remainder ? 1 : 0);
    long long local_start = rank * base_chunk + (rank < remainder ? rank : remainder);

    /* Allocate local memory chunk */
    double* local_data = (double*)malloc(local_count * sizeof(double));
    if (!local_data && local_count > 0) {
        fprintf(stderr, "Rank %d failed to allocate memory for %lld elements\n", rank, local_count);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Generate local data chunk */
    for (long long i = 0; i < local_count; i++) {
        local_data[i] = generate_value(local_start + i);
    }

    /* Synchronize before benchmarking */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Pass 1: Compute local metrics */
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    double local_min = (local_count > 0) ? local_data[0] : 1e18;
    double local_max = (local_count > 0) ? local_data[0] : -1e18;

    for (long long i = 0; i < local_count; i++) {
        double v = local_data[i];
        local_sum += v;
        local_sum_sq += v * v;
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    /* Collective Reduction to Rank 0 */
    double global_sum = 0.0;
    double global_sum_sq = 0.0;
    double global_min = 0.0;
    double global_max = 0.0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Rank 0 computes global mean and broadcasts it for Pass 2 */
    double global_mean = 0.0;
    if (rank == 0) {
        global_mean = global_sum / (double)N;
    }
    MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Pass 2: Local conditional filter above mean */
    long long local_above_count = 0;
    for (long long i = 0; i < local_count; i++) {
        if (local_data[i] > global_mean) {
            local_above_count++;
        }
    }

    /* Collective reduction for count above mean */
    long long global_above_count = 0;
    MPI_Reduce(&local_above_count, &global_above_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double exec_time_ms = (t_end - t_start) * 1000.0;

    /* Rank 0 computes variance, std deviation, and prints outputs */
    if (rank == 0) {
        double variance = (global_sum_sq / (double)N) - (global_mean * global_mean);
        if (variance < 0.0) variance = 0.0;
        double std_dev = sqrt(variance);

        printf("====================================================\n");
        printf(" Student: Vinay V Hiremath (USN: 01FE24BCI069)\n");
        printf(" Topic 9: OpenMP vs MPI for Data Processing\n");
        printf(" MPI Distributed Data Processing\n");
        printf(" Dataset Size: %lld elements\n", N);
        printf(" MPI Processes: %d\n", num_procs);
        printf("====================================================\n");
        printf("Results:\n");
        printf("  Sum:               %.4f\n", global_sum);
        printf("  Mean:              %.4f\n", global_mean);
        printf("  Variance:          %.4f\n", variance);
        printf("  Std Deviation:     %.4f\n", std_dev);
        printf("  Minimum:           %.4f\n", global_min);
        printf("  Maximum:           %.4f\n", global_max);
        printf("  Count > Mean:      %lld\n", global_above_count);
        printf("  Execution Time:    %.3f ms\n", exec_time_ms);
        printf("====================================================\n");

        /* Machine readable output line for benchmarking scripts */
        printf("RESULT_CSV:MPI,%lld,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%lld,%.3f\n",
               N, num_procs, global_sum, global_mean, global_min, global_max, std_dev, global_above_count, exec_time_ms);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}
