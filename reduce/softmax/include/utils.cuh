#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <random>

#define CEIL(a, b) ((a) + (b) - 1) / (b)
#define FLOAT4(value) (*(float4*)(&(value)))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
#define TIME_RECORD(N, func)                                                                    \
    [&] {                                                                                       \
        float total_time = 0;                                                                   \
        for (int repeat = 0; repeat <= N; ++repeat) {                                           \
            cudaEvent_t start, stop;                                                            \
            cudaCheck(cudaEventCreate(&start));                                                 \
            cudaCheck(cudaEventCreate(&stop));                                                  \
            cudaCheck(cudaEventRecord(start));                                                  \
            cudaEventQuery(start);                                                              \
            func();                                                                             \
            cudaCheck(cudaEventRecord(stop));                                                   \
            cudaCheck(cudaEventSynchronize(stop));                                              \
            float elapsed_time;                                                                 \
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
            if (repeat > 0) total_time += elapsed_time;                                         \
            cudaCheck(cudaEventDestroy(start));                                                 \
            cudaCheck(cudaEventDestroy(stop));                                                  \
        }                                                                                       \
        if (N == 0) return (float)0.0;                                                          \
        return total_time;                                                                      \
    }()

void _cudaCheck(cudaError_t error, const char *file, int line);
void randomize_matrix(float *mat, int N);
void print_matrix(float* a, int M, int N);
bool verify_matrix(float *mat1, float *mat2, size_t N);