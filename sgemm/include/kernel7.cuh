#pragma once
#include <stdio.h>

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
