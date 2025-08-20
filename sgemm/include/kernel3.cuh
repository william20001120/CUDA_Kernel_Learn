#pragma once
#include <stdio.h>

template<const int BM,
         const int BN,
         const int BK,
         const int TM>
__global__ void sgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
