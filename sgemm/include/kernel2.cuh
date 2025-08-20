#pragma once
#include <stdio.h>

template<const int BLOCK_SIZE>
__global__ void sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
