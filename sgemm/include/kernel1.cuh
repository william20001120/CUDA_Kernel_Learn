#pragma once
#include <stdio.h>

__global__ __launch_bounds__(1024) 
void sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
