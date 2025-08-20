#include "kernel1.cuh"

__global__ __launch_bounds__(1024) 
void sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int id_x = blockIdx.x * blockDim.x + threadIdx.x; // id x
    int id_y = blockIdx.y * blockDim.y + threadIdx.y; // id y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[id_y * K + i] * B[i * N + id_x]; // 两次全局内存访问和一次FMA（累加乘）
    }
    C[id_y * N + id_x] = alpha * tmp + beta * C[id_y * N + id_x];
}
