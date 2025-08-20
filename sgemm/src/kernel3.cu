#include "kernel3.cuh"

template<const int BM,
         const int BN,
         const int BK,
         const int TM>
__global__ void sgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM; // thread_num是一个block中线程的数量，TM表示一个线程负责计算TM个元素

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 当前线程负责搬运全局内存矩阵A的中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    // a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存
    // b_tile_* 同理

    // 若BM=64, BK=8, thread_num=512, 则a_tile_stride=64, a_tile_stride=BM，表示每个线程搬运 1 轮即可完成所需元素的搬运
    // 若BM=128, BK=8, thread_num=512, 则a_tile_stride=64, a_tile_stride=BM/2，表示每个线程搬运 2 轮即可完成所需元素的搬运
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.};  // 每个线程负责TM个元素，则需要申请TM个寄存器保存累加值，额外的一个寄存器用于缓存
#pragma unroll
    for (int k = 0; k < K; k += BK) {  // 窗口滑动
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] = Bs[tx + i * BN];  // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
#pragma unroll
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    }
}

// template instantiation declaration
template __global__ void sgemm_v3<64, 64, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
