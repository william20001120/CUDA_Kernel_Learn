#include "kernel2.cuh"

template<const int BLOCK_SIZE>
__global__ void sgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // blockId and threadId
    int bx = blockIdx.x;
    int by = blockIdx.y;    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 申请共享内存空间
    // NVIDIA GeForce GTX 1050's sharedMemPerBlock is 48KB = 48*1024B = 49152B(0xc000)
    // 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <= 48*1024/4 = 12288
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {  // 窗口滑动
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();  // 同步同一个线程块(block)中的线程，执行到同一个点
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

// template instantiation declaration
template __global__ void sgemm_v2<16>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
template __global__ void sgemm_v2<32>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
template __global__ void sgemm_v2<64>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

// Note: pay attention to the `sharedMemPerBlock`,
// for example, when there is a template instantiation declaration like below:
// template __global__ void sgemm_v2<128>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
// compiler will throw error like below:
// ptxas error   : Entry function '_Z8sgemm_v2ILi128EEviiifPfS0_fS0_' uses too much shared data (0x20000 bytes, 0xc000 max)
