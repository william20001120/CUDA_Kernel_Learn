#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

#define CEIL(a,b) ((a)+((b)-1))/(b)
#define checkCudaErrors(func) {                                                   \
    cudaError_t e = (func);                                                       \
    if(e != cudaSuccess)                                                          \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));  \
}

// dim3 dimGrid(M);
// dim3 dimBlock(32);
// 适合K∈[32,128]使用，小于32或大于128有进一步的优化方法
__global__ void sgemv_k32(float* A, float* x, float* y, int M, int K) {
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x;  // 0~M-1
    if (row >= M) return;

    float res = 0.0f;
    int kIteration = CEIL(K, warpSize);  // 每个线程需要负责计算的数据个数

    #pragma unroll
    for(int i = 0; i < kIteration; i++){
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }

    if(laneId == 0) y[row] = res;
}

int main() {
    size_t M = 1024;
    size_t K = 32;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_x = sizeof(float) * K;
    size_t bytes_y = sizeof(float) * M;
    float* h_A  = (float*)malloc(bytes_A);
    float* h_x  = (float*)malloc(bytes_x);
    float* h_y  = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    double duration[2] = {0, 0};
    double GFLOPS[2] = {0, 0};
    double GFLOPs = 2.0 * M * 1 * K;

    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        h_A[i] = (float)i/K;
    }

    // 生成x的数据
    for( int i = 0; i < K; i++ ) {
        h_x[i] = 1;
    }
    memset(h_y,  0, M * sizeof(float));
    memset(h_y1, 0, M * sizeof(float));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int iteration = 1000;

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaEventRecord(start));

    for (int run = 0 ; run < iteration; run ++ ) {
        dim3 dimGrid(M);
        dim3 dimBlock(32);
        sgemv_k32<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, K);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[0] = msecTotal / iteration;
    GFLOPS[0] = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            K, M, &alpha, 
            d_A, K, d_x, 1, &beta, d_y, 1
        );
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[1] = msecTotal / iteration;
    GFLOPS[1] = (GFLOPs * 1.0e-9f) / (duration[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[1],
        duration[1],
        GFLOPs);

    cublasDestroy(blas_handle);
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}
