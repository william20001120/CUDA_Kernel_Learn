#include <stdio.h>
#include "utils.cuh"

void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void randomize_matrix(float *mat, size_t N) {
    std::random_device rd;  
    std::mt19937 gen(rd()); // 使用随机设备初始化生成器  

    // 创建一个在[0, 2000)之间均匀分布的分布对象  
    std::uniform_int_distribution<> dis(0, 2000); 
    for (int i = 0; i < N; i++) {
        // 生成随机数，限制范围在[-1.0,1.0]
        mat[i] = (dis(gen)-1000)/1000.0;  
    }
}

void copy_matrix(float *src, float *dest, size_t N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %lu elements in total.\n", i, N);
}

void print_matrix(const float *A, size_t M, size_t N) {
    printf("[");
    for (size_t i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, size_t N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("Error: mat1[%d]=%5.2f, mat2[%d]=%5.2f, \n", i, mat1[i], i, mat2[i]);
            return false;
        }
    }
    return true;
}

float call_kernel(int kernel_num, bool record, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    float total_time = 0;
    float repeat_times = 0;
    if (record) repeat_times = REPEAT_TIMES;
    if (kernel_num == 0) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        total_time = TIME_RECORD(repeat_times, ([&]{cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);}));
        cublasDestroy(handle);
    }
    else if (kernel_num == 1) {
        dim3 blockDim(32, 32);
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    }
    else if (kernel_num == 2) {
        dim3 blockDim(1024);
        dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    }
    else if (kernel_num == 3) {
        dim3 blockDim(512);
        dim3 gridDim(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    }
    else if (kernel_num == 4) {
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    }
    else if (kernel_num == 5) {
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v5<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    }
    else if (kernel_num == 6) {
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v6<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    }
    else if (kernel_num == 7) {
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
        total_time = TIME_RECORD(repeat_times, ([&]{sgemm_v7<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}));
    } else {
        printf("Error: kernel %d not found.\n", kernel_num);
        exit(EXIT_FAILURE);
    }
    return total_time;
}
