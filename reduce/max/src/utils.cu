#include "utils.cuh"

void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}

void randomize_matrix(float *mat, int N) {
    std::random_device rd;  
    std::mt19937 gen(rd()); // 使用随机设备初始化生成器  

    // 创建一个在[0, 2000)之间均匀分布的分布对象  
    std::uniform_int_distribution<> dis(0, 2000); 
    for (int i = 0; i < N; i++) {
        // 生成随机数，限制范围在[-10.0,10.0]
        mat[i] = (dis(gen)-100)/100.0;  
    }
}

void print_matrix(float* a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%7.3f", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool verify_matrix(float *mat1, float *mat2, size_t N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-4) {
            printf("Error: mat1[%d]=%5.6f, mat2[%d]=%5.6f, \n", i, mat1[i], i, mat2[i]);
            return false;
        }
    }
    return true;
}