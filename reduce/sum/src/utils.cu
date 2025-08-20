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
        // 生成随机数，限制范围在[-1.0,1.0]
        mat[i] = (dis(gen)-1000)/1000.0;  
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
