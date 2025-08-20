#include <stdio.h>
#include <cublas_v2.h>

int main() {
    unsigned int m = 2, n = 4, k = 3;

    // 1. host malloc for matrix A, B, C
    unsigned int size_A = m * k;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);

    unsigned int size_B = k * n;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    unsigned int size_C = m * n;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_CUBLAS = (float *) malloc(mem_size_C);

    // 2. init elements in A and B
    for (int i = 0; i < size_A; i++)
        h_A[i] = i + 1;

    for (int i = 0; i < size_B; i++)
        h_B[i] = i + 1;

    // 3. cuda malloc for A, B, C and copy A, B from host to device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);
    cudaMalloc((void **) &d_C, mem_size_C);
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    
    // 4. call cublasSgemm
    const float alpha = 1.0f;     
    const float beta  = 0.0f;
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    cublasDestroy(handle);

    // 5. copy C from device to host
    cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    // 6. print matrix C
    printf("C=\n");
    for (int i = 0; i < m * n; i++) {
        printf("%3.2f\t", h_CUBLAS[i]);
        if ((i + 1) % n == 0) printf("\n");
    }
    
    // 7. free memory
    free(h_A);
    free(h_B);
    free(h_CUBLAS);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}