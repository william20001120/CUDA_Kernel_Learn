#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello World from the GPU\n");
}

int main() {
    hello_from_gpu<<<2, 2>>>();
    cudaDeviceSynchronize();  // 同步
    return 0;
}