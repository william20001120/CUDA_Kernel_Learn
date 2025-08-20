#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"

void max_cpu(float* input, float* output, int N) {
    *output =  *(std::max_element(input, input + N));  // 计算输入数组的最大值
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;  // address转为int指针
    int old = *address_as_i;  // address中的旧值，用int解码
    int assumed;
    do {
        assumed = old;  // assumed存储旧值
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float* input, float* output, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // 求M(max)
    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

	if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneId == 0) atomicMax(output, val);
    }
}


int main() {
    size_t N = 12800;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;

    float* input  = (float*)malloc(sizeof(float) * N);
    for (int i = N; i > 0; i--) {
        input[i] = i;
    }
    float* output_ref = (float*)malloc(1 * sizeof(float));
    float total_time_h = TIME_RECORD(repeat_times, ([&]{max_cpu(input, output_ref, N);}));
    printf("[max_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* output = (float*)malloc(1 * sizeof(float));
    float* input_device  = nullptr;
    float* output_device = nullptr;

    cudaMalloc(&input_device, N * sizeof(float));
    cudaMalloc(&output_device, 1 * sizeof(float));
    cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // max
    int block_size = BLOCK_SIZE;
    int grid_size  = CEIL(N, BLOCK_SIZE);
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{max_kernel<<<grid_size, block_size>>>(input_device, output_device, N);}));
    printf("[max_kernel]: total_time_1 = %f ms\n", total_time_1 / repeat_times);
    cudaMemcpy(output, output_device, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("output = %f, output_ref = %f\n", *output, *output_ref);

    free(input);
    free(output);
    free(output_ref);
    cudaFree(input_device);
    cudaFree(output_device);
    return 0;
}