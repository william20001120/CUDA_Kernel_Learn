# 归约计算-Softmax
内容：求给定数组的softmax

需要注意的是，`__threadfence()` 做不到 grid 级别的同步，所以不能在一个核函数里，先求 sum 再除以 sum，因为核函数中间无法保证求得的 sum 是所有元素中的和，求 max 同理，所以，softmax.cu 中，求max和sum都是使用了单独的kernel，因为单独的kernel是一定可以保证 grid 中所有 block 同步的：
```cpp
void call_softmax_v2(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N) {
    int block_size = BLOCK_SIZE;
    int grid_size  = CEIL(N, BLOCK_SIZE);

    // 1. 初始化
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));  // total需要设置为0
    setToNegativeMax<<<1,1>>>(total_max_device);            // total_max_device设置为最小FLOAT值

    // 2. 计算最大值
    max_kernel<<<grid_size, block_size>>>(input_device, total_max_device, N);

    // 3. 计算和
    sum_kernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 4. 计算softmax (减去最大值避免溢出)
    softmax_kernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}
```
