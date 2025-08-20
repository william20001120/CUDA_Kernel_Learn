# 归约计算-sum
内容：求给定数组求和

## v0~v3
1. device_reduce_v0：仅使用**全局内存**，且 N 必须是 BLOCK_SIZE 的整数倍
2. device_reduce_v1：使用（静态）**共享内存**，不再要求 N 是 BLOCK_SIZE 的整数倍，归约的过程中不会改变全局内存的数据
3. device_reduce_v2：在v1基础上修改，使用（动态）**共享内存**，性能不变
4. device_reduce_v3：在v2基础上修改，通过原子函数，不再需要到CPU上再归约一次，**v3缺点是BLOCK_SIZE必须是2的幂次，否则折半操作时会计算出错，导致误差很大**

## v4、v5
device_reduce_v4：使用 warp shuffle 进行计算，在一个warp里进行归约，**BLOCK_SIZE需要是32的整数倍，否则产生线程数不足32的warp，可能会导致访问到无效数据。**

因为一个 warp 里的线程是**天然同步的（硬件级同步）**，所以不需要手动调用 `__syncthreads()`，并行性更好，效率更高。

device_reduce_v5：在 v4 基础上使用 float4 向量化读取。

## 测试
N=100000000，BLOCK_SIZE = 128 的测试结果：
```
[reduce_host]: sum = -1392.220947, total_time_h = 383.848877 ms
[reduce_v0]: sum = 12188807.000000, total_time_0 = 31.068247 ms
[reduce_v1]: sum = -1392.776123, total_time_1 = 19.648817 ms
[reduce_v2]: sum = -1392.776123, total_time_2 = 19.483204 ms
[reduce_v3]: sum = -1392.776123, total_time_3 = 15.859097 ms
[reduce_v4]: sum = -1392.792847, total_time_4 = 11.208912 ms
[reduce_v5]: sum = -1392.694214, total_time_5 = 4.105523 ms
```

## 参考：
1. cuda编程基础与实践 (樊哲勇)