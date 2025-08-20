# 归约计算-max

使用 warp shuffle 进行归约，因为 AtomicMax 不支持 float 类型，需要自己手写实现。

## 测试

```log
[max_cpu]: total_time_h = 0.126157 ms
[max_kernel]: total_time_1 = 0.052634 ms
output = 12799.000000, output_ref = 12799.000000
```
