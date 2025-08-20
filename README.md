## CUDA Kernel Learn

学习与实践 CUDA Kernel 优化的示例仓库，涵盖矩阵乘法（SGEMM）、矩阵转置、各类归约（sum/max/softmax/矩阵 softmax）、GEMV、逐元素算子、LayerNorm，以及 cuBLAS 对比与若干入门示例。目标是以循序渐进的方式，拆解典型优化技巧并给出可复现实验。

### 环境与依赖
- 必备：NVIDIA GPU、CUDA Toolkit（含 nvcc）、CMake（用于部分子项目）、GNU Make
- 可选：Python 3 + matplotlib（用于 `sgemm/tools/plot.py` 画图）
- 示例设备（来自 `sgemm/README.md`）：GeForce GTX 1050（SM=5，CC=6.1，sharedMemPerBlock=48KB，Warp Size=32）

### 目录概览
- `sgemm/`：从 naive 到多级优化的单精度矩阵乘法（7 个 kernel），涵盖 shared memory、thread tile、寄存器缓存、float4 向量化、双缓存预取等，并与 cuBLAS 对比；带测试与绘图脚本
- `transpose/`：矩阵转置优化，含全局内存合并访问、shared memory + padding 消除 bank conflict、swizzling 等；提供性能对比
- `reduce/`
  - `sum/`：v0–v5，依次引入（静态/动态）shared memory、原子操作、warp shuffle、float4 向量化
  - `max/`：基于 warp shuffle 的浮点最大值（手写 AtomicMax float）
  - `softmax/`：标量 softmax，分 kernel 完成 max/sum/softmax，避免单 kernel 内的 grid 级同步问题
  - `softmax_matrix/`：对 M×N 矩阵按行/列 softmax，基于 warp 归约（`__shfl_xor_sync` 等）
- `gemv/`：矩阵乘向量（适配 K∈[32,128] 的 kernel），含一键编译命令
- `elementwise/`：逐元素算子样例（以 add 为例，可扩展到 relu/sigmoid/histogram 等）
- `laynorm/`：LayerNorm 前向/反向的 kernel 示例（`layernorm_forward.cu`、`layernorm_backward.cu`）
- `cublas/`：通用头文件与算子片段（`gelu.cuh`、`matmul.cuh`、`cuda_utils.cuh` 等），便于与 cuBLAS/自定义 kernel 结合
- `example/`
  - `hello_cuda/`：最小可运行 CUDA 示例（CMake）
  - `cuda_info/`：打印设备与内存限制信息
  - `matrix_copy/`：矩阵拷贝/基础示例（CMake）
  - `cublas_example/`：cuBLAS `cublasSgemm` 的最小示例（含列主序/行主序转换公式与 CMake）

### 快速开始
以下给出各模块常见的构建与运行方式（不同子目录彼此独立，互不依赖）。

- 通用（基于 CMake 的子项目，如 `sgemm/`、`transpose/`、`reduce/*/`、`example/*` 中大多数）：
```bash
cd 子目录
mkdir build && cd build
cmake .. && make
# 例如（sgemm）：
./main 0    # 运行 cuBLAS 对比
./main 1    # 运行 kernel1（其余依次递增）
```

- sgemm 额外工具：
```bash
cd sgemm
pip install matplotlib
bash tools/test.sh   # 生成 ./test/*.log，并在 ./images 输出对比图
```

- 单文件 nvcc 编译示例（如 `gemv/`、`elementwise/`）：
```bash
# sgemv（K∈[32,128]）：
cd gemv
nvcc sgemv_k32.cu -o sgemv_k32 -lcublas && ./sgemv_k32

# elementwise-add：
cd elementwise
nvcc add.cu -o add && ./add
```

- 其他示例：
```bash
# hello_cuda（CMake）
cd example/hello_cuda && mkdir build && cd build && cmake .. && make && ./hello_cuda

# cuda_info（打印设备/共享内存限制等）
cd example/cuda_info && nvcc main.cu -o main && ./main

# cuBLAS 示例（CMake）
cd example/cublas_example && mkdir build && cd build && cmake .. && make && ./cublas_exmple
```

### 学习要点（摘录）
- 全局内存访问：尽量合并访问；若难以同时合并读/写，优先保证合并写；必要时使用 `__ldg()`（部分架构需显式）
- 共享内存优化：tile 化；避免 bank conflict（padding 或 swizzling）；控制共享内存占用以保持足够并行度
- 访存与计算重叠：寄存器缓存、double-buffer 预取，减少 `__syncthreads()` 造成的阻塞
- 向量化指令：`float4` 提升带宽与降低指令数，但会增加寄存器压力与指针对齐要求
- Warp 原语：`__shfl_*` 进行寄存器内归约，避免共享内存与同步开销
- Grid 级同步：单 kernel 内无法通过 `__threadfence()` 实现 grid 同步，需拆分为多个 kernel

### 性能/结果
- `sgemm/`：提供 `test/` 日志和 `images/` 对比图（包含各版本 kernel 与 cuBLAS 的对照）
- `transpose/`：README 中给出多版本 kernel 的耗时对比（GTX1050，M=12800, N=1280, BLOCK_SIZE=32）
- `reduce/`：`sum`/`max`/`softmax` 均提供示例输出/耗时，便于横向比较

### 参考资料
- NVIDIA_SGEMM_PRACTICE、Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
- CUDA Pro Tip: Increase Performance with Vectorized Memory Access
- CUDA 内存合并访问、共享内存冲突（bank conflict）与 swizzling 文章
- 知乎/博客/专栏若干链接（详见各子目录 README）
- 《cuda 编程基础与实践》（樊哲勇）




