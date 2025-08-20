# CUDA SGEMM 优化

本项目参考并修改自 [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)，加入了更多的图例和 CUDA 编程解释，相关引用见文末。

## 开发环境
设备：NVIDIA GeForce GTX 1050
```
Device ID: 0
       *Number of SMs: 5
       Compute Capability Major: 6
       Compute Capability Minor: 1
       memoryBusWidth: 128
       *maxThreadsPerBlock: 1024
       maxThreadsPerMultiProcessor: 2048
       *totalGlobalMem: 2047M
       sharedMemPerBlock: 48KB
       *sharedMemPerMultiprocessor: 96KB
       totalConstMem: 64KB
       *multiProcessorCount: 5
       *Warp Size: 32
```

## 开发流程
1. 在src下编写kernel.cu
2. 在include下编写对应头文件，并在include/kernel.cuh中包含该头文件
3. 在src/utils.cu的call_kernel函数中调用编写的kernel
4. 编译：
```bash
mkdir build && cd build
cmake ..
make
```
5. 运行：
```bash
# run cuBLAS(0) or custom kernel(>0)
./main 0  # cuBLAS
./main 1  # kernel1
...
```
6. 测试并画图：
```bash
pip install matplotlib
bash tools/test.sh  # 日志保存在./test, 图片保存在./images
```

## CUDA名词解释
- **访存量**”（memory access）：通常指的是GPU核心（CUDA核心）或线程所需从全局内存中读取或写入的数据量。
- **计算访存比**：每秒计算量与每秒访存量之比。

## Kernel1：Naive 实现 (global memory)
<div align=center>
<img src="./images/kernel_cublas_vs_1.png" width = "700"/>
</div>

### 代码
```cpp
__global__ __launch_bounds__(1024) 
void sgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int id_x = blockIdx.x * blockDim.x + threadIdx.x; // id x
    int id_y = blockIdx.y * blockDim.y + threadIdx.y; // id y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[id_y * K + i] * B[i * N + id_x]; // 两次全局内存访问和一次FMA（累加乘）
    }
    C[id_y * N + id_x] = alpha * tmp + beta * C[id_y * N + id_x];
}
```

### 计算步骤 (图解)
将每个逻辑线程与矩阵C的每一个元素相对应，每个线程负责C中一个元素的计算：
<div align=center>
<img src="./images/image.png" width = "500"/><img src="./images/image-1.png" width = "500"/>
</div>

### 分析
未经过优化的矩阵乘法性能不足cuBLAS的1/10，具体分析如下：

1. **访存比低**：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2；
2. **访存延迟高**：访问**全局内存**，**延迟高**，需要几百个时钟周期 (cycle)
3. **较低的访存比无法有效隐藏访存延迟**
4. 访存量：矩阵C的每个元素计算需要访问2K个单精度浮点数，完成全部计算需要 $2\times K\times M\times N$
5. 相同位置元素被重复读取（C中同一行元素计算共享A中同一行元素，C中同一列元素计算共享B中同一列元素）

> 动态全局内存是在运行时动态分配的内存，**所有线程**可见（主机端也可见），使用 `cudaMalloc()` 和 `cudaFree()` 函数来分配和释放。

## Kernel2：从 global memory 到 shared memory
<div align=center>
<img src="./images/kernel_1_vs_2.png" width = "500"/><img src="./images/kernel_cublas_vs_2.png" width = "500"/>
</div>

### 计算步骤 (图解)
<div align=center>
<img src="./images/describe_kernel_2.png" width = "400"/>
</div>
<div align=center>
<img src="./images/image-3.png" width = "800"/>
</div>

如上图左边所示，矩阵乘时，矩阵C的每一个行中的结果在计算时，都要重复读取矩阵A中的同一行（同理，矩阵C的每一个列中的结果在计算时，都要重复读取矩阵B中的同一列）。

利用这个特点，可以把A、B、C按 $BM\times BK$， $BK\times BN$， $BM\times BN$ 的切分，三个矩阵形成 $\frac{M}{BM}\times \frac{K}{BK}$， $\frac{K}{BK}\times \frac{N}{BN}$， $\frac{M}{BM}\times \frac{N}{BN}$ 的网格，如上图右边所示：
1. 在block中申请等同于块大小的共享内存，每个 block 从全局内存 (global memory) 中读取数据并保存在共享内存中
2. 由于块的尺寸大于 $1\times 1$，所以读取全局内存的次数会按块的尺寸成倍减小
3. 因为共享内存在一个 block 中是共享的，这样一个block内的元素在重复读取同一行（列）时，可以直接从共享内存中读取
4. 尽管总的读取次数增加了（上图中全局内存的访问次数变为原来的一半，共享内存的访问次数等于naive实现的读取次数），但是全局内存的访问次数显著减少，而共享内存的访问次数虽然很多但由于共享内存的访问延迟是远小于全局内存的，所以**总的访问延迟还是显著减小**的

### 分析
性能相比kernel1有所提升，但相比cuBLAS的实现，差距还是很大，具体分析如下：

1. **访存量显著减小**：完成C中所有元素的计算一共需要从global memory中读取 $\frac{M}{BM}\times \frac{N}{BN} \times \frac{K}{BK} \times (BM \times BK+BK \times BN)=M \times N \times K \times (\frac{1}{BM}+ \frac{1}{BN})$，访存量是 kernel1 的 $0.5 \times(\frac{1}{BM}+ \frac{1}{BN})$，代码中使用BM=BN=32，此时访存量变为原来的 1/32；
2. **访存比没有变化**：每次计算仍然需要2个访存指令和1个计算指令。

> 从上面分析可以看出，增加block的大小（BM和BN）可以进一步降低全局内存的访问量，但是也会增加共享内存的使用量。因为每个SM的共享内存数量是一定的，如果在单个线程块中分配过度的共享内存，将会限制线程束的数量 (比如固定线程块中的线程数量不变，而增加线程块的共享内存的分配量，那么分配给一个SM的线程块数量将减少，线程总数减少，线程束减少)。

## Kernel3：引入一维thread tile和寄存器
<div align=center>
<img src="./images/kernel_2_vs_3.png" width = "500"/><img src="./images/kernel_cublas_vs_3.png" width = "500"/>
</div>

### 计算步骤
> `pragma unroll` 用于展开循环（告诉编译器将循环体复制多次），好处是可以消除循环开销（如循环索引计算和循环终止检查等）

<div align=center>
<img src="./images/image-4.png" width = "800"/>
</div>

1. 引入thread tile，即一个线程负责block中多个元素的计算，TM和TN分别表示thread tile的高和宽。上图中TM=2，TN=1。
2. 定义了一个一维的长度为TM+1的 `tmp`，其中`tmp[TM]`用于缓存Bs中的元素到寄存器中，访问几乎无时延。
3. 当TM=8时，下面的代码中，外层循环每进行1次，访问1次Bs，然后访问8次As，并计算8次，计算-访存比是8:9，比原始版本1:2高了不少。
```cpp
for (int i = 0; i < BK; i++) {
    tmp[TM] = Bs[tx + i * BN]; // 额外的一个寄存器，避免反复从共享内存中读取Bs[tx + i * BN]
    #pragma unroll  // 循环展开，增加指令并行度
    for (int j = 0; j < TM; j++) {  // 8次循环
        tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
    }
}
```

### 分析
1. 全局内存访存量：相比于初始版本，通过对 $64\times 64$ block size进行缓存，访存量降至1/64；
2. 计算-访存比：提高至8:9，可以有效隐藏访存延迟；

## Kernel4：引入二维thread tile
<div align=center>
<img src="./images/kernel_3_vs_4.png" width = "500"/><img src="./images/kernel_cublas_vs_4.png" width = "500"/>
</div>

## Kernel5：引入二维thread tile，并用寄存器避免重复读取shared memory
<div align=center>
<img src="./images/kernel_4_vs_5.png" width = "500"/><img src="./images/kernel_cublas_vs_5.png" width = "500"/>
</div>

## Kernel6：向量内存指令FLOAT4优化
<div align=center>
<img src="./images/kernel_5_vs_6.png" width = "500"/><img src="./images/kernel_cublas_vs_6.png" width = "500"/>
</div>

### 计算步骤
<div align=center>
<img src="./images/image6-1.png" width = "800"/>
</div>
因为拷贝是以float4为单位进行的，为了便于后续的乘法计算（保证数据连续），在A拷贝到As时，对As进行转置，Bs则不用：
<div align=center>
<img src="./images/image6-2.png" width = "800"/>
</div>
<div align=center>
<img src="./images/image6-3.png" width = "500"/>
</div>

### 分析
在kernel5的基础上，引入`float4`类型，这个是CUDA的扩展类型，被用于“向量化存取”，可以将4个浮点数为一组进行拷贝，减少了内存指令。

使用向量化存取需要注意以下几点：
1. float4等类型会增加寄存器压力，减少总体并行性
2. 如果指针没有对齐或者数据类型的尺寸不是2的幂次，则不能使用

> In almost all cases vectorized loads are preferable to scalar loads. Note however that using vectorized loads **increases register pressure** and **reduces overall parallelism**. So if you have a kernel that is already register limited or has very low parallelism, you may want to stick to scalar loads. Also, as discussed earlier, if your pointer is **not aligned** or your **data type size in bytes is not a power of two** you cannot use vectorized loads.

向量化存取有以下好处：
1. 增加了带宽
2. 减少了内存指令（4次内存拷贝指令→1次内存拷贝指令）
3. 减少了延迟

> Vectorized loads are a fundamental CUDA optimization that you should use when possible, because they **increase bandwidth**, reduce **instruction count**, and **reduce latency**. In this post, I’ve shown how you can easily incorporate vectorized loads into existing kernels with relatively few changes.

更多细节可以阅读[官方blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)。

## Kernel7：数据预取 (双缓存)
<div align=center>
<img src="./images/kernel_6_vs_7.png" width = "500"/><img src="./images/kernel_cublas_vs_7.png" width = "500"/>
</div>

### 计算步骤
总体流程图：
<div align=center>
<img src="./images/double-buffer.png" width = "800"/>
</div>

以下是具体流程。

1、初始化数据：

<div align=center>
<img src="./images/image7-1.png" width = "800"/>
</div>

2、第一次 global memory 到 shared memory 的拷贝: 

<div align=center>
<img src="./images/image7-2.png" width = "800"/>
</div>

3、**进入循环**，global memory to shared memory：

<div align=center>
<img src="./images/image7-3.png" width = "800"/>
</div>

4、计算，并拷贝下一轮数据到寄存器：

<div align=center>
<img src="./images/image7-4.png" width = "800"/>
</div>

5、从寄存器拷贝下一轮数据到另一块shared memory：

<div align=center>
<img src="./images/image7-5.png" width = "800"/>
</div>

6、结束本轮迭代，开启下一轮迭代(回第3步)。

### 分析
不要在循环中过度使用`__syncthreads()`：过度使用`__syncthreads()`可能会降低性能，因为它会阻止线程并行地执行。参考[这篇文章](https://blog.csdn.net/weixin_43844521/article/details/133945535)。

> 以下内容摘录并修改自：https://blog.csdn.net/LostUnravel/article/details/138324342

在 GPU 上, **访存和计算对应着不同的硬件单元**, **这两个计算单元是可以并行执行的**。

**代码的顺序执行**对应的是编译后**硬件指令发射的顺序**, 指令的发射过程虽然是顺序的, 但发射速度很快, 而指令发出后需要一段时间才能执行完成, 这也就对应着某个指令需要相应的时钟周期才能完成, 访存的延迟也就是访存指令相比于计算指令有更长的时钟周期。

在 kernel 6 中, 需要两个 `__syncthreads()`, 一个是在从 global memory 加载数据到 shared memory 后, 一个是在从shared memory取数据放到register，并完成计算后。因此，kernel6中，在当前线程块的所有线程加载完数据到 shared memory 之前, 当前线程块的所有线程都无法开始计算; 同样的, 在所有线程计算完毕前, 也不能加载下一轮的数据到 shared memory。

而kernel7的双缓存实现中, 具有以下优势：
1. **线程块层面**：shared memory扩大了一倍，一半用加载下一轮的数据，一半用于当前计算，在循环开始前，先加载第一轮的数据，然后在循环中，使用当前已加载的数据进行计算，并加载下一轮数据，这样当前这一轮计算时就不必等待数据加载，因为这一轮所需的数据在上一轮已经加载完成，因此可以节约掉 1 个 `__syncthreads()`，这样在**线程块层面**, GPU 可以提前发射后面的计算的指令, 从而掩盖从 global memory 加载到 shared memory 的访存延迟。
2. **线程层面**：寄存器也扩大了一倍，采用和shared memory同样的处理思路，一半用于加载下一轮的数据，一半用于当前计算，这样在**线程层面**，GPU 可以提前发射GPU 可以提前发射后面的计算的指令，从而掩盖从 shared memory 加载到寄存器的访存延迟。

还有两点需要注意：
1. 第一次准备数据和第一次计算之间的延迟不能隐藏，因为第一次计算依赖于第一次数据的加载。而后续每次计算都会同时加载下一轮数据，所以可以隐藏访问延迟。
2. 只是节约了一个线程块内部，加载数据到计算前的那个`__syncthreads()`，但是只有当前线程块的所有线程都计算完毕后，才能移动窗口（即kernel2中引入的tile，只有当前线程块的线程都完成计算了，才能移动到下一个tile），这里的`__syncthreads()`是不能省略的。
3. 线程层面（寄存器存取）的访问延迟隐藏不是通过`__syncthreads()`实现的，只是通过避开计算和数据存取的依赖关系，让编译器提前发射计算指令。

# Q&A
## 为什么pre-fetch的时候，global memory 要先放到寄存器中再挪到 shared memory 中
答：因为硬件的限制，在安培架构之前，global memory 和shared memory没有直连，所以搬运逻辑就是先搬寄存器，再搬共享内存。

代码中哪怕直接写了global memory赋值给shared memory，中间也包含了“先写寄存器再写到shared memory”的，只是编译器隐藏了这一点。

## Kernel7的代码看上去还是顺序执行的，似乎没有做到“数据读取和计算”的重叠
建议阅读：[CUDA 学习笔记-GEMM 优化: 双缓冲 (Prefetch) 和 Bank Conflict 解决](https://blog.csdn.net/LostUnravel/article/details/138324342)

> 直观看来, 代码整体上仍然是顺序执行的逻辑, 感觉好像并不能达到 overlap 的目的, 因为还是读一个分片写一个分片的代码逻辑.
实则不然. 核心在于要理解代码对应的指令发射与执行完成的过程. 在 GPU 上, 访存和计算对应着不同的硬件单元, 这两个计算单元是可以并行执行的, **代码的顺序执行**对应的是**编译后硬件指令发射的顺序过程**, 指令的发射过程虽然是顺序的, 但发射速度很快, 而指令发出后需要一段时间才能执行完成, 这也就对应着某个指令需要相应的时钟周期才能完成, 访存的延迟也就是访存指令相比于计算指令有更长的时钟周期.

# 参考
1. https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE
2. https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
3. https://zhuanlan.zhihu.com/p/410278370
4. https://zhuanlan.zhihu.com/p/435908830
5. https://blog.csdn.net/u013013023/article/details/127245181
6. bank conflict: https://blog.csdn.net/xysjj/article/details/103885803
7. bank conflict: https://segmentfault.com/a/1190000007533157
8. vectorized loads: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
9. vectorized loads: https://www.zhihu.com/question/574968879/answer/3005751704