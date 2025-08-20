# 使用cuBLAS中的矩阵乘法
> 参考博文：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html

存储矩阵时，C/C++中是行优先，而cuBLAS是列优先，利用公式 $AB={(B^TA^T)}^T$，调整输入：
```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, *A, m, *B, k, &beta, *C, m);
                                              ↓  ↓              ↓  ↓   ↓                ↓
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, *B, n, *A, k, &beta, *C, n);
```

编译：
```bash
mkdir build
cd build
cmake ..
make
```

运行：
```bash
./cublas_exmple
```

输出：
```
C=
38.00   44.00   50.00   56.00
83.00   98.00   113.00  128.0
```