#include "kernel7.cuh"

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    // 当前线程对应thread tile的左上角元素在block中的位置
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[2][BK * BM]; // 增加一倍共享内存大小用于缓存
    __shared__ float Bs[2][BK * BN];


    const int ldg_a_num = BK * BM / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至As需要所有线程搬运ldg_a_num轮
    const int ldg_b_num = BK * BN / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至Bs需要所有线程搬运ldg_b_num轮

    int a_tile_row = threadIdx.x / (BK / 4); // 每行4个字节作为一个内存块，当前线程负责第a_tile_row行的第a_tile_col个内存块的搬运
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; // 一共BM行，搬运ldg_a_num轮，每论搬运a_tile_stride行

    int b_tile_row = threadIdx.x / (BN / 4); // 每行4个字节作为一个内存块，当前线程负责第b_tile_row行的第b_tile_col个内存块的搬运
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; // 一共BK行，搬运ldg_b_num轮，每论搬运b_tile_stride行

    float accum[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；

    // 计算ldg_a_num的所有参数必须全部是const，否则不能用来申明数组大小
    float ldg_a_reg[4 * ldg_a_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，并用于转置As矩阵
    float ldg_b_reg[4 * ldg_b_num] = {0.}; // 每个线程搬运ldg_b_num轮，寄存器缓存ldg_b_num个float4元素

    float a_frag[2][TM];  // 缓存As共享内存,增加一倍寄存器大小用于缓存
    float b_frag[2][TN];  // 缓存Bs共享内存,增加一倍寄存器大小用于缓存

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // first global to shared
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
        As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 不需要转置
    }
    
    int write_index = 1;
    int load_index;
    int k = 0;
    do {  // 进入循环
        __syncthreads();  // 循环开始时同步一次
        // A += BK;
        // B += BK * N;
        // 窗口滑动的逻辑直接用k写到A和B的索引中了，不用再滑动了
        k += BK;
        // load global to reg
        if (k < K) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                        FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }
        load_index = write_index ^ 1;
        // first shared to frag，这里，第109和第114行的accum[m][n]计算需要等待第一个shared to frag完成才可以继续
        // 也就是说，这里第一次加载shared memory到寄存器的操作，无法隐藏“从shared memory加载到寄存器的访存延迟”。
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(0, ty + m, BM)]); // 偏移到当前thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(0, tx + n, BN)]); // 偏移到当前thread tile
            }
        // finished first shared to frag
#pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {  // 计算了BK-1次，因为是加载下一次迭代的数据，所以可以隐藏“从shared memory加载到寄存器的访存延迟”。
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(bk + 1, ty + m, BM)]); // 偏移到当前thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); // 偏移到当前thread tile
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }
#pragma unroll
        for (int m = 0; m < TM; m++) {  // 前面只计算了BK-1次，这里计算第BK次
#pragma unroll
            for (int n = 0; n < TN; n++) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }
        // __syncthreads();  // 这里不需要同步了，因为上面的As(Bs)[load_index]和下面的As(Bs)[write_index]的内存是分开的
        if (k < K) {
            // load reg to shared
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                        FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }

            write_index ^= 1;
        }
    } while (k < K);
    
    // C = alpha*AB+C
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

// template instantiation declaration
template __global__ void sgemm_v7<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
