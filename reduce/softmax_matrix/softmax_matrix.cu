#include <cstdio>
#include <cstdlib>
#include <cuda_rutime.h>
#include <algorithm>
#include <float.h>
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


__global__ void softmax(float *input,float *output,int h,int w){
   int row=blockIdx.x;
   int laneid=threadIdx.x%warpSize;
   int col=threadIdx.x;
   __shared__ s_max;
   __shared__ s_sum;
   if(row<h){
      float max_val=-FLT_MAX;
      #pragma unroll   
      for(int i=col;i<w;i+=warpSize){
      	 max_val=fmaxf(max_val,input[row*w+i]);
      }
      #pragma unroll
      for(int offset=warpSize>>1;offset>0;offset>>=1){
         max_val=fmaxf(max_val,__shfl_down_sync(0xffffffff,max_val,offset));
      }
      if(laneid==0)s_max=max_val;
      float sum_val=0.f;
      #pragma unroll
      for(int i=col;i<w;i+=warpSize){
         sum_val+=expf(input[row*w+i]-s_max);
      }      
      #pragma unroll
      for(int offset=warpSize>>1;offset>0;offset>>=1){
         sum_val+=__shfl_down_sync(0xffffffff,sum_val,offset);
      }
      if(laneid==0)s_sum=sum_val;
      #pragma unroll
      for(int i=col;i<w;i+=warpSize){
         output[row*w+i]=expf(input[row*w+i]-s_max)/s_sum;
      }
   }


}

__inline__ __device__ float warpReduceMax(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  return v; // 通常返回每个 lane 都相同的规约结果（或至少 lane0 正确）
}
__inline__ __device__ float warpReduceSum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}




__global__ void softmax1(float *input,float *output,int h,int w){
      int idx=blockIdx.x;
      int tid=threadIdx.x;
      int warpid=threadIdx.x/32;
      int laneid=threadIdx.x%32;
      extern __shared__ float shared[];
      int warpsPerBlock=blockDim.x/32;
      if(idx>=h)return;
      float max_val=-FLT_MAX;
      const float *x=idx*w+input;
      for(int i=tid;i<w;i+=blockDim.x){
         max_val=fmaxf(max_val,x[i]);
      }
      
      max_val=warpReduceMax(max_val);
      if(laneid==0)shared[warpid]=max_val;
      __syncthreads();
      if(warpid==0){
         max_val=(laneid<warpsPerBlock)?shared[laneid]:-FLT_MAX;
         max_val=warpReduceMax(max_val);
         if(laneid==0)shared[0]=max_val;
      }
      __syncthreads();
      float offset=shared[0];
      for(int i=tid;i<w;i+=blockDim.x){
         output[i+idx*w]=expf(x[i]-offset);
      }
      float sum_val=0.f;
      x=output+idx*w;
      for(int i=tid;i<w;i+=blockDim.x){
         sum_val+=x[i];
      }
      sum_val=warpReduceSum(sum_val);
      if(laneid==0)shared[warpid]=sum_val;
      __syncthreads();
      if(warpid==0){
         sum_val=(laneid<warpsPerBlock)?shared[laneid]:0.f;
         sum_val=warpReduceSum(sum_val);
         if(laneid==0)shared[0]=sum_val;
      }
      __syncthreads();
      sum_val=shared[0];
      for(int i=tid;i<w;i+=blockDim.x){
         x[i]/=sum_val;
      }

}