#include "processing_gpu.cuh"
#include <device_launch_parameters.h>


__global__ void rdComplexMultiply(cuFloatComplex* s, cuFloatComplex* w, long int M, long int N)         //这个是nmlgb的脉冲压缩, 那那个b匹配滤波器是干啥的
{
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        long int n = i % N;
        printf("%d %d          ", i, n);
        s[i] = cuCmulf(s[i], cuConjf(w[n]));
    }
}


void readData(cuFloatComplex* signal)
{
    int a = 1;                              //不知道咋写
}

 void pulseCompression(cuFloatComplex* signal, cuFloatComplex *ori, int M, int N)
{
    size_t memSize = M * N * sizeof(cuFloatComplex);
    cuFloatComplex* d_signal, * d_ori;
    cudaMalloc((void**)&d_signal, memSize);
    cudaMalloc((void**)&d_ori, memSize / M);
    cudaMemcpy(d_signal, signal, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ori, ori, memSize, cudaMemcpyHostToDevice);

    cufftHandle plan1;
    cufftPlan1d(&plan1, N, CUFFT_C2C, M);
    cufftExecC2C(plan1, (cufftComplex*)d_signal, (cufftComplex*)d_signal, CUFFT_FORWARD);
    cufftDestroy(plan1);

    cufftHandle plan2;
    cufftPlan1d(&plan2, N, CUFFT_C2C, 1);
    cufftExecC2C(plan2, (cufftComplex*)d_ori, (cufftComplex*)d_ori, CUFFT_FORWARD);
    cufftDestroy(plan2);
    
    dim3 block, grid;
    block.x = 1024;
    grid.x = (M * N + block.x - 1) / block.x;
    rdComplexMultiply<<<block,grid>>>(d_signal, d_ori, M, N);                               //我突然发现,他都没用过流, nvprof和啥nightSystem啥的也都没分析过
    
    cufftHandle plan3;
    cufftPlan1d(&plan3, M, CUFFT_C2C, N);
    cufftExecC2C(plan3, d_signal, d_signal, CUFFT_INVERSE);
    cufftDestroy(plan3);
}