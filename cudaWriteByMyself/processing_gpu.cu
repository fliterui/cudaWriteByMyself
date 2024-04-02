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


__global__ void rdComplexTranspose(cuFloatComplex* sout, cuFloatComplex* sin, long int M, long int N)       //矩阵转置???   是的
{
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        long int n = i % N;                                                             //这个是行
        long int m = (long int)(i - n) / N;                                             //这个是列

        sout[m + n * M] = sin[n + m * N];
    }
 }


void readData(cuFloatComplex* signal)
{
    int a = 1;                              //不知道咋写     返回的是Host变量
}

 void pulseCompression(cuFloatComplex* d_signal, cuFloatComplex * d_ori, long int M, long int N)
 { 

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
    cufftPlan1d(&plan3, N, CUFFT_C2C, M);
    cufftExecC2C(plan3, d_signal, d_signal, CUFFT_INVERSE);
    cufftDestroy(plan3);

 }

 void mtd(cuFloatComplex* d_signal, long int M, long int N)
 {
     /*cuFloatComplex* signal;
     size_t memSize = M* N * sizeof(cuFloatComplex);
     cudaMalloc((void**)&signal, memSize);
     cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToDevice);               //不对, 后面还得转置回去, cfar要用mtd的结果*/
     dim3 block, grid;
     block.x = 1024;
     grid.x = (M * N + block.x - 1) / block.x;
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);                                  //这个得狠狠的优化
     cufftHandle plan;
     cufftPlan1d(&plan,M,CUFFT_C2C,N);
     cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
     cufftDestroy(plan);
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);
 }

 void CFAR(cuFloatComplex* d_signal, long int M, long int N, int rnum, int prum)                                  //这个取元素[i-pnum+rnum:ii-pnum-1 i+pnum+1:ii+rnum+pnum]有没有优化区间捏, 另外信噪比什么的都还没弄
 {
     long int i = blockIdx.x* blockDim.x + threadIdx.x;         //这个咋弄aaaaa
     if(i>)
 }
 void doGpuProcessing(cuFloatComplex* signal, cuFloatComplex* ori, long int M, long int N)
{
     size_t memSize = M * N * sizeof(cuFloatComplex);
     cuFloatComplex* d_signal, * d_ori;
     cudaMalloc((void**)&d_signal, memSize);
     cudaMalloc((void**)&d_ori, memSize / M);
     cudaMemcpy(d_signal, signal, memSize, cudaMemcpyHostToDevice);
     cudaMemcpy(d_ori, ori, memSize, cudaMemcpyHostToDevice);
     pulseCompression(d_signal, d_ori, M, N);
     mtd(d_signal, M, N);

}