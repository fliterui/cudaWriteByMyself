#include "processing_gpu.cuh"
#include <device_launch_parameters.h>

#define BLOCKX 1024


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


/*__global__ void rdSquareCopy(cuFloatComplex* sout, cuFloatComplex* sin, long int M, long int N) {           //sin平方然后输出到cout里   //你说你都平方了, 还复数干啥, 改了!?
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        float x = cuCabsf(sin[i]);                                          //求模
        sout[i] = make_cuFloatComplex(x * x, 0);                            //平方, 转复数
    }
}*/


__global__ void rdSquareCopy(float* sout, cuFloatComplex* sin, long int M, long int N) {           //sin平方然后输出到cout里   //你说你都平方了, 还复数干啥, 改了!?
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        float x = cuCabsf(sin[i]);                                          //求模
        sout[i] = x*x;                                               //平方, 转复数
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
    block.x = BLOCKX;
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
     block.x = BLOCKX;
     grid.x = (M * N + block.x - 1) / block.x;
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);                                  //这个得狠狠的优化
     cufftHandle plan;
     cufftPlan1d(&plan,M,CUFFT_C2C,N);                                                                    //按理说fft点数应该大于M来着 即一般k>M
     cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
     cufftDestroy(plan);
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);                                
 }

 __global__ void CFAR(float *d_out, float* d_signal, long int M, long int N ,int rnum, int pnum, float k)                // 另外信噪比什么的都还没弄
 {
     long int i = blockIdx.x* blockDim.x + threadIdx.x;                                     //这个咋弄aaaaa
     long int col = i % N;                                                                  //i% N是纵坐标, 即每一行的第几个元素
     float thold=0;
     if (col >= rnum + pnum && col < N - rnum - pnum && i < M * N)                          //边缘的先没管 这个取元素[i-pnum+rnum:ii-pnum-1 i+pnum+1:ii+rnum+pnum]有没有优化区间捏
     {                                                                                        //这个判断信号大不大于门限的函数感觉会有一堆分支啥的
         for (int aaa = pnum + 1; aaa <=pnum + rnum; aaa++)                                                     //咋优化, 感觉是不是得换个法子算这玩意
         {
             d_out[i] = d_out[i] + d_signal[i + aaa] + d_signal[i - aaa];
         }
         d_out[i] = d_out[i] / (float)rnum;
         thold = d_out[i] * k;
         if(d_signal[i]<= thold)
         { 
             d_out[i] = 0;                                                                      //就是反过来嘛, 大于门限的保留, 小于的清零
         }
     }
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

     float* d_sqSignal;
     cudaMalloc((void**)&d_sqSignal, memSize);
     float* d_out;
     cudaMalloc((void**)&d_out, memSize);
     cudaMemset(d_out, 0, memSize);                                 //咱也不知道有没有意义这一步
     dim3 block1, grid1;
     block1.x = BLOCKX;
     grid1.x = (M * N + block1.x - 1) / block1.x;
     rdSquareCopy << <block1, grid1 >> > (d_sqSignal, d_signal, M, N);
     dim3 block2, grid2;
     block1.x = BLOCKX;
     grid1.x = (M * N + block2.x - 1) / block2.x;
    int pnum = 4;                                  //保护单元
    int rnum = 10;                                  // 参考单元
    float pfa = 1e-6;                                 // 恒虚警率               //这个可以考虑用那个什么什么常量内存啥的
    float k = powf(pfa, (-1 / (2 * (float)rnum))) - 1;
    CFAR << <block2, grid2 >> > (d_out, d_sqSignal, M, N, rnum, pnum, k);
}