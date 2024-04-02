#include "processing_gpu.cuh"
#include <device_launch_parameters.h>


__global__ void rdComplexMultiply(cuFloatComplex* s, cuFloatComplex* w, long int M, long int N)         //�����nmlgb������ѹ��, ���Ǹ�bƥ���˲����Ǹ�ɶ��
{
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        long int n = i % N;
        printf("%d %d          ", i, n);
        s[i] = cuCmulf(s[i], cuConjf(w[n]));
    }
}


__global__ void rdComplexTranspose(cuFloatComplex* sout, cuFloatComplex* sin, long int M, long int N)       //����ת��???   �ǵ�
{
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        long int n = i % N;                                                             //�������
        long int m = (long int)(i - n) / N;                                             //�������

        sout[m + n * M] = sin[n + m * N];
    }
 }


void readData(cuFloatComplex* signal)
{
    int a = 1;                              //��֪��զд     ���ص���Host����
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
    rdComplexMultiply<<<block,grid>>>(d_signal, d_ori, M, N);                               //��ͻȻ����,����û�ù���, nvprof��ɶnightSystemɶ��Ҳ��û������
    
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
     cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToDevice);               //����, ���滹��ת�û�ȥ, cfarҪ��mtd�Ľ��*/
     dim3 block, grid;
     block.x = 1024;
     grid.x = (M * N + block.x - 1) / block.x;
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);                                  //����úݺݵ��Ż�
     cufftHandle plan;
     cufftPlan1d(&plan,M,CUFFT_C2C,N);
     cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
     cufftDestroy(plan);
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);
 }

 void CFAR(cuFloatComplex* d_signal, long int M, long int N, int rnum, int prum)                                  //���ȡԪ��[i-pnum+rnum:ii-pnum-1 i+pnum+1:ii+rnum+pnum]��û���Ż�������, ���������ʲô�Ķ���ûŪ
 {
     long int i = blockIdx.x* blockDim.x + threadIdx.x;         //���զŪaaaaa
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