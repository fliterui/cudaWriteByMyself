#include "processing_gpu.cuh"
#include <device_launch_parameters.h>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#define BLOCKX 1024


__global__ void rdComplexMultiply(cuFloatComplex* s, cuFloatComplex* w, long int M, long int N)         //�����nmlgb������ѹ��, ���Ǹ�bƥ���˲����Ǹ�ɶ��
{
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        long int n = i % N;
        //printf("%d %d          ", i, n);
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


/*__global__ void rdSquareCopy(cuFloatComplex* sout, cuFloatComplex* sin, long int M, long int N) {           //sinƽ��Ȼ�������cout��   //��˵�㶼ƽ����, ��������ɶ, ����!?
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        float x = cuCabsf(sin[i]);                                          //��ģ
        sout[i] = make_cuFloatComplex(x * x, 0);                            //ƽ��, ת����
    }
}*/


__global__ void rdSquareCopy(float* sout, cuFloatComplex* sin, long int M, long int N) {           //sinƽ��Ȼ�������cout��   //��˵�㶼ƽ����, ��������ɶ, ����!?
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        float x = cuCabsf(sin[i]);                                          //��ģ
        sout[i] = x*x;                                               //ƽ��, ת����
    }
}



void readData(cuFloatComplex* signal, cuFloatComplex *ori, long int M, long int N)                    //�����ʱ���ù̶��ڴ��ܺõ�
{
    FILE* fp;//�ļ�ָ��
    fp = fopen("signal_real.txt", "r");//���ı���ʽ���ļ���
    if (fp == NULL) //���ļ�����
        printf("error1");
    for (int i = 0; i<M*N; i++)
    {
        fscanf(fp, "%lf", &signal[i].x);
       /* if (signal[i].x == EOF)
        {
            printf("signal num is %d", i);
            break;
        }*/
    }
    fclose(fp);//�ر��ļ�
    fp = fopen("signal_imag.txt", "r");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i < M * N; i++)
    {
        fscanf(fp, "%lf", &signal[i].y);
        /*if (signal[i].y == EOF)
        {
            break;
        }*/
    }
    fclose(fp);
    fp = fopen("ori_real.txt", "r");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i<N; i++)
    {
        fscanf(fp, "%lf", &ori[i].x);
        /*if (ori[i].x == EOF)
        {
            printf("ori num is %d", i);
            break;
        }*/
    }
    fclose(fp);
    fp = fopen("ori_imag.txt", "r");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i < N; i++)
    {
        fscanf(fp, "%lf", &ori[i].y);
       /* if (ori[i].y == EOF)
        {
            break;
        }*/
    }
    fclose(fp);
    printf("������, �����! \n");
}

void writeData(float *result, long int M, long int N)
{
    FILE* fpWrite;
    fpWrite = fopen("out.txt", "w");
    if (fpWrite == NULL)
    {
        printf("error");
        return;
    }
    for (int i = 0; i < M * N; i++)
        fprintf(fpWrite, "%2.15f\n", result[i]);
    fclose(fpWrite);
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
     block.x = BLOCKX;
     grid.x = (M * N + block.x - 1) / block.x;
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);                                  //����úݺݵ��Ż�
     cufftHandle plan;
     cufftPlan1d(&plan,M,CUFFT_C2C,N);                                                                    //����˵fft����Ӧ�ô���M���� ��һ��k>M
     cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
     cufftDestroy(plan);
     rdComplexTranspose << <block, grid >> > (d_signal, d_signal, M, N);                                
 }

 __global__ void CFAR(float *d_out, float* d_signal, long int M, long int N ,int rnum, int pnum, float k)                // ���������ʲô�Ķ���ûŪ
 {
     long int i = blockIdx.x* blockDim.x + threadIdx.x;                                     //���զŪaaaaa
     long int col = i % N;                                                                  //i% N��������, ��ÿһ�еĵڼ���Ԫ��
     float thold=0;
     if (col >= rnum + pnum && col < N - rnum - pnum && i < M * N)                          //��Ե����û�� ���ȡԪ��[i-pnum+rnum:ii-pnum-1 i+pnum+1:ii+rnum+pnum]��û���Ż�������
     {                                                                                        //����ж��źŴ󲻴������޵ĺ����о�����һ�ѷ�֧ɶ��
         for (int aaa = pnum + 1; aaa <=pnum + rnum; aaa++)                                                     //զ�Ż�, �о��ǲ��ǵû���������������
         {
             d_out[i] = d_out[i] + d_signal[i + aaa] + d_signal[i - aaa];
         }
         d_out[i] = d_out[i] / (float)rnum;
         thold = d_out[i] * k;
         if(d_signal[i]<= thold)
         { 
             d_out[i] = 0;                                                                      //���Ƿ�������, �������޵ı���, С�ڵ�����
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
     cudaMemset(d_out, 0, memSize);                                 //��Ҳ��֪����û��������һ��         �е�, ��Ȼ��Ե�ľ�û��ֵ��
     dim3 block1, grid1;
     block1.x = BLOCKX;
     grid1.x = (M * N + block1.x - 1) / block1.x;
     rdSquareCopy << <block1, grid1 >> > (d_sqSignal, d_signal, M, N);
     dim3 block2, grid2;
     block1.x = BLOCKX;
     grid1.x = (M * N + block2.x - 1) / block2.x;
    int pnum = 4;                                  //������Ԫ
    int rnum = 10;                                  // �ο���Ԫ
    float pfa = 1e-6;                                 // ���龯��               //������Կ������Ǹ�ʲôʲô�����ڴ�ɶ��
    float k = powf(pfa, (-1 / (2 * (float)rnum))) - 1;
    CFAR << <block2, grid2 >> > (d_out, d_sqSignal, M, N, rnum, pnum, k);

    float* out;
    cudaMallocHost((void**)&out, memSize);
    cudaMemcpy(out, d_out, memSize, cudaMemcpyDeviceToHost);
    writeData(out, M, N);
    printf("��������, ����� \n");
}