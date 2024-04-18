#include "processing_gpu.cuh"
#include <device_launch_parameters.h>

//#include "matplotlibcpp.h"
//namespace plt = matplotlibcpp;

#define BLOCKX 256

void dev_setup(int M,int N)
{
    // set up device
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    printf("  GPU�豸����                         %s \n", deviceProp.name);
    printf("  GPU������������SM������             %d \n", (int)deviceProp.multiProcessorCount);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver �汾 / Runtime �汾     %d.%d / %d.%d\n",
        driverVersion / 1000, (driverVersion % 100) / 10,
        runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA ������:                       %d.%d\n",
        deviceProp.major, deviceProp.minor);
    printf("  �Դ��С:                          %.2f GBytes (%llu "
        "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
        (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU ʱ��Ƶ��:                      %.0f MHz (%0.2f "
        "GHz)\n", deviceProp.clockRate * 1e-3f,
        deviceProp.clockRate * 1e-6f);
    printf("  Memory ʱ��Ƶ��:                   %.0f Mhz\n",
        deviceProp.memoryClockRate * 1e-3f);
    printf("  �����ģ��                         %d * %d\n", M, N);
}

__global__ void rdComplexMultiply(cuDoubleComplex* s, cuDoubleComplex* w, int M, int N)         //�����nmlgb������ѹ��, ���Ǹ�bƥ���˲����Ǹ�ɶ��
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        int n = i % N;
        //printf("%d %d          ", i, n);
        s[i] = cuCmul(s[i], cuConj(w[n]));
    }
}


__global__ void rdComplexTranspose(cuDoubleComplex* sout, cuDoubleComplex* sin, int M, int N)       //����ת��???   �ǵ�      ��������nmdgb bug, sin��sout������ͬһ����Ȼ���ͻ
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        int n = i % N;                                                             //�������
        int m = (int)(i - n) / N;                                             //�������

        sout[m + n * M] = sin[n + m * N];
    }
 }


/*__global__ void rdSquareCopy(cuDoubleComplex* sout, cuDoubleComplex* sin, int M, int N) {           //sinƽ��Ȼ�������cout��   //��˵�㶼ƽ����, ��������ɶ, ����!?
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        double x = cuCabsf(sin[i]);                                          //��ģ
        sout[i] = make_cuFloatComplex(x * x, 0);                            //ƽ��, ת����
    }
}*/


__global__ void rdSquareCopy(double* sout, cuDoubleComplex* sin, int M, int N) {           //sinƽ��Ȼ�������cout��   //��˵�㶼ƽ����, ��������ɶ, ����!?
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N * M)
    {
        double x = cuCabs(sin[i]);                                          //��ģ
        sout[i] = x*x;                                               //ƽ��, ת����
    }
    
}



void readData(cuDoubleComplex* signal, cuDoubleComplex *ori, int M, int N)                    //�����ʱ���ù̶��ڴ��ܺõ�
{
    FILE* fp;//�ļ�ָ��
    fp = fopen("signal_real.txt", "r");//���ı���ʽ���ļ���
    if (fp == NULL) //���ļ�����
        printf("error1");
    for (int i = 0; i<M*N; i++)
    {
        fscanf(fp, "%lf", &signal[i].x);
  
    }
    fclose(fp);//�ر��ļ�
    fp = fopen("signal_imag.txt", "r");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i < M * N; i++)
    {
        fscanf(fp, "%lf", &signal[i].y);

    }
    fclose(fp);
    fp = fopen("ori_real.txt", "r");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i<N; i++)
    {
        fscanf(fp, "%lf", &ori[i].x);

    }
    fclose(fp);
    fp = fopen("ori_imag.txt", "r");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i < N; i++)
    {
        fscanf(fp, "%lf", &ori[i].y);
    }
    fclose(fp);
    printf("������, �����! \n");
}

void writeDataComplex(cuDoubleComplex* d_signal, int M, int N)                    
{
    int memSize = M * N * sizeof(cuDoubleComplex);
    cuDoubleComplex* signal;
    cudaMallocHost((void**)&signal,memSize);
    cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToHost);
    //cudaMemset(signal, 1145, memSize);
    test(d_signal, M, N);
    FILE* fp;//�ļ�ָ��
    fp = fopen("signal_real_out.txt", "w");//���ı���ʽ���ļ���
    printf("caonima, %lf", signal[6].y);
    if (fp == NULL) //���ļ�����
        printf("error1");
    for (int i = 0; i < M * N; i++)
    {
        fprintf(fp, "%lf\n", signal[i].x);

    }
    fclose(fp);//�ر��ļ�
    fp = fopen("signal_imag_out.txt", "w");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i < M * N; i++)
    {
        fprintf(fp, "%lf\n", signal[i].y);
        

    }
    fclose(fp);
    
    printf("д����, �����! \n");
}

void writeData (double *d_signal, int M, int N)               //����������gpu���ڴ������, �����Զ����㿽����Host��
{
    size_t memSize = M * N * sizeof(double);
    double* out;
    cudaMallocHost((void**)&out, memSize);
    cudaMemcpy(out, d_signal, memSize, cudaMemcpyDeviceToHost);
    //out[1] = 1;
    FILE* fpWrite;
    fpWrite = fopen("writeData_out.txt", "w");
    if (fpWrite == NULL)
    {
        printf("error");
        return;
    }
    for (int i = 0; i < M * N; i++)
        fprintf(fpWrite, "%2.15f\n", out[i]);
    fclose(fpWrite);
}

/*����ѹ��
* d_signal�ǻز�, d_ori��sin�ź�, M, N���źŵ�M��N��
*/

 void pulseCompression(cuDoubleComplex* d_signal, cuDoubleComplex * d_ori, int M, int N)                
 { 
    int inembed[1] = { 0 };
    int onembed[1] = { 0 };
    int number_N[1] = { (int)N };                                         //�ⲻ��longint��ͻ��
    int istride = 1;
    int rank = 1;
    int ostride = 1;
    cufftHandle plan1;
    
    //cufftPlanMany(&plan1, rank, number_N, inembed, istride, N, onembed, ostride, N, CUFFT_Z2Z, M);
    cufftPlan1d(&plan1, N, CUFFT_Z2Z, M);
    cufftExecZ2Z(plan1, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_FORWARD);               //�ź�fft, �������
    //printf("fuck ");
    //printGpuModComplex(d_signal);
    cufftDestroy(plan1);
    //test(d_signal, M, N);
    cufftHandle plan2;
    cufftPlan1d(&plan2, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan2, (cufftDoubleComplex*)d_ori, (cufftDoubleComplex*)d_ori, CUFFT_FORWARD);                     //sin fft, �������
    cufftDestroy(plan2);
    
    dim3 block, grid;
    block.x = BLOCKX;
    grid.x = (M * N + block.x - 1) / block.x;
    //printf("%d, %d \n", block.x, grid.x);
    rdComplexMultiply<<<grid, block >>>(d_signal, d_ori, M, N);                                              //���Թ���ֱ, ֱ�Ӹĵ�d_signal
    //test(d_ori, 1, N);                                                                                                                                               //��ͻȻ����,����û�ù���, nvprof��ɶnightSystemɶ��Ҳ��û������
    
    cufftHandle plan3;
    cufftPlan1d(&plan3, N, CUFFT_Z2Z, M);
    cufftExecZ2Z(plan3, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_INVERSE);                                                 //ifft
    cufftDestroy(plan3);
 }

 /*mtd
* d_signal�ǻز�, d_ori��sin�ź�, M, N���źŵ�M��N��
*/

 void mtd(cuDoubleComplex* d_signal, int M, int N)
 {
     /*cuDoubleComplex* signal;
     size_t memSize = M* N * sizeof(cuDoubleComplex);
     cudaMalloc((void**)&signal, memSize);  
     cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToDevice);                                                                         //����, ���滹��ת�û�ȥ, cfarҪ��mtd�Ľ��*/
     dim3 block, grid;
     block.x = BLOCKX;
     grid.x = (M * N + block.x - 1) / block.x;
     cuDoubleComplex* dd_signal;
     size_t memSize = M * N * sizeof(cuDoubleComplex);
     cudaMalloc((void**)&dd_signal, memSize);
     rdComplexTranspose <<< grid, block >>> (dd_signal, d_signal, M, N);                                    //��ת�ú����е�fft           //����úݺݵ��Ż�
     //writeDataComplex(dd_signal, M, N);
     cufftHandle plan;
     cufftPlan1d(&plan,M,CUFFT_Z2Z,N);                                                                                                          //����˵fft����Ӧ�ô���M���� ��һ��k>M
     cufftExecZ2Z(plan, (cufftDoubleComplex*)dd_signal, (cufftDoubleComplex*)dd_signal, CUFFT_FORWARD);                                                 //��fft
     cufftDestroy(plan);
     rdComplexTranspose <<< grid, block >>> (d_signal, dd_signal, N, M);                                 //ת�û�ȥ��ʱ����N��M��, ������N,M!!!!!
     cudaFree(dd_signal);
 }


 /*CFAR
* d_signal�ǻز�, d_ori��sin�ź�, M, N���źŵ�M��N��, rnum�ǲο���Ԫ����, pnum�Ǳ�����Ԫ����, k���ĸ����ڹ���ƽ��ֵ�ϵ��Ǹ�ϵ��(����pfa�����˵��Ǹ�)
*/

 __global__ void CFAR(double *d_out, double* d_signal, int M, int N ,int rnum, int pnum, double k)                                                  // ���������ʲô�Ķ���ûŪ
 {
     int i = blockIdx.x* blockDim.x + threadIdx.x;                                                                                                            //���զŪaaaaa
     int col = i % N;                                                                  //i% N��������, ��ÿһ�еĵڼ���Ԫ��
     double thold=0;
     if (col >= rnum + pnum && col < N - rnum - pnum && i < M * N)                          //��Ե����û�� ���ȡԪ��[i-pnum+rnum:ii-pnum-1 i+pnum+1:ii+rnum+pnum]��û���Ż�������
     {                                                                                        //����ж��źŴ󲻴������޵ĺ����о�����һ�ѷ�֧ɶ��
         for (int aaa = pnum + 1; aaa <=pnum + rnum; aaa++)                                                     //զ�Ż�, �о��ǲ��ǵû���������������
         {
             d_out[i] = d_out[i] + d_signal[i + aaa] + d_signal[i - aaa];
         }
         d_out[i] = d_out[i] / (double)rnum;
         thold = d_out[i] * k;
         if(d_signal[i]<= thold)
         { 
             d_out[i] = 0;                                                                      //���Ƿ�������, �������޵ı���, С�ڵ�����    //�о���Ҳ���Ż��ռ�
         }
     }
 }
 /*
 void test(cuDoubleComplex* d_signal, int M, int N)
 {
     size_t memSize;
     memSize = M * N * sizeof(double);
     double* a;
     cudaMalloc((void**)&a, memSize);
     cudaMemset(a, 0, memSize);
     dim3 blockkk, griddd;
     blockkk.x = 1024;
     griddd.x = (M * N + blockkk.x - 1) / blockkk.x;
     rdSquareCopy << < griddd, blockkk >> > (a, d_signal, M, N);
     printf("test out: ");
     printGpuModFloat(a);
     writeData(a, M, N);
     cudaFree(a);
 }

 void printGpuModComplex(cuDoubleComplex  *d_signal)
 {
     cuDoubleComplex* a;
     cudaMallocHost((void**)&a, sizeof(cuDoubleComplex));
     cudaMemcpy(a, d_signal, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
     printf("%f   \n", cuCabs(a[0]));
     cudaFree(a);
 }
 
 void printGpuModFloat(double* d_signal)
 {
     double* a;
     cudaMallocHost((void**)&a, sizeof(double));
     cudaMemcpy(a, d_signal, sizeof(double), cudaMemcpyDeviceToHost);
     printf("%f\n", a[0]);
 }

 void makeSmall(cuDoubleComplex* d_signal, int M, int N)
 {
     dim3 block, grid;
     block.x = BLOCKX;
     grid.x = (M * N + block.x - 1) / block.x;
     mod1w << <grid, block >> > (d_signal, M, N);
 }


 __global__ void mod1w(cuDoubleComplex* d_signal, int M, int N)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < M * N)
     {
         d_signal[i] = make_cuDoubleComplex(cuCreal(d_signal[i]) / 100000, cuCimag(d_signal[i]) / 100000);
     }
 }
 */
 LARGE_INTEGER nFreq;
 LARGE_INTEGER nLastTime1;
 LARGE_INTEGER nLastTime2;
 static int __count = 0;
 float doGpuProcessing(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N)
{
     //printf("%d ", ++__count);
     QueryPerformanceFrequency(&nFreq);
     QueryPerformanceCounter(&nLastTime1);
     size_t memSize = M * N * sizeof(cuDoubleComplex);
     cuDoubleComplex* d_signal, * d_ori;
     cudaMalloc((void**)&d_signal, memSize);
     cudaMalloc((void**)&d_ori, memSize / M);
     cudaMemcpy(d_signal, signal, memSize, cudaMemcpyHostToDevice);
     cudaMemcpy(d_ori, ori, memSize/M, cudaMemcpyHostToDevice);                           //signal��oriŪ��gpu��
     //test(d_signal, M, N);
     pulseCompression(d_signal, d_ori, M, N);                                           //����ѹ��
     //writeDataComplex(d_signal, M, N);
     //makeSmall(d_signal, M, N);
     //cudaDeviceSynchronize();
     //test(d_signal, M, N);                                                            
     //makeSmall(d_signal, M, N);
     mtd(d_signal, M, N);                                                               //����ѹ���Ľ���͵�mtd
     //test(d_signal, M, N);
     double* d_sqSignal;
     cudaMalloc((void**)&d_sqSignal, memSize);
     double* d_out;
     cudaMalloc((void**)&d_out, memSize);
     cudaMemset(d_out, 1, memSize);                                 //��Ҳ��֪����û��������һ��         �е�, ��Ȼ��Ե�ľ�û��ֵ��
     dim3 block1, grid1;
     block1.x = BLOCKX;
     grid1.x = (M * N + block1.x - 1) / block1.x;
     rdSquareCopy << < grid1, block1 >> > (d_sqSignal, d_signal, M, N);
     dim3 block2, grid2;
     block2.x = BLOCKX;
     grid2.x = (M * N + block2.x - 1) / block2.x;
     int pnum = 4;                                  //������Ԫ
     int rnum = 10;                                  // �ο���Ԫ
     double pfa = 1e-6;                                 // ���龯��               //������Կ������Ǹ�ʲôʲô�����ڴ�ɶ��
     double k = powf(pfa, (-1 / (2 * (double)rnum))) - 1;
     CFAR << < grid2, block2 >> > (d_out, d_sqSignal, M, N, rnum, pnum, k);
     QueryPerformanceCounter(&nLastTime2);
     float fInterval = nLastTime2.QuadPart - nLastTime1.QuadPart;
    
     //writeData(d_out, M, N);
     //printf("��������, ����� \n");
     cudaFree(d_signal);
     cudaFree(d_ori);
     cudaFree(d_sqSignal);
     cudaFree(d_out);
     return  fInterval / (float)nFreq.QuadPart;
}