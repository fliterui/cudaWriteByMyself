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

    printf("  GPU设备名称                         %s \n", deviceProp.name);
    printf("  GPU中流处理器（SM）个数             %d \n", (int)deviceProp.multiProcessorCount);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver 版本 / Runtime 版本     %d.%d / %d.%d\n",
        driverVersion / 1000, (driverVersion % 100) / 10,
        runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA 计算力:                       %d.%d\n",
        deviceProp.major, deviceProp.minor);
    printf("  显存大小:                          %.2f GBytes (%llu "
        "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
        (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU 时钟频率:                      %.0f MHz (%0.2f "
        "GHz)\n", deviceProp.clockRate * 1e-3f,
        deviceProp.clockRate * 1e-6f);
    printf("  Memory 时钟频率:                   %.0f Mhz\n",
        deviceProp.memoryClockRate * 1e-3f);
    printf("  矩阵规模：                         %d * %d\n", M, N);
}

__global__ void rdComplexMultiply(cuDoubleComplex* s, cuDoubleComplex* w, int M, int N)         //这个是nmlgb的脉冲压缩, 那那个b匹配滤波器是干啥的
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        int n = i % N;
        //printf("%d %d          ", i, n);
        s[i] = cuCmul(s[i], cuConj(w[n]));
    }
}


__global__ void rdComplexTranspose(cuDoubleComplex* sout, cuDoubleComplex* sin, int M, int N)       //矩阵转置???   是的      这玩意有nmdgb bug, sin和sout不能是同一个不然会冲突
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        int n = i % N;                                                             //这个是行
        int m = (int)(i - n) / N;                                             //这个是列

        sout[m + n * M] = sin[n + m * N];
    }
 }


/*__global__ void rdSquareCopy(cuDoubleComplex* sout, cuDoubleComplex* sin, int M, int N) {           //sin平方然后输出到cout里   //你说你都平方了, 还复数干啥, 改了!?
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        double x = cuCabsf(sin[i]);                                          //求模
        sout[i] = make_cuFloatComplex(x * x, 0);                            //平方, 转复数
    }
}*/


__global__ void rdSquareCopy(double* sout, cuDoubleComplex* sin, int M, int N) {           //sin平方然后输出到cout里   //你说你都平方了, 还复数干啥, 改了!?
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N * M)
    {
        double x = cuCabs(sin[i]);                                          //求模
        sout[i] = x*x;                                               //平方, 转复数
    }
    
}



void readData(cuDoubleComplex* signal, cuDoubleComplex *ori, int M, int N)                    //这个到时候用固定内存能好点
{
    FILE* fp;//文件指针
    fp = fopen("signal_real.txt", "r");//以文本方式打开文件。
    if (fp == NULL) //打开文件出错。
        printf("error1");
    for (int i = 0; i<M*N; i++)
    {
        fscanf(fp, "%lf", &signal[i].x);
  
    }
    fclose(fp);//关闭文件
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
    printf("读完了, 你真棒! \n");
}

void writeDataComplex(cuDoubleComplex* d_signal, int M, int N)                    
{
    int memSize = M * N * sizeof(cuDoubleComplex);
    cuDoubleComplex* signal;
    cudaMallocHost((void**)&signal,memSize);
    cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToHost);
    //cudaMemset(signal, 1145, memSize);
    test(d_signal, M, N);
    FILE* fp;//文件指针
    fp = fopen("signal_real_out.txt", "w");//以文本方式打开文件。
    printf("caonima, %lf", signal[6].y);
    if (fp == NULL) //打开文件出错。
        printf("error1");
    for (int i = 0; i < M * N; i++)
    {
        fprintf(fp, "%lf\n", signal[i].x);

    }
    fclose(fp);//关闭文件
    fp = fopen("signal_imag_out.txt", "w");
    if (fp == NULL)
        printf("error");
    for (int i = 0; i < M * N; i++)
    {
        fprintf(fp, "%lf\n", signal[i].y);
        

    }
    fclose(fp);
    
    printf("写完了, 你真棒! \n");
}

void writeData (double *d_signal, int M, int N)               //这个输入的是gpu的内存就行了, 他会自动给你拷贝到Host里
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

/*脉冲压缩
* d_signal是回波, d_ori是sin信号, M, N是信号的M行N列
*/

 void pulseCompression(cuDoubleComplex* d_signal, cuDoubleComplex * d_ori, int M, int N)                
 { 
    int inembed[1] = { 0 };
    int onembed[1] = { 0 };
    int number_N[1] = { (int)N };                                         //这不跟longint冲突了
    int istride = 1;
    int rank = 1;
    int ostride = 1;
    cufftHandle plan1;
    
    //cufftPlanMany(&plan1, rank, number_N, inembed, istride, N, onembed, ostride, N, CUFFT_Z2Z, M);
    cufftPlan1d(&plan1, N, CUFFT_Z2Z, M);
    cufftExecZ2Z(plan1, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_FORWARD);               //信号fft, 用来卷积
    //printf("fuck ");
    //printGpuModComplex(d_signal);
    cufftDestroy(plan1);
    //test(d_signal, M, N);
    cufftHandle plan2;
    cufftPlan1d(&plan2, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan2, (cufftDoubleComplex*)d_ori, (cufftDoubleComplex*)d_ori, CUFFT_FORWARD);                     //sin fft, 用来卷积
    cufftDestroy(plan2);
    
    dim3 block, grid;
    block.x = BLOCKX;
    grid.x = (M * N + block.x - 1) / block.x;
    //printf("%d, %d \n", block.x, grid.x);
    rdComplexMultiply<<<grid, block >>>(d_signal, d_ori, M, N);                                              //乘以共轭直, 直接改的d_signal
    //test(d_ori, 1, N);                                                                                                                                               //我突然发现,他都没用过流, nvprof和啥nightSystem啥的也都没分析过
    
    cufftHandle plan3;
    cufftPlan1d(&plan3, N, CUFFT_Z2Z, M);
    cufftExecZ2Z(plan3, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_INVERSE);                                                 //ifft
    cufftDestroy(plan3);
 }

 /*mtd
* d_signal是回波, d_ori是sin信号, M, N是信号的M行N列
*/

 void mtd(cuDoubleComplex* d_signal, int M, int N)
 {
     /*cuDoubleComplex* signal;
     size_t memSize = M* N * sizeof(cuDoubleComplex);
     cudaMalloc((void**)&signal, memSize);  
     cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToDevice);                                                                         //不对, 后面还得转置回去, cfar要用mtd的结果*/
     dim3 block, grid;
     block.x = BLOCKX;
     grid.x = (M * N + block.x - 1) / block.x;
     cuDoubleComplex* dd_signal;
     size_t memSize = M * N * sizeof(cuDoubleComplex);
     cudaMalloc((void**)&dd_signal, memSize);
     rdComplexTranspose <<< grid, block >>> (dd_signal, d_signal, M, N);                                    //先转置好做列的fft           //这个得狠狠的优化
     //writeDataComplex(dd_signal, M, N);
     cufftHandle plan;
     cufftPlan1d(&plan,M,CUFFT_Z2Z,N);                                                                                                          //按理说fft点数应该大于M来着 即一般k>M
     cufftExecZ2Z(plan, (cufftDoubleComplex*)dd_signal, (cufftDoubleComplex*)dd_signal, CUFFT_FORWARD);                                                 //做fft
     cufftDestroy(plan);
     rdComplexTranspose <<< grid, block >>> (d_signal, dd_signal, N, M);                                 //转置回去的时候是N列M行, 所以是N,M!!!!!
     cudaFree(dd_signal);
 }


 /*CFAR
* d_signal是回波, d_ori是sin信号, M, N是信号的M行N列, rnum是参考单元个数, pnum是保护单元个数, k是哪个乘在功率平均值上的那个系数(根据pfa算完了的那个)
*/

 __global__ void CFAR(double *d_out, double* d_signal, int M, int N ,int rnum, int pnum, double k)                                                  // 另外信噪比什么的都还没弄
 {
     int i = blockIdx.x* blockDim.x + threadIdx.x;                                                                                                            //这个咋弄aaaaa
     int col = i % N;                                                                  //i% N是纵坐标, 即每一行的第几个元素
     double thold=0;
     if (col >= rnum + pnum && col < N - rnum - pnum && i < M * N)                          //边缘的先没管 这个取元素[i-pnum+rnum:ii-pnum-1 i+pnum+1:ii+rnum+pnum]有没有优化区间捏
     {                                                                                        //这个判断信号大不大于门限的函数感觉会有一堆分支啥的
         for (int aaa = pnum + 1; aaa <=pnum + rnum; aaa++)                                                     //咋优化, 感觉是不是得换个法子算这玩意
         {
             d_out[i] = d_out[i] + d_signal[i + aaa] + d_signal[i - aaa];
         }
         d_out[i] = d_out[i] / (double)rnum;
         thold = d_out[i] * k;
         if(d_signal[i]<= thold)
         { 
             d_out[i] = 0;                                                                      //就是反过来嘛, 大于门限的保留, 小于的清零    //感觉这也有优化空间
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
     cudaMemcpy(d_ori, ori, memSize/M, cudaMemcpyHostToDevice);                           //signal和ori弄成gpu的
     //test(d_signal, M, N);
     pulseCompression(d_signal, d_ori, M, N);                                           //脉冲压缩
     //writeDataComplex(d_signal, M, N);
     //makeSmall(d_signal, M, N);
     //cudaDeviceSynchronize();
     //test(d_signal, M, N);                                                            
     //makeSmall(d_signal, M, N);
     mtd(d_signal, M, N);                                                               //脉冲压缩的结果送到mtd
     //test(d_signal, M, N);
     double* d_sqSignal;
     cudaMalloc((void**)&d_sqSignal, memSize);
     double* d_out;
     cudaMalloc((void**)&d_out, memSize);
     cudaMemset(d_out, 1, memSize);                                 //咱也不知道有没有意义这一步         有的, 不然边缘的就没赋值了
     dim3 block1, grid1;
     block1.x = BLOCKX;
     grid1.x = (M * N + block1.x - 1) / block1.x;
     rdSquareCopy << < grid1, block1 >> > (d_sqSignal, d_signal, M, N);
     dim3 block2, grid2;
     block2.x = BLOCKX;
     grid2.x = (M * N + block2.x - 1) / block2.x;
     int pnum = 4;                                  //保护单元
     int rnum = 10;                                  // 参考单元
     double pfa = 1e-6;                                 // 恒虚警率               //这个可以考虑用那个什么什么常量内存啥的
     double k = powf(pfa, (-1 / (2 * (double)rnum))) - 1;
     CFAR << < grid2, block2 >> > (d_out, d_sqSignal, M, N, rnum, pnum, k);
     QueryPerformanceCounter(&nLastTime2);
     float fInterval = nLastTime2.QuadPart - nLastTime1.QuadPart;
    
     //writeData(d_out, M, N);
     //printf("处理完了, 你真棒 \n");
     cudaFree(d_signal);
     cudaFree(d_ori);
     cudaFree(d_sqSignal);
     cudaFree(d_out);
     return  fInterval / (float)nFreq.QuadPart;
}