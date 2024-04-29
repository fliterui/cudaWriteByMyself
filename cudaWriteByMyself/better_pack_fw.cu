#include"better_pack_fw.cuh"
#include"processing_gpu.cuh"

__global__ void better_rdComplexMultiply(cuDoubleComplex* s, cuDoubleComplex* w, int M, int N)         //Ϊ�˸���ת��Ū��
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        int n = i % N;
        //printf("%d %d          ", i, n);
        s[i] = cuCmul(s[i], cuConj(w[n]));
    }
}

__global__ void better_rdComplexTranspose(cuDoubleComplex* sout, cuDoubleComplex* sin, int M, int N)       //����ת��???   �ǵ�      ��������nmdgb bug, sin��sout������ͬһ����Ȼ���ͻ
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N * M)
    {
        int n = i % N;                                                             //�������
        int m = (int)(i - n) / N;                                             //�������

        sout[m + n * M] = sin[n + m * N];
    }
}

void better_mtd(cuDoubleComplex* d_signal, int M, int N, dim3 grid1, dim3 block1, dim3 grid2, dim3 block2)
{
    /*cuDoubleComplex* signal;
    size_t memSize = M* N * sizeof(cuDoubleComplex);
    cudaMalloc((void**)&signal, memSize);
    cudaMemcpy(signal, d_signal, memSize, cudaMemcpyDeviceToDevice);                                                                         //����, ���滹��ת�û�ȥ, cfarҪ��mtd�Ľ��*/
    // block, grid;
    //block.x = BLOCKX;
    //grid.x = (M * N + block.x - 1) / block.x;
    cuDoubleComplex* dd_signal;
    size_t memSize = M * N * sizeof(cuDoubleComplex);
    cudaMalloc((void**)&dd_signal, memSize);
    better_rdComplexTranspose << < grid1, block1 >> > (dd_signal, d_signal, M, N);                                    //��ת�ú����е�fft           //����úݺݵ��Ż�
    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, N);                                                                                                          //����˵fft����Ӧ�ô���M���� ��һ��k>M
    cufftExecZ2Z(plan, (cufftDoubleComplex*)dd_signal, (cufftDoubleComplex*)dd_signal, CUFFT_FORWARD);                                                 //��fft
    cufftDestroy(plan);
    rdComplexTranspose << < grid2, block2 >> > (d_signal, dd_signal, N, M);                 //ת�û�ȥ��ʱ����N��M��, ������N,M!!!!!   ������Ҳ�֪��զŪ,�ȴպ��ð�
    cudaFree(dd_signal);
}
LARGE_INTEGER nFreq_1;
LARGE_INTEGER nLastTime1_1;
LARGE_INTEGER nLastTime2_1;
float goodTransDoGpuProcessing(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N, dim3* grid, dim3* block)
{

    //printf("%d ", ++__count);
    QueryPerformanceFrequency(&nFreq_1);
    QueryPerformanceCounter(&nLastTime1_1);
    size_t memSize = M * N * sizeof(cuDoubleComplex);
    cuDoubleComplex* d_signal, * d_ori;
    cudaMalloc((void**)&d_signal, memSize);
    cudaMalloc((void**)&d_ori, memSize / M);
    //eee                     //signal��oriŪ��gpu��
    ///////////////////////////////Ϊ�˷��������ѹ���ĺ�������////////////////
    //pulseCompression(d_signal, d_ori, M, N, grid[0], block[0]);                                           //����ѹ��

    cudaStream_t ps1;
    cudaStream_t ps2;
    cudaStreamCreateWithFlags(&ps1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&ps2, cudaStreamNonBlocking);
    cudaMemcpyAsync(d_signal, signal, memSize, cudaMemcpyHostToDevice, ps1);
    cudaMemcpyAsync(d_ori, ori, memSize / M, cudaMemcpyHostToDevice, ps2);

    cufftHandle plan1;
    cufftSetStream(plan1, ps1);
    cufftHandle plan2;
    cufftSetStream(plan2, ps2);
    cufftPlan1d(&plan1, N, CUFFT_Z2Z, M);
    cufftExecZ2Z(plan1, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_FORWARD);               //�ź�fft, �������
    cufftDestroy(plan1);

    cufftPlan1d(&plan2, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan2, (cufftDoubleComplex*)d_ori, (cufftDoubleComplex*)d_ori, CUFFT_FORWARD);                     //sin fft, �������
    cufftDestroy(plan2);
    cudaStreamDestroy(ps1);
    cudaStreamDestroy(ps2);
    rdComplexMultiply << <grid[0], block[0] >> > (d_signal, d_ori, M, N);                                              //���Թ���ֱ, ֱ�Ӹĵ�d_signal
    cufftHandle plan3;
    cufftPlan1d(&plan3, N, CUFFT_Z2Z, M);
    cufftExecZ2Z(plan3, (cufftDoubleComplex*)d_signal, (cufftDoubleComplex*)d_signal, CUFFT_INVERSE);                                                 //ifft
    cufftDestroy(plan3);
    ////////////////////////////////////////////////////////////////////////////////
    better_mtd(d_signal, M, N, grid[1], block[1], grid[2], block[2]);                                                               //����ѹ���Ľ���͵�mtd
    //test(d_signal, M, N);
    double* d_sqSignal;
    cudaMalloc((void**)&d_sqSignal, memSize);
    double* d_out;
    cudaMalloc((void**)&d_out, memSize);
    cudaMemset(d_out, 1, memSize);                                 //��Ҳ��֪����û��������һ��         �е�, ��Ȼ��Ե�ľ�û��ֵ��
    //dim3 block1, grid1;
    //block1.x = BLOCKX;
    //grid1.x = (M * N + block1.x - 1) / block1.x;
    rdSquareCopy << < grid[3], block[3] >> > (d_sqSignal, d_signal, M, N);
    //dim3 block2, grid2;
    //block2.x = BLOCKX;
    //grid2.x = (M * N + block2.x - 1) / block2.x;
    int pnum = 4;                                  //������Ԫ
    int rnum = 10;                                  // �ο���Ԫ
    double pfa = 1e-6;                                 // ���龯��               //������Կ������Ǹ�ʲôʲô�����ڴ�ɶ��
    double k = powf(pfa, (-1 / (2 * (double)rnum))) - 1;
    CFAR << < grid[4], block[4] >> > (d_out, d_sqSignal, M, N, rnum, pnum, k);
    QueryPerformanceCounter(&nLastTime2_1);
    float fInterval = nLastTime2_1.QuadPart - nLastTime1_1.QuadPart;

    //writeData(d_out, M, N);
    //printf("��������, ����� \n");
    cudaFree(d_signal);
    cudaFree(d_ori);
    cudaFree(d_sqSignal);
    cudaFree(d_out);
    return  fInterval / (float)nFreq_1.QuadPart;
}