#include"fileOp.cuh"


void readData(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N)                    //这个到时候用固定内存能好点
{
    FILE* fp;//文件指针
    fp = fopen("signal_real.txt", "r");//以文本方式打开文件。
    if (fp == NULL) //打开文件出错。
        printf("error1");
    for (int i = 0; i < M * N; i++)
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
    for (int i = 0; i < N; i++)
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
    printf("读完了! \n");
}

void writeDataComplex(cuDoubleComplex* d_signal, int M, int N)
{
    int memSize = M * N * sizeof(cuDoubleComplex);
    cuDoubleComplex* signal;
    cudaMallocHost((void**)&signal, memSize);
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

void writeData(double* d_signal, int M, int N)               //这个输入的是gpu的内存就行了, 他会自动给你拷贝到Host里
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
