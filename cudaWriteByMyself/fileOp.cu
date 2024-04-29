#include"fileOp.cuh"


void readData(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N)                    //�����ʱ���ù̶��ڴ��ܺõ�
{
    FILE* fp;//�ļ�ָ��
    fp = fopen("signal_real.txt", "r");//���ı���ʽ���ļ���
    if (fp == NULL) //���ļ�����
        printf("error1");
    for (int i = 0; i < M * N; i++)
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
    printf("������! \n");
}

void writeDataComplex(cuDoubleComplex* d_signal, int M, int N)
{
    int memSize = M * N * sizeof(cuDoubleComplex);
    cuDoubleComplex* signal;
    cudaMallocHost((void**)&signal, memSize);
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

void writeData(double* d_signal, int M, int N)               //����������gpu���ڴ������, �����Զ����㿽����Host��
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
