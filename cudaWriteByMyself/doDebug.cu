#include "doDebug.cuh"
#include"processing_gpu.cuh"
#include <device_launch_parameters.h>

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

void printGpuModComplex(cuDoubleComplex* d_signal)
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
    block.x = 256;
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