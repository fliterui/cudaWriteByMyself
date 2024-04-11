#ifndef PROCESSING_GPU_CUH
#define PROCESSING_GPU_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>
//#include<pyconfig.h>
__global__ void rdComplexMultiply(cuDoubleComplex* s, cuDoubleComplex* w, int M, int N);
__global__ void rdComplexTranspose(cuDoubleComplex* sout, cuDoubleComplex* sin, int M, int N);
__global__ void rdSquareCopy(double* sout, cuDoubleComplex* sin, int M, int N);
void readData(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N);
void writeData(double* result, int M, int N);
void readDataComplex(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N);
void pulseCompression(cuDoubleComplex* d_signal, cuDoubleComplex* d_ori, int M, int N);
void mtd(cuDoubleComplex* d_signal, int M, int N);
__global__ void CFAR(double *d_out, double* d_signal, int M, int N, int rnum, int prum, double k);
void doGpuProcessing(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N);


void test(cuDoubleComplex* d_signal, int M, int N);
void printGpuModComplex(cuDoubleComplex* d_signal);
void printGpuModFloat(double* d_signal);
void makeSmall(cuDoubleComplex* d_signal, int M, int N);
__global__ void mod1w(cuDoubleComplex* d_signal, int M, int N);




#endif // !PROCESSING_GPU_H

