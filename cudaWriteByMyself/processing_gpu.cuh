#ifndef PROCESSING_GPU_CUH
#define PROCESSING_GPU_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>

__global__ void rdComplexMultiply(cuFloatComplex* s, cuFloatComplex* w, long int M, long int N);
__global__ void rdComplexTranspose(cuFloatComplex* sout, cuFloatComplex* sin, long int M, long int N);
__global__ void rdSquareCopy(float* sout, cuFloatComplex* sin, long int M, long int N);
void readData(cuFloatComplex* signal);
void pulseCompression(cuFloatComplex* d_signal, cuFloatComplex* d_ori, int M, int N);
void mtd(cuFloatComplex* d_signal, int M, int N);
__global__ void CFAR(float *d_out, float* d_signal, long int M, long int N, int rnum, int prum, float k);
void doGpuProcessing(cuFloatComplex* signal, cuFloatComplex* ori, int M, int N);


#endif // !PROCESSING_GPU_H

