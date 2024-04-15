#ifndef DODEBUG_H
#define DODEBUG_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>
#include <windows.h>

void test(cuDoubleComplex* d_signal, int M, int N);
void printGpuModComplex(cuDoubleComplex* d_signal);
void printGpuModFloat(double* d_signal);
void makeSmall(cuDoubleComplex* d_signal, int M, int N);
__global__ void mod1w(cuDoubleComplex* d_signal, int M, int N);



#endif // !DODEBUG_H
