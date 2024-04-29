#ifndef FILEOP_H
#define FILEOP_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <math.h>
#include <cuComplex.h>
#include <windows.h>

void readData(cuDoubleComplex* signal, cuDoubleComplex* ori, int M, int N);
void writeDataComplex(cuDoubleComplex* d_signal, int M, int N);
void writeData(double* d_signal, int M, int N);


#endif // !FILEOP_H
