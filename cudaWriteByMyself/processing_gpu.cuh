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
void readData(cuFloatComplex* signal);
void pulseCompression(cuFloatComplex* signal, cuFloatComplex* ori, int M, int N);

#endif // !PROCESSING_GPU_H

